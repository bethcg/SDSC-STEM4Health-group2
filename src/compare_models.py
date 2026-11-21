"""Compare three supervised models for BRPMM, per process, with the nested CV of ridge_classifier.py.

Every option turns each batch, whatever its length, into a fixed-size input that uses every row:

  1. Ridge on trajectories: the batch is cut into k equal slices (k picked by CV from 20/50/100)
     and each value is the mean of all rows in its slice (average pooling). Batch length is
     added as an extra input. Variant 1b also keeps each slice's std (how noisy it was). The ridge weights show which sensor, at which % of the batch,
     separates rejected from accepted batches (saved as a figure).
  2. MiniRocket + ridge: thousands of random convolution kernels slide over the curves
     (slice-averaged to a common length); their pooled outputs feed a ridge classifier.
     Needs the `aeon` package; skipped if it is not installed.
  3. Trees on phase features: the batch is split into thirds; mean, std, min, max, slope and
     roughness (std of the step-to-step change) per sensor and third, plus batch length.
     Random forest and gradient boosting.

Scoring is F1 of the "rejected" class. Each option is refit on all calibration batches and its
test predictions are written to outputs/submission_<option>.csv (template format).
Nothing is submitted anywhere.

    python src/compare_models.py --data-dir "/path/to/Project 2 - BRPMM"
"""
import argparse
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ridge_classifier import METRICS, batch_features, find_data_dir, make_search

MAX_SLICES = 100   # trajectories are stored at 100 slices; coarser k must divide it
ROCKET_LEN = 500   # common length fed to MiniRocket


def load_batches(csv_path):
    """{batch_id: array (rows, sensors)} in file order, plus the sensor names."""
    df = pd.read_csv(csv_path, dtype={"batch_id": str})
    sensors = [c for c in df.columns if c != "batch_id"]
    return {b: g[sensors].to_numpy(float) for b, g in df.groupby("batch_id", sort=True)}, sensors


def resample(a, length):
    """(rows, sensors) -> (sensors, length). Slice means when the batch is long enough
    (every row is used), linear interpolation otherwise."""
    n = len(a)
    if n >= length:
        idx = np.arange(n) * length // n
        sums = np.zeros((length, a.shape[1]))
        np.add.at(sums, idx, a)
        return (sums / np.bincount(idx, minlength=length)[:, None]).T
    old, new = np.linspace(0, 1, n), np.linspace(0, 1, length)
    return np.stack([np.interp(new, old, a[:, j]) for j in range(a.shape[1])])


def slice_stats(a, length):
    """(rows, sensors) -> (2 * sensors, length): slice means, then slice stds (rows >= length)."""
    idx = np.arange(len(a)) * length // len(a)
    counts = np.bincount(idx, minlength=length)[:, None]
    s1, s2 = np.zeros((length, a.shape[1])), np.zeros((length, a.shape[1]))
    np.add.at(s1, idx, a)
    np.add.at(s2, idx, a ** 2)
    mean = s1 / counts
    std = np.sqrt(np.maximum(s2 / counts - mean ** 2, 0))
    return np.vstack([mean.T, std.T])


def phase_features(a, sensors, n_phases=3):
    feats = {}
    for j, part in enumerate(np.array_split(a, n_phases), start=1):
        t = np.linspace(0, 1, len(part))
        stats = {
            "mean": part.mean(0), "std": part.std(0), "min": part.min(0), "max": part.max(0),
            "slope": np.polyfit(t, part, 1)[0], "rough": np.diff(part, axis=0).std(0),
        }
        for stat, values in stats.items():
            feats.update({f"{s}__p{j}_{stat}": v for s, v in zip(sensors, values)})
    feats["n_rows"] = len(a)
    return feats


class Coarsen(BaseEstimator, TransformerMixin):
    """[sensors x MAX_SLICES slice means, n_rows] -> [sensors x k slice means, n_rows]."""

    def __init__(self, n_sensors=1, k=MAX_SLICES):
        self.n_sensors, self.k = n_sensors, k

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        s = X[:, :-1].reshape(len(X), self.n_sensors, self.k, MAX_SLICES // self.k).mean(axis=3)
        return np.hstack([s.reshape(len(X), -1), X[:, -1:]])


class MiniRocketFeatures(BaseEstimator, TransformerMixin):
    """Standardise each sensor (training stats), then MiniRocket's pooled convolution features."""

    def __init__(self, n_sensors=1, n_kernels=10000, seed=0):
        self.n_sensors, self.n_kernels, self.seed = n_sensors, n_kernels, seed

    def _scaled(self, X):
        X3 = np.asarray(X).reshape(len(X), self.n_sensors, -1)
        return (X3 - self.mu_) / self.sd_

    def fit(self, X, y=None):
        from aeon.transformations.collection.convolution_based import MiniRocket
        X3 = np.asarray(X).reshape(len(X), self.n_sensors, -1)
        self.mu_ = X3.mean(axis=(0, 2), keepdims=True)
        self.sd_ = np.where(X3.std(axis=(0, 2), keepdims=True) > 0, X3.std(axis=(0, 2), keepdims=True), 1.0)
        self.rocket_ = MiniRocket(n_kernels=self.n_kernels, random_state=self.seed).fit(self._scaled(X))
        return self

    def transform(self, X):
        return np.asarray(self.rocket_.transform(self._scaled(X)))


def grid(pipe, params, n_min, seed, n_jobs=-1):
    inner = StratifiedKFold(n_splits=min(5, max(2, n_min - 1)), shuffle=True, random_state=seed)
    return GridSearchCV(pipe, params, cv=inner, scoring="f1", n_jobs=n_jobs)


def trajectory_ridge(n_channels, n_min, seed):
    pipe = Pipeline([
        ("coarsen", Coarsen(n_channels)), ("impute", SimpleImputer(strategy="median")),
        ("drop_constant", VarianceThreshold()), ("scale", StandardScaler()),
        ("model", RidgeClassifier(class_weight="balanced"))])
    return grid(pipe, {"coarsen__k": [20, 50, 100], "model__alpha": np.logspace(-1, 6, 15)}, n_min, seed)


def build_models(n_sensors, n_min, seed):
    """{name: (model, input view)}."""
    prep = [("impute", SimpleImputer(strategy="median")), ("drop_constant", VarianceThreshold())]
    models = {
        "1_trajectory_ridge": (trajectory_ridge(n_sensors, n_min, seed), "slices"),
        "1b_trajectory_ridge_mean_std": (trajectory_ridge(2 * n_sensors, n_min, seed), "slices_std"),
        "3a_phase_random_forest": (grid(
            Pipeline(prep + [("model", RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample", random_state=seed))]),
            {"model__min_samples_leaf": [1, 3, 5], "model__max_features": ["sqrt", 0.3]}, n_min, seed), "phases"),
        "3b_phase_gradient_boosting": (grid(
            Pipeline(prep + [("model", HistGradientBoostingClassifier(
                max_iter=200, class_weight="balanced", random_state=seed))]),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]}, n_min, seed), "phases"),
    }
    try:
        import aeon  # noqa: F401
        # cache the MiniRocket transform so the alpha grid does not recompute it per candidate
        rocket_pipe = Pipeline([
            ("rocket", MiniRocketFeatures(n_sensors, seed=seed)), ("scale", StandardScaler()),
            ("model", RidgeClassifier(class_weight="balanced"))], memory=tempfile.mkdtemp(prefix="rocket_"))
        # n_jobs=1: numba's thread pool aborts when called from several threads at once
        models["2_minirocket_ridge"] = (
            grid(rocket_pipe, {"model__alpha": np.logspace(-1, 5, 7)}, n_min, seed, n_jobs=1), "rocket")
    except ImportError:
        print("  (aeon not installed: skipping MiniRocket)")
    return models


def plot_ridge_weights(search, channels, p, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pipe = search.best_estimator_
    k = pipe.named_steps["coarsen"].k
    w = np.zeros(len(channels) * k + 1)
    w[pipe.named_steps["drop_constant"].get_support()] = pipe.named_steps["model"].coef_.ravel()
    w = w[:-1].reshape(len(channels), k)
    x = (np.arange(k) + 0.5) * 100 / k
    lim = np.abs(w).max() * 1.05
    fig, axes = plt.subplots(len(channels), 1, figsize=(8, 0.9 * len(channels) + 1), sharex=True)
    for ax, s, ws in zip(axes, channels, w):
        ax.bar(x, ws, width=100 / k, color=np.where(ws > 0, "tab:red", "tab:blue"))
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylim(-lim, lim)
        ax.set_ylabel(s, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_yticks([])
    axes[0].set_title(f"Process {p}: ridge weights (red = pushes towards rejected), k={k} slices")
    axes[-1].set_xlabel("% of batch")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def run_process(p, data_dir, seed, repeats, fig_dir):
    proc = data_dir / f"process {p}"
    cal_csv = proc / "calibration data" / "calibration_data.csv"
    test_csv = proc / "test data" / "test_data.csv"
    cal, sensors = load_batches(cal_csv)
    test, _ = load_batches(test_csv)
    y = pd.read_csv(proc / "calibration data" / "calibration_labels.csv",
                    dtype={"batch_id": str}).set_index("batch_id")["label"]
    ids, ids_test = list(y.index), sorted(test)
    y_bin = (y == "rejected").astype(int).to_numpy()
    n_min = int(np.bincount(y_bin).min())
    print(f"\n=== Process {p}: {len(ids)} labelled batches ({y_bin.sum()} rejected), {len(sensors)} sensors ===")

    def views(batches, keys):
        arrs = [batches[b] for b in keys]
        return {
            "slices": np.stack([np.r_[resample(a, MAX_SLICES).ravel(), len(a)] for a in arrs]),
            "slices_std": np.stack([np.r_[slice_stats(a, MAX_SLICES).ravel(), len(a)] for a in arrs]),
            "rocket": np.stack([resample(a, ROCKET_LEN).ravel() for a in arrs]),
            "phases": pd.DataFrame([phase_features(a, sensors) for a in arrs], index=keys).to_numpy(),
        }

    X, Xt = views(cal, ids), views(test, ids_test)
    X["current"] = batch_features(cal_csv).loc[ids].to_numpy()
    Xt["current"] = batch_features(test_csv).loc[ids_test].to_numpy()

    models = {"current_summary_ridge": (make_search("ridge", n_min, seed), "current")}
    models.update(build_models(len(sensors), n_min, seed))
    outer = RepeatedStratifiedKFold(n_splits=min(5, n_min), n_repeats=repeats, random_state=seed)
    rows, preds = [], {}
    for name, (model, view) in models.items():
        cv = cross_validate(model, X[view], y_bin, cv=outer, scoring=METRICS, n_jobs=-1)
        row = {"process": p, "model": name}
        for m in METRICS:
            row[m], row[f"{m}_sd"] = cv[f"test_{m}"].mean(), cv[f"test_{m}"].std()
        final = model.fit(X[view], y_bin)
        pred = final.predict(Xt[view])
        row["test_pred_rejected"] = int(pred.sum())
        rows.append(row)
        print(f"  {name:28s} f1 {row['f1']:.3f}±{row['f1_sd']:.3f}  precision {row['precision']:.3f}  "
              f"recall {row['recall']:.3f}  accuracy {row['accuracy']:.3f}  "
              f"| test: {pred.sum()} of {len(pred)} predicted rejected")
        preds[name] = pd.DataFrame({"batch_id": ids_test, "project": p,
                                    "label": np.where(pred == 1, "rejected", "accepted")})
        if name.startswith("1"):
            channels = sensors + [f"{s} (std)" for s in sensors] if view == "slices_std" else sensors
            print(f"    chosen: {final.best_params_}")
            plot_ridge_weights(final, channels, p, fig_dir / f"weights_{name}_{p}.png")
    return rows, preds


def write_submission(preds, template_path, out):
    template = pd.read_csv(template_path, dtype={"batch_id": str})
    template.columns = template.columns.str.strip()
    template["batch_id"] = template["batch_id"].str.strip()
    if set(template.batch_id) != set(preds.batch_id):
        raise SystemExit(f"Predicted batches differ from the template: {out.name}")
    sub = template[["batch_id", "project"]].merge(preds[["batch_id", "label"]], on="batch_id", how="left")
    assert len(sub) == len(template) and sub.label.isin(["accepted", "rejected"]).all()
    sub.to_csv(out, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path)
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--out-dir", type=Path, default=repo / "outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5, help="repeats of the outer 5-fold CV")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    data_dir = (args.data_dir or find_data_dir()).resolve()
    out_dir = args.out_dir.resolve()
    if "input-data" in out_dir.parts or out_dir.is_relative_to(data_dir):
        raise SystemExit(f"Refusing to write inside the data folder: {out_dir}")
    print(f"data: {data_dir}")

    rows, preds = [], {}
    for p in "AB":
        r, pr = run_process(p, data_dir, args.seed, args.repeats, out_dir / "figures")
        rows += r
        for name, df in pr.items():
            preds.setdefault(name, []).append(df)

    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "model_comparison.csv", index=False)
    f1 = table.pivot(index="model", columns="process", values="f1")
    f1["mean"] = f1.mean(axis=1)
    print("\nCV F1 of rejected (estimated score = mean of A and B):")
    print(f1.sort_values("mean", ascending=False).round(3).to_string())

    for name, parts in preds.items():
        if not name.startswith("current"):
            write_submission(pd.concat(parts, ignore_index=True), data_dir / "BRPMM_template.csv",
                             out_dir / f"submission_{name}.csv")
    print(f"\nwrote {out_dir}/model_comparison.csv, submission_<model>.csv and figures/ - not submitted anywhere")


if __name__ == "__main__":
    # run through the module name so the custom transformers pickle for parallel workers
    from compare_models import main as run
    run()
