"""Ridge (L2-regularised) classifier for BRPMM: accepted / rejected per batch.

One model per process (A and B have different sensors and sizes). Each batch's
time series is collapsed into summary statistics, then a RidgeClassifier is
fitted with alpha chosen by inner CV on F1 of the "rejected" class (the
leaderboard metric). Tuning the decision threshold as well is optional
(--tune-threshold): on the calibration data it lowered CV F1 in both processes. Repeated outer CV on the
calibration data gives an honest estimate of that F1; the final model is then
refit on all calibration batches and used to label the test batches.

Nothing is sent anywhere: the result is only written to outputs/submission.csv
(batch_id,project,label) in the template's order.

    python src/ridge_classifier.py --data-dir "/path/to/Project 2 - BRPMM"
    python src/ridge_classifier.py --model logreg       # L2 logistic instead
    python src/ridge_classifier.py --tune-threshold     # also tune the decision threshold

Rows within a batch are assumed to be in time order (there is no time column).
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     TunedThresholdClassifierCV, cross_val_score, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_NAME = "Project 2 - BRPMM"
STATS = ["mean", "std", "min", "max", "median", "first", "last"]
METRICS = ["f1", "precision", "recall", "accuracy"]  # as on the leaderboard, for "rejected"


def find_data_dir():
    repo = Path(__file__).resolve().parent.parent
    candidates = [
        repo.parent / DATA_NAME,                          # data folder next to the repo
        repo / "input-data" / DATA_NAME,
        repo.parent / "input-data" / DATA_NAME,           # Renku: /home/renku/work/input-data
        Path.home() / "work" / "input-data" / DATA_NAME,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise SystemExit(f"Could not find '{DATA_NAME}'. Use --data-dir.")


def batch_features(csv_path):
    """One row per batch: summary stats of every sensor over the batch."""
    df = pd.read_csv(csv_path, dtype={"batch_id": str})
    sensors = [c for c in df.columns if c != "batch_id"]
    agg = df.groupby("batch_id", sort=True)[sensors].agg(STATS)
    agg.columns = [f"{s}__{stat}" for s, stat in agg.columns]
    agg["n_rows"] = df.groupby("batch_id").size()
    return agg


def make_search(kind, n_min, seed, k_best=None, tune_threshold=False):
    """Pipeline + alpha grid search (optionally + F1-tuned threshold), sized for the rare class."""
    if kind == "ridge":
        est, grid = RidgeClassifier(class_weight="balanced"), {"model__alpha": np.logspace(-1, 5, 13)}
    else:
        est = LogisticRegression(class_weight="balanced", max_iter=5000)  # L2 is the default penalty
        grid = {"model__C": np.logspace(-4, 2, 13)}
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("drop_constant", VarianceThreshold()),
        ("scale", StandardScaler()),
    ]
    if k_best:  # baseline: the single most discriminative feature only
        steps.append(("select", SelectKBest(f_classif, k=k_best)))
    pipe = Pipeline(steps + [("model", est)])
    # training folds keep ~(k-1)/k of the rare class, so nested CVs need fewer splits
    k = min(5, max(2, n_min - 1))
    inner = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, grid, cv=inner, scoring="f1", n_jobs=-1)
    if not tune_threshold:
        return search
    return TunedThresholdClassifierCV(search, scoring="f1", cv=inner, random_state=seed)


def report_cv(name, model, X, y, outer):
    cv = cross_validate(model, X, y, cv=outer, scoring=METRICS, n_jobs=-1)
    parts = [f"{m} {cv[f'test_{m}'].mean():.3f}±{cv[f'test_{m}'].std():.3f}" for m in METRICS]
    print(f"  {name:28s} " + "  ".join(parts))
    return cv["test_f1"].mean()


def shift_check(X_cal, X_test, seed):
    """Can a classifier tell calibration from test batches? AUC ~0.5 means they look alike."""
    X = pd.concat([X_cal, X_test[X_cal.columns]])
    is_test = np.r_[np.zeros(len(X_cal)), np.ones(len(X_test))]
    clf = Pipeline([("impute", SimpleImputer(strategy="median")), ("drop_constant", VarianceThreshold()),
                    ("scale", StandardScaler()), ("model", LogisticRegression(C=0.1, max_iter=5000))])
    auc = cross_val_score(clf, X, is_test, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean()
    sd = X_cal.std().replace(0, np.nan)
    diff = ((X_test[X_cal.columns].mean() - X_cal.mean()) / sd).abs().dropna().sort_values(ascending=False)
    top = ", ".join(f"{c} ({v:.2f} sd)" for c, v in diff.head(3).items())
    print(f"  shift check: calibration-vs-test AUC {auc:.3f} (0.5 = same distribution); largest mean shifts: {top}")


def run_process(p, data_dir, kind, seed, tune):
    proc = data_dir / f"process {p}"
    X = batch_features(proc / "calibration data" / "calibration_data.csv")
    y = pd.read_csv(proc / "calibration data" / "calibration_labels.csv", dtype={"batch_id": str}).set_index("batch_id")["label"]
    X = X.loc[y.index]
    y_bin = (y == "rejected").astype(int)  # positive class = rejected (the leaderboard's F1 class)
    n_min = int(y_bin.value_counts().min())
    print(f"\n=== Process {p}: {X.shape[0]} labelled batches, {X.shape[1]} features, "
          f"{y_bin.sum()} rejected / {(1 - y_bin).sum()} accepted ===")

    outer = RepeatedStratifiedKFold(n_splits=min(5, n_min), n_repeats=5, random_state=seed)
    prev = y_bin.mean()
    print(f"  {'baseline: everything rejected':28s} f1 {2 * prev / (1 + prev):.3f} (precision {prev:.3f}, recall 1.000)")
    report_cv("baseline: 1 best feature", make_search(kind, n_min, seed, k_best=1, tune_threshold=tune), X, y_bin, outer)
    f1 = report_cv(f"{kind}: all features", make_search(kind, n_min, seed, tune_threshold=tune), X, y_bin, outer)

    final = make_search(kind, n_min, seed, tune_threshold=tune).fit(X, y_bin)
    if tune:
        print(f"  final model (all calibration): {final.estimator_.best_params_}, threshold {final.best_threshold_:.3f}")
    else:
        print(f"  final model (all calibration): {final.best_params_}, default threshold")

    Xt = batch_features(proc / "test data" / "test_data.csv")
    shift_check(X, Xt, seed)
    pred = final.predict(Xt[X.columns])
    print(f"  test batches: {len(pred)}, predicted rejected: {pred.sum()} ({pred.mean():.0%}; "
          f"calibration has {prev:.0%})")
    out = pd.DataFrame({"batch_id": Xt.index, "project": p,
                        "label": np.where(pred == 1, "rejected", "accepted")})
    return out, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--model", choices=["ridge", "logreg"], default="ridge")
    # anchored to the repo (not the current folder), so it never writes into the data folder
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--out", type=Path, default=repo / "outputs" / "submission.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune-threshold", action="store_true")
    args = ap.parse_args()
    # B has 5 rejected batches: some CV folds hold a single class, which makes sklearn chatty
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    data_dir = (args.data_dir or find_data_dir()).resolve()
    print(f"data: {data_dir}  model: {args.model}")
    out = args.out.resolve()
    if "input-data" in out.parts or out.is_relative_to(data_dir):
        raise SystemExit(f"Refusing to write inside the data folder: {out}")

    results = {p: run_process(p, data_dir, args.model, args.seed, args.tune_threshold) for p in "AB"}
    preds = pd.concat([r[0] for r in results.values()], ignore_index=True)
    f1 = {p: r[1] for p, r in results.items()}
    print(f"\nestimated score (mean CV F1 of rejected, A and B): "
          f"{(f1['A'] + f1['B']) / 2:.3f}  (A {f1['A']:.3f}, B {f1['B']:.3f})")

    template = pd.read_csv(data_dir / "BRPMM_template.csv", dtype={"batch_id": str})
    template.columns = template.columns.str.strip()
    template["batch_id"] = template["batch_id"].str.strip()
    missing = set(template.batch_id) ^ set(preds.batch_id)
    if missing:
        raise SystemExit(f"Predicted batches differ from template ({len(missing)} ids), e.g. {sorted(missing)[:5]}")
    sub = template[["batch_id", "project"]].merge(preds[["batch_id", "label"]], on="batch_id", how="left")
    assert list(sub.columns) == list(template.columns) and len(sub) == len(template)
    assert sub.label.isin(["accepted", "rejected"]).all()

    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(sub)} rows) - not submitted anywhere")
    print(sub.groupby(["project", "label"]).size().to_string())


if __name__ == "__main__":
    main()
