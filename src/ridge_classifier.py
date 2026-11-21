"""Ridge (L2-regularised) classifier for BRPMM: accepted / rejected per batch.

One model per process (A and B have different sensors and sizes). Each batch's
time series is collapsed into summary statistics, then a RidgeClassifier is
fitted with alpha chosen by inner CV on F1 of the "rejected" class (the
leaderboard metric); outer repeated CV gives an honest estimate of that F1.

    python src/ridge_classifier.py                      # CV + predictions
    python src/ridge_classifier.py --data-dir "/home/renku/work/input-data/Project 2 - BRPMM"
    python src/ridge_classifier.py --model logreg       # L2 logistic instead

Writes outputs/submission.csv (batch_id,project,label) in the template's order.
Rows within a batch are assumed to be in time order (there is no time column).
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_NAME = Path("input-data") / "Project 2 - BRPMM"
STATS = ["mean", "std", "min", "max", "median", "first", "last"]


def find_data_dir():
    here = Path(__file__).resolve().parent
    for c in [DATA_NAME, here.parent / DATA_NAME, here.parent.parent / DATA_NAME, Path.home() / "work" / DATA_NAME]:
        if c.is_dir():
            return c
    raise SystemExit("Could not find 'input-data/Project 2 - BRPMM'. Use --data-dir.")


def batch_features(csv_path):
    """One row per batch: summary stats of every sensor over the batch."""
    df = pd.read_csv(csv_path)
    sensors = [c for c in df.columns if c != "batch_id"]
    agg = df.groupby("batch_id", sort=True)[sensors].agg(STATS)
    agg.columns = [f"{s}__{stat}" for s, stat in agg.columns]
    agg["n_rows"] = df.groupby("batch_id").size()
    return agg


def make_model(kind):
    if kind == "ridge":
        est, grid = RidgeClassifier(class_weight="balanced"), {"model__alpha": np.logspace(-1, 5, 13)}
    else:
        est = LogisticRegression(class_weight="balanced", max_iter=5000)  # L2 is the default penalty
        grid = {"model__C": np.logspace(-4, 2, 13)}
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("drop_constant", VarianceThreshold()),
        ("scale", StandardScaler()),
        ("model", est),
    ])
    return pipe, grid


def run_process(p, data_dir, kind, seed):
    proc = data_dir / f"process {p}"
    X = batch_features(proc / "calibration data" / "calibration_data.csv")
    y = pd.read_csv(proc / "calibration data" / "calibration_labels.csv", dtype={"batch_id": str}).set_index("batch_id")["label"]
    X = X.loc[y.index]
    y_bin = (y == "rejected").astype(int)  # positive class = rejected (the rare, costly one)
    print(f"\n=== Process {p}: {X.shape[0]} labelled batches, {X.shape[1]} features, "
          f"{y_bin.sum()} rejected / {(1 - y_bin).sum()} accepted ===")

    pipe, grid = make_model(kind)
    n_min = int(y_bin.value_counts().min())
    # outer training folds keep ~(k-1)/k of the rare class, so the inner CV needs fewer splits
    inner = StratifiedKFold(n_splits=min(5, max(2, n_min - 1)), shuffle=True, random_state=seed)
    # leaderboard metric: F1 of the rejected class (y_bin == 1)
    search = GridSearchCV(pipe, grid, cv=inner, scoring="f1", n_jobs=-1)

    outer = RepeatedStratifiedKFold(n_splits=min(5, n_min), n_repeats=3, random_state=seed)
    metrics = ["f1", "precision", "recall", "accuracy"]  # same as the leaderboard, for "rejected"
    cv = cross_validate(search, X, y_bin, cv=outer, scoring=metrics, n_jobs=1)
    for m in metrics:
        v = cv[f"test_{m}"]
        tag = "" if m == "accuracy" else " (rejected)"
        print(f"  nested CV {m + tag:21s}: {v.mean():.3f} +/- {v.std():.3f}")
    prev = y_bin.mean()
    print(f"  baseline 'everything rejected' F1: {2 * prev / (1 + prev):.3f}")

    search.fit(X, y_bin)
    print(f"  chosen hyperparameter: {search.best_params_}")

    Xt = batch_features(proc / "test data" / "test_data.csv")
    pred = search.predict(Xt[X.columns])
    out = pd.DataFrame({"batch_id": Xt.index, "project": p,
                        "label": np.where(pred == 1, "rejected", "accepted")})
    print(f"  test batches: {len(out)}, predicted rejected: {(pred == 1).sum()}")
    return out, cv["test_f1"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--model", choices=["ridge", "logreg"], default="ridge")
    # anchored to the repo (not the current folder), so running from input-data never writes there
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--out", type=Path, default=repo / "outputs" / "submission.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # B has ~5 rejected batches: some CV folds hold a single class, which makes sklearn chatty
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    data_dir = args.data_dir or find_data_dir()
    print(f"data: {data_dir}  model: {args.model}")
    out = args.out.resolve()
    if "input-data" in out.parts or out.is_relative_to(data_dir.resolve()):
        raise SystemExit(f"Refusing to write inside input-data: {out}")

    results = {p: run_process(p, data_dir, args.model, args.seed) for p in "AB"}
    preds = pd.concat([r[0] for r in results.values()], ignore_index=True)
    f1 = {p: r[1] for p, r in results.items()}
    print(f"\nestimated leaderboard score (mean CV F1 of A and B): "
          f"{(f1['A'] + f1['B']) / 2:.3f}  (A {f1['A']:.3f}, B {f1['B']:.3f})")

    template = pd.read_csv(data_dir / "BRPMM_template.csv", dtype={"batch_id": str})
    missing = set(template.batch_id) ^ set(preds.batch_id)
    if missing:
        raise SystemExit(f"Predicted batches differ from template ({len(missing)} ids), e.g. {sorted(missing)[:5]}")
    sub = template[["batch_id", "project"]].merge(preds[["batch_id", "label"]], on="batch_id", how="left")
    assert list(sub.columns) == list(template.columns) and len(sub) == len(template)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(sub)} rows)")
    print(sub.groupby(["project", "label"]).size().to_string())


if __name__ == "__main__":
    main()
