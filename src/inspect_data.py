"""Compact summary of the BRPMM data folder, meant to be pasted back into chat.

Run from the JupyterLab terminal (data can't be downloaded, so we look at it there):

    python src/inspect_data.py
    python src/inspect_data.py "/path/to/input-data/Project 2 - BRPMM" > summary.txt

Only reads files; never writes into input-data.
"""
import json
import sys
from pathlib import Path

import pandas as pd

DEFAULT = Path("input-data") / "Project 2 - BRPMM"
MAX_FILES_PER_DIR = 8  # only summarise the first N tabular files per folder
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 40)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def summarise_table(path):
    df = pd.read_csv(path, sep=None, engine="python", nrows=200_000)
    print(f"  shape (first 200k rows max): {df.shape}")
    print(f"  columns ({df.shape[1]}): {list(df.columns)[:40]}{' ...' if df.shape[1] > 40 else ''}")
    print("  dtypes:", df.dtypes.value_counts().to_dict())
    na = df.isna().sum()
    na = na[na > 0]
    print(f"  columns with NaN: {len(na)}", dict(na.head(10)) if len(na) else "")
    print("  head:")
    print(df.head(5).to_string(max_cols=15))
    num = df.select_dtypes("number")
    if not num.empty:
        print("  describe (first 10 numeric cols):")
        print(num.iloc[:, :10].describe().T[["count", "mean", "std", "min", "max"]].to_string())


def summarise_notebook(path):
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", []))
        print(f"  --- cell {i} [{cell['cell_type']}] ---")
        print("  " + src.replace("\n", "\n  ")[:1500])


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    print(f"ROOT: {root.resolve()}\n")

    print("== TREE ==")
    for p in sorted(root.rglob("*")):
        depth = len(p.relative_to(root).parts) - 1
        if depth > 2:
            continue
        size = human(p.stat().st_size) if p.is_file() else ""
        print(f"{'  ' * depth}{p.name}{'/' if p.is_dir() else ''} {size}")

    print("\n== TOP-LEVEL FILES ==")
    for p in sorted(root.glob("*")):
        if p.suffix == ".csv":
            print(f"\n# {p.name}")
            summarise_table(p)
        elif p.suffix == ".ipynb":
            print(f"\n# {p.name}")
            summarise_notebook(p)

    for sub in sorted(d for d in root.iterdir() if d.is_dir()):
        print(f"\n== FOLDER: {sub.name} ==")
        tables = sorted(f for f in sub.rglob("*") if f.suffix in {".csv", ".tsv", ".txt"})
        print(f"  {len(tables)} tabular-looking files")
        for f in tables[:MAX_FILES_PER_DIR]:
            print(f"\n# {f.relative_to(root)} ({human(f.stat().st_size)})")
            try:
                summarise_table(f)
            except Exception as e:  # noqa: BLE001
                print(f"  could not parse: {e}")


if __name__ == "__main__":
    main()
