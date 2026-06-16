"""Show how predictions.csv changed vs the last committed version.

Used by the /reforecast workflow so each re-run reports only the fixtures
whose pick moved. Compares the working-tree predictions.csv against
`git show HEAD:predictions.csv`.

    uv run python -m scripts.diff_predictions
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "predictions.csv"
KEY = ["home", "away"]


def _committed() -> pd.DataFrame | None:
    try:
        out = subprocess.run(
            ["git", "show", "HEAD:predictions.csv"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return pd.read_csv(io.StringIO(out))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    if not CSV.exists():
        print("no predictions.csv yet — run scripts.run_schedule first")
        return
    new = pd.read_csv(CSV)
    old = _committed()
    if old is None:
        print(f"no committed predictions.csv to diff against; "
              f"{len(new)} fixtures in working tree.")
        return

    m = old.merge(new, on=KEY, how="outer", suffixes=("_old", "_new"))
    changed = m[m["pick_old"] != m["pick_new"]]
    print(f"=== reforecast diff: {len(changed)} of {len(new)} picks changed ===")
    if changed.empty:
        print("(no pick changes)")
        return
    for _, r in changed.iterrows():
        fixture = f"{r['home']} vs {r['away']}"
        old_p = r.get("pick_old", "—")
        new_p = r.get("pick_new", "—")
        if pd.isna(old_p):
            print(f"  + {fixture:42s} new fixture -> {new_p}")
        elif pd.isna(new_p):
            print(f"  - {fixture:42s} dropped (was {old_p})")
        else:
            print(f"  ~ {fixture:42s} {old_p} -> {new_p}")


if __name__ == "__main__":
    main()
