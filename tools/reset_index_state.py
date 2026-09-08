#!/usr/bin/env python3
"""Replace state values in prediction index CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

STATE_LABELS = {
    -1.0: "pending",
    0.0: "in_progress",
    1.0: "success",
    2.0: "success_multi_person",
    -2.0: "multi_person_reject",
    -3.0: "empty_or_no_person",
}


def collect_index_csvs(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise SystemExit(f"Expected a .csv file, got: {path}")
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Index path not found: {path}")
    csvs = sorted(path.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No .csv files found in: {path}")
    return csvs


def state_label(value: float) -> str:
    return STATE_LABELS.get(float(value), "unknown")


def replace_state(
    csv_paths: Sequence[Path],
    old_state: float,
    new_state: float,
    dry_run: bool = False,
) -> int:
    total_replaced = 0
    for path in csv_paths:
        df = pd.read_csv(path, dtype={"file_names": str, "state": float})
        if "state" not in df.columns:
            print(f"  SKIP {path.name}: no 'state' column")
            continue
        mask = df["state"] == old_state
        n = int(mask.sum())
        total_replaced += n
        print(f"  {path.name}: {n} row(s) {old_state:g} -> {new_state:g}")
        if n == 0 or dry_run:
            continue
        df.loc[mask, "state"] = new_state
        df.to_csv(path, index=False)
    return total_replaced


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace one state value with another in index CSV file(s). "
            "Pass a single .csv or a folder containing index_file_*.csv."
        )
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        required=True,
        help="Path to an index CSV or a folder of index CSVs (e.g. predict_*/_index)",
    )
    parser.add_argument(
        "--old-state",
        type=float,
        required=True,
        help="State value to replace (e.g. 0)",
    )
    parser.add_argument(
        "--new-state",
        type=float,
        required=True,
        help="Replacement state value (e.g. -1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count matches without writing changes",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    index_path = args.index_path.resolve()
    csv_paths = collect_index_csvs(index_path)

    print(
        f"Replacing state {args.old_state:g} ({state_label(args.old_state)}) "
        f"-> {args.new_state:g} ({state_label(args.new_state)}) "
        f"in {len(csv_paths)} file(s)"
        + (" [dry-run]" if args.dry_run else "")
    )
    total = replace_state(csv_paths, args.old_state, args.new_state, dry_run=args.dry_run)
    print(f"Total rows {'matched' if args.dry_run else 'updated'}: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
