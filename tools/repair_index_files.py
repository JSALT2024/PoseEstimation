#!/usr/bin/env python3
"""Rebuild corrupted index CSVs and drop invalid prediction JSON files."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SUCCESS_STATES = {1.0, 2.0}


@dataclass
class ShardReport:
    path: Path
    unreadable: bool = False
    names_mismatch: bool = False
    existing_rows: int = 0
    expected_rows: int = 0
    preserved_states: int = 0
    json_deleted: List[str] = field(default_factory=list)
    marked_done: int = 0
    reset_missing_output: int = 0
    reset_in_progress: int = 0
    index_rewritten: bool = False

    @property
    def would_regenerate_index(self) -> bool:
        return self.unreadable or self.names_mismatch or self.index_rewritten


def list_input_videos(input_folder: Path, video_suffixes: Sequence[str]) -> List[str]:
    suffixes = tuple(s if s.startswith(".") else f".{s}" for s in video_suffixes)
    file_names = [
        file_name
        for file_name in os.listdir(input_folder)
        if any(file_name.lower().endswith(suffix.lower()) for suffix in suffixes)
    ]
    file_names.sort()
    return file_names


def split_index_file(index_file: pd.DataFrame, num_index_files: int) -> List[pd.DataFrame]:
    """Same split as utils_predict.create_index_files."""
    if num_index_files == 1:
        return [index_file]
    num_files = len(index_file)
    step = int(np.round(num_files / num_index_files))
    shards = []
    for i in range(1, num_index_files):
        start = (i - 1) * step
        end = i * step
        shards.append(index_file[start:end])
    shards.append(index_file[end::])
    return shards


def expected_index_shards(
    input_folder: Path,
    num_index_files: int,
    video_suffixes: Sequence[str],
) -> List[pd.DataFrame]:
    file_names = list_input_videos(input_folder, video_suffixes)
    index_file = pd.DataFrame({"file_names": file_names, "state": [-1] * len(file_names)})
    return split_index_file(index_file, num_index_files)


def clip_stem(file_name: str) -> str:
    return ".".join(Path(file_name).name.split(".")[:-1]) or Path(file_name).stem


def json_path_for_clip(output_folder: Path, file_name: str) -> Path:
    return output_folder / f"{clip_stem(file_name)}.json"


def load_index_csv(path: Path) -> Tuple[Optional[pd.DataFrame], bool]:
    if not path.is_file() or path.stat().st_size == 0:
        return None, True
    try:
        df = pd.read_csv(path, dtype={"file_names": str, "state": float})
    except Exception:
        return None, True
    if "file_names" not in df.columns:
        return None, True
    if "state" not in df.columns:
        df["state"] = -1.0
    df["file_names"] = df["file_names"].astype(str)
    df["state"] = pd.to_numeric(df["state"], errors="coerce").fillna(-1.0)
    return df, False


def json_is_valid(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        return True
    except Exception:
        return False


def names_match(existing: pd.DataFrame, expected: pd.DataFrame) -> bool:
    return existing["file_names"].tolist() == expected["file_names"].tolist()


def rebuild_shard_frame(
    expected: pd.DataFrame,
    existing: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, int]:
    rebuilt = expected.copy().reset_index(drop=True)
    rebuilt["file_names"] = rebuilt["file_names"].astype(str)
    rebuilt["state"] = -1.0
    preserved = 0
    if existing is not None and len(existing):
        old_states = dict(zip(existing["file_names"].astype(str), existing["state"]))
        for idx, name in enumerate(rebuilt["file_names"]):
            if name in old_states:
                rebuilt.at[idx, "state"] = old_states[name]
                preserved += 1
    return rebuilt, preserved


def apply_output_checks(
    df: pd.DataFrame,
    output_folder: Path,
    dry_run: bool,
) -> Tuple[pd.DataFrame, List[str], int, int, int]:
    deleted: List[str] = []
    marked_done = 0
    reset_missing_output = 0
    reset_in_progress = 0
    updated = df.copy().reset_index(drop=True)

    for idx, row in updated.iterrows():
        json_path = json_path_for_clip(output_folder, str(row["file_names"]))
        state = float(row["state"])

        if json_path.is_file():
            if json_is_valid(json_path):
                if state not in SUCCESS_STATES:
                    updated.at[idx, "state"] = 1.0
                    marked_done += 1
                continue
            deleted.append(json_path.name)
            if not dry_run:
                json_path.unlink()
            updated.at[idx, "state"] = -1.0
            continue

        if state in SUCCESS_STATES:
            updated.at[idx, "state"] = -1.0
            reset_missing_output += 1
        elif state == 0.0:
            updated.at[idx, "state"] = -1.0
            reset_in_progress += 1

    return updated, deleted, marked_done, reset_missing_output, reset_in_progress


def repair_index_files(
    input_folder: Path,
    index_folder: Path,
    output_folder: Path,
    num_tasks: int,
    video_suffixes: Sequence[str],
    dry_run: bool,
) -> List[ShardReport]:
    shards = expected_index_shards(input_folder, num_tasks, video_suffixes)
    reports: List[ShardReport] = []

    for shard_id, expected in enumerate(shards):
        path = index_folder / f"index_file_{shard_id:03d}.csv"
        existing, unreadable = load_index_csv(path)
        report = ShardReport(
            path=path,
            unreadable=unreadable or existing is None,
            existing_rows=0 if existing is None else len(existing),
            expected_rows=len(expected),
        )
        if existing is not None:
            report.names_mismatch = not names_match(existing, expected)
        else:
            report.names_mismatch = True

        if report.unreadable or report.names_mismatch:
            repaired, preserved = rebuild_shard_frame(expected, existing)
            report.preserved_states = preserved
        else:
            repaired = existing.copy()
            report.preserved_states = len(existing)

        repaired, deleted, marked_done, reset_missing, reset_in_progress = apply_output_checks(
            repaired, output_folder, dry_run=dry_run
        )
        report.json_deleted = deleted
        report.marked_done = marked_done
        report.reset_missing_output = reset_missing
        report.reset_in_progress = reset_in_progress

        original_names = [] if existing is None else existing["file_names"].tolist()
        original_states = [] if existing is None else [float(v) for v in existing["state"].tolist()]
        new_names = repaired["file_names"].tolist()
        new_states = [float(v) for v in repaired["state"].tolist()]
        report.index_rewritten = original_names != new_names or original_states != new_states

        if report.index_rewritten and not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            repaired.to_csv(path, index=False)

        reports.append(report)
    return reports


def print_report(reports: List[ShardReport], dry_run: bool) -> None:
    prefix = "[dry-run] " if dry_run else ""
    print(f"{prefix}Checked {len(reports)} index file(s)")

    regenerated = 0
    deleted_json = 0
    for report in reports:
        reasons = []
        if report.unreadable:
            reasons.append("empty/unreadable")
        if report.names_mismatch:
            reasons.append(
                f"names mismatch (got {report.existing_rows}, expected {report.expected_rows})"
            )
        if not reasons and report.index_rewritten:
            reasons.append("states updated from outputs")

        action = "would rewrite" if dry_run else "rewrote"
        if report.would_regenerate_index or report.index_rewritten:
            regenerated += 1
            print(f"  {report.path.name}: {action} ({', '.join(reasons) or 'updated'})")
        else:
            print(f"  {report.path.name}: OK ({report.expected_rows} rows)")

        print(f"    preserved states from old file: {report.preserved_states}")
        print(f"    valid JSON marked processed: {report.marked_done}")
        print(f"    missing JSON, reset 1/2 -> -1: {report.reset_missing_output}")
        print(f"    in-progress 0 -> -1: {report.reset_in_progress}")
        print(f"    corrupt JSON {'would be removed' if dry_run else 'removed'}: {len(report.json_deleted)}")
        for name in report.json_deleted[:20]:
            print(f"      {name}")
        if len(report.json_deleted) > 20:
            print(f"      ... {len(report.json_deleted) - 20} more")
        deleted_json += len(report.json_deleted)

    print(
        f"{prefix}Index files to regenerate: {regenerated}; "
        f"corrupt JSON files: {deleted_json}"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair truncated or conflicting index CSVs using the same deterministic "
            "split as create_index_files. Validates prediction JSON files and resets "
            "rows whose outputs are missing or unreadable."
        )
    )
    parser.add_argument(
        "--index-folder",
        type=Path,
        required=True,
        help="Folder with index_file_XXX.csv files",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Folder with prediction JSON files (cropped_keypoints)",
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        required=True,
        help="Folder with source clips; needed to rebuild expected shard names",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        required=True,
        help="Number of index shards / workers (same as --num_index_files)",
    )
    parser.add_argument(
        "--video-suffixes",
        nargs="*",
        default=[".mp4"],
        help="Video suffixes used when the index was created (default: .mp4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which index files and JSON outputs would be regenerated, without writing",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.num_tasks < 1:
        raise SystemExit("--num-tasks must be >= 1")
    if not args.input_folder.is_dir():
        raise SystemExit(f"Input folder not found: {args.input_folder}")
    if not args.output_folder.is_dir():
        raise SystemExit(f"Output folder not found: {args.output_folder}")

    args.index_folder.mkdir(parents=True, exist_ok=True)
    reports = repair_index_files(
        input_folder=args.input_folder.resolve(),
        index_folder=args.index_folder.resolve(),
        output_folder=args.output_folder.resolve(),
        num_tasks=args.num_tasks,
        video_suffixes=args.video_suffixes,
        dry_run=args.dry_run,
    )
    print_report(reports, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
