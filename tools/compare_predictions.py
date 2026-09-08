#!/usr/bin/env python3
"""Compare prediction folders: clip overlap, index state counts, and overlay viz."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".avi", ".mkv")
# Bright BGR colors so overlays stay visible after subplot downscaling
KEYPOINT_COLORS_BGR = {
    "pose_landmarks": (0, 165, 255),       # orange
    "right_hand_landmarks": (255, 0, 255),  # magenta
    "left_hand_landmarks": (255, 255, 0),   # cyan
    "face_landmarks": (0, 255, 255),        # yellow
}
SIGN_SPACE_BGR = (0, 255, 0)       # green
YOLO_SIGN_SPACE_BGR = (255, 255, 0)  # cyan
DISPLAY_WIDTH = 480
STATE_LABELS = {
    -1.0: "pending",
    0.0: "in_progress",
    1.0: "success",
    2.0: "success_multi_person",
    -2.0: "multi_person_reject",
    -3.0: "empty_or_no_person",
}


def resolve_folder(input_path: Path, folder: str) -> Path:
    path = Path(folder)
    if path.is_absolute():
        return path
    return input_path / folder


def collect_clips(folder: Path) -> Set[str]:
    kp_dir = folder / "cropped_keypoints"
    if not kp_dir.is_dir():
        return set()
    return {p.stem for p in kp_dir.glob("*.json")}


def index_state_counts(folder: Path) -> Optional[pd.Series]:
    index_dir = folder / "_index"
    if not index_dir.is_dir():
        return None
    csv_paths = sorted(index_dir.glob("*.csv"))
    if not csv_paths:
        return None
    frames = [
        pd.read_csv(path, dtype={"file_names": str, "state": float}) for path in csv_paths
    ]
    combined = pd.concat(frames, ignore_index=True)
    return combined["state"].value_counts(dropna=False).sort_index()


def format_state_counts(counts: Optional[pd.Series]) -> str:
    if counts is None:
        return "  (no _index/*.csv found)\n"
    lines = []
    for state, count in counts.items():
        label = STATE_LABELS.get(float(state), "unknown")
        lines.append(f"  state {state:g} ({label}): {int(count)}")
    return "\n".join(lines) + "\n"


def build_report(
    folder_names: Sequence[str],
    folder_paths: Sequence[Path],
    clip_sets: Sequence[Set[str]],
) -> str:
    lines: List[str] = []
    lines.append("Prediction folder comparison report")
    lines.append("=" * 40)
    lines.append("")

    lines.append("Clip counts (cropped_keypoints/*.json)")
    lines.append("-" * 40)
    for name, clips in zip(folder_names, clip_sets):
        lines.append(f"  {name}: {len(clips)}")
    lines.append("")

    lines.append("Pairwise overlap")
    lines.append("-" * 40)
    for i in range(len(folder_names)):
        for j in range(i + 1, len(folder_names)):
            overlap = clip_sets[i] & clip_sets[j]
            lines.append(
                f"  {folder_names[i]} AND {folder_names[j]}: {len(overlap)}"
            )
    lines.append("")

    lines.append("Index state value counts")
    lines.append("-" * 40)
    for name, path in zip(folder_names, folder_paths):
        lines.append(f"{name}:")
        lines.append(format_state_counts(index_state_counts(path)).rstrip("\n"))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def find_video_path(original_clips: Path, stem: str) -> Optional[Path]:
    for suffix in VIDEO_SUFFIXES:
        candidate = original_clips / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    # Case-insensitive fallback
    if not original_clips.is_dir():
        return None
    lower_stem = stem.lower()
    for path in original_clips.iterdir():
        if path.is_file() and path.stem.lower() == lower_stem:
            return path
    return None


def count_video_frames(video_path: Path) -> int:
    """Count frames by sequential decode (CAP_PROP_FRAME_COUNT is often wrong)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def read_frames_at_indices(
    video_path: Path, frame_indices: Sequence[int]
) -> List[Optional[np.ndarray]]:
    """Sequentially decode and collect selected frames (seek is unreliable near EOF)."""
    if not frame_indices:
        return []
    wanted = set(int(i) for i in frame_indices)
    max_idx = max(wanted)
    found: dict = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [None] * len(frame_indices)

    i = 0
    while i <= max_idx:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        if i in wanted:
            found[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return [found.get(int(idx)) for idx in frame_indices]


def prediction_frame_count(pred: Optional[dict]) -> Optional[int]:
    if not pred:
        return None
    keypoints = pred.get("keypoints")
    if isinstance(keypoints, list) and len(keypoints) > 0:
        return len(keypoints)
    cropped = pred.get("cropped_keypoints")
    if isinstance(cropped, list) and len(cropped) > 0:
        return len(cropped)
    return None


def _as_bbox(value) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 4:
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in value[:4]):
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    return None


def get_bbox_for_frame(pred: dict, key: str, frame_idx: int) -> Optional[List[float]]:
    """Return [x0,y0,x1,y1] for a fixed or per-frame (windowed) bbox field."""
    if key not in pred:
        return None
    value = pred[key]
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        return None
    # Fixed box for the whole clip: [x0, y0, x1, y1]
    if len(value) == 4 and all(
        isinstance(v, (int, float, np.integer, np.floating)) for v in value
    ):
        return _as_bbox(value)
    # Windowed: list of per-frame boxes
    if isinstance(value[0], (list, tuple, np.ndarray)):
        if 0 <= frame_idx < len(value):
            return _as_bbox(value[frame_idx])
        return None
    return None


def _scale_bbox(bbox: List[float], sx: float, sy: float) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    )


def draw_overlay(
    image: np.ndarray,
    pred: dict,
    frame_idx: int,
    display_width: int = DISPLAY_WIDTH,
) -> np.ndarray:
    """Resize for display, then draw bboxes/keypoints so overlays stay visible in subplots."""
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image.copy()

    sx = display_width / float(w)
    display_h = max(1, int(round(h * sx)))
    sy = display_h / float(h)
    resized = cv2.resize(image, (display_width, display_h), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

    thickness = max(2, int(round(2 * display_width / 320)))
    kp_radius = max(3, int(round(3 * display_width / 320)))
    face_radius = max(1, int(round(1.5 * display_width / 320)))

    sign_space = get_bbox_for_frame(pred, "sign_space", frame_idx)
    if sign_space is not None:
        x0, y0, x1, y1 = _scale_bbox(sign_space, sx, sy)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), SIGN_SPACE_BGR, thickness)

    yolo_ss = get_bbox_for_frame(pred, "yolo_sign_space", frame_idx)
    if yolo_ss is not None:
        x0, y0, x1, y1 = _scale_bbox(yolo_ss, sx, sy)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), YOLO_SIGN_SPACE_BGR, max(1, thickness - 1))

    keypoints = pred.get("keypoints")
    if isinstance(keypoints, list) and 0 <= frame_idx < len(keypoints):
        frame_kps = keypoints[frame_idx] or {}
        for name, color_bgr in KEYPOINT_COLORS_BGR.items():
            pts = frame_kps.get(name) if isinstance(frame_kps, dict) else None
            if not pts:
                continue
            radius = face_radius if name == "face_landmarks" else kp_radius
            for kp in pts:
                if kp is None or len(kp) < 2:
                    continue
                xy = (int(round(float(kp[0]) * sx)), int(round(float(kp[1]) * sy)))
                cv2.circle(bgr, xy, radius, color_bgr, thickness=-1)
                # Outline for contrast on any background
                cv2.circle(bgr, xy, radius, (0, 0, 0), thickness=1)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_prediction(folder: Path, stem: str) -> Optional[dict]:
    path = folder / "cropped_keypoints" / f"{stem}.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_clips(intersection: Set[str], n_videos: int, seed: int) -> List[str]:
    stems = sorted(intersection)
    if len(stems) <= n_videos:
        return stems
    rng = random.Random(seed)
    return sorted(rng.sample(stems, n_videos))


def make_subplot(
    stem: str,
    folder_names: Sequence[str],
    folder_paths: Sequence[Path],
    original_clips: Path,
    n_frames: int,
    output_path: Path,
) -> Optional[Path]:
    video_path = find_video_path(original_clips, stem)
    if video_path is None:
        print(f"  WARNING: original video not found for stem '{stem}', skipping viz")
        return None

    preds: List[Optional[dict]] = []
    for folder in folder_paths:
        pred = load_prediction(folder, stem)
        if pred is None:
            print(f"  WARNING: missing JSON for '{stem}' in {folder.name}")
        preds.append(pred)

    pred_lengths = [prediction_frame_count(p) for p in preds]
    pred_lengths = [n for n in pred_lengths if n is not None]
    # Prefer prediction length (matches overlay indices); fall back to sequential video count
    if pred_lengths:
        total_frames = min(pred_lengths)
    else:
        total_frames = count_video_frames(video_path)

    if total_frames <= 0:
        print(f"  WARNING: no usable frames for '{stem}', skipping viz")
        return None

    sample_n = min(n_frames, total_frames)
    frame_indices = np.linspace(0, total_frames - 1, num=sample_n, dtype=int).tolist()
    # Deduplicate while preserving order (short clips + many requested frames)
    deduped: List[int] = []
    seen = set()
    for idx in frame_indices:
        if idx not in seen:
            deduped.append(int(idx))
            seen.add(idx)
    frame_indices = deduped

    frames = read_frames_at_indices(video_path, frame_indices)
    # If sequential decode got fewer frames than predicted, resample from what exists
    if any(f is None for f in frames):
        readable = count_video_frames(video_path)
        if readable <= 0:
            print(f"  WARNING: could not decode video '{video_path}', skipping viz")
            return None
        usable = min(readable, total_frames)
        sample_n = min(n_frames, usable)
        frame_indices = np.linspace(0, usable - 1, num=sample_n, dtype=int).tolist()
        deduped = []
        seen = set()
        for idx in frame_indices:
            if idx not in seen:
                deduped.append(int(idx))
                seen.add(idx)
        frame_indices = deduped
        frames = read_frames_at_indices(video_path, frame_indices)
        print(
            f"  WARNING: adjusted frame sample for '{stem}' "
            f"to {len(frame_indices)} readable frames (video has {readable})"
        )

    n_rows = len(folder_names)
    n_cols = len(frame_indices)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )

    for r, (name, pred) in enumerate(zip(folder_names, preds)):
        for c, (frame_idx, frame) in enumerate(zip(frame_indices, frames)):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if frame is None:
                ax.set_facecolor("0.85")
                ax.text(0.5, 0.5, "no frame", ha="center", va="center", transform=ax.transAxes)
            elif pred is None:
                h, w = frame.shape[:2]
                sx = DISPLAY_WIDTH / float(w)
                display = cv2.resize(
                    frame,
                    (DISPLAY_WIDTH, max(1, int(round(h * sx)))),
                    interpolation=cv2.INTER_AREA,
                )
                ax.imshow(display)
                ax.text(
                    0.5,
                    0.05,
                    "no prediction",
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes,
                    color="red",
                    fontsize=8,
                )
            else:
                use_idx = frame_idx
                n_kp = prediction_frame_count(pred)
                if n_kp is not None and frame_idx >= n_kp:
                    use_idx = n_kp - 1
                overlay = draw_overlay(frame, pred, use_idx)
                ax.imshow(overlay)

            if r == 0:
                ax.set_title(f"f{frame_idx}", fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)

    fig.suptitle(stem, fontsize=12)
    fig.tight_layout()
    out_file = output_path / f"viz_{stem}.png"
    fig.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_file


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare prediction folders: report clip overlap and index state counts, "
            "then visualize N shared videos x M frames with bboxes/keypoints."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Parent directory containing prediction folders",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="Prediction folder names under input-path (or absolute paths)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory for comparison_report.txt and viz_*.png",
    )
    parser.add_argument(
        "--original-clips",
        type=Path,
        required=True,
        help="Directory with original video clips",
    )
    parser.add_argument("--n-videos", type=int, default=3, help="Number of shared videos to visualize")
    parser.add_argument("--n-frames", type=int, default=5, help="Number of evenly spaced frames per video")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for video selection")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.n_videos < 1:
        raise SystemExit("--n-videos must be >= 1")
    if args.n_frames < 1:
        raise SystemExit("--n-frames must be >= 1")

    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve()
    original_clips = args.original_clips.resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    folder_names = list(args.folders)
    folder_paths = [resolve_folder(input_path, name).resolve() for name in folder_names]

    for name, path in zip(folder_names, folder_paths):
        if not path.is_dir():
            raise SystemExit(f"Prediction folder not found: {path} (from '{name}')")

    # Use basename for display when absolute paths were given
    display_names = [Path(name).name for name in folder_names]

    clip_sets = [collect_clips(path) for path in folder_paths]
    report = build_report(display_names, folder_paths, clip_sets)
    report_path = output_path / "comparison_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote report: {report_path}")

    if not clip_sets or not any(clip_sets):
        print("No clips found; skipping visualization.")
        return 0

    intersection = set.intersection(*clip_sets)
    if not intersection:
        print("Empty all-folder intersection; skipping visualization.")
        return 0

    selected = select_clips(intersection, args.n_videos, args.seed)
    print(f"Visualizing {len(selected)} clip(s): {', '.join(selected)}")

    for stem in selected:
        out = make_subplot(
            stem=stem,
            folder_names=display_names,
            folder_paths=folder_paths,
            original_clips=original_clips,
            n_frames=args.n_frames,
            output_path=output_path,
        )
        if out is not None:
            print(f"  Saved {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
