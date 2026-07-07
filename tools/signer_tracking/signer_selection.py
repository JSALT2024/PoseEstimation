from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tools.signer_tracking.track import Track


@dataclass
class SignerSelectionConfig:
    """Thresholds and weights for multi-criteria signer selection."""

    # Minimum coverage, presence, and confidence required to be a signer candidate.
    min_coverage: float = 0.25
    min_frames: int = 10
    min_mean_conf: float = 0.3

    # Weights for each metric.
    weight_hand_movement: float = 0.45
    weight_coverage: float = 0.25
    weight_confidence: float = 0.15
    weight_bbox_movement: float = 0.10
    weight_bbox_area: float = 0.05

    # Thresholds and penalties for short-lived tracks and walking.
    short_span_threshold: float = 0.3
    short_span_penalty: float = 0.2
    walking_penalty: float = 0.3
    walking_bbox_movement_threshold: float = 0.5
    walking_hand_movement_threshold: float = 0.3


@dataclass
class SignerSelectionResult:
    """Output of signer selection: chosen track ID, scores, and per-track metrics."""

    track_id: int
    scores: Dict[int, float] = field(default_factory=dict)
    metrics: Dict[int, dict] = field(default_factory=dict)


def _normalize_metrics(values: Dict[int, float]) -> Dict[int, float]:
    """Min-max normalize metric values across tracks to [0, 1]."""
    if not values:
        return {}

    arr = np.array(list(values.values()), dtype=float)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))

    if max_v - min_v < 1e-9:
        return {k: 1.0 for k in values}
    
    return {k: (v - min_v) / (max_v - min_v) for k, v in values.items()}


def _compute_track_metrics(tracks: Dict[int, Track], num_frames: int, frame_size: Tuple[int, int]) -> Dict[int, dict]:
    """Compute metrics for each track."""
    metrics = {}

    for track_id, track in tracks.items():
        metrics[track_id] = track.to_summary_dict(num_frames, frame_size)

    return metrics


def _passes_filters(metrics: dict, config: SignerSelectionConfig) -> bool:
    """Minimum coverage, presence, and confidence required to be a signer candidate."""

    return (
        metrics["coverage_frac"] >= config.min_coverage
        and metrics["frames_present"] >= config.min_frames
        and metrics["mean_confidence"] >= config.min_mean_conf
    )


def _score_track(metrics: dict, num_frames: int, config: SignerSelectionConfig) -> float:
    """Score a track based on its metrics."""

    # Get normalized metrics for the track.
    norm_hand = metrics.get("_norm_hand_movement", 0.0)
    norm_coverage = metrics.get("_norm_coverage", 0.0)
    norm_conf = metrics.get("_norm_confidence", 0.0)
    norm_bbox_mov = metrics.get("_norm_bbox_movement", 0.0)
    norm_area = metrics.get("_norm_bbox_area", 0.0)

    # weighted sum of normalized criteria
    score = (
        config.weight_hand_movement * norm_hand
        + config.weight_coverage * norm_coverage
        + config.weight_confidence * norm_conf
        + config.weight_bbox_movement * norm_bbox_mov
        + config.weight_bbox_area * norm_area
    )

    # penalize short-lived tracks
    if num_frames > 0 and metrics["temporal_span"] / num_frames < config.short_span_threshold:
        score -= config.short_span_penalty
    
    # penalize walking tracks
    if metrics["bbox_movement_norm"] > config.walking_bbox_movement_threshold and metrics["hand_movement_norm"] < config.walking_hand_movement_threshold:
        score -= config.walking_penalty

    return score


def select_signer(
    tracks: Dict[int, Track],
    num_frames: int,
    frame_size: Tuple[int, int] = (720, 1280),
    config: Optional[SignerSelectionConfig] = None,
) -> SignerSelectionResult:
    """Select the best signer from the tracks."""

    if config is None:
        config = SignerSelectionConfig()

    if not tracks:
        return SignerSelectionResult(track_id=-1)

    # collect per-track metrics used for filtering and scoring
    raw_metrics = _compute_track_metrics(tracks, num_frames, frame_size)

    # require minimum coverage, frame count, and confidence
    eligible_ids = [track_id for track_id, m in raw_metrics.items() if _passes_filters(m, config)]
    candidate_ids = eligible_ids if eligible_ids else list(tracks.keys())

    # normalize each criterion across candidates
    norm_hand = _normalize_metrics({tid: raw_metrics[tid]["hand_movement_norm"] for tid in candidate_ids})
    norm_coverage = _normalize_metrics({tid: raw_metrics[tid]["coverage_frac"] for tid in candidate_ids})
    norm_conf = _normalize_metrics({tid: raw_metrics[tid]["mean_confidence"] for tid in candidate_ids})
    norm_bbox_mov = _normalize_metrics({tid: raw_metrics[tid]["bbox_movement_norm"] for tid in candidate_ids})
    norm_area = _normalize_metrics({tid: raw_metrics[tid]["mean_bbox_area_frac"] for tid in candidate_ids})

    # score each candidate track
    scores: Dict[int, float] = {}
    for track_id in candidate_ids:
        m = dict(raw_metrics[track_id])
        m["_norm_hand_movement"] = norm_hand.get(track_id, 0.0)
        m["_norm_coverage"] = norm_coverage.get(track_id, 0.0)
        m["_norm_confidence"] = norm_conf.get(track_id, 0.0)
        m["_norm_bbox_movement"] = norm_bbox_mov.get(track_id, 0.0)
        m["_norm_bbox_area"] = norm_area.get(track_id, 0.0)
        scores[track_id] = _score_track(m, num_frames, config)
        raw_metrics[track_id] = m

    if not scores:
        return SignerSelectionResult(track_id=-1, scores=scores, metrics=raw_metrics)

    # pick highest score; fall back to hand movement if no track passed filters
    if not eligible_ids:
        best_id = max(scores, key=lambda tid: raw_metrics[tid]["hand_movement_norm"])
    else:
        best_id = max(scores, key=scores.get)

    # strip internal normalized fields before returning
    clean_metrics = {
        tid: {k: v for k, v in m.items() if not k.startswith("_norm_")}
        for tid, m in raw_metrics.items()
    }
    return SignerSelectionResult(track_id=best_id, scores=scores, metrics=clean_metrics)


# rebuild per-frame YOLO output containing only the selected track
def filter_predictions_to_track(yolo_predictions: List[List], track: Track) -> List[List]:
    """Filter the YOLO predictions to only include the selected track."""

    filtered = []
    for frame_idx, _prediction in enumerate(yolo_predictions):
        obs = track.get_observation(frame_idx)
        if obs is None:
            filtered.append([[], [], []])
        else:
            filtered.append([[obs.bbox], [obs.keypoints], [obs.conf]])

    return filtered


def selection_result_to_metadata(result: SignerSelectionResult) -> dict:
    """Convert selection result to JSON-serializable metadata."""
    
    return {
        "track_id": int(result.track_id),
        "scores": {int(k): float(v) for k, v in result.scores.items()},
        "metrics": {
            int(k): {mk: (float(mv) if isinstance(mv, (int, float, np.floating)) else mv) for mk, mv in m.items()}
            for k, m in result.metrics.items()
        },
    }
