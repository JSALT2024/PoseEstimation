from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# COCO / YOLO pose keypoint indices
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

INVALID_COST = 1e6


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])


def max_pairwise_center_distance(centers: np.ndarray) -> float:
    if len(centers) < 2:
        return 0.0
    diffs = centers[:, None, :] - centers[None, :, :]
    return float(np.max(np.linalg.norm(diffs, axis=2)))


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    denom = box_a_area + box_b_area - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / float(denom)


@dataclass
class TrackObservation:
    """Single YOLO pose detection for one person in one frame."""

    frame_idx: int
    bbox: np.ndarray
    keypoints: np.ndarray
    conf: float

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=float)
        self.keypoints = np.asarray(self.keypoints, dtype=float)


@dataclass
class Track:
    """Tracked person with per-frame history and clip-level movement metrics."""

    track_id: int
    observations: List[TrackObservation] = field(default_factory=list)

    # --- presence / confidence ---
    def frames_present(self) -> int:
        return len(self.observations)

    def coverage_frac(self, total_frames: int) -> float:
        if total_frames <= 0:
            return 0.0
        return self.frames_present() / total_frames

    def temporal_span(self) -> int:
        if not self.observations:
            return 0
        indices = [obs.frame_idx for obs in self.observations]
        return int(max(indices) - min(indices))

    def mean_confidence(self) -> float:
        if not self.observations:
            return 0.0
        return float(np.mean([obs.conf for obs in self.observations]))

    def get_observation(self, frame_idx: int) -> Optional[TrackObservation]:
        for obs in self.observations:
            if obs.frame_idx == frame_idx:
                return obs
        return None

    # --- normalization scale helpers ---
    def _shoulder_widths(self) -> List[float]:
        widths = []
        for obs in self.observations:
            left = obs.keypoints[LEFT_SHOULDER]
            right = obs.keypoints[RIGHT_SHOULDER]
            if np.any(left) and np.any(right):
                widths.append(float(np.linalg.norm(left - right)))
        return widths

    def _median_scale(self) -> float:
        diagonals = []
        for obs in self.observations:
            w = obs.bbox[2] - obs.bbox[0]
            h = obs.bbox[3] - obs.bbox[1]
            diagonals.append(float(np.sqrt(w * w + h * h)))
        if diagonals:
            return max(float(np.median(diagonals)), 1e-6)
        shoulder_widths = self._shoulder_widths()
        if shoulder_widths:
            return max(float(np.median(shoulder_widths)), 1e-6)
        return 1.0

    # --- movement and size metrics ---
    def hand_movement_norm(self, percentile: float = 80) -> float:
        """Clip-level hand activity score normalized by shoulder width.

        For each pair of consecutive tracked frames, computes Euclidean wrist
        displacement (left and right). Zero displacements and values at or above
        ``percentile`` are dropped as outliers, then the remaining displacements
        are summed and divided by mean shoulder width. If shoulders are
        unavailable, returns the unnormalized sum instead.

        Args:
            percentile: Outlier cutoff; displacements >= this percentile are
                excluded (default 80).

        Returns:
            Normalized hand movement, or 0.0 when not computable.
        """
        # Early exit if not enough observations
        if len(self.observations) < 2:
            return 0.0

        # Organize observations by frame index for fast lookup
        obs_by_frame = {obs.frame_idx: obs for obs in self.observations}
        frame_indices = sorted(obs_by_frame.keys())
        displacements = []

        # Compute wrist displacement between consecutive frames
        for prev_idx, curr_idx in zip(frame_indices[:-1], frame_indices[1:]):
            if curr_idx - prev_idx > 1:
                continue
            prev_obs = obs_by_frame[prev_idx]
            curr_obs = obs_by_frame[curr_idx]
            for wrist_idx in (LEFT_WRIST, RIGHT_WRIST):
                prev_wrist = prev_obs.keypoints[wrist_idx]
                curr_wrist = curr_obs.keypoints[wrist_idx]
                if np.any(prev_wrist) and np.any(curr_wrist):
                    displacements.append(float(np.linalg.norm(curr_wrist - prev_wrist)))

        # Exit if no displacements computed
        if not displacements:
            return 0.0

        # Convert list to array and filter to only positive displacements
        displacements = np.array(displacements)
        displacements = displacements[displacements > 0]
        if len(displacements) == 0:
            return 0.0

        # Drop outlier spikes above the given percentile
        thr = np.percentile(displacements, percentile)
        filtered = displacements[displacements < thr]
        if len(filtered) == 0:
            return 0.0

        # Get shoulder widths across observations
        shoulder_widths = self._shoulder_widths()
        if not shoulder_widths:
            # If shoulder width is not available, return sum of filtered displacements
            return float(np.sum(filtered))

        # Normalize sum of filtered displacements by average shoulder width
        avg_shoulder = float(np.mean(shoulder_widths))
        if avg_shoulder <= 0:
            return 0.0
        return float(np.sum(filtered) / avg_shoulder)

    def bbox_movement_norm(self) -> float:
        if len(self.observations) < 2:
            return 0.0

        centers = np.array([bbox_center(obs.bbox) for obs in self.observations])
        displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        scale = self._median_scale()
        return float(np.sum(displacements) / scale)

    def max_pairwise_center_dist_norm(self) -> float:
        if len(self.observations) < 2:
            return 0.0
        centers = np.array([bbox_center(obs.bbox) for obs in self.observations])
        scale = self._median_scale()
        return max_pairwise_center_distance(centers) / scale

    def mean_bbox_area_frac(self, frame_size: Tuple[int, int]) -> float:
        if not self.observations:
            return 0.0
        height, width = frame_size
        frame_area = max(height * width, 1.0)
        areas = []
        for obs in self.observations:
            w = obs.bbox[2] - obs.bbox[0]
            h = obs.bbox[3] - obs.bbox[1]
            areas.append((w * h) / frame_area)
        return float(np.mean(areas))

    def mean_center_norm(self, frame_size: Tuple[int, int]) -> Tuple[float, float]:
        if not self.observations:
            return 0.0, 0.0
        height, width = frame_size
        centers = np.array([bbox_center(obs.bbox) for obs in self.observations])
        return (
            float(np.mean(centers[:, 0] / max(width, 1.0))),
            float(np.mean(centers[:, 1] / max(height, 1.0))),
        )

    def to_summary_dict(self, total_frames: int, frame_size: Tuple[int, int]) -> dict:
        center_x, center_y = self.mean_center_norm(frame_size)
        return {
            "track_id": self.track_id,
            "frames_present": self.frames_present(),
            "coverage_frac": self.coverage_frac(total_frames),
            "temporal_span": self.temporal_span(),
            "mean_confidence": self.mean_confidence(),
            "hand_movement_norm": self.hand_movement_norm(),
            "bbox_movement_norm": self.bbox_movement_norm(),
            "max_pairwise_center_dist_norm": self.max_pairwise_center_dist_norm(),
            "mean_bbox_area_frac": self.mean_bbox_area_frac(frame_size),
            "mean_center_x": center_x,
            "mean_center_y": center_y,
            "first_frame": int(min(obs.frame_idx for obs in self.observations)) if self.observations else -1,
            "last_frame": int(max(obs.frame_idx for obs in self.observations)) if self.observations else -1,
        }


def _valid_keypoint_mask(kpts_a: np.ndarray, kpts_b: np.ndarray) -> np.ndarray:
    return np.any(kpts_a, axis=1) & np.any(kpts_b, axis=1)


def _keypoint_distance_norm(
    kpts_a: np.ndarray,
    kpts_b: np.ndarray,
    bbox_a: np.ndarray,
    bbox_b: np.ndarray,
) -> float:
    """Normalized keypoint distance between two bounding boxes."""

    mask = _valid_keypoint_mask(kpts_a, kpts_b)
    if not np.any(mask):
        return INVALID_COST

    dist = float(np.linalg.norm(kpts_a[mask] - kpts_b[mask], axis=1).mean())

    # normalize by shoulder width, or bbox diagonal as fallback
    left_a, right_a = kpts_a[LEFT_SHOULDER], kpts_a[RIGHT_SHOULDER]
    left_b, right_b = kpts_b[LEFT_SHOULDER], kpts_b[RIGHT_SHOULDER]

    scales = []
    if np.any(left_a) and np.any(right_a):
        scales.append(float(np.linalg.norm(left_a - right_a)))

    if np.any(left_b) and np.any(right_b):
        scales.append(float(np.linalg.norm(left_b - right_b)))

    if not scales:
        w = max(bbox_a[2] - bbox_a[0], bbox_b[2] - bbox_b[0], 1e-6)
        h = max(bbox_a[3] - bbox_a[1], bbox_b[3] - bbox_b[1], 1e-6)
        scales.append(float(np.sqrt(w * w + h * h)))

    scale = max(float(np.mean(scales)), 1e-6)
    return dist / scale


class PoseTracker:
    """Multi-frame pose tracker that assigns stable IDs via IoU and keypoint matching.

    Each ``update`` call associates new YOLO detections to existing active tracks
    with the Hungarian algorithm. Pairs are gated by minimum bbox IoU and maximum
    normalized keypoint distance; eligible pairs are scored by
    ``(1 - IoU) + alpha * norm_kp_dist``. Unmatched detections spawn new tracks;
    unmatched active tracks age out after ``max_disappeared`` frames.
    """

    def __init__(
        self,
        max_disappeared: int = 24,
        iou_threshold: float = 0.3,
        keypoint_dist_threshold: float = 2.0,
        keypoint_cost_alpha: float = 0.5,
    ):
        """Initialize tracker state and association thresholds.

        Args:
            max_disappeared: Remove an active track after this many consecutive
                frames without a match.
            iou_threshold: Minimum bbox IoU required for a detection/track pair
                to be considered for assignment.
            keypoint_dist_threshold: Maximum normalized keypoint distance
                (see ``_keypoint_distance_norm``) for a pair to be eligible.
            keypoint_cost_alpha: Weight of normalized keypoint distance in the
                assignment cost; IoU contributes ``(1 - IoU)``.
        """
        self.next_id = 0
        self.active_tracks: Dict[int, dict] = {}
        self.tracks: Dict[int, Track] = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.keypoint_dist_threshold = keypoint_dist_threshold
        self.keypoint_cost_alpha = keypoint_cost_alpha

    def _get_or_create_track(self, track_id: int) -> Track:
        """Return the persistent ``Track`` for ``track_id``, creating it if needed."""
        if track_id not in self.tracks:
            self.tracks[track_id] = Track(track_id=track_id)
        return self.tracks[track_id]

    def _record(
        self,
        track_id: int,
        frame_idx: int,
        bbox: np.ndarray,
        keypoints: np.ndarray,
        conf: float,
    ) -> None:
        """Append a detection to the persistent history for ``track_id``."""
        track = self._get_or_create_track(track_id)
        track.observations.append(
            TrackObservation(
                frame_idx=frame_idx,
                bbox=bbox.copy(),
                keypoints=keypoints.copy(),
                conf=float(conf),
            )
        )

    def update(self, detections: List[dict], frame_idx: int) -> Dict[int, Track]:
        """Associate detections to tracks and advance tracker state by one frame.

        On the first call (no active tracks), each detection receives a new ID.
        Otherwise, builds a gated cost matrix between active tracks and new
        detections, runs Hungarian assignment, updates matched tracks, ages
        unmatched tracks, and spawns tracks for unmatched detections.

        Args:
            detections: Per-person detections for ``frame_idx``. Each dict must
                contain ``bbox``, ``keypoints``, and ``conf``.
            frame_idx: Zero-based index of the current frame.

        Returns:
            All tracks accumulated so far (``track_id`` -> ``Track``), including
            tracks that are no longer active.
        """
        new_bboxes = [det["bbox"] for det in detections]
        new_keypoints = [det["keypoints"] for det in detections]
        new_confs = [det["conf"] for det in detections]

        # first frame: assign a new track per detection
        if len(self.active_tracks) == 0:
            for bbox, kpts, conf in zip(new_bboxes, new_keypoints, new_confs):
                self.active_tracks[self.next_id] = {
                    "bbox": bbox,
                    "keypoints": kpts,
                    "disappeared": 0,
                }
                self._record(self.next_id, frame_idx, bbox, kpts, conf)
                self.next_id += 1
            return self.tracks

        existing_ids = list(self.active_tracks.keys())
        existing_bboxes = [self.active_tracks[tid]["bbox"] for tid in existing_ids]
        existing_keypoints = [self.active_tracks[tid]["keypoints"] for tid in existing_ids]

        # build gated cost matrix: (1 - IoU) + alpha * normalized keypoint distance
        cost_matrix = np.full((len(existing_ids), len(new_bboxes)), INVALID_COST, dtype=float)
        for i, (bbox_a, kpts_a) in enumerate(zip(existing_bboxes, existing_keypoints)):
            for j, (bbox_b, kpts_b) in enumerate(zip(new_bboxes, new_keypoints)):
                iou_score = bbox_iou(bbox_a, bbox_b)
                norm_kp_dist = _keypoint_distance_norm(kpts_a, kpts_b, bbox_a, bbox_b)
                if iou_score < self.iou_threshold or norm_kp_dist > self.keypoint_dist_threshold:
                    continue
                cost_matrix[i, j] = (1.0 - iou_score) + self.keypoint_cost_alpha * norm_kp_dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_cols = set()
        matched_rows = set()

        # apply valid matches and record observations
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= INVALID_COST:
                continue
            tid = existing_ids[r]
            self.active_tracks[tid] = {
                "bbox": new_bboxes[c],
                "keypoints": new_keypoints[c],
                "disappeared": 0,
            }
            self._record(tid, frame_idx, new_bboxes[c], new_keypoints[c], new_confs[c])
            assigned_cols.add(c)
            matched_rows.add(r)

        # age out tracks that were not matched this frame
        for i, tid in enumerate(existing_ids):
            if i not in matched_rows:
                self.active_tracks[tid]["disappeared"] += 1

        self.active_tracks = {
            tid: track
            for tid, track in self.active_tracks.items()
            if track["disappeared"] <= self.max_disappeared
        }

        # spawn new tracks for unmatched detections
        for i, (bbox, kpts, conf) in enumerate(zip(new_bboxes, new_keypoints, new_confs)):
            if i not in assigned_cols:
                self.active_tracks[self.next_id] = {
                    "bbox": bbox,
                    "keypoints": kpts,
                    "disappeared": 0,
                }
                self._record(self.next_id, frame_idx, bbox, kpts, conf)
                self.next_id += 1

        return self.tracks


def track_yolo_predictions(
    yolo_predictions: List[List],
    max_disappeared: int = 24,
    iou_threshold: float = 0.3,
    keypoint_dist_threshold: float = 2.0,
    keypoint_cost_alpha: float = 0.5,
) -> Dict[int, Track]:
    """Run pose tracking over all frames of YOLO predictions."""
    tracker = PoseTracker(
        max_disappeared=max_disappeared,
        iou_threshold=iou_threshold,
        keypoint_dist_threshold=keypoint_dist_threshold,
        keypoint_cost_alpha=keypoint_cost_alpha,
    )

    # track each frame independently
    for frame_idx, yolo_prediction in enumerate(yolo_predictions):
        bboxes, keypoints, confs = yolo_prediction
        detections = []
        for bbox, kp, conf in zip(bboxes, keypoints, confs):
            detections.append({"bbox": bbox, "keypoints": kp, "conf": conf})
        tracker.update(detections, frame_idx)
        
    return tracker.tracks
