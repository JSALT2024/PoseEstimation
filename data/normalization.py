import numpy as np
from typing import Tuple


def get_keypoints(joints, landmarks_name):
    frames_keypoints = joints[landmarks_name]
    frames_keypoints = frames_keypoints[:, :, :2]

    return frames_keypoints


def output_keypoints(joints, valid_frames, frames_keypoints):
    frames_names = np.array(list(joints.keys()))[valid_frames]
    frames_keypoints = frames_keypoints[valid_frames]

    return dict(zip(frames_names, frames_keypoints))


def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def local_keypoint_normalization(joints: dict, landmarks: str, select_idx: list = [], padding: float = 0.1) -> dict:
    frames_keypoints = get_keypoints(joints, landmarks)

    if select_idx:
        frames_keypoints = frames_keypoints[:, select_idx, :]

    # move to origin
    xmin = np.min(frames_keypoints[:, :, 0], axis=1)
    ymin = np.min(frames_keypoints[:, :, 1], axis=1)

    frames_keypoints[:, :, 0] -= xmin[:, np.newaxis]
    frames_keypoints[:, :, 1] -= ymin[:, np.newaxis]

    # pad to square
    xmax = np.max(frames_keypoints[:, :, 0], axis=1)
    ymax = np.max(frames_keypoints[:, :, 1], axis=1)

    dif_full = np.abs(xmax - ymax)
    dif = np.floor(dif_full / 2)

    for i in range(len(dif)):
        if xmax[i] > ymax[i]:
            ymax[i] += dif_full[i]
            frames_keypoints[i, :, 1] += dif[i]
        else:
            xmax[i] += dif_full[i]
            frames_keypoints[i, :, 0] += dif[i]

    # add padding to all sides
    side_size = np.max([xmax, ymax], axis=0)
    padding = side_size * padding

    frames_keypoints += padding[:, np.newaxis, np.newaxis]
    xmax += padding * 2
    ymax += padding * 2

    # normalize to [-1, 1]
    frames_keypoints = safe_divide(frames_keypoints, xmax[:, np.newaxis, np.newaxis])
    # frames_keypoints /= xmax[:, np.newaxis, np.newaxis]
    frames_keypoints = frames_keypoints * 2 - 1

    return frames_keypoints


def global_keypoint_normalization(
        joints: dict,
        landmarks: str,
        add_landmarks_names: list,
        face_select_idx: list = [],
        sign_area_size: tuple = (1.5, 1.5),
        l_shoulder_idx: int = 11,
        r_shoulder_idx: int = 12) -> Tuple[dict, dict]:
    frames_keypoints = get_keypoints(joints, landmarks)

    # get distance between right and left shoulder
    l_shoulder_points = frames_keypoints[:, l_shoulder_idx, :]
    r_shoulder_points = frames_keypoints[:, r_shoulder_idx, :]
    distance = np.sqrt((l_shoulder_points[:, 0] - r_shoulder_points[:, 0]) ** 2 + (
            l_shoulder_points[:, 1] - r_shoulder_points[:, 1]) ** 2)

    # get center point between shoulders
    center_x = np.abs(l_shoulder_points[:, 0] - r_shoulder_points[:, 0]) / 2 + np.min(
        [l_shoulder_points[:, 0], r_shoulder_points[:, 0]], 0)
    center_y = np.abs(l_shoulder_points[:, 1] - r_shoulder_points[:, 1]) / 2 + np.min(
        [l_shoulder_points[:, 1], r_shoulder_points[:, 1]], 0)
    sign_area_size = np.array(sign_area_size) * distance[:, np.newaxis]

    # normalize
    frames_keypoints[:, :, 0] -= center_x[:, np.newaxis]
    frames_keypoints[:, :, 1] -= center_y[:, np.newaxis]

    # frames_keypoints[:, :, 0] /= sign_area_size[:, 1, np.newaxis]
    # frames_keypoints[:, :, 1] /= sign_area_size[:, 0, np.newaxis]
    frames_keypoints[:, :, 0] = safe_divide(frames_keypoints[:, :, 0], sign_area_size[:, 0, np.newaxis])
    frames_keypoints[:, :, 1] = safe_divide(frames_keypoints[:, :, 1], sign_area_size[:, 1, np.newaxis])

    # normalize additional landmarks
    add_landmarks = {}
    for add_landmarks_name in add_landmarks_names:
        add_frames_keypoints = get_keypoints(joints, add_landmarks_name)

        if face_select_idx and add_landmarks_name == "face_landmarks":
            add_frames_keypoints = add_frames_keypoints[:, face_select_idx, :]

        add_frames_keypoints[:, :, 0] -= center_x[:, np.newaxis]
        add_frames_keypoints[:, :, 1] -= center_y[:, np.newaxis]

        # add_frames_keypoints[:, :, 0] /= sign_area_size[:, 1, np.newaxis]
        # add_frames_keypoints[:, :, 1] /= sign_area_size[:, 0, np.newaxis]
        add_frames_keypoints[:, :, 0] = safe_divide(add_frames_keypoints[:, :, 0], sign_area_size[:, 0, np.newaxis])
        add_frames_keypoints[:, :, 1] = safe_divide(add_frames_keypoints[:, :, 1], sign_area_size[:, 1, np.newaxis])

        add_landmarks[add_landmarks_name] = add_frames_keypoints

    return frames_keypoints, add_landmarks


def yasl_keypoint_normalization(frames_keypoints: np.ndarray, select_idx: list = []) -> np.ndarray:
    """Normalize all values to be in a unit box across the clip."""
    if select_idx:
        frames_keypoints = frames_keypoints[:, select_idx, :]

    # move to origin
    xmin = np.min(frames_keypoints[:, :, 0])
    ymin = np.min(frames_keypoints[:, :, 1])

    frames_keypoints[:, :, 0] -= xmin
    frames_keypoints[:, :, 1] -= ymin

    # normalize to [0, 1]
    xmax = np.max(frames_keypoints[:, :, 0])
    ymax = np.max(frames_keypoints[:, :, 1])

    frames_keypoints[:, :, 0] = safe_divide(frames_keypoints[:, :, 0], xmax)
    frames_keypoints[:, :, 1] = safe_divide(frames_keypoints[:, :, 1], ymax)

    return frames_keypoints


def yasl_keypoint_normalization2(frames_keypoints: np.ndarray, select_idx: list = []) -> np.ndarray:
    """Normalize all values to be in a unit box in each frame."""
    if select_idx:
        frames_keypoints = frames_keypoints[:, select_idx, :]

    # move to origin
    xmin = np.min(frames_keypoints[:, :, 0], axis=1)
    ymin = np.min(frames_keypoints[:, :, 1], axis=1)

    frames_keypoints[:, :, 0] -= xmin[:, np.newaxis]
    frames_keypoints[:, :, 1] -= ymin[:, np.newaxis]

    # normalize to [0, 1]
    xmax = np.max(frames_keypoints[:, :, 0], axis=1)
    ymax = np.max(frames_keypoints[:, :, 1], axis=1)

    frames_keypoints[:, :, 0] = safe_divide(frames_keypoints[:, :, 0], xmax[:, np.newaxis])
    frames_keypoints[:, :, 1] = safe_divide(frames_keypoints[:, :, 1], ymax[:, np.newaxis])

    return frames_keypoints
