import logging
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision
from scipy.optimize import linear_sum_assignment
from typing import List, Sequence
from ultralytics import YOLO

BaseOptions = mp.tasks.BaseOptions


def create_models(
    checkpoint_folder: str,
    min_confidence: float = 0.4,
    yolo_model_name: str = "yolov8n-pose.pt",
) -> tuple:
    # mediapipe
    num_poses = 1
    hand_model_path = os.path.join(checkpoint_folder, 'hand_landmarker.task')
    pose_model_path = os.path.join(checkpoint_folder, 'pose_landmarker_full.task')
    face_model_path = os.path.join(checkpoint_folder, 'face_landmarker.task')
    yolo_model_path = os.path.join(checkpoint_folder, yolo_model_name)

    # yolo model
    yolo_model = YOLO(yolo_model_path)

    # hand detector
    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        num_hands=num_poses * 2)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    # body pose detector
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        num_poses=num_poses
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    # face detector
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        num_faces=num_poses
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    return hand_detector, pose_detector, face_detector, yolo_model


def new_bbox(image, keypoints, lsi=5, rsi=6, sign_space=5):
    h, w = image.shape[:2]
    l_shoulder = keypoints[lsi]
    r_shoulder = keypoints[rsi]
    distance = np.sqrt((l_shoulder[0] - r_shoulder[0]) ** 2 + (l_shoulder[1] - r_shoulder[1]) ** 2)

    center_x = np.abs(l_shoulder[0] - r_shoulder[0]) / 2 + np.min([l_shoulder[0], r_shoulder[0]], 0)
    center_y = np.abs(l_shoulder[1] - r_shoulder[1]) / 2 + np.min([l_shoulder[1], r_shoulder[1]], 0)

    new_x0 = center_x - (distance * (sign_space / 2))
    new_x1 = center_x + (distance * (sign_space / 2))
    new_y0 = center_y - (distance * (sign_space / 2))
    new_y1 = center_y + (distance * (sign_space / 2))

    idx_x = keypoints[:, 0] > 0
    idx_y = keypoints[:, 1] > 0
    new_x0 = np.min([new_x0, *keypoints[idx_x, 0]])
    new_x1 = np.max([new_x1, *keypoints[idx_x, 0]])
    new_y0 = np.min([new_y0, *keypoints[idx_y, 1]])
    new_y1 = np.max([new_y1, *keypoints[idx_y, 1]])

    new_x0 = np.round(np.clip(new_x0, 0, w)).astype(int)
    new_x1 = np.round(np.clip(new_x1, 0, w)).astype(int)
    new_y0 = np.round(np.clip(new_y0, 0, h)).astype(int)
    new_y1 = np.round(np.clip(new_y1, 0, h)).astype(int)

    return new_x0, new_y0, new_x1, new_y1


def mdeiapipe_to_xy(data, image_size=None):
    """image_size: (height, width)"""
    x = np.array([kp.x for kp in data])
    y = np.array([kp.y for kp in data])

    if image_size is not None:
        x = x * image_size[1]
        y = y * image_size[0]

    return x, y


def yolo_predict(image: np.ndarray, model, min_conf: float = 0):
    yolo_results = model(image, verbose=False)

    bboxes = yolo_results[0].boxes.xyxy
    keypoints = yolo_results[0].keypoints.xy
    bboxes = bboxes.cpu().numpy()
    keypoints = keypoints.cpu().numpy()

    conf = yolo_results[0].boxes.conf
    conf = conf.cpu().numpy()
    select_mask_kp = np.sum(keypoints, axis=(1, 2)) > 0.0001
    select_mask_bb = conf > min_conf
    select_mask = select_mask_kp & select_mask_bb

    conf = conf[select_mask]
    bboxes = bboxes[select_mask]
    keypoints = keypoints[select_mask]

    return bboxes, keypoints, conf


def keypoints_out_format(mp_keypoints, image_size):
    """image_size = (ih, iw)"""
    if len(mp_keypoints) >= 1:
        data = mp_keypoints[0]
        x, y = mdeiapipe_to_xy(data, image_size)
        z = np.array([kp.z for kp in data])
        visibility = np.array([kp.visibility for kp in data])
        data = np.array([x, y, z, visibility], dtype=float).T
        return data
    else:
        return []


def distance_matrix(P, Q):
    dis_max = np.zeros([len(P), len(Q)])
    for i, p in enumerate(P):
        for j, q in enumerate(Q):
            dist = np.linalg.norm(np.array(p) - np.array(q))
            dis_max[i, j] = dist
    return dis_max


def process_hands(mp_hand_keypoints, mp_handedness, pose_keypoints, image_size, yolo_pose_keypoints=None):
    out = {"left": [], "right": []}

    if len(mp_hand_keypoints) == 0:
        return out

    hand_keypoints = []
    for data in mp_hand_keypoints:
        hand_keypoints.append(keypoints_out_format([data], image_size))

    if len(mp_hand_keypoints) == 1:
        side = mp_handedness[0][0].category_name.lower()
        out[side] = hand_keypoints[0]
        return out

    hand_centers = []
    for keypoints in hand_keypoints:
        x = keypoints[0, 0]
        y = keypoints[0, 1]
        hand_center = [x, y]
        hand_centers.append(hand_center)

    left_wrist = None
    right_wrist = None

    pose_keypoints = None if len(pose_keypoints) == 0 else pose_keypoints
    if pose_keypoints is not None:
        left_wrist = pose_keypoints[15, :2]
        right_wrist = pose_keypoints[16, :2]

    elif pose_keypoints is None and yolo_pose_keypoints is not None:
        left_wrist = yolo_pose_keypoints[9, :2]
        right_wrist = yolo_pose_keypoints[10, :2]
        if (np.sum(left_wrist) == 0) or (np.sum(right_wrist) == 0):
            left_wrist = None
            right_wrist = None

    if left_wrist is not None and right_wrist is not None:
        wrists = [left_wrist, right_wrist]

        dis_max = distance_matrix(wrists, hand_centers)
        row_idx, col_idx = linear_sum_assignment(dis_max)

        sides = list(out.keys())
        for ridx, cidx in zip(row_idx, col_idx):
            side = sides[ridx]
            keypoints = hand_keypoints[cidx]
            out[side] = keypoints
    else:
        hand_centers_x = np.array(hand_centers)[:, 0]
        right_idx = np.argmin(hand_centers_x)
        out["right"] = hand_keypoints[right_idx]
        left_idx = np.argmax(hand_centers_x)
        if right_idx != left_idx:
            out["left"] = hand_keypoints[left_idx]

    return out


def create_debug_image(image: np.ndarray, prediction: dict, idx: int):
    colors = {
        'pose_landmarks': [50, 50, 200],
        'right_hand_landmarks': [0, 0, 0],
        'left_hand_landmarks': [255, 255, 255],
        'face_landmarks': [200, 50, 50]
    }
    for name in prediction["results"]["cropped_keypoints"][idx]:
        if prediction["results"]["cropped_keypoints"][idx][name] is None:
            continue
        for kp in prediction["results"]["cropped_keypoints"][idx][name]:
            image = cv2.circle(
                image,
                np.round(kp[:2]).astype(int),
                3,
                colors[name],
                thickness=-1
            )
    bbox_names = ["bbox_left_hand", "bbox_right_hand", "bbox_face"]
    for bbox_name in bbox_names:
        bbox = prediction["results"][bbox_name][idx]
        if len(bbox) > 0:
            image = cv2.rectangle(image, np.round(bbox[:2]).astype(int),
                                  np.round(bbox[2:]).astype(int), [50, 50, 200], 3)
    return image


def list_input_videos(
    input_folder: str,
    video_suffixes: Sequence[str] = (".mp4",),
) -> List[str]:
    """Return sorted video file names used when creating index shards."""
    suffixes = tuple(s if s.startswith(".") else f".{s}" for s in video_suffixes)
    file_names = [
        file_name
        for file_name in os.listdir(input_folder)
        if any(file_name.lower().endswith(suffix.lower()) for suffix in suffixes)
    ]
    file_names.sort()
    return file_names


def split_index_file(index_file: pd.DataFrame, num_index_files: int) -> List[pd.DataFrame]:
    """Split an index DataFrame across workers. Must stay in sync with create_index_files."""
    if num_index_files < 1:
        raise ValueError(f"num_index_files must be >= 1, got {num_index_files}")
    if num_index_files == 1:
        return [index_file]

    num_files = len(index_file)
    step = int(np.round(num_files / num_index_files))
    logging.debug(f"Files: {num_files},  Step: {step}")

    index_file_split = []
    for i in range(1, num_index_files):
        start = (i - 1) * step
        end = i * step
        index_file_split.append(index_file[start:end])
        logging.debug(f"{i} {start} {end}")
    logging.debug(f"{i} {end} {num_files}")
    index_file_split.append(index_file[end::])
    return index_file_split


def expected_index_shards(
    input_folder: str,
    num_index_files: int,
    video_suffixes: Sequence[str] = (".mp4",),
) -> List[pd.DataFrame]:
    """Build the deterministic index shards that create_index_files would write."""
    file_names = list_input_videos(input_folder, video_suffixes=video_suffixes)
    index_file = pd.DataFrame({"file_names": file_names, "state": [-1] * len(file_names)})
    return split_index_file(index_file, num_index_files)


def create_index_files(
    input_folder: str,
    output_folder: str,
    num_index_files: int,
    video_suffixes: Sequence[str] = (".mp4",),
) -> List[str]:
    """Create index CSV files listing videos to process, split across workers."""
    index_file_split = expected_index_shards(
        input_folder, num_index_files, video_suffixes=video_suffixes
    )

    logging.info("Saving index files:")
    paths = []
    for i, file in enumerate(index_file_split):
        path = os.path.join(output_folder, f"index_file_{i:03d}.csv")
        file.to_csv(path, index=False)
        paths.append(path)
        logging.debug(path)
    return paths
