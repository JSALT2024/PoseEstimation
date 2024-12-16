import os
from copy import deepcopy
from typing import List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def crop_frame(image, bounding_box):
    x, y, w, h = bounding_box
    cropped_frame = image[y:y + h, x:x + w]
    return cropped_frame


def get_centered_box(keypoints, box_size, scale_factor=1.2):
    center_x, center_y = np.mean(keypoints, axis=0, dtype=int)
    half_size = box_size // 2
    x = center_x - half_size
    y = center_y - half_size
    w = box_size
    h = box_size

    w_padding = int((scale_factor - 1) * w / 2)
    h_padding = int((scale_factor - 1) * h / 2)
    x -= w_padding
    y -= h_padding
    w += 2 * w_padding
    h += 2 * h_padding

    return x, y, w, h


def get_bounding_box(keypoints, scale_factor=1.2):
    keypoints = np.round(keypoints).astype(int)
    x, y, w, h = cv2.boundingRect(keypoints)
    w_padding = int((scale_factor - 1) * w / 2)
    h_padding = int((scale_factor - 1) * h / 2)
    x -= w_padding
    y -= h_padding
    w += 2 * w_padding
    h += 2 * h_padding
    return x, y, w, h


def adjust_bounding_box(bounding_box, image_shape):
    x, y, w, h = bounding_box
    ih, iw, _ = image_shape

    # Adjust x-coordinate if the bounding box extends beyond the image's right edge
    if x + w > iw:
        x = iw - w

    # Adjust y-coordinate if the bounding box extends beyond the image's bottom edge
    if y + h > ih:
        y = ih - h

    # Ensure bounding box's x and y coordinates are not negative
    x = max(x, 0)
    y = max(y, 0)

    return x, y, w, h


def create_mediapipe_models(checkpoint_folder: str, min_confidence: float = 0.4) -> (object, object, object, object):
    BaseOptions = mp.tasks.BaseOptions

    # mediapipe
    num_poses = 1
    hand_model_path = os.path.join(checkpoint_folder, 'hand_landmarker.task')
    pose_model_path = os.path.join(checkpoint_folder, 'pose_landmarker_full.task')
    face_model_path = os.path.join(checkpoint_folder, 'face_landmarker.task')
    yolo_model_path = os.path.join(checkpoint_folder, "yolov8n-pose.pt")

    # yolov8
    yolo_model = YOLO(yolo_model_path)

    # define hand model
    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        num_hands=num_poses * 2)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    # define body model
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        num_poses=num_poses
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    # define face model
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        num_faces=num_poses
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    return hand_detector, pose_detector, face_detector, yolo_model


def yolo_predict(image: np.ndarray, model, min_conf: float = 0.5):
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


def load_video_cv(path: str):
    video = []

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(img)
    cap.release()
    return video, fps


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


def crop_pad_image(image: np.ndarray, bbox: np.ndarray, border: float = 0.25) -> np.ndarray:
    """Crop the image, pad to square and add a border."""
    # get bbox and image
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0

    # add padding
    dif = np.abs(w - h)
    pad_value_0 = np.floor(dif / 2).astype(int)
    pad_value_1 = dif - pad_value_0

    if w > h:
        y0 -= pad_value_0
        y1 += pad_value_1
    else:
        x0 -= pad_value_0
        x1 += pad_value_1

    border = np.round((np.max([w, h]) * border) / 2).astype(int)
    ih, iw = image.shape[:2]
    y0 -= border
    y1 += border
    x0 -= border
    x1 += border

    new_bbox = [x0, y0, x1, y1]

    y0 += ih
    y1 += ih
    x0 += iw
    x1 += iw

    image = np.pad(image, ((ih, ih), (iw, iw), (0, 0)), mode='constant', constant_values=0)  # mode="reflect"
    cropped_image = image[y0:y1, x0:x1]

    return cropped_image, new_bbox


def keypoints_out_format(mp_keypoints, image_size):
    """image_size = (ih, iw)"""
    if len(mp_keypoints) >= 1:
        data = mp_keypoints[0]
        x, y = mdeiapipe_to_xy(data, image_size)
        z = np.array([kp.z for kp in data])
        visibility = np.array([kp.visibility for kp in data])
        data = np.array([x, y, z, visibility]).T
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

    # transform keypoints
    hand_keypoints = []
    for data in mp_hand_keypoints:
        hand_keypoints.append(keypoints_out_format([data], image_size))

    if (mp_hand_keypoints) == 1:
        side = mp_handedness[0]["category_name"].lower
        out[side] = hand_keypoints[0]
        return out

    # calculate centers
    hand_centers = []
    for keypoints in hand_keypoints:
        x = keypoints[0, 0]
        y = keypoints[0, 1]
        hand_center = [x, y]
        hand_centers.append(hand_center)

    # assign hands to sides
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


def predict_pose(video: List[np.ndarray], models: tuple, sign_space=4, yolo_sign_space=4) -> dict:
    """
        This function processes a video to detect and extract pose, hand, and face landmarks using Mediapipe models.
        It also calculates the signing space and crops the images accordingly.

        Parameters:
            video (list): A list of images.
            models (tuple): A tuple containing the Mediapipe models for pose, hand, and face detection and yolo model.
            sign_space (int): The desired size of the signing space.
                              Width and height calculated as shoulder distance * sign_space  Default is 4.

        Returns:
            (dict): A dictionary containing the processed video data, including images, keypoints, cropped images, cropped keypoints,
        signing space, and bounding boxes for different body parts.
    """
    hand_detector, pose_detector, face_detector, yolo_model = models
    results = {
        "images": video,
        "keypoints": [],
        "cropped_images": [],
        "cropped_keypoints": [],
        "sign_space": [],
        "cropped_left_hand": [],
        "cropped_right_hand": [],
        "cropped_face": [],
        "bbox_left_hand": [],
        "bbox_right_hand": [],
        "bbox_face": [],
    }

    # yolo predict + crop images
    yolo_predictions = []
    num_predictions = []
    for idx, image in enumerate(results["images"]):
        bboxes, keypoints, confs = yolo_predict(image, yolo_model)
        yolo_predictions.append([bboxes, keypoints, confs])
        num_predictions.append(len(bboxes))
        
    # no predictions -> add empty values and return
    if np.sum(num_predictions) == 0:
        _h, _w = results["images"][0].shape[:2]
        for idx in range(len(results["images"])):
            results["keypoints"].append({'pose_landmarks': [], 'right_hand_landmarks': [], 'left_hand_landmarks': [], 'face_landmarks': []})
            results["cropped_images"].append(results["images"][idx])
            results["cropped_keypoints"].append({'pose_landmarks': [], 'right_hand_landmarks': [], 'left_hand_landmarks': [], 'face_landmarks': []})
            results["sign_space"].append([0, 0, _w, _h])
            results["cropped_left_hand"].append(np.zeros([224, 224, 3], dtype=np.uint8))
            results["cropped_right_hand"].append(np.zeros([224, 224, 3], dtype=np.uint8))
            results["cropped_face"].append(np.zeros([224, 224, 3], dtype=np.uint8))
            results["bbox_left_hand"].append([])
            results["bbox_right_hand"].append([])
            results["bbox_face"].append([])    
        return results

    # get signing bbox
    x0, y0, x1, y1 = [], [], [], []
    for idx, (image, prediction) in enumerate(zip(results["images"], yolo_predictions)):
        _, keypoints, _ = prediction
        if len(keypoints) != 1:
            continue

        _x0, _y0, _x1, _y1 = new_bbox(image, keypoints[0], lsi=5, rsi=6, sign_space=yolo_sign_space)

        x0.append(_x0)
        y0.append(_y0)
        x1.append(_x1)
        y1.append(_y1)

    x0y = np.round(np.median(x0)).astype(int)
    y0y = np.round(np.median(y0)).astype(int)
    x1y = np.round(np.median(x1)).astype(int)
    y1y = np.round(np.median(y1)).astype(int)

    # mediapipe predict + signing space
    mp_predictions = []
    x0, y0, x1, y1 = [], [], [], []
    for idx, image in enumerate(results["images"]):
        yolo_image = image[y0y:y1y, x0y:x1y]

        ih, iw = yolo_image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(yolo_image))

        # HACK:
        # if the YOLO model does not detect anything,
        # pretend it detects a black square and give it to mediapipe
        if yolo_image.shape == (0, 0, 3):
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.zeros(shape=(256, 256, 3), dtype=np.uint8)
            )

        pose_prediction = pose_detector.detect(mp_image)
        hand_prediction = hand_detector.detect(mp_image)
        face_prediction = face_detector.detect(mp_image)

        mp_predictions.append([hand_prediction, face_prediction, pose_prediction])

        if len(pose_prediction.pose_landmarks) != 1:
            continue

        kp_all_x = []
        kp_all_y = []
        mp_keypoints = [
            pose_prediction.pose_landmarks[0][:25],
            *face_prediction.face_landmarks,
            *hand_prediction.hand_landmarks,
        ]

        for p in mp_keypoints:
            x, y = mdeiapipe_to_xy(p, (ih, iw))
            kp_all_x.extend(x)
            kp_all_y.extend(y)
        kp_all = np.array((kp_all_x, kp_all_y)).T

        kp_all[:, 0] = kp_all[:, 0] + x0y
        kp_all[:, 1] = kp_all[:, 1] + y0y

        if len(kp_all) == 0:
            continue

        _x0, _y0, _x1, _y1 = new_bbox(image, kp_all, lsi=11, rsi=12, sign_space=sign_space)

        x0.append(_x0)
        y0.append(_y0)
        x1.append(_x1)
        y1.append(_y1)

    # create signing space as median of all signing spaces
    if len(x0) == 0:
        ih, iw = video[0].shape[:2]
        x0mp = 0
        y0mp = 0
        x1mp = iw
        y1mp = ih
    else:
        x0mp = np.round(np.median(x0)).astype(int)
        y0mp = np.round(np.median(y0)).astype(int)
        x1mp = np.round(np.median(x1)).astype(int)
        y1mp = np.round(np.median(y1)).astype(int)

    for idx, (image, prediction) in enumerate(zip(results["images"], mp_predictions)):
        yolo_image = image[y0y:y1y, x0y:x1y]
        yih, yiw = yolo_image.shape[:2]

        cropped_image, pad_bbox = crop_pad_image(image, (x0mp, y0mp, x1mp, y1mp), border=0)

        hand_prediction, face_prediction, pose_prediction = prediction

        face_keypoints = keypoints_out_format(face_prediction.face_landmarks, (yih, yiw))
        pose_keypoints = keypoints_out_format(pose_prediction.pose_landmarks, (yih, yiw))
        hand_keypoints = process_hands(
            hand_prediction.hand_landmarks,
            hand_prediction.handedness,
            pose_keypoints,
            (yih, yiw),
            None
        )

        keypoints = {
            'pose_landmarks': pose_keypoints,
            'right_hand_landmarks': hand_keypoints["right"],
            'left_hand_landmarks': hand_keypoints["left"],
            'face_landmarks': face_keypoints
        }

        # move kp
        x_move = x0y
        y_move = y0y
        for name in keypoints:
            if len(keypoints[name]) > 0:
                keypoints[name][:, 0] += x_move
                keypoints[name][:, 1] += y_move

        # get dino crops
        name_to_keypoints = [
            ("face", face_keypoints),
            ("left_hand", hand_keypoints["left"]),
            ("right_hand", hand_keypoints["right"])
        ]
        for name, kp in name_to_keypoints:
            if len(kp) > 0:
                kp = np.round(kp[:, :2]).astype(int)
                x, y, w, h = cv2.boundingRect(kp)
                cropped_local_bbox = get_centered_box(kp, np.max([w, h]), scale_factor=1.2)
                cropped_local_bbox = adjust_bounding_box(cropped_local_bbox, image.shape)
                cropped_local_image = crop_frame(image, cropped_local_bbox)
                x0, y0, w, h = cropped_local_bbox
                cropped_local_bbox = [x0, y0, x0 + w, y0 + h]

            else:
                cropped_local_image = np.zeros([224, 224, 3], dtype=np.uint8)
                cropped_local_bbox = []
            results[f"bbox_{name}"].append(cropped_local_bbox)
            results[f"cropped_{name}"].append(cropped_local_image)

        # move kp
        x_move = pad_bbox[0]
        y_move = pad_bbox[1]
        keypoints_cropped = deepcopy(keypoints)
        for name in keypoints_cropped:
            if len(keypoints_cropped[name]) > 0:
                keypoints_cropped[name][:, 0] -= x_move
                keypoints_cropped[name][:, 1] -= y_move
                keypoints_cropped[name] = np.round(keypoints_cropped[name], 3).tolist()
                keypoints[name] = np.round(keypoints[name], 3).tolist()

        # save processed data
        results["keypoints"].append(keypoints)
        results["cropped_images"].append(cropped_image)
        results["cropped_keypoints"].append(keypoints_cropped)
        results["sign_space"].append(pad_bbox)
    results["images"] = video

    return results