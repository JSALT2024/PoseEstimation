import argparse
import json
import logging
import os
import random
import shutil
import time
from copy import deepcopy
from typing import List

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import vision
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from utils import crop_pad_image, load_video_cv, get_state_counts

logging.basicConfig(level=logging.INFO)
VisionRunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions


def create_models(checkpoint_folder: str, min_confidence: float = 0.4) -> (object, object, object, object):
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


def create_index_files(input_folder: str, output_folder: str, num_index_files: int):
    # get names of clips and prepare index file
    file_names = os.listdir(input_folder)
    file_names = [file_name for file_name in file_names if ".mp4" in file_name]

    index_file = pd.DataFrame({"file_names": file_names, "state": [-1] * len(file_names)})
    num_files = len(index_file)
    step = int(np.round(num_files / num_index_files))
    logging.debug(f"Files: {num_files},  Step: {step}")

    # split file
    index_file_split = []
    for i in range(1, num_index_files):
        start = (i - 1) * step
        end = i * step
        index_file_split.append(index_file[start:end])
        logging.debug(f"{i} {start} {end}")
    logging.debug(f"{i} {end} {num_files}")
    index_file_split.append(index_file[end::])

    # save index files
    logging.info("Saving index files:")
    for i, file in enumerate(index_file_split):
        path = os.path.join(output_folder, f"index_file_{i:03d}.csv")
        file.to_csv(path, index=False)
        index_files.append(path)
        logging.debug(path)


def _create_debug_image(image: np.ndarray, prediction: dict):
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


def predict(
        video: List[np.ndarray],
        models: tuple,
        sign_space: int = 4,
        yolo_sign_space: int = 2,
        yolo_min_conf: float = 0.5,
        single_person_frames: float = 0.05,
        no_person_frames: float = 0.2
) -> dict:
    """
    This function processes a video to detect and extract pose, hand, and face landmarks using Mediapipe models. It also calculates the signing space and crops the images accordingly.

    Args:
        video: A list of images.
        models: A tuple containing the Mediapipe models for pose, hand, and face detection and yolo model.
        sign_space: The desired size of the signing space. Width and height calculated as shoulder distance * sign_space  Default is 4.
        yolo_sign_space: Size of yolo signing space. Can be small, mediapipe is used to detect keypoints in this crop.
        yolo_min_conf: Min confidence of yolo predictions to be used.
        single_person_frames: Portion of how many frames do not have to contain only single person.
        no_person_frames: Maximal portion of frames without people.

    Returns:
        A dictionary containing process state and results. If clip was not processed results will be None. Possible state codes:
            -1 = not processed,
             0 = ended with exception,
             1 = finished successfully,
            -2 = multiple people,
            -3 = no person

    """

    hand_detector, pose_detector, face_detector, yolo_model = models
    results = {
        "images": video,
        "keypoints": [],
        "cropped_images": [],
        "cropped_keypoints": [],
        "bbox_left_hand": [],
        "bbox_right_hand": [],
        "bbox_face": [],
        "yolo_sign_space": None,
        "sign_space": None,
        "size": None,
        "cropped_size": None,
    }

    # yolo predict + crop images
    yolo_predictions = []
    num_predictions = []
    for idx, image in enumerate(results["images"]):
        bboxes, keypoints, confs = yolo_predict(image, yolo_model, yolo_min_conf)
        yolo_predictions.append([bboxes, keypoints, confs])
        num_predictions.append(len(bboxes))

    # multiple people in video
    single_person = np.sum(np.array(num_predictions) <= 1) / len(num_predictions)
    logging.info(f"Single person images: {single_person:.3f}")
    single_person = (1 - single_person) < single_person_frames
    if not single_person:
        return {"state": -2, "results": None}

    # no person in video
    no_person = np.sum(np.array(num_predictions) == 0) / len(num_predictions)
    logging.info(f"No person images: {no_person:.3f}")
    no_person = no_person > no_person_frames
    if no_person:
        return {"state": -3, "results": None}

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
        oh, ow = cropped_image.shape[:2]
        cropped_image = cv2.resize(cropped_image, [512, 512])  #
        nh, nw = cropped_image.shape[:2]

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
        for name in keypoints:
            if len(keypoints[name]) == 0:
                keypoints[name] = np.array([])
                continue
            keypoints[name] = keypoints[name][:, :2]

        # move kp
        x_move = x0y
        y_move = y0y
        for name in keypoints:
            if len(keypoints[name]) > 0:
                keypoints[name][:, 0] += x_move
                keypoints[name][:, 1] += y_move

        # move kp
        x_move = pad_bbox[0] * nw / ow  # SCALE
        y_move = pad_bbox[1] * nh / oh  # SCALE
        keypoints_cropped = deepcopy(keypoints)
        for name in keypoints_cropped:
            if len(keypoints_cropped[name]) > 0:
                keypoints_cropped[name][:, 0] *= nw / ow  # SCALE
                keypoints_cropped[name][:, 1] *= nh / oh  # SCALE
                keypoints_cropped[name][:, 0] -= x_move
                keypoints_cropped[name][:, 1] -= y_move

            keypoints_cropped[name] = np.round(keypoints_cropped[name], 3).tolist()
            keypoints[name] = np.round(keypoints[name], 3) .tolist()

        # get dino crops
        name_to_keypoints = [
            ("face", keypoints_cropped["face_landmarks"]),
            ("left_hand", keypoints_cropped["left_hand_landmarks"]),
            ("right_hand", keypoints_cropped["right_hand_landmarks"])
        ]
        for name, kp in name_to_keypoints:
            if len(kp) > 0:
                kp = np.round(kp).astype(int)
                x0, x1 = np.min(kp[:, 0]), np.max(kp[:, 0])
                y0, y1 = np.min(kp[:, 1]), np.max(kp[:, 1])
                cropped_local_image, cropped_local_bbox = crop_pad_image(image, np.array([x0, y0, x1, y1]), 0.25)
                cropped_local_bbox = np.array(cropped_local_bbox).astype(int).tolist()
            else:
                cropped_local_bbox = []
            results[f"bbox_{name}"].append(cropped_local_bbox)

        # save processed data
        results["keypoints"].append(keypoints)
        results["cropped_images"].append(cropped_image)
        results["cropped_keypoints"].append(keypoints_cropped)

    results["size"] = results["images"][0].shape[:2]
    results["cropped_size"] = results["cropped_images"][0].shape[:2]
    results["yolo_sign_space"] = np.array([x0y, y0y, x1y, y1y]).astype(int).tolist()
    results["sign_space"] = np.array(pad_bbox).astype(int).tolist()
    results["images"] = None

    return {"state": 1, "results": results}


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--input_folder', type=str, help="Folder with clips.")
    parser.add_argument('--output_folder', type=str, help="Folder where to save cropped clips.")
    parser.add_argument('--index_path', type=str, help="Path to folder with index files, if index files does not "
                                                       "exist, they will be created.")
    parser.add_argument('--index_file_id', type=int, help="Id of specific index file. If not provided file "
                                                                      "will be chosen randomly form index_path after "
                                                                      "each clip.")
    parser.add_argument('--num_index_files', type=int, default="", help="Number of index files to generate.")
    parser.add_argument('--tmp_folder', type=str, help="If provided, cropped clips are first saved in this folder and "
                                                       "than copied to input_folder.")
    parser.add_argument('--checkpoint_folder', default="", type=str, help="Path to folder with MediaPipe checkpoints.")
    parser.add_argument('--sign_space', type=int, default=4, help="Size of the signing space (n * "
                                                                  "distance_between_shoulders)")
    parser.add_argument('--yolo_sign_space', type=int, default=2, help="Size of the signing space (n * "
                                                                  "distance_between_shoulders)")
    parser.add_argument('--debug', action='store_true', default=False, help="Save clip with predicted keypoints.")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    yolo_min_confidence = 0.5
    mp_min_confidence = 0.4
    single_person_frames = 0.05
    no_person_frames = 0.2

    # mediapipe
    models = create_models(args.checkpoint_folder, mp_min_confidence)

    input_folder = args.input_folder
    output_folder = args.output_folder
    index_folder = args.index_path
    os.makedirs(index_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    run_stats = {
        "all_times": [],
        "all_frames": [],
        "processed_videos": 0,
        "failed_videos": 0,
        "predictions": {
            "cropped_keypoints": [],
            "bbox_left_hand": [],
            "bbox_right_hand": [],
            "bbox_face": [],
        }
    }

    while True:
        start_time = time.time()
        index_files = [os.path.join(index_folder, file) for file in os.listdir(index_folder) if ".csv" in file]
        index_row_idx = -1

        # create index files
        if len(index_files) == 0:
            create_index_files(input_folder, index_folder, args.num_index_files)

        # select index file
        for _ in range(len(index_files)):
            num_files = len(index_files)
            idx = random.randint(0, num_files - 1)

            # get index file path
            if args.index_file_id:
                index_path = os.path.join(index_folder, f"index_file_{args.index_file_id:03d}.csv")
            else:
                index_path = index_files[idx]

            # load index file
            try:
                index_file = pd.read_csv(index_path, dtype={"file_names": str, "state": float})
                idx_list = index_file[index_file["state"] == -1]
                logging.debug(f"Unprocessed files: {len(idx_list)}")
            except Exception as e:
                logging.error(f"Loading csv failed: {e}")
                idx_list = []

            # select unprocessed file
            if len(idx_list) > 0:
                idx_list = idx_list.sample(1).index.tolist()
                index_row_idx = idx_list[0]
                logging.info(f"\n")
                logging.info(f"Processing file: {index_path}")
                logging.info(f"Row: {index_row_idx}")
                break
            index_files.pop(idx)

        if index_row_idx == -1:
            state_counts = get_state_counts(args.index_path)
            logging.info(state_counts)
            logging.info(f"All files processed exiting")
            break

        # set selected file
        index_file.at[index_row_idx, "state"] = 0
        index_file.to_csv(index_path, index=False)

        # get path to video
        video_name = index_file.iloc[index_row_idx]["file_names"]
        video_path = os.path.join(input_folder, video_name)

        # predict
        prediction = None
        try:
            video, fps = load_video_cv(video_path)
            prediction = predict(
                video,
                models,
                args.sign_space,
                args.yolo_sign_space,
                yolo_min_confidence,
                single_person_frames,
                no_person_frames
            )
            video = None

            # save predictions and video
            if prediction["state"] == 1:
                # paths
                file_name = ".".join(os.path.basename(video_path).split(".")[:-1])
                keypoints_path = os.path.join(output_folder, f"{file_name}.json")
                video_path = os.path.join(output_folder, f"{file_name}.mp4")
                if args.tmp_folder:
                    os.makedirs(args.tmp_folder, exist_ok=True)
                    keypoints_path = os.path.join(args.tmp_folder, f"{file_name}.json")
                    video_path = os.path.join(args.tmp_folder, f"{file_name}.mp4")

                # save video
                h, w = prediction["results"]["cropped_size"]
                result = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (h, w))
                for idx in range(len(prediction["results"]["cropped_images"])):
                    cropped_image = prediction["results"]["cropped_images"][idx]
                    if args.debug:
                        cropped_image = _create_debug_image(cropped_image, prediction)
                    result.write(cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                result.release()

                # save keypoints
                results = prediction["results"]
                del results["images"]
                del results["cropped_images"]
                with open(keypoints_path, "w") as f:
                    json.dump(results, f)

                # move to output folder and set access for all
                if args.tmp_folder:
                    if not os.path.exists(os.path.join(output_folder, f"{file_name}.json")):
                        _ = shutil.move(keypoints_path, output_folder)
                    if not os.path.exists(os.path.join(output_folder, f"{file_name}.mp4")):
                        _ = shutil.move(video_path, output_folder)
                    shutil.rmtree(args.tmp_folder)
                os.chmod(os.path.join(output_folder, f"{file_name}.json"), 0o0666)
                os.chmod(os.path.join(output_folder, f"{file_name}.mp4"), 0o0666)
            else:
                logging.info(f"State: {prediction['state']}. Multiple or none people in video: {video_path}")

            # save data
            index_file = pd.read_csv(index_path)
            index_file.at[index_row_idx, "state"] = prediction["state"]
            index_file.to_csv(index_path, index=False)

        except Exception as e:
            logging.error(f"Processing failed: {video_path}")
            logging.error(f"Message: {e}")

        # get run stats
        process_time = time.time() - start_time
        run_stats["all_times"].append(process_time)

        if prediction["results"] is not None:
            run_stats["all_frames"].append(len(prediction["results"]["keypoints"]))
            if prediction['state'] == 1:
                run_stats["processed_videos"] += 1
                for name in run_stats["predictions"]:
                    num_p = np.sum([1 for p in prediction['results'][name] if len(p) > 0])
                    run_stats["predictions"][name].append(num_p / len(prediction["results"]["keypoints"]))
            else:
                run_stats["failed_videos"] += 1
        else:
            run_stats["failed_videos"] += 1

        finished_videos = run_stats["processed_videos"] + run_stats["failed_videos"]
        average_frames = np.sum(run_stats["all_frames"]) / finished_videos
        logging.info(
            f"Processing time: {process_time}, "
            f"Average time: {np.mean(run_stats['all_times'])}, "
            f"Average frames: {average_frames}, "
            f"Processed videos: {run_stats['processed_videos']}, "
            f"Failed videos: {run_stats['failed_videos']}"
        )
        logging.info(f"\n")

    logging.info(f"\n")
    logging.info(f"Videos: {len(run_stats['all_times'])}")
    logging.info(f"Full processing time: {np.sum(run_stats['all_times']):.3f} s")
    logging.info(f"Average processing time: {np.mean(run_stats['all_times']):.3f} s")
    logging.info(f"Average processing time: {np.mean(run_stats['all_times']):.3f} s")
    logging.info(f"Average frames: {np.mean(run_stats['all_frames']):.3f}")
    for name in run_stats["predictions"]:
        logging.info(f"Average {name} predictions: {np.mean(run_stats['predictions'][name]):.3f}")
    state_counts = get_state_counts(args.index_path)
    logging.info(state_counts)
