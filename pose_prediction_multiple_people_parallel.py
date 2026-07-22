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
from scipy.optimize import linear_sum_assignment

from utils import crop_pad_image, load_video_cv, get_state_counts
from utils_predict import (
    create_debug_image as _create_debug_image,
    create_index_files,
    create_models,
    keypoints_out_format,
    mdeiapipe_to_xy,
    new_bbox,
    process_hands,
    yolo_predict,
)

logging.basicConfig(level=logging.INFO)


class Tracker:
    def __init__(self, max_disappeared=5, iou_threshold=0.3, keypoint_threshold=0.5):
        self.next_id = 0
        self.tracks = {}  # {id: {"bbox": last_bbox, "keypoints": last_keypoints, "disappeared": 0}}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.keypoint_threshold = keypoint_threshold

    def iou(self, boxA, boxB):
        """Compute IoU between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        denom = boxAArea + boxBArea - interArea
        if denom <= 0:
            return 0.0
        return interArea / float(denom)

    def keypoint_distance(self, kptsA, kptsB):
        """Compute mean Euclidean distance between corresponding keypoints."""
        kptsA = np.array(kptsA)
        kptsB = np.array(kptsB)
        return np.linalg.norm(kptsA - kptsB, axis=1).mean()

    def update(self, detections):
        """Update tracking with new detections."""
        new_bboxes = [det["bbox"] for det in detections]
        new_keypoints = [det["keypoints"] for det in detections]

        if len(self.tracks) == 0:
            # Initialize tracks
            for bbox, kpts in zip(new_bboxes, new_keypoints):
                self.tracks[self.next_id] = {"bbox": bbox, "keypoints": kpts, "disappeared": 0}
                self.next_id += 1
            return self.tracks

        # Prepare cost matrix for Hungarian algorithm
        existing_ids = list(self.tracks.keys())
        existing_bboxes = [self.tracks[tid]["bbox"] for tid in existing_ids]
        existing_keypoints = [self.tracks[tid]["keypoints"] for tid in existing_ids]

        cost_matrix = np.zeros((len(existing_bboxes), len(new_bboxes)))

        for i, (bboxA, kptsA) in enumerate(zip(existing_bboxes, existing_keypoints)):
            for j, (bboxB, kptsB) in enumerate(zip(new_bboxes, new_keypoints)):
                iou_score = self.iou(bboxA, bboxB)
                keypoint_dist = self.keypoint_distance(kptsA, kptsB)
                
                if iou_score < self.iou_threshold:
                    cost_matrix[i, j] = 1e6  # Large cost for mismatches
                else:
                    cost_matrix[i, j] = keypoint_dist / iou_score
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e6:
                # print(r, c, cost_matrix[r, c])
                tid = existing_ids[r]
                self.tracks[tid] = {"bbox": new_bboxes[c], "keypoints": new_keypoints[c], "disappeared": 0}
                assigned.add(c)

        # Mark unassigned existing tracks as disappeared
        for tid in existing_ids:
            if tid not in [existing_ids[r] for r in row_ind]:
                self.tracks[tid]["disappeared"] += 1
                    
        # Remove disappeared tracks
        self.tracks = {tid: t for tid, t in self.tracks.items() if t["disappeared"] <= self.max_disappeared}

        # Assign new IDs to unassigned detections
        for i, (bbox, kpts) in enumerate(zip(new_bboxes, new_keypoints)):
            if i not in assigned:
                self.tracks[self.next_id] = {"bbox": bbox, "keypoints": kpts, "disappeared": 0}
                self.next_id += 1

        return self.tracks

def compute_hand_movement(keypoints, percentile=80):
    """
    Identify the person who moves their hands the most, normalized by their torso size.
    
    Parameters:
        keypoints (numpy array): Shape (num_frames, num_people, num_keypoints, 2)
                                 Missing keypoints are represented as [0, 0].
    
    Returns:
        int: Index of the person who moved their hands the most.
    """
    num_frames, num_people, num_keypoints, _ = keypoints.shape
    
    # Define indices for hands and torso keypoints (assuming COCO format)
    left_wrist, right_wrist = 9, 10
    left_shoulder, right_shoulder, left_hip, right_hip = 5, 6, 11, 12
    
    hand_movements = np.zeros(num_people)
    hand_occurances = np.zeros(num_people)
    shoulders = [[] for p in range(num_people)]
    hand_movements_full = [[] for p in range(num_people)]
    for t in range(1, num_frames):
        for p in range(num_people):
            # Extract hand keypoints
            prev_left_hand = keypoints[t-1, p, left_wrist]
            prev_right_hand = keypoints[t-1, p, right_wrist]
            curr_left_hand = keypoints[t, p, left_wrist]
            curr_right_hand = keypoints[t, p, right_wrist]

            if np.any(prev_left_hand) and np.any(curr_left_hand):
                hand_movements_full[p].append(np.linalg.norm(curr_left_hand - prev_left_hand))
            
            if np.any(prev_right_hand) and np.any(curr_right_hand):
                hand_movements_full[p].append(np.linalg.norm(curr_right_hand - prev_right_hand))
    
    for p in range(num_people):
        if not hand_movements_full[p]:
            continue
        
        hand_movements_full[p] = np.array(hand_movements_full[p])
        hand_movements_full[p] = hand_movements_full[p][hand_movements_full[p] > 0]
        if len(hand_movements_full[p]) == 0:
            hand_movements_full[p] = 0
            continue
            
        thr = np.percentile(hand_movements_full[p], percentile)
        hand_movements_full[p] = hand_movements_full[p][hand_movements_full[p] < thr]
        hand_movements[p] = np.sum(hand_movements_full[p])
        

    # Normalize by shoulder width
    normalized_movements = np.zeros(num_people)
    for p in range(num_people):
        left_shoulder_pts = keypoints[:, p, left_shoulder]
        right_shoulder_pts = keypoints[:, p, right_shoulder]
        
        valid_left = np.any(left_shoulder_pts, axis=1)
        valid_right = np.any(right_shoulder_pts, axis=1)
        
        shoulder_dist = [] #np.zeros(num_frames)
        for t in range(num_frames):
            if valid_left[t] and valid_right[t]:
                shoulder_dist.append(np.linalg.norm(left_shoulder_pts[t] - right_shoulder_pts[t]))
        if not shoulder_dist:
            continue
        
        avg_shoulder_width = np.nanmean(shoulder_dist) #** (1/2)  # Avoid division by zero
        
        if avg_shoulder_width > 0:
            normalized_movements[p] = hand_movements[p] / avg_shoulder_width
    return np.argmax(normalized_movements)  # Return the index of the most active person


def get_signer(yolo_predictions):
    tracker = Tracker(24, iou_threshold =  0.5)

    yolo_predictions_new = []
    for yolo_prediction in yolo_predictions:
        # gathere values for tracking
        bboxes, keypoints, confs = yolo_prediction
        _detections = []
        for bbox, kp in zip(bboxes, keypoints):
            _detections.append({"bbox": bbox, "keypoints": kp})

        # update tracker and get sorted results
        tracker_result = tracker.update(_detections)
        
        # put sorted results in initial format
        n_tracks = max(tracker_result.keys()) + 1 if tracker_result else 0
        bboxes_new = [np.zeros((4)) for _ in range(n_tracks)]
        keypoints_new = [np.zeros((17, 2)) for _ in range(n_tracks)]
        confs_new = list(np.zeros(n_tracks))
        for idx in tracker_result:
            bboxes_new[idx] = tracker_result[idx]['bbox']
            keypoints_new[idx] = tracker_result[idx]['keypoints']
        yolo_predictions_new.append([bboxes_new, keypoints_new, confs_new])

    # Pad every frame to the same track count. Track IDs grow over time and some
    # frames can be empty, so lengths differ and np.array(...) would otherwise fail
    # with an inhomogeneous shape (e.g. (num_frames,) + inhomogeneous part).
    n_tracks = tracker.next_id
    if n_tracks == 0:
        return [[[], [], []] for _ in yolo_predictions_new]

    for frame_idx, (bboxes, keypoints, confs) in enumerate(yolo_predictions_new):
        pad = n_tracks - len(keypoints)
        if pad > 0:
            bboxes = list(bboxes) + [np.zeros((4)) for _ in range(pad)]
            keypoints = list(keypoints) + [np.zeros((17, 2)) for _ in range(pad)]
            confs = list(confs) + [0.0] * pad
            yolo_predictions_new[frame_idx] = [bboxes, keypoints, confs]

    # select person with most wrist movement
    clip_keypoints = [kp for _, kp, _ in yolo_predictions_new]
    top_move_idx = compute_hand_movement(np.array(clip_keypoints))

    yolo_predictions_selected = []
    for yolo_prediction in yolo_predictions_new:
        bboxes, keypoints, confs = yolo_prediction
        yolo_predictions_selected.append([[bboxes[top_move_idx]], [keypoints[top_move_idx]], [confs[top_move_idx]]])
        
    return yolo_predictions_selected


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

    if len(results["images"]) == 0:
        return {"state": -3, "results": None}

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
    multiple_people = False
    if not single_person:
        yolo_predictions = get_signer(yolo_predictions)
        multiple_people = True
        # return {"state": -2, "results": None}

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
        if sum([_x0, _y0, _x1, _y1]) == 0:
            continue

        x0.append(_x0)
        y0.append(_y0)
        x1.append(_x1)
        y1.append(_y1)

    if len(x0) == 0:
        ih, iw = video[0].shape[:2]
        x0y = 0
        y0y = 0
        x1y = iw
        y1y = ih
    else:
        x0y = np.round(np.median(x0)).astype(int)
        y0y = np.round(np.median(y0)).astype(int)
        x1y = np.round(np.median(x1)).astype(int)
        y1y = np.round(np.median(y1)).astype(int)
    
    # mediapipe predict + signing space
    mp_predictions = []
    x0, y0, x1, y1 = [], [], [], []
    for idx, (image, prediction) in enumerate(zip(results["images"], yolo_predictions)):
        # mask image if multiple people detected
        yolo_image = image.copy()
        _, keypoints, _ = prediction
        if multiple_people and len(keypoints) == 1:
            _x0, _y0, _x1, _y1 = new_bbox(image, keypoints[0][:11], lsi=5, rsi=6, sign_space=2)
            yolo_image[:, :_x0] = 0 
            yolo_image[:, _x1:] = 0 
            yolo_image[:_y0, :] = 0 
            yolo_image[_y1:, :] = 0
        
        yolo_image = yolo_image[y0y:y1y, x0y:x1y]

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
    state = 1
    if multiple_people:
        state = 2
    return {"state": state, "results": results}


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--input_folder', type=str, help="Folder with clips.")
    parser.add_argument('--output_folder', type=str, help="Folder where to save cropped clips.")
    parser.add_argument('--index_path', type=str, help="Path to folder with index files, if index files does not "
                                                       "exist, they will be created.")
    parser.add_argument('--index_file_id', type=int, help="Id of specific index file. If not provided file "
                                                                      "will be chosen randomly form index_path after "
                                                                      "each clip.")
    parser.add_argument('--num_index_files', type=int, default=1, help="Number of index files to generate.")
    parser.add_argument('--video_suffixes', type=str, nargs='*',default=['.mp4'],
        help="Video file suffixes to include when creating index files (e.g. .mp4 .avi .mov).",
    )
    parser.add_argument('--tmp_folder', type=str, help="If provided, cropped clips are first saved in this folder and "
                                                       "than copied to input_folder.")
    parser.add_argument('--checkpoint_folder', default="", type=str, help="Path to folder with MediaPipe checkpoints.")
    parser.add_argument('--sign_space', type=float, default=4, help="Size of the signing space (n * "
                                                                  "distance_between_shoulders)")
    parser.add_argument('--yolo_sign_space', type=float, default=4, help="Size of the signing space (n * "
                                                                  "distance_between_shoulders)")
    parser.add_argument('--debug', action='store_true', default=False, help="Save clip with predicted keypoints.")
    parser.add_argument('--json_only', action='store_true', default=False,
                        help="Save only JSON keypoint files, skip writing cropped videos.")

    parser.add_argument('--target_index_values', type=int, default=[], nargs="*")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    yolo_min_confidence = 0.5
    mp_min_confidence = 0.4
    single_person_frames = 0.05
    no_person_frames = 0.2

    # mediapipe
    models = create_models(args.checkpoint_folder, mp_min_confidence, yolo_model_name="yolov8m-pose.pt")

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
    
    target_index_value = args.target_index_values if args.target_index_values else [-1]
    while True:
        start_time = time.time()
        index_files = [os.path.join(index_folder, file) for file in os.listdir(index_folder) if ".csv" in file]
        index_row_idx = -1

        # create index files
        if len(index_files) == 0:
            index_files = create_index_files(input_folder, index_folder, args.num_index_files, video_suffixes=args.video_suffixes)

        # select index file
        for _ in range(len(index_files)):
            num_files = len(index_files)
            idx = random.randint(0, num_files - 1)

            # get index file path
            if args.index_file_id is not None:
                index_path = os.path.join(index_folder, f"index_file_{args.index_file_id:03d}.csv")
            else:
                index_path = index_files[idx]

            # load index file
            try:
                index_file = pd.read_csv(index_path, dtype={"file_names": str, "state": float})
                idx_list = index_file[index_file["state"].isin(target_index_value)]
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
        video_name = index_file.loc[index_row_idx, "file_names"]
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
            if prediction["state"] in [1, 2]:
                # paths
                file_name = ".".join(os.path.basename(video_path).split(".")[:-1])
                keypoints_path = os.path.join(output_folder, f"{file_name}.json")
                video_path = os.path.join(output_folder, f"{file_name}.mp4")
                if args.tmp_folder:
                    os.makedirs(args.tmp_folder, exist_ok=True)
                    keypoints_path = os.path.join(args.tmp_folder, f"{file_name}.json")
                    video_path = os.path.join(args.tmp_folder, f"{file_name}.mp4")

                # save video
                if not args.json_only:
                    h, w = prediction["results"]["cropped_size"]
                    result = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))
                    for idx in range(len(prediction["results"]["cropped_images"])):
                        cropped_image = prediction["results"]["cropped_images"][idx]
                        if args.debug:
                            cropped_image = _create_debug_image(cropped_image, prediction, idx)
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
                    if not args.json_only and not os.path.exists(os.path.join(output_folder, f"{file_name}.mp4")):
                        _ = shutil.move(video_path, output_folder)
                    shutil.rmtree(args.tmp_folder)
                os.chmod(os.path.join(output_folder, f"{file_name}.json"), 0o0666)
                if not args.json_only:
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

        if prediction is not None and prediction["results"] is not None:
            run_stats["all_frames"].append(len(prediction["results"]["keypoints"]))
            if prediction['state'] in [1, 2]:
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
