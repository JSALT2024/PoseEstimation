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


def rolling_median_boxes(boxes_raw, window_frames, image_shape):
    """Per-frame rolling median of bounding boxes.

    For each frame the box is the median (per coordinate) of all valid boxes within
    a window of `window_frames` centered on that frame. Frames without a valid box
    (gaps) fall back to the global median; if no valid box exists at all, the full
    frame is returned for every frame.
    """
    n = len(boxes_raw)
    ih, iw = image_shape
    valid = [b for b in boxes_raw if b is not None]
    if len(valid) == 0:
        return [(0, 0, iw, ih)] * n
    global_med = np.median(np.array(valid), axis=0)
    half = window_frames // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = [boxes_raw[j] for j in range(lo, hi) if boxes_raw[j] is not None]
        med = np.median(np.array(win), axis=0) if win else global_med
        out.append(tuple(np.round(med).astype(int).tolist()))
    return out


def predict(
        video: List[np.ndarray],
        models: tuple,
        sign_space: int = 4,
        yolo_sign_space: int = 2,
        yolo_min_conf: float = 0.5,
        single_person_frames: float = 0.05,
        no_person_frames: float = 0.2,
        window_seconds: float = 2.0,
        fps: float = 25.0
) -> dict:
    """
    This function processes a video to detect and extract pose, hand, and face landmarks using Mediapipe models. It also calculates the signing space and crops the images accordingly.

    Unlike the single-box variant, the signing space is computed per frame as a rolling
    (sliding) median over a time window, so the crop follows a moving signer without jitter.

    Args:
        video: A list of images.
        models: A tuple containing the Mediapipe models for pose, hand, and face detection and yolo model.
        sign_space: The desired size of the signing space. Width and height calculated as shoulder distance * sign_space  Default is 4.
        yolo_sign_space: Size of yolo signing space. Can be small, mediapipe is used to detect keypoints in this crop.
        yolo_min_conf: Min confidence of yolo predictions to be used.
        single_person_frames: Portion of how many frames do not have to contain only single person.
        no_person_frames: Maximal portion of frames without people.
        window_seconds: Length of the rolling window in seconds used to compute per-frame
            signing-space boxes. A value <= 0 uses the whole video (single box, original behavior).
        fps: Frames per second of the video, used to convert window_seconds to frames.

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
        "yolo_sign_space": [],
        "sign_space": [],
        "size": None,
        "cropped_size": None,
    }

    if len(results["images"]) == 0:
        return {"state": -3, "results": None}

    num_frames = len(results["images"])
    window_frames = (2 * num_frames + 1) if window_seconds <= 0 else max(1, int(round(window_seconds * fps)))

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

    # get per-frame yolo signing bbox
    yolo_boxes_raw = [None] * num_frames
    for idx, (image, prediction) in enumerate(zip(results["images"], yolo_predictions)):
        _, keypoints, _ = prediction
        if len(keypoints) != 1:
            continue

        _x0, _y0, _x1, _y1 = new_bbox(image, keypoints[0], lsi=5, rsi=6, sign_space=yolo_sign_space)
        if sum([_x0, _y0, _x1, _y1]) == 0:
            continue

        yolo_boxes_raw[idx] = (_x0, _y0, _x1, _y1)

    yolo_boxes = rolling_median_boxes(yolo_boxes_raw, window_frames, video[0].shape[:2])

    # mediapipe predict + per-frame signing space
    mp_predictions = []
    mp_boxes_raw = [None] * num_frames
    for idx, image in enumerate(results["images"]):
        x0y, y0y, x1y, y1y = yolo_boxes[idx]
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

        mp_boxes_raw[idx] = (_x0, _y0, _x1, _y1)

    # create per-frame signing space as rolling median of all signing spaces
    mp_boxes = rolling_median_boxes(mp_boxes_raw, window_frames, video[0].shape[:2])

    for idx, (image, prediction) in enumerate(zip(results["images"], mp_predictions)):
        x0y, y0y, x1y, y1y = yolo_boxes[idx]
        x0mp, y0mp, x1mp, y1mp = mp_boxes[idx]

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
            keypoints[name] = np.asarray(keypoints[name][:, :2], dtype=float)

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
        results["yolo_sign_space"].append([int(v) for v in yolo_boxes[idx]])
        results["sign_space"].append([int(v) for v in pad_bbox])

    results["size"] = results["images"][0].shape[:2]
    results["cropped_size"] = results["cropped_images"][0].shape[:2]
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
    parser.add_argument('--window_seconds', type=float, default=2.0, help="Length of the rolling window in seconds "
                                                                          "used to compute per-frame signing-space "
                                                                          "boxes. A value <= 0 uses the whole video "
                                                                          "(single box, original behavior).")
    parser.add_argument('--debug', action='store_true', default=False, help="Save clip with predicted keypoints.")
    parser.add_argument('--json_only', action='store_true', default=False,
                        help="Save only JSON keypoint files, skip writing cropped videos.")

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
                no_person_frames,
                args.window_seconds,
                fps
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
