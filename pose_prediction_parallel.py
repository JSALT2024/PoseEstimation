import argparse
import json
import os
import pandas as pd

import cv2
import mediapipe as mp
import numpy as np
from decord import VideoReader, cpu
from mediapipe.tasks.python import vision
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import gc
import time
import random
import shutil

VisionRunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions


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


def load_video_decord(path: str, workers: int = 1):
    video = []
    video_reader = VideoReader(path, num_threads=workers, ctx=cpu(0))
    fps = video_reader.get_avg_fps()
    for frame in video_reader:
        video.append(frame.asnumpy())
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

    image = np.pad(image, ((ih, ih), (iw, iw), (0, 0)), mode='constant', constant_values=114)  # mode="reflect"
    cropped_image = image[y0:y1, x0:x1]

    return cropped_image, new_bbox


def yolo_predict(image: np.ndarray, model, min_conf: float = 0):
    yolo_results = model(image, verbose=False)

    bboxes = yolo_results[0].boxes.xyxy
    keypoints = yolo_results[0].keypoints.xy
    bboxes = bboxes.cpu().numpy()
    keypoints = keypoints.cpu().numpy()

    conf = yolo_results[0].boxes.conf
    conf = conf.cpu().numpy()
    select_mask_kp = np.sum(keypoints, axis=(1, 2)) > 0.0001
    select_mask_bb = conf > yolo_min_confidence
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


def predict(video_path, output_folder, workers, sign_space, tmp_folder="", video_size=(512, 512), debug=False):
    print("video_path", video_path, "output_folder", output_folder, "workers", workers, "sign_space", sign_space,
          "tmp_folder", tmp_folder)
    predictions = []
    num_predictions = []

    # predict
    images, fps = load_video_cv(video_path)
    print(f"video: {video_path}", f"fps: {fps}", f"frames: {len(images)}", f"frame size: {images[0].shape}")
    # for idx, image in enumerate(tqdm(images, desc="yolov8 predict")):
    for idx, image in enumerate(images):
        bboxes, keypoints, confs = yolo_predict(image, yolo_model)

        predictions.append([bboxes, keypoints, confs])
        num_predictions.append(len(bboxes))

    # single person video
    single_person = np.sum(np.array(num_predictions) <= 1) / len(num_predictions)
    print(f"Single person images: {single_person:.3f}")
    single_person = (1 - single_person) < single_person_frames

    if not single_person:
        # save filename + num_predictions in file
        return {"state": 2}

    # no person video
    no_person = np.sum(np.array(num_predictions) == 0) / len(num_predictions)
    print(f"No person images: {no_person:.3f}")
    no_person = no_person > no_person_frames

    if no_person:
        return {"state": -2}

    # get signing bbox
    x0, y0, x1, y1 = [], [], [], []
    # for idx, (image, prediction) in enumerate(tqdm(zip(images, predictions), desc="get signing space")):
    for idx, (image, prediction) in enumerate(zip(images, predictions)):
        _, keypoints, _ = prediction
        if len(keypoints) != 1:
            continue

        _x0, _y0, _x1, _y1 = new_bbox(image, keypoints[0], lsi=5, rsi=6, sign_space=sign_space)

        x0.append(_x0)
        y0.append(_y0)
        x1.append(_x1)
        y1.append(_y1)

    x0y = np.round(np.median(x0)).astype(int)
    y0y = np.round(np.median(y0)).astype(int)
    x1y = np.round(np.median(x1)).astype(int)
    y1y = np.round(np.median(y1)).astype(int)

    # mediapipe predict
    mp_predictions = []
    # new_video_frames = []
    x0, y0, x1, y1 = [], [], [], []
    # for idx, (image, prediction) in enumerate(tqdm(zip(images, predictions), desc="mediapipe prediction")):
    for idx, (image, prediction) in enumerate(zip(images, predictions)):
        cropped_image = image[y0y:y1y, x0y:x1y]
        ih, iw = cropped_image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(cropped_image))

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

        if len(kp_all) == 0:
            continue

        _x0, _y0, _x1, _y1 = new_bbox(cropped_image, kp_all, lsi=11, rsi=12, sign_space=sign_space)

        x0.append(_x0)
        y0.append(_y0)
        x1.append(_x1)
        y1.append(_y1)

    x0mp = np.round(np.median(x0)).astype(int)
    y0mp = np.round(np.median(y0)).astype(int)
    x1mp = np.round(np.median(x1)).astype(int)
    y1mp = np.round(np.median(y1)).astype(int)

    # save predictionas and video
    file_name = ".".join(os.path.basename(video_path).split(".")[:-1])
    keypoints_path = os.path.join(output_folder, f"{file_name}.json")
    video_path = os.path.join(output_folder, f"{file_name}.mp4")
    if tmp_folder:
        os.makedirs(tmp_folder, exist_ok=True)
        keypoints_path = os.path.join(tmp_folder, f"{file_name}.json")
        video_path = os.path.join(tmp_folder, f"{file_name}.mp4")

    result = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
                             video_size)  # frames[0].shape[:2]

    predictions_out = {}
    # for idx, (image, prediction, prediction_yolo) in enumerate(tqdm(zip(images, mp_predictions, predictions), desc="crop video")):
    for idx, (image, prediction, prediction_yolo) in enumerate(zip(images, mp_predictions, predictions)):
        cropped_image = image[y0y:y1y, x0y:x1y]
        ih, iw = cropped_image.shape[:2]

        cropped_image_final, pad_bbox = crop_pad_image(cropped_image, (x0mp, y0mp, x1mp, y1mp), border=0)
        oh, ow = cropped_image_final.shape[:2]

        cropped_image_final = cv2.resize(cropped_image_final, video_size)
        nh, nw = cropped_image_final.shape[:2]

        hand_prediction, face_prediction, pose_prediction = prediction

        # 1 = keyponts position in output tuple, 0 = first prediction -> only one person expected
        yolo_keypoints = None if len(prediction_yolo[1]) != 1 else prediction_yolo[1][0]

        face_keypoints = keypoints_out_format(face_prediction.face_landmarks, (ih, iw))
        pose_keypoints = keypoints_out_format(pose_prediction.pose_landmarks, (ih, iw))
        hand_keypoints = process_hands(
            hand_prediction.hand_landmarks,
            hand_prediction.handedness,
            pose_keypoints,
            (ih, iw),
            yolo_keypoints
        )

        keypoints = {
            'pose_landmarks': pose_keypoints,
            'right_hand_landmarks': hand_keypoints["right"],
            'left_hand_landmarks': hand_keypoints["left"],
            'face_landmarks': face_keypoints
        }

        # move kp
        x_move = pad_bbox[0] * nw / ow  # mp_bbox[0] #- pad_bbox[0]
        y_move = pad_bbox[1] * nh / oh  # mp_bbox[1] #- pad_bbox[1]

        for name in keypoints:
            if len(keypoints[name]) > 0:
                keypoints[name][:, 0] *= nw / ow
                keypoints[name][:, 1] *= nh / oh
                keypoints[name][:, 0] -= x_move
                keypoints[name][:, 1] -= y_move
                keypoints[name] = np.round(keypoints[name], 3).tolist()

        # remove used image
        images[idx] = None
        predictions[idx] = None
        if (idx + 1) % 100 == 0:
            gc.collect()

        if debug:
            colors = {
                'pose_landmarks': [50, 50, 200],
                'right_hand_landmarks': [0, 0, 0],
                'left_hand_landmarks': [255, 255, 255],
                'face_landmarks': [200, 50, 50]
            }

            for name in keypoints:
                if keypoints[name] is None:
                    continue
                for kp in keypoints[name]:
                    cropped_image_final = cv2.circle(cropped_image_final, np.round(kp[:2]).astype(int), 3, colors[name],
                                                     thickness=-1)

        # save processed data
        predictions_out[str(idx)] = keypoints
        result.write(cv2.cvtColor(cropped_image_final, cv2.COLOR_RGB2BGR))

    result.release()

    keypoints = {"joints": predictions_out}
    with open(keypoints_path, "w") as f:
        json.dump(keypoints, f)

    if tmp_folder:
        if not os.path.exists(os.path.join(output_folder, f"{file_name}.json")):
            _ = shutil.move(keypoints_path, output_folder)

        if not os.path.exists(os.path.join(output_folder, f"{file_name}.mp4")):
            _ = shutil.move(video_path, output_folder)

        shutil.rmtree(tmp_folder)

    os.chmod(os.path.join(output_folder, f"{file_name}.json"), 0o0777)
    os.chmod(os.path.join(output_folder, f"{file_name}.mp4"), 0o0777)

    return {"frames": len(images)}


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--index_path', type=str)
    parser.add_argument('--index_file', type=str, default="")
    parser.add_argument('--tmp_folder', type=str)
    parser.add_argument('--checkpoint_folder', default="", type=str)

    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--sign_space', type=int, default=5)
    parser.add_argument('--debug', action='store_true', default=False)

    return parser


if __name__ == "__main__":
    """
    state: -1 = not processed
            0 = ended with exception
            1 = finished successfully
            2 = more than 1 people
    """
    args = get_args_parser().parse_args()

    yolo_min_confidence = 0.5
    mp_min_confidence = 0.4
    single_person_frames = 0.05
    no_person_frames = 0.2

    # mediapipe
    num_poses = 1
    hand_model_path = os.path.join(args.checkpoint_folder, 'hand_landmarker.task')
    pose_model_path = os.path.join(args.checkpoint_folder, 'pose_landmarker_full.task')
    face_model_path = os.path.join(args.checkpoint_folder, 'face_landmarker.task')
    yolo_model_path = os.path.join(args.checkpoint_folder, "yolov8n-pose.pt")

    # yolov8
    yolo_model = YOLO(yolo_model_path)

    # define hand model
    hand_options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        min_hand_detection_confidence=mp_min_confidence,  # 0.1,
        min_hand_presence_confidence=mp_min_confidence,  # 0.05,
        num_hands=num_poses * 2)
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    # define body model
    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        min_pose_detection_confidence=mp_min_confidence,
        min_pose_presence_confidence=mp_min_confidence,
        num_poses=num_poses
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

    # define face model
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        min_face_detection_confidence=mp_min_confidence,
        min_face_presence_confidence=mp_min_confidence,
        num_faces=num_poses
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    input_folder = args.input_folder
    output_folder = args.output_folder
    index_folder = args.index_path

    all_times = 0
    all_frames = 0
    processed_videos = 0
    failed_videos = 0

    num_index_files = 500

    while True:
        start_time = time.time()
        index_files = [os.path.join(index_folder, file) for file in os.listdir(index_folder) if ".csv" in file]

        # create index file
        index_row_idx = -1
        if len(index_files) == 0:
            file_names = os.listdir(input_folder)
            file_names = [file_name for file_name in file_names if ".mp4" in file_name]

            index_file = pd.DataFrame({"file_names": file_names, "state": [-1] * len(file_names)})
            num_files = len(index_file)
            step = int(np.round(num_files / num_index_files))
            print("files", num_files, "step", step)

            # split file
            index_file_split = []
            for i in range(1, num_index_files):
                start = (i - 1) * step
                end = i * step
                index_file_split.append(index_file[start:end])
                print(i, start, end)
            print(i, end, num_files)
            index_file_split.append(index_file[end::])

            print("Saving index files:")
            for i, file in enumerate(index_file_split):
                path = os.path.join(index_folder, f"index_file_{i:03d}.csv")
                file.to_csv(path, index=False)
                index_files.append(path)
                print(path)

        # read file
        for _ in range(len(index_files)):
            num_files = len(index_files)
            idx = random.randint(0, num_files - 1)

            if args.index_file:
                index_path = args.index_file
                index_file = pd.read_csv(index_path, dtype={"file_names": str, "state": float})
                idx_list = index_file[index_file["state"] == -1]
            else:
                try:
                    index_path = index_files[idx]
                    index_file = pd.read_csv(index_path, dtype={"file_names": str, "state": float})
                    idx_list = index_file[index_file["state"] == -1]
                    print(len(idx_list))
                except Exception as e:
                    print(f"Loading csv failed: {e}")
                    idx_list = []

            if len(idx_list) > 0:
                print("idx_list:", len(idx_list))
                idx_list = idx_list.sample(1).index.tolist()
                index_row_idx = idx_list[0]

                print(index_path)
                break
            index_files.pop(idx)

        print("index_row_idx:", index_row_idx)
        if index_row_idx == -1:
            break

        # set selected file
        index_file.at[index_row_idx, "state"] = 0
        index_file.to_csv(index_path, index=False)

        video_name = index_file.iloc[index_row_idx]["file_names"]
        video_path = os.path.join(input_folder, video_name)

        # predict
        try:
            predictions = predict(video_path, output_folder, args.workers, args.sign_space, tmp_folder=args.tmp_folder,
                                  debug=args.debug)
            if "state" in predictions:
                print(f"Multiple people in video: {video_path}")
                index_file = pd.read_csv(index_path)
                index_file.at[index_row_idx, "state"] = predictions["state"]
                index_file.to_csv(index_path, index=False)
        except Exception as e:
            print(f"Processing failed: {video_path}")
            print(f"Message: {e}")
            predictions = {}

        # save data
        if "frames" in predictions:
            n_frames = predictions["frames"]

            index_file = pd.read_csv(index_path)
            index_file.at[index_row_idx, "state"] = 1
            index_file.to_csv(index_path, index=False)

            all_frames += n_frames
        else:
            failed_videos += 1

        process_time = time.time() - start_time
        processed_videos += 1
        all_times += process_time

        average_frames = all_frames / (processed_videos - failed_videos) if processed_videos - failed_videos > 0 else 0
        print(
            f"Processing time: {process_time}, Average time: {all_times / processed_videos}, Average frames: {average_frames}, Failed videos: {failed_videos}")
        print()

    print("Full processing time:", all_times)

