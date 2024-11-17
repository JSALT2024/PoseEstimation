import json
import os

import numpy as np
from torch.utils.data import Dataset

from .normalization import local_keypoint_normalization, global_keypoint_normalization


def get_keypoints(json_data, data_key='cropped_keypoints'):
    right_hand_landmarks = []
    left_hand_landmarks = []
    face_landmarks = []
    pose_landmarks = []

    keypoints = json_data[data_key]
    for frame_id in range(len(keypoints)):
        if len(keypoints[frame_id]['pose_landmarks']) == 0:
            pose_landmarks.append(np.zeros((33, 2)))
        else:
            pose_landmarks.append(np.array(keypoints[frame_id]['pose_landmarks']))

        if len(keypoints[frame_id]['right_hand_landmarks']) == 0:
            right_hand_landmarks.append(np.zeros((21, 2)))
        else:
            right_hand_landmarks.append(np.array(keypoints[frame_id]['right_hand_landmarks']))

        if len(keypoints[frame_id]['left_hand_landmarks']) == 0:
            left_hand_landmarks.append(np.zeros((21, 2)))
        else:
            left_hand_landmarks.append(np.array(keypoints[frame_id]['left_hand_landmarks']))

        if len(keypoints[frame_id]['face_landmarks']) == 0:
            face_landmarks.append(np.zeros((478, 2)))
        else:
            face_landmarks.append(np.array(keypoints[frame_id]['face_landmarks']))

    pose_landmarks = np.array(pose_landmarks)[:, :25]
    return pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks


def get_json_files(json_dir):
    json_files = [os.path.join(json_dir, json_file) for json_file in os.listdir(json_dir) if
                  json_file.endswith('.json')]
    return json_files


class How2SignDatasetJSON(Dataset):
    def __init__(
            self,
            json_folder: str,
            clip_to_video: dict,
            kp_normalization: tuple = (),
            data_key: str = "cropped_keypoints"
    ):
        json_list = get_json_files(json_folder)
        self.clip_to_video = clip_to_video
        self.video_to_files = {}
        for idx, path in enumerate(json_list):
            name = os.path.basename(path)
            name_split = name.split(".")[:-1]
            clip_name = ".".join(name_split)
            video_name = clip_to_video[clip_name]

            if video_name in self.video_to_files:
                self.video_to_files[video_name].append(path)
            else:
                self.video_to_files[video_name] = [path]
        self.video_names = list(self.video_to_files.keys())

        self.face_landmarks = [
            0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81,
            93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276,
            282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405, 468, 473
        ]
        self.kp_normalization = kp_normalization
        self.data_key = data_key

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        clip_paths = self.video_to_files[video_name]

        output_data = []
        for clip_path in clip_paths:
            name = os.path.basename(clip_path)
            name_split = name.split(".")
            clip_name = ".".join(name_split[:-1])

            clip_data = self.load_data(clip_path)
            clip_data = {"data": clip_data, "video_name": video_name, "clip_name": clip_name}
            output_data.append(clip_data)

        return output_data

    def __len__(self):
        return len(self.video_to_files)

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            keypoints_meta = json.load(file)
        keypoints = get_keypoints(keypoints_meta, data_key=self.data_key)
        pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = keypoints
        joints = {
            'face_landmarks': np.array(face_landmarks)[:, self.face_landmarks, :],
            'left_hand_landmarks': np.array(left_hand_landmarks),
            'right_hand_landmarks': np.array(right_hand_landmarks),
            'pose_landmarks': np.array(pose_landmarks)
        }

        if self.kp_normalization:
            local_landmarks = {}
            global_landmarks = {}

            for idx, landmarks in enumerate(self.kp_normalization):
                prefix, landmarks = landmarks.split("-")
                if prefix == "local":
                    local_landmarks[idx] = landmarks
                elif prefix == "global":
                    global_landmarks[idx] = landmarks

            # local normalization
            for idx, landmarks in local_landmarks.items():
                normalized_keypoints = local_keypoint_normalization(joints, landmarks, padding=0.2)
                local_landmarks[idx] = normalized_keypoints

            # global normalization
            additional_landmarks = list(global_landmarks.values())
            if "pose_landmarks" in additional_landmarks:
                additional_landmarks.remove("pose_landmarks")

            keypoints, additional_keypoints = global_keypoint_normalization(
                joints,
                "pose_landmarks",
                additional_landmarks
            )

            for k, landmark in global_landmarks.items():
                if landmark == "pose_landmarks":
                    global_landmarks[k] = keypoints
                else:
                    global_landmarks[k] = additional_keypoints[landmark]

            all_landmarks = {**local_landmarks, **global_landmarks}
            data = []
            for idx in range(len(self.kp_normalization)):
                data.append(all_landmarks[idx])

            data = np.concatenate(data, axis=1)
        else:
            data = [joints["pose_landmarks"], joints["right_hand_landmarks"], joints["left_hand_landmarks"],
                    joints["face_landmarks"]]
            data = np.concatenate(data, axis=1)
        data = data.reshape(data.shape[0], -1)

        return data
