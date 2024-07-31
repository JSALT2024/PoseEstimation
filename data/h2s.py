import json
import os

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from normalization import local_keypoint_normalization, global_keypoint_normalization


class How2SignDataset(Dataset):
    """ Custom dataset for how2sign dataset on pose features.
    args:
        h5_path (str): path to h5 file
        video_file_path (str, optional): path to video files
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self,
                 h5_path,
                 video_file_path=None,
                 transform=None,
                 kp_normalization: list = [],
                 face_landmarks: str = "YouTubeASL"):

        self.video_path = video_file_path
        self.h5_path = h5_path

        self.video_names = {}

        if face_landmarks == "YouTubeASL":
            self.face_landmarks = [
                0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81,
                93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276,
                282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405, 468, 473
            ]
        elif face_landmarks == "UWB":
            self.face_landmarks = [
                101, 214,  # left cheek top, bot
                330, 434,  # right cheek top, bot
                197, 195, 4, 1,  # nose rigid1, rigid2, flex, tip
                295, 282, 283,  # right eyebrow
                53, 52, 65,  # left eyebrow
                263, 386, 362, 374,  # right eye
                33, 159, 133, 145,  # left eye
                40, 270, 91, 321,  # outer mouth sqruare
                311, 81, 178, 402,  # inner mouth square
                78, 308  # inner mouth corners
            ]
        self.transform = transform
        self.kp_normalization = kp_normalization

        self.get_video_names()

    def __getitem__(self, index):

        data, sentence, video_name = self.load_data(idx=index)
        if self.transform:
            data = self.transform(data)

        return {"data": torch.tensor(data).float(), "sentence": sentence, "video_name": video_name}

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as f:
            return len(f.keys())

    def get_video_names(self):
        """
        Get video names for .mp4 videos in self.video_path dir.
        returns:
            video_paths (list): list of .mp4 files available in self.video_path
        """
        if self.video_path is None:
            return

        if not os.path.exists(self.video_path):
            raise ValueError(f'Error: video_path does not exist \n {self.video_path}')
        if not os.path.isdir(self.video_path):
            raise ValueError(f'Error: video_path is not a directory \n {self.video_path} ')

        for filename in os.listdir(self.video_path):
            if filename.endswith(".mp4"):
                self.video_names[filename.strip('.mp4')] = os.path.join(self.video_path, filename)

    def load_data(self, idx=0):

        with h5py.File(self.h5_path, 'r') as f:
            video_name = list(f.keys())[idx]
            joints = {l[0]: l[1][()] for l in f[video_name]['joints'].items()}
            face_landmarks = f[video_name]['joints']['face_landmarks'][()]
            left_hand_landmarks = f[video_name]['joints']['left_hand_landmarks'][()]
            right_hand_landmarks = f[video_name]['joints']['right_hand_landmarks'][()]
            pose_landmarks = f[video_name]['joints']['pose_landmarks'][()]
            sentence = f[video_name]['sentence'][()].decode('utf-8')

        pose_landmarks = pose_landmarks[:, :25]
        joints["pose_landmarks"] = joints["pose_landmarks"][:, :25]

        # TODO implement once visual training is relevant
        # if self.video_path is None:
        #     pass
        # elif video_name in self.video_names:
        #     video_path = os.path.join(self.video_path + video_name + '.mp4')
        # else:
        #     warnings.warn(
        #         f'Warning: video_path does not contain video with name {video_name}')

        if self.kp_normalization:
            joints["face_landmarks"] = joints["face_landmarks"][:, self.face_landmarks, :]

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

            for k,  landmark in global_landmarks.items():
                if landmark == "pose_landmarks":
                    global_landmarks[k] = keypoints
                else:
                    global_landmarks[k] = additional_keypoints[landmark]

            all_landmarks = {**local_landmarks, **global_landmarks}
            data = []
            for idx in range(len(self.kp_normalization)):
                data.append(all_landmarks[idx])

            data = np.concatenate(data, axis=1)
            data = data.reshape(data.shape[0], -1)
        else:
            face_landmarks = face_landmarks[:, self.face_landmarks, 0:2]  # select only wanted KPI and  x, y
            left_hand_landmarks = left_hand_landmarks[:, :, 0:2]
            right_hand_landmarks = right_hand_landmarks[:, :, 0:2]
            pose_landmarks = pose_landmarks[:, :, 0:2]

            data = np.concatenate((pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks),
                                  axis=1).reshape(len(face_landmarks), 214)

        return data, sentence, video_name

    def plot_points2video(self, index, video_name):
        if self.video_path is None:
            raise ValueError("Error: video_path is None, cannot plot. \n Aborting.")
        item = self.__getitem__(index)
        plot_metadata = item['plot_metadata']

        cap = cv2.VideoCapture(item['metadata']['VIDEO_PATH'])

        # Check if the video file opened successfully
        if not cap.isOpened():
            raise ValueError(f'Error: Couldnt open the video file. \n {video_path} \n Aborting.')

        ret, frame = cap.read()

        height, width, layers = frame.shape
        idx = 0
        video = cv2.VideoWriter(video_name, 0, 3, (width, height))

        while ret:
            frame = self.anotate_img(frame, plot_metadata, idx, (125, 255, 10))
            video.write(frame)
            ret, frame = cap.read()
            idx += 1

        cap.release()
        cv2.destroyAllWindows()
        video.release()



def get_keypoints(json_data):
    right_hand_landmarks = []
    left_hand_landmarks = []
    face_landmarks = []
    pose_landmarks = []
    for frame_id in json_data['joints']:
        if len(json_data['joints'][frame_id]['pose_landmarks']) == 0:
            pose_landmarks.append(np.zeros((33, 4)))
        else:
            pose_landmarks.append(np.array(json_data['joints'][frame_id]['pose_landmarks']))

        if len(json_data['joints'][frame_id]['right_hand_landmarks']) == 0:
            right_hand_landmarks.append(np.zeros((21, 4)))
        else:
            right_hand_landmarks.append(np.array(json_data['joints'][frame_id]['right_hand_landmarks']))

        if len(json_data['joints'][frame_id]['left_hand_landmarks']) == 0:
            left_hand_landmarks.append(np.zeros((21, 4)))
        else:
            left_hand_landmarks.append(np.array(json_data['joints'][frame_id]['left_hand_landmarks']))

        if len(json_data['joints'][frame_id]['face_landmarks']) == 0:
            face_landmarks.append(np.zeros((478, 4)))
        else:
            face_landmarks.append(np.array(json_data['joints'][frame_id]['face_landmarks']))


    pose_landmarks = np.array(pose_landmarks)[:, :25]
    return pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks


def get_json_files(json_dir):
    json_files = [os.path.join(json_dir, json_file) for json_file in os.listdir(json_dir) if json_file.endswith('.json')]
    return json_files


class How2SignDatasetJSON(Dataset):
    def __init__(self,
                 json_folder,
                 kp_normalization: list = [],
                 face_landmarks: str = "YouTubeASL"):


        json_list = get_json_files(json_folder)

        self.video_to_files = {}
        for idx, path in enumerate(json_list):
            name = os.path.basename(path)
            name_split = name.split(".")[:-1]
            clip_name = ".".join(name_split)
            video_name = name_split[0]

            if video_name in self.video_to_files:
                self.video_to_files[video_name].append(path)
            else:
                self.video_to_files[video_name] = [path]
        self.video_names = list(self.video_to_files.keys())

        if face_landmarks == "YouTubeASL":
            self.face_landmarks = [
                0, 4, 13, 14, 17, 33, 39, 46, 52, 55, 61, 64, 81,
                93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276,
                282, 285, 291, 294, 311, 323, 362, 386, 397, 402, 405, 468, 473
            ]
        elif face_landmarks == "UWB":
            self.face_landmarks = [
                101, 214,  # left cheek top, bot
                330, 434,  # right cheek top, bot
                197, 195, 4, 1,  # nose rigid1, rigid2, flex, tip
                295, 282, 283,  # right eyebrow
                53, 52, 65,  # left eyebrow
                263, 386, 362, 374,  # right eye
                33, 159, 133, 145,  # left eye
                40, 270, 91, 321,  # outer mouth sqruare
                311, 81, 178, 402,  # inner mouth square
                78, 308  # inner mouth corners
            ]

        self.kp_normalization = kp_normalization

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
        pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = get_keypoints(keypoints_meta)
        joints = {
            'face_landmarks': np.array(face_landmarks),
            'left_hand_landmarks': np.array(left_hand_landmarks),
            'right_hand_landmarks': np.array(right_hand_landmarks),
            'pose_landmarks': np.array(pose_landmarks)
        }

        if self.kp_normalization:
            joints["face_landmarks"] = joints["face_landmarks"][:, self.face_landmarks, :]

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

            for k,  landmark in global_landmarks.items():
                if landmark == "pose_landmarks":
                    global_landmarks[k] = keypoints
                else:
                    global_landmarks[k] = additional_keypoints[landmark]

            all_landmarks = {**local_landmarks, **global_landmarks}
            data = []
            for idx in range(len(self.kp_normalization)):
                data.append(all_landmarks[idx])

            data = np.concatenate(data, axis=1)
            data = data.reshape(data.shape[0], -1)
        else:
            face_landmarks = face_landmarks[:, self.face_landmarks, 0:2]  # select only wanted KPI and  x, y
            left_hand_landmarks = left_hand_landmarks[:, :, 0:2]
            right_hand_landmarks = right_hand_landmarks[:, :, 0:2]
            pose_landmarks = pose_landmarks[:, :, 0:2]

            data = np.concatenate((pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks),
                                  axis=1).reshape(len(face_landmarks), 214)
        return data