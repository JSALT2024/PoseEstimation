import h5py
import json
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import argparse

from data.h2s import How2SignDataset


def save_to_h5(features_list_h5, label, index_dataset, chunk_batch, chunk_size):
    if index_dataset == chunk_batch * chunk_size:
        chunk_batch += 1
        features_list_h5.resize(chunk_batch * chunk_size, axis=0)
    features_list_h5[index_dataset:index_dataset + chunk_size] = label
    index_dataset += chunk_size
    return index_dataset, chunk_batch


def add_to_h5(clip_name, clip_features, index_dataset, chunk_batch, chunk_size):
    feature_shape = clip_features.shape
    features_list_h5 = video_h5.create_dataset(
        clip_name, 
        shape=feature_shape, 
        maxshape=(None,feature_shape[-1]), 
        dtype=np.dtype('float16')
    )
    num_full_chunks = len(clip_features) // chunk_size
    last_chunk_size = len(clip_features) % chunk_size
    for c in range(num_full_chunks):
        feature = clip_features[index_dataset:index_dataset + chunk_size]
        index_dataset, chunk_batch = save_to_h5(features_list_h5, feature, index_dataset, chunk_batch,
                                                chunk_size)
    if last_chunk_size > 0:
        feature = clip_features[index_dataset:index_dataset + last_chunk_size]
        index_dataset, chunk_batch = save_to_h5(features_list_h5, feature, index_dataset, chunk_batch,
                                                last_chunk_size)
        
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

    return pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks


def get_json_files(json_dir):
    json_files = [os.path.join(json_dir, json_file) for json_file in os.listdir(json_dir) if json_file.endswith('.json')]
    return json_files

        
def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--dataset_split', type=str)
    parser.add_argument('--root_folder', type=str)
    parser.add_argument('--annotation_file', type=str)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    
    # prepare variables
    dataset_split = args.dataset_split
    root_folder = args.root_folder

    dataset_file = os.path.join(root_folder, "h5_dataset", f"H2S_{dataset_split}.h5")

    split = 0
    output_folder =  os.path.join(root_folder, "h5_features")
    output_file_name = f"H2S.keypoints.{dataset_split}.{split}.h5"
    output_features_name = f"H2S.keypoints.{dataset_split}.json"

    annotation_file = args.annotation_file
    
    
    json_dir = os.path.join(root_folder, dataset_split, "clips_cropped")

    # create h5 dataset
    json_list = get_json_files(json_dir)
    with h5py.File(dataset_file, 'w') as f:
        for json_file in tqdm(json_list):

            with open(json_file, 'r') as file:
                keypoints_meta = json.load(file)

            video_name = os.path.basename(json_file).replace(".json", "")
            video_name_g = f.create_group(video_name)

            metadata = video_name_g.create_group('metadata')
            metadata.create_dataset(name='start_time', data=0, dtype=np.float32)
            metadata.create_dataset(name='end_time', data=0, dtype=np.float32)
            metadata.create_dataset(name='video_name', data=video_name)
            metadata.create_dataset(name='full_video_name', data="")

            video_name_g.create_dataset(name='sentence', data="")
            joints_g = video_name_g.create_group('joints')

            pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = get_keypoints(keypoints_meta)

            joints_g.create_dataset(name='pose_landmarks', data=pose_landmarks, dtype=np.float32)
            joints_g.create_dataset(name='right_hand_landmarks', data=right_hand_landmarks, dtype=np.float32)
            joints_g.create_dataset(name='left_hand_landmarks', data=left_hand_landmarks, dtype=np.float32)
            joints_g.create_dataset(name='face_landmarks', data=face_landmarks, dtype=np.float32)
    
    
    
    # load dataset
    dataset = How2SignDataset(
        dataset_file, 
        kp_normalization=[
            "global-pose_landmarks",
            "local-right_hand_landmarks",
            "local-left_hand_landmarks",
            "local-face_landmarks",
            ]
    )

    annotations = pd.read_csv(annotation_file, sep='\t')
    video_names = set(annotations.VIDEO_ID)

    video_to_idx = {}
    for idx, data in enumerate(tqdm(dataset)):
        clip_name = data["video_name"]

        for video_name in video_names:
            if clip_name.startswith(video_name):
                break

        if video_name in video_to_idx:
            video_to_idx[video_name].append(idx)
        else:
            video_to_idx[video_name] = [idx]
            
            
    # save features
    f_out = h5py.File(os.path.join(output_folder, output_file_name), 'w')
    dt = h5py.vlen_dtype(np.dtype('float16'))

    metadata = {}
    for video_name in tqdm(video_to_idx):

        idxs = video_to_idx[video_name]
        video_h5 = f_out.create_group(video_name)
        for idx in idxs:
            data = dataset[idx]
            features = data["data"]
            clip_name = data["video_name"]

            metadata[video_name] = split

            # save features in hd5
            add_to_h5(
                clip_name,
                features,
                index_dataset=0,
                chunk_batch=1,
                chunk_size=len(features)
            )

    f_out.close()

    with open(os.path.join(output_folder, output_features_name), "w") as f:
        json.dump(metadata, f)
    

