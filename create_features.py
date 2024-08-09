import h5py
import json
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import argparse

from data.h2s import How2SignDatasetJSON


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

    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split_name', default="train", type=str)

    parser.add_argument('--annotation_file', type=str)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    # prepare names
    output_file_name = f"{args.dataset_name}.keypoints.{args.split_name}.0.h5"
    output_features_name = f"{args.dataset_name}.keypoints.{args.split_name}.json"
    os.makedirs(args.output_folder, exist_ok=True)

    # prepare mapping between clip names and video names
    clip_to_video = {}
    clip_names = os.listdir(args.input_folder)
    clip_names = [file for file in clip_names if file.endswith(".mp4")]
    for idx in range(len(clip_names)):
        name_split = clip_names[idx].split(".")[:-1]
        clip_names[idx] = ".".join(name_split)
    if args.annotation_file:
        annotations = pd.read_csv(args.annotation_file, sep='\t')
        _clip_to_video = dict(zip(annotations.SENTENCE_NAME, annotations.VIDEO_ID))
        for name in clip_names:
            clip_to_video[name] = _clip_to_video[name]
    else:
        for name in clip_names:
            name_split = name.split(".")[:-1]
            video_name = ".".join(name_split)
            clip_to_video[name] = video_name

    # load dataset
    dataset = How2SignDatasetJSON(
        args.input_folder,
        clip_to_video,
        kp_normalization=[
            "global-pose_landmarks",
            "local-right_hand_landmarks",
            "local-left_hand_landmarks",
            "local-face_landmarks",
            ]
    )    
    print(len(dataset))
            
    # save features
    f_out = h5py.File(os.path.join(args.output_folder, output_file_name), 'w')

    metadata = {}
    for data in tqdm(dataset):
        video_name = data[0]["video_name"]
        video_h5 = f_out.create_group(video_name)
        
        for clip in data:
            features = clip["data"]
            clip_name = clip["clip_name"]

            metadata[video_name] = 0

            # save features in hd5
            add_to_h5(
                clip_name,
                features,
                index_dataset=0,
                chunk_batch=1,
                chunk_size=len(features)
            )

    f_out.close()

    with open(os.path.join(args.output_folder, output_features_name), "w") as f:
        json.dump(metadata, f)


