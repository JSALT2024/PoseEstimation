import h5py
import json
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import argparse

from data.keypoint_dataset import KeypointDatasetJSON


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
        maxshape=(None, feature_shape[-1]),
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
        _ = save_to_h5(features_list_h5, feature, index_dataset, chunk_batch, last_chunk_size)


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--input_folder', type=str, help='Path to folder with video and json files.')
    parser.add_argument('--output_folder', type=str, help="Data will be saved in this folder.")
    parser.add_argument('--dataset_name', type=str, help="Name of the dataset. Used only for naming of the "
                                                         "output file.")
    parser.add_argument('--split_name', default="train", type=str, help="Name of the data subset examples: dev, "
                                                                        "train, test. Used only for naming of the "
                                                                        "output file.")

    parser.add_argument('--annotation_file', type=str, help="If the name is not in the format: "
                                                            "'video_name.time_stamp.mp4' and can't be parsed, "
                                                            "annotation file with: SENTENCE_NAME and VIDEO_ID columns "
                                                            "should be provided.")
    parser.add_argument('--normalization', type=str, default="default",
                        choices=["none", "default"],  help='How to normalize keypoints (local or global)')
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
    default_normalization = (
            "global-pose_landmarks",
            "local-right_hand_landmarks",
            "local-left_hand_landmarks",
            "local-face_landmarks",
    )

    normalization = None
    if args.normalization == "default":
        normalization = default_normalization

    dataset = KeypointDatasetJSON(
        args.input_folder,
        clip_to_video,
        kp_normalization=normalization
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
