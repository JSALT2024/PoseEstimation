import argparse
import json
import multiprocessing
import os
from functools import partial

from tqdm import tqdm


def process_files(files, root_folder, key="joints"):
    face = 0
    l_hand = 0
    r_hand = 0
    frames = 0

    for idx, file in enumerate(tqdm(files)):
        path = os.path.join(root_folder, file)
        with open(path, "r") as f:
            keypoints = json.load(f)[key]

        frames += len(keypoints)
        for fidx in keypoints:
            if isinstance(fidx, str):
                kp = keypoints[fidx]
            else:
                kp = fidx
            if len(kp["face_landmarks"]) > 0:
                face += 1
            if len(kp["right_hand_landmarks"]) > 0:
                r_hand += 1
            if len(kp["left_hand_landmarks"]) > 0:
                l_hand += 1

    return face / 2 ** 16, l_hand / 2 ** 16, r_hand / 2 ** 16, frames / 2 ** 16


def split(data, n):
    splits = [data[i * n:(i + 1) * n] for i in range((len(data) + n - 1) // n)]
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--key', type=str, default="joints")
    args = parser.parse_args()

    files = os.listdir(args.root_folder)
    files = [f for f in files if ".json" in f]
    split_size = (len(files) // args.workers) + 1

    process_function = partial(process_files, root_folder=args.root_folder, key=args.key)
    file_splits = split(files, split_size)

    print(f"Num files: {len(files)}")
    print(f"Split size: {split_size}")
    print(f"Num split: {len(file_splits)}")

    pool = multiprocessing.Pool(processes=args.workers)
    results = pool.map(process_function, file_splits)
    face = 0
    l_hand = 0
    r_hand = 0
    frames = 0
    for _face, _l_hand, _r_hand, _frames in results:
        print(_face, _l_hand, _r_hand, _frames)
        face += _face
        l_hand += _l_hand
        r_hand += _r_hand
        frames += _frames

    print("face  mean:", face / frames)
    print("left  mean:", l_hand / frames)
    print("right mean:", r_hand / frames)
