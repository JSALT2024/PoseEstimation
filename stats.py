import json
import os
from tqdm import tqdm
import numpy as np
import logging
logger = logging.getLogger("")



if __name__=="__main__":
    files = os.listdir("/auto/plzen1/home/mhruz/JSALT2024/YouTubeASL/clips_cropped")
    files = [f for f in files if ".json" in f]

    face = []
    l_hand = []
    r_hand = []

    for idx, file in enumerate(files):
        path = os.path.join("/auto/plzen1/home/mhruz/JSALT2024/YouTubeASL/clips_cropped", file)
        with open(path, "r") as f:
            keypoints = json.load(f)['joints']

        frames = len(keypoints)
        _face = 0
        _l_hand = 0
        _r_hand = 0
        for fidx in keypoints:
            kp = keypoints[fidx]
            if len(kp["face_landmarks"]) > 0:
                _face += 1
            if len(kp["right_hand_landmarks"]) > 0:
                _r_hand += 1
            if len(kp["left_hand_landmarks"]) > 0:
                _l_hand += 1
        face.append(_face/frames)
        l_hand.append(_l_hand/frames)
        r_hand.append(_r_hand/frames)

        if ((idx+1) % 100) == 0:
            print(idx)
            print("face ", np.round(np.mean(face), 3), np.round(np.std(face), 3))
            print("left ", np.round(np.mean(l_hand), 3), np.round(np.std(l_hand), 3))
            print("right", np.round(np.mean(r_hand), 3), np.round(np.std(r_hand), 3))
            
            logger.info(f'{idx}')
            logger.info(f'face   {np.round(np.mean(face), 3)} {np.round(np.std(face), 3)}')
            logger.info(f'left   {np.round(np.mean(l_hand), 3)} {np.round(np.std(l_hand), 3)}')
            logger.info(f'right  {np.round(np.mean(r_hand), 3)} {np.round(np.std(r_hand), 3)}')