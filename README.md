# PoseEstimation

## Requirements (not complete)
```shell
pip install ultralytics==8.2.18
pip install decord
pip install mediapipe
```


## Download weights
```shell
mkdir checkpoints
wget -O checkpoints/hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
wget -O checkpoints/pose_landmarker_lite.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
wget -O checkpoints/face_landmarker.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## Run
```shell
python pose_prediction_parallel.py \
    --input_folder data/clips \
    --output_folder data/cropped_clips \
    --index_path data/index_file.csv \
    --checkpoint_folder checkpoints \
    --sign_space 5 \
    --debug
```