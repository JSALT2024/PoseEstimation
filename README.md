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

## Run parallel job
```shell
# create 100 index files and process fill index_file_000
# additional processes can be run in parallel with different index_file
python pose_prediction_parallel.py \
    --input_folder data/clips \
    --output_folder data/cropped_clips \
    --tmp_folder data/tmp_clips \
    --num_index_files 100 \
    --index_path data/index_files \
    --index_file data/index_files/index_file_000.csv \
    --checkpoint_folder checkpoints \
    --sign_space 4 
```

### TODO
 - [ ] Remove duplicate code 
 - [ ] Add better logging
 - [ ] Save more information (similar to predict script)
 - [ ] Use YOLO to crop smaller box, than expand after MediaPipe


## Create h5 features
### How2Sign
```shell
python create_features-hs.py \
  --dataset_split train \
  --root_folder data\h2s \
  --annotation_file data\h2s\how2sign_realigned_train.csv
```

### YouTubeASL
```shell
python create_features-yt.py \
  --dataset_split train \
  --root_folder data\yt 
```

### TODO 
 - [ ] Merge files
 - [ ] Describe folder structure
 - [ ] Add additional info into h5 (signing space, hand_crops...)


## Predict
```python
from predict_pose import predict_pose, create_mediapipe_models#

# load models: 
#   hand_landmarker.task
#   pose_landmarker_full.task
#   face_landmarker.task
checkpoint_folder = ""
models = create_mediapipe_models(checkpoint_folder)

# predict
video = []
prediction = predict_pose(video, models, 4)
```