# PoseEstimation

## Requirements (not complete)
```shell
pip install ultralytics==8.2.18
pip install mediapipe

# pip install decord
```


## Download weights
```shell
mkdir checkpoints
wget -O checkpoints/hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
wget -O checkpoints/pose_landmarker_lite.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
wget -O checkpoints/face_landmarker.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## Predict keypoints in parallel
### Descriptions:

 - input: folder with clips
 - output: spatial cropped clip with size (512, 512) and predicted keypoints in json file with same name as clip
 - crop is created based on sign space around the person
 - if input clip is not square, shorter side will be padded with (114,114,114) color
 - clips without predictions or multiple predictions are **skipped**
 - script can run in parallel, each process can access its own index file or select index files randomly
 


Prepare index files in advance (not necessary).
```python
from pose_prediction_parallel import create_index_files
clip_folder = ""
index_folder = ""
num_index_files = 100

create_index_files(clip_folder, index_folder, num_index_files)
```

Run multiple parallel jobs. If index files does not exist, first job will crete them.
If `index_file_id` is not specified, clips will be processed randomly.
```shell
# create 100 index files and process fill index_file_000
# additional processes can be run in parallel with different index_file_id
python pose_prediction_parallel.py \
    --input_folder data/clips \
    --output_folder data/cropped_clips \
    --tmp_folder data/tmp_clips \
    --num_index_files 100 \
    --index_path data/index_files \
    --index_file_id 0 \
    --checkpoint_folder checkpoints \
    --sign_space 4 
```

Parallel jobs can be run as array job
PBS example:
```shell
#PBS -J 0-9

python pose_prediction_parallel.py \
  --index_file_id "$PBS_ARRAY_INDEX"
  # ...
```


## Convert predictions to h5
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