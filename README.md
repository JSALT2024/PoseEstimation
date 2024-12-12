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
wget -O checkpoints/pose_landmarker_full.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
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
    --sign_space 4 \
    --yolo_sign_space 4
```

Parallel jobs can be run as array job
PBS example:
```shell
#PBS -J 0-9

python pose_prediction_parallel.py \
  --index_file_id "$PBS_ARRAY_INDEX"
  # ...
```


## Create normalized keypoint feature h5 dataset
Descriptions:
 - Converts features from json files into h5
 - Structure of the h5: `{"video_name_00": {clip_name_00: features_00_00, clip_name_01: features_00_01, ...}, ...}`
 - Shape of the features: `number of frames` x `embedding dimension`
 - Face keypoints are reduced (see `data/h2s.py -> How2SignDatasetJSON.face_landmarks`)
 - Keypoint prediction scripts does not save leg keypoints
 - Keypoints are normalized before saving:
   - global-pose_landmarks
   - local-right_hand_landmarks
   - local-left_hand_landmarks
   - local-face_landmarks
 - Local normalization: moves keypoints to origin adds square padding and normalizes the values in local space -> captures local shape, independent of position in space and scale
 - Global normalization: keypoints are normalized in relation to signing space -> captures absolute position and relation between parts
 - If the name of the clips is not in the format: `video_name.time_stamp.mp4` annotation file with columns `SENTENCE_NAME` and `VIDEO_ID` should be provided

```shell
python create_keypoint_features.py \
  --input_folder data/cropped_clips
  --output_folder data/features
  --dataset_name h2s \
  --split_name train \
  --annotation_file data\how2sign_realigned_train.csv   # only if the name is in bad format
```

## Keypoint dataset
`data/keypoint_dataset.py` -> `KeypointDatasetJSON`
Description:
   - Load and normalize keypoints from json files

Output:
   - List of clip keypoints for one video
```python
# output example
[
   {
      'data': np.empty([n_frames_00, 208]),
      'video_name': 'video_name_00',
      'clip_name': 'clip_name_00'
   },
   {
      'data': np.empty([n_frames_01, 208]),
      'video_name': 'video_name_00',
      'clip_name': 'clip_name_01'
   },
]
```


## Predict
Descriptions:
 - Prediction script for demo
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
