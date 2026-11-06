import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import cv2
import json
import subprocess
import tempfile
from fractions import Fraction
from typing import List


def load_video_cv(path: str) -> (List[np.ndarray], float):
    """Load a video."""
    video = []

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(img)
    cap.release()
    return video, fps


def deinterlace_from_annotations(annotations: dict, video_names):
    """Map actual video basenames using nested clip_order annotations, case-sensitively."""
    clips = {}
    for group in annotations.values():
        for clip_id in group["clip_order"]:
            record = group[clip_id]
            value = (record.get("interlaced"), record.get("field_order"))
            if clip_id in clips and clips[clip_id] != value:
                raise ValueError(f"Conflicting interlace annotations for {clip_id}")
            clips[clip_id] = value
    manifest = {}
    for name in video_names:
        clip_id = os.path.splitext(name)[0]
        if clip_id not in clips:
            raise ValueError(f"Missing interlace annotation for {name}")
        interlaced, order = clips[clip_id]
        if interlaced is False and order == "progressive":
            continue
        if interlaced is True and order in ("tff", "bff"):
            manifest[name] = order
        else:
            raise ValueError(f"Unknown or inconsistent interlace annotation for {name}: {interlaced}, {order}")
    return manifest


def load_video_deinterlaced(path: str, parity: str):
    """Deinterlace a flagged CFR clip at source frame rate; never write beside input."""
    if parity not in ("tff", "bff"):
        raise ValueError("Interlace parity must be tff or bff")
    def run(command):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"{command[0]} failed: {result.stderr[-4000:]}")
        return result.stdout

    probe = json.loads(run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_streams", "-show_entries",
        "stream=avg_frame_rate:frame=best_effort_timestamp_time",
        "-of", "json", os.path.abspath(path),
    ]))
    rate = Fraction(probe["streams"][0]["avg_frame_rate"])
    timestamps = [float(frame["best_effort_timestamp_time"]) for frame in probe["frames"]]
    if rate <= 0 or not timestamps or not np.all(np.isfinite(timestamps)):
        raise ValueError("Flagged clip has missing/invalid timing")
    fps = float(rate)
    if len(timestamps) > 1 and not np.allclose(
        np.diff(timestamps), 1 / fps, rtol=0, atol=max(0.00001, 0.002 / fps)
    ):
        raise ValueError("Deinterlacing currently requires constant frame rate")
    filter_spec = f"bwdif=mode=send_frame:parity={parity}:deint=all"
    with tempfile.TemporaryDirectory(prefix="pose-deinterlace-") as directory:
        output = os.path.join(directory, "progressive.mkv")
        run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", os.path.abspath(path),
             "-map", "0:v:0", "-an", "-vf", filter_spec, "-fps_mode", "passthrough",
             "-c:v", "ffv1", "-threads", "1", output])
        video, _ = load_video_cv(output)
    if len(video) != len(timestamps):
        raise RuntimeError(f"Deinterlacing frame count mismatch: {len(timestamps)} -> {len(video)}")
    return video, fps, {
        "deinterlace": "bwdif", "filter": filter_spec,
        "ffmpeg_version": run(["ffmpeg", "-version"]).splitlines()[0],
        "source_fps": str(rate), "output_fps": str(rate),
        "source_frame_count": len(timestamps), "output_frame_count": len(video),
        "source_start_time_seconds": timestamps[0], "time_origin": "first decoded frame",
    }


def crop_pad_image(image: np.ndarray, bbox: np.ndarray, border: float = 0.25, color: int = 114) -> (np.ndarray, list):
    """
    Crop and pad an image based on a given bounding box and border.

    Parameters:
        image (np.ndarray): The input image as a numpy array.
        bbox (np.ndarray): The bounding box coordinates as a numpy array in the format [x0, y0, x1, y1].
        border (float): The percentage of the maximum image dimension to use as border. Default is 0.25.
        color (int): The color value to use for padding. Default is 114.

    Returns:
        (tuple): A tuple containing the cropped and padded image as a numpy array and the new bounding box
    coordinates as a list.
    """
    # get bbox and image
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0

    # add padding
    dif = np.abs(w - h)
    pad_value_0 = np.floor(dif / 2).astype(int)
    pad_value_1 = dif - pad_value_0

    if w > h:
        y0 -= pad_value_0
        y1 += pad_value_1
    else:
        x0 -= pad_value_0
        x1 += pad_value_1

    # add border
    border = np.round((np.max([w, h]) * border) / 2).astype(int)
    ih, iw = image.shape[:2]
    y0 -= border
    y1 += border
    x0 -= border
    x1 += border

    new_bbox = [x0, y0, x1, y1]

    y0 += ih
    y1 += ih
    x0 += iw
    x1 += iw

    image = np.pad(image, ((ih, ih), (iw, iw), (0, 0)), mode='constant', constant_values=color)  # mode="reflect"
    cropped_image = image[y0:y1, x0:x1]

    return cropped_image, new_bbox


def get_state_counts(index_folder: str) -> str:
    """
    Load index files from a given folder and count the occurrences of each state value.

    Parameters:
        index_folder (str): The folder containing the index files.
    """
    index_files = [os.path.join(index_folder, file) for file in os.listdir(index_folder) if ".csv" in file]
    index_files = [pd.read_csv(file, dtype={"file_names": str, "state": float}) for file in tqdm(index_files)]

    index_file = pd.concat(index_files)
    # print(len(index_file))
    text = ""
    counts = dict(index_file["state"].value_counts())
    for state, count in counts.items():
        text += f"State {state}:   {count}\n"
        # print(f"State {state}:   {count}")
    return text


def reset_state(index_file: str, old_state: int, new_state: int):
    """
    Reset the state of all records in an index file to a new state.

    Parameters:
        index_file (str): The path to the index file or a folder with index files.
        old_state (int): The old state value to be replaced.
        new_state (int): The new state value to replace the old state with.
    """
    if index_file.endswith(".csv"):
        index_files = [index_file]
    else:
        index_files = [os.path.join(index_file, file) for file in os.listdir(index_file) if ".csv" in file]

    for file in tqdm(index_files):
        index_file = pd.read_csv(file, dtype={"file_names": str, "state": float})
        index_file.loc[index_file["state"] == old_state, 'state'] = new_state
        index_file.to_csv(file, index=False)
