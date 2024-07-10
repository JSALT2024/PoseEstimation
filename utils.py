import numpy as np


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
