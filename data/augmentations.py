import numpy as np
import random
import cv2


def apply_transform(keypoints, transformation_matrix):
    """Apply transformation to the keypoints."""
    num_kp = keypoints.shape[0]

    keypoints_h = np.hstack([keypoints, np.ones((num_kp, 1))])
    transformed_h = keypoints_h @ transformation_matrix.T
    transformed = transformed_h[:, :2] / transformed_h[:, 2:]

    return transformed


def get_rotation_matrix(angle=0, center=0):
    """Generate a 2D rotation matrix."""
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cx, cy = center

    rotation_matrix = np.array([
        [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
        [sin_a, cos_a, cy - cx * sin_a - cy * cos_a],
        [0, 0, 1]
    ])
    return rotation_matrix


def get_shear_matrix(angle_x=0, angle_y=0):
    """Generate a 2D shear transformation matrix."""
    angle_rad_x = np.radians(angle_x)
    angle_rad_y = np.radians(angle_y)
    shear_x = np.tan(angle_rad_x)
    shear_y = np.tan(angle_rad_y)

    shear_matrix = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0],
        [0, 0, 1]
    ])
    return shear_matrix


def get_perspective_matrix(portion=0, reference_size=512):
    """Generate a 2D perspective transformation matrix."""

    src = np.array(((0, 1), (1, 1), (0, 0), (1, 0)), dtype=np.float32) * reference_size
    dest = np.array(((0 + portion, 1), (1 - portion, 1), (0, 0), (1, 0)), dtype=np.float32) * reference_size

    perspective_matrix = cv2.getPerspectiveTransform(src, dest)

    return perspective_matrix


def get_bbox(keypoints):
    """Get bbox from a keypoints."""
    # _keypoints = []
    # for kp in keypoints:
    #     _keypoints.extend(kp)
    _keypoints = np.array(keypoints)
    return np.min(_keypoints[:, 0]), np.min(_keypoints[:, 1]), np.max(_keypoints[:, 0]), np.max(_keypoints[:, 1])


def use_augmentation(p=0):
    return random.random() <= p


def all_same(keypoints):
    return np.sum(keypoints == keypoints[0, 0]) == keypoints.size
