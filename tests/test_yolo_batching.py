import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from ultralytics.cfg import DEFAULT_CFG_DICT
from utils_predict import _resolve_yolo_device, yolo_predict_batch


class FakeYoloModel:
    def __init__(self):
        self.yolo_device = "cuda:0"
        self.yolo_half = False
        self.overrides = {}
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append((len(source), kwargs))
        result = SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=torch.empty((0, 4)),
                conf=torch.empty((0,)),
            ),
            keypoints=SimpleNamespace(xy=torch.empty((0, 17, 2))),
        )
        return [result for _ in source]


class YoloBatchingTests(unittest.TestCase):
    def test_splits_input_into_bounded_batches_and_preserves_order(self):
        model = FakeYoloModel()
        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(5)]

        predictions = yolo_predict_batch(images, model, batch_size=2)

        self.assertEqual(len(predictions), 5)
        self.assertEqual([size for size, _ in model.calls], [2, 2, 1])
        self.assertTrue(all(call[1]["device"] == "cuda:0" for call in model.calls))

    def test_rejects_invalid_batch_size(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            yolo_predict_batch([np.zeros((1, 1, 3))], FakeYoloModel(), batch_size=0)

    def test_requests_half_precision_with_supported_ultralytics_argument(self):
        model = FakeYoloModel()
        model.yolo_half = True

        yolo_predict_batch([np.zeros((8, 8, 3), dtype=np.uint8)], model)

        precision_argument = "quantize" if "quantize" in DEFAULT_CFG_DICT else "half"
        expected_value = 16 if precision_argument == "quantize" else True
        self.assertEqual(model.calls[0][1][precision_argument], expected_value)


class ResolveYoloDeviceTests(unittest.TestCase):
    @mock.patch("utils_predict.torch.cuda.device_count", return_value=1)
    @mock.patch("utils_predict.torch.cuda.is_available", return_value=True)
    def test_auto_selects_visible_gpu(self, _available, _count):
        self.assertEqual(_resolve_yolo_device("auto"), "cuda:0")

    @mock.patch("utils_predict.torch.cuda.is_available", return_value=False)
    def test_explicit_gpu_fails_when_pytorch_cannot_see_one(self, _available):
        with self.assertRaisesRegex(RuntimeError, "torch.cuda.is_available"):
            _resolve_yolo_device("cuda:0")

if __name__ == "__main__":
    unittest.main()
