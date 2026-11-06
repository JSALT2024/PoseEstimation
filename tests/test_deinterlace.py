import json
import unittest
from unittest.mock import patch

from utils import load_video_deinterlaced, deinterlace_from_annotations


class AnnotationTests(unittest.TestCase):
    def test_mixed_annotations_match_stems_case_sensitively(self):
        annotations = {"group": {"clip_order": ["A", "a", "other"],
            "A": {"interlaced": False, "field_order": "progressive"},
            "a": {"interlaced": True, "field_order": "bff"},
            "other": {"interlaced": True, "field_order": "unknown"}}}
        self.assertEqual(deinterlace_from_annotations(annotations, ["A.mp4", "a.mpg"]), {"a.mpg": "bff"})

    def test_unknown_and_missing_are_rejected(self):
        annotations = {"group": {"clip_order": ["clip"],
            "clip": {"interlaced": True, "field_order": "unknown"}}}
        for name in ("clip.mp4", "missing.mp4"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                deinterlace_from_annotations(annotations, [name])

    def test_conflicting_duplicate_ids_are_rejected(self):
        annotations = {str(i): {"clip_order": ["clip"],
            "clip": {"interlaced": True, "field_order": order}}
            for i, order in enumerate(["tff", "bff"])}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            deinterlace_from_annotations(annotations, ["clip.mp4"])


class DeinterlaceTests(unittest.TestCase):
    def run_clip(self, timestamps, decoded_count=3):
        probe = {"streams": [{"avg_frame_rate": "25/1"}], "frames": [
            {"best_effort_timestamp_time": t} for t in timestamps]}
        def command(args, **kwargs):
            from types import SimpleNamespace
            output = json.dumps(probe) if args[0] == "ffprobe" else "ffmpeg test\n"
            return SimpleNamespace(returncode=0, stdout=output, stderr="")
        with patch("utils.subprocess.run", side_effect=command) as run, patch(
            "utils.load_video_cv", return_value=([None] * decoded_count, 25)
        ):
            result = load_video_deinterlaced("input.mpg", "bff")
            self.assertIn("bwdif=mode=send_frame:parity=bff:deint=all", run.call_args_list[1].args[0])
            return result

    def test_preserves_frame_rate_count_and_origin(self):
        video, fps, metadata = self.run_clip([2, 2.04, 2.08])
        self.assertEqual((len(video), fps), (3, 25))
        self.assertEqual(metadata["source_start_time_seconds"], 2)
        self.assertEqual(metadata["output_frame_count"], 3)

    def test_rejects_variable_timing(self):
        with self.assertRaisesRegex(ValueError, "constant frame rate"):
            self.run_clip([0, 0.04, 0.12])

    def test_rejects_partial_decode(self):
        with self.assertRaisesRegex(RuntimeError, "frame count mismatch"):
            self.run_clip([0, 0.04, 0.08], decoded_count=2)

    def test_requires_explicit_parity(self):
        with self.assertRaises(ValueError):
            load_video_deinterlaced("input.mpg", "auto")
