import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import get_state_counts


class IndexSummaryTests(unittest.TestCase):
    def test_worker_finishes_with_empty_sibling_at_both_summary_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index_file_000.csv").write_text("file_names,state\nclip.mp4,1\n")
            # Another worker has truncated its shard and is still rewriting it.
            (root / "index_file_001.csv").write_text("")
            script = Path(__file__).resolve().parents[1] / "pose_prediction_multiple_people_parallel_v2.py"
            argv = [str(script), "--input_folder", directory, "--output_folder", directory,
                    "--index_path", directory, "--index_file_id", "0"]
            with patch.object(sys, "argv", argv), patch("utils_predict.create_models"), patch(
                "utils.get_state_counts", wraps=get_state_counts
            ) as summary:
                runpy.run_path(str(script), run_name="__main__")
            self.assertEqual(summary.call_count, 2)
            for call in summary.call_args_list:
                self.assertEqual(call.args, (directory, 0))

    def test_combined_summary_after_workers_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index_file_000.csv").write_text("file_names,state\na.mp4,1\n")
            (root / "index_file_001.csv").write_text("file_names,state\nb.mp4,-3\n")
            self.assertEqual(get_state_counts(directory, 0), "State 1.0:   1\n")
            combined = get_state_counts(directory)
            self.assertIn("State 1.0:   1\n", combined)
            self.assertIn("State -3.0:   1\n", combined)
