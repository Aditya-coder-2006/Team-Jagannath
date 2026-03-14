import unittest
import numpy as np
import torch
import os
import sys
from unittest import mock

# Ensure `project_root` is on sys.path so imports work regardless of CWD.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_processing.frame_sampler import FrameSampler
from inference.ranking import aggregate_ranking_scores

class TestDataProcessing(unittest.TestCase):
    def test_uniform_sampler_pad(self):
        sampler = FrameSampler(target_frames=10, method="uniform")
        frames = [np.zeros((10,10,3)) for _ in range(5)]
        indices = sampler.sample(frames)
        self.assertEqual(len(indices), 10)
        self.assertEqual(indices[-1], 4)
        
    def test_ranking_aggregation(self):
        # 3 frames, P(0 < 1) = 0.9, P(0 < 2) = 0.8, P(1 < 2) = 0.7
        # Expected order: 0, 1, 2
        probs = torch.tensor([
            [0.5, 0.9, 0.8],
            [0.1, 0.5, 0.7],
            [0.2, 0.3, 0.5]
        ])
        
        pred_order = aggregate_ranking_scores(probs, 3)
        self.assertEqual(pred_order, [0, 1, 2])

    def test_predict_sequence_one_indexed(self):
        # monkeypatch parts of VideoPredictor to avoid heavy dependencies
        from inference.predict_sequence import VideoPredictor, build_pairwise_matrix, aggregate_ranking_scores
        # create a dummy predictor instance without loading weights
        vp = VideoPredictor.__new__(VideoPredictor)
        vp.device = torch.device("cpu")
        vp.config = type("C", (), {"model": type("M", (), {"num_frames": 3}), "training": type("T", (), {"device": "cpu"})})()
        # patch required methods
        vp.extractor = type("E", (), {"get_frames": lambda self, x: [np.zeros((2,2,3))]*3})()
        vp.flow_extractor = type("F", (), {
            "sequence_flow_magnitudes": lambda self, frames: [0]*len(frames),
            "compute_sequence_flow": lambda self, frames: [np.zeros((2,2,2))]*len(frames)
        })()
        vp.sampler = type("S", (), {"sample": lambda self, f, m=None: list(range(3))})()
        class DummyModel:
            def to(self, d): pass
            def eval(self): pass
            def encode_frames(self, f, fl): return torch.zeros((1,3,10))
        vp.model = DummyModel()
        # override build_pairwise_matrix and aggregation to return predictable order
        def fake_build(model, emb, device):
            return torch.zeros((3,3))
        def fake_agg(pm, n):
            return [0,1,2]
        import inference.predict_sequence as ps
        ps.build_pairwise_matrix = fake_build
        ps.aggregate_ranking_scores = fake_agg

        out = vp.predict("dummy_path")
        # check that the returned string is 1-indexed
        self.assertEqual(out, "1 2 3")

    def test_submission_generator_limit(self):
        import shutil
        from scripts.submission_generator import generate_submission

        class DummyPredictor:
            def __init__(self, checkpoint_path, config_path):
                self.checkpoint_path = checkpoint_path
                self.config_path = config_path

            def predict(self, video_path):
                return "1 2 3"

        tmpdir = os.path.join(_PROJECT_ROOT, "tests_runtime_submission")
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        os.makedirs(tmpdir, exist_ok=True)

        try:
            for i in range(150):
                open(os.path.join(tmpdir, f"video_{i}.mp4"), "a").close()
            out_csv = os.path.join(tmpdir, "out.csv")
            with mock.patch("scripts.submission_generator.VideoPredictor", DummyPredictor):
                generate_submission(
                    tmpdir,
                    checkpoint_path="/dev/null",
                    config_path="/dev/null",
                    output_csv=out_csv,
                    max_videos=120
                )
            # read csv and ensure 121 lines (header + 120 entries)
            with open(out_csv) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 121)
        finally:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
