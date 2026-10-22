"""No-model tests for project orchestration and ball-overlays."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import numpy as np
import run as runner
from annotate_ball import load_observations, render_video
from track_ball import BallReferenceMatcher, MotionBallFilter
from web_pipeline_entry import configure


class IntegrationTests(unittest.TestCase):
    def test_saved_config_for_another_clip(self):
        settings = configure('test_match_2')
        import importlib
        p = importlib.import_module('possession_settings')
        # It has a real dynamic source, not a literal hard-coded bluevsred.
        self.assertEqual(p.CLIP, 'test_match_2')
        self.assertEqual(p.BALL_CSV, 'outputs/csv/test_match_2_professional_ball.csv')
        self.assertEqual(settings.VIDEO, 'sample_data/input_videos/test_match_2.mp4')

    def test_clip_sanitization(self):
        with self.assertRaises(ValueError):
            configure('../bad')

    def test_three_reference_matcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            image = np.zeros((24, 24, 3), dtype=np.uint8)
            cv2.circle(image, (12, 12), 9, (225, 220, 220), -1)
            for name in ('ball.png', 'ball_2.png', 'ball_3.png'):
                cv2.imwrite(str(folder / name), image)
            matcher = BallReferenceMatcher(folder)
            score = matcher.score(image, (0, 0, 24, 24))
            self.assertGreater(score, .9)

    def test_overlay_uses_measured_pixels_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, output = folder/'in.mp4', folder/'out.mp4'
            csvpath = folder/'ball.csv'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (160, 120))
            self.assertTrue(writer.isOpened())
            for i in range(4):
                writer.write(np.full((120, 160, 3), (30, 100, 50), dtype=np.uint8))
            writer.release()
            with csvpath.open('w', newline='') as f:
                csvwriter = csv.DictWriter(f, fieldnames=['frame', 'source', 'pixel_x', 'pixel_y'])
                csvwriter.writeheader()
                csvwriter.writerow(dict(frame=0,source='predicted',pixel_x='',pixel_y=''))
                csvwriter.writerow(dict(frame=1,source='measured_yolo',pixel_x=75,pixel_y=53))
            positions=load_observations(csvpath)
            self.assertEqual(set(positions), {1})
            self.assertEqual(render_video(source, csvpath, output), 4)
            self.assertGreater(output.stat().st_size, 1000)

    def test_motion_gate_bootstrap(self):
        m = MotionBallFilter(25.0)
        self.assertFalse(m.candidate(1, np.array([30.,20.]), 'yolo', .9)[0])
        self.assertTrue(m.candidate(2, np.array([30.2,20.]), 'yolo', .9)[0])


if __name__ == '__main__':
    unittest.main()
