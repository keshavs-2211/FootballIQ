"""Unified FootballIQ pipeline for both web worker and offline CLI."""
from __future__ import annotations
import importlib
import re
import sys
from pathlib import Path

STAGES = {'detection', 'refine', 'ball', 'ball-annotate', 'render', 'analytics', 'possession', 'possession-video'}


def configure(clip: str):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', clip):
        raise ValueError('Invalid clip name. Use letters, numbers, underscores and dashes.')
    import offline_settings as s
    s.CLIP = clip
    s.VIDEO = f'sample_data/input_videos/{clip}.mp4'
    s.ANNOTATED_VIDEO = f'outputs/videos/{clip}_improved.mp4'
    s.FIRST_PASS_CSV = f'outputs/csv/{clip}_improved_coordinates.csv'
    s.CALIBRATION_JSON = f'references/{clip}_pitch_calibration.json'
    s.REFERENCE_DIR = f'references/{clip}'
    s.OUT_CSV = f'outputs/csv/{clip}_professional_coordinates.csv'
    s.PROPOSALS_CSV = f'outputs/csv/{clip}_identity_review.csv'
    s.APPROVALS_CSV = f'outputs/csv/{clip}_identity_approvals.csv'
    s.BALL_CSV = f'outputs/csv/{clip}_professional_ball.csv'
    s.BALL_AUDIT_CSV = f'outputs/csv/{clip}_ball_candidate_audit.csv'
    s.OUT_VIDEO = f'outputs/videos/{clip}_professional_birdeye.mp4'
    s.OUT_SIDE = f'outputs/videos/{clip}_professional_side_by_side.mp4'
    s.REVIEW_DIR = f'outputs/diagnostics/{clip}_id_review'
    return s


def run_stage(clip: str, stage: str):
    if stage not in STAGES:
        raise ValueError(f'Invalid stage {stage}; expected one of {sorted(STAGES)}')
    s = configure(clip)
    if stage == 'detection':
        from detector.jersey_classifier import _saved_references
        refs = _saved_references(Path(s.REFERENCE_DIR))
        from process_video import process_video
        process_video(s.VIDEO, s.ANNOTATED_VIDEO, show_pitch_debug=False,
                      use_legacy_pitch_filter=False, reference_paths=refs)
    else:
        modules = {
            'refine': ('refine_coordinates', 'run'),
            'analytics': ('run_analytics', 'run'),
            'render': ('render_refined_birdeye', 'run'),
            'ball': ('track_ball', 'run'),
            'ball-annotate': ('annotate_ball', 'run'),
            'possession': ('run_possession', 'run'),
            'possession-video': ('render_possession_birdeye', 'run'),
        }
        mod, call = modules[stage]
        getattr(importlib.import_module(mod), call)()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python -m web_pipeline_entry <clip> <stage>')
    run_stage(sys.argv[1], sys.argv[2])
