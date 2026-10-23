"""FootballIQ single-entry CLI. Run from project root: python run.py help."""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
REQUIRED_JERSEYS = ('team_a.png', 'team_a_2.png', 'team_a_3.png',
                    'team_b.png', 'team_b_2.png', 'team_b_3.png')
BALL_IMAGES = ('ball.png', 'ball_2.png', 'ball_3.png')
STAGES = {'annotate': 'detection', 'refine': 'refine', 'analytics': 'analytics',
          'ball': 'ball', 'ball-annotate': 'ball-annotate', 'birdseye': 'render',
          'possession': 'possession', 'possession-video': 'possession-video'}


def check(clip, *, require_model=False):
    """Human-readable audit, no model/network imports required."""
    print(f'FootballIQ environment | clip: {clip}\nProject: {ROOT}')
    mandatory = [ROOT / 'src' / 'process_video.py', ROOT / 'src' / 'footballiq_web.py',
                 ROOT / 'web' / 'templates' / 'index.html',
                 ROOT / 'web' / 'static' / 'app.js',
                 ROOT / 'web' / 'static' / 'styles.css', ROOT / 'src' / 'track_ball.py']
    optional_inputs = [(ROOT / 'models' / 'best.pt', 'model'),
                       (ROOT / 'sample_data' / 'input_videos' / f'{clip}.mp4', 'video'),
                       (ROOT / 'references' / f'{clip}_pitch_calibration.json', 'calibration')]
    for path in mandatory:
        print(f'{"OK" if path.is_file() else "MISSING"}  {path.relative_to(ROOT)}')
    for path, label in optional_inputs:
        print(f'{"OK" if path.is_file() else "ADD"}  {label}: {path.relative_to(ROOT)}')
    for name in REQUIRED_JERSEYS + BALL_IMAGES:
        path = ROOT / 'references' / clip / name
        print(f'{"OK" if path.is_file() else "ADD"}  reference: {clip}/{name}')
    print('\nPython package availability:')
    packages = ('cv2', 'numpy', 'pandas', 'scipy', 'pyarrow', 'ultralytics',
                'supervision', 'flask', 'imageio_ffmpeg', 'matplotlib')
    for name in packages:
        print(f'{"OK" if importlib.util.find_spec(name) else "INSTALL"}  {name}')
    ready = all(p.exists() for p in mandatory)
    ready &= all(p.exists() for p, _ in optional_inputs[:2])
    ready &= all((ROOT / 'references' / clip / name).is_file() for name in REQUIRED_JERSEYS)
    ready &= all(importlib.util.find_spec(name) is not None for name in packages)
    print('\n' + ('READY: source, footage, model and packages found.' if ready else
                   'NOT READY: restore/install the entries marked ADD, MISSING or INSTALL.'))
    return ready


def require_inputs(clip, stage):
    if stage in ('detection', 'refine', 'analytics', 'ball', 'render', 'possession', 'ball-annotate', 'possession-video', 'local-prep'):
        video = ROOT / 'sample_data' / 'input_videos' / f'{clip}.mp4'
        if not video.is_file():
            raise SystemExit(f'Missing input footage: {video}')
    if stage in ('detection', 'ball') and not (ROOT / 'models' / 'best.pt').is_file():
        raise SystemExit('Missing YOLO weights. Add models/best.pt before running detection.')
    if stage == 'detection':
        names = REQUIRED_JERSEYS
    elif stage == 'ball':
        names = BALL_IMAGES
    else:
        names = ()
    missing = [name for name in names if not (ROOT / 'references' / clip / name).is_file()]
    if missing:
        raise SystemExit(f'Missing {clip} reference images: {", ".join(missing)}')
    if stage in ('refine', 'ball') and not (ROOT / 'references' / f'{clip}_pitch_calibration.json').is_file():
        raise SystemExit(f'Missing pitch calibration for {clip}. Use Match Studio to calibrate the pitch.')


def execute(clip, stage):
    require_inputs(clip, stage)
    print(f'\n====== FootballIQ {clip}: {stage} ======', flush=True)
    command = [sys.executable, '-u', str(SRC / 'web_pipeline_entry.py'), clip, stage]
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='FootballIQ: one current pipeline, one website.')
    parser.add_argument('command', nargs='?', default='help',
                        choices=['web', 'check', 'full', 'annotated-ball', 'help', 'references', 'calibrate', *STAGES])
    parser.add_argument('--clip', default='bluevsred', help='Input video filename without .mp4')
    parser.add_argument('--with-ball', action='store_true', help='Include reference-verified ball and possession')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', args.clip):
        parser.error('Clip must contain only letters, numbers, underscore or dash.')
    os.chdir(ROOT)
    if args.command == 'help':
        parser.print_help()
        print('\nExamples:')
        print('  python run.py check --clip bluevsred')
        print('  python run.py web')
        print('  python run.py annotate --clip bluevsred')
        print('  python run.py annotated-ball --clip bluevsred')
        print('  python run.py references --clip bluevsred')
        print('  python run.py calibrate --clip bluevsred')
        print('  python run.py full --clip bluevsred --with-ball')
        return 0
    if args.command == 'check':
        return 0 if check(args.clip) else 1
    if args.command == 'web':
        sys.path.insert(0, str(SRC))
        import runpy
        runpy.run_module('footballiq_web', run_name='__main__')
        return 0
    if args.command in ('references', 'calibrate'):
        require_inputs(args.clip, 'local-prep')
        sys.path.insert(0, str(SRC))
        from web_pipeline_entry import configure
        settings = configure(args.clip)
        if args.command == 'references':
            from capture_references import capture
            capture(settings.VIDEO, settings.REFERENCE_DIR)
        else:
            from utils.manual_pitch_calibration import calibrate
            calibrate(settings.VIDEO, settings.CALIBRATION_JSON)
        return 0
    if args.command == 'full':
        steps = ['detection', 'refine', 'analytics']
        if args.with_ball:
            steps += ['ball', 'ball-annotate']
        steps += ['render']
        if args.with_ball:
            steps += ['possession']
    elif args.command == 'annotated-ball':
        steps = ['detection', 'refine', 'ball', 'ball-annotate']
    else:
        steps = [STAGES[args.command]]
    try:
        for stage in steps:
            execute(args.clip, stage)
    except subprocess.CalledProcessError as exc:
        print(f'\nStage failed ({exc.returncode}). Last stage: {stage}', file=sys.stderr)
        return exc.returncode or 1
    print('\nFinished. Videos: outputs/videos/ | CSV and JSON: outputs/csv/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
