"""FootballIQ local Flask interface. Run from the FootballIQ project root.

This is a LOCAL development server, not an authenticated public deployment.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import subprocess
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent.parent
JOB_DIR = ROOT / 'web_data' / 'jobs'
INPUT_DIR = ROOT / 'sample_data' / 'input_videos'
REF_DIR = ROOT / 'references'
CSV_DIR = ROOT / 'outputs' / 'csv'
VIDEO_DIR = ROOT / 'outputs' / 'videos'
ANALYTICS_DIR = ROOT / 'outputs' / 'analytics'
for directory in (JOB_DIR, INPUT_DIR, REF_DIR, CSV_DIR, VIDEO_DIR, ANALYTICS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(ROOT / 'web' / 'templates'),
            static_folder=str(ROOT / 'web' / 'static'), static_url_path='/fi-assets')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GiB local prototype
app.json.sort_keys = False

_LOCK = threading.RLock()
_active_id: str | None = None
# Three samples per outfield team and three ball crops are mandatory.
# Keep team_a.png / team_b.png as the first samples: the existing
# jersey_classifier._saved_references() reads these + _2.png + _3.png.
REFERENCE_COUNT = 3

REQUIRED_ROLES = (
    "team_a",
    "team_b",
    "team_a_gk",
    "team_b_gk",
    "referee",
    "ball",
)

ROLE_STEMS = {
    role: f"{role}.png"
    for role in REQUIRED_ROLES
}


def _reference_filename(role, sample=1):
    if role not in ROLE_STEMS:
        raise ValueError("Unknown reference role")

    if (
        not isinstance(sample, int)
        or not 1 <= sample <= REFERENCE_COUNT
    ):
        raise ValueError("Invalid reference sample")

    suffix = "" if sample == 1 else f"_{sample}"

    return f"{role}{suffix}.png"


LANDMARKS = {
    'top_left': ('Top-left pitch corner', 0., 0.),
    'top_right': ('Top-right pitch corner', 105., 0.),
    'bottom_left': ('Bottom-left pitch corner', 0., 68.),
    'bottom_right': ('Bottom-right pitch corner', 105., 68.),
    'halfway_top': ('Halfway / top touchline', 52.5, 0.),
    'halfway_bottom': ('Halfway / bottom touchline', 52.5, 68.),
    'center_spot': ('Centre spot', 52.5, 34.),
    'left_penalty_top': ('Left penalty area top corner', 16.5, 13.84),
    'left_penalty_bottom': ('Left penalty area bottom corner', 16.5, 54.16),
    'right_penalty_top': ('Right penalty area top corner', 88.5, 13.84),
    'right_penalty_bottom': ('Right penalty area bottom corner', 88.5, 54.16),
    'left_penalty_spot': ('Left penalty spot', 11., 34.),
    'right_penalty_spot': ('Right penalty spot', 94., 34.),
    'left_goalarea_top': ('Left goal area upper corner', 5.5, 24.84),
    'left_goalarea_bottom': ('Left goal area lower corner', 5.5, 43.16),
    'right_goalarea_top': ('Right goal area upper corner', 99.5, 24.84),
    'right_goalarea_bottom': ('Right goal area lower corner', 99.5, 43.16),
}


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _job_path(job_id):
    if not re.fullmatch(r'[a-f0-9]{12}', job_id or ''):
        abort(404, description='Project not found')
    return JOB_DIR / (job_id + '.json')


def _read_job(job_id):
    p = _job_path(job_id)
    if not p.exists():
        abort(404, description='Project not found')
    return json.loads(p.read_text(encoding='utf-8'))


def _write_job(job):
    job['updated_at'] = _utc()
    p = _job_path(job['id'])
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(p)


def _update_job(job_id, **changes):
    with _LOCK:
        job = _read_job(job_id)
        job.update(changes)
        _write_job(job)
        return job


def _safe_error(message, status=400):
    return jsonify(error=str(message)), status


def _video_meta(filename):
    cap = cv2.VideoCapture(str(filename))
    try:
        if not cap.isOpened():
            raise ValueError('Unsupported or damaged video. Use a valid MP4 file.')
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames < 2 or fps <= 0 or not math.isfinite(fps) or w < 64 or h < 64:
            raise ValueError('Cannot read valid video metadata.')
        return {'frames': frames, 'fps': round(fps, 3), 'duration_s': round(frames/fps, 2),
                'width': w, 'height': h}
    finally:
        cap.release()


def _frame(job, seconds=0):
    cap = cv2.VideoCapture(str(INPUT_DIR / (job['clip'] + '.mp4')))
    try:
        if not cap.isOpened():
            raise ValueError('Unable to open uploaded video')
        max_sec = max(0, job['meta']['duration_s'] - 1/max(1, job['meta']['fps']))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, min(float(seconds), max_sec))*1000)
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Unable to read requested video frame')
        return frame
    finally:
        cap.release()


# OpenCV commonly writes MPEG-4 Part 2 (mp4v). Chrome/Firefox browsers often
# cannot decode it, although Windows media players can. Keep original videos and
# publish separate fast-start H.264 copies under outputs/videos/.
VIDEO_SOURCES = (
    ('annotated', 'Annotated match', ('_ball_annotated.mp4', '_improved.mp4')),
    ('side_by_side', 'Side-by-side comparison', ('_professional_side_by_side.mp4',)),
    ('birdseye', "Bird's-eye tactical view", ('_professional_birdeye.mp4',)),
    ('possession', 'Possession replay', ('_possession_side_by_side.mp4',)),
)


def _existing_video(clip, suffixes):
    for suffix in suffixes:
        candidate = VIDEO_DIR / (clip + suffix)
        if candidate.is_file() and candidate.stat().st_size > 4096:
            return candidate
    return None


def _ffmpeg_executable():
    program = shutil.which('ffmpeg')
    if program:
        return program
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        raise RuntimeError('FFmpeg is required for browser playback. Run: python -m pip install imageio-ffmpeg')


def _transcode_one(source):
    """Return a new H.264 MP4, NEVER overwrite OpenCV's original file."""
    dest = source.with_name(source.stem + '_web.mp4')
    if dest.is_file() and dest.stat().st_size > 4096 and dest.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return dest
    temp = source.with_name(source.stem + '_web_tmp.mp4')
    temp.unlink(missing_ok=True)
    cmd = [_ffmpeg_executable(), '-hide_banner', '-loglevel', 'error', '-y',
           '-i', str(source), '-map', '0:v:0', '-an',
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temp)]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60*30)
        if proc.returncode or not temp.is_file() or temp.stat().st_size < 4096:
            raise RuntimeError('FFmpeg conversion failed: ' + (proc.stderr or '')[-700:])
        # OpenCV checks that the compressed video is actually readable.
        cap = cv2.VideoCapture(str(temp))
        valid = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 1
        cap.release()
        if not valid:
            raise RuntimeError('Converted video cannot be decoded.')
        temp.replace(dest)
        return dest
    finally:
        temp.unlink(missing_ok=True)


def _prepare_web_media(job_id):
    """Runs in one worker thread. Also repairs files from already-finished jobs."""
    clip = _read_job(job_id)['clip']
    failures = []
    sources = [(role, source) for role, _, suffixes in VIDEO_SOURCES
               if (source := _existing_video(clip, suffixes)) is not None]
    for index, (role, source) in enumerate(sources):
        try:
            _update_job(job_id, media_stage=f'Converting {role.replace("_", " ")} for browser ({index+1}/{len(sources)})')
            _transcode_one(source)
        except Exception as exc:
            failures.append(f'{role}: {str(exc)[:400]}')
    _update_job(job_id, media_state='partial' if failures else 'done',
                media_stage='Browser videos ready' if not failures else 'Some videos need attention',
                media_errors=failures)
    return failures


def _prepare_web_media_async(job_id):
    with _LOCK:
        job = _read_job(job_id)
        if job.get('state') == 'running' or job.get('media_state') == 'running':
            return False
        _update_job(job_id, media_state='running', media_stage='Preparing browser videos', media_errors=[])
    def worker():
        try:
            _prepare_web_media(job_id)
        except Exception as exc:
            _update_job(job_id, media_state='partial', media_errors=[str(exc)[:600]])
    threading.Thread(target=worker, daemon=True, name=f'media-{job_id}').start()
    return True


def _overview(job):
    result = dict(job)
    clip = job['clip']
    rd = REF_DIR / clip
    result['reference_samples'] = {
        role: [(rd / _reference_filename(role, i)).is_file()
               for i in range(1, (REFERENCE_COUNT) + 1)]
        for role in ROLE_STEMS
    }
    result['reference_counts'] = {
        role: sum(samples) for role, samples in result['reference_samples'].items()
    }
    # Keep the older references flags for compatibility with existing clients.
    result['references'] = {role: result['reference_counts'][role] > 0 for role in ROLE_STEMS}
    result['references_ready'] = all(result['reference_counts'][role] == REFERENCE_COUNT
                                      for role in REQUIRED_ROLES)
    calibration_path = REF_DIR / f'{clip}_pitch_calibration.json'
    result['calibrated'] = calibration_path.is_file()
    result['calibration_time_s'] = 0.0
    if calibration_path.is_file():
        try:
            result['calibration_time_s'] = float(json.loads(calibration_path.read_text(encoding='utf-8')).get('frame_time_seconds', 0))
        except (ValueError, TypeError, OSError):
            pass
    return result


def _send_allowed(path):
    allowed = [INPUT_DIR, VIDEO_DIR, CSV_DIR, ANALYTICS_DIR, REF_DIR]
    resolved = path.resolve()
    if not any(resolved.is_relative_to(base.resolve()) for base in allowed):
        abort(403)
    if not resolved.is_file():
        abort(404)
    return send_file(resolved, mimetype='video/mp4' if resolved.suffix.lower() == '.mp4' else None, conditional=True, max_age=0)


@app.errorhandler(413)
def too_large(_):
    return _safe_error('File too large. Maximum upload size is 1 GB.', 413)


@app.get('/')
def home():
    return render_template('index.html')


@app.get('/api/projects')
def projects():
    items = []
    for p in JOB_DIR.glob('*.json'):
        try:
            items.append(_overview(json.loads(p.read_text(encoding='utf-8'))))
        except (ValueError, KeyError):
            pass
    items.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return jsonify(projects=items[:60])


@app.post('/api/projects')
def create_project():
    video = request.files.get('video')
    if video is None or not video.filename:
        return _safe_error('Choose an MP4 video to upload.')
    if Path(secure_filename(video.filename)).suffix.lower() != '.mp4':
        return _safe_error('Please upload an .mp4 video.')
    project_id = uuid.uuid4().hex[:12]
    slug = re.sub(r'[^a-z0-9]+', '_', Path(video.filename).stem.lower()).strip('_')[:26] or 'match'
    clip = f'fi_{project_id}_{slug}'
    destination = INPUT_DIR / (clip + '.mp4')
    video.save(destination)
    try:
        meta = _video_meta(destination)
    except (ValueError, cv2.error) as err:
        destination.unlink(missing_ok=True)
        return _safe_error(err)
    team_a = request.form.get('team_a', 'Home').strip()[:35] or 'Home'
    team_b = request.form.get('team_b', 'Away').strip()[:35] or 'Away'
    job = {'id': project_id, 'clip': clip, 'original_filename': secure_filename(video.filename),
           'title': request.form.get('title', Path(video.filename).stem).strip()[:70] or 'Match',
           'team_a': team_a, 'team_b': team_b, 'meta': meta,
           'state': 'setup', 'stage': 'Upload complete', 'progress': 0,
           'created_at': _utc(), 'updated_at': _utc(), 'error': None,
           'optional_ball': False, 'log_tail': [], 'media_state': 'idle',
           'media_stage': '', 'media_errors': []}
    with _LOCK:
        _write_job(job)
    return jsonify(project=_overview(job)), 201


@app.get('/api/local-matches')
def local_matches():
    """List existing inputs the user already analyzed outside the web app."""
    names = []
    for path in sorted(INPUT_DIR.glob('*.mp4')):
        if path.stem.startswith('fi_') or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', path.stem):
            continue
        has_video = any(_existing_video(path.stem, suffixes) for _, _, suffixes in VIDEO_SOURCES)
        has_stats = (CSV_DIR / f'{path.stem}_analytics_summary.json').is_file()
        names.append({'clip': path.stem, 'video': path.name,
                      'has_outputs': has_video or has_stats})
    return jsonify(matches=names[:100])


@app.post('/api/projects/import')
def import_existing_project():
    """Add an already-processed match without renaming or copying its files."""
    data = request.get_json(silent=True) or {}
    clip = data.get('clip', '')
    if not isinstance(clip, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,80}', clip):
        return _safe_error('Invalid local match name.')
    original = INPUT_DIR / f'{clip}.mp4'
    if not original.is_file():
        return _safe_error('Match input video is not in sample_data/input_videos/.', 404)
    for file in JOB_DIR.glob('*.json'):
        try:
            old = json.loads(file.read_text(encoding='utf-8'))
            if old.get('clip') == clip:
                return jsonify(project=_overview(old), existing=True)
        except (ValueError, KeyError):
            continue
    try:
        meta = _video_meta(original)
    except (ValueError, cv2.error) as exc:
        return _safe_error(exc)
    completed = (CSV_DIR / f'{clip}_analytics_summary.json').is_file() or any(
        _existing_video(clip, suffixes) for _, _, suffixes in VIDEO_SOURCES)
    job_id = uuid.uuid4().hex[:12]
    job = {'id': job_id, 'clip': clip, 'original_filename': original.name,
           'title': clip.replace('_', ' ').title(), 'team_a': 'Team A', 'team_b': 'Team B',
           'meta': meta, 'state': 'done' if completed else 'setup',
           'stage': 'Imported existing outputs' if completed else 'Imported video',
           'progress': 100 if completed else 0, 'error': None,
           'created_at': _utc(), 'updated_at': _utc(), 'optional_ball': False,
           'log_tail': [], 'media_state': 'idle', 'media_errors': [], 'media_stage': ''}
    with _LOCK:
        _write_job(job)
    return jsonify(project=_overview(job), existing=False), 201


@app.get('/api/projects/<job_id>')
def get_project(job_id):
    return jsonify(project=_overview(_read_job(job_id)))


@app.get('/api/projects/<job_id>/video')
def source_video(job_id):
    job = _read_job(job_id)
    return _send_allowed(INPUT_DIR / (job['clip'] + '.mp4'))


@app.get('/api/projects/<job_id>/frame')
def frame_image(job_id):
    job = _read_job(job_id)
    try:
        t = float(request.args.get('time', '0'))
        if not math.isfinite(t):
            return _safe_error('Invalid timestamp')
        frame = _frame(job, t)
        # Original resolution, so relative UI coordinates map exactly.
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 87])
        if not ok:
            raise ValueError('JPEG encoding failed')
        return send_file(io.BytesIO(encoded.tobytes()), mimetype='image/jpeg', download_name='frame.jpg')
    except (ValueError, cv2.error) as err:
        return _safe_error(err)


@app.post('/api/projects/<job_id>/reference')
def add_reference(job_id):
    job = _read_job(job_id)
    if job['state'] == 'running':
        return _safe_error('Cannot change reference images while processing.', 409)
    data = request.get_json(silent=True) or {}
    role = data.get('role')
    if role not in ROLE_STEMS:
        return _safe_error('Invalid reference role')
    try:
        sample = int(data.get('sample', 1))
        filename = _reference_filename(role, sample)
        t = float(data.get('time', 0))
        if not math.isfinite(t) or not (0 <= t <= job['meta']['duration_s'] + 0.01):
            raise ValueError('Reference timestamp is outside the video')
        box = data.get('box')
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('Select a crop around the jersey, referee shirt or ball')
        vals = [float(v) for v in box]
        if not all(math.isfinite(v) for v in vals) or not all(0 <= v <= 1 for v in vals):
            raise ValueError('Crop coordinates must be normalized between 0 and 1')
        x1, y1, x2, y2 = vals
        if x2-x1 < .005 or y2-y1 < .005:
            raise ValueError('Crop is too small. Select the visible object.')
        im = _frame(job, t)
        height, width = im.shape[:2]
        px1, px2 = round(x1*width), round(x2*width)
        py1, py2 = round(y1*height), round(y2*height)
        crop = im[py1:py2, px1:px2]
        if crop.shape[0] < 5 or crop.shape[1] < 5:
            raise ValueError('Crop is too small')
        directory = REF_DIR / job['clip']
        directory.mkdir(parents=True, exist_ok=True)
        out = directory / filename
        if not cv2.imwrite(str(out), crop):
            raise ValueError('Failed to save reference crop')
        overview = _overview(job)
        return jsonify(ok=True, role=role, sample=sample,
                       url=f'/api/projects/{job_id}/reference/{role}?sample={sample}&v={int(time.time())}',
                       references=overview['references'],
                       reference_samples=overview['reference_samples'],
                       reference_counts=overview['reference_counts'],
                       references_ready=overview['references_ready'])
    except (TypeError, ValueError, cv2.error) as err:
        return _safe_error(err)


@app.get('/api/projects/<job_id>/reference/<role>')
def reference_image(job_id, role):
    job = _read_job(job_id)
    try:
        sample = int(request.args.get('sample', '1'))
        filename = _reference_filename(role, sample)
    except (ValueError, TypeError):
        abort(404)
    return _send_allowed(REF_DIR / job['clip'] / filename)


@app.get('/api/landmarks')
def landmarks():
    return jsonify(landmarks=[{'id': key, 'label': label, 'x': x, 'y': y}
                              for key, (label, x, y) in LANDMARKS.items()])


@app.post('/api/projects/<job_id>/calibrate')
def calibrate(job_id):
    job = _read_job(job_id)
    if job['state'] == 'running':
        return _safe_error('Cannot recalibrate while processing.', 409)
    data = request.get_json(silent=True) or {}
    pts = data.get('points', [])
    try:
        calib_time = float(data.get('time', 0))
        if not math.isfinite(calib_time) or not (0 <= calib_time <= job['meta']['duration_s']):
            raise ValueError('Calibration frame time must be within the video')
    except (ValueError, TypeError) as err:
        return _safe_error(err)
    if not isinstance(pts, list) or not (4 <= len(pts) <= 17):
        return _safe_error('Select at least four real, visible pitch landmarks.')
    keys = [p.get('id') for p in pts if isinstance(p, dict)]
    if len(keys) != len(pts) or len(set(keys)) != len(keys) or any(k not in LANDMARKS for k in keys):
        return _safe_error('Each landmark must be unique and valid.')
    try:
        src_norm = np.asarray([p['uv'] for p in pts], dtype=float)
        if src_norm.shape != (len(pts), 2) or not np.isfinite(src_norm).all() or (src_norm < 0).any() or (src_norm > 1).any():
            raise ValueError('Image coordinates must lie within the video frame')
        src = src_norm * [job['meta']['width'], job['meta']['height']]
        dst = np.asarray([[LANDMARKS[k][1], LANDMARKS[k][2]] for k in keys], dtype=float)
        if cv2.contourArea(cv2.convexHull(src.astype(np.float32))) < 150:
            raise ValueError('Landmarks are too close together or collinear.')
        if len(src) == 4:
            H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        else:
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-15:
            raise ValueError('Calibration is degenerate. Select better spaced landmarks.')
        pred = cv2.perspectiveTransform(src.astype(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(pred-dst, axis=1)
        median_error = float(np.median(errors))
        if not math.isfinite(median_error) or median_error > 8:
            raise ValueError(f'Calibration mismatch (median error {median_error:.1f} m). Check the landmark labels.')
        # The existing coordinate refiner maps every video frame BACK TO FRAME 0.
        # When users calibrate a later frame, convert its H into a frame-0 H.
        # Using the later-frame H unmodified makes ALL projected distances wrong.
        selected_H = H.copy()
        selected_frame = min(job['meta']['frames']-1, int(round(calib_time * job['meta']['fps'])))
        if selected_frame:
            try:
                from camera.camera_movement import CameraMotionCompensator
                cap = cv2.VideoCapture(str(INPUT_DIR / (job['clip'] + '.mp4')))
                try:
                    ok, first = cap.read()
                    if not ok:
                        raise ValueError('Cannot read first frame for calibration alignment.')
                    motion = CameraMotionCompensator(first)
                    to_first = np.eye(3, dtype=np.float64)
                    for _ in range(selected_frame):
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError('Video ended during calibration alignment.')
                        to_first = motion.update(frame)
                finally:
                    cap.release()
                if abs(np.linalg.det(to_first)) < 1e-12:
                    raise ValueError('Camera motion alignment matrix is degenerate.')
                H = selected_H @ np.linalg.inv(to_first)
                H /= H[2,2] if abs(H[2,2]) > 1e-10 else np.linalg.norm(H)
            except ImportError as exc:
                raise ValueError('Nonzero-frame calibration requires camera/camera_movement.py with CameraMotionCompensator.') from exc
        payload = {'video_width': job['meta']['width'], 'video_height': job['meta']['height'],
                   'pitch_length': 105.0, 'pitch_width': 68.0,
                   'homography': H.tolist(), 'selected_frame_homography': selected_H.tolist(),
                   'image_points': src.tolist(), 'pitch_points': dst.tolist(),
                   'frame_time_seconds': round(calib_time, 5),
                   'frame_index': min(job['meta']['frames'] - 1, round(calib_time * job['meta']['fps']))}
        path = REF_DIR / (job['clip'] + '_pitch_calibration.json')
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return jsonify(ok=True, median_error_m=round(median_error, 2),
                       warning='High reprojection error; double-check the points.' if median_error > 2 else None)
    except (ValueError, TypeError, KeyError, cv2.error) as err:
        return _safe_error(err)


def _run_script(job_id, label, progress, script, optional=False):
    job = _update_job(job_id, stage=label, progress=progress)
    clip = job['clip']
    command = [sys.executable, '-u', str(ROOT / 'src' / 'web_pipeline_entry.py'), clip, script]
    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors='replace',
                            bufsize=1, stdin=subprocess.DEVNULL)
    tail = []
    assert proc.stdout is not None
    log_path = JOB_DIR / (job_id + '.log')
    with log_path.open('a', encoding='utf-8') as logfile:
        logfile.write(f'\n=== {label} ===\n')
        for line in proc.stdout:
            logfile.write(line)
            tail.append(line.rstrip()[:280])
            if len(tail) > 24:
                tail = tail[-24:]
            if len(tail) % 5 == 0:
                _update_job(job_id, log_tail=tail[-12:])
    rc = proc.wait()
    _update_job(job_id, log_tail=tail[-12:])
    if rc:
        raise RuntimeError(f'{label} failed (exit {rc}). See web_data/jobs/{job_id}.log. ' + ' '.join(tail[-3:])[:500])
    clip = job['clip']
    expected = {
        'ball-annotate': VIDEO_DIR / f'{clip}_ball_annotated.mp4',
        'detection': CSV_DIR / f'{clip}_improved_coordinates.csv',
        'refine': CSV_DIR / f'{clip}_professional_coordinates.csv',
        'analytics': CSV_DIR / f'{clip}_analytics_summary.json',
        'render': VIDEO_DIR / f'{clip}_professional_side_by_side.mp4',
        'ball': CSV_DIR / f'{clip}_professional_ball.csv',
        'possession': CSV_DIR / f'{clip}_possession_summary.json',
    }
    required = expected.get(script)
    if required is not None and (not required.is_file() or required.stat().st_size == 0):
        raise RuntimeError(f'{label} finished without producing {required.relative_to(ROOT)}. See {log_path.relative_to(ROOT)}')


def _pipeline_worker(job_id, with_ball):
    global _active_id
    try:
        _run_script(job_id, 'Detecting players and assigning teams', 10, 'detection')
        _run_script(job_id, 'Refining pitch coordinates', 40, 'refine')
        _run_script(job_id, 'Generating heatmaps and analytics', 62, 'analytics')
        if with_ball:
            _run_script(job_id, 'Tracking the ball (experimental)', 76, 'ball')
            _run_script(job_id, 'Overlaying verified ball locations', 80, 'ball-annotate')
        _run_script(job_id, 'Rendering tactical video', 84, 'render')
        if with_ball:
            _run_script(job_id, 'Calculating possession (experimental)', 92, 'possession')
        _update_job(job_id, stage='Preparing browser-compatible videos', progress=95,
                    media_state='running', media_errors=[])
        failures = _prepare_web_media(job_id)
        _update_job(job_id, state='done', stage='Complete' if not failures else 'Complete (video warnings)',
                    progress=100, error=None)
    except Exception as error:
        _update_job(job_id, state='failed', stage='Processing stopped', error=str(error))
    finally:
        with _LOCK:
            _active_id = None


@app.post('/api/projects/<job_id>/start')
def start_job(job_id):
    global _active_id
    with _LOCK:
        job = _read_job(job_id)
        if job['state'] == 'running':
            return _safe_error('Already processing.', 409)
        if _active_id is not None:
            return _safe_error('Another match is being processed. This CPU-friendly app handles one job at a time.', 409)
        existing = _overview(job)
        if not existing['references_ready']:
            missing = [f'{role}: {existing["reference_counts"][role]}/{REFERENCE_COUNT}'
                       for role in REQUIRED_ROLES if existing['reference_counts'][role] != REFERENCE_COUNT]
            return _safe_error('Save three photos for each of the six roles (18 total). Missing: ' + ', '.join(missing))
        if not existing['calibrated']:
            return _safe_error('Calibrate the pitch with at least four landmarks.')
        for script in ('process_video.py', 'refine_coordinates.py', 'run_analytics.py',
                       'render_refined_birdeye.py', 'offline_settings.py'):
            if not (ROOT / 'src' / script).is_file():
                return _safe_error(f'Missing existing pipeline file: {script}. Follow the installation guide.')
        data = request.get_json(silent=True) or {}
        with_ball = bool(data.get('with_ball', False))
        if with_ball:
            for script in ('track_ball.py', 'annotate_ball.py', 'run_possession.py'):
                if not (ROOT / 'src' / script).is_file():
                    return _safe_error(f'Missing optional pipeline file: {script}.')
        _active_id = job_id
        _update_job(job_id, state='running', stage='Starting processing', progress=2,
                    error=None, optional_ball=with_ball, log_tail=[], media_state='idle', media_errors=[])
        threading.Thread(target=_pipeline_worker, args=(job_id, with_ball), daemon=True).start()
    return jsonify(ok=True, project=_overview(_read_job(job_id)))


@app.get('/api/projects/<job_id>/results')
def results(job_id):
    job = _read_job(job_id)
    clip = job['clip']
    report = {}
    for kind, path in {'analytics': CSV_DIR / f'{clip}_analytics_summary.json',
                       'possession': CSV_DIR / f'{clip}_possession_summary.json'}.items():
        if path.is_file():
            try:
                report[kind] = json.loads(path.read_text(encoding='utf-8'))
            except ValueError:
                pass
    files = []
    candidates = [
        ('Analytics overview', ANALYTICS_DIR / clip / 'dashboard' / 'analytics_overview.png', 'image'),
        ('Team A heatmap', ANALYTICS_DIR / clip / 'heatmaps' / 'team_a_heatmap.png', 'image'),
        ('Team B heatmap', ANALYTICS_DIR / clip / 'heatmaps' / 'team_b_heatmap.png', 'image'),
        ('Team A positioning', ANALYTICS_DIR / clip / 'formations' / 'team_a_average_positions.png', 'image'),
        ('Team B positioning', ANALYTICS_DIR / clip / 'formations' / 'team_b_average_positions.png', 'image'),
        ('Team spread', ANALYTICS_DIR / clip / 'formations' / 'team_spread_over_time.png', 'image'),
        ('Player distance CSV', CSV_DIR / f'{clip}_player_statistics.csv', 'csv'),
        ('Team distance CSV', CSV_DIR / f'{clip}_team_statistics.csv', 'csv'),
        ('Player positions CSV', CSV_DIR / f'{clip}_professional_coordinates.csv', 'csv'),
        ('Analytics JSON', CSV_DIR / f'{clip}_analytics_summary.json', 'json'),
        ('Possession JSON', CSV_DIR / f'{clip}_possession_summary.json', 'json'),
    ]
    for heatmap in sorted((ANALYTICS_DIR / clip / 'heatmaps' / 'players').glob('*.png'))[:120]:
        candidates.append((f'Player heatmap · {heatmap.stem.replace("_", " ")}', heatmap, 'player_image'))
    for role, label, suffixes in VIDEO_SOURCES:
        path = _existing_video(clip, suffixes)
        if path is None:
            continue
        web_copy = path.with_name(path.stem + '_web.mp4')
        is_playable = web_copy.is_file() and web_copy.stat().st_size > 4096 and web_copy.stat().st_mtime_ns >= path.stat().st_mtime_ns
        selected = web_copy if is_playable else path
        rel = selected.relative_to(ROOT).as_posix()
        orig_rel = path.relative_to(ROOT).as_posix()
        files.append({'label': label, 'role': role, 'path': rel, 'kind': 'video',
                      'playable': is_playable, 'url': f'/api/projects/{job_id}/artifact/{rel}',
                      'download_url': f'/api/projects/{job_id}/artifact/{orig_rel}'})
    for label, path, kind in candidates:
        if path.is_file() and path.stat().st_size:
            rel = path.relative_to(ROOT).as_posix()
            files.append({'label': label, 'path': rel, 'kind': kind,
                          'url': f'/api/projects/{job_id}/artifact/{rel}'})
    team_shape = []
    team_shape_path = CSV_DIR / f'{clip}_team_shape_summary.csv'
    if team_shape_path.is_file():
        import csv
        with team_shape_path.open(encoding='utf-8-sig', newline='') as f:
            team_shape = list(csv.DictReader(f))[:2]
    player_stats = []
    stats_path = CSV_DIR / f'{clip}_player_statistics.csv'
    if stats_path.is_file():
        import csv
        with stats_path.open(encoding='utf-8-sig', newline='') as f:
            player_stats = list(csv.DictReader(f))[:150]
    # Repair media from jobs processed by older releases automatically.
    if job.get('state') in ('done', 'failed') and any(f['kind']=='video' and not f.get('playable') for f in files) and job.get('media_state') not in ('running', 'partial'):
        _prepare_web_media_async(job_id)
        job = _read_job(job_id)
    return jsonify(project=_overview(job), summaries=report, files=files, players=player_stats,
                   team_shape=team_shape, media_state=job.get('media_state', 'idle'),
                   media_stage=job.get('media_stage', ''), media_errors=job.get('media_errors', []))



@app.post('/api/projects/<job_id>/prepare-videos')
def prepare_videos(job_id):
    job = _read_job(job_id)
    if job['state'] == 'running':
        return _safe_error('Video generation is still running.', 409)
    started = _prepare_web_media_async(job_id)
    return jsonify(ok=True, started=started, project=_overview(_read_job(job_id)))


@app.get('/api/projects/<job_id>/artifact/<path:rel>')
def project_artifact(job_id, rel):
    job = _read_job(job_id)
    path = (ROOT / rel).resolve()
    if job['clip'] not in path.name and job['clip'] not in path.parts:
        abort(403)
    return _send_allowed(path)


if __name__ == '__main__':
    # Jobs cannot continue after a server restart. Make stale records re-runnable.
    for old_path in JOB_DIR.glob('*.json'):
        try:
            old = json.loads(old_path.read_text(encoding='utf-8'))
            if old.get('state') == 'running':
                old.update(state='failed', stage='Interrupted by server restart',
                           error='The server stopped during processing. You can restart this project.')
                _write_job(old)
            elif old.get('media_state') == 'running':
                old['media_state'] = 'idle'
                _write_job(old)
        except (ValueError, KeyError):
            continue
    # A single Flask worker so that the CPU-friendly one-job guard stays consistent.
    app.run(host='127.0.0.1', port=int(os.environ.get('FOOTBALLIQ_PORT', '5000')),
            debug=False, threaded=True, use_reloader=False)
