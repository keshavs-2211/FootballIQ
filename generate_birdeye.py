"""FootballIQ: generate a 2D tactical simulation from existing coordinate CSV.

Run from the FootballIQ project root:
    python generate_birdeye.py

Requires only cv2, numpy, and the standard library.
"""
import csv
from pathlib import Path

import cv2
import numpy as np

from camera.camera_movement import CameraMotionCompensator
from utils.manual_pitch_calibration import calibrate
from utils.pitch_projector import PitchProjector
from visualization.birdeye_renderer import BirdEyeRenderer


# EDIT THESE THREE PATHS FOR YOUR FILES
INPUT_VIDEO = 'sample_data/input_videos/whitevsred.mp4'
TRACKING_CSV = 'outputs/csv/whitevsred_reference_debug_coordinates.csv'
# Optional: replace with an existing annotated video of identical frame alignment.
SIDE_BY_SIDE_SOURCE = INPUT_VIDEO
OUTPUT_DIRECTORY = 'outputs/videos'
CALIBRATION_FILE = 'references/whitevsred_pitch_calibration.json'
PROJECTED_CSV = 'outputs/csv/whitevsred_pitch_coordinates.csv'

# Set to False to start fresh calibration; existing calibration is reused otherwise.
REUSE_CALIBRATION = False
CAMERA_COMPENSATION = True
# Small smoothing factor: 1 = no smoothing; lower = more smoothing, more lag.
SMOOTHING_ALPHA = 0.60


def iter_csv_records(file_obj):
    """CSV must be sorted ascending by frame; no all-rows load into memory."""
    reader = csv.DictReader(file_obj)
    required = {'frame', 'track_id', 'foot_x', 'foot_y', 'team', 'class_name'}
    if not required.issubset(set(reader.fieldnames or [])):
        raise ValueError(f'CSV missing columns: {sorted(required - set(reader.fieldnames or []))}')
    for row in reader:
        yield int(row['frame']), row


def write_video(path, fps, dims):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, dims)
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video output: {path}')
    return writer


def generate():
    video_path = Path(INPUT_VIDEO)
    csv_path = Path(TRACKING_CSV)
    if not video_path.is_file():
        raise FileNotFoundError(f'Video not found: {video_path}')
    if not csv_path.is_file():
        raise FileNotFoundError(
            f'Tracking CSV not found: {csv_path}\n'
            'Check CSV_OUTPUT_DIR in your config.py and the filename printed by process_video.py.\n'
            'Set EXPORT_UNKNOWN_PLAYERS=True and rerun processing if you also want grey Unknown markers.')

    cal = Path(CALIBRATION_FILE)
    if not (REUSE_CALIBRATION and cal.exists()):
        calibrate(video_path, cal)
    projector = PitchProjector(cal)
    pitch = BirdEyeRenderer()

    out = Path(OUTPUT_DIRECTORY)
    out.mkdir(parents=True, exist_ok=True)
    Path(PROJECTED_CSV).parent.mkdir(parents=True, exist_ok=True)
    tactical_video = out / f'{video_path.stem}_birdeye.mp4'
    side_video = out / f'{video_path.stem}_side_by_side.mp4'

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError('Cannot open source video')
    side_source = cv2.VideoCapture(str(SIDE_BY_SIDE_SOURCE))
    if not side_source.isOpened():
        video.release()
        raise RuntimeError('Cannot open source for side-by-side video')

    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    outsize = (pitch.width, pitch.height)
    tact_writer = write_video(tactical_video, fps, outsize)
    side_writer = write_video(side_video, fps, (pitch.width * 2, pitch.height))

    # Tracking CSV may contain repeated observations for each frame, but is streamed.
    tracks = {}
    smooth_positions = {}
    last_seen = {}
    frame_index = 0
    try:
        with csv_path.open('r', newline='', encoding='utf-8') as f, \
                Path(PROJECTED_CSV).open('w', newline='', encoding='utf-8') as outf:
            csv_records = iter_csv_records(f)
            pending = next(csv_records, None)
            fieldnames = ['frame', 'track_id', 'class_name', 'team',
                          'pitch_x_m', 'pitch_y_m', 'timestamp_seconds']
            csv_writer = csv.DictWriter(outf, fieldnames=fieldnames)
            csv_writer.writeheader()
            compensator = None

            while True:
                ok, frame = video.read()
                ok_side, frame_side = side_source.read()
                if not ok or not ok_side:
                    break

                if compensator is None:
                    compensator = CameraMotionCompensator(frame)
                    camera_to_first = np.eye(3)
                else:
                    camera_to_first = (compensator.update(frame)
                                       if CAMERA_COMPENSATION else np.eye(3))

                # Every frame begins with its own tracked observations.
                observations = []
                while pending is not None and pending[0] < frame_index:
                    pending = next(csv_records, None)
                while pending is not None and pending[0] == frame_index:
                    observations.append(pending[1])
                    pending = next(csv_records, None)

                markers = []
                for row in observations:
                    try:
                        track_id = int(row['track_id'])
                        foot_x = float(row['foot_x'])
                        foot_y = float(row['foot_y'])
                    except (ValueError, TypeError):
                        continue
                    point = projector.project(foot_x, foot_y, camera_to_first)
                    if point is None:
                        continue

                    # Do not interpolate missing detections; this avoids phantom players.
                    prev = smooth_positions.get(track_id)
                    if prev is not None and frame_index - last_seen.get(track_id, -100) == 1:
                        x = SMOOTHING_ALPHA * point[0] + (1-SMOOTHING_ALPHA)*prev[0]
                        y = SMOOTHING_ALPHA * point[1] + (1-SMOOTHING_ALPHA)*prev[1]
                    else:
                        x, y = point
                    smooth_positions[track_id] = (x, y)
                    last_seen[track_id] = frame_index
                    # Set team strings from config, if your project uses custom names.
                    team = row.get('team', 'Unknown')
                    markers.append({'track_id': track_id, 'class_name': row['class_name'],
                                    'team': team, 'pitch_x': x, 'pitch_y': y})
                    csv_writer.writerow({'frame':frame_index, 'track_id':track_id,
                                         'class_name':row['class_name'], 'team':team,
                                         'pitch_x_m':round(x,3), 'pitch_y_m':round(y,3),
                                         'timestamp_seconds':round(frame_index/fps,3)})

                # Clean long-dead IDs to bound memory.
                if frame_index % 250 == 0:
                    for tid in list(last_seen):
                        if frame_index - last_seen[tid] > 125:
                            last_seen.pop(tid, None)
                            smooth_positions.pop(tid, None)

                tactical = pitch.render(markers, frame_index=frame_index, fps=fps)
                tact_writer.write(tactical)
                original = cv2.resize(frame_side, outsize, interpolation=cv2.INTER_AREA)
                side_writer.write(np.hstack((original, tactical)))

                frame_index += 1
                if frame_index % 50 == 0:
                    print(f'Processed {frame_index}/{total_frames} frames | '
                          f'{len(markers)} projected players | '
                          f'camera inliers: {compensator.last_inlier_count}')
    finally:
        tact_writer.release()
        side_writer.release()
        video.release()
        side_source.release()

    print('\nFinished!')
    print('2D simulation:', tactical_video)
    print('Side-by-side:', side_video)
    print('Projected coordinates:', PROJECTED_CSV)
    print('Note: tracking CSV does not contain ball coordinates, so the ball is not simulated yet.')


if __name__ == '__main__':
    generate()
