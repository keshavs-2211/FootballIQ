import json
from pathlib import Path
import cv2
import numpy as np

# Pitch dimensions in metres; X left-to-right, Y top-to-bottom.
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# Only choose landmarks visible in the starting frame.
LANDMARKS = {
    '1': ('Top-left pitch corner', (0.0, 0.0)),
    '2': ('Top-right pitch corner', (105.0, 0.0)),
    '3': ('Bottom-left pitch corner', (0.0, 68.0)),
    '4': ('Bottom-right pitch corner', (105.0, 68.0)),
    '5': ('Top touchline / halfway line', (52.5, 0.0)),
    '6': ('Bottom touchline / halfway line', (52.5, 68.0)),
    '7': ('Centre spot', (52.5, 34.0)),
    '8': ('Left penalty area / top edge', (16.5, (68.0 - 40.32) / 2)),
    '9': ('Left penalty area / bottom edge', (16.5, (68.0 + 40.32) / 2)),
    '10': ('Right penalty area / top edge', (105.0 - 16.5, (68.0 - 40.32) / 2)),
    '11': ('Right penalty area / bottom edge', (105.0 - 16.5, (68.0 + 40.32) / 2)),
    '12': ('Left penalty spot', (11.0, 34.0)),
    '13': ('Right penalty spot', (94.0, 34.0)),
    '14': ('Left goal-area top-right corner', (5.5, (68.0 - 18.32) / 2)),
    '15': ('Left goal-area bottom-right corner', (5.5, (68.0 + 18.32) / 2)),
    '16': ('Right goal-area top-left corner', (99.5, (68.0 - 18.32) / 2)),
    '17': ('Right goal-area bottom-left corner', (99.5, (68.0 + 18.32) / 2)),
}


def calibrate(video_path, output_json, preview_width=1100):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError('Could not read the first video frame.')

    h, w = frame.shape[:2]
    scale = min(1.0, preview_width / w)
    preview = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame
    selected_image, selected_pitch = [], []
    print('\nPITCH CALIBRATION — frame 0')
    print('Click at least FOUR real, visible pitch landmarks.')
    print('IMPORTANT: Landmarks use the tactical pitch orientation: LEFT goal = X=0, RIGHT goal = X=105.')
    print('If the camera faces the opposite direction, use the corresponding reverse-side landmarks.')
    print('Avoid selecting imaginary or hidden intersections. More spread-out landmarks improve stability.')
    print('Press ESC after selecting 4+ points to finish. R to remove the last point.')
    for k, (label, _) in LANDMARKS.items():
        print(f'{k:>2} : {label}')

    window = 'FootballIQ Calibration - click requested landmark'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    clicked = []

    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x / scale, y / scale))

    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            prompt = input('\nLandmark number (or done / undo): ').strip().lower()
            if prompt in ('done', 'd'):
                break
            if prompt in ('undo', 'u'):
                if selected_image:
                    selected_image.pop()
                    selected_pitch.pop()
                continue
            if prompt not in LANDMARKS:
                print('Choose a number from the list.')
                continue
            name, xy = LANDMARKS[prompt]
            print(f'Click exactly at: {name} (press ESC to cancel this point).')
            clicked.clear()
            while not clicked:
                canvas = preview.copy()
                for idx, (ix, iy) in enumerate(selected_image):
                    pt = (int(ix * scale), int(iy * scale))
                    cv2.circle(canvas, pt, 5, (0, 255, 255), -1)
                    cv2.putText(canvas, str(idx + 1), (pt[0]+6, pt[1]-6),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
                cv2.putText(canvas, f'Click: {name}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
                cv2.imshow(window, canvas)
                key = cv2.waitKey(25) & 0xFF
                if key == 27:
                    break
            if clicked:
                selected_image.append(clicked[0])
                selected_pitch.append(xy)
                print(f'Selected {len(selected_image)}: {name}')
    finally:
        cv2.destroyAllWindows()

    if len(selected_image) < 4:
        raise ValueError('At least four corresponding landmarks are required.')
    src = np.array(selected_image, dtype=np.float32)
    dst = np.array(selected_pitch, dtype=np.float32)
    if len(src) == 4:
        H = cv2.getPerspectiveTransform(src, dst)
        inliers = np.ones(4, dtype=np.uint8)
    else:
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or not np.isfinite(H).all():
        raise ValueError('Calibration failed; choose well-separated, non-collinear landmarks.')
    projected = cv2.perspectiveTransform(src.reshape(-1,1,2), H).reshape(-1,2)
    errors = np.linalg.norm(projected - dst, axis=1)
    print(f'Calibration point median error: {np.median(errors):.2f} metres')
    if np.median(errors) > 3:
        print('WARNING: Large calibration error; check your selected landmark locations.')
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({
        'video_width': w, 'video_height': h,
        'pitch_length': PITCH_LENGTH, 'pitch_width': PITCH_WIDTH,
        'homography': H.tolist(), 'image_points': src.tolist(),
        'pitch_points': dst.tolist(),
    }, indent=2), encoding='utf-8')
    print('Saved calibration:', output_json)
    return output_json
