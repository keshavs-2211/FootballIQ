"""Standalone capture of all 9 FootballIQ team and ball reference crops.

No website needed. Requires an OpenCV desktop window (not Google Colab).
"""
from __future__ import annotations
from pathlib import Path
import cv2
import offline_settings as S

ROLES = (('team_a', 'Team A shirt'), ('team_b', 'Team B shirt'),
         ('ball', 'Actual ball'))


def capture(video, output, *, roles=ROLES, samples=3):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open: {video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        for stem, label in roles:
            for idx in range(1, samples + 1):
                filename = f'{stem}{"" if idx == 1 else "_"+str(idx)}.png'
                dest = output / filename
                if dest.is_file():
                    answer = input(f'{dest} already exists. Replace it? [y/N]: ').strip().lower()
                    if answer not in ('y', 'yes'):
                        continue
                while True:
                    time_raw = input(f'{label} reference {idx}/{samples} - frame time (seconds) [0]: ').strip()
                    try:
                        seconds = max(0.0, float(time_raw or 0.0))
                    except ValueError:
                        print('Enter a valid timestamp.')
                        continue
                    frame_num = min(count-1, int(seconds*fps)) if count else int(seconds*fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0,frame_num))
                    ok, frame = cap.read()
                    if not ok:
                        print('Cannot read that frame. Try another timestamp.')
                        continue
                    height, width = frame.shape[:2]
                    scale = min(1.0, 1200.0/max(1,width))
                    preview = (cv2.resize(frame, (round(width*scale), round(height*scale)))
                               if scale < 1 else frame.copy())
                    name = f'{label} / sample {idx} — drag ROI and press ENTER'
                    rect = cv2.selectROI(name, preview, showCrosshair=True, fromCenter=False)
                    cv2.destroyWindow(name)
                    x,y,w,h = (int(i) for i in rect)
                    if w < 4 or h < 4:
                        print('Crop too small. Try again.')
                        continue
                    x1,y1=round(x/scale),round(y/scale)
                    x2,y2=round((x+w)/scale),round((y+h)/scale)
                    crop = frame[max(0,y1):min(height,y2), max(0,x1):min(width,x2)]
                    if crop.size==0 or min(crop.shape[:2])<4:
                        print('Invalid crop; retry.')
                        continue
                    if not cv2.imwrite(str(dest), crop):
                        raise RuntimeError(f'Could not write image: {dest}')
                    print('Saved:', dest)
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f'Finished reference capture at {output}')


def run():
    capture(S.VIDEO, S.REFERENCE_DIR)


if __name__=='__main__':
    run()
