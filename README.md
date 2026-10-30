# FootballIQ — rebuilt unified project

One maintained web app, one offline command-line entry point, shared detection/tracking/analytics stages.
**Python 3.11 recommended** (Windows 11 or Linux). The supplied source ZIP intentionally does not contain `models/best.pt` or large input videos, as requested. Restore those from your own existing project. Never overwrite your current copy before local testing.

## Folders

| Folder | Purpose |
|---|---|
| `src/` | Active Python processing stages, settings, and reusable modules |
| `src/detector/`, `src/tracker/`, `src/camera/` | Model inference, player tracking and camera movement |
| `src/analytics/`, `src/visualization/`, `src/utils/` | Metrics, pitch drawing, video support |
| `web/templates`, `web/static` | Only maintained website; no older Flask app or legacy templates |
| `models/` | Put trained `best.pt` here |
| `sample_data/input_videos/` | Put your `<clip>.mp4` here |
| `references/<clip>/` | Three jersey images per team and three ball crops |
| `references/<clip>_pitch_calibration.json` | Saved pitch-to-pixel homography |
| `outputs/` | Generated CSV, JSON, videos, heatmaps and charts |
| `web_data/jobs/` | Local website job logs and state |
| `tests/` | Standalone smoke tests that do not require YOLO weights |

## Install and launch on Windows

From the **root of this rebuilt project** in the VS Code terminal (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py check --clip bluevsred
python run.py web
```

Open `http://127.0.0.1:5000`. If you use Git Bash to activate: `source .venv/Scripts/activate`.

If Python 3.11 is not installed, first check your environment's compatibility with PyTorch, Ultralytics, `trackers`, `pyarrow`, and supervision before choosing a different Python version.

## Offline commands (no Flask required)

```powershell
python run.py annotate --clip bluevsred
python run.py annotated-ball --clip bluevsred
python run.py full --clip bluevsred --with-ball
python run.py refine --clip bluevsred
python run.py analytics --clip bluevsred
python run.py ball --clip bluevsred
python run.py ball-annotate --clip bluevsred
python run.py birdseye --clip bluevsred
python run.py possession --clip bluevsred
python run.py possession-video --clip bluevsred
```

- `annotate`: reads saved jerseys and produces an annotated video + player tracking coordinates without opening website or interactive crop windows.
- `annotated-ball`: runs detection → refinement → reference-aware ball tracking → ball overlay to create `<clip>_ball_annotated.mp4`. It requires a saved calibration and all three ball references.
- `full --with-ball`: also makes analytics, tactical bird's-eye / side-by-side videos, and possession. Experimental ball-tracking measurements are explicitly marked.
- `full` without `--with-ball`: creates annotated player, refined tracking, analytics and tactical outputs, without requiring the ball pipeline to run. The website still asks you to save all nine reference crops during setup.
- To set up a new clip's pitch landmarks without coding, open the website's Match Studio and calibrate at the desired frame. Saved calibration persists across frame changes; unsaved points reset.

## Outputs

- `outputs/videos/<clip>_improved.mp4` — YOLO/player tracking annotated video.
- `outputs/videos/<clip>_ball_annotated.mp4` — *measured* ball positions overlaid on player annotations; only after ball tracking, not merely generic YOLO ball predictions.
- `outputs/videos/<clip>_professional_birdeye.mp4` and `_professional_side_by_side.mp4` — tactical videos.
- `outputs/videos/*_web.mp4` — H.264 copies created by the Flask website for Chrome/Firefox playback. Originals remain untouched.
- `outputs/csv/<clip>_professional_ball.csv` — pitch and observed pixel positions. Predicted frame rows have empty pixel columns and are **not** drawn as measured detections.
- `outputs/csv/<clip>_ball_candidate_audit.csv` and `<clip>_ball_reference_report.json` — debug evidence for reference-matching quality.

## Known limitations

The full pipeline cannot be validated without **your actual trained model and input videos**. Tiny balls, occlusions, player re-identification, strong zooms/camera cuts and ambiguous kit colours still present experimental accuracy limitations. Calibration alignment assumes reasonably smooth camera movement; after a hard broadcast cut, recalibrate/resegment for trustworthy positions. This local Flask application is not designed for deployment on a public network.

The `references/bluevsred/ball*.png` included here were carried over from your existing `fi_cfe7fabfdb82_bluevsred` references to give the offline Blue vs Red clip three candidate ball crops. Confirm visually that they belong to the same video and accurately show the ball before relying on their scores.

## Checks

```powershell
python -m compileall -q src run.py
python -m unittest discover -s tests -v
```

Start with `python run.py check --clip bluevsred`, then run one offline annotated job before replacing your original project.

### Offline setup with no website

`python run.py references --clip bluevsred` lets you capture 3 Team A, 3 Team B, and 3 ball crops from chosen frames using OpenCV ROI windows. `python run.py calibrate --clip bluevsred` opens a first-frame pitch landmark picker. Either route writes the same reference/calibration paths consumed by the web and CLI pipelines.

Built By Pranjal,Keshav & Milind 
5Th Sem Mini Project
