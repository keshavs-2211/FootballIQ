# Quick start

1. Place `models/best.pt` and `sample_data/input_videos/bluevsred.mp4` in the extracted folder.
2. Install Python 3.11, create/activate a virtual environment, install `requirements.txt`.
3. Run `python run.py check --clip bluevsred`.
4. Run `python run.py web` to upload, crop 9 references, calibrate at a chosen frame and process a match.
5. Or run `python run.py references --clip bluevsred` and `python run.py calibrate --clip bluevsred` for offline setup; then `python run.py full --clip bluevsred --with-ball`.

`python run.py annotate --clip bluevsred` creates player annotations without Flask, without rerunning analytics, and without using ball references. `python run.py annotated-ball --clip bluevsred` additionally verifies balls using saved crops and overlays those observations.

See project `README.md` for all commands. **Run in a fresh folder and retain your old folder as backup until a full real-video run passes.**
