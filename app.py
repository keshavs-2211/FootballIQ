from pathlib import Path
from threading import Thread, Lock
from uuid import uuid4

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from config import UPLOAD_DIR, VIDEO_OUTPUT_DIR
from process_video import process_video


app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
JOBS = {}
JOBS_LOCK = Lock()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def update_job(job_id, **values):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)



def run_analysis(job_id, input_path, output_path):
    update_job(
        job_id,
        status="processing",
        message="Loading model and processing video",
    )

    try:
        result = process_video(input_path, output_path)

        update_job(
            job_id,
            status="completed",
            message="Analysis complete",
            statistics=result,
            video_url=f"/outputs/{Path(output_path).name}",
        )

    except Exception as exc:
        app.logger.exception(
            "Video processing failed for job %s",
            job_id,
        )
        update_job(
            job_id,
            status="failed",
            message=str(exc),
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify(error="Choose a video file first."), 400

    video = request.files["video"]

    if not video.filename:
        return jsonify(error="Choose a video file first."), 400

    if not allowed_file(video.filename):
        return jsonify(error="Unsupported format. Use MP4, AVI, MOV, or MKV."), 400

    job_id = uuid4().hex
    original_name = secure_filename(video.filename)
    input_path = Path(UPLOAD_DIR) / f"{job_id}_{original_name}"
    output_path = Path(VIDEO_OUTPUT_DIR) / f"{job_id}_annotated.mp4"

    try:
        video.save(str(input_path))
    except Exception:
        app.logger.exception("Could not save uploaded video")
        return jsonify(error="The uploaded video could not be saved."), 500

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "message": "Video uploaded. Waiting to start.",
            "filename": original_name,
        }

    worker = Thread(
        target=run_analysis,
        args=(job_id, str(input_path), str(output_path)),
        daemon=True,
    )
    worker.start()

    return jsonify(
        job_id=job_id,
        filename=original_name,
        status_url=url_for("job_status", job_id=job_id),
    ), 202


@app.route("/status/<job_id>")
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            job = dict(job)

    if job is None:
        return jsonify(error="Analysis job not found."), 404

    return jsonify(job)


@app.route("/outputs/<path:filename>")
def output_video(filename):
    from flask import send_from_directory

    return send_from_directory(VIDEO_OUTPUT_DIR, filename, conditional=True)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="Video exceeds the 500 MB upload limit."), 413


if __name__ == "__main__":
    app.run(debug=True, threaded=True)