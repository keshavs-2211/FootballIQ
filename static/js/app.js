const form = document.getElementById("upload-form");
const fileInput = document.getElementById("video-input");
const dropzone = document.getElementById("dropzone");
const fileLabel = document.getElementById("file-label");
const fileSubtitle = document.getElementById("file-subtitle");
const submitButton = document.getElementById("submit-button");

const resultsSection = document.getElementById("results-section");
const resultTitle = document.getElementById("result-title");
const jobStatus = document.getElementById("job-status");
const statusMessage = document.getElementById("status-message");
const statusDetail = document.getElementById("status-detail");
const progressPanel = document.getElementById("progress-panel");
const spinner = document.getElementById("spinner");
const errorMessage = document.getElementById("error-message");
const completedContent = document.getElementById("completed-content");
const outputVideo = document.getElementById("output-video");

let selectedFile = null;
let activeJob = false;

function formatNumber(value) {
    return Number(value || 0).toLocaleString();
}

function selectFile(file) {
    if (!file) return;

    const allowed = /\.(mp4|avi|mov|mkv)$/i.test(file.name);
    if (!allowed) {
        showError("Choose a video in MP4, AVI, MOV, or MKV format.");
        return;
    }

    if (file.size > 500 * 1024 * 1024) {
        showError("The video must be smaller than 500 MB.");
        return;
    }

    selectedFile = file;
    fileLabel.textContent = file.name;
    fileSubtitle.textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MB · Ready to upload`;
    errorMessage.hidden = true;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
}

fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragging");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragging");
    });
});

dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (!file) return;

    selectFile(file);
});

function showProcessing(message) {
    progressPanel.hidden = false;
    spinner.hidden = false;
    completedContent.hidden = true;
    errorMessage.hidden = true;
    statusMessage.textContent = message;
    statusDetail.textContent = "Processing runs on the server. Larger videos can take several minutes.";
}

function displayResults(job) {
    const stats = job.statistics || {};

    document.getElementById("frames-stat").textContent =
        formatNumber(stats.frames_processed);
    document.getElementById("detections-stat").textContent =
        formatNumber(stats.total_detections);
    document.getElementById("tracked-stat").textContent =
        formatNumber(stats.tracked_player_and_referee_detections);
    document.getElementById("ball-stat").textContent =
        formatNumber(stats.frames_with_ball_detection);

    outputVideo.src = job.video_url;
    outputVideo.load();

    progressPanel.hidden = true;
    completedContent.hidden = false;
    resultTitle.textContent = "Analysis complete";
    jobStatus.textContent = "COMPLETED";
}

async function pollStatus(statusUrl) {
    while (activeJob) {
        try {
            const response = await fetch(statusUrl);
            const job = await response.json();

            if (!response.ok) {
                throw new Error(job.error || "Could not retrieve job status.");
            }

            jobStatus.textContent = job.status.toUpperCase();
            statusMessage.textContent = job.message || "Processing video...";

            if (job.status === "completed") {
                displayResults(job);
                activeJob = false;
                submitButton.disabled = false;
                submitButton.innerHTML = "<span>Analyze another video</span><span>→</span>";
                return;
            }

            if (job.status === "failed") {
                progressPanel.hidden = true;
                showError(job.message || "Video processing failed.");
                resultTitle.textContent = "Analysis failed";
                activeJob = false;
                submitButton.disabled = false;
                return;
            }
        } catch (error) {
            progressPanel.hidden = true;
            showError(error.message || "Lost connection while checking processing status.");
            activeJob = false;
            submitButton.disabled = false;
            return;
        }

        await new Promise((resolve) => setTimeout(resolve, 2000));
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (activeJob) return;

    if (!selectedFile) {
        showError("Select a video before starting analysis.");
        return;
    }

    const formData = new FormData();
    formData.append("video", selectedFile);

    activeJob = true;
    submitButton.disabled = true;
    resultsSection.hidden = false;
    resultTitle.textContent = "Processing your match";
    jobStatus.textContent = "UPLOADING";
    showProcessing("Uploading video...");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Upload failed.");
        }

        showProcessing("Video uploaded. Starting analysis...");
        await pollStatus(result.status_url);
    } catch (error) {
        showError(error.message || "Could not upload the video.");
        progressPanel.hidden = true;
        activeJob = false;
        submitButton.disabled = false;
    }
});