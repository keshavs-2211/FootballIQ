from utils.video_reader import VideoReader
from utils.video_writer import VideoWriter
from detector.yolo_detector import YOLODetector


def process_video(input_path: str, output_path: str):
    """
    Process a video frame-by-frame using YOLO.

    The video is never loaded completely into RAM.
    """

    # --------------------------------------------------------
    # Initialize video reader
    # --------------------------------------------------------

    reader = VideoReader(input_path)

    print("Video information:")
    print(f"Frames : {reader.frame_count}")
    print(f"FPS    : {reader.fps}")
    print(f"Width  : {reader.width}")
    print(f"Height : {reader.height}")

    # --------------------------------------------------------
    # Initialize YOLO
    # --------------------------------------------------------

    detector = YOLODetector()

    # --------------------------------------------------------
    # Initialize video writer
    # --------------------------------------------------------

    writer = VideoWriter(
        output_path=output_path,
        fps=reader.fps,
        width=reader.width,
        height=reader.height,
    )

    processed_frames = 0
    total_detections = 0

    try:

        # ----------------------------------------------------
        # Stream frames
        # ----------------------------------------------------

        for frame_index, frame in reader.frames():

            # -----------------------------------------------
            # YOLO detection
            # -----------------------------------------------

            detections = detector.detect(frame)

            total_detections += len(detections)

            # -----------------------------------------------
            # Write original frame for now
            # -----------------------------------------------

            writer.write(frame)

            processed_frames += 1

            if processed_frames % 50 == 0:

                print(
                    f"Processed "
                    f"{processed_frames}/"
                    f"{reader.frame_count} frames | "
                    f"Detections so far: "
                    f"{total_detections}"
                )

    finally:

        reader.release()
        writer.release()

    print("\nProcessing complete.")

    print(
        f"Total frames processed: "
        f"{processed_frames}"
    )

    print(
        f"Total detections: "
        f"{total_detections}"
    )


if __name__ == "__main__":

    input_video = (
        "sample_data/input_videos/08fd33_4.mp4"
    )

    output_video = (
        "outputs/videos/yolo_test.mp4"
    )

    process_video(
        input_video,
        output_video,
    )