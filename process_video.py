from utils.video_reader import VideoReader
from utils.video_writer import VideoWriter


def process_video(input_path: str, output_path: str):
    """
    Read a video frame-by-frame and write it back
    without storing the complete video in RAM.
    """

    reader = VideoReader(input_path)

    print("Video information:")
    print(f"Frames : {reader.frame_count}")
    print(f"FPS    : {reader.fps}")
    print(f"Width  : {reader.width}")
    print(f"Height : {reader.height}")

    writer = VideoWriter(
        output_path=output_path,
        fps=reader.fps,
        width=reader.width,
        height=reader.height,
    )

    processed_frames = 0

    try:

        for frame_index, frame in reader.frames():

            # No AI processing yet.
            # Simply write the frame.
            writer.write(frame)

            processed_frames += 1

            if processed_frames % 100 == 0:
                print(
                    f"Processed {processed_frames} frames..."
                )

    finally:

        reader.release()
        writer.release()

    print(
        f"\nFinished processing {processed_frames} frames."
    )


if __name__ == "__main__":

    input_video = (
        "sample_data/input_videos/08fd33_4.mp4"
    )

    output_video = (
        "outputs/videos/test_stream.mp4"
    )

    process_video(
        input_video,
        output_video,
    )