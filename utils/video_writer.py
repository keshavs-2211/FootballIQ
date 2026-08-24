import cv2


class VideoWriter:
    """
    Writes video frames directly to disk.

    Frames are written immediately instead of
    being accumulated in RAM.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        width: int,
        height: int,
    ):
        self.output_path = output_path

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height),
        )

        if not self.writer.isOpened():
            raise ValueError(
                f"Could not create output video: {output_path}"
            )

    def write(self, frame):
        """
        Write one frame directly to the output video.
        """
        self.writer.write(frame)

    def release(self):
        """
        Release the video writer.
        """
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()