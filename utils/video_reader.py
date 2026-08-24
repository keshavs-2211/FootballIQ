import cv2
from typing import Generator, Tuple


class VideoReader:
    """
    Streams video frames one at a time.

    Frames are NOT stored in memory.
    This keeps RAM usage approximately constant
    regardless of video length.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

    def frames(self) -> Generator[Tuple[int, any], None, None]:
        """
        Yield one frame at a time.

        Returns:
            frame_index: Index of the current frame
            frame: OpenCV image (numpy array)
        """

        frame_index = 0

        while True:

            success, frame = self.cap.read()

            if not success:
                break

            yield frame_index, frame

            frame_index += 1

    def release(self):
        """Release the video capture resource."""
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()