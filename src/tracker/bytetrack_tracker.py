
import numpy as np
import supervision as sv

from trackers import ByteTrackTracker as SupervisionByteTrack


class ByteTrackTracker:
    """
    Standard ByteTrack wrapper for FootballIQ.

    Uses ByteTrack's built-in Kalman filtering
    and association for player tracking.

    No additional predicted-position recovery
    or manual ID reassignment.
    """

    def __init__(
        self,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.3,
        frame_rate=25,
    ):

        self.tracker = SupervisionByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_iou_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections):
        """
        Update ByteTrack and return tracked objects.
        """

        if not detections:
            self.tracker.update(
                sv.Detections.empty()
            )
            return []

        xyxy = np.array(
            [d["bbox"] for d in detections],
            dtype=np.float32,
        )

        confidence = np.array(
            [d["confidence"] for d in detections],
            dtype=np.float32,
        )

        class_id = np.array(
            [d["class_id"] for d in detections],
            dtype=np.int32,
        )

        class_names = np.array(
            [d["class_name"] for d in detections],
            dtype=object,
        )

        supervision_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        supervision_detections.data[
            "class_name"
        ] = class_names

        tracked = self.tracker.update(
            supervision_detections
        )

        tracked_objects = []

        for i in range(len(tracked.xyxy)):

            track_id = tracked.tracker_id[i]

            if (
                track_id is None
                or int(track_id) < 0
            ):
                continue

            class_id_value = int(
                tracked.class_id[i]
            )

            if "class_name" in tracked.data:
                class_name = str(
                    tracked.data["class_name"][i]
                )
            else:
                class_name = str(
                    class_id_value
                )

            tracked_objects.append({
                "bbox": tracked.xyxy[i].tolist(),
                "class_id": class_id_value,
                "class_name": class_name,
                "confidence": float(
                    tracked.confidence[i]
                ),
                "track_id": int(track_id),
            })

        return tracked_objects
