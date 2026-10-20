import numpy as np
import supervision as sv
from trackers import ByteTrackTracker as SupervisionByteTrack


class ByteTrackTracker:
    """
    ByteTrack-based multi-object tracker.

    Responsible only for assigning persistent track IDs
    to YOLO detections.

    It does not perform:
    - YOLO detection
    - team classification
    - ball tracking
    - possession
    - analytics
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
        Update ByteTrack using YOLO detections.

        Input:
            detections: list of YOLO detection dictionaries

        Output:
            list of dictionaries containing:
            bbox, class_id, class_name, confidence, track_id
        """

        if not detections:
            self.tracker.update(sv.Detections.empty())
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

        # Keep the original class name attached to each detection.
        class_names = np.array(
            [d["class_name"] for d in detections],
            dtype=object,
        )

        supervision_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        # Store class names as custom detection data.
        supervision_detections.data["class_name"] = class_names

        tracked = self.tracker.update(supervision_detections)

        tracked_detections = []

        for i in range(len(tracked.xyxy)):
            track_id = tracked.tracker_id[i]

            if track_id is None or int(track_id)<0:
                continue

            class_id_value = int(tracked.class_id[i])

            if "class_name" in tracked.data:
                class_name = tracked.data["class_name"][i]
            else:
                class_name = str(class_id_value)

            tracked_detection = {
                "bbox": tracked.xyxy[i].tolist(),
                "class_id": class_id_value,
                "class_name": str(class_name),
                "confidence": float(tracked.confidence[i]),
                "track_id": int(track_id),
            }

            tracked_detections.append(tracked_detection)

        return tracked_detections