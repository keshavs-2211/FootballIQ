import cv2
import numpy as np


class CameraMotionCompensator:
    """Track background features; map each CURRENT frame back to frame 0.

    Approximation is appropriate for mild pans/zooms, not cuts or large changes.
    On uncertain frames, holds last transform instead of inventing movement.
    """

    def __init__(self, first_frame, max_width=800):
        h, w = first_frame.shape[:2]
        self.scale = min(1.0, max_width / w)
        self.prev_gray = self._gray(first_frame)
        self.to_first = np.eye(3, dtype=np.float64)
        self.prev_points = self._features(self.prev_gray, self._mask(first_frame))
        self.last_inlier_count = 0

    def _gray(self, frame):
        if self.scale != 1:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _mask(self, frame):
        if self.scale != 1:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Grass pixels and nearby fixed pitch markings; suppress stands.
        grass = cv2.inRange(hsv, (30, 35, 30), (95, 255, 255))
        grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8))
        return cv2.dilate(grass, np.ones((11,11), np.uint8))

    @staticmethod
    def _features(gray, mask):
        return cv2.goodFeaturesToTrack(gray, maxCorners=350, qualityLevel=.01,
                                       minDistance=9, mask=mask, blockSize=5)

    def update(self, frame):
        gray = self._gray(frame)
        mask = self._mask(frame)
        if self.prev_points is None or len(self.prev_points) < 12:
            self.prev_points = self._features(self.prev_gray, mask)
        if self.prev_points is None or len(self.prev_points) < 8:
            self.prev_gray = gray
            self.prev_points = self._features(gray, mask)
            return self.to_first.copy()

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        if next_points is None or status is None:
            good_previous = np.empty((0, 2), np.float32)
            good_current = good_previous
        else:
            good = status.reshape(-1) == 1
            good_previous = self.prev_points.reshape(-1, 2)[good]
            good_current = next_points.reshape(-1, 2)[good]

        self.last_inlier_count = 0
        if len(good_previous) >= 10:
            # Estimate CURRENT -> PREVIOUS, ignoring motion outliers (players).
            affine, inliers = cv2.estimateAffinePartial2D(
                good_current, good_previous, method=cv2.RANSAC,
                ransacReprojThreshold=2.5, maxIters=1000)
            if affine is not None and inliers is not None:
                num_inliers = int(inliers.sum())
                fraction = num_inliers / len(good_previous)
                motion = np.eye(3, dtype=np.float64)
                motion[:2, :] = affine
                zoom = np.linalg.norm(affine[:, 0])
                dx, dy = affine[:, 2]
                # Reject uncertain and implausibly large frame-to-frame motion.
                if (num_inliers >= 10 and fraction >= .45
                        and .96 <= zoom <= 1.04
                        and abs(dx) <= 35 and abs(dy) <= 35):
                    self.to_first = self.to_first @ motion
                    self.last_inlier_count = num_inliers

        self.prev_gray = gray
        # Refresh regularly to prevent drift from tracking disappearing features.
        self.prev_points = self._features(gray, mask)

        # Transform is in reduced-image pixels; convert to original dimensions.
        S = np.diag([self.scale, self.scale, 1.0])
        return np.linalg.inv(S) @ self.to_first @ S
