# Kalman Filter Tracker

import numpy as np
from filterpy.kalman import KalmanFilter
from config import DT


class ObjectTracker:
    """
    Tracks a single object using a Kalman Filter.
    State = [x, y, vx, vy]
    """

    def __init__(self, initial_detection):
        self.id = id(self)  # unique ID for this tracker
        self.age = 0        # how many frames this tracker has existed
        self.hits = 0       # how many times it got a detection match
        self.no_detection_count = 0  # frames without a match

        # History for visualization
        self.position_history = []
        self.velocity_history = []

        # Initialize Kalman Filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, DT, 0],
            [0, 1, 0, DT],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # Measurement matrix (we only measure x, y)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Measurement noise (how much we trust the camera)
        self.kf.R = np.array([
            [10, 0],
            [0, 10]
        ])

        # Process noise (how much we trust the motion model)
        self.kf.Q = np.eye(4) * 0.1

        # Initial covariance (uncertainty at start)
        self.kf.P = np.eye(4) * 100

        # Set initial state from first detection
        cx, cy = initial_detection['center']
        self.kf.x = np.array([[cx], [cy], [0], [0]], dtype=float)

    def predict(self):
        """Predict next position using motion model."""
        self.kf.predict()
        self.age += 1

    def update(self, detection):
        """Correct prediction using actual detection."""
        cx, cy = detection['center']
        measurement = np.array([[cx], [cy]], dtype=float)
        self.kf.update(measurement)
        self.hits += 1
        self.no_detection_count = 0

        # Save history
        x, y, vx, vy = self.kf.x.flatten()
        self.position_history.append((x, y))
        self.velocity_history.append((vx, vy))

    def get_state(self):
        """Return current estimated state."""
        x, y, vx, vy = self.kf.x.flatten()
        return {
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'speed': np.sqrt(vx**2 + vy**2)
        }

    def get_position(self):
        """Return estimated (x, y) position."""
        x, y = self.kf.x.flatten()[:2]
        return (int(x), int(y))


class MultiObjectTracker:
    """
    Manages multiple ObjectTrackers.
    Matches detections to existing trackers.
    """

    def __init__(self):
        self.trackers = []
        self.max_no_detection = 5   # remove tracker after 5 missed frames
        self.min_hits = 2           # confirm tracker after 2 hits

    def update(self, detections):
        """
        Main update function.
        - Predicts all trackers forward
        - Matches detections to trackers
        - Creates new trackers for unmatched detections
        - Removes dead trackers
        """

        # Step 1: Predict all trackers
        for tracker in self.trackers:
            tracker.predict()

        # Step 2: Match detections to trackers
        matched, unmatched_dets = self._match_detections(detections)

        # Step 3: Update matched trackers
        for tracker_idx, det_idx in matched:
            self.trackers[tracker_idx].update(detections[det_idx])

        # Step 4: Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            new_tracker = ObjectTracker(detections[det_idx])
            self.trackers.append(new_tracker)

        # Step 5: Remove dead trackers
        self.trackers = [
            t for t in self.trackers
            if t.no_detection_count <= self.max_no_detection
        ]

        # Mark unmatched trackers
        matched_tracker_ids = [t for t, d in matched]
        for i, tracker in enumerate(self.trackers):
            if i not in matched_tracker_ids:
                tracker.no_detection_count += 1

        # Return only confirmed trackers
        return [t for t in self.trackers if t.hits >= self.min_hits]

    def _match_detections(self, detections):
        """
        Simple distance-based matching.
        Match each detection to the nearest tracker.
        """
        matched = []
        unmatched_dets = list(range(len(detections)))

        if not self.trackers or not detections:
            return matched, unmatched_dets

        MAX_DISTANCE = 100  # pixels — max distance to consider a match

        for det_idx, det in enumerate(detections):
            best_tracker = None
            best_dist = MAX_DISTANCE

            for t_idx, tracker in enumerate(self.trackers):
                tx, ty = tracker.get_position()
                dx = det['center'][0] - tx
                dy = det['center'][1] - ty
                dist = np.sqrt(dx**2 + dy**2)

                if dist < best_dist:
                    best_dist = dist
                    best_tracker = t_idx

            if best_tracker is not None:
                matched.append((best_tracker, det_idx))
                unmatched_dets.remove(det_idx)

        return matched, unmatched_dets