# DeepSORT Tracker

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np


class DeepSORTTracker:
    """
    Wraps DeepSORT for drop-in replacement of our
    simple Kalman tracker. Gives each object a
    persistent ID even after occlusion.
    """

    def __init__(self):
        self.tracker = DeepSort(
            max_age=5,           # frames to keep lost track
            n_init=2,            # frames before track confirmed
            max_cosine_distance=0.3,
            nn_budget=None,
        )
        self.active_tracks = []
        print("[DeepSORT] Tracker initialized!")

    def update(self, detections, frame):
        """
        detections: list of dicts from detector.py
        frame: current BGR frame (needed for appearance features)
        Returns list of active track objects.
        """
        # Convert to DeepSORT input format
        # Each item: ([x1,y1,w,h], confidence, class_name)
        ds_input = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            ds_input.append(
                ([x1, y1, w, h],
                 det['confidence'],
                 det['class_name'])
            )

        # Run DeepSORT update
        tracks = self.tracker.update_tracks(ds_input, frame=frame)

        # Build our standard track format
        self.active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            self.active_tracks.append({
                'id':       track.track_id,
                'bbox':     [x1, y1, x2, y2],
                'center':   (cx, cy),
                'class':    track.get_det_class(),
                'confidence': track.get_det_conf() or 0.0
            })

        return self.active_tracks

    def get_track_count(self):
        return len(self.active_tracks)