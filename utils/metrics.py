# Evaluation Metrics

import numpy as np


class MetricsTracker:
    """
    Tracks performance metrics over the entire run.
    - RMSE for position and velocity
    - Precision and Recall for detection
    """

    def __init__(self):
        self.position_errors   = []
        self.velocity_errors   = []
        self.true_positives    = 0
        self.false_positives   = 0
        self.false_negatives   = 0
        self.frame_count       = 0
        self.total_detections  = 0
        self.total_tracks      = 0

    def update(self, detections, active_trackers, fusion_results):
        self.frame_count      += 1
        self.total_detections += len(detections)
        self.total_tracks     += len(active_trackers)

        # Simple TP/FP/FN based on detection vs track match
        det_count   = len(detections)
        track_count = len(active_trackers)

        matched = min(det_count, track_count)
        self.true_positives  += matched
        self.false_positives += max(0, track_count - det_count)
        self.false_negatives += max(0, det_count - track_count)

        # Velocity error from fusion results
        for res in fusion_results:
            if res['fusion_active'] and res['radar_velocity'] is not None:
                cam_spd   = res['fused_speed']
                radar_vel = abs(res['radar_velocity'])
                error     = abs(cam_spd - radar_vel)
                self.velocity_errors.append(error)

    def get_rmse_velocity(self):
        if not self.velocity_errors:
            return 0.0
        return float(np.sqrt(np.mean(np.array(self.velocity_errors) ** 2)))

    def get_precision(self):
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    def get_recall(self):
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    def get_f1(self):
        p = self.get_precision()
        r = self.get_recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def summary(self):
        print("\n" + "="*50)
        print("        EVALUATION SUMMARY")
        print("="*50)
        print(f"  Total Frames     : {self.frame_count}")
        print(f"  Total Detections : {self.total_detections}")
        print(f"  Total Tracks     : {self.total_tracks}")
        print(f"  RMSE Velocity    : {self.get_rmse_velocity():.4f}")
        print(f"  Precision        : {self.get_precision():.4f}")
        print(f"  Recall           : {self.get_recall():.4f}")
        print(f"  F1 Score         : {self.get_f1():.4f}")
        print("="*50)