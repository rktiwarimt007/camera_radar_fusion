# KITTI + DeepSORT + Full Evaluation

import cv2
import numpy as np
import os
from modules.detector          import CameraDetector, draw_detections
from modules.deepsort_tracker  import DeepSORTTracker
from modules.radar             import RadarSimulator, draw_radar
from modules.fusion            import SensorFusion, draw_fusion
from utils.evaluator           import Evaluator
from config import (VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT,
                    USE_KITTI, KITTI_SEQUENCE,
                    KITTI_IMAGE_DIR, KITTI_LABEL_DIR)


def get_color(track_id):
    np.random.seed(int(track_id) % 100)
    return tuple(np.random.randint(50, 255, 3).tolist())

def draw_deepsort_tracks(frame, tracks):
    for t in tracks:
        color = get_color(t['id'])
        x1, y1, x2, y2 = t['bbox']
        cx, cy = t['center']

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 6, color, -1)
        cv2.putText(frame, f"ID:{t['id']} {t['class']}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def draw_legend(frame):
    items = [
        ("GREEN  = YOLO Detection",  (0, 255, 0)),
        ("COLOR  = DeepSORT Track",  (200, 200, 200)),
        ("BLUE   = Radar",           (255, 100, 0)),
        ("YELLOW = Fused Estimate",  (255, 255, 0)),
    ]
    for i, (text, color) in enumerate(items):
        cv2.putText(frame, text, (10, 60 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return frame


def convert_deepsort_to_tracker_format(ds_tracks):
    """Convert DeepSORT output to format radar/fusion expects."""
    class FakeTracker:
        def __init__(self, t):
            self.id  = t['id']
            self.hits = 3
            self._cx, self._cy = t['center']
            self._state = {
                'x': float(self._cx),
                'y': float(self._cy),
                'vx': 0.0, 'vy': 0.0, 'speed': 0.0
            }
            self.velocity_history  = []
            self.position_history  = [t['center']]

        def get_position(self):
            return (self._cx, self._cy)

        def get_state(self):
            return self._state

    return [FakeTracker(t) for t in ds_tracks]


def main():
    os.makedirs("outputs", exist_ok=True)

    detector  = CameraDetector()
    tracker   = DeepSORTTracker()
    radar     = RadarSimulator()
    fusion    = SensorFusion()
    evaluator = Evaluator()

    # ── Choose data source ────────────────────────────
    if USE_KITTI:
        from utils.kitti_loader import KITTILoader
        cap = KITTILoader(KITTI_SEQUENCE,
                          KITTI_IMAGE_DIR,
                          KITTI_LABEL_DIR)
        print("[Main] Using KITTI dataset")
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print("[Main] Using webcam")

    if not cap.isOpened():
        print("[ERROR] Cannot open source!")
        return

    # Get first frame size
    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] No frames!")
        return

    h, w = first_frame.shape[:2]
    radar.set_origin(w, h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(
        'outputs/tracked_output.mp4', fourcc, 20.0, (w, h))

    print("[Main] Pipeline running... Press 'q' to quit")
    frame = first_frame

    while True:
        # ── Core Pipeline ──────────────────────────────
        detections  = detector.detect(frame)
        ds_tracks   = tracker.update(detections, frame)
        fake_tracks = convert_deepsort_to_tracker_format(ds_tracks)
        radar_meas  = radar.simulate(fake_tracks)
        fusion_res  = fusion.fuse(fake_tracks, radar_meas)
        # ──────────────────────────────────────────────

        # Get GT if using KITTI
        gt = []
        if USE_KITTI and hasattr(cap, 'get_ground_truth'):
            gt = cap.get_ground_truth()

        evaluator.update(detections, ds_tracks, fusion_res, gt)

        # Draw
        frame = draw_detections(frame, detections)
        frame = draw_deepsort_tracks(frame, ds_tracks)
        frame = draw_radar(frame, radar_meas, radar.origin)
        frame = draw_fusion(frame, fusion_res)
        frame = draw_legend(frame)

        fused = sum(1 for r in fusion_res if r['fusion_active'])
        cv2.putText(frame,
            f"Det:{len(detections)}  "
            f"Tracks:{len(ds_tracks)}  "
            f"Fused:{fused}  "
            f"Frame:{evaluator.frame_count}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Camera-Radar Fusion Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, frame = cap.read()
        if not ret:
            break

    # Cleanup + save report
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    evaluator.save_report("outputs/evaluation_report.png")
    print("[Main] Done! Check outputs/ folder")


if __name__ == "__main__":
    main()