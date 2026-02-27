# KITTI Dataset Loader

import os
import cv2
import numpy as np


class KITTILoader:
    """
    Loads KITTI tracking dataset frames and ground truth labels.
    Replaces webcam feed with real driving footage.
    """

    def __init__(self, sequence="0000",
                 image_dir="data/kitti/image_02",
                 label_dir="data/kitti/label_02"):

        self.sequence   = sequence
        self.image_dir  = os.path.join(image_dir, sequence)
        self.label_file = os.path.join(label_dir, f"{sequence}.txt")

        # Load all frame paths sorted
        self.frames = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.endswith('.png')
        ])

        self.total_frames = len(self.frames)
        self.current_idx  = 0

        # Load ground truth labels
        self.ground_truth = self._load_labels()
        print(f"[KITTI] Loaded {self.total_frames} frames "
              f"from sequence {sequence}")

    def _load_labels(self):
        """
        Parse KITTI label file into per-frame ground truth.
        Returns dict: {frame_id: [list of objects]}
        """
        gt = {}
        if not os.path.exists(self.label_file):
            print("[KITTI] No label file found — running without GT")
            return gt

        with open(self.label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                frame_id  = int(parts[0])
                obj_type  = parts[2]
                # Skip background classes
                if obj_type in ['DontCare', 'Misc']:
                    continue

                # Bounding box in pixels
                x1 = float(parts[6])
                y1 = float(parts[7])
                x2 = float(parts[8])
                y2 = float(parts[9])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if frame_id not in gt:
                    gt[frame_id] = []

                gt[frame_id].append({
                    'bbox':       [int(x1), int(y1),
                                   int(x2), int(y2)],
                    'center':     (cx, cy),
                    'class_name': obj_type,
                    'confidence': 1.0,   # GT = perfect confidence
                    'class_id':   -1     # not a COCO ID
                })

        return gt

    def read(self):
        """Mimic cv2.VideoCapture.read() interface."""
        if self.current_idx >= self.total_frames:
            return False, None

        frame = cv2.imread(self.frames[self.current_idx])
        self.current_idx += 1
        return True, frame

    def get_ground_truth(self, frame_id=None):
        """Return GT boxes for current or given frame."""
        fid = frame_id if frame_id is not None \
              else self.current_idx - 1
        return self.ground_truth.get(fid, [])

    def isOpened(self):
        return self.current_idx < self.total_frames

    def release(self):
        pass  # nothing to release for image sequences