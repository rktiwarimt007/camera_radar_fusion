# YOLO Camera Detection Module

from ultralytics import YOLO
import cv2
import numpy as np
from config import YOLO_MODEL, CONFIDENCE_THRESHOLD, TARGET_CLASSES


class CameraDetector:
    def __init__(self):
        print("[Detector] Loading YOLO model...")
        self.model = YOLO(YOLO_MODEL)
        print("[Detector] Model loaded successfully!")

    def detect(self, frame):
        """
        Takes a video frame, returns list of detections.
        Each detection = {
            'bbox': [x1, y1, x2, y2],
            'center': (cx, cy),
            'confidence': float,
            'class_id': int,
            'class_name': str
        }
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Filter by class and confidence
            if class_id not in TARGET_CLASSES:
                continue
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Center point of bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'center': (cx, cy),
                'confidence': confidence,
                'class_id': class_id,
                'class_name': results.names[class_id]
            })

        return detections


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame."""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx, cy = det['center']
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw center point
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Draw label
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame