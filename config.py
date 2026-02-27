# All project settings

# Video source: 0 = webcam, or path to a video file
VIDEO_SOURCE = 0  # we'll change this later

# YOLO model to use (nano = smallest, fastest on CPU)
YOLO_MODEL = "yolov8n.pt"

# Confidence threshold — detections below this are ignored
CONFIDENCE_THRESHOLD = 0.5

# Classes we care about (COCO dataset IDs)
# 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES = [0, 2, 3, 5, 7]

# Frame size
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Time between frames (seconds) — for Kalman Filter
DT = 0.033  # ~30 FPS

# ---- Radar Settings ----

# Maximum radar detection range (pixels)
RADAR_MAX_RANGE = 800

# Gaussian noise standard deviation for range measurement
RADAR_RANGE_NOISE = 5.0

# Gaussian noise standard deviation for velocity measurement
RADAR_VELOCITY_NOISE = 0.5

# ---- Fusion Settings ----

# How much to trust camera velocity vs radar velocity (must add to 1.0)
FUSION_CAMERA_TRUST = 0.4
FUSION_RADAR_TRUST  = 0.6


# ---- Dataset Settings ----
# Set to True to use KITTI dataset instead of webcam
USE_KITTI        = True
KITTI_SEQUENCE   = "0006"
KITTI_IMAGE_DIR  = "data/kitti/image_02"
KITTI_LABEL_DIR  = "data/kitti/label_02"