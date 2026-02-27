# Radar Simulation Module

import numpy as np
from config import RADAR_RANGE_NOISE, RADAR_VELOCITY_NOISE, RADAR_MAX_RANGE


class RadarSimulator:
    """
    Simulates a radar sensor.
    Takes ground truth positions and adds realistic noise.
    Outputs range, angle, and radial velocity measurements.
    """

    def __init__(self):
        print("[Radar] Radar simulator initialized!")
        # Radar origin = center bottom of frame (where radar would be mounted)
        self.origin = None

    def set_origin(self, frame_width, frame_height):
        """Set radar mounting position (bottom center of frame)."""
        self.origin = (frame_width // 2, frame_height)

    def simulate(self, trackers):
        """
        For each tracker, simulate a radar measurement.
        Returns list of radar measurements.
        """
        if self.origin is None:
            return []

        measurements = []
        ox, oy = self.origin

        for tracker in trackers:
            state = tracker.get_state()
            true_x, true_y = state['x'], state['y']
            true_vx, true_vy = state['vx'], state['vy']

            # --- True range and angle ---
            dx = true_x - ox
            dy = true_y - oy
            true_range = np.sqrt(dx**2 + dy**2)
            true_angle = np.arctan2(dy, dx)  # radians

            # Skip if object is too far for radar
            if true_range > RADAR_MAX_RANGE:
                continue

            # --- Add Gaussian noise (simulating real radar error) ---
            noisy_range = true_range + np.random.normal(0, RADAR_RANGE_NOISE)
            noisy_angle = true_angle + np.random.normal(0, 0.02)  # small angle noise

            # --- Radial velocity (how fast object moves toward/away) ---
            # Project velocity onto range direction
            range_unit_x = dx / (true_range + 1e-6)
            range_unit_y = dy / (true_range + 1e-6)
            true_radial_vel = true_vx * range_unit_x + true_vy * range_unit_y
            noisy_radial_vel = true_radial_vel + np.random.normal(0, RADAR_VELOCITY_NOISE)

            # --- Convert back to Cartesian ---
            radar_x = ox + noisy_range * np.cos(noisy_angle)
            radar_y = oy + noisy_range * np.sin(noisy_angle)

            measurements.append({
                'tracker_id': tracker.id,
                'range': noisy_range,
                'angle': noisy_angle,
                'radial_velocity': noisy_radial_vel,
                'x': radar_x,
                'y': radar_y,
                'true_range': true_range,
                'true_radial_vel': true_radial_vel
            })

        return measurements


def draw_radar(frame, measurements, origin):
    """Draw radar measurements on frame."""
    if origin is None:
        return frame

    # Draw radar origin
    cv_origin = (int(origin[0]), int(origin[1]))

    for m in measurements:
        rx, ry = int(m['x']), int(m['y'])

        # Draw radar detected position (blue dot)
        import cv2
        cv2.circle(frame, (rx, ry), 6, (255, 100, 0), -1)

        # Draw line from radar origin to detected position
        cv2.line(frame, cv_origin, (rx, ry), (255, 100, 0), 1)

        # Show range and velocity
        label = f"r:{m['range']:.0f} v:{m['radial_velocity']:.1f}"
        cv2.putText(frame, label, (rx + 8, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

    return frame