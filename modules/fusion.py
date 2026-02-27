# Sensor Fusion Module

import numpy as np
from config import (FUSION_CAMERA_TRUST, FUSION_RADAR_TRUST)


class SensorFusion:
    """
    Fuses camera position measurements and radar velocity measurements
    into the Kalman Filter state of each tracker.
    """

    def __init__(self):
        print("[Fusion] Sensor fusion initialized!")

    def fuse(self, trackers, radar_measurements):
        """
        For each tracker that has a matching radar measurement,
        perform a fused Kalman update using both camera + radar.

        Camera gives us: x, y (position)
        Radar gives us:  radial_velocity (speed toward/away)
        """

        # Build a lookup: tracker_id -> radar measurement
        radar_lookup = {m['tracker_id']: m for m in radar_measurements}

        fusion_results = []

        for tracker in trackers:
            state = tracker.get_state()
            result = {
                'tracker_id': tracker.id,
                'position': tracker.get_position(),
                'camera_velocity': (state['vx'], state['vy']),
                'radar_velocity': None,
                'fused_speed': state['speed'],
                'fusion_active': False
            }

            if tracker.id in radar_lookup:
                radar = radar_lookup[tracker.id]

                # --- Fuse radar velocity into Kalman state ---
                # Radar gives radial velocity (toward/away from sensor)
                # We use it to correct the vx, vy estimates

                radar_vel = radar['radial_velocity']
                angle = radar['angle']

                # Decompose radial velocity into vx, vy components
                radar_vx = radar_vel * np.cos(angle)
                radar_vy = radar_vel * np.sin(angle)

                # Weighted fusion of camera velocity and radar velocity
                cam_vx = state['vx']
                cam_vy = state['vy']

                fused_vx = (FUSION_CAMERA_TRUST * cam_vx +
                            FUSION_RADAR_TRUST * radar_vx)
                fused_vy = (FUSION_CAMERA_TRUST * cam_vy +
                            FUSION_RADAR_TRUST * radar_vy)
                
                # Calculate fused speed first
                fused_speed = np.sqrt(fused_vx**2 + fused_vy**2)
                
                # Inject fused velocity back into state
                # (only if tracker has Kalman Filter)
                if hasattr(tracker, 'kf'):
                    tracker.kf.x[2] = fused_vx
                    tracker.kf.x[3] = fused_vy
                else:
                    tracker._state['vx'] = fused_vx
                    tracker._state['vy'] = fused_vy
                    tracker._state['speed'] = fused_speed

                fused_speed = np.sqrt(fused_vx**2 + fused_vy**2)

                result['radar_velocity'] = radar_vel
                result['fused_speed'] = fused_speed
                result['fusion_active'] = True

                # Save to tracker history
                tracker.velocity_history.append((fused_vx, fused_vy))

            fusion_results.append(result)

        return fusion_results


def draw_fusion(frame, fusion_results):
    """Draw fusion info on frame."""
    import cv2

    for res in fusion_results:
        if not res['fusion_active']:
            continue

        px, py = res['position']

        # Draw fusion indicator ring (cyan)
        cv2.circle(frame, (px, py), 15, (255, 255, 0), 2)

        # Show fused speed
        label = f"F:{res['fused_speed']:.1f}"
        cv2.putText(frame, label, (px + 18, py + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    return frame