# Real-time Plot Visualizer

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import threading


class VelocityPlotter:
    """
    Shows a live matplotlib graph of fused speed over time.
    Runs in a separate thread so it doesn't block the video.
    """

    def __init__(self, max_points=100):
        self.max_points   = max_points
        self.speed_data   = deque(maxlen=max_points)
        self.frame_data   = deque(maxlen=max_points)
        self.frame_count  = 0
        self.lock         = threading.Lock()
        self.running      = True

        # Setup plot
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#1e1e1e')
        self.ax.set_title("Fused Speed Over Time",
                           color='white', fontsize=12)
        self.ax.set_xlabel("Frame", color='white')
        self.ax.set_ylabel("Speed (px/frame)", color='white')
        self.ax.tick_params(colors='white')
        self.line, = self.ax.plot([], [], color='cyan', linewidth=2)
        plt.tight_layout()
        plt.ion()
        plt.show()

    def update(self, fusion_results):
        self.frame_count += 1
        speeds = [r['fused_speed'] for r in fusion_results
                  if r['fusion_active']]
        avg_speed = np.mean(speeds) if speeds else 0.0

        with self.lock:
            self.speed_data.append(avg_speed)
            self.frame_data.append(self.frame_count)

        # Refresh plot every 5 frames
        if self.frame_count % 5 == 0:
            with self.lock:
                x = list(self.frame_data)
                y = list(self.speed_data)
            self.line.set_data(x, y)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def save(self, path="outputs/velocity_plot.png"):
        self.fig.savefig(path, facecolor=self.fig.get_facecolor())
        print(f"[Plotter] Saved velocity plot → {path}")

    def close(self):
        plt.close(self.fig)