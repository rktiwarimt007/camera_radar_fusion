# Comprehensive Evaluation & Graphs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


class Evaluator:
    """
    Records metrics per frame and generates
    a comprehensive evaluation report with graphs.
    """

    def __init__(self):
        self.frames           = []
        self.det_counts       = []
        self.track_counts     = []
        self.fused_speeds     = []
        self.radar_speeds     = []
        self.camera_speeds    = []
        self.precision_list   = []
        self.recall_list      = []
        self.tp_list = []
        self.fp_list = []
        self.fn_list = []
        self.frame_count      = 0

    def update(self, detections, active_trackers,
               fusion_results, ground_truth=None):
        self.frame_count += 1
        self.frames.append(self.frame_count)
        self.det_counts.append(len(detections))
        self.track_counts.append(len(active_trackers))

        # Speed data
        fused = [r['fused_speed'] for r in fusion_results
                 if r['fusion_active']]
        radar = [abs(r['radar_velocity']) for r in fusion_results
                 if r['radar_velocity'] is not None]
        cam   = [np.sqrt(r['camera_velocity'][0]**2 +
                         r['camera_velocity'][1]**2)
                 for r in fusion_results]

        self.fused_speeds.append(np.mean(fused) if fused else 0)
        self.radar_speeds.append(np.mean(radar) if radar else 0)
        self.camera_speeds.append(np.mean(cam)  if cam  else 0)

        # Precision / Recall
        det   = len(detections)
        trk   = len(active_trackers)
        tp    = min(det, trk)
        fp    = max(0, trk - det)
        fn    = max(0, det - trk)
        self.tp_list.append(tp)
        self.fp_list.append(fp)
        self.fn_list.append(fn)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        self.precision_list.append(prec)
        self.recall_list.append(rec)

    def get_rmse(self):
        errors = [abs(f - r) for f, r in
                  zip(self.fused_speeds, self.radar_speeds) if r > 0]
        return np.sqrt(np.mean(np.array(errors)**2)) if errors else 0

    def save_report(self, path="outputs/evaluation_report.png"):
        os.makedirs("outputs", exist_ok=True)
        fig = plt.figure(figsize=(16, 10),
                         facecolor='#1a1a2e')
        fig.suptitle("Camera-Radar Fusion — Evaluation Report",
                     color='white', fontsize=16, fontweight='bold')

        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                hspace=0.4, wspace=0.35)

        style = {'facecolor': '#16213e',
                 'label.color': 'white'}

        # ── Plot 1: Speed Comparison ──────────────────
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_facecolor('#16213e')
        ax1.plot(self.frames, self.fused_speeds,
                 color='cyan',   linewidth=2, label='Fused Speed')
        ax1.plot(self.frames, self.radar_speeds,
                 color='orange', linewidth=1,
                 linestyle='--', label='Radar Speed')
        ax1.plot(self.frames, self.camera_speeds,
                 color='lime',   linewidth=1,
                 linestyle=':',  label='Camera Speed')
        ax1.set_title("Speed Comparison Over Time",
                      color='white')
        ax1.set_xlabel("Frame", color='white')
        ax1.set_ylabel("Speed (px/frame)", color='white')
        ax1.tick_params(colors='white')
        ax1.legend(facecolor='#1a1a2e', labelcolor='white')

        # ── Plot 2: RMSE gauge ────────────────────────
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor('#16213e')
        rmse = self.get_rmse()
        ax2.bar(['RMSE\nVelocity'], [rmse],
                color='cyan', width=0.4)
        ax2.set_title("Velocity RMSE", color='white')
        ax2.set_ylabel("Error (px/frame)", color='white')
        ax2.tick_params(colors='white')
        ax2.text(0, rmse + 0.1, f"{rmse:.3f}",
                 ha='center', color='white', fontsize=12)

        # ── Plot 3: Detection vs Track count ─────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor('#16213e')
        ax3.plot(self.frames, self.det_counts,
                 color='lime',   label='Detections')
        ax3.plot(self.frames, self.track_counts,
                 color='yellow', label='Tracks')
        ax3.set_title("Detections vs Tracks", color='white')
        ax3.set_xlabel("Frame", color='white')
        ax3.set_ylabel("Count",  color='white')
        ax3.tick_params(colors='white')
        ax3.legend(facecolor='#1a1a2e', labelcolor='white')

        # ── Plot 4: Precision & Recall ────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#16213e')
        ax4.plot(self.frames, self.precision_list,
                 color='cyan',   label='Precision')
        ax4.plot(self.frames, self.recall_list,
                 color='magenta', label='Recall')
        ax4.set_ylim(0, 1.1)
        ax4.set_title("Precision & Recall", color='white')
        ax4.set_xlabel("Frame",  color='white')
        ax4.set_ylabel("Score",  color='white')
        ax4.tick_params(colors='white')
        ax4.legend(facecolor='#1a1a2e', labelcolor='white')

        # ── Plot 5: Summary Stats ─────────────────────
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.set_facecolor('#16213e')
        ax5.axis('off')

        avg_prec = np.mean(self.precision_list)
        avg_rec  = np.mean(self.recall_list)
        f1       = (2 * avg_prec * avg_rec /
                    (avg_prec + avg_rec + 1e-6))

        summary = [
            ("Total Frames",  str(self.frame_count)),
            ("Avg Precision", f"{avg_prec:.3f}"),
            ("Avg Recall",    f"{avg_rec:.3f}"),
            ("F1 Score",      f"{f1:.3f}"),
            ("RMSE Velocity", f"{rmse:.3f}"),
            ("Total TPs",     str(sum(self.tp_list))),
            ("Total FPs",     str(sum(self.fp_list))),
            ("Total FNs",     str(sum(self.fn_list))),
        ]

        ax5.set_title("Summary", color='white')
        for i, (key, val) in enumerate(summary):
            ax5.text(0.05, 0.88 - i*0.12, f"{key}:",
                     color='gray',  fontsize=10,
                     transform=ax5.transAxes)
            ax5.text(0.60, 0.88 - i*0.12, val,
                     color='cyan',  fontsize=10,
                     fontweight='bold',
                     transform=ax5.transAxes)

        plt.savefig(path, facecolor=fig.get_facecolor(),
                    bbox_inches='tight')
        plt.close(fig)
        print(f"[Evaluator] Report saved → {path}")