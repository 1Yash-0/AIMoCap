"""
Phase B - Gate 2: Visual Rendering
Reads immutable outputs from Gate 1 and renders synchronized clips for first-pass scoring.
"""

import sys, json, csv, uuid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

OUT_VIS = ROOT / "outputs/phase_b_final_visuals"
OUT_VIS.mkdir(parents=True, exist_ok=True)
GATE1_DIR = ROOT / "outputs/phase_b_gate1"

# We use the same bone connections as viz_rest_plot.py for plotting
# COCO 17 layout connections
CONNECTIONS = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 0), (6, 0), (0, 1), (1, 2), (2, 3), (1, 4)]
CAMS = ["00_11", "00_12", "00_23"]

def plot_skeleton_3d(ax, pts3d, color, label, alpha=1.0):
    if not np.isfinite(pts3d).all(): return
    for p, c in CONNECTIONS:
        if p < len(pts3d) and c < len(pts3d):
            ax.plot([pts3d[p,0], pts3d[c,0]], [pts3d[p,1], pts3d[c,1]], [pts3d[p,2], pts3d[c,2]], color=color, alpha=alpha)
    ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2], color=color, s=10, label=label, alpha=alpha)

def plot_skeleton_2d(ax, pts2d, conf, mask_log_for_frame_cam, title):
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.set_title(title)
    if not np.isfinite(pts2d).any(): return
    for p, c in CONNECTIONS:
        if p < len(pts2d) and c < len(pts2d):
            if conf[p] > 0.1 and conf[c] > 0.1:
                ax.plot([pts2d[p,0], pts2d[c,0]], [pts2d[p,1], pts2d[c,1]], color='gray', alpha=0.5)
    
    # Plot masked points in red
    masked_j = [m["j"] for m in mask_log_for_frame_cam]
    for j in range(len(pts2d)):
        if conf[j] > 0.1:
            color = 'red' if j in masked_j else 'blue'
            ax.scatter(pts2d[j,0], pts2d[j,1], color=color, s=15)

def render_clip(fi_center, duration, name, arrays, events_data):
    start_f = max(0, fi_center - 15)
    end_f = min(1800, fi_center + duration + 15)
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3)
    ax_cam1 = fig.add_subplot(gs[0, 0])
    ax_cam2 = fig.add_subplot(gs[0, 1])
    ax_cam3 = fig.add_subplot(gs[0, 2])
    ax_3d = fig.add_subplot(gs[1, :], projection='3d')
    
    gt = arrays["gt"]
    b_s6 = arrays["b_stage6"]
    c_s6 = arrays["c_stage6"]
    mask_log = arrays["mask_c_log"]
    npz_obs = np.load(ROOT / "outputs/canonical_dataset/canonical_detector_pose_observations.npz")
    kpts = npz_obs["kpts"]
    scores = npz_obs["scores"]
    
    # Pre-filter mask log
    mask_log_list = mask_log.tolist() if hasattr(mask_log, "tolist") else mask_log
    
    def update(frame_idx):
        ax_cam1.clear(); ax_cam2.clear(); ax_cam3.clear(); ax_3d.clear()
        
        # 3D plot
        plot_skeleton_3d(ax_3d, gt[frame_idx]/10.0, 'green', 'GT', alpha=0.8)
        plot_skeleton_3d(ax_3d, b_s6[frame_idx]/10.0, 'blue', 'B', alpha=0.6)
        plot_skeleton_3d(ax_3d, c_s6[frame_idx]/10.0, 'red', 'C', alpha=0.6)
        
        ax_3d.set_xlim(-150, 150)
        ax_3d.set_ylim(-50, 200)
        ax_3d.set_zlim(-150, 150)
        ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z')
        ax_3d.legend()
        
        # 2D plots
        ml1 = [m for m in mask_log_list if m["f"] == frame_idx and m["c"] == 0]
        ml2 = [m for m in mask_log_list if m["f"] == frame_idx and m["c"] == 1]
        ml3 = [m for m in mask_log_list if m["f"] == frame_idx and m["c"] == 2]
        
        plot_skeleton_2d(ax_cam1, kpts[frame_idx, 0], scores[frame_idx, 0], ml1, "Cam 00_11")
        plot_skeleton_2d(ax_cam2, kpts[frame_idx, 1], scores[frame_idx, 1], ml2, "Cam 00_12")
        plot_skeleton_2d(ax_cam3, kpts[frame_idx, 2], scores[frame_idx, 2], ml3, "Cam 00_23")
        
        # Overlay metrics
        ev_text = f"Frame: {frame_idx}\nOrigin B: {arrays['b_origin'][frame_idx, 15]}\nOrigin C: {arrays['c_origin'][frame_idx, 15]}"
        fig.suptitle(f"{name} | {ev_text}")
        
    ani = animation.FuncAnimation(fig, update, frames=range(start_f, end_f), interval=1000/30.0)
    out_path = OUT_VIS / f"{name}.mp4"
    ani.save(str(out_path), writer='ffmpeg')
    plt.close(fig)
    return name

def main():
    if not (GATE1_DIR/"gate1_metrics.json").exists():
        print("Gate 1 outputs not found. Run Gate 1 first.")
        return
        
    with open(GATE1_DIR/"gate1_metrics.json") as f: metrics = json.load(f)
    with open(GATE1_DIR/"b_events.json") as f: b_events = json.load(f)
    with open(GATE1_DIR/"starvations.json") as f: starvations = json.load(f)
    arrays = np.load(GATE1_DIR/"gate1_arrays.npz", allow_pickle=True)
    
    print(f"Loaded {len(b_events)} B-events and {len(starvations)} C-starvations.")
    
    b_only = b_events.copy()
    b_only.sort(key=lambda x: x["peak_acc"], reverse=True)
    
    rendered = set()
    csv_rows = []
    
    def render_and_log(category, ev):
        if ev["id"] in rendered: return
        rendered.add(ev["id"])
        print(f"Rendering {category}: {ev['id']}")
        name = f"{category}_{ev['id']}"
        render_clip(ev["peak_fi"], ev.get("duration", 1), name, arrays, ev)
        csv_rows.append({"clip_name": name, "category": category, "event_id": ev["id"], "snap": "", "limb_flip": "", "skating": "", "penetration": "", "torso_distortion": "", "missing_span": "", "severity_0_3": "", "notes": ""})

    # 1. 10 highest-severity B-only events
    for ev in b_only[:10]: render_and_log("worst_severity", ev)
    
    # 2. 10 longest B-only events
    b_only_longest = sorted(b_only, key=lambda x: x.get("duration", 0), reverse=True)
    for ev in b_only_longest[:10]: render_and_log("longest", ev)
        
    # 3. 74 C starvation cases
    for i, st in enumerate(starvations):
        ev = {"id": f"starve_{i}", "peak_fi": st["fi"], "duration": 1}
        render_and_log("starvation", ev)
        
    # 4. 10 random B-only events
    rng = np.random.RandomState(42)
    random_evs = rng.choice(b_only, min(10, len(b_only)), replace=False)
    for ev in random_evs: render_and_log("random", ev)
        
    # Write CSV for scoring
    with open(OUT_VIS/"visual_scoring.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"Rendered {len(rendered)} clips. CSV written to {OUT_VIS}/visual_scoring.csv")

if __name__ == "__main__":
    main()
