import json, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

# COCO 17-keypoint connections
SKELETON_COCO = [(15,13), (13,11), (16,14), (14,12), (11,12), (5,11), (6,12), (5,6), (5,7), (7,9), (6,8), (8,10), (5,0), (6,0), (0,1), (0,2), (1,3), (2,4)]

def plot_skel(ax, pts3d, color='blue'):
    if np.isnan(pts3d).all(): return
    ax.scatter(pts3d[:,0], pts3d[:,2], pts3d[:,1], c=color, s=10)
    for bone in SKELETON_COCO:
        if not np.isnan(pts3d[bone[0]]).any() and not np.isnan(pts3d[bone[1]]).any():
            ax.plot([pts3d[bone[0],0], pts3d[bone[1],0]],
                    [pts3d[bone[0],2], pts3d[bone[1],2]],
                    [pts3d[bone[0],1], pts3d[bone[1],1]], c=color)

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
GATE1 = ROOT / "outputs/phase_b_gate1"
OUT_GIFS = ROOT / "outputs/phase_b_final_visuals_gifs"
OUT_GIFS.mkdir(parents=True, exist_ok=True)

def generate_decisive_gifs():
    arrays = np.load(GATE1 / "gate1_arrays.npz")
    b_s6 = arrays["b_stage6"] / 10.0
    c_s6 = arrays["c_stage6"] / 10.0
    gt = arrays["gt"] / 10.0
    
    with open(GATE1 / "b_events.json") as f:
        b_ev = {str(e["id"]): e for e in json.load(f)}
    with open(GATE1 / "starvations.json") as f:
        starvs = json.load(f)
        
    with open(ROOT / "outputs/phase_b_final_visuals/visual_scoring.csv") as f:
        scores = list(csv.DictReader(f))
        
    # We want to select the compact decisive set:
    # 1. 3 sample C starvations (to keep artifact compact)
    # 2. 3 worst B-vs-C ankle diffs
    # 3. 3 worst sliding
    # 4. 3 B-only high severity
    # We will just pick the top 3 of each from the CSV rows (since the CSV was generated with these strata in order: 10 worst B-events, 10 longest B, 74 starvations, 10 B vs C diffs)
    
    decisive_rows = []
    strata_counts = {"B-Event-Severity":0, "B-Event-Length":0, "Starvation":0, "B-vs-C Ankle Diff":0}
    
    for row in scores:
        cat = row["category"]
        if strata_counts.get(cat, 0) < 3:
            decisive_rows.append(row)
            strata_counts[cat] = strata_counts.get(cat, 0) + 1
            
    print(f"Generating GIFs for {len(decisive_rows)} decisive clips...")
    
    for row in decisive_rows:
        clip_name = row["clip_name"]
        ev_id = row["event_id"]
        
        if "starve_" in ev_id:
            idx = int(ev_id.split("_")[1])
            fi = starvs[idx]["fi"]
        else:
            fi = b_ev.get(ev_id, {}).get("peak_fi", 0)
            
        start_f = max(0, fi - 15)
        end_f = min(1800, fi + 15)
        
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        
        def update(f):
            ax1.clear(); ax2.clear(); ax3.clear()
            ax1.set_title("GT")
            ax2.set_title("Cand B (No Gate)")
            ax3.set_title("Cand C (Masks)")
            
            for ax in (ax1, ax2, ax3):
                ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_zlim(-100, 100)
                ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
                
            plot_skel(ax1, gt[f], 'black')
            plot_skel(ax2, b_s6[f], 'green')
            plot_skel(ax3, c_s6[f], 'red')
            
            fig.suptitle(f"{clip_name} | Frame {f}")
            
        ani = animation.FuncAnimation(fig, update, frames=range(start_f, end_f), interval=33)
        out_path = OUT_GIFS / f"{clip_name}.gif"
        ani.save(out_path, writer='pillow')
        plt.close(fig)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_decisive_gifs()
