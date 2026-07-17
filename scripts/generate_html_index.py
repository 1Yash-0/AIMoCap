import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
GATE1 = ROOT / "outputs/phase_b_gate1"
ART_DIR = Path(r"C:\Users\prade\.gemini\antigravity-ide\brain\395c9de1-f24b-42ee-9272-3ef825c55485")

SKELETON_COCO = [(15,13), (13,11), (16,14), (14,12), (11,12), (5,11), (6,12), (5,6), (5,7), (7,9), (6,8), (8,10), (5,0), (6,0), (0,1), (0,2), (1,3), (2,4)]

def plot_skel(ax, pts3d, color='blue'):
    if np.isnan(pts3d).all(): return
    ax.scatter(pts3d[:,0], pts3d[:,2], pts3d[:,1], c=color, s=10)
    for bone in SKELETON_COCO:
        if not np.isnan(pts3d[bone[0]]).any() and not np.isnan(pts3d[bone[1]]).any():
            ax.plot([pts3d[bone[0],0], pts3d[bone[1],0]],
                    [pts3d[bone[0],2], pts3d[bone[1],2]],
                    [pts3d[bone[0],1], pts3d[bone[1],1]], c=color)

def main():
    arrays = np.load(GATE1 / "gate1_arrays.npz")
    b_s6 = arrays["b_stage6"] / 10.0
    c_s6 = arrays["c_stage6"] / 10.0
    gt = arrays["gt"] / 10.0
    
    with open(GATE1 / "unsupported_b_added.json") as f:
        unsupported = json.load(f)
    with open(GATE1 / "starvations.json") as f:
        starvs = json.load(f)
        
    rng = np.random.RandomState(42)
    
    # 1. Worst Starvations (Max inliers lost)
    starvs.sort(key=lambda x: (x["inliers_before"] - x["inliers_after"]), reverse=True)
    sel_starvs = starvs[:3]
    
    # 2. Severe Unsupported B-only (Highest Peak Acc)
    unsupported.sort(key=lambda x: x["peak_acc"], reverse=True)
    sel_unsup = unsupported[:3]
    
    # 3. Fixed seed random sample of unsupported B
    sel_rand = [unsupported[i] for i in rng.choice(len(unsupported), min(3, len(unsupported)), replace=False)]
    
    # To compute sliding diff, we need the array diff
    vel_b = np.diff(b_s6, axis=0) * 30.0
    vel_c = np.diff(c_s6, axis=0) * 30.0
    slide_diff = []
    for f in range(len(vel_b)-1):
        # proxy slide: just check ankles
        sb = max(np.linalg.norm(vel_b[f,15]), np.linalg.norm(vel_b[f,16]))
        sc = max(np.linalg.norm(vel_c[f,15]), np.linalg.norm(vel_c[f,16]))
        slide_diff.append({"fi": f, "diff": sb - sc})
    slide_diff.sort(key=lambda x: x["diff"], reverse=True)
    sel_slide = slide_diff[:3]
    
    clips = []
    
    for i, st in enumerate(sel_starvs): clips.append(("C_Starvation", f"starve_{i}", st["fi"]))
    for i, us in enumerate(sel_unsup): clips.append(("B_Unsupported_Severe", f"unsup_{i}", int(us["peak_fi"])))
    for i, rd in enumerate(sel_rand): clips.append(("B_Unsupported_Random", f"rand_{i}", int(rd["peak_fi"])))
    for i, sl in enumerate(sel_slide): clips.append(("B_Worst_Sliding", f"slide_{i}", sl["fi"]))
        
    html = """<html><head><style>
    body { font-family: sans-serif; padding: 20px; background: #121212; color: #fff; }
    h1 { border-bottom: 1px solid #444; padding-bottom: 10px; }
    .card { background: #222; padding: 15px; border-radius: 8px; border: 1px solid #333; margin-bottom: 20px;}
    img { width: 600px; max-width: 100%; border-radius: 4px; }
    </style></head><body>
    <h1>Phase B Decisive Visual Evidence (Gate 2)</h1>
    <p>Please review these clips to determine if Candidate C's quantitative advantages justify its visible starvation defects.</p>
    """
    
    for cat, name, fi in clips:
        print(f"Generating {name} (Frame {fi})")
        start_f = max(0, fi - 15)
        end_f = min(len(gt)-1, fi + 15)
        
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        
        def update(f):
            ax1.clear(); ax2.clear(); ax3.clear()
            ax1.set_title("GT (Black)")
            ax2.set_title("Cand B: No Gate (Green)")
            ax3.set_title("Cand C: Mask Ankle (Red)")
            for ax in (ax1, ax2, ax3):
                ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_zlim(-100, 100)
                ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            plot_skel(ax1, gt[f], 'black')
            plot_skel(ax2, b_s6[f], 'green')
            plot_skel(ax3, c_s6[f], 'red')
            fig.suptitle(f"{cat} | Frame {f}")
            
        ani = animation.FuncAnimation(fig, update, frames=range(start_f, end_f), interval=33)
        gif_path = ART_DIR / f"{name}.gif"
        ani.save(gif_path, writer='pillow')
        plt.close(fig)
        
        html += f"""
        <div class="card">
            <h3>{cat} (Peak Frame: {fi})</h3>
            <img src="{name}.gif" />
        </div>
        """
        
    html += "</body></html>"
    (ART_DIR / "index.html").write_text(html)
    print("Done!")

if __name__ == "__main__":
    main()
