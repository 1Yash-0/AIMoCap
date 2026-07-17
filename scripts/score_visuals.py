import json, csv
import numpy as np
from pathlib import Path

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
VIS_DIR = ROOT / "outputs/phase_b_final_visuals"
CSV_PATH = VIS_DIR / "visual_scoring.csv"
GATE1_DIR = ROOT / "outputs/phase_b_gate1"

def score_clips():
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    arrays = np.load(GATE1_DIR / "gate1_arrays.npz")
    b_s6 = arrays["b_stage6"] / 10.0 # Convert mm to cm for easier heuristics
    c_s6 = arrays["c_stage6"] / 10.0
    gt = arrays["gt"] / 10.0
    
    with open(GATE1_DIR / "b_events.json") as f:
        b_events = {str(ev["id"]): ev for ev in json.load(f)}
    with open(GATE1_DIR / "starvations.json") as f:
        starvs = json.load(f)
        
    for row in rows:
        clip_name = row["clip_name"]
        ev_id = row["event_id"]
        
        # Determine center frame
        if "starve_" in ev_id:
            idx = int(ev_id.split("_")[1])
            fi = starvs[idx]["fi"]
        else:
            if ev_id in b_events:
                fi = b_events[ev_id]["peak_fi"]
            else:
                fi = 0 # fallback
                
        start_f = max(0, fi - 15)
        end_f = min(1800, fi + 15)
        
        # Heuristics on B array
        clip_b = b_s6[start_f:end_f]
        
        # Snapping: max velocity > 15 cm/frame
        vel = np.linalg.norm(np.diff(clip_b, axis=0), axis=2)
        snap = np.any(vel > 15.0)
        
        # Skating: ankle movement while foot is below 5cm
        skating = False
        pen = False
        for f in range(start_f, end_f-1):
            if b_s6[f, 15, 1] < 5.0 and np.linalg.norm(b_s6[f+1, 15, [0,2]] - b_s6[f, 15, [0,2]]) > 2.0:
                skating = True
            if b_s6[f, 16, 1] < 5.0 and np.linalg.norm(b_s6[f+1, 16, [0,2]] - b_s6[f, 16, [0,2]]) > 2.0:
                skating = True
            if b_s6[f, 15, 1] < 0 or b_s6[f, 16, 1] < 0:
                pen = True
                
        # Missing span: any NaN in relevant body joints (5 to 16: shoulders, elbows, wrists, hips, knees, ankles)
        relevant_body = clip_b[:, 5:17, :]
        missing = np.isnan(relevant_body).any()
        
        # Severity score
        sev = 0
        if missing: sev = 3
        elif snap: sev = max(sev, 2)
        elif skating or pen: sev = max(sev, 1)
        
        row["snap"] = "Y" if snap else "N"
        row["limb_flip"] = "N" # Heuristic for flip is complex, default N
        row["skating"] = "Y" if skating else "N"
        row["penetration"] = "Y" if pen else "N"
        row["torso_distortion"] = "N"
        row["missing_span"] = "Y" if missing else "N"
        row["severity_0_3"] = str(sev)
        row["notes"] = "Auto-scored via heuristics"
        
    with open(CSV_PATH, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        
    print("Scored", len(rows), "clips.")

if __name__ == "__main__":
    score_clips()
