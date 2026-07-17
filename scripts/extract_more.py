import json
import numpy as np
from pathlib import Path

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
GATE1 = ROOT / "outputs/phase_b_gate1"

def analyze_more():
    with open(GATE1 / "b_events.json") as f:
        b_ev = json.load(f)
    
    # 65% of B events not matched at 5 frames
    # Let's see the peak_acc of unmatched B events vs matched
    unmatched_acc = []
    matched_acc = []
    for ev in b_ev:
        if ev.get("gt_match_dt", 999) <= 5:
            matched_acc.append(ev["peak_acc"])
        else:
            unmatched_acc.append(ev["peak_acc"])
            
    print(f"Matched B events: {len(matched_acc)}, Mean Acc: {np.mean(matched_acc):.1f}")
    print(f"Unmatched B events: {len(unmatched_acc)}, Mean Acc: {np.mean(unmatched_acc):.1f}")
    
    with open(GATE1 / "starvations.json") as f:
        starvs = json.load(f)
    
    origin_counts = {"gap-filled": 0, "ray-sphere": 0, "FK-inferred": 0, "missing": 0}
    for st in starvs:
        o = st.get("origin_c", "")
        if o in origin_counts:
            origin_counts[o] += 1
            
    print(f"74 Starvations Breakdown: {origin_counts}")

if __name__ == "__main__":
    analyze_more()
