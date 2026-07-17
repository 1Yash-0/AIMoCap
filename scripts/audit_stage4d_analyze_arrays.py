"""Stage 4D.2: Numeric Analysis of Motion Energy Arrays"""

import numpy as np
from pathlib import Path

SEQ = "171204_pose3"
CAMS = ["00_03", "00_04", "00_28", "00_24"]

def main():
    print(f"Numeric Analysis of Cross-Correlation (SEQ: {SEQ})\\n")
    
    for cn in CAMS:
        npz_path = Path(f'data/panoptic/{SEQ}/energy_{cn}.npz')
        if not npz_path.exists():
            print(f"Camera {cn}: NPZ file missing!")
            continue
            
        data = np.load(npz_path)
        obs = data['obs']
        gt = data['gt']
        global_lag = data['lag']
        
        # Calculate full correlation
        corr_full = np.correlate(obs, gt, mode='full')
        lags_full = np.arange(-len(obs) + 1, len(gt))
        
        peak_idx = np.argmax(corr_full)
        peak_val = corr_full[peak_idx]
        mean_val = np.mean(corr_full)
        std_val = np.std(corr_full)
        z_score = (peak_val - mean_val) / (std_val + 1e-8)
        
        # Split into halves to check drift
        mid = len(obs) // 2
        
        obs1, gt1 = obs[:mid], gt[:mid]
        corr1 = np.correlate(obs1, gt1, mode='full')
        lags1 = np.arange(-len(obs1) + 1, len(gt1))
        lag1 = lags1[np.argmax(corr1)]
        z1 = (np.max(corr1) - np.mean(corr1)) / (np.std(corr1) + 1e-8)
        
        obs2, gt2 = obs[mid:], gt[mid:]
        corr2 = np.correlate(obs2, gt2, mode='full')
        lags2 = np.arange(-len(obs2) + 1, len(gt2))
        lag2 = lags2[np.argmax(corr2)]
        z2 = (np.max(corr2) - np.mean(corr2)) / (np.std(corr2) + 1e-8)
        
        print(f"--- Camera {cn} ---")
        print(f"  Global Lag: {global_lag} (Z-score: {z_score:.1f})")
        print(f"  First Half Lag : {lag1} (Z-score: {z1:.1f})")
        print(f"  Second Half Lag: {lag2} (Z-score: {z2:.1f})")
        
        if lag1 == lag2:
            print(f"  Result: STABLE (Fixed offset of {lag1} frames)")
        else:
            print(f"  Result: DRIFT DETECTED (Frames dropped or variable FPS)")
        print("")

if __name__ == "__main__":
    main()
