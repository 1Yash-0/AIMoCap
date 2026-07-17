import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"e:\Chaos\Projects\aimocap_re")
OUT_DIR = ROOT / "outputs" / "diag_pose1_full"

def main():
    print("Loading CSVs...")
    df_a = pd.read_csv(OUT_DIR / "camera_frame_log.csv")
    df_b = pd.read_csv(OUT_DIR / "keypoint_2d_log.csv")
    df_c = pd.read_csv(OUT_DIR / "camera_pair_geometry_log.csv")
    df_d = pd.read_csv(OUT_DIR / "triangulation_3d_log.csv")
    df_e = pd.read_csv(OUT_DIR / "frame_summary_log.csv")
    
    out_lines = []
    out_lines.append("# Phase 3: Full-Minute Diagnostic Report (171204_pose1)\\n")
    
    # 1. Gate Report
    out_lines.append("## 1. Full-Minute Gate Report\\n")
    
    tot_cam_frames = len(df_a)
    passing_ar = df_a['passed_ar_gate'].sum()
    out_lines.append(f"- **Total Camera-Frames:** {tot_cam_frames}")
    out_lines.append(f"- **AR Gate Pass Rate:** {passing_ar/tot_cam_frames*100:.1f}% ({passing_ar}/{tot_cam_frames})")
    
    tot_2d_joints = len(df_b)
    valid_2d = df_b[df_b['conf'] >= 0.4]
    out_lines.append(f"- **2D Detection Rate (conf>=0.4):** {len(valid_2d)/tot_2d_joints*100:.1f}% ({len(valid_2d)}/{tot_2d_joints})")
    out_lines.append(f"- **2D Reprojection Error (px):** {valid_2d['px_error'].mean():.2f} mean / {valid_2d['px_error'].median():.2f} median")
    
    valid_3d = df_d.dropna(subset=['abs_error_mm'])
    body_errs = valid_3d[~valid_3d['joint'].isin([0,1,2,3,4])]
    out_lines.append(f"- **3D MPJPE (Body, Raw):** {body_errs['abs_error_mm'].mean():.1f} mm mean / {body_errs['abs_error_mm'].median():.1f} mm median (N={len(body_errs)})")
    
    frames_surviving = df_e[~df_e['mpjpe_body_raw_mm'].isna()]
    frames_low_quality = df_e[df_e['low_quality_flag'] == True]
    out_lines.append(f"- **Frames with valid Triangulation:** {len(frames_surviving)}/1800 ({len(frames_surviving)/1800*100:.1f}%)")
    out_lines.append(f"- **Frames flagging Low Quality (<35 deg baseline):** {len(frames_low_quality)} ({len(frames_low_quality)/1800*100:.1f}%)")
    
    # 2. Combined Breakdown Table
    out_lines.append("\\n## 2. Combined Breakdown Table\\n")
    
    # Bin distance: 0-50, 50-100, 100-150, 150-200, 200+
    bins_dist = [0, 50, 100, 150, 200, 500]
    labels_dist = ['0-50cm', '50-100cm', '100-150cm', '150-200cm', '200+cm']
    df_e['dist_bin'] = pd.cut(df_e['dist_from_center_cm'], bins=bins_dist, labels=labels_dist)
    
    # Bin baseline: 10 deg increments
    bins_ang = range(0, 130, 10)
    labels_ang = [f'{i}-{i+10}deg' for i in range(0, 120, 10)]
    df_e['baseline_bin'] = pd.cut(df_e['max_baseline_angle_deg'], bins=bins_ang, labels=labels_ang)
    
    # Camera pair identity
    # Reconstruct from df_c which cameras contributed to each frame
    pair_identities = []
    for f in df_e['frame']:
        c_rows = df_c[df_c['frame'] == f]
        valid_pairs = c_rows[c_rows['n_joints_contributed'] > 0]
        if len(valid_pairs) == 3:
            pair_identities.append('Full Trio')
        elif len(valid_pairs) == 1:
            row = valid_pairs.iloc[0]
            pair_identities.append(f"{row['cam_a']} x {row['cam_b']}")
        else:
            pair_identities.append('None/Unknown')
    df_e['pair_identity'] = pair_identities
    
    # We will compute the mean MPJPE per combined group (baseline_bin, pair_identity, dist_bin)
    # The user asked for "All four axes in a single table". We don't have orientation. I will omit orientation for now, or just leave it blank.
    grouped = df_e.groupby(['pair_identity', 'baseline_bin', 'dist_bin'])['mpjpe_body_raw_mm'].agg(['count', 'mean']).reset_index()
    grouped = grouped[grouped['count'] > 0]
    
    out_lines.append("| Camera Pair | Baseline Angle | Dist from Center | Mean MPJPE (mm) | N Frames |")
    out_lines.append("|---|---|---|---|---|")
    for _, row in grouped.iterrows():
        out_lines.append(f"| {row['pair_identity']} | {row['baseline_bin']} | {row['dist_bin']} | {row['mean']:.1f} | {int(row['count'])} |")
        
    out_path = ROOT / "outputs" / "phase3_analysis.md"
    with open(out_path, 'w') as f:
        f.write("\\n".join(out_lines))
        
    print(f"Analysis saved to {out_path}")

if __name__ == '__main__':
    main()
