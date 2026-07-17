"""
Visualization: IK Solver Truth (GT vs Solved)
==============================================
Shows ground-truth vs IK-solved skeleton on all 3 camera angles + a 3D view,
over the most-moved 15-second window of the real Panoptic dataset.

Design decisions (addressing reviewer feedback):
  1. SYNC: data local index j corresponds to video frame j + OFFSET (150).
     Verified empirically by cross-correlating the person's horizontal
     centroid in the video against the projected-GT centroid: correlation
     peaks at D=150 (0.97) vs 0.62 at D=0. So we seek video to start+OFFSET.
  2. QUALITY: each camera view is cropped/zoomed to the person (fixed square
     window, re-centered per frame) so the skeleton fills the panel, then
     encoded at high resolution via ffmpeg/libx264.
  3. CLARITY: only two skeletons are drawn — GT (green) and Solved (red) —
     in both the 2D overlays and the 3D view. The 3D view is centered on the
     person each frame. Triangulated-input error is reported in the console,
     not drawn, to keep the picture readable.

Coordinate systems:
  - gate1_arrays.npz['gt']:        mm, internal (Y-up, Z-backward)
  - Manny FK:                      cm, Manny (Z-up, X-forward, Y-right)
  - internal → Panoptic:           (X, -Y, -Z)
  - Manny → internal:              (X, Z, -Y)
  - Manny → Panoptic:              (X, -Z, Y)
  - gate1 → Panoptic:              (X/10, -Y/10, -Z/10)
"""
import sys
import cv2
import subprocess
import tempfile
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(r"E:\Chaos\Projects\aimocap_re")
sys.path.insert(0, str(ROOT))

from aimocap.data.panoptic import load_calibration
from aimocap.retarget.mocap_skeleton import MocapSkeleton, extract_mocap_points
from aimocap.retarget.mocap_ik import MocapIKSolver
from aimocap.retarget.fbx_rig import Skeleton

# ── Constants ──────────────────────────────────────────────────────────────────
SEQ = "171204_pose1"
CAMS = ["00_11", "00_12", "00_23"]
OFFSET = 150                 # data index j -> video frame j + OFFSET
FPS = 30
WINDOW_FRAMES = 450          # 15 seconds at 30 fps
SUBSAMPLE = 3                # 30 fps -> 10 fps for output
OUT_FPS = 10
IMG_W, IMG_H = 1920, 1080    # full camera resolution
CAM_PANEL = 420              # per-camera square panel (px)
CANVAS_W = CAM_PANEL * 3     # 1260
THREE_D_W = CANVAS_W
THREE_D_H = 640
GIF_MAX_W = 720              # downscale GIF for size; MP4 stays full res

# COCO-17 bone connections + colors for a clean, readable skeleton
BONES_COCO = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (0, 5), (0, 6),                        # neck to shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 11), (6, 12),                      # torso sides
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (11, 12),                      # shoulder/hip lines
]

# Manny bones to draw (matched to a COCO-like body for clarity)
PREVIEW_BONES = {
    "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05",
    "neck_01", "neck_02", "head",
    "clavicle_l", "upperarm_l", "lowerarm_l", "hand_l",
    "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
    "thigh_l", "calf_l", "foot_l", "ball_l",
    "thigh_r", "calf_r", "foot_r", "ball_r",
}

# Colors (BGR for OpenCV 2D, RGB for matplotlib 3D)
GT_BGR = (0, 200, 0)         # green
SOLVED_BGR = (0, 0, 255)     # red
GT_RGB = "#00c800"
SOLVED_RGB = "#e00000"


# ── Coordinate conversions ─────────────────────────────────────────────────────
def manny_to_panoptic(pts):
    out = np.zeros_like(pts, dtype=np.float64)
    out[..., 0] = pts[..., 0]
    out[..., 1] = -pts[..., 2]
    out[..., 2] = pts[..., 1]
    return out


def manny_to_internal(pts):
    out = np.zeros_like(pts, dtype=np.float64)
    out[..., 0] = pts[..., 0]
    out[..., 1] = pts[..., 2]
    out[..., 2] = -pts[..., 1]
    return out


def gate1_to_panoptic(pts_mm):
    out = np.zeros_like(pts_mm, dtype=np.float64)
    out[..., 0] = pts_mm[..., 0] / 10.0
    out[..., 1] = -pts_mm[..., 1] / 10.0
    out[..., 2] = -pts_mm[..., 2] / 10.0
    return out


# ── Window selection ───────────────────────────────────────────────────────────
def find_most_moved_window(gt_mm, window=WINDOW_FRAMES):
    valid = ~np.isnan(gt_mm[..., 0])
    diff = np.diff(gt_mm, axis=0)
    disp = np.linalg.norm(np.nan_to_num(diff), axis=-1)
    disp[~valid[1:]] = 0
    total_motion = disp.sum(axis=-1)
    window_motion = np.convolve(total_motion, np.ones(window), mode="valid")
    best_start = int(np.argmax(window_motion))
    return best_start, best_start + window


# ── IK solve (unchanged pipeline) ───────────────────────────────────────────────
def solve_ik_window(b_stage6_cm, start, end):
    pts3d_raw = b_stage6_cm[start:end].copy()  # already cm, internal

    # Y-up (internal) -> Z-up (Manny): [x, y, z] -> [x, -z, y]
    pts3d = np.zeros_like(pts3d_raw)
    pts3d[..., 0] = pts3d_raw[..., 0]
    pts3d[..., 1] = -pts3d_raw[..., 2]
    pts3d[..., 2] = pts3d_raw[..., 1]

    weights = np.any(np.isfinite(pts3d), axis=-1).astype(np.float32)
    pts3d = np.nan_to_num(pts3d, nan=0.0)

    fbx_skel = Skeleton(str(ROOT / "Manny.FBX"))
    mocap_skel = MocapSkeleton(pts3d, weights, fbx_skel=fbx_skel)
    mocap_ik = MocapIKSolver(mocap_skel)
    mocap_target_pts = extract_mocap_points(pts3d)

    num_frames = len(pts3d)
    solved_root, solved_local_quats = [], []
    prev_x = None
    for f in range(num_frames):
        if f % 50 == 0:
            print(f"    IK solving frame {f}/{num_frames}")
        measured = {k: v[f] for k, v in mocap_target_pts.items()}
        x_opt = mocap_ik.solve_frame(measured, prev_x=prev_x, temporal_weight=0.03)
        prev_x = x_opt
        root_t, local_quats = mocap_ik._state_to_local_rotations(x_opt)
        solved_root.append(root_t)
        solved_local_quats.append(local_quats)

    solved_root = np.array(solved_root)
    solved_local_quats = np.array(solved_local_quats)

    fbx_rest_global_pos, fbx_rest_global_rot = fbx_skel.get_forward_kinematics()
    fbx_rest_global = Rotation.from_quat(fbx_rest_global_rot)
    fbx_rest_local = Rotation.from_quat(np.array(fbx_skel.rest_rotations))

    mocap_to_fbx_idx = {
        mocap_i: fbx_skel.name_to_idx[name]
        for mocap_i, name in mocap_skel.fbx_mapping.items()
    }
    fbx_to_mocap_idx = {fbx_i: m for m, fbx_i in mocap_to_fbx_idx.items()}

    fbx_root_idx = [i for i, p in enumerate(fbx_skel.parents) if p == -1][0]
    pelvis_idx = fbx_skel.name_to_idx["pelvis"]
    num_joints = fbx_skel.num_joints
    manny_pos = np.zeros((num_frames, num_joints, 3))

    for f in range(num_frames):
        if f % 50 == 0:
            print(f"    Rotation transfer frame {f}/{num_frames}")
        _, mocap_global_quats = mocap_ik.forward_kinematics(
            solved_root[f], solved_local_quats[f]
        )
        mocap_global = Rotation.from_quat(mocap_global_quats)

        frame_global = [None] * num_joints
        frame_local = [None] * num_joints
        for fbx_i in range(num_joints):
            p = fbx_skel.parents[fbx_i]
            if fbx_i in fbx_to_mocap_idx:
                mocap_i = fbx_to_mocap_idx[fbx_i]
                R_global_target = mocap_global[mocap_i] * fbx_rest_global[fbx_i]
            elif p == -1:
                R_global_target = fbx_rest_global[fbx_i]
            else:
                R_global_target = frame_global[p] * fbx_rest_local[fbx_i]
            R_local = (R_global_target if p == -1
                       else frame_global[p].inv() * R_global_target)
            frame_global[fbx_i] = R_global_target
            frame_local[fbx_i] = R_local

        root_world = (
            solved_root[f]
            - frame_global[fbx_root_idx].apply(
                fbx_skel.rest_translations[pelvis_idx]
            )
        )
        root_t = root_world - fbx_skel.rest_translations[fbx_root_idx]
        frame_local_quats = np.array([r.as_quat() for r in frame_local])
        local_eulers = Rotation.from_quat(frame_local_quats).as_euler("xyz")
        local_rot = Rotation.from_euler("xyz", local_eulers).as_quat()
        pos, _ = fbx_skel.get_forward_kinematics(local_rot, root_translation=root_t)
        manny_pos[f] = pos

    preview_set = {
        i for i, name in enumerate(fbx_skel.node_names) if name in PREVIEW_BONES
    }
    connections_manny = [
        (fbx_skel.parents[i], i)
        for i in range(num_joints)
        if fbx_skel.parents[i] != -1
        and i in preview_set and fbx_skel.parents[i] in preview_set
    ]
    return manny_pos, connections_manny, fbx_skel


# ── Projection / 2D drawing ─────────────────────────────────────────────────────
def project_pts(pts3d_cm, K, R, t, dist):
    valid = np.all(np.isfinite(pts3d_cm), axis=-1)
    out = np.full((len(pts3d_cm), 2), np.nan)
    if not valid.any():
        return out
    proj, _ = cv2.projectPoints(
        pts3d_cm[valid].astype(np.float64),
        R.astype(np.float64), t.astype(np.float64).reshape(3, 1),
        K.astype(np.float64), dist.astype(np.float64),
    )
    out[valid] = proj.reshape(-1, 2)
    return out


def draw_skeleton_2d(img, pts2d, bones, color, joint_radius, bone_thickness):
    for j1, j2 in bones:
        if j1 < len(pts2d) and j2 < len(pts2d):
            if np.all(np.isfinite(pts2d[[j1, j2]])):
                p1 = (int(pts2d[j1, 0]), int(pts2d[j1, 1]))
                p2 = (int(pts2d[j2, 0]), int(pts2d[j2, 1]))
                cv2.line(img, p1, p2, color, bone_thickness, cv2.LINE_AA)
    for j in range(len(pts2d)):
        if np.all(np.isfinite(pts2d[j])):
            cv2.circle(img, (int(pts2d[j, 0]), int(pts2d[j, 1])),
                       joint_radius, color, -1, cv2.LINE_AA)


def crop_and_resize(frame, center_xy, half_size, out_size):
    """Crop a square window centered on the person, then resize to panel size."""
    cx, cy = int(center_xy[0]), int(center_xy[1])
    hs = int(half_size)
    x0, y0 = cx - hs, cy - hs
    x1, y1 = cx + hs, cy + hs
    # Pad if crop extends beyond the frame
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - IMG_W); pad_b = max(0, y1 - IMG_H)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(IMG_W, x1), min(IMG_H, y1)
    crop = frame[y0c:y1c, x0c:x1c]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_CONSTANT, value=(30, 30, 30))
    if crop.size == 0:
        crop = np.full((2 * hs, 2 * hs, 3), 30, np.uint8)
    return cv2.resize(crop, (out_size, out_size))


# ── 3D drawing ──────────────────────────────────────────────────────────────────
def draw_3d_panel(ax, gt_cm, solved_cm, solved_bones, frame_idx, center):
    """GT (green) vs Solved (red), centered on the person. internal Y-up cm.
    Display maps (X, Z, Y) so Y-up is vertical."""
    cx, cz, cy_up = center[0], center[2], center[1]
    R = 80  # cm half-range around the person horizontally
    ax.set_xlim(cx - R, cx + R)
    ax.set_ylim(cz - R, cz + R)
    ax.set_zlim(max(0, cy_up - 95), cy_up + 95)
    ax.set_box_aspect((1, 1, 1.2))
    ax.set_xlabel("X (cm)", fontsize=8, labelpad=-4)
    ax.set_ylabel("depth (cm)", fontsize=8, labelpad=-4)
    ax.set_zlabel("height (cm)", fontsize=8, labelpad=-4)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=8, azim=-70)
    ax.set_title(f"3D  —  GT (green)  vs  Solved (red)   |   frame {frame_idx}",
                 fontsize=12)

    for j1, j2 in BONES_COCO:
        if np.all(np.isfinite(gt_cm[[j1, j2]])):
            ax.plot([gt_cm[j1, 0], gt_cm[j2, 0]],
                    [gt_cm[j1, 2], gt_cm[j2, 2]],
                    [gt_cm[j1, 1], gt_cm[j2, 1]],
                    color=GT_RGB, alpha=0.9, linewidth=3, solid_capstyle="round")
    ax.plot([], [], [], color=GT_RGB, label="GT", linewidth=3)

    for p, c in solved_bones:
        if np.all(np.isfinite(solved_cm[[p, c]])):
            ax.plot([solved_cm[p, 0], solved_cm[c, 0]],
                    [solved_cm[p, 2], solved_cm[c, 2]],
                    [solved_cm[p, 1], solved_cm[c, 1]],
                    color=SOLVED_RGB, alpha=0.9, linewidth=3,
                    solid_capstyle="round")
    ax.plot([], [], [], color=SOLVED_RGB, label="Solved", linewidth=3)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=== Step 1: Finding most-moved window ===")
    g1 = np.load(ROOT / "outputs/phase_b_gate1/gate1_arrays.npz", allow_pickle=True)
    gt_mm = g1["gt"].copy()
    # Mask untracked markers: Panoptic sets position=(0,0,0) when conf=-1.
    # Without this, the viz draws them at the projection of origin, making
    # the GT skeleton fly apart during bends when the tracker loses legs.
    zero_mask = np.all(gt_mm == 0, axis=-1)
    gt_mm[zero_mask] = np.nan
    n_masked = int(zero_mask.sum())
    if n_masked:
        print(f"    Masked {n_masked} untracked GT markers (origin -> NaN)")
    # Build the PRODUCTION 3D input (filtered raw triangulation) — NOT the audit b_stage6.
    from aimocap.math.filter import filter_skeleton3d
    from scripts.measure_pipeline_accuracy import load_canonical_data, triangulate_raw
    _d = load_canonical_data()
    b_s6_cm = filter_skeleton3d(triangulate_raw(_d))  # (1800,17,3) cm internal
    print(f"    Production input: {b_s6_cm.shape}, "
          f"finite={np.mean(np.all(np.isfinite(b_s6_cm), axis=-1)):.1%}")
    start, end = find_most_moved_window(gt_mm)
    num_frames = end - start
    print(f"    Window: local frames {start}-{end-1} "
          f"(video frames {start+OFFSET}-{end+OFFSET-1})")

    print("\n=== Step 2: IK solver on window ===")
    cache_path = ROOT / "outputs" / "viz_ik_truth_cache.npz"
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=True)
        manny_pos = cache["manny_pos"]
        connections_manny = [tuple(c) for c in cache["connections"]]
        fbx_skel = Skeleton(str(ROOT / "Manny.FBX"))
        # Use the cached frame count to set the actual window end
        end = start + len(manny_pos)
        num_frames = end - start
        print(f"    Loaded {len(manny_pos)} frames from cache "
              f"(window {start}-{end-1})")
    else:
        manny_pos, connections_manny, fbx_skel = solve_ik_window(b_s6_cm, start, end)
        np.savez(cache_path, manny_pos=manny_pos,
                 connections=np.array(connections_manny))
        print(f"    Solved {len(manny_pos)} frames (cached)")

    print("\n=== Step 3: Coordinate conversions ===")
    gt_cm_window = gt_mm[start:end] / 10.0                 # internal cm
    solved_internal = manny_to_internal(manny_pos)          # internal cm
    gt_pan = gate1_to_panoptic(gt_mm[start:end])            # Panoptic cm
    solved_pan = manny_to_panoptic(manny_pos)               # Panoptic cm

    print("\n=== Step 4: Camera calibration ===")
    calib = load_calibration(ROOT / f"data/panoptic/{SEQ}/calibration_{SEQ}.json")
    cam_params = []
    for cn in CAMS:
        cam_params.append((
            calib[cn].K.astype(np.float64),
            calib[cn].R.astype(np.float64),
            calib[cn].t.astype(np.float64).reshape(3, 1),
            calib[cn].dist_coef.astype(np.float64),
        ))

    # Pre-pass: per-camera fixed crop half-size from GT bbox across the window.
    print("\n=== Step 5: Computing per-camera zoom ===")
    render_idx = list(range(0, num_frames, SUBSAMPLE))
    cam_halfsize = []
    for c in range(len(CAMS)):
        K, R, t, dist = cam_params[c]
        spans = []
        for i in render_idx:
            proj = project_pts(gt_pan[i], K, R, t, dist)
            v = np.all(np.isfinite(proj), axis=1)
            if v.sum() >= 4:
                p = proj[v]
                spans.append(max(p[:, 0].ptp(), p[:, 1].ptp()))
        # fixed square half-size = 0.75 * (p90 span) + margin, so the person fills
        hs = 0.5 * (np.percentile(spans, 90) if spans else 700) * 1.6
        hs = float(np.clip(hs, 250, 700))
        cam_halfsize.append(hs)
        print(f"    {CAMS[c]}: crop half-size = {hs:.0f}px")

    print("\n=== Step 6: Opening videos (seek to start+OFFSET) ===")
    vid_dir = ROOT / f"data/panoptic/{SEQ}/hdVideos"
    caps = [cv2.VideoCapture(str(vid_dir / f"hd_{cn}.mp4")) for cn in CAMS]
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + OFFSET)  # <-- SYNC FIX

    print("\n=== Step 7: Rendering ===")
    tmpdir = Path(tempfile.mkdtemp())
    fig_3d = plt.figure(figsize=(THREE_D_W / 100, THREE_D_H / 100), dpi=100)
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    fig_3d.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    out_count = 0

    for i in range(num_frames):
        vid_frames = []
        for cap in caps:
            ret, fr = cap.read()
            vid_frames.append(fr if ret
                              else np.zeros((IMG_H, IMG_W, 3), np.uint8))
        if i % SUBSAMPLE != 0:
            continue

        # 2D camera panels (draw on full frame, then crop+zoom to person)
        cam_canvases = []
        for c in range(len(CAMS)):
            fr = vid_frames[c].copy()
            K, R, t, dist = cam_params[c]

            gt_proj = project_pts(gt_pan[i], K, R, t, dist)
            solved_proj = project_pts(solved_pan[i], K, R, t, dist)
            draw_skeleton_2d(fr, gt_proj, BONES_COCO, GT_BGR,
                             joint_radius=7, bone_thickness=4)
            draw_skeleton_2d(fr, solved_proj, connections_manny, SOLVED_BGR,
                             joint_radius=5, bone_thickness=3)

            v = np.all(np.isfinite(gt_proj), axis=1)
            if v.sum() >= 4:
                center = gt_proj[v].mean(axis=0)
            else:
                center = np.array([IMG_W / 2, IMG_H / 2])
            panel = crop_and_resize(fr, center, cam_halfsize[c], CAM_PANEL)

            # header bar with camera name + legend swatches
            cv2.rectangle(panel, (0, 0), (CAM_PANEL, 26), (0, 0, 0), -1)
            cv2.putText(panel, f"cam {CAMS[c]}", (8, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.circle(panel, (CAM_PANEL - 150, 13), 6, GT_BGR, -1)
            cv2.putText(panel, "GT", (CAM_PANEL - 138, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_BGR, 1, cv2.LINE_AA)
            cv2.circle(panel, (CAM_PANEL - 95, 13), 6, SOLVED_BGR, -1)
            cv2.putText(panel, "Solved", (CAM_PANEL - 83, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, SOLVED_BGR, 1, cv2.LINE_AA)
            cam_canvases.append(panel)

        # 3D panel, centered on GT pelvis (mid-hip)
        hips = gt_cm_window[i][[11, 12]]
        pelvis = np.nanmean(hips, axis=0)
        if not np.all(np.isfinite(pelvis)):
            pelvis = np.array([0.0, 90.0, 0.0])
        ax_3d.clear()
        draw_3d_panel(ax_3d, gt_cm_window[i], solved_internal[i],
                      connections_manny, start + i, pelvis)
        fig_3d.canvas.draw()
        img_rgba = np.frombuffer(fig_3d.canvas.buffer_rgba(), dtype=np.uint8)
        wh = fig_3d.canvas.get_width_height()
        img_3d = img_rgba.reshape(wh[::-1] + (4,))[:, :, :3]
        img_3d = cv2.resize(img_3d, (CANVAS_W, THREE_D_H))

        top_bgr = np.hstack(cam_canvases)
        top_rgb = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)
        canvas = np.vstack([top_rgb, img_3d])
        Image.fromarray(canvas).save(str(tmpdir / f"frame_{out_count:04d}.png"))
        out_count += 1
        if out_count % 30 == 0:
            print(f"    Rendered {out_count} frames")

    for cap in caps:
        cap.release()
    plt.close(fig_3d)
    print(f"    Total {out_count} frames")

    print("\n=== Step 8: Encoding ===")
    out_dir = ROOT / "outputs"
    mp4_path = out_dir / "viz_ik_truth.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(OUT_FPS),
        "-i", str(tmpdir / "frame_%04d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(mp4_path),
    ], check=True, capture_output=True)
    print(f"    MP4: {mp4_path} ({mp4_path.stat().st_size/1e6:.1f} MB)")

    gif_path = out_dir / "viz_ik_truth.gif"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(OUT_FPS),
        "-i", str(tmpdir / "frame_%04d.png"),
        "-vf", f"scale={GIF_MAX_W}:-1:flags=lanczos,"
               "split[s0][s1];[s0]palettegen=max_colors=200[p];"
               "[s1][p]paletteuse=dither=sierra2_4a",
        str(gif_path),
    ], check=True, capture_output=True)
    print(f"    GIF: {gif_path} ({gif_path.stat().st_size/1e6:.1f} MB)")
    shutil.rmtree(tmpdir)
    print(f"    {out_count} frames @ {OUT_FPS} fps = {out_count/OUT_FPS:.1f} s")

    # ── Report: MPJPE and triangulated-input error (console, not drawn) ──────────
    print("\n=== Step 9: Numbers ===")
    manny_to_coco = {
        "head": 0, "upperarm_l": 5, "upperarm_r": 6, "lowerarm_l": 7,
        "lowerarm_r": 8, "hand_l": 9, "hand_r": 10, "thigh_l": 11,
        "thigh_r": 12, "calf_l": 13, "calf_r": 14, "foot_l": 15, "foot_r": 16,
    }
    # Align each frame's solved skeleton to GT pelvis before measuring shape error
    solved_err, tri_err = [], []
    for f in range(num_frames):
        gt_f = gt_cm_window[f]
        sol_f = solved_internal[f]
        tri_f = b_s6_cm[start + f]
        gt_pelvis = np.nanmean(gt_f[[11, 12]], axis=0)
        sol_pelvis = np.nanmean(
            sol_f[[fbx_skel.name_to_idx["thigh_l"],
                   fbx_skel.name_to_idx["thigh_r"]]], axis=0)
        if not (np.all(np.isfinite(gt_pelvis)) and np.all(np.isfinite(sol_pelvis))):
            continue
        shift = gt_pelvis - sol_pelvis
        for mn, ci in manny_to_coco.items():
            mi = fbx_skel.name_to_idx[mn]
            if np.all(np.isfinite(gt_f[ci])) and np.all(np.isfinite(sol_f[mi])):
                solved_err.append(np.linalg.norm(gt_f[ci] - (sol_f[mi] + shift)))
        for ci in range(17):
            if np.all(np.isfinite(gt_f[ci])) and np.all(np.isfinite(tri_f[ci])):
                tri_err.append(np.linalg.norm(gt_f[ci] - tri_f[ci]))

    def stats(a, label):
        a = np.array(a)
        print(f"    {label}: mean={a.mean():.2f} median={np.median(a):.2f} "
              f"p95={np.percentile(a,95):.2f} max={a.max():.2f} cm  (n={len(a)})")
    if solved_err:
        stats(solved_err, "Solved vs GT (pelvis-aligned)")
    if tri_err:
        stats(tri_err, "Production input (filtered raw triang) vs GT")


if __name__ == "__main__":
    main()
