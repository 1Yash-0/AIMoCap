"""aimocap command-line interface.

Entry point: ``aimocap <subcommand> [options]`` (installed via pyproject.toml).

Current subcommands:
    pose <video> [-o out.mp4] [--max-frames N] [--threshold T]
        Run 2D whole-body pose estimation over a video, write annotated output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

import cv2

# Importing aimocap top-level ensures CUDA DLLs load before onnxruntime.
import aimocap  # noqa: F401
from aimocap.pose.infer import PoseEstimator
from aimocap.video import run_video, summarize
from aimocap.pose.multi_video import run_multi_video
from aimocap.calib.intrinsics import guess_intrinsics
from aimocap.calib.extrinsics import calibrate_all, align_to_floor
from aimocap.calib.io import save_calibration, load_calibration
from aimocap.calib.scale import apply_metric_scale
from aimocap.triangulate.engine import triangulate_sequence_with_diagnostics
from aimocap.math.filter import filter_skeleton3d
from aimocap.viz.plot3d import plot_scene
from aimocap.retarget.engine import retarget_to_fbx


def _cmd_pose(args: argparse.Namespace) -> int:
    src = Path(args.video)
    if not src.exists():
        print(f"error: input video not found: {src}", file=sys.stderr)
        return 2

    if args.output:
        dst = Path(args.output)
    else:
        dst = src.with_name(f"{src.stem}_annotated.mp4")
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input : {src}")
    print(f"Output: {dst}")
    print(f"Threshold: {args.threshold}  max_frames: {args.max_frames or 'all'}")

    # Warm up the estimator so the first frame isn't paying for CUDA init
    # in the timing.
    kwargs = {}
    if args.pose_model:
        kwargs["pose_model"] = args.pose_model
    est = PoseEstimator(**kwargs)
    print(f"Providers: {est.active_providers}")

    stats = run_video(
        src,
        dst,
        estimator=est,
        threshold=args.threshold,
        max_frames=args.max_frames,
        pick=args.pick,
        progress_every=args.progress_every,
    )

    summary = summarize(stats)
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k:20s} {v}")

    # Stability gate: a good single-person run should have a high detection
    # rate and low body-confidence variance (no flicker). Print a verdict.
    det_rate = summary.get("detection_rate", 0)
    conf_std = summary.get("body_conf_std", 0)
    print("\n=== stability verdict ===")
    print(f"  detection_rate = {det_rate:.2%}  (want > 0.95 for a clear subject)")
    print(f"  body_conf_std  = {conf_std:.3f}     (want < 0.10 for no flicker)")
    return 0


def _cmd_multipose(args: argparse.Namespace) -> int:
    sources = [Path(s) for s in args.videos]
    for src in sources:
        if not src.exists():
            print(f"error: input video not found: {src}", file=sys.stderr)
            return 2
            
    npz_dst = Path(args.output)
    grid_dst = Path(args.grid_video) if args.grid_video else None

    print(f"Inputs : {[str(s) for s in sources]}")
    print(f"Output : {npz_dst}")
    if grid_dst:
        print(f"Grid   : {grid_dst}")
        
    kwargs = {}
    if args.pose_model:
        kwargs["pose_model"] = args.pose_model
    est = PoseEstimator(**kwargs)
    print(f"Providers: {est.active_providers}")
    
    run_multi_video(
        sources,
        npz_dst,
        grid_dst=grid_dst,
        estimator=est,
        threshold=args.threshold,
        max_frames=args.max_frames,
        progress_every=args.progress_every,
    )
    return 0


def _cmd_calib_auto(args: argparse.Namespace) -> int:
    npz_src = Path(args.npz)
    if not npz_src.exists():
        print(f"error: input npz not found: {npz_src}", file=sys.stderr)
        return 2
        
    print(f"Loading keypoints from {npz_src}")
    data = np.load(npz_src)
    keypoints = data["keypoints"]  # (F, C, 133, 2)
    scores = data["scores"]        # (F, C, 133)
    
    num_frames, num_cameras, _, _ = keypoints.shape
    print(f"Found {num_frames} frames, {num_cameras} cameras.")
    
    if "image_sizes" in data:
        img_sizes = data["image_sizes"]
    else:
        img_sizes = np.array([[1920, 1080]] * num_cameras)
        
    from aimocap.calib.intrinsics import guess_intrinsics, extract_intrinsics
    from aimocap.calib.focal_search import focal_grid_search
    
    K_list = []
    has_guess = False
    for i in range(num_cameras):
        w, h = img_sizes[i]
        
        if args.videos and i < len(args.videos):
            K, method = extract_intrinsics(args.videos[i], w, h)
            print(f"Cam {i}: Intrinsics via {method} -> focal ~ {K[0,0]:.1f}")
            if method == "guess":
                has_guess = True
        else:
            K = guess_intrinsics(w, h)
            print(f"Cam {i}: Intrinsics via guess -> focal ~ {K[0,0]:.1f}")
            has_guess = True
        K_list.append(K)
        
    if has_guess:
        print("\nSome cameras lack EXIF focal length data. Running focal grid search...")
        try:
            K_list, extrinsics, best_ratio = focal_grid_search(
                keypoints, scores, img_sizes, min_conf=args.threshold, steps=10, stride=5
            )
        except Exception as e:
            print(f"Focal grid search failed: {e}. Falling back to initial guess.")
            print("Solving extrinsics...")
            extrinsics = calibrate_all(keypoints, scores, K_list, min_conf=args.threshold)
            print("Aligning to floor...")
            extrinsics = align_to_floor(extrinsics, K_list, keypoints, scores, min_conf=args.threshold)
    else:
        print("Solving extrinsics...")
        try:
            extrinsics = calibrate_all(keypoints, scores, K_list, min_conf=args.threshold)
            print("Aligning to floor...")
            extrinsics = align_to_floor(extrinsics, K_list, keypoints, scores, min_conf=args.threshold)
        except Exception as e:
            print(f"error during calibration: {e}", file=sys.stderr)
            return 1
        
    out_path = Path(args.output)
    save_calibration(out_path, K_list, extrinsics, image_sizes=[(int(s[0]), int(s[1])) for s in img_sizes])
    print(f"Saved calibration to {out_path}")
    return 0


def _cmd_triangulate(args: argparse.Namespace) -> int:
    npz_src = Path(args.npz)
    calib_src = Path(args.calib)
    
    if not npz_src.exists():
        print(f"error: input npz not found: {npz_src}", file=sys.stderr)
        return 2
    if not calib_src.exists():
        print(f"error: calibration json not found: {calib_src}", file=sys.stderr)
        return 2
        
    print(f"Loading keypoints from {npz_src}")
    data = np.load(npz_src)
    keypoints = data["keypoints"]
    scores = data["scores"]
    
    if args.filter_2d:
        print("Applying Savitzky-Golay filter to 2D keypoints before triangulation...")
        from scipy.signal import savgol_filter
        
        # 1. Interpolate missing points (scores < threshold) to prevent savgol overshoot
        F, C, J, _ = keypoints.shape
        for c in range(C):
            for j in range(J):
                valid = scores[:, c, j] >= args.threshold
                if np.any(valid) and not np.all(valid):
                    # Linearly interpolate X and Y
                    valid_idx = np.where(valid)[0]
                    invalid_idx = np.where(~valid)[0]
                    keypoints[invalid_idx, c, j, 0] = np.interp(invalid_idx, valid_idx, keypoints[valid_idx, c, j, 0])
                    keypoints[invalid_idx, c, j, 1] = np.interp(invalid_idx, valid_idx, keypoints[valid_idx, c, j, 1])
                    
        # 2. Apply filter
        window = 15
        poly = 3
        if keypoints.shape[0] > window:
            keypoints = savgol_filter(keypoints, window_length=window, polyorder=poly, axis=0)
        else:
            print(f"Warning: Sequence length ({keypoints.shape[0]}) too short for 2D filtering (requires > {window}).")
            
    if "camera_names" in data:
        camera_names = [str(x) for x in data["camera_names"]]
        print(f"Using camera mapping: {camera_names}")
    else:
        camera_names = None
        
    print(f"Loading cameras from {calib_src}")
    K_list, extrinsics = load_calibration(calib_src, camera_names=camera_names)
    
    print("Triangulating sequence...")
    tri = triangulate_sequence_with_diagnostics(
        keypoints,
        scores,
        K_list,
        extrinsics,
        min_conf=args.threshold,
        reproj_threshold_px=args.reproj_threshold,
    )
    skeleton3d = tri.points3d
    finite_reproj = tri.reprojection_error_px[np.isfinite(tri.reprojection_error_px)]
    if finite_reproj.size:
        print(
            "Triangulation reprojection error: "
            f"median={np.median(finite_reproj):.2f}px "
            f"p90={np.percentile(finite_reproj, 90):.2f}px "
            f"p95={np.percentile(finite_reproj, 95):.2f}px"
        )
    supported = tri.num_inliers[tri.num_inliers > 0]
    mean_inliers = float(np.mean(supported)) if supported.size else 0.0
    print(
        "Triangulation support: "
        f"mean_inliers={mean_inliers:.2f} "
        f"valid_points={np.count_nonzero(tri.num_inliers >= 2)}/{tri.num_inliers.size}"
    )
    if hasattr(tri, 'low_quality_mask') and tri.low_quality_mask is not None:
        num_low_quality = np.sum(tri.low_quality_mask)
        if num_low_quality > 0:
            print(f"WARNING: Detected {num_low_quality} frames with extremely narrow baseline (< 35 deg). These frames may have unstable depth.")
    
    print("Applying anthropometric scale...")
    skeleton3d, _, scale_factor = apply_metric_scale(skeleton3d, extrinsics)
    print(f"Computed metric scale factor: {scale_factor:.3f}")
    
    from aimocap.math.kinematics import compute_median_bone_lengths, fit_skeleton_sequence, forward_kinematics_sequence, CLUSTER_DEFS
    from aimocap.math.filter import filter_params_one_euro_quaternion
    from aimocap.math.clusters import reject_cluster_outliers_anchor_distance, anchor_cluster_bone_constrained, filter_cluster_relative, fill_cluster_gaps
    
    # Extract 17 body joints for Kinematics
    body3d = skeleton3d[:, :17, :]
    
    print("Computing median bone lengths...")
    bone_lengths = compute_median_bone_lengths(body3d)
    
    body3d_rigid, params_seq = fit_skeleton_sequence(body3d, bone_lengths)
    
    print("Applying One-Euro temporal filter...")
    params_seq_filtered = filter_params_one_euro_quaternion(params_seq, fps=30.0)
    
    print("Reconstructing 3D body from filtered parameters...")
    body3d_final = forward_kinematics_sequence(params_seq_filtered, bone_lengths)
    body3d_final_17 = body3d_final[:, :17, :]
    
    print("Processing non-body clusters (hands, face, feet)...")
    cleaned_3d = reject_cluster_outliers_anchor_distance(skeleton3d, body3d_final_17, CLUSTER_DEFS)
    anchored_3d = anchor_cluster_bone_constrained(cleaned_3d, body3d_final_17, CLUSTER_DEFS)
    filtered_clusters = filter_cluster_relative(anchored_3d, body3d_final_17, CLUSTER_DEFS, fps=30.0)
    final_clusters = fill_cluster_gaps(filtered_clusters, body3d_final_17, CLUSTER_DEFS, max_gap=15)
    
    skeleton3d_final = np.copy(final_clusters)
    skeleton3d_final[:, :17, :] = body3d_final_17
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "skeleton3d": skeleton3d_final,
        "scale_factor": scale_factor,
        "confidence": tri.confidence,
        "triangulation_inlier_mask": tri.inlier_mask,
        "triangulation_reprojection_error_px": tri.reprojection_error_px,
        "triangulation_num_inliers": tri.num_inliers,
    }
    if camera_names is not None:
        save_dict["camera_names"] = np.array(camera_names, dtype=str)
        
    np.savez(out_path, **save_dict)
    print(f"Saved 3D sequence to {out_path} with shape {skeleton3d_final.shape}")
    
    return 0


def _cmd_viz(args: argparse.Namespace) -> int:
    print(f"Loading 3D skeleton from {args.npz}")
    data = np.load(args.npz)
    skeleton3d = data["skeleton3d"]
    
    if args.skip_frames > 0:
        print(f"Skipping first {args.skip_frames} frames...")
        skeleton3d = skeleton3d[args.skip_frames:]
    
    camera_names = None
    if "camera_names" in data:
        camera_names = [str(x) for x in data["camera_names"]]
        print(f"Using camera mapping: {camera_names}")
        
    print(f"Loading cameras from {args.cameras}")
    K_list, extrinsics = load_calibration(args.cameras, camera_names=camera_names)
    
    if "scale_factor" in data:
        scale_factor = float(data["scale_factor"])
        print(f"Cameras scaled by loaded factor: {scale_factor:.3f}")
        extrinsics = [(R, t * scale_factor) for R, t in extrinsics]
    
    out_path = Path(args.output)
    print(f"Rendering to {out_path}...")
    from aimocap.viz.plot3d import plot_scene
    plot_scene(extrinsics, skeleton3d, out_path, animate=True)
    print("Done rendering.")
    return 0


def _cmd_retarget(args: argparse.Namespace) -> int:
    npz_src = Path(args.npz)
    fbx_src = Path(args.fbx)
    out_path = Path(args.output)
    
    if not npz_src.exists():
        print(f"error: input npz not found: {npz_src}", file=sys.stderr)
        return 2
    if not fbx_src.exists():
        print(f"error: FBX rig not found: {fbx_src}", file=sys.stderr)
        return 2
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    retarget_to_fbx(
        str(npz_src),
        str(fbx_src),
        str(out_path),
        fps=args.fps,
        mocap_target_stabilization=not args.disable_mocap_target_stabilization,
        mocap_target_cutoff_hz=args.mocap_target_cutoff,
        leg_stabilization=not args.disable_leg_stabilization,
        leg_stabilize_cutoff_hz=args.leg_stabilize_cutoff,
        leg_max_deg_per_frame=args.leg_max_deg_per_frame,
        foot_lock=not args.disable_foot_lock,
        foot_lock_debug_dir=args.foot_lock_debug_dir,
        pose2d_npz=args.pose2d_npz,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aimocap",
        description="Markerless multi-camera motion capture (Python core).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pose", help="Run 2D pose estimation over a video.")
    pp.add_argument("video", help="Path to input video (mp4/avi/mov/gif).")
    pp.add_argument("-o", "--output", help="Output video path (default: <name>_annotated.mp4).")
    pp.add_argument("--max-frames", type=int, default=None,
                    help="Stop after N frames (default: whole video).")
    pp.add_argument("--threshold", type=float, default=0.3,
                    help="Min keypoint confidence to draw (default: 0.3).")
    pp.add_argument("--pick", choices=["largest", "all"], default="largest",
                    help="Person selection each frame (default: largest).")
    pp.add_argument("--progress-every", type=int, default=10,
                    help="Print progress every N frames.")
    pp.add_argument("--pose-model", help="Path to CIGPose ONNX model (optional).")
    pp.set_defaults(func=_cmd_pose)

    mp = sub.add_parser("multipose", help="Run 2D pose estimation synchronously over multiple videos.")
    mp.add_argument("videos", nargs="+", help="Paths to input videos.")
    mp.add_argument("-o", "--output", required=True, help="Output .npz file path.")
    mp.add_argument("--grid-video", help="Optional output grid video path.")
    mp.add_argument("--max-frames", type=int, default=None,
                    help="Stop after N frames (default: whole video).")
    mp.add_argument("--threshold", type=float, default=0.3,
                    help="Min keypoint confidence to draw (default: 0.3).")
    mp.add_argument("--progress-every", type=int, default=10,
                    help="Print progress every N frames.")
    mp.add_argument("--pose-model", help="Path to CIGPose ONNX model (optional).")
    mp.set_defaults(func=_cmd_multipose)

    cp = sub.add_parser("calib", help="Camera calibration commands.")
    c_sub = cp.add_subparsers(dest="calib_cmd", required=True)
    
    ca = c_sub.add_parser("auto", help="Auto-calibrate from multi_pose2d.npz.")
    ca.add_argument("npz", help="Path to multi_pose2d.npz.")
    ca.add_argument("-o", "--output", required=True, help="Output extrinsics.json path.")
    ca.add_argument("--videos", nargs="*", help="Original videos to extract EXIF focal length.")
    ca.add_argument("--threshold", type=float, default=0.5,
                    help="Min keypoint confidence to use for calibration (default: 0.5).")
    ca.set_defaults(func=_cmd_calib_auto)

    tp = sub.add_parser("triangulate", help="Triangulate 2D keypoints into 3D using calibrated cameras.")
    tp.add_argument("npz", help="Path to multi_pose2d.npz.")
    tp.add_argument("calib", help="Path to extrinsics.json.")
    tp.add_argument("-o", "--output", required=True, help="Output skeleton3d.npz path.")
    tp.add_argument("--threshold", type=float, default=0.5,
                    help="Min keypoint confidence to triangulate (default: 0.5).")
    tp.add_argument("--reproj-threshold", type=float, default=25.0,
                    help="Pixel reprojection threshold for triangulation inlier diagnostics (default: 25).")
    tp.add_argument("--filter-2d", action="store_true",
                    help="Apply Savitzky-Golay temporal filter to 2D keypoints prior to 3D lifting.")
    tp.set_defaults(func=_cmd_triangulate)

    vp = sub.add_parser("viz", help="Visualize triangulated 3D skeleton.")
    vp.add_argument("npz", help="Path to skeleton3d.npz.")
    vp.add_argument("--cameras", required=True, help="Path to extrinsics.json.")
    vp.add_argument("-o", "--output", required=True, help="Output gif/mp4 path.")
    vp.add_argument("--skip-frames", type=int, default=0, help="Skip first N frames.")
    vp.set_defaults(func=_cmd_viz)

    rp = sub.add_parser("retarget", help="Retarget triangulated points to an FBX skeleton.")
    rp.add_argument("npz", help="Path to skeleton3d.npz.")
    rp.add_argument("fbx", help="Path to target FBX rig (e.g. Manny.FBX).")
    rp.add_argument("-o", "--output", required=True, help="Output .bvh path.")
    rp.add_argument("--fps", type=float, default=30.0, help="Output animation FPS (default: 30).")
    rp.add_argument(
        "--disable-mocap-target-stabilization",
        action="store_true",
        help="Disable lower-body target smoothing before proxy IK.",
    )
    rp.add_argument(
        "--mocap-target-cutoff",
        type=float,
        default=1.5,
        help="Lower-body proxy target smoothing cutoff in Hz (default: 1.5).",
    )
    rp.add_argument(
        "--disable-leg-stabilization",
        action="store_true",
        help="Disable target-space lower-body rotation stabilization.",
    )
    rp.add_argument(
        "--leg-stabilize-cutoff",
        type=float,
        default=1.5,
        help="Lower-body zero-phase rotation smoothing cutoff in Hz (default: 1.5).",
    )
    rp.add_argument(
        "--leg-max-deg-per-frame",
        type=float,
        default=30.0,
        help="Clamp lower-body one-frame rotation jumps after smoothing (default: 30).",
    )
    rp.add_argument(
        "--disable-foot-lock",
        action="store_true",
        help="Disable target-space IK foot locking.",
    )
    rp.add_argument(
        "--foot-lock-debug-dir",
        default=None,
        help="Optional directory for foot-lock contact and metric diagnostics.",
    )
    rp.add_argument(
        "--pose2d-npz",
        default=None,
        help="Optional multi-camera 2D pose NPZ for image-space foot contact detection.",
    )
    rp.set_defaults(func=_cmd_retarget)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
