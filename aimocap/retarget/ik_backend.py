"""IK backend dispatcher — selects the fastest available solver.

Priority:
  1. PyTorch CUDA (GPU, ~100x faster) — if torch.cuda.is_available()
  2. Scipy LM with precompute (CPU, ~2x faster than original) — always available
  3. (Future: Numba JIT CPU fallback)

The backend is selected automatically.  Override with env var:
  AIMOCAP_IK_BACKEND=torch|scipy|auto  (default: auto)
"""
from __future__ import annotations

import os
import numpy as np

# ── Backend selection ────────────────────────────────────────────────────

_BACKEND_OVERRIDE = os.environ.get("AIMOCAP_IK_BACKEND", "auto").lower()
_BACKEND_NAME = None  # set on first dispatch


def _try_torch_cuda():
    """Check if torch CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


_HAS_CUDA = _try_torch_cuda()


def dispatch_solve_frame(solver, ctx: dict, x0: np.ndarray) -> np.ndarray:
    """Select the best available backend and solve."""
    global _BACKEND_NAME

    backend = _BACKEND_OVERRIDE
    if backend == "auto":
        if _HAS_CUDA:
            backend = "torch"
        else:
            backend = "scipy"

    if backend == "torch" and _HAS_CUDA:
        _BACKEND_NAME = "torch-cuda"
        try:
            return _solve_torch(solver, ctx, x0)
        except Exception as e:
            import warnings
            warnings.warn(
                f"torch backend failed ({e}), falling back to scipy")
            _BACKEND_NAME = "scipy-fallback"
            return _solve_scipy_precomputed(solver, ctx, x0)
    else:
        _BACKEND_NAME = "scipy"
        return _solve_scipy_precomputed(solver, ctx, x0)


def get_backend_name() -> str:
    """Return the name of the backend used on the last solve."""
    return _BACKEND_NAME or "not-yet-dispatched"


# ── Scipy path (with precompute — always available) ─────────────────────

def _solve_scipy_precomputed(solver, ctx: dict, x0: np.ndarray) -> np.ndarray:
    """Scipy LM using precomputed frame context (faster than original)."""
    from scipy.optimize import least_squares
    res = least_squares(
        solver._residuals_with_ctx, x0,
        args=(ctx,),
        method='lm', max_nfev=500, ftol=1e-6, xtol=1e-6, gtol=1e-6,
    )
    return res.x


# ── Torch CUDA path ──────────────────────────────────────────────────────

def _solve_torch(solver, ctx: dict, x0: np.ndarray) -> np.ndarray:
    """GPU-accelerated solve using PyTorch CUDA."""
    import torch
    from aimocap.retarget.ik_torch import (
        _residuals_torch, _fk_torch, _rotvec_to_quat, _quat_to_rotvec,
        _quat_mul, _quat_inv,
    )

    device = torch.device("cuda")
    # FP64 is used for production accuracy.  Benchmarking showed FP64 and
    # FP32 per-FK-call times are identical on the RTX 4060 for our batch
    # size (76×75) — the Ada Lovelace FP64 throttle doesn't apply to small
    # tensors.  Set AIMOCAP_IK_PRECISION=fp32 for rapid iteration (slightly
    # fewer iterations due to larger eps, but causes temporal divergence
    # over 300+ frames: 101mm MPJPE → 210mm).
    _precision = os.environ.get("AIMOCAP_IK_PRECISION", "fp64").lower()
    dtype = torch.float32 if _precision == "fp32" else torch.float64
    num_joints = ctx["num_joints"]
    num_vars = 3 + num_joints * 3

    # ── Build torch tensors from ctx (move to GPU) ───────────────────
    def to_t(arr):
        if arr is None:
            return None
        return torch.tensor(np.ascontiguousarray(arr), device=device, dtype=dtype)

    # Build levels flat array (for the FK loop)
    levels_flat = []
    level_starts = [0]
    for lvl in solver.levels:
        levels_flat.extend(lvl.tolist())
        level_starts.append(len(levels_flat))
    levels_flat_t = torch.tensor(levels_flat, device=device, dtype=torch.long)
    level_starts_t = torch.tensor(level_starts, device=device, dtype=torch.long)

    torch_ctx = {
        "num_joints": num_joints,
        "rest_t0": to_t(ctx["rest_t0"]),
        "rest_offsets": to_t(ctx["rest_offsets"]),
        "parents": torch.tensor(solver.parents, device=device, dtype=torch.long),
        "levels_flat": levels_flat_t,
        "level_starts": level_starts_t,
        "tgt_pos": to_t(ctx["tgt_pos"]),
        "w": to_t(ctx["w"]),
        "R_target_quats": to_t(ctx["R_target_quats"]),
        "spine_indices": torch.tensor(ctx["spine_indices"], device=device, dtype=torch.long),
        "n_spine": ctx["n_spine"],
        "ori_weight": ctx["ori_weight"],
        "head_ori_active": ctx["head_ori_active"],
        "head_idx": ctx["head_idx"],
        "neck_idx": ctx["neck_idx"],
        "pelvis_idx": ctx["pelvis_idx"],
        "head_rest_dir": to_t(ctx["head_rest_dir"]) if ctx["head_rest_dir"] is not None else None,
        "nose": to_t(ctx["nose"]) if ctx["nose"] is not None else None,
        # Frame-based head orientation target (precomputed per frame)
        "head_target_quat": to_t(ctx["head_target_quat"]) if ctx.get("head_target_quat") is not None else None,
        # Kabsch head orientation data (disabled)
        "use_kabsch": ctx.get("use_kabsch", False),
        "canonical_head": to_t(ctx["canonical_head"]) if "canonical_head" in ctx else None,
        "face_kpts_obs": None,  # set below if Kabsch is active
        "face_kpts_template": None,  # set below
        "lat_axis": to_t(ctx["lat_axis"]),
        "prev_lat": ctx["prev_lat"],
        "prev_x": to_t(ctx["prev_x"]).unsqueeze(0) if ctx["prev_x"] is not None else None,
        "prev_prev_x": to_t(ctx["prev_prev_x"]).unsqueeze(0) if ctx.get("prev_prev_x") is not None else None,
        "accel_weight": ctx.get("accel_weight", 0.0),
        "init_x": to_t(ctx["init_x"]).unsqueeze(0) if ctx["init_x"] is not None else None,
        "temporal_weight": ctx["temporal_weight"],
        "init_weight": ctx["init_weight"],
        "limb_pinned_info": None,  # handled below
    }

    # Convert limb pinned info to torch tensors
    if ctx["limb_pinned_info"]:
        limb_infos = ctx["limb_pinned_info"]
        torch_ctx["limb_pinned_info"] = [{
            "p": info["p"],
            "c": info["c"],
            "bone": to_t(info["bone"]),
            "p_parent": info["p_parent"],
            "twist_target": info.get("twist_target", 0.0),
            "twist_weight": info.get("twist_weight", 1.0),
            "side": info["side"] if "side" in info else None,
        } for info in limb_infos]
    else:
        torch_ctx["limb_pinned_info"] = []

    # Convert wrist targets to torch tensors (legacy)
    wrist_targets = ctx.get("wrist_targets", {})
    if wrist_targets:
        torch_ctx["wrist_targets"] = {wi: to_t(q) for wi, q in wrist_targets.items()}
    else:
        torch_ctx["wrist_targets"] = {}

    # ── Build Kabsch head rotation ─────────────────────────────────────
    # When use_kabsch is True, we have ≥3 face keypoints.  Compute the
    # Kabsch rotation ONCE here (it depends only on measured data, not on
    # the optimization state), convert to a quaternion, and pass that into
    # the residual.  The residual then only needs to compare the current
    # head rotation to this static target — no per-iteration SVD.
    torch_ctx["kabsch_quat"] = None
    if torch_ctx.get("use_kabsch") and torch_ctx.get("canonical_head") is not None:
        from aimocap.math.kabsch import compute_head_rotation_kabsch
        face_kpts = ctx.get("face_kpts", {})
        neck_pos = ctx["tgt_pos"][ctx["neck_idx"]]  # (3,) numpy
        R_kabsch = compute_head_rotation_kabsch(face_kpts, neck_pos)
        if R_kabsch is not None:
            # R_kabsch maps template → observed, i.e. it's the head's
            # global orientation relative to the neck frame.
            torch_ctx["kabsch_quat"] = to_t(
                R_kabsch.as_quat().astype(np.float32))  # (4,) xyzw
        else:
            torch_ctx["use_kabsch"] = False  # not enough non-collinear points

    # ── Run LM loop on GPU ───────────────────────────────────────────
    x = torch.tensor(x0, device=device, dtype=dtype).unsqueeze(0)  # (1, num_vars)
    x = _lm_loop_torch(x, torch_ctx, num_vars)
    return x.squeeze(0).cpu().numpy()


def _lm_loop_torch(
    x: "torch.Tensor",
    ctx: dict,
    num_vars: int,
    max_iter: int = 200,
    ftol: float = 1e-5,
    xtol: float = 1e-5,
    lambda_init: float = 1e-3,
) -> "torch.Tensor":
    """Levenberg-Marquardt loop on GPU with batched finite-diff Jacobian.

    The key optimization: instead of 75 sequential residual evals for the
    Jacobian (like scipy does), we batch all 75 perturbations into ONE GPU
    call — (76, 75) tensor, where row 0 is the unperturbed state and rows
    1..75 are x + eps*e_i.  This turns 75x GPU overhead into 1x.

    Benchmarking showed FP64 and FP32 have identical per-call speed on the
    RTX 4060 for our batch size (76×75) — the Ada Lovelace FP64 throttle
    doesn't apply to small tensors.  FP64 is used for temporal accuracy
    (prevents MPJPE divergence over 300+ frames).
    """
    import torch
    from aimocap.retarget.ik_torch import _residuals_torch

    device = x.device
    dtype = x.dtype
    lam = lambda_init
    eps = 1e-5 if dtype == torch.float32 else 1e-7

    # Pre-build the perturbation matrix: (num_vars+1, num_vars)
    I_aug = torch.eye(num_vars, device=device, dtype=dtype)
    I_aug = torch.cat([torch.zeros(1, num_vars, device=device, dtype=dtype), I_aug])

    # Initial cost
    with torch.no_grad():
        r0 = _residuals_torch(x, ctx)
        cost = (r0 ** 2).sum().item()

    for iteration in range(max_iter):
        # ── Batched Jacobian: 1 GPU call for all 75 perturbations ──────
        with torch.no_grad():
            x_batch = x.expand(num_vars + 1, -1) + eps * I_aug  # (76, 75)
            r_batch = _residuals_torch(x_batch, ctx)             # (76, M)
            r0 = r_batch[0:1, :]  # (1, M)
            r_pert = r_batch[1:, :]  # (75, M)
            J = ((r_pert - r0) / eps).T  # (M, 75)

        # LM normal equations: (J^T J + lam * diag(JtJ)) dx = -J^T r
        JtJ = J.T @ J                         # (num_vars, num_vars)
        Jtr = J.T @ r0[0]                     # (num_vars,)
        diag_JtJ = torch.diag(JtJ)
        diag_JtJ = torch.where(diag_JtJ > 1e-10, diag_JtJ, torch.ones_like(diag_JtJ))

        # Try step with increasing lambda
        improved = False
        for _inner in range(20):
            A = JtJ + lam * torch.diag(diag_JtJ)
            try:
                dx = torch.linalg.solve(A, -Jtr)  # (num_vars,)
            except RuntimeError:
                lam *= 10.0
                continue

            x_new = x + dx.unsqueeze(0)
            with torch.no_grad():
                r_new = _residuals_torch(x_new, ctx)
                cost_new = (r_new ** 2).sum().item()

            if cost_new < cost:
                step_norm = dx.norm().item()
                x = x_new
                # ── Swing-twist twist clamping on hinge joints ─────────────
                # After each accepted LM step, decompose each constrained
                # bone's local rotation into swing + twist, clamp the twist
                # to anatomical limits, and reconstruct.  This prevents
                # caved-in knees, spinning hands, and twisted forearms.
                if ctx.get("twist_limits"):
                    import numpy as _np
                    from scipy.spatial.transform import Rotation as _Rot
                    from aimocap.retarget.swing_twist import clamp_twist
                    x_np = x[0].cpu().numpy()
                    num_j = ctx["num_joints"]
                    changed = False
                    for jname, max_tw in ctx["twist_limits"].items():
                        ji = ctx.get("twist_joint_indices", {}).get(jname)
                        if ji is None:
                            continue
                        bone_axis = ctx["twist_bone_axes"].get(jname)
                        if bone_axis is None:
                            continue
                        rv = x_np[3 + ji*3 : 3 + ji*3 + 3]
                        if _np.linalg.norm(rv) < 1e-8:
                            continue
                        q = _Rot.from_rotvec(rv)
                        q_clamped = clamp_twist(q, bone_axis, max_tw)
                        rv_new = q_clamped.as_rotvec()
                        if _np.linalg.norm(rv_new - rv) > 1e-6:
                            x_np[3 + ji*3 : 3 + ji*3 + 3] = rv_new
                            changed = True
                    if changed:
                        x = torch.tensor(x_np, device=device, dtype=dtype).unsqueeze(0)
                rel_change = abs(cost - cost_new) / max(cost, 1e-10)
                if rel_change < ftol:
                    return x
                cost = cost_new
                lam = max(lam * 0.3, 1e-12)
                improved = True
                break
            else:
                lam *= 10.0

        if not improved:
            return x  # converged or stuck
        if step_norm < xtol:
            return x

    return x
