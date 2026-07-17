# Spine Twist Root-Cause Analysis

## Symptom

The upper torso (chest) twists backward. The lower back around the hips is
fine, but above that the whole torso rotates as if the character is turning
their back to the camera. The twist accumulates up the spine chain — worse at
the chest than at the lower back.

## Root Cause: Three Compounding Problems

### Problem 1 — The spine init distributes the WRONG rotation

**File:** `aimocap/retarget/mocap_ik.py:187-203` (`analytic_init`)

The pelvis is set to `R_root` (the hip frame) at line 156:
```python
global_rot[0] = R_root
```

Then each spine joint multiplies `R_frac = R_root^(1/6)` **on top of** the
parent's already-`R_root` frame:
```python
# line 192-202
frac = 1.0 / (n_spine - 1)          # 1/6
R_frac = Rotation.from_rotvec(R_root.as_rotvec() * frac)
local_quats[p] = (global_rot[p_parent].inv() * R_frac * global_rot[p_parent]).as_quat()
global_rot[p] = R_frac * global_rot[p_parent]
```

This produces globals:
```
pelvis   = R_root^1.000
spine_01 = R_root^1.167   (over-rotated +20°)
spine_02 = R_root^1.333   (over-rotated +40°)
spine_03 = R_root^1.500   (over-rotated +60°)
spine_04 = R_root^1.667   (over-rotated +80°)
spine_05 = R_root^1.833   (over-rotated +83°)
neck_01  = R_upper         (set directly, NOT R_root^2.0)
```

The intended "distribute the bend evenly" only works if the pelvis starts at
**identity**. Because the pelvis starts at `R_root`, every spine segment stacks
an extra `R_root^(1/6)` on top of an already-`R_root` parent. The upper spine
is over-rotated by up to `R_root^(5/6)` ≈ **83°** at spine_05.

### Problem 2 — The shoulder-vs-hip yaw is dumped entirely into neck_01

**File:** `aimocap/retarget/mocap_ik.py:180-185`

The actual upper-body rotation relative to the pelvis is:
```
R_diff = R_upper · R_root⁻¹
```
This is the rotation that SHOULD be distributed across the spine (the torso
twist between hips and shoulders). But the code distributes `R_root` instead
(Problem 1), and sets `neck_01 = R_upper` directly:
```python
if p == neck_idx:
    R_local_neck = R_parent.inv() * R_upper    # line 182
    global_rot[p] = R_upper                     # line 184
```

So the entire `R_diff` (≈37° in a typical pose, up to 100°+ in a twisted pose)
lands as a single local twist at neck_01. The spine doesn't distribute it at
all — it gets it in one shot at the neck.

### Problem 3 — The IK solver cannot correct the twist (position-only residual)

**File:** `aimocap/retarget/mocap_ik.py:250-353` (`_residuals`)

The solver's residual is **position-only** (line 346):
```python
res_data = (global_pos - tgt_pos) * w[:, None]
```

There is **no orientation or roll constraint** anywhere in the residual. The
only other terms are:
- Temporal smoothing (smooth toward previous frame — line 349-350)
- Init regularizer with `init_weight=1e-3` (tiny pull toward `analytic_init` —
  line 351-352)

**Why this matters:** Rotation of a spine bone about its own long axis is a
**null direction** of the position Jacobian — it moves no joint position. The
synthesized spine intermediate targets (from `distribute_spine_targets`) are
**collinear** along the chord from pelvis to neck. Twisting any spine bone
about that chord moves zero joint positions to first order. So `least_squares`
cannot observe or correct spine roll at all. It stays at whatever
`analytic_init` set it to — which is the over-rotated value from Problem 1.

**Contrast with limbs:** Limb bones avoid this because `analytic_init` calls
`constrained_rotation` with a `roll_child` (lines 221-234). The roll child
(e.g., foot pins calf, hand pins upperarm) is NOT collinear with the bone,
so twisting the bone about its long axis WOULD move the roll child. This
constrains the roll. The spine branch (lines 187-203) does NOT call
`constrained_rotation` and does NOT use a roll child — even though
`RigTopology.roll_child()` returns valid children for spine bones (each
spine bone's roll child is the next spine bone).

## The Fundamental Issue

This is a classic **under-constrained inverse kinematics** problem. Given
only joint positions (point data), you can determine the **swing** (bone
direction) but not the **twist** (rotation about the bone's long axis). This
is a gauge freedom — a null space in the position Jacobian.

For the spine specifically:
- The 5 intermediate spine joints (spine_01 through spine_05) are
  **synthesized** (not measured) by `distribute_spine_targets`, and their
  targets are **collinear** along the pelvis→neck chord.
- Only pelvis and neck_01 are measured (from COCO keypoints 11/12 and 5/6).
- The IK weight for spine intermediates is 0.15 (low — line 342).
- There is no orientation measurement for any spine bone.

So the twist DOF for every spine bone is completely unobservable from the
input data, and the solver is free to set it to anything. The init sets it to
the over-rotated value, and the solver has no reason to change it.

## The Generalized Fix (Premium Feature)

The fix must address all three problems. Here is the most general approach,
designed to work on any humanoid rig (no hardcoded Manny indices):

### Fix 1: Distribute the correct rotation in `analytic_init`

Replace the `R_root` distribution with SLERP from `R_root` to `R_upper`:

```python
# Instead of: R_frac = R_root ** (1/(n-1))
# Use:        R_diff = R_upper * R_root.inv()
#             R_spine_k = R_root * R_diff ** (k / (n-1))

R_diff = R_upper * R_root.inv()
for k in range(1, n_spine):
    frac = k / (n_spine - 1)
    R_spine_k = R_root * (R_diff ** frac)
    global_rot[spine_joint_k] = R_spine_k
```

This smoothly interpolates from `R_root` at the pelvis to `R_upper` at the
neck, distributing only the **difference** (the actual torso twist) — not
re-applying the entire root rotation. The over-rotation vanishes.

### Fix 2: Pin the spine roll using `constrained_rotation` (like limbs)

The infrastructure already exists. `RigTopology.roll_child()` returns the
next spine bone for each spine joint. Use `constrained_rotation` in the
spine branch, exactly like the limb branch does:

```python
# Instead of the R_frac conjugation, use:
rest_off = self.skel.rest_offsets[child]           # spine_k → spine_{k+1}
desired_world = target_pos[child] - target_pos[spine_k]
desired_pl = global_rot[parent].inv().apply(desired_world)

rc_name = self.skel.topo.roll_child(rig_name_child)
# ... same roll-child logic as limbs (lines 221-234) ...

R_local = constrained_rotation(rest_off, desired_pl, roll_rest, roll_des)
```

This gives the **minimal-rotation** solution (zero excess twist) for
collinear targets, and a meaningful roll for the arc-curve case. It uses the
exact same mechanism as limbs — no spine-specific code path.

### Fix 3: Add an orientation residual to the solver

Even with a correct init, the solver can drift the twist because the
position residual has a null direction along the bone's long axis. Add a
weak orientation residual that penalizes deviation from the init's
per-bone orientation:

```python
# In _residuals, after the position term:
# Compute current global orientations from FK
_, global_rot = self.forward_kinematics(root_t, local_rotations)
# Target orientations from analytic_init (computed once, passed in)
ori_residual = log_map(global_rot[spine_joints] * target_rot[spine_joints].inv())
parts.append(ori_residual.flatten() * ori_weight)  # ori_weight ~ 0.1
```

This doesn't hard-pin the orientation (the solver can still adjust it to fit
positions), but it eliminates the null direction — the solver can no longer
add arbitrary twist for free.

### Why this is general

- **No hardcoded indices**: Uses `RigTopology.spine_chain()` and
  `roll_child()` which work on any rig.
- **No Manny-specific assumptions**: The SLERP interpolation and
  `constrained_rotation` are rig-agnostic.
- **Graceful degradation**: If the rig has a 1-bone spine (pelvis→neck
  directly), `R_diff` is applied at the single joint — no over-rotation.
- **Works with partial data**: If shoulder keypoints are NaN, `R_upper`
  falls back to `R_root`, and `R_diff` = identity → no spine twist (safe
  default).

## Verification

After the fix, check:
1. `spine_05` global rotation should be ≈ `R_root * R_diff^(5/6)`, not
   `R_root^(11/6)`. The over-rotation should be near zero.
2. `neck_01` local rotation should be small (the last fraction of R_diff,
   not the entire R_diff dumped at once).
3. The backward twist in the render should be gone — the chest faces the
   same direction as the hips, with only the actual shoulder-vs-hip twist
   distributed smoothly.
4. No regression on limbs (they use a separate code path).
5. Temporal stability (no jitter from the orientation residual).
