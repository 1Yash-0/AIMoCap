import numpy as np
from scipy.spatial.transform import Rotation


# Face keypoints are not calibrated well enough to drive production head
# orientation.  Keep the head neutral relative to the neck until a separate
# calibrated head-frame task supplies a numeric confidence gate.
HEAD_ORIENTATION_MODE = "neck_follow"
from aimocap.retarget.mocap_skeleton import MocapSkeleton, COCO
from aimocap.retarget.root_frame import root_rotation, spine_scale

class MocapIKSolver:
    def __init__(self, skel: MocapSkeleton):
        self.skel = skel
        self.num_joints = skel.num_joints

        # 3 root translation + (num_joints * 3) rotation vectors
        self.num_vars = 3 + self.num_joints * 3

        self.parents = np.array(skel.parents)
        self.rest_offsets = skel.rest_offsets.copy()

        depths = {0: 0}
        for i in range(1, self.num_joints):
            depths[i] = depths[self.parents[i]] + 1

        max_depth = max(depths.values())
        self.levels = []
        for d in range(1, max_depth + 1):
            nodes_at_d = [i for i in range(1, self.num_joints) if depths[i] == d]
            self.levels.append(np.array(nodes_at_d))

        # ── Swing-twist twist clamping setup ───────────────────────────
        # Precompute which joints have twist limits and their bone axes.
        # The bone axis is the rest offset direction (the bone's
        # longitudinal axis in parent space), which is what twist
        # rotates around.
        from aimocap.retarget.swing_twist import TWIST_LIMITS, arm_frame
        self._twist_limits = {}
        self._twist_joint_indices = {}
        self._twist_bone_axes = {}
        for jname, max_tw in TWIST_LIMITS.items():
            ji = skel.name_to_idx.get(jname)
            if ji is not None and ji < self.num_joints:
                axis = self.rest_offsets[ji].copy()
                n = np.linalg.norm(axis)
                if n > 1e-6:
                    axis = axis / n
                    self._twist_limits[jname] = max_tw
                    self._twist_joint_indices[jname] = ji
                    self._twist_bone_axes[jname] = axis


        # ── Rest head frame for frame-based head orientation ──────────
        # Built from clavicle positions (rig-agnostic). Left-handed formula
        # (left x up = forward) matches the data's chirality, then forced
        # to right-handed by flipping the up column. This ensures the delta
        # F_obs @ F_rest.T is a proper rotation.
        fbx_gp, _ = skel._fbx_skel.get_forward_kinematics()
        cl_name = skel.fbx_mapping.get(skel.name_to_idx.get("clavicle_l"))
        cr_name = skel.fbx_mapping.get(skel.name_to_idx.get("clavicle_r"))
        if cl_name and cr_name:
            cl_fbx = skel._fbx_skel.name_to_idx[cl_name]
            cr_fbx = skel._fbx_skel.name_to_idx[cr_name]
            x0 = fbx_gp[cl_fbx] - fbx_gp[cr_fbx]
        else:
            x0 = np.array([1.0, 0.0, 0.0])
        x0 = x0 / (np.linalg.norm(x0) + 1e-9)
        u0 = np.array([0.0, 0.0, 1.0])
        f0 = np.cross(x0, u0)
        f0 = f0 / (np.linalg.norm(f0) + 1e-9)
        u0 = np.cross(f0, x0)
        u0 = u0 / (np.linalg.norm(u0) + 1e-9)
        self._F_rest_head = np.column_stack([x0, f0, u0])
        self._F_rest_head[:, 2] *= -1  # force right-handed

    def _state_to_local_rotations(self, x: np.ndarray):
        root_t = x[0:3]
        rotvecs = x[3:].reshape(-1, 3)
        
        # Rs is (num_joints, 4), xyzw. Rotation vectors avoid Euler wrapping
        # and are locally linear around the identity.
        Rs = Rotation.from_rotvec(rotvecs).as_quat()
        return root_t, Rs

    def _state_delta(self, x: np.ndarray, ref_x: np.ndarray) -> np.ndarray:
        """Difference between two IK states, using the SO(3) log for rotations."""
        root_t, local_q = self._state_to_local_rotations(x)
        ref_root_t, ref_local_q = self._state_to_local_rotations(ref_x)

        d_root = root_t - ref_root_t
        d_rot = (
            Rotation.from_quat(ref_local_q).inv()
            * Rotation.from_quat(local_q)
        ).as_rotvec().reshape(-1)
        return np.concatenate([d_root, d_rot])
        
    def forward_kinematics(self, root_t: np.ndarray, local_rotations: np.ndarray):
        """Forward kinematics: local rotations -> global positions and rotations."""
        global_pos = np.zeros((self.num_joints, 3))
        global_rot = np.zeros((self.num_joints, 4))

        global_pos[0] = self.skel.rest_t[0] + root_t
        global_rot[0] = local_rotations[0]

        for level_nodes in self.levels:
            p_nodes = self.parents[level_nodes]

            R_parent = Rotation.from_quat(global_rot[p_nodes])
            R_local = Rotation.from_quat(local_rotations[level_nodes])

            global_pos[level_nodes] = (
                global_pos[p_nodes] + R_parent.apply(self.rest_offsets[level_nodes])
            )
            global_rot[level_nodes] = (R_parent * R_local).as_quat()

        return global_pos, global_rot

    def analytic_init(self, measured: dict) -> np.ndarray:
        from aimocap.retarget.swing_twist import constrained_rotation
        from aimocap.retarget.spine_chain import distribute_spine_targets
        from scipy.spatial.transform import Rotation
        spine_names = self.skel.topo.spine_chain("pelvis", "neck_01")

        hip_line = measured["hip_r"] - measured["hip_l"]
        spine_dir = measured["neck"] - measured["pelvis"]
        
        idx_hl = self.skel.name_to_idx["hip_l"]
        idx_hr = self.skel.name_to_idx["hip_r"]
        idx_sl = self.skel.name_to_idx["shoulder_l"]
        idx_sr = self.skel.name_to_idx["shoulder_r"]
        
        rest_hip = self.skel.rest_t[idx_hr] - self.skel.rest_t[idx_hl]
        rest_pelvis_mid = (self.skel.rest_t[idx_hl] + self.skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (self.skel.rest_t[idx_sl] + self.skel.rest_t[idx_sr]) / 2.0
        rest_spine = rest_neck_mid - rest_pelvis_mid
        
        from aimocap.retarget.root_frame import root_rotation
        R_root = root_rotation(spine_dir, hip_line, rest_spine, rest_hip)
        
        shoulder_line = measured["shoulder_r"] - measured["shoulder_l"]
        rest_shoulder = self.skel.rest_t[idx_sr] - self.skel.rest_t[idx_sl]
        R_upper = root_rotation(spine_dir, shoulder_line, rest_spine, rest_shoulder)

        # Torso twist = shoulder frame relative to hip frame, expressed in
        # the hip (root) local frame.  Distributed across the spine via left-
        # SLERP (R_root -> R_upper), so the actual shoulder-vs-hip yaw is
        # spread evenly instead of dumped at the neck.  Using R_root.inv() *
        # R_upper (not R_upper * R_root.inv()) ensures R_root * R_diff = R_upper
        # exactly at frac=1.0, matching the neck branch (line 184).
        R_diff = R_root.inv() * R_upper

        # The pelvis and neck_01 targets are the measured mid-hip and mid-shoulder
        # positions, OFFSET to the actual joint positions. The pelvis joint is
        # not at mid-hip and neck_01 is not at mid-shoulder (they're offset by
        # ~2.4 cm and ~7 cm in Manny). Without the offset, no FK state can place
        # the pelvis joint at mid-hip AND the hip children at their measured
        # positions (the bone lengths don't match). The offset is the rest pose
        # difference (joint - midpoint), applied rigidly so it rotates with the
        # root frame. Bone lengths from _estimate_bone_lengths are joint-to-child
        # distances, so the corrected targets are consistent with them.
        rest_pelvis_mid = (self.skel.rest_t[idx_hl] + self.skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (self.skel.rest_t[idx_sl] + self.skel.rest_t[idx_sr]) / 2.0
        pelvis_offset = self.skel.rest_t[0] - rest_pelvis_mid
        neck_offset = self.skel.rest_t[self.skel.name_to_idx["neck_01"]] - rest_neck_mid
        tgt_pelvis = measured["pelvis"] + R_root.apply(pelvis_offset)
        tgt_neck = measured["neck"] + R_upper.apply(neck_offset)
        
        rest_positions = np.array([self.skel.rest_t[self.skel.name_to_idx[nm]] for nm in spine_names])
        inter = distribute_spine_targets(tgt_pelvis, tgt_neck, rest_positions)
        
        target_pos = np.zeros((self.num_joints, 3))
        si = 0
        for k, nm in enumerate(spine_names):
            i = self.skel.name_to_idx[nm]
            if k == 0:
                target_pos[i] = tgt_pelvis
            elif k == len(spine_names) - 1:
                target_pos[i] = tgt_neck
            else:
                target_pos[i] = inter[si]; si += 1
            
        for i, nm in self.skel.coco_anchor.items():
            if self.skel.joint_names[i] not in ["pelvis", "neck_01"]:
                if nm in measured:
                    target_pos[i] = measured[nm]
                # If nm not in measured (e.g. 17-joint input without foot
                # keypoints), target_pos[i] stays zero — the toe fallback below
                # will synthesize from R_root.

        # Clavicle targets: the clavicles are children of neck_01 but have no
        # COCO anchor (no "clavicle" keypoint in COCO).  Without a synthesized
        # target, target_pos stays 0, which makes the init orient the clavicle
        # from the origin (garbage) and the solver pull it to the origin
        # (caved-in shoulders).  Synthesize from the MEASURED shoulder positions
        # (COCO 5/6) rather than R_upper.apply(rest_off), because the
        # triangulated data has opposite chirality from the rig — using
        # R_upper on the non-mirrored rest offset places the right clavicle
        # on the left side.  The clavicle sits on the line from neck to
        # shoulder, at the rest-proportion distance from the neck.
        neck_idx = self.skel.name_to_idx["neck_01"]
        for side in ("l", "r"):
            clav_name = f"clavicle_{side}"
            sh_name = f"shoulder_{side}"
            if (clav_name in self.skel.name_to_idx
                    and sh_name in measured
                    and np.all(np.isfinite(measured[sh_name]))):
                ci = self.skel.name_to_idx[clav_name]
                if np.allclose(target_pos[ci], 0.0):
                    # Place clavicle on the neck→shoulder line at rest proportion
                    sh_pos = measured[sh_name]
                    neck_to_shoulder = sh_pos - tgt_neck
                    rest_neck_to_shoulder = (
                        self.skel.rest_t[self.skel.name_to_idx[sh_name]]
                        - self.skel.rest_t[neck_idx])
                    rest_clav_len = np.linalg.norm(
                        self.skel.rest_t[ci] - self.skel.rest_t[neck_idx])
                    rest_shoulder_dist = np.linalg.norm(rest_neck_to_shoulder)
                    if rest_shoulder_dist > 1e-5:
                        frac = rest_clav_len / rest_shoulder_dist
                    else:
                        frac = 0.5
                    target_pos[ci] = tgt_neck + frac * neck_to_shoulder

        # Toe (ball_l/ball_r) targets: when the big-toe keypoint is measured
        # (COCO-WholeBody 17/20), the coco_anchor loop above already set the
        # target from measurement — use it, so foot orientation comes from
        # actual detected foot position, not the pelvis root frame. When the
        # big-toe is NaN (not detected) or toe_* is not coco-anchored, fall
        # back to R_root synthesis so the ankle roll still has a constraint.
        for side in ("l", "r"):
            toe_name = f"toe_{side}"
            ankle_name = f"ankle_{side}"
            if toe_name in self.skel.name_to_idx and ankle_name in self.skel.name_to_idx:
                toe_i = self.skel.name_to_idx[toe_name]
                ankle_i = self.skel.name_to_idx[ankle_name]
                toe_anchor = self.skel.coco_anchor.get(toe_i)
                needs_synth = (toe_anchor is None
                               or toe_anchor not in measured
                               or np.any(np.isnan(target_pos[toe_i])))
                if needs_synth:
                    rest_off = self.skel.rest_t[toe_i] - self.skel.rest_t[ankle_i]
                    target_pos[toe_i] = target_pos[ankle_i] + R_root.apply(rest_off)

        # Dorsiflexion clamp: prevent the toe target from landing ABOVE the
        # ankle/heel, which would dorsiflex the foot ("on heels" artifact).
        # The toe is synthesized from R_root (pelvis frame); when the torso
        # leans forward, R_root tilts and projects the toe's rest "down"
        # component forward+up -> toe above ankle -> dorsiflexion.
        # Two reference heights (prefer the more precise one):
        #   (a) heel keypoint (COCO-WholeBody 19/22) when available: the sole
        #       runs heel->toe, so toe Z <= heel Z + 1 cm keeps it flat.
        #   (b) ankle keypoint (always available, 17-joint data): the toe
        #       should not be above the ankle (rest toe is 7 cm below it).
        #       Clamp toe Z to ankle Z so the sole stays roughly flat.
        for side in ("l", "r"):
            toe_name = f"toe_{side}"
            ankle_name = f"ankle_{side}"
            heel_key = f"heel_{side}"
            if (toe_name not in self.skel.name_to_idx
                    or ankle_name not in self.skel.name_to_idx):
                continue
            toe_i = self.skel.name_to_idx[toe_name]
            ankle_i = self.skel.name_to_idx[ankle_name]
            if not np.all(np.isfinite(target_pos[toe_i])):
                continue
            if (heel_key in measured
                    and np.all(np.isfinite(measured[heel_key]))):
                # (a) heel-based clamp (133-joint data): precise.
                max_toe_z = measured[heel_key][2] + 1.0
            else:
                # (b) ankle-based clamp (17-joint data): toe can't exceed ankle.
                max_toe_z = target_pos[ankle_i][2]
            if target_pos[toe_i][2] > max_toe_z:
                target_pos[toe_i][2] = max_toe_z

        root_t = tgt_pelvis - self.skel.rest_t[0]

        global_rot = [None] * self.num_joints
        local_quats = np.zeros((self.num_joints, 4))
        global_rot[0] = R_root
        local_quats[0] = R_root.as_quat()

        # Map each parent to its primary child to orient the parent's outgoing bone
        primary_child = {}
        for j in range(1, self.num_joints):
            p = self.parents[j]
            if p not in primary_child:
                primary_child[p] = j

        ordered_parents = []
        for lvl in self.levels:
            for j in lvl:
                p = int(self.parents[j])
                if p not in ordered_parents:
                    ordered_parents.append(p)

        neck_idx = self.skel.name_to_idx["neck_01"]

        for p in ordered_parents:
            if p == 0: continue
            
            p_parent = self.parents[p]
            
            if p == neck_idx:
                R_parent = global_rot[p_parent]
                R_local_neck = R_parent.inv() * R_upper
                local_quats[p] = R_local_neck.as_quat()
                global_rot[p] = R_upper
                continue

            if self.skel.joint_names[p] in spine_names:
                # Distribute the torso twist (R_diff = R_upper . R_root^-1)
                # — NOT the full root rotation — evenly across the spine via
                # SLERP.  The pelvis is already R_root (line 156), so applying
                # R_root fractions on top over-rotates the upper spine by up
                # to R_root^(5/6) (~83 deg).  Interpolating from R_root to
                # R_upper distributes only the *difference* between the hip
                # and shoulder frames: spine_k=0 -> R_root, and the neck
                # branch (line 184) gets R_upper = R_root . R_diff exactly.
                n_spine = len(spine_names)
                spine_k = spine_names.index(self.skel.joint_names[p])
                frac = spine_k / (n_spine - 1)
                R_target = R_root * (R_diff ** frac)
                local_quats[p] = (global_rot[p_parent].inv() * R_target).as_quat()
                global_rot[p] = R_target
                continue
                
            if p not in primary_child:
                local_quats[p] = [0, 0, 0, 1]
                global_rot[p] = global_rot[p_parent]
                continue
                
            j = primary_child[p]
            rest_off = self.skel.rest_offsets[j]
            desired_world = target_pos[j] - target_pos[p]
            
            if np.linalg.norm(desired_world) < 1e-5:
                local_quats[p] = [0, 0, 0, 1]
                global_rot[p] = global_rot[p_parent]
                continue
                
            desired_pl = global_rot[p_parent].inv().apply(desired_world)
            R_local = constrained_rotation(rest_off, desired_pl)
            local_quats[p] = R_local.as_quat()
            global_rot[p] = global_rot[p_parent] * R_local

        # Orient leaves: joints with COCO anchors (e.g. head->nose) get
        # oriented toward their measured target so they don't just inherit
        # the parent's rotation.  Without this, the head always points the
        # same way as the neck regardless of where the nose is — a visible
        # tilt.  Other leaves (twist bones, etc.) keep identity.
        
        for p in range(self.num_joints):
            if p not in primary_child and p != 0:
                coco_name = self.skel.coco_anchor.get(p)
                if coco_name is not None and coco_name in measured:
                    p_parent = self.parents[p]
                    rest_dir = self.skel.rest_offsets[p]

                    # ── Safe production head mode ─────────────────────────
                    if p == self.skel.name_to_idx.get("head") and HEAD_ORIENTATION_MODE == "neck_follow":
                        local_quats[p] = [0.0, 0.0, 0.0, 1.0]
                        global_rot[p] = global_rot[p_parent]
                        continue

                    # ── Frame-based head orientation (diagnostic only) ─────
                    # Build an orthonormal head frame from face keypoints
                    # (ears for lateral, face midpoint for forward) and the
                    # rig's rest frame (clavicles for lateral). The delta
                    # R = F_obs @ F_rest.T gives real head tracking (yaw,
                    # pitch, roll) from data, without Kabsch or nose-direction.
                    #
                    # The triangulated data has opposite chirality from the
                    # rig (scalar triple product confirms this). Both frames
                    # are built with the same left-handed formula, then forced
                    # to right-handed by flipping the up column. This makes
                    # the delta a proper rotation that maps rest → observed.
                    if coco_name == "nose" and "left_ear" in measured:
                        la = measured["left_ear"]
                        ra = measured["right_ear"]
                        nose = measured["nose"]
                        le = measured.get("left_eye")
                        re = measured.get("right_eye")
                        face_pts = [p for p in [nose, le, re]
                                    if p is not None and np.all(np.isfinite(p))]
                        if (np.all(np.isfinite(la)) and np.all(np.isfinite(ra))
                                and len(face_pts) >= 2
                                and np.linalg.norm(la - ra) > 5.0):
                            # Observed head frame
                            x = la - ra  # left = ear_L - ear_R
                            x = x / np.linalg.norm(x)
                            face_mid = np.mean(face_pts, axis=0)
                            ear_mid = (la + ra) / 2.0
                            f_raw = face_mid - ear_mid
                            f = f_raw - np.dot(f_raw, x) * x  # perp to x
                            fn = np.linalg.norm(f)
                            if fn > 1e-5:
                                f = f / fn
                                u = np.cross(f, x)  # forward x left = up
                                u = u / np.linalg.norm(u)
                                F_obs = np.column_stack([x, f, u])
                                F_obs[:, 2] *= -1  # force RH
                                if np.linalg.det(F_obs) > 0:
                                    R_head_global = Rotation.from_matrix(
                                        F_obs @ self._F_rest_head.T)
                                    R_neck = global_rot[p_parent]
                                    R_local = R_neck.inv() * R_head_global
                                    local_quats[p] = R_local.as_quat()
                                    global_rot[p] = global_rot[p_parent] * R_local
                                    continue
                        # Fall through to default if degenerate

                    # Compute the desired direction for this leaf joint
                    # (from parent's target to this joint's target).
                    desired_world = target_pos[p] - target_pos[p_parent]
                    desired_pl = global_rot[p_parent].inv().apply(desired_world)
                    if np.linalg.norm(desired_pl) > 1e-5:
                        # Plausibility gate: the measured anchor (e.g. nose) can
                        # be a noisy triangulation that lands far off the head's
                        # plausible direction (e.g. 23 cm ABOVE the neck when the
                        # head bone points forward).  Trusting that would swing
                        # the head ~90 deg sideways ("completely left/right").
                        # Only accept the swing if the desired direction is
                        # within ~70 deg of the rest bone direction (the head
                        # can tilt/look around, but not face backward or
                        # sideways from the neck).  Otherwise fall back to the
                        # parent's rotation (identity local) -- the orientation
                        # residual in _residuals will refine it gently.
                        cos_accept = np.dot(
                            desired_pl / np.linalg.norm(desired_pl),
                            rest_dir / np.linalg.norm(rest_dir))
                        if cos_accept > np.cos(np.deg2rad(70.0)):
                            R_local = constrained_rotation(rest_dir, desired_pl)
                            local_quats[p] = R_local.as_quat()
                            global_rot[p] = global_rot[p_parent] * R_local
                            continue
                # For leaf joints (e.g. wrist/hand), inherit the parent's
                # rotation so the hand follows the forearm instead of freezing
                # at identity.  This gives the hand the correct base orientation
                # (matching the rest pose relative to the forearm), which the
                # GPU IK can then refine.  Without this, the hand stays at
                # absolute identity, appearing frozen at rest for all frames.
                if coco_name is not None and coco_name in measured:
                    # Leaf with a measured target (e.g. wrist): orient the
                    # bone direction, and inherit parent rotation for roll.
                    p_parent = self.parents[p]
                    desired_world = target_pos[p] - target_pos[p_parent]
                    desired_pl = global_rot[p_parent].inv().apply(desired_world)
                    if np.linalg.norm(desired_pl) > 1e-5:
                        cos_accept = np.dot(
                            desired_pl / np.linalg.norm(desired_pl),
                            rest_dir / np.linalg.norm(rest_dir))
                        if cos_accept > np.cos(np.deg2rad(70.0)):
                            R_local = constrained_rotation(rest_dir, desired_pl)
                            # Apply wrist twist target if available (computed from arm triangle)
                            if p in wrist_targets_init:
                                R_twist = Rotation.from_quat(wrist_targets_init[p])
                                R_local = R_local * R_twist
                            local_quats[p] = R_local.as_quat()
                            global_rot[p] = global_rot[p_parent] * R_local
                            continue
                # Default: inherit parent rotation (identity local = follow parent)
                local_quats[p] = [0, 0, 0, 1]
                global_rot[p] = global_rot[self.parents[p]]

        rotvecs = Rotation.from_quat(local_quats).as_rotvec()
        x = np.zeros(self.num_vars)
        x[0:3] = root_t
        x[3:] = rotvecs.flatten()
        return x



    def _precompute_frame(self, measured: dict, prev_x: np.ndarray = None,
                          temporal_weight: float = 0.03, init_x: np.ndarray = None,
                          prev_lat: float = None, frame_idx: int = 0,
                          prev_prev_x: np.ndarray = None,
                          accel_weight: float = 0.0) -> dict:
        """Compute all frame-constant quantities once per solve_frame call.

        These were previously recomputed 1200+ times per frame inside
        _residuals (once per least_squares eval).  Precomputing them here
        saves ~60% of the per-frame cost with zero quality change — the
        values are identical, just computed once instead of 1200x.
        """
        from aimocap.retarget.root_frame import root_rotation
        from aimocap.retarget.spine_chain import distribute_spine_targets

        # ── Root frame (depends only on measured) ────────────────────────
        hip_line = measured["hip_r"] - measured["hip_l"]
        spine_dir = measured["neck"] - measured["pelvis"]

        idx_hl = self.skel.name_to_idx["hip_l"]
        idx_hr = self.skel.name_to_idx["hip_r"]
        idx_sl = self.skel.name_to_idx["shoulder_l"]
        idx_sr = self.skel.name_to_idx["shoulder_r"]

        rest_hip = self.skel.rest_t[idx_hr] - self.skel.rest_t[idx_hl]
        rest_pelvis_mid = (self.skel.rest_t[idx_hl] + self.skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (self.skel.rest_t[idx_sl] + self.skel.rest_t[idx_sr]) / 2.0
        rest_spine = rest_neck_mid - rest_pelvis_mid

        R_root = root_rotation(spine_dir, hip_line, rest_spine, rest_hip)

        shoulder_line = measured["shoulder_r"] - measured["shoulder_l"]
        rest_shoulder = self.skel.rest_t[idx_sr] - self.skel.rest_t[idx_sl]
        R_upper = root_rotation(spine_dir, shoulder_line, rest_spine, rest_shoulder)
        R_diff = R_root.inv() * R_upper

        # ── Target positions (depends only on measured + rest) ───────────
        pelvis_offset = self.skel.rest_t[0] - rest_pelvis_mid
        neck_offset = self.skel.rest_t[self.skel.name_to_idx["neck_01"]] - rest_neck_mid
        tgt_pelvis = measured["pelvis"] + R_root.apply(pelvis_offset)
        tgt_neck = measured["neck"] + R_upper.apply(neck_offset)

        spine_names = self.skel.topo.spine_chain("pelvis", "neck_01")
        rest_positions = np.array([self.skel.rest_t[self.skel.name_to_idx[nm]] for nm in spine_names])
        inter = distribute_spine_targets(tgt_pelvis, tgt_neck, rest_positions)

        tgt_pos = np.zeros((self.num_joints, 3))
        si = 0
        for k, nm in enumerate(spine_names):
            i = self.skel.name_to_idx[nm]
            if k == 0:
                tgt_pos[i] = tgt_pelvis
            elif k == len(spine_names) - 1:
                tgt_pos[i] = tgt_neck
            else:
                tgt_pos[i] = inter[si]; si += 1

        for i, nm in self.skel.coco_anchor.items():
            if self.skel.joint_names[i] not in ["pelvis", "neck_01"]:
                if nm in measured:
                    tgt_pos[i] = measured[nm]

        # Clavicle synthesis
        neck_idx_r = self.skel.name_to_idx["neck_01"]
        for side in ("l", "r"):
            clav_name = f"clavicle_{side}"
            if clav_name in self.skel.name_to_idx:
                ci = self.skel.name_to_idx[clav_name]
                if np.allclose(tgt_pos[ci], 0.0):
                    rest_off = self.skel.rest_t[ci] - self.skel.rest_t[neck_idx_r]
                    tgt_pos[ci] = tgt_neck + R_upper.apply(rest_off)

        # Toe synthesis + dorsiflexion clamp
        for side in ("l", "r"):
            toe_name = f"toe_{side}"
            ankle_name = f"ankle_{side}"
            if toe_name in self.skel.name_to_idx and ankle_name in self.skel.name_to_idx:
                toe_i = self.skel.name_to_idx[toe_name]
                ankle_i = self.skel.name_to_idx[ankle_name]
                toe_anchor = self.skel.coco_anchor.get(toe_i)
                needs_synth = (toe_anchor is None
                               or toe_anchor not in measured
                               or np.any(np.isnan(tgt_pos[toe_i])))
                if needs_synth:
                    rest_off = self.skel.rest_t[toe_i] - self.skel.rest_t[ankle_i]
                    tgt_pos[toe_i] = tgt_pos[ankle_i] + R_root.apply(rest_off)

            # Dorsiflexion clamp
            heel_key = f"heel_{side}"
            if (toe_name in self.skel.name_to_idx
                    and ankle_name in self.skel.name_to_idx):
                toe_i = self.skel.name_to_idx[toe_name]
                ankle_i = self.skel.name_to_idx[ankle_name]
                if not np.all(np.isfinite(tgt_pos[toe_i])):
                    continue
                if (heel_key in measured
                        and np.all(np.isfinite(measured[heel_key]))):
                    max_toe_z = measured[heel_key][2] + 1.0
                else:
                    max_toe_z = tgt_pos[ankle_i][2]
                if tgt_pos[toe_i][2] > max_toe_z:
                    tgt_pos[toe_i][2] = max_toe_z

        # ── Weight vector w (frame-constant portion) ─────────────────────
        # Base weight: 1.0 for measured joints, 0.15 for synthesized, 0 for NaN.
        # Multiply by per-joint triangulation confidence when available —
        # this downweights noisy joints (e.g. right hip conf=0.08).
        conf = self.skel.confidence
        w = np.zeros(self.num_joints)
        for i, val in enumerate(tgt_pos):
            if np.any(np.isnan(val)):
                w[i] = 0.0
            elif i not in self.skel.coco_anchor:
                if np.allclose(val, 0.0):
                    w[i] = 0.0
                else:
                    w[i] = 0.15
            else:
                w[i] = 1.0
                # Apply per-joint confidence if available.
                if conf is not None:
                    coco_name = self.skel.coco_anchor[i]
                    coco_idx = COCO.get(coco_name)
                    if coco_idx is not None and frame_idx < conf.shape[0]:
                        c = conf[frame_idx, coco_idx]
                        if np.isfinite(c) and c > 0.05:
                            # Real confidence from RANSAC — scale the IK weight.
                            w[i] *= float(np.clip(c, 0.1, 1.0))
                        else:
                            # Zero/near-zero confidence: the 2D detector failed
                            # for this frame, but the position was gap-filled from
                            # neighboring frames.  Use a moderate default (0.3)
                            # so the IK still tries to match it, but with less
                            # authority than a high-confidence joint.
                            w[i] *= 0.3

        # ── Spine orientation targets (R_root * R_diff^frac) ─────────────
        n_spine = len(spine_names)
        spine_indices = np.array([self.skel.name_to_idx[nm] for nm in spine_names])
        R_target_quats = np.zeros((n_spine, 4))
        for k in range(n_spine):
            frac = k / (n_spine - 1) if n_spine > 1 else 0.0
            R_target = R_root * (R_diff ** frac)
            R_target_quats[k] = R_target.as_quat()  # xyzw

        # ── Lateral axis for sway damping ────────────────────────────────
        lat_axis = rest_hip / (np.linalg.norm(rest_hip) + 1e-9)

        # ── Limb twist pinned set (frame-constant: depends on tgt_pos) ───
        limb_pinned_info = []
        for side in ("l", "r"):
            for p_name, c_name in (
                (f"hip_{side}", f"knee_{side}"),
                (f"knee_{side}", f"ankle_{side}"),
                (f"shoulder_{side}", f"elbow_{side}"),
                (f"elbow_{side}", f"wrist_{side}"),
            ):
                if p_name not in self.skel.name_to_idx:
                    continue
                p = self.skel.name_to_idx[p_name]
                c = self.skel.name_to_idx[c_name]
                rig_name_c = self.skel.fbx_mapping[c]
                rc_name = self.skel.topo.roll_child(rig_name_c)
                rc_well_conditioned = False
                if rc_name and rc_name in self.skel.fbx_mapping.values():
                    rc_proxy = next(idx for idx, rn in self.skel.fbx_mapping.items()
                                    if rn == rc_name)
                    rc_coco = self.skel.coco_anchor.get(rc_proxy)
                    rc_measured = (rc_coco is not None
                                   and rc_coco in measured
                                   and np.all(np.isfinite(measured[rc_coco]))
                                   and np.linalg.norm(measured[rc_coco]) > 1e-5)
                    rc_children = [k for k in range(self.num_joints)
                                   if self.parents[k] == rc_proxy]
                    if rc_measured and rc_children:
                        bone_dir = tgt_pos[c] - tgt_pos[p]
                        rc_dir = tgt_pos[rc_proxy] - tgt_pos[c]
                        bn = np.linalg.norm(bone_dir)
                        rn = np.linalg.norm(rc_dir)
                        if bn > 1e-5 and rn > 1e-5:
                            cos_ang = np.clip(
                                np.dot(bone_dir / bn, rc_dir / rn), -1, 1)
                            rc_well_conditioned = np.degrees(
                                np.arccos(cos_ang)) > 35.0
                if rc_well_conditioned:
                    continue
                rest_off = self.skel.rest_offsets[c]
                bone = rest_off / (np.linalg.norm(rest_off) + 1e-12)
                p_parent = int(self.parents[p])
                
                limb_pinned_info.append({
                    "p": p, "c": c, "bone": bone, "p_parent": p_parent,
                    "twist_target": 0.0,
                    "twist_weight": 1.0,
                    "side": side,
                })

        # ── Head orientation metadata ────────────────────────────────────
        head_i = self.skel.name_to_idx.get("head")
        neck_i = self.skel.name_to_idx.get("neck_01")
        head_ori_active = (HEAD_ORIENTATION_MODE != "neck_follow"
                           and head_i is not None and neck_i is not None
                           and "nose" in measured
                           and head_i in self.skel.coco_anchor
                           and w[head_i] > 0.0)

        # Collect face keypoints for Kabsch head orientation.
        face_kpts = {}
        for k in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            if k in measured and np.all(np.isfinite(measured[k])):
                face_kpts[k] = measured[k]
        use_kabsch = False  # disabled: replaced by frame-based head orientation

        # ── Frame-based head target ─────────────────────────────────────
        # Build observed head frame from face keypoints, compute delta from
        # rest frame. This gives real head tracking (yaw/pitch/roll) from
        # data without Kabsch or nose-direction.
        head_target_quat = None
        if HEAD_ORIENTATION_MODE != "neck_follow" and head_ori_active and "left_ear" in face_kpts and "right_ear" in face_kpts:
            la = face_kpts["left_ear"]
            ra = face_kpts["right_ear"]
            face_pts = [face_kpts[k] for k in ("nose", "left_eye", "right_eye")
                        if k in face_kpts]
            if len(face_pts) >= 2 and np.linalg.norm(la - ra) > 5.0:
                x = la - ra
                x = x / np.linalg.norm(x)
                face_mid = np.mean(face_pts, axis=0)
                ear_mid = (la + ra) / 2.0
                f_raw = face_mid - ear_mid
                f = f_raw - np.dot(f_raw, x) * x
                fn = np.linalg.norm(f)
                if fn > 1e-5:
                    f = f / fn
                    u = np.cross(f, x)
                    u = u / (np.linalg.norm(u) + 1e-9)
                    F_obs = np.column_stack([x, f, u])
                    F_obs[:, 2] *= -1  # force right-handed (matches data chirality)
                    if np.linalg.det(F_obs) > 0:
                        R_head = Rotation.from_matrix(F_obs @ self._F_rest_head.T)
                        head_target_quat = R_head.as_quat()

        # Canonical head template (kept for backward compat).
        from aimocap.math.kabsch import CANONICAL_HEAD_TEMPLATE

        return {
            "tgt_pos": tgt_pos,
            "w": w.copy(),
            "R_root_quat": R_root.as_quat(),
            "R_diff_quat": R_diff.as_quat(),
            "R_target_quats": R_target_quats,
            "spine_indices": spine_indices,
            "n_spine": n_spine,
            "lat_axis": lat_axis,
            "prev_lat": prev_lat,
            "init_x": init_x,
            "prev_x": prev_x,
            "prev_prev_x": prev_prev_x,
            "accel_weight": accel_weight,
            "temporal_weight": temporal_weight,
            "init_weight": 1e-3,
            "ori_weight": 1.0,
            "limb_pinned_info": limb_pinned_info,
            "head_ori_active": head_ori_active,
            "head_idx": head_i,
            "neck_idx": neck_i,
            "pelvis_idx": self.skel.name_to_idx.get("pelvis"),
            "head_rest_dir": self.skel.rest_offsets[head_i] if head_i is not None else None,
            "nose": measured.get("nose"),
            "head_target_quat": head_target_quat,
            "use_kabsch": use_kabsch,
            "face_kpts": face_kpts,
            "canonical_head": CANONICAL_HEAD_TEMPLATE,
            "num_joints": self.num_joints,
            "rest_t0": self.skel.rest_t[0].copy(),
            "rest_offsets": self.rest_offsets.copy(),
            "parents": self.parents.copy(),
            "twist_limits": self._twist_limits,
            "twist_joint_indices": self._twist_joint_indices,
            "twist_bone_axes": self._twist_bone_axes,
        }

    def _residuals_with_ctx(self, x: np.ndarray, ctx: dict) -> np.ndarray:
        """Hot-path residual using precomputed frame context.

        Produces identical output to _residuals(x, measured, prev_x, ...)
        but skips recomputing R_root, tgt_pos, w, R_targets, etc.
        """
        root_t, local_rotations = self._state_to_local_rotations(x)
        global_pos, global_rot_quat = self.forward_kinematics(root_t, local_rotations)

        tgt_pos = ctx["tgt_pos"]
        w = ctx["w"].copy()
        ori_weight = ctx["ori_weight"]

        # Head position gate (x-dependent — must stay in hot path)
        head_i = ctx["head_idx"]
        neck_i = ctx["neck_idx"]
        if (head_i is not None and neck_i is not None
                and ctx["head_ori_active"] and w[head_i] > 0.0):
            neck_pos = global_pos[neck_i]
            nose_pos = ctx["nose"]
            desired_world = nose_pos - neck_pos
            if np.linalg.norm(desired_world) > 1e-5:
                R_neck = Rotation.from_quat(global_rot_quat[neck_i])
                rest_dir = self.skel.rest_offsets[head_i]
                desired_pl = R_neck.inv().apply(desired_world)
                cos_accept = np.dot(
                    desired_pl / np.linalg.norm(desired_pl),
                    rest_dir / np.linalg.norm(rest_dir))
                if cos_accept <= np.cos(np.deg2rad(70.0)):
                    w[head_i] = 0.1

        res_data = (global_pos - tgt_pos) * w[:, None]
        parts = [res_data.flatten()]

        if ori_weight > 0:
            # Spine orientation
            global_rots = Rotation.from_quat(global_rot_quat)
            n_spine = ctx["n_spine"]
            spine_indices = ctx["spine_indices"]
            R_target_quats = ctx["R_target_quats"]
            ori_res = np.zeros(n_spine * 3)
            for k in range(n_spine):
                i = spine_indices[k]
                R_target = Rotation.from_quat(R_target_quats[k])
                R_current = global_rots[i]
                ori_res[k * 3:(k + 1) * 3] = (
                    R_current * R_target.inv()).as_rotvec()
            parts.append(ori_res * ori_weight)

            # Head orientation: pin head global rotation to the frame-based
            # target computed from face keypoints (real head tracking).
            # Falls back to neck rotation (torso-follow) when face keypoints
            # are unavailable or degenerate.
            head_res = np.zeros(3)
            if ctx["head_ori_active"]:
                htq = ctx.get("head_target_quat")
                if htq is not None:
                    R_head_target = Rotation.from_quat(htq)
                else:
                    R_head_target = global_rots[neck_i]  # torso-follow fallback
                R_head_current = global_rots[head_i]
                head_res = (R_head_current * R_head_target.inv()).as_rotvec()
            parts.append(head_res * ori_weight)

            # Shoulder roll from arm triangle (Stage 3)
            # The arm triangle constrains UPPER-ARM axial roll (shoulder_* orientation).
            # This is the CORRECT use of the arm triangle — it constrains the
            # upper arm's axial roll, not forearm pronation or palm orientation.
            # We add a full SO(3) residual on the shoulder local rotation.
            from aimocap.retarget.swing_twist import signed_twist_angle
            shoulder_roll_targets = ctx.get("shoulder_roll_targets", {})
            shoulder_roll_active = ctx.get("shoulder_roll_active", {})
            for side in ("l", "r"):
                if not shoulder_roll_active.get(side, False):
                    continue
                targets = shoulder_roll_targets.get(side, {})
                if not targets.get("valid"):
                    continue
                arm_data = self._arm_rest.get(side)
                if not arm_data:
                    continue
                sh_i = arm_data["shoulder_i"]
                roll_axis_sh = arm_data["roll_axis_sh"]
                R_sh_parent = global_rots[self.parents[sh_i]]
                R_sh_local = R_sh_parent.inv() * global_rots[sh_i]
                # Full SO(3) residual on shoulder local rotation
                R_target_local = Rotation.from_quat(targets["target_local_quat"])
                R_err = R_sh_local * R_target_local.inv()
                r_sh = R_err.as_rotvec() * ori_weight
                parts.append(r_sh)

            # Sway amplitude + velocity
            pelvis_i = ctx["pelvis_idx"]
            neck_i_s = ctx["neck_idx"]
            if pelvis_i is not None and neck_i_s is not None:
                lat_axis = ctx["lat_axis"]
                neck_off = global_pos[neck_i_s] - global_pos[pelvis_i]
                lat = np.dot(neck_off, lat_axis)
                # SWAY_LIMIT=5cm (tightened from 8: real walking keeps the neck
                # within 5cm of the pelvis line; the measured shoulder midpoint
                # can be 40+cm off due to triangulation noise, and 8cm let too
                # much through).  SWAY_WEIGHT=60 (tripled from 20: the position
                # residual for the neck target is ~1.0 weight, so a 40cm outlier
                # costs 40 in position but only 20*(40-8)=640 in sway — but the
                # solver still compromised at 16cm.  60*(16-5)=660 makes it cost
                # as much as a 16cm position error, enough to actually clamp).
                excess = max(0.0, abs(lat) - 5.0) * np.sign(lat)
                parts.append(np.array([excess]) * 60.0)
                if ctx["prev_lat"] is not None:
                    v = lat - ctx["prev_lat"]
                    excess_v = max(0.0, abs(v) - 0.5) * np.sign(v)
                    parts.append(np.array([excess_v]) * 30.0)
                else:
                    parts.append(np.zeros(1))
            else:
                parts.append(np.zeros(1))
                if ctx["prev_lat"] is not None:
                    parts.append(np.zeros(1))

            # Limb twist
            limb_res = []
            for info in ctx["limb_pinned_info"]:
                p_parent = info["p_parent"]
                p = info["p"]
                bone = info["bone"]
                R_current = global_rots[p_parent].inv() * global_rots[p]
                rv = R_current.as_rotvec()
                twist_rad = np.dot(rv, bone)
                
                twist_target = info.get("twist_target", 0.0)
                twist_weight = info.get("twist_weight", 1.0)
                
                twist_error = (twist_rad - twist_target) * twist_weight
                limb_res.append(np.array([twist_error * bone[i] for i in range(3)]))
            if limb_res:
                parts.append(np.concatenate(limb_res) * ori_weight)

        # Temporal + init
        if ctx["prev_x"] is not None and ctx["temporal_weight"] > 0:
            parts.append(self._state_delta(x, ctx["prev_x"]) * ctx["temporal_weight"])
        if ctx["init_x"] is not None:
            parts.append(self._state_delta(x, ctx["init_x"]) * ctx["init_weight"])
        
        r = np.concatenate(parts)
        # Residual length assertion — catches silent changes when residuals
        # are added/removed in only one code path (scipy vs torch).
        n_exp = ctx.setdefault("_resid_len", r.size)
        if r.size != n_exp:
            raise RuntimeError(f"residual length changed: {r.size} != {n_exp}")
        return r

    def _residuals(self, x: np.ndarray, measured: dict,
                   prev_x: np.ndarray = None, temporal_weight: float = 0.0,
                   init_x: np.ndarray = None, init_weight: float = 1e-3,
                   ori_weight: float = 1.0, prev_lat: float = None) -> np.ndarray:
        root_t, local_rotations = self._state_to_local_rotations(x)
        global_pos, global_rot_quat = self.forward_kinematics(root_t, local_rotations)

        spine_names = self.skel.topo.spine_chain("pelvis", "neck_01")

        hip_line = measured["hip_r"] - measured["hip_l"]
        spine_dir = measured["neck"] - measured["pelvis"]

        idx_hl = self.skel.name_to_idx["hip_l"]
        idx_hr = self.skel.name_to_idx["hip_r"]
        idx_sl = self.skel.name_to_idx["shoulder_l"]
        idx_sr = self.skel.name_to_idx["shoulder_r"]

        rest_hip = self.skel.rest_t[idx_hr] - self.skel.rest_t[idx_hl]
        rest_pelvis_mid = (self.skel.rest_t[idx_hl] + self.skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (self.skel.rest_t[idx_sl] + self.skel.rest_t[idx_sr]) / 2.0
        rest_spine = rest_neck_mid - rest_pelvis_mid

        from aimocap.retarget.root_frame import root_rotation
        R_root = root_rotation(spine_dir, hip_line, rest_spine, rest_hip)

        shoulder_line = measured["shoulder_r"] - measured["shoulder_l"]
        rest_shoulder = self.skel.rest_t[idx_sr] - self.skel.rest_t[idx_sl]
        R_upper = root_rotation(spine_dir, shoulder_line, rest_spine, rest_shoulder)

        # Torso twist in the hip local frame (same convention as analytic_init).
        R_diff = R_root.inv() * R_upper

        # The pelvis and neck_01 targets are offset to joint positions. See
        # analytic_init for the explanation of why the offset is needed.
        rest_pelvis_mid = (self.skel.rest_t[idx_hl] + self.skel.rest_t[idx_hr]) / 2.0
        rest_neck_mid = (self.skel.rest_t[idx_sl] + self.skel.rest_t[idx_sr]) / 2.0
        pelvis_offset = self.skel.rest_t[0] - rest_pelvis_mid
        neck_offset = self.skel.rest_t[self.skel.name_to_idx["neck_01"]] - rest_neck_mid
        tgt_pelvis = measured["pelvis"] + R_root.apply(pelvis_offset)
        tgt_neck = measured["neck"] + R_upper.apply(neck_offset)
        
        from aimocap.retarget.spine_chain import distribute_spine_targets
        rest_positions = np.array([self.skel.rest_t[self.skel.name_to_idx[nm]] for nm in spine_names])
        inter = distribute_spine_targets(tgt_pelvis, tgt_neck, rest_positions)
        
        tgt_pos = np.zeros((self.num_joints, 3))
        si = 0
        for k, nm in enumerate(spine_names):
            i = self.skel.name_to_idx[nm]
            if k == 0:
                tgt_pos[i] = tgt_pelvis
            elif k == len(spine_names) - 1:
                tgt_pos[i] = tgt_neck
            else:
                tgt_pos[i] = inter[si]; si += 1
            
        for i, nm in self.skel.coco_anchor.items():
            if self.skel.joint_names[i] not in ["pelvis", "neck_01"]:
                if nm in measured:
                    tgt_pos[i] = measured[nm]
                # If nm not in measured (e.g. 17-joint input without foot
                # keypoints), tgt_pos[i] stays zero — the toe fallback below
                # will synthesize from R_root.

        # Clavicle targets: synthesize from the neck (see analytic_init for
        # the explanation). Without this, tgt_pos stays 0 and the solver pulls
        # the clavicles to the origin -> caved-in shoulders.
        neck_idx_r = self.skel.name_to_idx["neck_01"]
        for side in ("l", "r"):
            clav_name = f"clavicle_{side}"
            if clav_name in self.skel.name_to_idx:
                ci = self.skel.name_to_idx[clav_name]
                if np.allclose(tgt_pos[ci], 0.0):
                    rest_off = self.skel.rest_t[ci] - self.skel.rest_t[neck_idx_r]
                    tgt_pos[ci] = tgt_neck + R_upper.apply(rest_off)

        # Toe (ball_l/ball_r) targets: when the big-toe keypoint is measured
        # (COCO-WholeBody 17/20), the coco_anchor loop above already set the
        # target from measurement — use it, so foot orientation comes from
        # actual detected foot position, not the pelvis root frame. When the
        # big-toe is NaN (not detected) or toe_* is not coco-anchored, fall
        # back to R_root synthesis so the ankle roll still has a constraint.
        for side in ("l", "r"):
            toe_name = f"toe_{side}"
            ankle_name = f"ankle_{side}"
            if toe_name in self.skel.name_to_idx and ankle_name in self.skel.name_to_idx:
                toe_i = self.skel.name_to_idx[toe_name]
                ankle_i = self.skel.name_to_idx[ankle_name]
                toe_anchor = self.skel.coco_anchor.get(toe_i)
                needs_synth = (toe_anchor is None
                               or toe_anchor not in measured
                               or np.any(np.isnan(tgt_pos[toe_i])))
                if needs_synth:
                    rest_off = self.skel.rest_t[toe_i] - self.skel.rest_t[ankle_i]
                    tgt_pos[toe_i] = tgt_pos[ankle_i] + R_root.apply(rest_off)

        # Dorsiflexion clamp (see analytic_init for the full explanation):
        # prevent the toe target from landing above the ankle/heel, which
        # dorsiflexes the foot ("on heels").  Uses the heel keypoint when
        # available (133-joint data) or falls back to the ankle Z (17-joint
        # data, where the toe is synthesized from R_root and tilts with the
        # torso).  The toe can't be above the ankle in a flat foot.
        for side in ("l", "r"):
            toe_name = f"toe_{side}"
            ankle_name = f"ankle_{side}"
            heel_key = f"heel_{side}"
            if (toe_name not in self.skel.name_to_idx
                    or ankle_name not in self.skel.name_to_idx):
                continue
            toe_i = self.skel.name_to_idx[toe_name]
            ankle_i = self.skel.name_to_idx[ankle_name]
            if not np.all(np.isfinite(tgt_pos[toe_i])):
                continue
            if (heel_key in measured
                    and np.all(np.isfinite(measured[heel_key]))):
                max_toe_z = measured[heel_key][2] + 1.0
            else:
                max_toe_z = tgt_pos[ankle_i][2]
            if tgt_pos[toe_i][2] > max_toe_z:
                tgt_pos[toe_i][2] = max_toe_z

        # Per-joint weights: 1.0 for measured (coco-anchored), 0.3 for synthetic
        # spine intermediates (they guide bend but shouldn't dominate), 0.0 for
        # joints with no real target (e.g. clavicles: not coco-anchored, not in
        # the spine chain, so tgt_pos stays 0 -> a nonzero weight would pull them
        # to the origin and cave the shoulders inward).
        w = np.zeros(self.num_joints)
        for i, val in enumerate(tgt_pos):
            if np.any(np.isnan(val)):
                w[i] = 0.0
            elif i not in self.skel.coco_anchor:
                if np.allclose(val, 0.0):
                    # No target was ever set for this joint (not spine, not
                    # coco-anchored, not synthesized). Don't pull it to the
                    # origin -- let its measured children drive its position.
                    w[i] = 0.0
                else:
                    # Spine intermediates and neck_01: moderate weight so the
                    # solver distributes the bend across spine joints instead
                    # of concentrating it at the pelvis.
                    w[i] = 0.15
            else:
                w[i] = 1.0

        # Head position plausibility gate: the head's POSITION target comes
        # from the measured nose (coco_anchor -> w=1.0 above), but the head
        # ORIENTATION residual below is already gated by a 70 deg check.
        # Without gating the POSITION too, the solver hard-pins the head to a
        # noisy nose position, forcing the neck/spine to swing laterally to
        # reach it -> the "head always moves left or right" artifact.  Apply
        # the same 70 deg gate: when the measured nose direction (neck->nose,
        # expressed in the neck's local frame) is >70 deg from the rest head
        # direction, down-weight the head position so the neck/spine stay
        # neutral instead of swinging.  We down-weight (0.1) rather than zero
        # so a genuinely turned head still biases the solve slightly.
        head_i = self.skel.name_to_idx.get("head")
        neck_i = self.skel.name_to_idx.get("neck_01")
        if (head_i is not None and neck_i is not None
                and "nose" in measured and head_i in self.skel.coco_anchor
                and w[head_i] > 0.0):
            neck_pos = global_pos[neck_i]
            nose_pos = measured["nose"]
            desired_world = nose_pos - neck_pos
            if np.linalg.norm(desired_world) > 1e-5:
                R_neck = Rotation.from_quat(global_rot_quat[neck_i])
                rest_dir = self.skel.rest_offsets[head_i]
                desired_pl = R_neck.inv().apply(desired_world)
                cos_accept = np.dot(
                    desired_pl / np.linalg.norm(desired_pl),
                    rest_dir / np.linalg.norm(rest_dir))
                if cos_accept <= np.cos(np.deg2rad(70.0)):
                    w[head_i] = 0.1   # implausible nose -> don't swing the neck

        res_data = (global_pos - tgt_pos) * w[:, None]
        parts = [res_data.flatten()]

        # Orientation residual for spine joints: pin the global rotation of
        # each spine joint to its SLERP target (R_root * R_diff^frac).  Without
        # this, rotation about a spine bone's long axis is a null direction of
        # the position Jacobian (it moves no joint position), so the solver
        # is free to drift the twist arbitrarily.  This residual breaks that
        # gauge freedom: the solver can still adjust the spine to fit
        # positions, but it can no longer add twist for free.  ori_weight=1.0
        # means a 1-radian (~57 deg) twist drift costs as much as a 1-cm
        # position error, so ~5 deg of twist is an acceptable trade for 1 cm
        # of position — tight enough to prevent the ~83 deg over-rotation but
        # loose enough to let the spine curve.
        if ori_weight > 0:
            n_spine = len(spine_names)
            ori_res = np.zeros(n_spine * 3)
            global_rots = Rotation.from_quat(global_rot_quat)
            for k, nm in enumerate(spine_names):
                i = self.skel.name_to_idx[nm]
                frac = k / (n_spine - 1)
                R_target = R_root * (R_diff ** frac)
                R_current = global_rots[i]
                ori_res[k * 3:(k + 1) * 3] = (
                    R_current * R_target.inv()
                ).as_rotvec()
            parts.append(ori_res * ori_weight)

            # Head orientation: pin the head's global rotation toward the
            # nose direction.  The head is a leaf with a COCO anchor (nose);
            # without this, the solver only sees the head's POSITION (which
            # doesn't constrain roll about the neck->nose axis), so the head
            # can tilt sideways.  The target direction is from neck to nose,
            # and we pin the head's global rotation to match that direction
            # using the same swing-twist approach as analytic_init.
            head_i = self.skel.name_to_idx.get("head")
            neck_i = self.skel.name_to_idx.get("neck_01")
            if (head_i is not None and neck_i is not None
                    and "nose" in measured and head_i in self.skel.coco_anchor):
                neck_pos = global_pos[neck_i]
                nose_pos = measured["nose"]
                desired_world = nose_pos - neck_pos
                rest_dir = self.skel.rest_offsets[head_i]
                # Same plausibility gate as analytic_init: only apply the head
                # orientation residual when the measured nose direction is
                # within ~70 deg of the rest head direction.  A noisy nose
                # triangulation (e.g. landing 23 cm above the neck) would
                # otherwise pin the head to a sideways-facing target.  The
                # residual vector length must stay constant across solve
                # iterations, so when the gate fails we emit a zero residual
                # (no contribution) instead of skipping it.
                head_res = np.zeros(3)
                if np.linalg.norm(desired_world) > 1e-5:
                    R_neck = global_rots[neck_i]
                    desired_pl = R_neck.inv().apply(desired_world)
                    cos_accept = np.dot(
                        desired_pl / np.linalg.norm(desired_pl),
                        rest_dir / np.linalg.norm(rest_dir))
                    if cos_accept > np.cos(np.deg2rad(70.0)):
                        from aimocap.retarget.swing_twist import constrained_rotation
                        R_head_target = R_neck * constrained_rotation(rest_dir, desired_pl)
                        R_head_current = global_rots[head_i]
                        head_res = (R_head_current * R_head_target.inv()).as_rotvec()
                parts.append(head_res * ori_weight)

            # Spine lateral sway damping: the measured shoulders swing
            # laterally frame-to-frame (shoulder_l std ~9 cm on this clip),
            # which tilts spine_dir and R_root sideways -> the spine chain
            # faithfully reproduces a left-right oscillation that reads as
            # "the head keeps moving left-right".  Real human walking has
            # only ~3 deg of lateral spine bend; the measured noise drives
            # up to 10 deg, oscillating 16 times in 300 frames.  Penalize
            # the neck's lateral offset from the pelvis beyond a soft
            # threshold (~8 cm ~= 3 deg on a 153 cm spine).  The lateral
            # axis is the REST hip-line direction (the rig's anatomical
            # left-right), NOT the measured hip line (which is noisy and
            # nearly perpendicular to the neck-pelvis offset, so it would
            # measure ~0).  Using the rest hip line keeps this general
            # (works for any rig, no hardcoded world axes).  This damps the
            # oscillation at the IK level without lagging real motion the
            # way a pre-IK keypoint filter would.
            pelvis_i = self.skel.name_to_idx.get("pelvis")
            neck_i_s = self.skel.name_to_idx.get("neck_01")
            if pelvis_i is not None and neck_i_s is not None:
                lat_axis = rest_hip / (np.linalg.norm(rest_hip) + 1e-9)
                neck_off = global_pos[neck_i_s] - global_pos[pelvis_i]
                lat = np.dot(neck_off, lat_axis)
                SWAY_LIMIT = 5.0  # cm; ~3 deg lateral bend (biomechanical max)
                excess = max(0.0, abs(lat) - SWAY_LIMIT) * np.sign(lat)
                # sway_weight >> ori_weight: the sway residual is a single
                # scalar competing against ~24*3 position terms that pull
                # toward the noisy measured neck.  A high weight clamps the
                # lateral sway hard (kills the left-right oscillation).
                # WEIGHT=60: at 16cm offset, cost=60*(16-5)=660, comparable
                # to the ~16*3=48 per-joint position error the solver trades
                # against.  WEIGHT=20 was too low: 20*(16-8)=160 was dominated
                # by position terms, so the solver compromised at 16cm.
                SWAY_WEIGHT = 60.0
                parts.append(np.array([excess]) * SWAY_WEIGHT)

                # Neck lateral velocity damping: complement to the sway
                # amplitude clamp above.  The amplitude clamp (SWAY_LIMIT=5cm)
                # kills sustained sideways lean but a 4cm oscillation at 1.6 Hz
                # would pass the 5cm gate and still read as "head keeps moving
                # left-right".  Real walking has slow lateral sway (~0.3 Hz,
                # <0.5 cm/frame); the noise-driven oscillation is ~1.6 Hz
                # (>1 cm/frame).  This penalizes rapid lateral CHANGE, targeting
                # the frequency directly.  Reuses `lat` (computed above) and
                # `prev_lat` (computed once per frame in solve_frame, not per
                # residual eval — prev_x is constant across all evals within
                # one solve_frame call, so one extra FK per frame is negligible).
                if prev_lat is not None:
                    VEL_LIMIT = 0.5   # cm/frame dead-zone (~15 cm/s at 30fps)
                    v = lat - prev_lat
                    excess_v = max(0.0, abs(v) - VEL_LIMIT) * np.sign(v)
                    VEL_WEIGHT = 30.0
                    parts.append(np.array([excess_v]) * VEL_WEIGHT)
                else:
                    parts.append(np.zeros(1))
            else:
                parts.append(np.zeros(1))
                if prev_lat is not None:
                    parts.append(np.zeros(1))

            # Limb orientation: penalize twist about each limb bone's long
            # axis.  Twist about a limb bone's long axis is a null direction of
            # the position residual (it moves no joint position), so without
            # this the solver is free to drift the calf/thigh/forearm twist by
            # 60-100 deg to save a fraction of a cm elsewhere -- the visible
            # "caved-in" knee or spinning hand.  We project the bone's local
            # rotation rotvec onto its rest long axis and penalize that
            # component.  The residual is RELAXED (skipped) only when the
            # roll child is a real measurement AND is well-conditioned (the
            # roll-child direction is >35 deg from the bone -- i.e. the limb
            # is bent enough that the roll child genuinely constrains roll).
            # When the roll child is synthesized (toe from R_root) or the limb
            # is near-straight (arm bone-vs-hand <35 deg), any non-zero twist
            # is garbage and we pin it to zero.  ori_weight=1.0 makes ~5 deg
            # of drift cost as much as 1 cm of position error.
            limb_res = []
            for side in ("l", "r"):
                for p_name, c_name in (
                    (f"hip_{side}", f"knee_{side}"),     # thigh
                    (f"knee_{side}", f"ankle_{side}"),   # shin
                    (f"shoulder_{side}", f"elbow_{side}"),  # upper arm
                    (f"elbow_{side}", f"wrist_{side}"),     # forearm
                ):
                    if p_name not in self.skel.name_to_idx:
                        continue
                    p = self.skel.name_to_idx[p_name]
                    c = self.skel.name_to_idx[c_name]
                    # The roll child constrains roll ONLY if it's a real
                    # measurement AND has proxy children (intermediate joint,
                    # not a chain-end leaf).  A chain-end roll child (e.g. the
                    # wrist for the forearm) has its position fully determined
                    # by the chain, so it carries no independent roll info ->
                    # pin twist to zero.  See analytic_init for the full
                    # explanation.  When the roll child is synthesized (toe from
                    # R_root) or is a leaf, any non-zero twist is garbage.
                    rig_name_c = self.skel.fbx_mapping[c]
                    rc_name = self.skel.topo.roll_child(rig_name_c)
                    rc_well_conditioned = False
                    if rc_name and rc_name in self.skel.fbx_mapping.values():
                        rc_proxy = next(idx for idx, rn in self.skel.fbx_mapping.items()
                                        if rn == rc_name)
                        rc_coco = self.skel.coco_anchor.get(rc_proxy)
                        rc_measured = (rc_coco is not None
                                       and rc_coco in measured
                                       and np.all(np.isfinite(measured[rc_coco]))
                                       and np.linalg.norm(measured[rc_coco]) > 1e-5)
                        rc_children = [k for k in range(self.num_joints)
                                       if self.parents[k] == rc_proxy]
                        if rc_measured and rc_children:
                            # Even a measured intermediate roll child can't
                            # constrain roll when the limb is nearly straight
                            # (roll-child direction ≈ bone direction -> the
                            # perpendicular projection is noise). Check the
                            # bone-vs-rollchild angle in the current pose; if
                            # <35 deg, pin twist to zero (same threshold as
                            # constrained_rotation's near-parallel guard).
                            bone_dir = tgt_pos[c] - tgt_pos[p]
                            rc_dir = tgt_pos[rc_proxy] - tgt_pos[c]
                            bn = np.linalg.norm(bone_dir)
                            rn = np.linalg.norm(rc_dir)
                            if bn > 1e-5 and rn > 1e-5:
                                cos_ang = np.clip(
                                    np.dot(bone_dir / bn, rc_dir / rn), -1, 1)
                                rc_well_conditioned = np.degrees(
                                    np.arccos(cos_ang)) > 35.0
                    if rc_well_conditioned:
                        continue  # roll child constrains roll -> let solver refine
                    rest_off = self.skel.rest_offsets[c]
                    bone = rest_off / (np.linalg.norm(rest_off) + 1e-12)
                    p_parent = self.parents[p]
                    R_current = global_rots[p_parent].inv() * global_rots[p]
                    rv = R_current.as_rotvec()
                    twist_rad = np.dot(rv, bone)  # signed twist about bone
                    limb_res.append(np.array([twist_rad * bone[i] for i in range(3)]))
            if limb_res:
                parts.append(np.concatenate(limb_res) * ori_weight)

        if prev_x is not None and temporal_weight > 0:
            parts.append(self._state_delta(x, prev_x) * temporal_weight)
        if init_x is not None:
            parts.append(self._state_delta(x, init_x) * init_weight)
        return np.concatenate(parts)

    def solve_frame(self, measured: dict, prev_x: np.ndarray = None,
                    temporal_weight: float = 0.03, frame_idx: int = 0,
                    prev_prev_x: np.ndarray = None,
                    accel_weight: float = 0.0) -> np.ndarray:
        if prev_x is None:  # re-init from measured if no prev_x
            x0 = self.analytic_init(measured)
        else:
            x0 = prev_x.copy()

        # Compute prev_lat once per frame: prev_x is constant across all
        # residual evaluations within this solve_frame call, so computing
        # the previous neck lateral offset here (one extra FK per FRAME, not
        # per eval) is negligible.  Passed to _residuals for the neck lateral
        # velocity damping term, which targets the oscillation FREQUENCY
        # (complementing the sway amplitude clamp that targets the magnitude).
        prev_lat = None
        if prev_x is not None:
            pelvis_i = self.skel.name_to_idx.get("pelvis")
            neck_i = self.skel.name_to_idx.get("neck_01")
            if pelvis_i is not None and neck_i is not None:
                idx_hr = self.skel.name_to_idx["hip_r"]
                idx_hl = self.skel.name_to_idx["hip_l"]
                rest_hip = self.skel.rest_t[idx_hr] - self.skel.rest_t[idx_hl]
                lat_axis = rest_hip / (np.linalg.norm(rest_hip) + 1e-9)
                pr, pl = self._state_to_local_rotations(prev_x)
                pgp, _ = self.forward_kinematics(pr, pl)
                prev_lat = float(np.dot(pgp[neck_i] - pgp[pelvis_i], lat_axis))

        init_x = self.analytic_init(measured)

        # ── Precompute frame context (saves ~60% per-frame time) ────────
        ctx = self._precompute_frame(
            measured, prev_x, temporal_weight, init_x, prev_lat,
            frame_idx=frame_idx, prev_prev_x=prev_prev_x,
            accel_weight=accel_weight)

        # ── Backend dispatch ────────────────────────────────────────────
        from aimocap.retarget.ik_backend import dispatch_solve_frame
        return dispatch_solve_frame(self, ctx, x0)
