import numpy as np
from scipy.spatial.transform import Rotation
from aimocap.retarget.mocap_skeleton import MocapSkeleton
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
        # (caved-in shoulders).  Synthesize from the neck using the upper-body
        # frame R_upper, the same way the toe is synthesized from the ankle.
        neck_idx = self.skel.name_to_idx["neck_01"]
        for side in ("l", "r"):
            clav_name = f"clavicle_{side}"
            if clav_name in self.skel.name_to_idx:
                ci = self.skel.name_to_idx[clav_name]
                if np.allclose(target_pos[ci], 0.0):
                    rest_off = self.skel.rest_t[ci] - self.skel.rest_t[neck_idx]
                    target_pos[ci] = tgt_neck + R_upper.apply(rest_off)

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
            
            # The roll child must be a child of j, to pin the rotation of p!
            # e.g., if p is thigh, j is calf, rc is foot.
            rig_name_j = self.skel.fbx_mapping[j]
            rc_name = self.skel.topo.roll_child(rig_name_j)
            if rc_name is not None and rc_name in self.skel.fbx_mapping.values():
                rc_proxy = next(idx for idx, rn in self.skel.fbx_mapping.items() if rn == rc_name)
                # Only use the roll child as a twist constraint when BOTH:
                #  (a) its target is a real MEASUREMENT (not synthesized from
                #      R_root -- e.g. 17-joint data has no big_toe, so toe_l is
                #      synthesized; twisting to match an R_root-derived target
                #      is circular and injects garbage roll, the "caved-in"
                #      calf), AND
                #  (b) it has proxy children (it's an intermediate joint, not a
                #      chain-end leaf).  A chain-end roll child (e.g. the wrist
                #      for the forearm) has its position fully determined by
                #      the chain geometry, so it carries no INDEPENDENT roll
                #      information -- using it as a roll constraint produces
                #      large spurious twists when the limb is bent (the
                #      "hands rotated wrongly" artifact).  An intermediate roll
                #      child (e.g. ankle for the shin, elbow for the upper arm)
                #      has children beyond it, so its position genuinely
                #      constrains the parent's roll.
                # When either condition fails, skip the twist constraint:
                # constrained_rotation returns swing-only, leaving the bone's
                # long-axis twist at zero (identity).
                rc_children = [c for c in range(self.num_joints)
                               if self.parents[c] == rc_proxy]
                rc_coco = self.skel.coco_anchor.get(rc_proxy)
                rc_measured = (rc_coco is not None
                               and rc_coco in measured
                               and np.all(np.isfinite(measured[rc_coco]))
                               and np.linalg.norm(measured[rc_coco]) > 1e-5)
                if rc_measured and rc_children:
                    roll_rest = self.skel.rest_offsets[rc_proxy]
                    roll_des_world = target_pos[rc_proxy] - target_pos[j]
                    # transform roll_des into parent's rest frame, just like desired_world
                    roll_des = global_rot[p_parent].inv().apply(roll_des_world)
                else:
                    roll_rest = None; roll_des = None
            else:
                roll_rest = None; roll_des = None
                
            R_local = constrained_rotation(rest_off, desired_pl, roll_rest, roll_des)
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
                local_quats[p] = [0, 0, 0, 1]
                global_rot[p] = global_rot[self.parents[p]]

        rotvecs = Rotation.from_quat(local_quats).as_rotvec()
        x = np.zeros(self.num_vars)
        x[0:3] = root_t
        x[3:] = rotvecs.flatten()
        return x

    def _residuals(self, x: np.ndarray, measured: dict,
                   prev_x: np.ndarray = None, temporal_weight: float = 0.0,
                   init_x: np.ndarray = None, init_weight: float = 1e-3,
                   ori_weight: float = 1.0) -> np.ndarray:
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
                    temporal_weight: float = 0.03) -> np.ndarray:
        if prev_x is None:  # re-init from measured if no prev_x
            x0 = self.analytic_init(measured)
        else:
            x0 = prev_x.copy()

        from scipy.optimize import least_squares
        res = least_squares(
            self._residuals, x0,
            args=(measured, prev_x, temporal_weight, self.analytic_init(measured)),
            method='lm', max_nfev=500, ftol=1e-6, xtol=1e-6, gtol=1e-6,
        )
        return res.x
