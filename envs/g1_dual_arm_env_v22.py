# envs/g1_dual_arm_env_v22.py
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco as mj

try:
    import mujoco.viewer as mjv
    _HAVE_VIEWER = True
except Exception:
    mjv = None
    _HAVE_VIEWER = False

DEFAULT_XML = os.environ.get("G1_XML", "models/g1_dual_arm.xml")

ARM_ACTS = [
    "left_shoulder_pitch_act","left_shoulder_roll_act","left_shoulder_yaw_act",
    "left_elbow_act","left_wrist_roll_act","left_wrist_pitch_act","left_wrist_yaw_act",
    "right_shoulder_pitch_act","right_shoulder_roll_act","right_shoulder_yaw_act",
    "right_elbow_act","right_wrist_roll_act","right_wrist_pitch_act","right_wrist_yaw_act",
]
FINGER_SYNS = [
    "left_thumb_close","left_index_close","left_middle_close","left_ring_close","left_little_close",
    "right_thumb_close","right_index_close","right_middle_close","right_ring_close","right_little_close",
]
FIST_MACROS = ["left_fist","right_fist"]

BASE_SENSORS = [
    "right_ee_pos","right_ee_quat","left_ee_pos",
    "cube_top_pos","plate_top_pos",
    "right_thumb_touch","right_index_touch","right_middle_touch","right_ring_touch","right_little_touch",
    "right_wrist_vang","right_wrist_vlin",
    "u_left_fist","u_right_fist",
    "right_palm_force","right_palm_torque",
]

def _safe_name2id(model, obj_type, name):
    try:
        return mj.mj_name2id(model, obj_type, name)
    except Exception:
        return -1

def _get_sensor_slice(model, sid):
    adr = int(model.sensor_adr[sid]); dim = int(model.sensor_dim[sid])
    return slice(adr, adr+dim)

class DualArmGraspEnvV22(gym.Env):
    """
    Dual-arm grasp với reward TIẾN BỘ (progress), gating theo contact,
    latency & sensor delay, asymmetric obs (privileged cho critic).
    Có 'sanity_check' để bắt thiếu actuator/sensor ngay khi khởi tạo.
    """
    metadata = {"render_modes": ["human","none"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str = DEFAULT_XML,
        render_mode: str | None = None,
        max_steps: int = 400,
        use_privileged: bool = True,
        obs_noise_std: float = 0.004,
        latency_steps: int = 2,
        sensor_delay_steps: int = 1,
        hold_steps: int = 120,
        slip_torque_lim: float = 45.0,
        max_wrist_angvel: float = 20.0,
        strict_check: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.xml_path = xml_path
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.use_privileged = bool(use_privileged)

        self.model = mj.MjModel.from_xml_path(self.xml_path)
        self.data  = mj.MjData(self.model)

        # ----- map actuators (bắt buộc phải có) -----
        self.actuator_names = ARM_ACTS + FINGER_SYNS + FIST_MACROS
        self.aid = np.array([_safe_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, n)
                             for n in self.actuator_names], dtype=int)

        missing_acts = [n for n, i in zip(self.actuator_names, self.aid) if i < 0]
        if strict_check and missing_acts:
            raise RuntimeError(f"[ENV CHECK] Thiếu actuator: {missing_acts}. "
                               f"Đúng tên trong XML rồi chạy lại.")

        # ----- action space -----
        n_act = len(self.actuator_names)
        self.action_space = spaces.Box(low=-np.ones(n_act, np.float32),
                                       high= np.ones(n_act, np.float32),
                                       dtype=np.float32)

        # ----- sensors (có thể thiếu; nếu thiếu sẽ điền 0) -----
        self.sensor_names = list(BASE_SENSORS)
        for rn in ["right_rf_center_dist","right_rf_u_dist","right_rf_d_dist"]:
            sid = _safe_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, rn)
            if sid >= 0: self.sensor_names.append(rn)
        self.sids = [_safe_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, n) for n in self.sensor_names]

        # tay: joint pos/vel public (không strict ở đây)
        self.arm_joint_names = [n.replace("_act","_joint") for n in ARM_ACTS]
        self.jids = [_safe_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, n) for n in self.arm_joint_names]

        # privileged (critic)
        self._priv_items = [("cube_qpos",7), ("cube_qvel",6), ("contact_force_sum",1), ("plate_normal_z",1)]

        # ----- obs dim -----
        base_dim = 0
        for sid in self.sids:
            base_dim += int(self.model.sensor_dim[sid]) if sid >= 0 and self.model.sensor_dim[sid] > 0 else 1
        base_dim += len(self.jids)*2
        priv_dim = sum(dim for _, dim in self._priv_items) if self.use_privileged else 0
        self.obs_dim = base_dim + priv_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # smoothing & rate-limit (đủ để tránh giật)
        self._u_prev = np.zeros(n_act, dtype=np.float64)
        self._alpha_arm, self._alpha_finger = 0.4, 0.22
        self._du_max_arm, self._du_max_finger = 0.15, 0.08
        self._max_arm_qvel = 6.0

        # latency & sensor delay
        self.latency_steps = int(max(0, latency_steps))
        self.sensor_delay_steps = int(max(0, sensor_delay_steps))
        self._u_ring = [np.zeros(n_act, dtype=np.float64) for _ in range(self.latency_steps+1)]
        self._obs_ring = [np.zeros(self.obs_dim, dtype=np.float64) for _ in range(self.sensor_delay_steps+1)]
        self._ring_idx = 0

        # noise
        self.obs_noise_std = float(obs_noise_std)

        # curriculum & phase
        self.curr_level = 0.0
        self._succ_ema = 0.0
        self._ema_beta = 0.995
        self.curr_phase = 0
        self.hold_steps_required = int(hold_steps)
        self._hold_counter = 0

        # safety
        self.slip_torque_lim = float(slip_torque_lim)
        self.max_wrist_angvel = float(max_wrist_angvel)
        self._right_fist_id = _safe_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "right_fist")

        # RNG
        self._rng = np.random.default_rng(seed)

        # viewer
        self._viewer = None

        # cache ids
        self._jid_cube_free = _safe_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "grasp_cube_free")
        self._sid_wrist_vang = _safe_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "right_wrist_vang")
        self._sid_right_palm_torque = _safe_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "right_palm_torque")
        self._sid_cube_top = _safe_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "cube_top_pos")
        self._sid_plate_top = _safe_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "plate_top_pos")

        self._touch_sids = [_safe_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, n) for n in
                            ["right_thumb_touch","right_index_touch","right_middle_touch","right_ring_touch","right_little_touch"]]

        # progress state
        self._z0_cube_top = 0.0
        self._prev_dist = 0.0
        self._prev_dz_rel = 0.0
        self._jerk = 0.0

        # cuối cùng reset
        self._reset_internal(randomize=True)

    # --------------- helpers ---------------
    def _sensor_vec(self, sid:int):
        if sid < 0: return np.array([0.0], dtype=np.float64)
        s = _get_sensor_slice(self.model, sid)
        return self.data.sensordata[s]

    def _arm_state(self):
        qs, vs = [], []
        for j in self.jids:
            if j >= 0:
                qadr = int(self.model.jnt_qposadr[j]); dadr = int(self.model.jnt_dofadr[j])
                qs.append(float(self.data.qpos[qadr])); vs.append(float(self.data.qvel[dadr]))
            else:
                qs.append(0.0); vs.append(0.0)
        return np.array(qs, dtype=np.float64), np.array(vs, dtype=np.float64)

    def _priv_obs(self):
        out = []
        if self._jid_cube_free >= 0:
            qadr = int(self.model.jnt_qposadr[self._jid_cube_free])
            dadr = int(self.model.jnt_dofadr[self._jid_cube_free])
            out.extend(self.data.qpos[qadr:qadr+7].tolist())
            out.extend(self.data.qvel[dadr:dadr+6].tolist())
        else:
            out.extend([0.0]*13)
        touch_sum = 0.0
        for sid in self._touch_sids:
            v = self._sensor_vec(sid); touch_sum += float(v[0]) if v.size else 0.0
        out.append(touch_sum)
        pz = 0.0
        if self._sid_plate_top >= 0:
            p = self._sensor_vec(self._sid_plate_top)
            if p.size >= 3: pz = float(p[2])
        out.append(pz)
        return np.asarray(out, dtype=np.float64)

    def _obs_now(self):
        arr = []
        for sid in self.sids:
            v = self._sensor_vec(sid)
            if v.size == 0: arr.append(0.0)
            else:           arr.extend(v.tolist())
        q, v = self._arm_state()
        arr.extend(q.tolist()); arr.extend(v.tolist())
        if self.use_privileged: arr.extend(self._priv_obs().tolist())
        return np.asarray(arr, dtype=np.float64)

    def _apply_obs_noise(self, x):
        if self.obs_noise_std > 0:
            x = x.copy()
            x += self._rng.normal(0.0, self.obs_noise_std, size=x.shape)
        return x

    def _smooth_and_limit(self, a):
        n_arm = len(ARM_ACTS)
        arm, fin = a[:n_arm].copy(), a[n_arm:].copy()
        arm = self._alpha_arm*arm + (1-self._alpha_arm)*self._u_prev[:n_arm]
        fin = self._alpha_finger*fin + (1-self._alpha_finger)*self._u_prev[n_arm:]
        def limit(x, xprev, du): return xprev + np.clip(x-xprev, -du, du)
        arm = limit(arm, self._u_prev[:n_arm], self._du_max_arm)
        fin = limit(fin, self._u_prev[n_arm:], self._du_max_finger)
        return np.concatenate([arm, fin])

    def _apply_latency(self, u):
        self._ring_idx = (self._ring_idx + 1) % len(self._u_ring)
        self._u_ring[self._ring_idx] = u.copy()
        delayed_idx = (self._ring_idx - self.latency_steps) % len(self._u_ring)
        return self._u_ring[delayed_idx]

    def _set_action(self, action):
        a = np.clip(np.asarray(action, np.float64), -1.0, 1.0)
        u = self._smooth_and_limit(a)
        self._jerk = float(np.linalg.norm(u - self._u_prev))
        u_lat = self._apply_latency(u)
        self._u_prev = u.copy()
        for i, aid in enumerate(self.aid):
            if aid < 0: continue
            lo, hi = self.model.actuator_ctrlrange[aid]
            self.data.ctrl[aid] = lo + 0.5*(u_lat[i]+1.0)*(hi-lo)

    # --------------- reward ---------------
    def _compute_metrics(self):
        right_ee = self._sensor_vec(_safe_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "right_ee_pos"))
        cube = self._sensor_vec(self._sid_cube_top)
        plate = self._sensor_vec(self._sid_plate_top)
        dist = float(np.linalg.norm((right_ee - cube)[:3])) if (right_ee.size>=3 and cube.size>=3) else 0.0
        dz_abs = float(cube[2]-plate[2]) if (cube.size>=3 and plate.size>=3) else 0.0
        dz_rel = float(cube[2]-self._z0_cube_top) if (cube.size>=3) else 0.0
        touches = []
        for sid in self._touch_sids:
            v = self._sensor_vec(sid); touches.append(float(v[0]) if v.size else 0.0)
        touch_sum = float(np.sum(touches))
        n_fingers = int(np.sum(np.array(touches) > 1e-3))
        w_ang = self._sensor_vec(self._sid_wrist_vang)
        wrist_rate = float(np.linalg.norm(w_ang)) if w_ang.size else 0.0
        palm_torque = self._sensor_vec(self._sid_right_palm_torque)
        tau_mag = float(np.linalg.norm(palm_torque)) if palm_torque.size else 0.0
        return {"dist":dist,"dz":dz_abs,"dz_rel":dz_rel,"touch_sum":touch_sum,"nf":n_fingers,
                "wrist_rate":wrist_rate,"tau_mag":tau_mag}

    def _compute_reward(self, M):
        r_reach_prog = 2.0 * (self._prev_dist - M["dist"])
        gate_contact = 1.0 if M["touch_sum"] > 0.03 else 0.0
        r_lift_prog = 6.0 * (M["dz_rel"] - self._prev_dz_rel) * gate_contact
        r_lift_prog = float(np.clip(r_lift_prog, -0.5, 1.0))
        r_touch = 0.4*np.tanh(0.6*M["touch_sum"]) + 0.4*(M["nf"]>=2)*gate_contact
        a_ctrl = self.data.ctrl[self.aid[self.aid >= 0]]
        r_pen = -0.0015*M["wrist_rate"] \
                -0.0030*self._jerk \
                -0.0020*(float(np.linalg.norm(a_ctrl)) if a_ctrl.size>0 else 0.0) \
                -0.0025*max(0.0, M["tau_mag"] - self.slip_torque_lim)
        r_step = -0.002
        reward = r_reach_prog + r_lift_prog + r_touch + r_pen + r_step

        success = 1.0 if (M["dz_rel"] > 0.04 and M["touch_sum"] > 0.08) else 0.0
        if success > 0.5: self._hold_counter += 1
        else:             self._hold_counter = 0
        if self._hold_counter >= self.hold_steps_required:
            reward += 10.0
        info = {"success":success, "hold": float(self._hold_counter >= self.hold_steps_required)}
        info.update(M)
        self._prev_dist = M["dist"]; self._prev_dz_rel = M["dz_rel"]
        return reward, info

    def _safety_termination(self, M):
        if M["wrist_rate"] > self.max_wrist_angvel: return True, "wrist_angvel"
        if not np.isfinite(M["dist"]) or not np.isfinite(M["dz"]) or not np.isfinite(M["dz_rel"]): return True, "nan_guard"
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)): return True, "state_invalid"
        return False, ""

    # --------------- reset/randomize ---------------
    def _randomize_physics(self):
        L = float(np.clip(self.curr_level, 0.0, 1.0))
        gid = _safe_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "grasp_cube_geom")
        if gid >= 0:
            sz = self.model.geom_size[gid].copy()
            noise = self._rng.uniform(-0.003*L, 0.003*L, size=3)
            self.model.geom_size[gid] = np.clip(sz + noise, 0.012, 0.030)
            fr = self.model.geom_friction[gid].copy()
            fr *= (1.0 + self._rng.uniform(-0.3*L, 0.3*L, size=3))
            fr[0] = np.clip(fr[0], 0.5, 5.0); fr[1] = np.clip(fr[1], 0.005, 0.10); fr[2] = np.clip(fr[2], 0.002, 0.05)
            self.model.geom_friction[gid] = fr
        names = [mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i) for i in range(self.model.ngeom)]
        for i, n in enumerate(names):
            if n and n.endswith("_tip_pad"):
                fr = self.model.geom_friction[i].copy()
                fr *= (1.0 + self._rng.uniform(-0.25*L, 0.25*L, size=3))
                fr[0] = np.clip(fr[0], 3.0, 12.0); fr[1] = np.clip(fr[1], 0.02, 0.20); fr[2] = np.clip(fr[2], 0.002, 0.05)
                self.model.geom_friction[i] = fr

        def jitter_sol(i, sref_scale=0.15*L, simp_scale=0.005*L):
            if hasattr(self.model, "geom_solref"):
                sr = self.model.geom_solref[i].copy()
                sr[0] *= 1.0 + self._rng.uniform(-sref_scale, sref_scale)
                sr[1] *= 1.0 + self._rng.uniform(-sref_scale, sref_scale)
                self.model.geom_solref[i] = sr
            if hasattr(self.model, "geom_solimp"):
                si = self.model.geom_solimp[i].copy()
                si[:2] = np.clip(si[:2] + self._rng.uniform(-simp_scale, simp_scale, size=2), 0.90, 0.999)
                self.model.geom_solimp[i] = si

        important = []
        if gid >= 0: important.append(gid)
        for i, n in enumerate(names):
            if n and ("palm" in n or "tip_pad" in n): important.append(i)
        for i in set(important): jitter_sol(i)

        for aid in self.aid:
            if aid < 0: continue
            lo, hi = self.model.actuator_ctrlrange[aid]
            span = (hi - lo) * (1.0 + self._rng.uniform(-0.10*L, 0.10*L))
            self.model.actuator_ctrlrange[aid] = np.array([lo, lo+span], dtype=np.float64)

    def _place_cube_on_plate(self):
        if self._jid_cube_free < 0: return
        qadr = int(self.model.jnt_qposadr[self._jid_cube_free]); dadr = int(self.model.jnt_dofadr[self._jid_cube_free])
        dx = self._rng.uniform(-0.010, 0.010); dy = self._rng.uniform(-0.010, 0.010)
        pz = 0.04
        if self._sid_plate_top >= 0:
            p = self._sensor_vec(self._sid_plate_top)
            if p.size >= 3: pz = float(p[2])
        cz = 0.02
        sid_geom = _safe_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "grasp_cube_geom")
        if sid_geom >= 0: cz = float(self.model.geom_size[sid_geom][2])
        z = pz - cz + 2*cz
        self.data.qpos[qadr:qadr+7] = np.array([0.12+dx, 0.0+dy, z, 1,0,0,0], dtype=np.float64)
        self.data.qvel[dadr:dadr+6] = 0.0

    def _reset_internal(self, randomize: bool):
        mj.mj_resetData(self.model, self.data)
        if randomize: self._randomize_physics()
        self._place_cube_on_plate()
        self._u_prev[:] = 0.0
        for i in range(len(self._u_ring)): self._u_ring[i][:] = 0.0
        self._jerk = 0.0; self.step_count = 0; self._hold_counter = 0
        mj.mj_forward(self.model, self.data)
        if self._sid_cube_top >= 0:
            zt = self._sensor_vec(self._sid_cube_top); self._z0_cube_top = float(zt[2]) if zt.size >= 3 else 0.0
        else:
            self._z0_cube_top = 0.0
        right_ee = self._sensor_vec(_safe_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "right_ee_pos"))
        cube = self._sensor_vec(self._sid_cube_top)
        self._prev_dist = float(np.linalg.norm((right_ee - cube)[:3])) if (right_ee.size>=3 and cube.size>=3) else 0.0
        self._prev_dz_rel = 0.0
        if self.render_mode == "human" and _HAVE_VIEWER and self._viewer is None:
            self._viewer = mjv.launch_passive(self.model, self.data)
        if self._viewer is not None and getattr(self._viewer,"is_running",lambda:False)(): self._viewer.sync()
        o = self._apply_obs_noise(self._obs_now())
        for i in range(len(self._obs_ring)): self._obs_ring[i][:] = o
        self._ring_idx = 0

    # --------------- gym api ---------------
    def reset(self, *, seed=None, options=None):
        if seed is not None: self._rng = np.random.default_rng(seed)
        if options and options.get("xml_path") and options["xml_path"] != self.xml_path:
            self.xml_path = options["xml_path"]
            self.model = mj.MjModel.from_xml_path(self.xml_path); self.data = mj.MjData(self.model)
            if self._viewer is not None:
                try: self._viewer.close()
                except: pass
                self._viewer = None
        self._reset_internal(randomize=True)
        obs = self._apply_obs_noise(self._obs_now())
        self._ring_idx = (self._ring_idx + 1) % len(self._obs_ring)
        self._obs_ring[self._ring_idx] = obs
        delayed_idx = (self._ring_idx - self.sensor_delay_steps) % len(self._obs_ring)
        return self._obs_ring[delayed_idx].astype(np.float32), {}

    def step(self, action):
        for j in self.jids:
            if j >= 0:
                dadr = int(self.model.jnt_dofadr[j])
                self.data.qvel[dadr] = float(np.clip(self.data.qvel[dadr], -self._max_arm_qvel, self._max_arm_qvel))
        self._set_action(action)

        # simple grip reflex trên right_fist để hỗ trợ bóp
        try:
            touch_sum = 0.0
            for sid in self._touch_sids:
                v = self._sensor_vec(sid); touch_sum += float(v[0]) if v.size else 0.0
            if self._right_fist_id >= 0:
                lo, hi = self.model.actuator_ctrlrange[self._right_fist_id]
                cur = self.data.ctrl[self._right_fist_id]
                tau = self._sensor_vec(self._sid_right_palm_torque)
                tau_mag = float(np.linalg.norm(tau)) if tau.size else 0.0
                if touch_sum > 0.02 and tau_mag < self.slip_torque_lim*1.6:
                    cur = float(np.clip(cur + 0.0065, lo, hi))
                elif tau_mag > self.slip_torque_lim*2.2:
                    cur = float(np.clip(cur - 0.0100, lo, hi))
                self.data.ctrl[self._right_fist_id] = cur
        except Exception:
            pass

        mj.mj_step(self.model, self.data); self.step_count += 1
        obs_now = self._apply_obs_noise(self._obs_now())
        self._ring_idx = (self._ring_idx + 1) % len(self._obs_ring)
        self._obs_ring[self._ring_idx] = obs_now
        delayed_idx = (self._ring_idx - self.sensor_delay_steps) % len(self._obs_ring)
        obs = self._obs_ring[delayed_idx].astype(np.float32)

        M = self._compute_metrics()
        reward, info = self._compute_reward(M)

        terminated = False; truncated = False
        self._succ_ema = self._ema_beta*self._succ_ema + (1-self._ema_beta)*info["success"]
        if self._succ_ema > 0.35: self.curr_level = float(np.clip(self.curr_level + 0.002, 0.0, 1.0))
        else:                      self.curr_level = float(np.clip(self.curr_level - 0.001, 0.0, 1.0))
        if self.curr_level > 0.15: self.curr_phase = 1
        if self.curr_level > 0.45: self.curr_phase = 2

        unsafe, reason = self._safety_termination(M)
        if unsafe:
            terminated = True; info["termination_reason"] = reason
        if info["hold"] > 0.5:
            terminated = True; info["termination_reason"] = "success_hold"
        if self.step_count >= self.max_steps: truncated = True

        if self.render_mode == "human" and self._viewer is not None and getattr(self._viewer,"is_running",lambda:False)():
            self._viewer.sync()

        info["curr_level"] = self.curr_level; info["succ_ema"] = self._succ_ema
        info["phase"] = self.curr_phase; info["jerk"] = self._jerk
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._viewer is not None:
            try: self._viewer.close()
            except: pass
            self._viewer = None

def make_env(xml_path=DEFAULT_XML, render=False, **kwargs):
    return DualArmGraspEnvV22(
        xml_path=xml_path,
        render_mode="human" if render else "none",
        **kwargs
    )
