"""Run a trained TacDiffusion wrench policy (Wu et al., arXiv:2409.11047) on the FORGE peg-in-hole scene.

The model is the conditional DDPM from `diffusion/TacDiffusion` (loaded in-process from its `.pth`):

    input  (36) = [s_t, s_{t-hist}], s = [F_ext (6), F_in (6), v (3), w (3)]   (18 per sample, 1 kHz)
    output (6)  = commanded wrench [f (3) N, tau (3) Nm]

All 18-D quantities are expressed in the TCP frame (z along the peg, into the hole), as in the
authors' dataset (f_z, v_z > 0 while pushing in):
    F_ext: wrench the robot exerts on the environment at the TCP (= -contact wrench on hand + peg),
    F_in:  wrench commanded by the controller (impedance part + diffusion feed-forward),
    v, w:  TCP twist.

Episode: FORGE reset (random socket pose) -> impedance approach: peg tip `hover` above the hole,
offset laterally by xy_err in a random direction (closed loop on the tip), guarded descent until contact -> pre-press:
ramp a push of `f_push` in (the real skill starts pressing; in the dataset F_in_z ~= action_z, and
the model largely keeps the current push) -> diffusion phase for `time_max` s. Control law in the
pre-press and diffusion phases (per 1 ms tick):
    tau = J^T [ K_xy (x_0 - x) + K_rot (q_0 ⊖ q) - D v + R F_ff ] + null-space posture + gravity/Coriolis
with springs only laterally and in rotation around the handover pose (none along z), and F_ff the
model output (re-sampled every `infer_every` ticks, smoothed by the paper's dynamic-system filter, Eq. 11). Success = FORGE success (peg tip centered and
within 1 mm of the hole floor). Peg/hole friction defaults to 0.3 (see `friction`).

Usage (`conda activate dqn`, from any directory):
    python projects/tacdiff/run.py mode=offline                     # check the weights on the test set
    python projects/tacdiff/run.py episodes=5                       # viewer, 5 insertions
    python projects/tacdiff/run.py render=false episodes=50 xy_err=0.001
    python projects/tacdiff/run.py model=/path/to/TacDiffusion_model_512.pth n_hidden=512 log=true
    python projects/tacdiff/run.py model=../diffusion/TacDiffusion/output/TacDiffusion_model_512.onnx   # authors' model

Keys (key=value): see DEFAULTS below.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]                       # .../RL_Mujoco
TACDIFF = REPO_ROOT / "diffusion" / "TacDiffusion"

DEFAULTS = {
    "mode": "sim",                 # sim | offline
    "model": str(TACDIFF / "output" / "TacDiffusion_model_512_e1500.pth"),
    "repo": str(TACDIFF),          # for helper_functions.models (and dataset/ in offline mode)
    "n_hidden": 512,               # must match training (1_model_train.py)
    "n_T": 50,
    "extra_steps": 8,              # extra denoising steps at t=1 (as in the ONNX export's forward)
    "threads": 2,                  # torch CPU threads (batch-1 inference is latency bound)
    "episodes": 5,
    "seed": 0,
    "render": True,
    "realtime": True,
    "time_max": 10.0,              # diffusion phase length [s] (skill time_max in the paper's setup)
    "hist": 7,                     # history offset [ticks @ 1 kHz]; the dataset uses 7
    "infer_every": 7,              # ticks between model samples (paper Table II: the 512 model runs at ~142 Hz)
    "filter": "ds",                # ds: paper Eq. 11 dynamic-system filter | lowpass: first order | none
    "ds_alpha": 0.9,               # Eq. 11: F_ff'' = alpha (beta (F_df - F_ff) - F_ff'), per 1 kHz tick (paper:
    "ds_beta": 0.3,                #   alpha = 0.9, beta = 0.3 -> ~80 Hz, damping ~0.87)
    "filter_tc": 0.02,             # lowpass time constant [s]
    "clip": True,                  # clip the model wrench to the training action range
    "xy_err": 0.0005,              # lateral start error: this distance, random direction [m] (radial clearance 0.25 mm)
    "xy_err_from_env": False,      # use the env's hole-estimate error (forge_reset_fixed pos_noise_*) instead of xy_err
    "hole_radius": 0.00425,        # hole radius [m] (peg radius 4 mm: 0.00425 = 0.5 mm diametrical clearance)
    "force_noise_mean": 0.0,       # Gaussian noise added to the sensed contact force (TCP frame, per axis, per tick) [N]
    "force_noise_std": 0.0,        #   std [N]
    "frame_yaw": 0.0,              # rotation of the model's I/O frame about the TCP z axis [deg] (the real
                                   # Franka EE x/y axes vs this TCP site are not documented)
    "hover": 0.002,                # peg-tip height above the hole tip before descending [m]
    "friction": 0.3,               # peg/hole sliding friction (scene default 0.75 needs > 6 N to slide at 8 N push,
                                   # beyond the +-4 N lateral range the model was trained on); None = keep scene
    "contact_force": 1.0,          # guarded-descent stop force [N]
    "damp_lin": 45.0,              # translational damping during the diffusion phase [Ns/m]
    "lat_kp": 500.0,               # lateral (x, y) stiffness around the handover pose [N/m] (paper K_x = 500)
    "rot_kp": 100.0,               # rotational stiffness holding gripper-down [Nm/rad] (paper K_x rot = 100)
    "f_push": 8.0,                 # push-down force ramped in before handover [N] (paper skill f_push = 8)
    "pre_press": 0.3,              # ramp + hold time of f_push before the model takes over [s]
    "log": False,                  # save per-tick traces to projects/tacdiff/runs/<time>.npz
    "offline_samples": 2000,
}

# Range of the authors' training actions (robot_action_*.pkl), used for `clip`.
ACTION_LO = np.array([-4.0, -4.0, 1.977, -2.0, -1.734, -3.361])
ACTION_HI = np.array([4.0, 4.0, 10.0, 2.0, 1.734, 3.361])
SIM_DT = 0.001


# ---------------------------------------------------------------------------------------- model
def load_model(path: str, repo: str, n_hidden: int, n_T: int, extra_steps: int, threads: int = 0):
    """Load a policy: a `.pth` state dict (torch) or an exported `.onnx` (onnxruntime, e.g. the authors' models).

    Args:
        path: `.pth` or `.onnx` file.
        repo: TacDiffusion repository root (for `.pth`).
        n_hidden: Hidden width used in training (`.pth` only).
        n_T: Diffusion steps used in training (`.pth` only).
        extra_steps: Extra denoising steps at t = 1 (`.pth` only; baked into an `.onnx` export, default 8).
        threads: onnxruntime intra-op threads (0 = its default, all cores).

    Returns:
        Callable mapping states (n, 36) float32 -> wrenches (n, 6).
    """
    if path.endswith(".onnx"):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads > 0:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
        sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        return lambda x: sess.run(["output"], {"input": x})[0]
    sys.path.insert(0, repo)
    from helper_functions.models import Model_Cond_Diffusion, Model_mlp_diff_embed

    x_dim, y_dim = 36, 6
    net = Model_mlp_diff_embed(x_dim, n_hidden, y_dim, embed_dim=128, net_type="fc", use_prev=True)
    model = Model_Cond_Diffusion(net, betas=(1e-4, 0.02), n_T=n_T, device="cpu", x_dim=x_dim, y_dim=y_dim,
                                 drop_prob=0.0, guide_w=0.0)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return lambda x: sample(model, torch.from_numpy(x), extra_steps).numpy()


@torch.no_grad()
def sample(model: torch.nn.Module, x: torch.Tensor, extra_steps: int) -> torch.Tensor:
    """DDPM sampling, identical to `Model_Cond_Diffusion.forward` (guide_w = 0) without its print.

    Args:
        model: Loaded model.
        x: States, shape (n, 36).
        extra_steps: Additional denoising iterations at t = 1.

    Returns:
        Wrenches, shape (n, 6).
    """
    n = x.shape[0]
    y = torch.randn(n, model.y_dim)
    mask = torch.zeros(n)
    for k in range(model.n_T, -extra_steps, -1):
        i = max(k, 1)
        t = torch.full((n, 1), i / model.n_T)
        z = torch.randn_like(y) if i > 1 else 0
        eps = model.nn_model(y, x, t, mask)
        y = model.oneover_sqrta[i] * (y - eps * model.mab_over_sqrtmab[i]) + model.sqrt_beta_t[i] * z
    return y


def offline_check(model, cfg: dict) -> None:
    """Compare predictions with the recorded expert wrench on random test-set samples.

    Args:
        model: Policy from `load_model`.
        cfg: Resolved config.
    """
    import pickle

    ds = Path(cfg["repo"]) / "dataset"
    with open(ds / "robot_state_test.pkl", "rb") as f:
        s = pickle.load(f)
    with open(ds / "robot_action_test.pkl", "rb") as f:
        a = pickle.load(f)
    idx = np.random.default_rng(cfg["seed"]).choice(len(s), size=min(cfg["offline_samples"], len(s)), replace=False)
    t0 = time.perf_counter()
    pred = model(s[idx].astype(np.float32))
    dt = time.perf_counter() - t0
    truth = a[idx]
    mae = np.abs(pred - truth).mean(0)
    base = np.abs(truth - a.mean(0)).mean(0)
    np.set_printoptions(precision=3, suppress=True)
    print(f"{len(idx)} test samples, batch inference {dt:.2f} s")
    print("             [f_x   f_y   f_z   t_x   t_y   t_z]")
    print(f"model MAE    {mae}")
    print(f"mean-baseline{base}   (predicting the dataset mean)")
    print("The model should be well below the baseline; if not, n_hidden / weights do not match.")


# ---------------------------------------------------------------------------------------- sim
class WrenchSensing:
    """Contact wrench on the hand subtree (hand, fingers, peg) and TCP-frame state assembly."""

    def __init__(self, env, frame_yaw_deg: float = 0.0, noise_mean: float = 0.0, noise_std: float = 0.0,
                 rng: np.random.Generator | None = None) -> None:
        """Cache ids and buffers.

        Args:
            env: A `forge_peg` TorqueEnv.
            frame_yaw_deg: Rotation of the model frame about the TCP z axis [deg].
            noise_mean: Mean of the Gaussian noise added to the sensed contact force (TCP frame, per axis) [N].
            noise_std: Its std [N] (0 and mean 0 = noise-free sensing).
            rng: Noise generator.
        """
        self.noise_mean, self.noise_std = float(noise_mean), float(noise_std)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        import mujoco

        self._mj = mujoco
        self.m, self.d, self.plant = env.plant.model, env.plant.data, env.plant
        ee = env.ctx.handles.ee_body_id
        sub = []
        for b in range(self.m.nbody):
            p = b
            while p not in (0, ee):
                p = self.m.body_parentid[p]
            if p == ee:
                sub.append(b)
        self.in_hand = np.zeros(self.m.nbody, dtype=bool)
        self.in_hand[sub] = True
        self._f6 = np.zeros(6)
        self.f_ext = np.zeros(6)        # robot on environment, world frame, about the TCP
        self.state = np.zeros(18)
        c, s = np.cos(np.radians(frame_yaw_deg)), np.sin(np.radians(frame_yaw_deg))
        self._Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self._R = np.zeros((3, 3))

    def frame(self) -> np.ndarray:
        """Model frame -> world rotation (TCP frame rotated by `frame_yaw` about its z).

        Returns:
            Owned buffer, shape (3, 3).
        """
        np.dot(self.plant.ee_rot.reshape(3, 3), self._Rz, out=self._R)
        return self._R

    def contact_wrench(self) -> np.ndarray:
        """Wrench the robot exerts on the environment at the TCP, world frame.

        Returns:
            Owned buffer, shape (6,).
        """
        m, d, f = self.m, self.d, self.f_ext
        f.fill(0.0)
        tcp = self.plant.ee_pos
        for i in range(d.ncon):
            c = d.contact[i]
            h1, h2 = self.in_hand[m.geom_bodyid[c.geom1]], self.in_hand[m.geom_bodyid[c.geom2]]
            if h1 == h2:
                continue
            self._mj.mj_contactForce(m, d, i, self._f6)
            fw = c.frame.reshape(3, 3).T @ self._f6[:3]       # force of geom1 on geom2, world frame
            on_env = fw if h1 else -fw
            f[:3] += on_env
            f[3:] += np.cross(c.pos - tcp, on_env)
        return f

    def build_state(self, f_in_world: np.ndarray) -> np.ndarray:
        """Assemble [F_ext, F_in, v, w] in the TCP frame.

        Args:
            f_in_world: Commanded task-space wrench (without gravity), world frame, shape (6,).

        Returns:
            Owned buffer, shape (18,).
        """
        R = self.frame()
        fe = self.contact_wrench()
        tw = self.plant.ee_vel()
        s = self.state
        s[0:3], s[3:6] = R.T @ fe[:3], R.T @ fe[3:]
        if self.noise_std > 0.0 or self.noise_mean != 0.0:   # force-sensor noise (forces only, sensor frame)
            s[0:3] += self.rng.normal(self.noise_mean, self.noise_std, size=3)
        s[6:9], s[9:12] = R.T @ f_in_world[:3], R.T @ f_in_world[3:]
        s[12:15], s[15:18] = R.T @ tw[:3], R.T @ tw[3:]
        return s


def run_episode(env, model, sens: WrenchSensing, cfg: dict, rng: np.random.Generator, viewer,
                seed: int | None = None) -> dict:
    """Approach with the impedance controller, then insert with the diffusion policy.

    Args:
        env: `forge_peg` TorqueEnv (sim_dt = 1 ms).
        model: Loaded diffusion model.
        sens: Sensing helper.
        cfg: Resolved config.
        rng: Episode RNG (start offset).
        viewer: The env's viewer (`sync()` / `is_running()`).
        seed: Env reset seed (None = drawn from `rng`).

    Returns:
        Episode metrics (and traces if `log`). f_mean: mean contact force over the policy phase; f_max_50ms:
        max of its 50 ms moving average (comparable to FORGE's step-averaged F_max); f_max: max single tick.
    """
    env.reset(seed=int(rng.integers(1 << 31)) if seed is None else int(seed))
    ctx, plant, ctrl = env.ctx, env.plant, env.controller
    st = ctx.state["forge"]
    ctx.episode_step = 0                                 # makes ForgeState.update use the instantaneous force
    st.update(ctx)
    tick = plant.control_step
    sync_every = 16                                      # ~60 Hz viewer refresh

    def run(n: int, fn) -> None:
        for k in range(n):
            tick(fn)
            if k % sync_every == 0:
                viewer.sync()

    # --- approach: impedance, setpoint moved at <= 5 cm/s to the hover pose (+ lateral error)
    kp_app = np.array([600.0, 600.0, 600.0, 100.0, 100.0, 100.0])
    ctrl.set_target(kp=kp_app)                           # Kd = 2 sqrt(Kp)
    tip = plant.data.site_xpos[st.peg_tip]
    tcp_minus_tip = plant.ee_pos - tip
    ang = rng.uniform(0.0, 2.0 * np.pi)
    offset = np.array([cfg["xy_err"] * np.cos(ang), cfg["xy_err"] * np.sin(ang), cfg["hover"]])
    if cfg["xy_err_from_env"]:   # aim at the env's (noisy) hole estimate, like a policy using that estimate
        offset[:2] = (ctx.state["fixed_anchor"] - st.hole_tip)[:2]
    goal = st.hole_tip + offset + tcp_minus_tip
    start = plant.ee_pos.copy()
    n_move = max(1, int(np.linalg.norm(goal - start) / 0.05 / SIM_DT))
    for k in range(n_move):
        ctrl.set_target(pos=start + (goal - start) * (k + 1) / n_move)
        run(1, ctrl.torque)
    # Hover and guarded descent with integral correction of the peg-TIP lateral error: the impedance
    # controller alone leaves ~0.1 mm oscillation at hover and drifts ~0.3 mm laterally while descending.
    aim = st.hole_tip[:2] + offset[:2]
    pos_d = goal.copy()

    def track(fn_z) -> bool:
        pos_d[:2] -= 0.005 * (plant.data.site_xpos[st.peg_tip][:2] - aim)   # ~0.2 s time constant
        fn_z()
        ctrl.set_target(pos=pos_d)
        run(1, ctrl.torque)
        st.update(ctx)
        return st.force_norm > cfg["contact_force"] or st.tip_height < st.hole_tip[2] - st.hole_floor[2] - 0.001

    run(1000, ctrl.torque)                               # let the approach transient die out
    calm = 0
    for _ in range(4000):                                # hover until the tip is aligned for 100 ms
        track(lambda: None)
        err = plant.data.site_xpos[st.peg_tip][:2] - aim
        calm = calm + 1 if err @ err < 3e-5 ** 2 else 0
        if calm >= 100:
            break

    def lower() -> None:
        pos_d[2] -= 0.005 * SIM_DT                       # 5 mm/s

    for _ in range(4000):                                # guarded descent until contact
        if track(lower):
            break
    run(100, ctrl.torque)                                # let the contact impact settle

    # --- pre-press (ramp f_push in, as the real skill starts pressing) then diffusion phase.
    # Same law in both: lateral + rotational springs around the handover pose, no z stiffness,
    # plus the feed-forward wrench. In the dataset F_in_z ~= action_z (no z spring) while F_in_xy
    # and F_in_tau differ from the action (springs present).
    kp = np.array([cfg["lat_kp"], cfg["lat_kp"], 0.0] + [cfg["rot_kp"]] * 3)
    kd = np.array([cfg["damp_lin"]] * 3 + [2.0 * np.sqrt(cfg["rot_kp"])] * 3)
    ctrl.set_target(pos=plant.ee_pos.copy(), quat=st.home_quat, kp=kp, kd=kd)
    push = np.array([0.0, 0.0, cfg["f_push"], 0.0, 0.0, 0.0])
    n_pre = int(cfg["pre_press"] / SIM_DT)
    lim = plant.torque_limits
    hist, every = int(cfg["hist"]), int(cfg["infer_every"])
    filt = cfg["filter"]
    if filt not in ("ds", "lowpass", "none"):
        raise ValueError("filter must be 'ds', 'lowpass' or 'none'")
    alpha = 1.0 if filt == "none" else min(1.0, SIM_DT / cfg["filter_tc"])
    ds_a, ds_b = float(cfg["ds_alpha"]), float(cfg["ds_beta"])
    F_rate = np.zeros(6)                                 # F_ff' of the ds filter (per tick), zero at handover
    ring = np.zeros((hist + 1, 18))                      # ring[-1] = newest
    F_target = push.copy()                               # model output, TCP frame
    F_filt = np.zeros(6)
    F_ff_world = np.zeros(6)
    f_in_world = ctrl._F.copy()                          # commanded wrench of the last approach tick
    tau = np.zeros(plant.n)
    x_in = np.zeros((1, 36), dtype=np.float32)
    infer_ms, f_max, t_success = [], 0.0, -1.0
    f_hist = []                                          # contact force per tick of the policy phase
    traces = {"state": [], "wrench": [], "tip": []} if cfg["log"] else None
    n_steps = int(cfg["time_max"] / SIM_DT)

    def torque_fn(p) -> np.ndarray:
        tau[:] = ctrl.torque(p)
        J = p.jacobian()
        tau[:] += J.T @ F_ff_world
        np.clip(tau, -lim, lim, out=tau)
        return tau

    n_ramp = max(1, 2 * n_pre // 3)
    for k in range(-n_pre, n_steps):                     # k < 0: pre-press, k >= 0: diffusion policy
        # sense (state at the start of the tick, commanded wrench of the previous tick)
        s = sens.build_state(f_in_world)
        ring[:-1] = ring[1:]
        ring[-1] = s
        if k >= 0 and k % every == 0:
            x_in[0, :18] = ring[-1]
            x_in[0, 18:] = ring[0]                       # hist ticks earlier
            t0 = time.perf_counter()
            F_target[:] = model(x_in)[0]
            infer_ms.append(1e3 * (time.perf_counter() - t0))
            if cfg["clip"]:
                np.clip(F_target, ACTION_LO, ACTION_HI, out=F_target)
        if k < 0:
            F_filt[:] = push * min(1.0, (k + n_pre + 1) / n_ramp)
        elif filt == "ds":                               # Eq. 11, discretized per tick
            F_rate += ds_a * (ds_b * (F_target - F_filt) - F_rate)
            F_filt += F_rate
        else:
            F_filt += alpha * (F_target - F_filt)
        R = sens.frame()
        F_ff_world[:3], F_ff_world[3:] = R @ F_filt[:3], R @ F_filt[3:]

        tick(torque_fn)
        f_in_world[:] = ctrl._F + F_ff_world             # impedance wrench (Kp e - Kd v) + feed-forward

        st.update(ctx)
        f_max = max(f_max, st.force_norm)
        if k >= 0:
            f_hist.append(st.force_norm)
        if traces is not None:
            traces["state"].append(s.copy())
            traces["wrench"].append(F_filt.copy())
            traces["tip"].append([st.xy_dist, st.tip_height])
        if k % sync_every == 0:
            viewer.sync()
            if not viewer.is_running():
                break
        if st.success and k >= 0:
            t_success = (k + 1) * SIM_DT
            break

    fh = np.asarray(f_hist) if f_hist else np.zeros(1)
    win = min(50, len(fh))                               # 50 ms at 1 kHz
    out = {"success": st.success, "t_success": t_success, "xy_err_mm": 1e3 * st.xy_dist,
           "f_mean": float(fh.mean()), "f_max_50ms": float(np.convolve(fh, np.ones(win) / win, "valid").max()),
           "tip_height_mm": 1e3 * st.tip_height, "f_max": f_max, "start_offset_mm": 1e3 * offset[:2],
           "infer_ms": float(np.mean(infer_ms)) if infer_ms else float("nan")}
    if traces is not None:
        out["traces"] = {k: np.asarray(v) for k, v in traces.items()}
    return out


def make_sim_env(cfg: dict, overrides: dict | None = None):
    """`forge_peg` env for this harness: 1 ms physics, FORGE randomization off, harness friction and hole size.

    The FORGE task randomizes gains, a controller dead zone, friction, the in-hand offset and yaw by
    default; all of that is switched off (`forge.task.NO_DR`), since this harness runs its own wrench control.

    Args:
        cfg: Resolved config (render, realtime, friction, hole_radius).
        overrides: Extra env overrides (applied last), e.g. hole-estimate error for `xy_err_from_env`.

    Returns:
        The env.
    """
    import forge  # noqa: F401  (registers forge_peg)
    from forge.task import NO_DR
    from mujoco_rl_bed.tasks.registry import make_env

    ov = {**NO_DR, "sim_dt": str(SIM_DT), "render": str(cfg["render"]).lower(),
          "realtime": str(cfg["realtime"]).lower(), "task.scene.assets.1.inner_radius": str(cfg["hole_radius"]),
          "task.events.reset_fixed.params.pos_noise_std": "0.0"}
    if cfg["friction"] is not None:   # set by the reset event (it would overwrite a one-off model edit)
        ov["task.events.randomize_friction.params.lo"] = ov["task.events.randomize_friction.params.hi"] = \
            str(float(cfg["friction"]))
    ov.update(overrides or {})
    return make_env("forge_peg", ov)


def main(argv: list[str]) -> None:
    """Parse key=value args and run the offline check or the simulation.

    Args:
        argv: key=value arguments.
    """
    from mujoco_rl_bed.utils.config import _literal, parse_cli

    cfg = dict(DEFAULTS)
    for k, v in parse_cli(argv).items():
        if k not in cfg:
            raise KeyError(f"unknown key '{k}' (valid: {sorted(cfg)})")
        val = _literal(v)
        cfg[k] = v if isinstance(DEFAULTS[k], str) else (None if val is None else type(DEFAULTS[k])(val))
    torch.set_num_threads(int(cfg["threads"]))
    if not Path(cfg["model"]).is_file():
        raise FileNotFoundError(f"model not found: {cfg['model']} (copy it from the training PC, or pass model=...)")

    # Load torch/the model BEFORE creating the env: torch imports segfault once the viewer thread runs.
    model = load_model(cfg["model"], cfg["repo"], int(cfg["n_hidden"]), int(cfg["n_T"]), int(cfg["extra_steps"]),
                       int(cfg["threads"]))
    print(f"loaded {cfg['model']} (n_hidden={cfg['n_hidden']}, n_T={cfg['n_T']})")
    if cfg["mode"] == "offline":
        offline_check(model, cfg)
        return
    if cfg["mode"] != "sim":
        raise ValueError("mode must be 'sim' or 'offline'")

    env = make_sim_env(cfg)
    viewer = env._viewer                                  # stepping the plant directly, so sync ourselves
    rng = np.random.default_rng(int(cfg["seed"]))
    sens = WrenchSensing(env, float(cfg["frame_yaw"]), cfg["force_noise_mean"], cfg["force_noise_std"], rng)
    results = []
    try:
        for ep in range(int(cfg["episodes"])):
            r = run_episode(env, model, sens, cfg, rng, viewer)
            results.append(r)
            print(f"ep {ep:3d}  success={r['success']!s:5}  t={r['t_success']:6.3f}s  "
                  f"xy={r['xy_err_mm']:.2f}mm  h={r['tip_height_mm']:6.2f}mm  Fmax={r['f_max']:6.1f}N  "
                  f"start={np.round(r['start_offset_mm'], 2)}mm  infer={r['infer_ms']:.1f}ms")
            if cfg["log"]:
                run_dir = HERE / "runs"
                run_dir.mkdir(exist_ok=True)
                path = run_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_ep{ep}.npz"
                np.savez(path, **r.pop("traces"))
                print(f"  traces -> {path}")
            if not viewer.is_running():
                break
    finally:
        env.close()
    if results:
        succ = np.array([r["success"] for r in results])
        ts = [r["t_success"] for r in results if r["success"]]
        print(f"\nsuccess {succ.sum()}/{len(succ)} ({100 * succ.mean():.0f}%)"
              + (f", mean time {np.mean(ts):.2f}s" if ts else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
