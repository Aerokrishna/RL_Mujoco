"""Warm start for the spiral catch policy: behaviour-clone the scripted catch rule into a recurrent asymmetric policy.

PPO from scratch on the try-insert action learns to try everywhere (run G: 35/60 at 1M steps vs 49/60 for the plain
spiral). Starting from a copy of the scripted rule avoids that (51/60). Steps:

1. Create a run folder with the exact network (a 1-step `train.py` run: config.json, final_model.zip, vecnormalize.pkl).
2. Roll out `scripted.CatchRule` in `n_envs` envs (seeds from `seed`, away from the evaluation seeds 2000+) with fresh
   observation/return normalization statistics.
3. Actor: weighted MSE (tries x `w_pos`) of the Gaussian mean to the rule action (`no_try` when not trying, 1 while
   trying), or for a binary action (`compliant_peg_s2i`) weighted cross-entropy of the push/no-push logits, over
   whole episodes from a zero LSTM state. Critic: MSE to the discounted normalized return to the episode end.
4. Save final_model.zip and vecnormalize.pkl into the run folder. Fine-tune with PPO via `train.py init_from=<run>`.

Usage (`conda activate dqn`, from `mujoco_rl_bed/`):
    python projects/compliant_RL/bc_warmstart.py                          # compliant_peg_s2g, 30 episodes/env, 40 epochs
    python projects/compliant_RL/bc_warmstart.py task=compliant_peg_s2h run_name=s2h_bc
    python projects/compliant_RL/bc_warmstart.py task=compliant_peg_s2i run_name=s2i_bc     # binary push, 17 inputs

Keys (defaults in brackets): task [compliant_peg_s2g], run_name [s2g_bc], n_envs [10], episodes_per_env [30],
epochs [40], seed [5000], gamma [0.995], net_arch [[64,64]], lstm_hidden_size [128], log_std_init [-1.5],
no_try [-0.3; 0 for a binary action], w_pos [5; 50 for a binary action: push onsets are ~0.3 % of the steps (a push
mostly ends within 1-2 steps when the peg drops), and at 5x the classifier learnt never to push (recall 0)],
explore_push [0.002: binary action only; mean push probability on the rule's no-push steps after BC, set by a
temperature on the push logits that keeps every decision; a saturated BC policy (p ~ 1e-4) gave PPO entropy ~0.001 and
KL 0, i.e. no exploration and no learning].
The network keys must match the later PPO run (`train.py` checks them for `init_from`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULTS = {"task": "compliant_peg_s2g", "run_name": "s2g_bc", "n_envs": 10, "episodes_per_env": 30, "epochs": 40,
            "seed": 5000, "gamma": 0.995, "net_arch": "[64,64]", "lstm_hidden_size": 128, "log_std_init": -1.5,
            "no_try": -0.3, "w_pos": "auto", "explore_push": 0.002}


def collect(task: str, n: int, n_ep: int, seed: int, gamma: float, no_try: float, pd: int, phase_idx: int,
            discrete: bool):
    """Roll out the catch rule; return whole episodes (normalized obs, rule action, normalized return) + the venv.

    Episodes that end early (the `unstable` termination: the simulation blew up) are dropped, and the observation and
    return normalization statistics are fitted on the remaining episodes only: the steps before a blow-up carry huge
    forces and velocities that would otherwise dominate the running variance and squash the catch signal (server run:
    3 of 320 episodes, held-out push precision 0.3 vs 0.95 without them).
    """
    from stable_baselines3.common.running_mean_std import RunningMeanStd
    from stable_baselines3.common.vec_env import VecNormalize

    from compliant_RL.scripted import CatchRule
    from mujoco_rl_bed.wrappers.sb3 import make_vec_env

    venv = VecNormalize(make_vec_env(task, n, seed, {"obs_mode": "asymmetric"}, task_modules=("compliant_RL",)),
                        norm_obs=True, norm_reward=True, gamma=gamma)
    rules = [CatchRule(no_try=no_try, phase_idx=phase_idx) for _ in range(n)]
    venv.reset()
    raw = venv.get_original_obs()
    O, A, R, D, succ = [], [], [], [], []
    while len(succ) < n * n_ep:
        act = np.array([rules[i](raw[i, :pd]) for i in range(n)])
        O.append(raw.copy())
        A.append(act)
        _, _, done, infos = venv.step(act.astype(np.int64) if discrete else act[:, None].astype(np.float32))
        R.append(venv.get_original_reward().copy())
        D.append(done.copy())
        raw = venv.get_original_obs()
        for i in np.flatnonzero(done):
            rules[i].reset()
            succ.append(bool(infos[i].get("is_success", False)))
    O, A, R, D = map(np.array, (O, A, R, D))                      # (steps, n, ...)
    print(f"collected {len(succ)} episodes: rule success {np.mean(succ):.3f}, try steps {(A > 0).mean():.4f}", flush=True)
    raw_eps = []                                                  # (obs, action, reward) per whole episode
    for i in range(n):
        s = 0
        for e in np.flatnonzero(D[:, i]):
            raw_eps.append((O[s:e + 1, i], A[s:e + 1, i], R[s:e + 1, i]))
            s = e + 1
    full = max(len(e[0]) for e in raw_eps)
    kept = [e for e in raw_eps if len(e[0]) == full and np.isfinite(e[0]).all() and np.isfinite(e[2]).all()]
    print(f"[bc] dropped {len(raw_eps) - len(kept)} of {len(raw_eps)} episodes that ended early or are non-finite",
          flush=True)
    venv.obs_rms = RunningMeanStd(shape=O.shape[-1:])
    venv.obs_rms.update(np.concatenate([e[0] for e in kept]).astype(np.float64))
    venv.ret_rms = RunningMeanStd(shape=())
    for e in kept:                                                # discounted running return, as VecNormalize does
        ret, rets = 0.0, []
        for r in e[2]:
            ret = ret * gamma + float(r)
            rets.append(ret)
        venv.ret_rms.update(np.array(rets))
    print(f"[bc] obs scale (std): contact force xy {np.sqrt(venv.obs_rms.var[:2]).round(2)} N, "
          f"return std {np.sqrt(venv.ret_rms.var):.1f}", flush=True)
    eps = []
    for o, a, r in kept:
        on = venv.normalize_obs(o).astype(np.float32)
        rn = venv.normalize_reward(r)
        g, acc = np.zeros(len(rn)), 0.0
        for t in range(len(rn) - 1, -1, -1):
            acc = rn[t] + gamma * acc
            g[t] = acc
        eps.append((on, a, g))
    return eps, venv


def train_bc(pol, eps: list, epochs: int, w_pos: float, discrete: bool, explore_push: float = 0.002) -> float:
    """Fit the actor to the rule actions (Gaussian mean, or push/no-push logits when `discrete`) and the critic to the
    returns. Episodes are zero-padded to the longest one and the padding is masked out of the losses (an episode
    that ends early, e.g. on the `unstable` termination, must not truncate the others)."""
    import torch as th

    lens = np.array([len(e[0]) for e in eps])
    T = int(lens.max())
    print(f"[bc] {len(eps)} episodes, length {lens.min()}-{T} steps ({int((lens < T).sum())} ended early)", flush=True)

    def pad(x):
        return np.concatenate([x, np.zeros((T - len(x),) + x.shape[1:], dtype=np.float32)])

    Xo = th.tensor(np.stack([pad(e[0]) for e in eps]))
    Xa = th.tensor(np.stack([pad(np.asarray(e[1], dtype=np.float32)) for e in eps]))[..., None]
    Xg = th.tensor(np.stack([pad(np.asarray(e[2], dtype=np.float32)) for e in eps]))[..., None]
    Xm = th.tensor(np.stack([np.arange(T) < n for n in lens]), dtype=th.float32)[..., None]   # 1 = real step
    params = [p for name, p in pol.named_parameters() if name != "log_std"]
    opt = th.optim.Adam(params, lr=1e-3)
    H = pol.lstm_actor.hidden_size

    def forward(x):
        B = x.shape[0]
        pi_f, vf_f = pol.extract_features(x.reshape(-1, x.shape[-1]))
        z = th.zeros(pol.lstm_actor.num_layers, B, H)
        lp, _ = pol.lstm_actor(pi_f.reshape(B, T, -1).transpose(0, 1), (z, z))
        lv, _ = pol.lstm_critic(vf_f.reshape(B, T, -1).transpose(0, 1), (z, z))
        mean = pol.action_net(pol.mlp_extractor.forward_actor(lp.transpose(0, 1).reshape(B * T, -1)))
        val = pol.value_net(pol.mlp_extractor.forward_critic(lv.transpose(0, 1).reshape(B * T, -1)))
        return mean.reshape(B, T, -1), val.reshape(B, T, 1)

    def actor_loss(out, a, w):
        """Weighted, masked (w = 0 on padding) imitation loss."""
        if discrete:                                      # out: logits (B, T, 2), a: 0 / 1
            ce = th.nn.functional.cross_entropy(out.reshape(-1, 2), a.reshape(-1).long(), reduction="none")
            return (w.reshape(-1) * ce).sum() / w.sum()
        return (w * (out - a) ** 2).sum() / w.sum()

    E = len(Xo)
    n_val = max(E // 10, 1)
    tr, va = np.arange(E - n_val), np.arange(E - n_val, E)
    for ep in range(epochs):
        np.random.shuffle(tr)
        for k in range(0, len(tr), 16):
            b = tr[k:k + 16]
            mean, val = forward(Xo[b])
            m = Xm[b]
            w = (1.0 + (w_pos - 1.0) * (Xa[b] > 0).float()) * m
            loss = actor_loss(mean, Xa[b], w) + 0.5 * (m * (val - Xg[b]) ** 2).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            with th.no_grad():
                mean, val = forward(Xo[va])
                m = Xm[va] > 0
                pred = (mean[..., 1:] > mean[..., :1] if discrete else mean > 0.3) & m
                lab = (Xa[va] > 0) & m
                tp = float((pred & lab).sum())
                g, v = Xg[va][m], val[m]
                r2 = 1 - float(((v - g) ** 2).mean() / g.var())
                recall = tp / max(float(lab.sum()), 1)
                print(f"epoch {ep + 1}: held-out try recall {recall:.2f}, precision "
                      f"{tp / max(float(pred.sum()), 1):.2f}, value R2 {r2:.2f}", flush=True)
    if discrete:
        recall = calibrate_push(pol, forward, Xo, Xa, Xm, tr, va, explore_push)
    return recall


def calibrate_push(pol, forward, Xo, Xa, Xm, tr, va, explore_push: float) -> float:
    """Shift the push logit so that push decisions best match the rule on the training episodes (max F1), then soften
    the push probabilities for PPO exploration.

    The `w_pos` weighting makes the classifier find the rare push steps but inflates every borderline push
    probability (held-out precision ~0.2 at recall ~0.97 on the server data); the shift keeps the ranking and moves the
    decision threshold. Prints held-out recall / precision after the shift.

    Exploration: both logits are divided by a temperature T >= 1 chosen so that the mean push probability on the rule's
    no-push training steps is `explore_push`. Dividing by T keeps the sign of every logit difference, so the
    deterministic (argmax) policy is unchanged; only the sampled policy that PPO trains with gets softer.

    Returns:
        Held-out recall after the shift.
    """
    import torch as th

    def margins(idx):
        with th.no_grad():
            out = th.cat([forward(Xo[idx[k:k + 32]])[0] for k in range(0, len(idx), 32)])
        m = Xm[idx][..., 0] > 0
        return (out[..., 1] - out[..., 0])[m].numpy(), (Xa[idx][..., 0] > 0)[m].numpy()

    def pr(d, y, shift):
        pred = d > shift
        tp = float((pred & y).sum())
        return tp / max(float(y.sum()), 1.0), tp / max(float(pred.sum()), 1.0)

    d, y = margins(tr)
    shifts = np.linspace(0.0, 10.0, 201)
    f1 = [2 * r * q / max(r + q, 1e-9) for r, q in (pr(d, y, sh) for sh in shifts)]
    shift = float(shifts[int(np.argmax(f1))])
    pol.action_net.bias.data[1] -= shift
    dv, yv = margins(va)
    recall, precision = pr(dv, yv, 0.0)
    print(f"[bc] push threshold calibrated (logit shift {shift:.2f}): held-out try recall {recall:.2f}, "
          f"precision {precision:.2f}", flush=True)

    dn, dp = d[~y] - shift, d[y] - shift

    def p_neg(T):
        return float((1.0 / (1.0 + np.exp(-dn / T))).mean())

    lo, hi = 1.0, 1000.0
    if p_neg(lo) < explore_push:
        for _ in range(60):                               # p_neg grows with T (no-push margins are mostly negative)
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if p_neg(mid) < explore_push else (lo, mid)
    T = lo
    pol.action_net.weight.data /= T
    pol.action_net.bias.data /= T
    p_pos = float((1.0 / (1.0 + np.exp(-dp / T))).mean())
    print(f"[bc] exploration temperature {T:.2f}: mean push probability {p_neg(T):.4f} on no-push steps, "
          f"{p_pos:.2f} on push steps (sampled policy)", flush=True)
    return recall


def main(argv: list[str]) -> None:
    import gymnasium as gym
    import torch as th
    from sb3_contrib import RecurrentPPO

    from mujoco_rl_bed.env.torque_env import TorqueEnv
    from mujoco_rl_bed.rl.train import main as train_main
    from mujoco_rl_bed.tasks.registry import make_env_cfg
    from mujoco_rl_bed.utils.config import parse_cli

    a = {**DEFAULTS, **parse_cli(argv)}
    unknown = set(a) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown keys {sorted(unknown)} (valid: {sorted(DEFAULTS)})")
    th.set_num_threads(4)
    run = train_main([f"task={a['task']}", "algo=recurrent_ppo", "asymmetric=true", f"net_arch={a['net_arch']}",
                      f"lstm_hidden_size={a['lstm_hidden_size']}", f"log_std_init={a['log_std_init']}",
                      f"n_envs={a['n_envs']}", "n_steps=256", f"batch_size={256 * int(a['n_envs'])}",
                      f"gamma={a['gamma']}", "total_timesteps=1", f"run_name={a['run_name']}"],
                     task_modules=("compliant_RL",), defaults={"run_root": str(HERE / "runs")})
    model = RecurrentPPO.load(str(run / "final_model.zip"), device="cpu")
    discrete = isinstance(model.action_space, gym.spaces.Discrete)
    env = TorqueEnv(make_env_cfg(a["task"], {"obs_mode": "asymmetric"}))
    phase_idx = dict(env.obs_mgr.layout["policy"])["sp_phase"].start
    env.close()
    print(f"[bc] action {'binary' if discrete else 'continuous'}, actor obs {model.policy.actor_dim}, "
          f"phase index {phase_idx}", flush=True)
    eps, venv = collect(a["task"], int(a["n_envs"]), int(a["episodes_per_env"]), int(a["seed"]), float(a["gamma"]),
                        0.0 if discrete else float(a["no_try"]), model.policy.actor_dim, phase_idx, discrete)
    w_pos = (50.0 if discrete else 5.0) if a["w_pos"] == "auto" else float(a["w_pos"])
    recall = train_bc(model.policy, eps, int(a["epochs"]), w_pos, discrete, float(a["explore_push"]))
    if recall < 0.7:
        print(f"[bc] WARNING: held-out try recall {recall:.2f} < 0.7: the warm start does not imitate the rule; "
              f"raise w_pos (now {w_pos:g}) or epochs before fine-tuning", flush=True)
    model.save(str(run / "final_model.zip"))
    venv.save(str(run / "vecnormalize.pkl"))
    venv.close()
    print(f"warm start saved -> {run}\nfine-tune: python projects/compliant_RL/train.py task={a['task']} "
          f"algo=recurrent_ppo asymmetric=true net_arch={a['net_arch']} lstm_hidden_size={a['lstm_hidden_size']} "
          f"init_from={run.name} ...", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent))
    main(sys.argv[1:])
