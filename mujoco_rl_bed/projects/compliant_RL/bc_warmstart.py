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
mostly ends within 1-2 steps when the peg drops), and at 5x the classifier learnt never to push (recall 0)].
The network keys must match the later PPO run (`train.py` checks them for `init_from`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULTS = {"task": "compliant_peg_s2g", "run_name": "s2g_bc", "n_envs": 10, "episodes_per_env": 30, "epochs": 40,
            "seed": 5000, "gamma": 0.995, "net_arch": "[64,64]", "lstm_hidden_size": 128, "log_std_init": -1.5,
            "no_try": -0.3, "w_pos": "auto"}


def collect(task: str, n: int, n_ep: int, seed: int, gamma: float, no_try: float, pd: int, phase_idx: int,
            discrete: bool):
    """Roll out the catch rule; return whole episodes (normalized obs, rule action, normalized return) + the venv."""
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
    on = venv.normalize_obs(O.reshape(-1, O.shape[-1])).reshape(O.shape).astype(np.float32)
    rn = venv.normalize_reward(R.reshape(-1)).reshape(R.shape)
    eps = []
    for i in range(n):
        s = 0
        for e in np.flatnonzero(D[:, i]):
            g, acc = np.zeros(e + 1 - s), 0.0
            for t in range(e - s, -1, -1):
                acc = rn[s + t, i] + gamma * acc
                g[t] = acc
            eps.append((on[s:e + 1, i], A[s:e + 1, i], g))
            s = e + 1
    return eps, venv


def train_bc(pol, eps: list, epochs: int, w_pos: float, discrete: bool) -> float:
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
    recall = train_bc(model.policy, eps, int(a["epochs"]), w_pos, discrete)
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
