"""Machine check before a long FORGE training run: how free are the CPU and GPU, and which n_envs to use.

Usage (`conda activate dqn`, from the `mujoco_rl_bed` folder):
    python scripts/system_check.py                    # full check, about 5-10 minutes
    python scripts/system_check.py quick=true         # fewer n_envs, shorter end-to-end run (2-4 minutes)
    python scripts/system_check.py envs=8,16,32,48 e2e_envs=32 device=cuda:1

Sections:
  1. host      CPU model, cores usable by this process, load average, RAM, disk, busiest processes
  2. gpu       nvidia-smi memory / utilization sampled over a few seconds, compute processes and owners,
               torch's view (free memory per device), a short fp32 matmul throughput probe (a shared GPU
               shows up as low TFLOPS)
  3. sim       FORGE simulation throughput (random actions, no learner) vs. number of env processes
  4. e2e       a short real training run (the planned recurrent PPO network, Isaac Lab FORGE sizes) on
               the chosen device: steps/s with the learner, time per PPO update, peak GPU memory

Everything is printed and also written to `system_check_<host>_<time>.json`. Paste the printed summary
(or the JSON) back to choose the final training command. Nothing here is kept on disk except that file
(the end-to-end run uses a temporary folder).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from mujoco_rl_bed.utils.config import parse_cli

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# The network/PPO settings planned for the GPU run (Isaac Lab FORGE rl_games config, mapped to SB3).
PLANNED = {
    "algo": "recurrent_ppo", "asymmetric": "true", "n_steps": "128", "batch_size": "512", "n_epochs": "4",
    "learning_rate": "1e-4", "lr_schedule": "adaptive", "kl_threshold": "0.008", "gamma": "0.995",
    "gae_lambda": "0.95", "clip_range": "0.2", "clip_range_vf": "0.2", "vf_coef": "1.0", "ent_coef": "0.0",
    "max_grad_norm": "1.0", "log_std_init": "0.0", "lstm_hidden_size": "1024", "n_lstm_layers": "2",
    "net_arch": "(512,128,64)", "activation": "elu",
}


def sh(cmd: list[str], timeout: float = 20.0) -> str:
    """Run a command and return stdout ('' if it is missing or fails).

    Args:
        cmd: Command and arguments.
        timeout: Seconds.

    Returns:
        Stripped stdout.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def section(title: str) -> None:
    """Print a section header.

    Args:
        title: Header text.
    """
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)), flush=True)


# ---------------------------------------------------------------------------------- 1. host
def host_info() -> dict:
    """CPU, memory, disk and load.

    Returns:
        Dict of host facts.
    """
    info: dict = {"hostname": socket.gethostname(), "os": platform.platform(), "python": sys.version.split()[0]}
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info["cpu_model"] = cpu
    info["logical_cpus"] = os.cpu_count()
    info["usable_cpus"] = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    lscpu = sh(["lscpu"])
    for line in lscpu.splitlines():
        k, _, v = line.partition(":")
        if k.strip() in ("Core(s) per socket", "Socket(s)", "Thread(s) per core"):
            info[k.strip()] = v.strip()
    try:
        cores, sockets = int(info.get("Core(s) per socket", 0)), int(info.get("Socket(s)", 1))
        info["physical_cores"] = cores * sockets if cores else None
    except ValueError:
        info["physical_cores"] = None
    info["loadavg_1_5_15"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                mem[k] = round(int(v.split()[0]) / 1024 / 1024, 1)  # GiB
    except OSError:
        pass
    info["mem_gib"] = mem
    du = shutil.disk_usage(ROOT)
    info["disk_free_gib"] = round(du.free / 2**30, 1)
    me = os.environ.get("USER", "") or sh(["whoami"])
    info["user"] = me
    top = sh(["ps", "-eo", "user:32,pid,pcpu,pmem,etime,comm", "--sort=-pcpu"]).splitlines()[:13]
    info["top_processes"] = top
    other = 0.0
    for line in top[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[0] != me:
            try:
                other += float(parts[2])
            except ValueError:
                pass
    info["cpu_pct_used_by_other_users_top12"] = round(other, 1)
    return info


# ---------------------------------------------------------------------------------- 2. gpu
def gpu_info(samples: int = 10, interval: float = 0.5) -> dict:
    """nvidia-smi + torch view of the GPUs.

    Args:
        samples: nvidia-smi utilization samples.
        interval: Seconds between samples.

    Returns:
        Dict with per-GPU stats.
    """
    out: dict = {"nvidia_smi": bool(shutil.which("nvidia-smi"))}
    q = "index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu," \
        "power.draw,power.limit,driver_version"
    gpus: dict[int, dict] = {}
    if out["nvidia_smi"]:
        for k in range(samples):
            txt = sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
            for line in txt.splitlines():
                f = [x.strip() for x in line.split(",")]
                if len(f) < 11:
                    continue
                i = int(f[0])
                g = gpus.setdefault(i, {"name": f[1], "mem_total_mib": float(f[2]), "driver": f[10],
                                        "util": [], "mem_used_mib": [], "power_w": [], "temp_c": []})
                g["util"].append(float(f[5]) if f[5].replace(".", "").isdigit() else np.nan)
                g["mem_used_mib"].append(float(f[3]))
                try:
                    g["power_w"].append(float(f[8]))
                except ValueError:
                    pass
                g["temp_c"].append(float(f[7]) if f[7].isdigit() else np.nan)
                g["power_limit_w"] = f[9]
            if k < samples - 1:
                time.sleep(interval)
        for g in gpus.values():
            g["util_mean"] = round(float(np.nanmean(g["util"])), 1) if g["util"] else None
            g["util_max"] = float(np.nanmax(g["util"])) if g["util"] else None
            g["mem_used_mib_max"] = max(g["mem_used_mib"]) if g["mem_used_mib"] else None
            g["mem_free_mib_min"] = g["mem_total_mib"] - g["mem_used_mib_max"] if g["mem_used_mib"] else None
            g["power_w_mean"] = round(float(np.mean(g["power_w"])), 1) if g["power_w"] else None
            for k in ("util", "mem_used_mib", "power_w", "temp_c"):
                g.pop(k)
        apps = sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                   "--format=csv,noheader,nounits"])
        uuids = sh(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"])
        uuid_to_idx = {u.split(",")[1].strip(): int(u.split(",")[0]) for u in uuids.splitlines() if "," in u}
        procs = []
        for line in apps.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) < 4:
                continue
            owner = sh(["ps", "-o", "user:32=", "-p", f[1]]) or "?"
            procs.append({"gpu": uuid_to_idx.get(f[0], f[0]), "pid": f[1], "user": owner, "name": f[2],
                          "mem_mib": f[3]})
        out["compute_processes"] = procs
    out["gpus"] = gpus

    try:
        import torch
    except ImportError:
        out["torch"] = None
        return out
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["cuda_available"] = torch.cuda.is_available()
    tv = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            tv[i] = {"name": torch.cuda.get_device_name(i), "free_gib": round(free / 2**30, 2),
                     "total_gib": round(total / 2**30, 2), "matmul_tflops": matmul_probe(i)}
    out["torch_devices"] = tv
    return out


def matmul_probe(dev: int, n: int = 4096, seconds: float = 2.0) -> float | None:
    """Rough fp32 matmul throughput; low values on a big GPU mean someone else is using it.

    Args:
        dev: CUDA device index.
        n: Matrix size.
        seconds: Measurement time.

    Returns:
        TFLOPS, or None on error.
    """
    import torch

    try:
        d = torch.device(f"cuda:{dev}")
        a = torch.randn(n, n, device=d)
        b = torch.randn(n, n, device=d)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize(d)
        k, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            a @ b
            k += 1
            if k % 10 == 0:
                torch.cuda.synchronize(d)
        torch.cuda.synchronize(d)
        dt = time.perf_counter() - t0
        del a, b
        torch.cuda.empty_cache()
        return round(2 * n**3 * k / dt / 1e12, 2)
    except Exception as e:  # noqa: BLE001  (report, don't crash the check)
        print(f"  matmul probe on cuda:{dev} failed: {e}")
        return None


# ---------------------------------------------------------------------------------- 3. sim scan
def sim_scan(n_list: list[int], seconds: float) -> list[dict]:
    """FORGE simulation throughput with random actions (no learner) per number of env processes.

    Args:
        n_list: n_envs values.
        seconds: Measurement time per value (after a warm-up).

    Returns:
        [{n_envs, steps_per_s, per_env}, ...].
    """
    from mujoco_rl_bed.wrappers.sb3 import make_vec_env

    res = []
    for n in n_list:
        venv = make_vec_env("forge_peg", n, seed=0, task_modules=("forge",))
        try:
            venv.reset()
            rng = np.random.default_rng(0)
            act = lambda: rng.uniform(-1, 1, (n, venv.action_space.shape[0])).astype(np.float32)  # noqa: E731
            for _ in range(20):  # warm-up
                venv.step(act())
            k, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < seconds:
                venv.step(act())
                k += 1
            sps = k * n / (time.perf_counter() - t0)
        finally:
            venv.close()
        res.append({"n_envs": n, "steps_per_s": round(sps), "per_env": round(sps / n, 1)})
        print(f"  n_envs={n:4d}: {sps:8.0f} env steps/s ({sps / n:6.1f} per env)", flush=True)
    return res


# ---------------------------------------------------------------------------------- 4. e2e
def e2e(n_envs: int, device: str, updates: int) -> dict:
    """Short real training run with the planned network.

    Args:
        n_envs: Env processes.
        device: Torch device.
        updates: PPO updates to run (the first one includes start-up overhead).

    Returns:
        Measured fps, time per update, peak GPU memory.
    """
    import torch

    from mujoco_rl_bed.rl.train import main as train_main

    n_steps = int(PLANNED["n_steps"])
    batch = int(PLANNED["batch_size"])
    while (n_steps * n_envs) % batch:  # keep the check's own run valid for any n_envs
        batch //= 2
    total = n_steps * n_envs * updates
    tmp = Path(tempfile.mkdtemp(prefix="forge_syscheck_"))
    args = [f"{k}={v}" for k, v in PLANNED.items()] + [
        f"n_envs={n_envs}", f"device={device}", f"batch_size={batch}", f"total_timesteps={total}",
        f"run_root={tmp}", "verbose=0", "checkpoint_every=100000000"]
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    t0 = time.perf_counter()
    try:
        run = train_main(args, task_modules=("forge",), defaults={"task": "forge_peg"})
        dt = time.perf_counter() - t0
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(str(next((run / "tb").iterdir())))
        ea.Reload()
        fps = [e.value for e in ea.Scalars("time/fps")] if "time/fps" in ea.Tags()["scalars"] else []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = {"n_envs": n_envs, "device": device, "batch_size": batch, "timesteps": total,
           "wall_s": round(dt, 1), "fps_overall": round(total / dt), "fps_sb3_last": round(fps[-1]) if fps else None,
           "s_per_update": round(dt / updates, 1)}
    if device.startswith("cuda"):
        out["peak_gpu_mem_gib"] = round(torch.cuda.max_memory_allocated(torch.device(device)) / 2**30, 2)
    return out


# ---------------------------------------------------------------------------------- main
def main(argv: list[str]) -> None:
    """Run the checks and write the JSON report.

    Args:
        argv: key=value options: quick, envs (comma list), e2e_envs, device, sim_seconds, e2e_updates, skip_e2e.
    """
    opt = parse_cli(argv)
    quick = opt.get("quick", "false").lower() in ("1", "true", "yes")
    report: dict = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "planned_train_cfg": PLANNED}

    section("1. host")
    h = host_info()
    report["host"] = h
    print(f"  {h['hostname']}  {h['cpu_model']}")
    print(f"  cpus: logical {h['logical_cpus']}, usable {h['usable_cpus']}, physical cores {h.get('physical_cores')}")
    print(f"  load avg (1/5/15 min): {h['loadavg_1_5_15']}   RAM GiB: {h['mem_gib']}   disk free: {h['disk_free_gib']} GiB")
    print(f"  CPU % used by other users (top processes): {h['cpu_pct_used_by_other_users_top12']}")
    for line in h["top_processes"]:
        print("    " + line)

    section("2. gpu")
    g = gpu_info(samples=6 if quick else 10)
    report["gpu"] = g
    if not g["gpus"]:
        print("  nvidia-smi: no GPU found")
    for i, d in g["gpus"].items():
        print(f"  [{i}] {d['name']}: mem used {d['mem_used_mib_max']:.0f}/{d['mem_total_mib']:.0f} MiB, "
              f"util mean {d['util_mean']}% max {d['util_max']}%, power {d['power_w_mean']}/{d['power_limit_w']} W")
    for p in g.get("compute_processes", []):
        print(f"    gpu {p['gpu']}: pid {p['pid']} user {p['user']} {p['name']} {p['mem_mib']} MiB")
    print(f"  torch {g.get('torch')} (CUDA {g.get('torch_cuda')}), cuda available: {g.get('cuda_available')}")
    for i, d in (g.get("torch_devices") or {}).items():
        print(f"  cuda:{i} {d['name']}: free {d['free_gib']}/{d['total_gib']} GiB, matmul ~{d['matmul_tflops']} TFLOPS")

    device = opt.get("device")
    if device is None:
        tv = g.get("torch_devices") or {}
        device = f"cuda:{max(tv, key=lambda i: tv[i]['free_gib'])}" if tv else "cpu"
    report["device_used"] = device

    section("3. sim throughput (random actions, no learner)")
    usable = int(h["usable_cpus"] or 1)
    if "envs" in opt:
        n_list = [int(x) for x in opt["envs"].split(",") if x]
    else:
        cand = [4, 8, 16, 24, 32, 48, 64, 96, 128]
        n_list = [n for n in cand if n <= max(4, usable)]
        if usable not in n_list and usable > 4:
            n_list.append(usable)
        if quick:
            n_list = sorted({n_list[0], n_list[len(n_list) // 2], n_list[-1]})
    sim = sim_scan(n_list, seconds=float(opt.get("sim_seconds", 4 if quick else 8)))
    report["sim_scan"] = sim
    best = max(sim, key=lambda r: r["steps_per_s"])
    knee = next(r for r in sim if r["steps_per_s"] >= 0.9 * best["steps_per_s"])
    report["sim_best"], report["sim_knee_90pct"] = best, knee
    print(f"  best: n_envs={best['n_envs']} ({best['steps_per_s']} steps/s); 90% of best from n_envs={knee['n_envs']}")

    if opt.get("skip_e2e", "false").lower() not in ("1", "true", "yes"):
        section(f"4. end-to-end training probe on {device} (planned network: LSTM 2x1024, MLP 512-128-64)")
        e2e_n = [int(x) for x in opt["e2e_envs"].split(",")] if "e2e_envs" in opt else sorted({knee["n_envs"], best["n_envs"]})
        updates = int(opt.get("e2e_updates", 2 if quick else 3))
        runs = []
        for n in e2e_n:
            try:
                r = e2e(n, device, updates)
            except Exception as e:  # noqa: BLE001  (report, keep going)
                r = {"n_envs": n, "device": device, "error": repr(e)}
            runs.append(r)
            print(f"  {r}", flush=True)
        report["e2e"] = runs

    out = ROOT / f"system_check_{socket.gethostname()}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    section("done")
    print(f"  report written to {out}")
    print("  Paste everything above (or the JSON file) back to get the final training command.")


if __name__ == "__main__":
    main(sys.argv[1:])
