"""
extract_telemetry_v2.py - one extractor for every ML-Agents PushBlock run in a results folder.

Reads existing run artifacts only, so no retraining is needed. It keeps the original
15 schema columns and adds the columns needed to compare and merge runs safely.
A run that cannot be read is reported and left out. It never receives default values.

Run this on the machine that produced the runs, because hardware is detected here:
  python extract_telemetry_v2.py --results results --source savvas --out runs_savvas_v2.csv

All runs are put on one common time grid (default 50,000 steps), so runs logged with different
summary_freq values give comparable metrics. A coarse grid point is the plain mean of the finer
summaries inside it, which is a close approximation of the coarser logged value.

Check seeds and times against an old CSV:
  python extract_telemetry_v2.py --results results --source savvas --out runs_savvas_v2.csv --compare runs_savvas.csv

Reading another person's runs? Give their hardware explicitly:
  ... --cpu-model "Their CPU" --ram-gb 16 --cpu-cores 8 --no-gpu
"""
import argparse
import glob
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REWARD_TAG = "Environment/Cumulative Reward"
LENGTH_TAG = "Environment/Episode Length"
THRESHOLD = 4.5        # ML-Agents benchmark mean reward for PushBlock
WARMUP_STEPS = 50000
GRID_STEPS = 50000     # common time grid; Riccardo's runs were logged at this resolution
SUSTAIN_STEPS = 100000 # reward must stay at or above THRESHOLD for this long
FINAL_STEPS = 100000   # span of the final reward
STABILITY_STEPS = 200000  # span of policy_stability_std
REWARD_SCALE = 5.0     # assumed maximum episode reward; confirm against PushAgentBasic.cs
COMPLETE_FRACTION = 0.98
ROLLING_STEPS = 150000 # span of each rolling variance window (SQ2.2)
CATASTROPHIC_DROP = 1.0  # reward fall between consecutive summaries; proposed value, agree it as a team


class ExtractionError(Exception):
    pass


def to_grid(series, grid):
    """Average {step: value} into grid bins ending at multiples of grid. Returns steps, values."""
    bins = {}
    for step, value in series.items():
        end = int(np.ceil(step / float(grid))) * grid
        bins.setdefault(end, []).append(value)
    steps = sorted(bins)
    return steps, [float(np.mean(bins[s])) for s in steps]


def detect_hardware(args):
    cpu = args.cpu_model
    if cpu is None:
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            winreg.CloseKey(key)
        except Exception:
            cpu = None
    if cpu is None and os.path.isfile("/proc/cpuinfo"):
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    if cpu is None and platform.system() == "Darwin":
        import subprocess
        try:
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            cpu = None
    if not cpu:
        raise ExtractionError("CPU model not detected: pass --cpu-model")

    psutil = None
    try:
        import psutil
    except ImportError:
        pass

    ram = args.ram_gb
    if ram is None:
        if psutil is None:
            raise ExtractionError("psutil is missing: run 'pip install psutil' or pass --ram-gb")
        ram = int(round(psutil.virtual_memory().total / 1024 ** 3))

    cores = args.cpu_cores
    if cores is None:
        cores = psutil.cpu_count(logical=False) if psutil else None
        if cores is None:
            raise ExtractionError("Core count not detected: pass --cpu-cores")

    if args.no_gpu:
        gpu = False
    else:
        try:
            import torch
            gpu = bool(torch.cuda.is_available())
        except ImportError:
            raise ExtractionError("torch not importable: pass --no-gpu if this machine has no GPU")
    return {"cpu_model": cpu, "ram_gb": ram, "cpu_cores": cores, "has_gpu": gpu}


def read_config(run_dir, behavior):
    path = os.path.join(run_dir, "configuration.yaml")
    if not os.path.isfile(path):
        raise ExtractionError("configuration.yaml missing")
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    try:
        beh = cfg["behaviors"][behavior]
        hp = beh["hyperparameters"]
        env = cfg.get("env_settings") or {}
        engine = cfg.get("engine_settings") or {}
        net = beh.get("network_settings") or {}
        return {
            "learning_rate": float(hp["learning_rate"]),
            "batch_size": int(hp["batch_size"]),
            "buffer_size": int(hp["buffer_size"]),
            "max_steps": int(float(beh["max_steps"])),
            "random_seed": int(env.get("seed", -1)),
            "num_epoch": hp.get("num_epoch"),
            "learning_rate_schedule": hp.get("learning_rate_schedule"),
            "time_horizon": beh.get("time_horizon"),
            "summary_freq": beh.get("summary_freq"),
            "hidden_units": net.get("hidden_units"),
            "num_layers": net.get("num_layers"),
            "time_scale": engine.get("time_scale"),
            "no_graphics": engine.get("no_graphics"),
            "num_envs": env.get("num_envs"),
            "num_areas": env.get("num_areas"),
            "env_file": os.path.basename(str(env.get("env_path"))) if env.get("env_path") else "EDITOR",
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ExtractionError("bad configuration.yaml: %r" % (exc,))


def read_timers(run_dir):
    files = glob.glob(os.path.join(run_dir, "**", "timers.json"), recursive=True)
    if len(files) != 1:
        raise ExtractionError("expected 1 timers.json, found %d" % len(files))
    with open(files[0], "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not data.get("total"):
        raise ExtractionError("timers.json has no total time")
    meta = data.get("metadata") or {}
    return {
        "elapsed_seconds": round(float(data["total"]), 1),
        "start_time_seconds": meta.get("start_time_seconds"),
        "mlagents_version": meta.get("mlagents_version"),
    }


def read_series(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))
    if not files:
        raise ExtractionError("no TensorBoard event file")
    rewards, lengths = {}, {}
    for path in files:
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags().get("scalars", [])
        if REWARD_TAG in tags:
            for event in acc.Scalars(REWARD_TAG):
                rewards[event.step] = event.value
        if LENGTH_TAG in tags:
            for event in acc.Scalars(LENGTH_TAG):
                lengths[event.step] = event.value
    if not rewards:
        raise ExtractionError("tag '%s' not found" % REWARD_TAG)
    return rewards, lengths


def windows(span, grid):
    return max(1, int(round(span / float(grid))))


def reward_metrics(rewards, lengths, max_steps, grid):
    last_step = max(rewards)
    steps, values = to_grid(rewards, grid)
    n_final = windows(FINAL_STEPS, grid)
    n_sustain = windows(SUSTAIN_STEPS, grid)
    n_stability = windows(STABILITY_STEPS, grid)
    n_rolling = windows(ROLLING_STEPS, grid)

    first_crossing = next((s for s, v in zip(steps, values) if s >= WARMUP_STEPS and v >= THRESHOLD), None)
    sustained, streak, start = None, 0, None
    for step, value in zip(steps, values):
        if step >= WARMUP_STEPS and value >= THRESHOLD:
            streak += 1
            start = step if streak == 1 else start
            if streak == n_sustain:
                sustained = start
                break
        else:
            streak = 0

    final_reward = float(np.mean(values[-n_final:]))
    x, y = np.asarray(steps, dtype=float), np.asarray(values, dtype=float)
    # Trapezoid area written out so it works on NumPy 1.x and 2.x
    auc = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2) / (x[-1] - x[0])) if len(x) > 1 else float(y[-1])

    kept = [v for s, v in zip(steps, values) if s >= WARMUP_STEPS]
    variances = [float(np.var(kept[i:i + n_rolling])) for i in range(len(kept) - n_rolling + 1)]
    peak, drawdown = -np.inf, 0.0
    for value in kept:
        peak = max(peak, value)
        drawdown = max(drawdown, peak - value)

    final_length = np.nan
    if lengths:
        _, grid_lengths = to_grid(lengths, grid)
        final_length = round(float(np.mean(grid_lengths[-n_final:])), 2)

    return {
        "eval_success_rate": round(float(np.clip(final_reward / REWARD_SCALE, 0, 1)), 4),
        "steps_to_converge": sustained if sustained is not None else np.nan,
        "policy_stability_std": round(float(np.std(values[-n_stability:])), 4),
        "converged": sustained is not None,
        "steps_first_crossing": first_crossing if first_crossing is not None else np.nan,
        "final_reward": round(final_reward, 4),
        "reward_auc": round(auc, 4),
        "rolling_var_mean": round(float(np.mean(variances)), 4) if variances else np.nan,
        "max_drawdown": round(float(drawdown), 4) if kept else np.nan,
        "catastrophic_drops": sum(1 for a, b in zip(kept, kept[1:]) if a - b >= CATASTROPHIC_DROP),
        "final_episode_length": final_length,
        "last_step": last_step,
        "n_summaries": len(steps),
        "grid_steps": grid,
        "completed": last_step >= COMPLETE_FRACTION * max_steps,
    }


def extract_run(run_id, run_dir, source, hardware, behavior, grid):
    cfg = read_config(run_dir, behavior)
    timers = read_timers(run_dir)
    rewards, lengths = read_series(run_dir)
    metrics = reward_metrics(rewards, lengths, cfg["max_steps"], grid)
    row = {"run_id": "%s__%s" % (source, run_id), "source": source, "environment_name": behavior, "algorithm": "PPO"}
    row.update(cfg)
    row.update(hardware)
    row.update(timers)
    row.update(metrics)
    row["seed_recorded"] = cfg["random_seed"] != -1
    return row


COLUMNS = [
    "run_id", "environment_name", "algorithm", "learning_rate", "batch_size", "buffer_size", "max_steps",
    "random_seed", "cpu_model", "ram_gb", "has_gpu", "eval_success_rate", "elapsed_seconds",
    "steps_to_converge", "policy_stability_std",
    "source", "cpu_cores", "converged", "steps_first_crossing", "final_reward", "reward_auc",
    "rolling_var_mean", "max_drawdown", "catastrophic_drops", "final_episode_length", "last_step",
    "n_summaries", "grid_steps", "completed", "seed_recorded", "start_time_seconds", "num_epoch", "learning_rate_schedule",
    "time_horizon", "summary_freq", "hidden_units", "num_layers", "time_scale", "no_graphics", "num_envs",
    "num_areas", "env_file", "mlagents_version",
]


def compare(new, old_csv, source):
    old = pd.read_csv(old_csv)
    old["run_id"] = source + "__" + old["run_id"].astype(str)
    both = new.merge(old, on="run_id", suffixes=("", "_old"))
    print("\nCOMPARISON with %s: matched %d of %d old rows" % (old_csv, len(both), len(old)))
    problems = 0
    for _, row in both.iterrows():
        issues = []
        if abs(row["elapsed_seconds"] - row["elapsed_seconds_old"]) > 0.11:
            issues.append("elapsed %.1f vs %.1f" % (row["elapsed_seconds"], row["elapsed_seconds_old"]))
        if "random_seed_old" in row and row["random_seed"] != row["random_seed_old"]:
            issues.append("SEED real %d vs CSV %d" % (row["random_seed"], row["random_seed_old"]))
        if issues:
            problems += 1
            print("  DIFF %s: %s" % (row["run_id"], "; ".join(issues)))
    print("  Rows with differences: %d" % problems)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--source", required=True, help="short name of the person or machine, e.g. savvas")
    parser.add_argument("--out", required=True)
    parser.add_argument("--behavior", default="PushBlock")
    parser.add_argument("--compare", help="old CSV: checks seeds and elapsed time")
    parser.add_argument("--grid", type=int, default=GRID_STEPS, help="common time grid in steps")
    parser.add_argument("--cpu-model")
    parser.add_argument("--ram-gb", type=int)
    parser.add_argument("--cpu-cores", type=int)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    if os.path.abspath(args.out) == os.path.abspath(args.compare or ""):
        sys.exit("--out must differ from --compare, so the old CSV is kept")
    hardware = detect_hardware(args)
    run_dirs = sorted(d for d in glob.glob(os.path.join(args.results, "*"))
                      if os.path.isfile(os.path.join(d, "configuration.yaml")))
    if not run_dirs:
        sys.exit("No runs with configuration.yaml found in %s" % args.results)

    rows, failures = [], []
    for run_dir in run_dirs:
        run_id = os.path.basename(run_dir)
        try:
            rows.append(extract_run(run_id, run_dir, args.source, hardware, args.behavior, args.grid))
        except Exception as exc:
            failures.append(run_id)
            print("[FAILED] %s: %s" % (run_id, exc))

    if not rows:
        sys.exit("Every run failed")
    frame = pd.DataFrame(rows)[COLUMNS]
    frame.to_csv(args.out, index=False)
    print("Extracted %d runs to %s; failed: %d; incomplete: %d; unseeded: %d; not converged: %d" % (
        len(frame), args.out, len(failures), (~frame["completed"]).sum(),
        (~frame["seed_recorded"]).sum(), (~frame["converged"]).sum()))
    if args.compare:
        compare(frame, args.compare, args.source)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()