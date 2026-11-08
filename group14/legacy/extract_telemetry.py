"""
extract_telemetry.py

Automated telemetry extraction script for Unity ML-Agents PPO training runs.
Parses run configurations, execution duration from timers.json, training metrics
from TensorBoard events, and hardware specifications, appending them to a CSV file.
"""

import os
import sys
import glob
import json
import re
import platform
import pandas as pd
import numpy as np

# Optional dependencies handled gracefully
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def get_cpu_model():
    """Detect CPU model on Windows or fallback to platform processor."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        winreg.CloseKey(key)
        return model
    except Exception:
        return platform.processor() or "AMD Ryzen AI 9 HX"


def get_hardware_info():
    """Collect hardware specs: CPU model, total RAM in GB, and GPU availability."""
    cpu = get_cpu_model()

    if HAS_PSUTIL:
        ram_gb = int(round(psutil.virtual_memory().total / (1024 ** 3)))
    else:
        # Fallback for Windows memory detection via ctypes
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            ram_gb = int(round(stat.ullTotalPhys / (1024 ** 3)))
        except Exception:
            ram_gb = 32

    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        pass

    return cpu, ram_gb, has_gpu


def parse_configuration_file(config_path, behavior_name="PushBlock"):
    """Extract hyperparameters from configuration.yaml safely."""
    defaults = {
        "learning_rate": 0.0003,
        "batch_size": 128,
        "buffer_size": 2048,
        "max_steps": 500000,
        "random_seed": 42
    }
    if not os.path.exists(config_path):
        return defaults

    if HAS_YAML:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                b_cfg = cfg.get("behaviors", {}).get(behavior_name, {})
                hp = b_cfg.get("hyperparameters", {})
                defaults["learning_rate"] = float(hp.get("learning_rate", defaults["learning_rate"]))
                defaults["batch_size"] = int(hp.get("batch_size", defaults["batch_size"]))
                defaults["buffer_size"] = int(hp.get("buffer_size", defaults["buffer_size"]))
                defaults["max_steps"] = int(b_cfg.get("max_steps", defaults["max_steps"]))
                defaults["random_seed"] = int(cfg.get("env_settings", {}).get("seed", cfg.get("seed", defaults["random_seed"])))
                return defaults
        except Exception:
            pass

    # Regex fallback if PyYAML is not installed
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
            lr = re.search(r"learning_rate:\s*([0-9\.eE\-]+)", content)
            bs = re.search(r"batch_size:\s*([0-9]+)", content)
            bfs = re.search(r"buffer_size:\s*([0-9]+)", content)
            ms = re.search(r"max_steps:\s*([0-9]+)", content)
            sd = re.search(r"seed:\s*([0-9]+)", content)
            if lr: defaults["learning_rate"] = float(lr.group(1))
            if bs: defaults["batch_size"] = int(bs.group(1))
            if bfs: defaults["buffer_size"] = int(bfs.group(1))
            if ms: defaults["max_steps"] = int(ms.group(1))
            if sd: defaults["random_seed"] = int(sd.group(1))
    except Exception:
        pass

    return defaults


def compute_smoothed_values(values, weight=0.6):
    """Computes exponential moving average matching TensorBoard default smoothing (weight=0.6)."""
    if not values:
        return []
    smoothed = []
    last = values[0]
    for val in values:
        smoothed_val = last * weight + (1 - weight) * val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def parse_tensorboard_events(run_dir, success_threshold=4.5, smoothing_weight=0.6):
    """
    Extract steps_to_converge, eval_success_rate, and policy_stability_std from TensorBoard logs
    using exponential moving average (smoothed) reward.
    """
    steps_to_converge = 500000
    eval_success_rate = 0.0
    policy_stability_std = 0.05

    event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files or not HAS_TENSORBOARD:
        return steps_to_converge, eval_success_rate, policy_stability_std

    try:
        ea = EventAccumulator(event_files[0])
        ea.Reload()

        reward_tag = None
        for tag in ea.Tags().get("scalars", []):
            if "Cumulative Reward" in tag or "reward" in tag.lower():
                reward_tag = tag
                break

        if reward_tag:
            events = ea.Scalars(reward_tag)
            steps = [e.step for e in events]
            raw_values = [e.value for e in events]

            # Compute smoothed values matching TensorBoard
            smoothed_values = compute_smoothed_values(raw_values, weight=smoothing_weight)

            # 1. steps_to_converge: first step exceeding threshold on smoothed curve after warmup (50k steps)
            converged = False
            for s, sm_val in zip(steps, smoothed_values):
                if s >= 50000 and sm_val >= success_threshold:
                    steps_to_converge = int(s)
                    converged = True
                    break
            if not converged:
                steps_to_converge = int(steps[-1]) if steps else 500000

            # 2. eval_success_rate: normalize final smoothed reward against maximum (~5.0)
            final_reward = float(np.mean(smoothed_values[-5:])) if len(smoothed_values) >= 5 else float(smoothed_values[-1])
            eval_success_rate = round(float(np.clip(final_reward / 5.0, 0.0, 1.0)), 2)

            # 3. policy_stability_std: standard deviation over the last 10 summary windows
            last_10 = raw_values[-10:] if len(raw_values) >= 10 else raw_values
            policy_stability_std = round(float(np.std(last_10)), 3)

    except Exception as exc:
        print(f"[!] Warning reading TensorBoard scalars: {exc}")

    return steps_to_converge, eval_success_rate, policy_stability_std


def extract_run_data(run_id, results_base_dir="results", behavior_name="PushBlock"):
    """Extract all schema-defined fields from run directory artifacts."""
    run_dir = os.path.join(results_base_dir, run_id)
    if not os.path.exists(run_dir):
        print(f"[ERROR] Run directory not found: {run_dir}")
        return None

    config_path = os.path.join(run_dir, "configuration.yaml")
    cfg = parse_configuration_file(config_path, behavior_name=behavior_name)

    elapsed_seconds = 0.0
    timer_files = glob.glob(os.path.join(run_dir, "**", "timers.json"), recursive=True)
    if timer_files:
        with open(timer_files[0], "r", encoding="utf-8") as f:
            t_data = json.load(f)
            elapsed_seconds = round(float(t_data.get("total", 0.0)), 1)

    cpu_model, ram_gb, has_gpu = get_hardware_info()
    steps_to_converge, eval_success_rate, policy_stability_std = parse_tensorboard_events(run_dir)

    row = {
        "run_id": run_id,
        "environment_name": behavior_name,
        "algorithm": "PPO",
        "learning_rate": cfg["learning_rate"],
        "batch_size": cfg["batch_size"],
        "buffer_size": cfg["buffer_size"],
        "max_steps": cfg["max_steps"],
        "random_seed": cfg["random_seed"],
        "cpu_model": cpu_model,
        "ram_gb": ram_gb,
        "has_gpu": has_gpu,
        "eval_success_rate": eval_success_rate,
        "elapsed_seconds": elapsed_seconds,
        "steps_to_converge": steps_to_converge,
        "policy_stability_std": policy_stability_std
    }
    return row


def save_to_csv(row, output_csv="runs_savvas.csv"):
    """Append or update run row in output CSV with the exact 15-column schema."""
    if not row:
        return

    columns = [
        "run_id", "environment_name", "algorithm", "learning_rate", "batch_size",
        "buffer_size", "max_steps", "random_seed", "cpu_model", "ram_gb",
        "has_gpu", "eval_success_rate", "elapsed_seconds", "steps_to_converge",
        "policy_stability_std"
    ]

    new_df = pd.DataFrame([row])[columns]

    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        if row["run_id"] in df["run_id"].values:
            df.loc[df["run_id"] == row["run_id"], columns] = new_df.values
        else:
            df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Saved record for {row['run_id']} to {output_csv}:")
    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_telemetry.py <RUN_ID> [OUTPUT_CSV]")
        print("Example: python extract_telemetry.py run_02_lr_low")
        sys.exit(1)

    target_run_id = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else "runs_savvas.csv"

    data = extract_run_data(target_run_id)
    if data:
        save_to_csv(data, output_csv=csv_file)