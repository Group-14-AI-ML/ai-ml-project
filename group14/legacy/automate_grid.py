"""
automate_grid.py

Automated batch runner for Unity ML-Agents PPO hyperparameter grid search.
Iterates through hyperparameter configurations, skips any runs already present
in runs_savvas.csv, generates temporary configs, executes training, and logs
telemetry automatically upon completion.
"""

import os
import sys
import subprocess
import yaml
import pandas as pd
import glob

# Import our existing verified telemetry extractor
import extract_telemetry


CSV_OUTPUT_PATH = "runs_savvas.csv"
UNITY_EXE_PATH = os.path.join("build", "UnityEnvironment.exe")
BASE_CONFIG_PATH = os.path.join("config", "ppo", "PushBlock.yaml")

# Define the experimental grid for Task C1
EXPERIMENTS_GRID = [
    # Seed 42 - Learning Rate & Batch variations
    {"run_id": "test_run_01",         "lr": 0.0003, "batch": 128, "buffer": 2048, "seed": 42},
    {"run_id": "run_02_lr_low",       "lr": 0.0001, "batch": 128, "buffer": 2048, "seed": 42},
    {"run_id": "run_03_lr_high",      "lr": 0.0010, "batch": 128, "buffer": 2048, "seed": 42},
    {"run_id": "run_04_batch_small",   "lr": 0.0003, "batch": 64,  "buffer": 1024, "seed": 42},
    {"run_id": "run_05_batch_large",   "lr": 0.0003, "batch": 256, "buffer": 2048, "seed": 42},
    
    # Remaining combinations to complete the 3x3 grid (Seed 42)
    {"run_id": "run_06_lr_low_bs_small",   "lr": 0.0001, "batch": 64,  "buffer": 1024, "seed": 42},
    {"run_id": "run_07_lr_low_bs_large",   "lr": 0.0001, "batch": 256, "buffer": 2048, "seed": 42},
    {"run_id": "run_08_lr_high_bs_small",  "lr": 0.0010, "batch": 64,  "buffer": 1024, "seed": 42},
    {"run_id": "run_09_lr_high_bs_large",  "lr": 0.0010, "batch": 256, "buffer": 2048, "seed": 42},

    # Seed 123 - Repeat key baseline and extreme configs for stochasticity check
    {"run_id": "run_10_seed_baseline",      "lr": 0.0003, "batch": 128, "buffer": 2048, "seed": 123},
    {"run_id": "run_11_seed_lr_low",        "lr": 0.0001, "batch": 128, "buffer": 2048, "seed": 123},
    {"run_id": "run_12_seed_lr_high",       "lr": 0.0010, "batch": 128, "buffer": 2048, "seed": 123},
]


def get_completed_run_ids():
    """Retrieve list of already executed run IDs from CSV to avoid duplicate work."""
    if not os.path.exists(CSV_OUTPUT_PATH):
        return set()
    try:
        df = pd.read_csv(CSV_OUTPUT_PATH)
        if "run_id" in df.columns:
            return set(df["run_id"].astype(str).tolist())
    except Exception as exc:
        print(f"[!] Warning reading {CSV_OUTPUT_PATH}: {exc}")
    return set()


def create_temp_config(base_path, lr, batch, buffer, max_steps=500000):
    """Generate a run-specific YAML config with requested hyperparameters."""
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Update hyperparameters for PushBlock behavior
    b_cfg = cfg.setdefault("behaviors", {}).setdefault("PushBlock", {})
    hp = b_cfg.setdefault("hyperparameters", {})
    hp["learning_rate"] = float(lr)
    hp["batch_size"] = int(batch)
    hp["buffer_size"] = int(buffer)
    b_cfg["max_steps"] = int(max_steps)

    temp_cfg_path = os.path.join("config", "temp_current_run.yaml")
    with open(temp_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return temp_cfg_path


def run_training(run_id, cfg_path, seed, time_scale=20):
    """Invoke mlagents-learn via subprocess with headless Unity build."""
    cmd = [
        sys.executable, "-m", "mlagents.trainers.learn",
        cfg_path,
        f"--run-id={run_id}",
        f"--seed={seed}",
        f"--time-scale={time_scale}",
        "--force"
    ]

    if os.path.exists(UNITY_EXE_PATH):
        cmd.extend([f"--env={UNITY_EXE_PATH}", "--no-graphics"])
        print(f"[INFO] Using Standalone Executable: {UNITY_EXE_PATH} (Headless Mode)")
    else:
        print(f"[WARNING] {UNITY_EXE_PATH} not found. Waiting for Unity Editor Play button...")

    print(f"\n[STARTING] Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)

    # Check if the training actually finished and produced telemetry logs
    results_dir = os.path.join("results", run_id)
    timers_exist = len(glob.glob(os.path.join(results_dir, "**", "timers.json"), recursive=True)) > 0
    events_exist = len(glob.glob(os.path.join(results_dir, "**", "events.out.tfevents.*"), recursive=True)) > 0

    return timers_exist or events_exist


def main():
    import glob
    completed_runs = get_completed_run_ids()
    print(f"[STATUS] Detected {len(completed_runs)} completed run(s) in {CSV_OUTPUT_PATH}: {completed_runs}")

    for idx, exp in enumerate(EXPERIMENTS_GRID, 1):
        run_id = exp["run_id"]

        if run_id in completed_runs:
            print(f"\n[SKIP] Experiment {idx}/{len(EXPERIMENTS_GRID)}: {run_id} is already completed.")
            continue

        print(f"\n{'='*70}")
        print(f"RUNNING EXPERIMENT {idx}/{len(EXPERIMENTS_GRID)}: {run_id}")
        print(f"Hyperparameters: LR={exp['lr']}, Batch={exp['batch']}, Buffer={exp['buffer']}, Seed={exp['seed']}")
        print(f"{'='*70}")

        temp_cfg = create_temp_config(BASE_CONFIG_PATH, exp["lr"], exp["batch"], exp["buffer"])
        success = run_training(run_id, temp_cfg, exp["seed"])

        if success:
            print(f"[SUCCESS] Training logs verified for {run_id}. Extracting telemetry...")
            row = extract_telemetry.extract_run_data(run_id)
            if row:
                extract_telemetry.save_to_csv(row, output_csv=CSV_OUTPUT_PATH)
                completed_runs.add(run_id)
        else:
            print(f"[ERROR] No logs found for {run_id}. Halting grid execution.")
            break

    print("\n[ALL DONE] All planned grid search runs have been processed!")


def main():
    completed_runs = get_completed_run_ids()
    print(f"[STATUS] Detected {len(completed_runs)} completed run(s) in {CSV_OUTPUT_PATH}: {completed_runs}")

    for idx, exp in enumerate(EXPERIMENTS_GRID, 1):
        run_id = exp["run_id"]

        # 1. Skip already executed experiments
        if run_id in completed_runs:
            print(f"\n[SKIP] Experiment {idx}/{len(EXPERIMENTS_GRID)}: {run_id} is already completed.")
            continue

        print(f"\n{'='*70}")
        print(f"RUNNING EXPERIMENT {idx}/{len(EXPERIMENTS_GRID)}: {run_id}")
        print(f"Hyperparameters: LR={exp['lr']}, Batch={exp['batch']}, Buffer={exp['buffer']}, Seed={exp['seed']}")
        print(f"{'='*70}")

        # 2. Create YAML configuration
        temp_cfg = create_temp_config(BASE_CONFIG_PATH, exp["lr"], exp["batch"], exp["buffer"])

        # 3. Execute training
        success = run_training(run_id, temp_cfg, exp["seed"])

        if success:
            print(f"[SUCCESS] Finished training for {run_id}. Extracting telemetry...")
            # 4. Extract telemetry and update CSV
            row = extract_telemetry.extract_run_data(run_id)
            if row:
                extract_telemetry.save_to_csv(row, output_csv=CSV_OUTPUT_PATH)
                completed_runs.add(run_id)
        else:
            print(f"[ERROR] Training failed for {run_id}. Halting grid execution.")
            break

    print("\n[ALL DONE] All planned grid search runs have been processed!")


if __name__ == "__main__":
    main()