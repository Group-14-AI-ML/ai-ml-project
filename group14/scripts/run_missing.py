"""
run_missing.py - runs the 11 PushBlock runs missing from the Project Plan design.

Run from the project folder with the venv active, laptop plugged in and on a normal power mode:
  python run_missing.py --dry-run     shows the list
  python run_missing.py               runs it; about 2 hours

Finished runs are skipped, so it is safe to stop with Ctrl+C and start it again.
Settings match config/ppo/PushBlock.yaml. Only learning rate, batch size and seed change.
"""
import argparse
import os
import shutil
import subprocess
import sys

TEMPLATE = """behaviors:
  PushBlock:
    trainer_type: ppo
    hyperparameters:
      batch_size: {batch}
      buffer_size: 2048
      learning_rate: {lr}
      beta: 0.01
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: false
      hidden_units: 256
      num_layers: 2
      vis_encode_type: simple
    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0
    keep_checkpoints: 5
    max_steps: 500000
    time_horizon: 64
    summary_freq: 10000
"""

RUNS = [  # learning rate, batch size, seed
    ("0.0001", 64, 42), ("0.0001", 64, 2024),
    ("0.0001", 256, 101),
    ("0.0003", 64, 42), ("0.0003", 64, 101), ("0.0003", 64, 2024),
    ("0.0003", 256, 101), ("0.0003", 256, 2024),
    ("0.001", 64, 101), ("0.001", 64, 2024),
    ("0.001", 256, 2024),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", default=os.path.join("build", "UnityEnvironment.exe"))
    parser.add_argument("--results", default="results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        if not os.path.isfile(args.env):
            sys.exit("Build not found: %s. Pass the right path with --env" % args.env)
        if shutil.which("mlagents-learn") is None:
            sys.exit("mlagents-learn not found: activate the venv first")
    os.makedirs("configs_missing", exist_ok=True)

    for index, (lr, batch, seed) in enumerate(RUNS, 1):
        run_id = "gap_lr%s_bs%d_buf2048_s%d" % (lr, batch, seed)
        run_dir = os.path.join(args.results, run_id)
        label = "[%d/%d] %s" % (index, len(RUNS), run_id)
        if os.path.isfile(os.path.join(run_dir, "run_logs", "timers.json")):
            print(label, "already done, skipping")
            continue
        if args.dry_run:
            print(label, "would run")
            continue
        config = os.path.join("configs_missing", run_id + ".yaml")
        with open(config, "w", encoding="utf-8") as handle:
            handle.write(TEMPLATE.format(batch=batch, lr=lr))
        print(label, "starting", flush=True)
        command = ["mlagents-learn", config, "--run-id=" + run_id, "--env=" + args.env, "--seed=%d" % seed,
                   "--no-graphics", "--time-scale=20", "--results-dir=" + args.results, "--force"]
        code = subprocess.run(command).returncode
        # The ONNX export can fail after training has finished, so judge the run by its files.
        if os.path.isfile(os.path.join(run_dir, "run_logs", "timers.json")):
            print(label, "finished" if code == 0 else "finished (exit code %d, check it in the extraction)" % code)
        else:
            sys.exit("%s failed with exit code %d and left no timers.json. Stopping." % (label, code))
    print("All done.")


if __name__ == "__main__":
    main()