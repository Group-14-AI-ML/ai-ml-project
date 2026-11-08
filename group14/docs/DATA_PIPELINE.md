# Group 14: PushBlock data pipeline

This guide explains how to run the missing experiments, extract results and merge everything into one dataset. Anyone in the group should be able to follow it alone.

**Goal:** build one table, `runs_master_v0.csv`, with one row per training run. All values are calculated the same way. The ML models use this table to predict a run's outcome from its settings.

## Current status

| Item | Status |
|---|---|
| Rows in `runs_master_v0.csv` | 59 |
| Savvas's runs | 12 complete: 7 headless build, 5 Unity Editor |
| Riccardo's runs | 46 complete, including the 11 missing runs; 40 after removing exact repeats |
| Usable reward runs | 52, from 28 configurations |
| Usable runtime runs | 24 (Riccardo 17, Savvas 7) |
| Project Plan coverage | All 9 LR x batch cells have 3 runs at buffer 2048 |
| Master dataset | Core settings match. Open items below are still unconfirmed |

`runs_master_v0.csv` is ready for first reward models. Do not use `success_rate_est` until Max Step and Decision Period are confirmed.

## The files

All files live in the `group14/` folder of the group repository:

| Folder | Contents |
|---|---|
| `group14/scripts/` | The scripts below |
| `group14/data/raw/` | One CSV per person, `runs_<name>_v2.csv` |
| `group14/data/master/` | The merged dataset, `runs_master_v0.csv` |
| `group14/docs/` | This guide and `HARDWARE_RUNS.md` |
| `group14/legacy/` | Old scripts and the old CSV, kept for history only |

| Script | Purpose |
|---|---|
| `run_missing.py` | Runs the 11 missing experiments (done) |
| `run_hardware_check.py` | Runs the same 9 experiments on another laptop; see `HARDWARE_RUNS.md` |
| `extract_telemetry_v2.py` | Reads a `results` folder and writes one CSV for one person |
| `merge_and_audit.py` | Merges the CSVs, checks them and marks which runs each model can use |

Do not use the old `extract_telemetry.py` or `sync_results.py`. The old scripts write default values when something fails, and `sync_results.py` guesses the seed from the run name.

Never edit values in a CSV by hand. Fix the cause and extract again.

## Step 1: Set up

Use the project's Python 3.10 virtual environment. Run every command from the root of the repository, the folder that contains `group14` and your `results` folder.

**Windows**

```
<venv_folder>\Scripts\activate
```

**Linux or macOS**

```
source <venv_folder>/bin/activate
```

If a package is missing:

```
pip install tensorboard pyyaml pandas psutil
```

Before running experiments:

- Plug in your laptop.
- Use normal or best performance mode, never low power.
- Close heavy apps, because run times are part of the dataset.
- Do not delete your `results` folder.

## Step 2: Run the missing experiments

**Only Riccardo needs this step.** If someone else takes it over, agree this with the group first.

All 11 runs use buffer size 2048. The other settings match `config/ppo/PushBlock.yaml`, and `max_steps` is 500,000.

| Learning rate | Batch size | Seeds |
|---|---|---|
| 0.0001 | 64 | 42, 2024 |
| 0.0001 | 256 | 101 |
| 0.0003 | 64 | 42, 101, 2024 |
| 0.0003 | 256 | 101, 2024 |
| 0.001 | 64 | 101, 2024 |
| 0.001 | 256 | 2024 |

Check the list:

```
python group14/scripts/run_missing.py --dry-run
```

Start the runs:

**Linux**

```
python group14/scripts/run_missing.py --env Project/PushBlock.x86_64
```

**Windows**

```
python group14/scripts/run_missing.py --env build\UnityEnvironment.exe
```

**macOS**

```
python group14/scripts/run_missing.py --env Project/PushBlock.app
```

Change the path if your build is somewhere else. If the script stops, run the same command again. Finished runs are skipped, and a run that stopped halfway restarts from the beginning.

When everything is done, `--dry-run` should show `already done, skipping` for all 11 runs.

## Step 3: Extract your results

Run this on the laptop that produced the runs, because it records that laptop's hardware. Replace `<name>` with your lowercase name.

```
python group14/scripts/extract_telemetry_v2.py --results results --source <name> --out group14/data/raw/runs_<name>_v2.csv
```

Example:

```
python group14/scripts/extract_telemetry_v2.py --results results --source riccardo --out group14/data/raw/runs_riccardo_v2.csv
```

If you see `torch not importable` and the laptop has no GPU for PyTorch, add `--no-gpu`.

### Check the result

- `Extracted` shows the number of runs you expected.
- `grid_steps` is `50000`.
- `cpu_model` shows the real CPU name, not `x86_64`.
- `completed` is `True` for finished runs.

Riccardo should have about 47 extracted runs: 36 existing and 11 new. His 3 known failures can be ignored: `RF-20261029_135053`, `RF-20261029_135101` and `RF-20261029_135108`.

Report any other `FAILED` run. Do not try to fix it inside the CSV.

## Step 4: Merge the CSVs

Make sure every `runs_<name>_v2.csv` is in `group14/data/raw/`, then run:

```
python group14/scripts/merge_and_audit.py group14/data/raw/runs_savvas_v2.csv group14/data/raw/runs_riccardo_v2.csv --out group14/data/master/runs_master_v0.csv
```

Add more `group14/data/raw/` CSVs to the command when other group members upload theirs.

## Step 5: Check the merge

Read the last line of the output.

- `VERDICT: settings match, data can be pooled`: the core training settings match. Finish the checks below before sharing the file.
- `VERDICT: STOP`: do not share or use the file. Read the `BLOCKER` lines and fix the cause.

Then check:

| Output section | What to check |
|---|---|
| `SETTINGS THAT MUST MATCH` | No `BLOCKER` lines |
| `DATA QUALITY` | Numbers look reasonable |
| `PROJECT PLAN COVERAGE` | Each cell should reach 3 where possible |
| `seed x source` | Seeds are the expected values |

LR 0.0001 with batch 128 may show 2 recorded seeds. It also has one run with no recorded seed, so it still has 3 runs in total.

These are expected to differ between laptops: `summary_freq`, `env_file`, `no_graphics`, `cpu_model`, `ram_gb` and `cpu_cores`.

## Still to confirm before modelling

- [ ] Max Step from the Unity Inspector
- [ ] Decision Period from the Unity Inspector
- [ ] Riccardo's low power mode ended around 10:30 on 29 Oct
- [ ] Unity Editor version on each laptop
- [ ] Whether the Windows and Linux builds came from the same project state

To find the two Unity values, open the PushBlock scene and select the blue agent cube.

If they are not 5000 and 5, add your real values to the merge command:

```
python group14/scripts/merge_and_audit.py group14/data/raw/runs_savvas_v2.csv group14/data/raw/runs_riccardo_v2.csv --out group14/data/master/runs_master_v0.csv --max-step <value> --decision-period <value>
```

## How to use the master CSV

Always filter the rows first.

| Task | Use rows where |
|---|---|
| Reward models and reward analysis | `use_for_reward == True` |
| Runtime models and runtime analysis | `use_for_time == True` |

Why rows are excluded:

- **Exact repeats:** the same settings and seed gave the same reward on one laptop, so the repeat adds no new reward information.
- **Incomplete runs:** training stopped early.
- **Low power runs:** rewards are valid, but the times are not comparable.
- **Unity Editor runs:** rewards stay in the dataset, but the times are not comparable.

### Main columns

| Purpose | Columns |
|---|---|
| Model inputs | `learning_rate`, `batch_size`, `buffer_size` |
| Hardware inputs for runtime | `source`, `cpu_cores`, `ram_gb` |
| Reward target | `final_reward` |
| Runtime target | `elapsed_seconds` |
| Convergence targets | `converged`, `steps_to_converge`, `reward_auc` |
| Stability for RQ2 | `policy_stability_std`, `rolling_var_mean`, `max_drawdown`, `catastrophic_drops` |
| Filtering | `use_for_reward`, `use_for_time`, `config_id` |

Do not use `random_seed` as a model input.

### Modelling steps

1. Use log scale for `learning_rate` and `batch_size`.
2. Start with a model that predicts the mean, then use a linear model and Random Forest, as promised in the Project Plan.
3. Keep all runs of one configuration together when testing. Use `LeaveOneGroupOut` from scikit-learn with `config_id` as the group.
4. Fit all scaling inside each fold.
5. Plot model error against 20, 40 and all available runs. If the error stops improving, more runs are unlikely to help much.

## How the metrics are calculated

All runs are placed on a common 50,000-step grid, because Savvas logged every 10,000 steps and Riccardo every 50,000.

| Column | Meaning |
|---|---|
| `final_reward` | Mean reward over the last 100,000 steps |
| `eval_success_rate` | `final_reward / 5`, clipped from 0 to 1 |
| `success_rate_est` | Estimated success rate, not validated yet |
| `converged` | Reward stayed at 4.5 or above for at least 100,000 steps |
| `steps_to_converge` | First step of that period; empty if the run did not converge |
| `reward_auc` | Average reward across training |
| `policy_stability_std` | Reward standard deviation over the last 200,000 steps |
| `rolling_var_mean` | Mean reward variance across rolling 150,000-step windows |
| `max_drawdown` | Largest reward fall after a peak |
| `catastrophic_drops` | Falls of at least 1.0 between consecutive grid points; this threshold is a proposal |

`eval_success_rate` is **not** the percentage of successful episodes. Call it normalized final reward in slides and reports.

## Changes from the Project Plan

Keep these visible in the report and presentation.

- **Rewards:** the code gives +5 for the goal and `-1 / MaxStep` per step, not +1 and -0.0025.
- **Success metric:** the Plan defines success over the last 100 episodes. The current metric uses reward over the last 100,000 steps.
- **Buffer size:** the Plan keeps it fixed, but the collected data also varies it.
- **Design:** the dataset combines structured runs and random runs.

## Known limitations

- Rewards vary strongly by seed, so no model can predict every run exactly.
- Runs are reproducible on one laptop, but the same seed can give different results on another laptop.
- Time differences between laptops can come from hardware, the operating system or the build. Two laptops cannot separate those causes.
- Do not describe all of Savvas's runs as headless, because the first 5 used the Unity Editor.
- 2 of Savvas's runs have no recorded seed. They are still usable, but they cannot be reproduced exactly.
- The learning rate schedule is linear, so the learning rate is close to zero at the end. This can make late training look more stable.

## Troubleshooting

| Problem | What to do |
|---|---|
| `mlagents-learn not found` | Activate the virtual environment |
| `Build not found` | Pass the correct path with `--env` |
| `torch not importable` | Add `--no-gpu` if PyTorch has no GPU on that laptop |
| `CPU model not detected` | Add `--cpu-model "Your CPU name"` |
| `STOP: a CSV came from an old script version` | Replace the script and extract again |
| `BLOCKER` | Do not merge the data until the cause is understood |
| A run fails after training | Run the same command again and check the output; the ONNX export step can fail at the end |

## Order of work

1. The group confirms the open items.
2. Ivana and Anna start preprocessing and the first models on `runs_master_v0.csv`.
3. The group builds the charts for the presentation.
4. The final data, scripts and this guide go into the public repository.
