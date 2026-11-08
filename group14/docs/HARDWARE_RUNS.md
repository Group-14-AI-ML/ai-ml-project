# Extra runs on your laptop: step-by-step guide

You'll run the same 9 PushBlock training runs that teammates run on their own laptops. We use these to study how hardware changes training time.

- **Your time:** about 1 hour to set up, the first time.
- **Laptop time:** about 1–2 hours of training, running on its own.
- **You don't need to understand or change any code.** Copy the commands exactly.

If you get stuck, take a screenshot of the terminal and send it to Savvas. Don't try to fix files by hand.

## What you need

Download these two files from the group repository, https://github.com/Group-14-AI-ML/ai-ml-project, in the folder `group14/scripts/`:

- `run_hardware_check.py`: runs the 9 trainings
- `extract_telemetry_v2.py`: turns the results into one CSV file

To download one, open it on GitHub and click the **Download raw file** button. Always use the repo version, not an older copy.

## Step 0: Do you already have the project set up?

If you completed the Phase 1 setup and `mlagents-learn --help` works in your virtual environment, go to **Step 5**. Otherwise, start at Step 1.

## Step 1: Install the tools

Install these, in this order:

1. **Git:** https://git-scm.com/install/
2. **Python 3.10.11:** https://www.python.org/downloads/release/python-31011/
   - On Windows, tick **Add python.exe to PATH** in the installer.
   - Use exactly version 3.10. Python 3.11 or 3.12 will not work.
3. **Unity Hub:** https://unity.com/download
4. **Unity Editor 2022.3.62f3:** install it from Unity Hub, under **Installs**. Add the build support module for your system:
   - Windows: **Windows Build Support (Mono)**
   - macOS: **Mac Build Support (Mono)**
   - Linux: **Linux Build Support (Mono)**

Check Python in a new terminal:

**Windows (Command Prompt)**

```
py -3.10 --version
```

**macOS or Linux**

```
python3.10 --version
```

It must print `Python 3.10.x`.

## Step 2: Download ML-Agents

Open a terminal in the folder where you keep your projects, then run:

```
git clone --branch fix-numpy-release-21-branch https://github.com/DennisSoemers/ml-agents.git
cd ml-agents
```

**From now on, run every command inside this `ml-agents` folder.**

## Step 3: Create and activate the virtual environment

**Windows (Command Prompt)**

```
py -3.10 -m venv venv
venv\Scripts\activate
```

**macOS or Linux**

```
python3.10 -m venv venv
source venv/bin/activate
```

The terminal line now starts with `(venv)`. **Activate it again every time you open a new terminal**, using the second command above.

## Step 4: Install the Python packages

Run these one at a time:

```
python -m pip install --upgrade pip
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
pip install "numpy>=1.23.5,<1.24.0" "protobuf==3.19.6" tensorboard pyyaml pandas psutil
```

The last line fixes two known version problems. Then check:

```
mlagents-learn --help
```

If it prints a list of options, the installation works. If it shows an error, see **Problems and fixes** at the end.

## Step 5: Build the PushBlock game

The training needs a standalone version of the game, called a build. You make it once.

1. Open **Unity Hub**, click **Add** (or **Open**) and choose the `Project` folder inside `ml-agents`.
2. Open the project with Unity **2022.3.62f3**. The first opening can take several minutes.
3. In the **Project** panel, open `Assets/ML-Agents/Examples/PushBlock/Scenes/` and double-click **PushBlock**.
4. **Don't change anything in the scene.**
5. Go to **File → Build Settings…**
6. Click **Add Open Scenes**. In **Scenes In Build**, make sure **only PushBlock** is ticked.
7. Choose **Windows, Mac, Linux** (or **PC, Mac & Linux Standalone**), select your system, and set Architecture to x86_64 or your Mac's chip.
8. Click **Build**, not Build And Run.
9. Inside the `ml-agents` folder, create a new folder called `build`. Save the build there with the name `PushBlock`.

Your build is now here:

| System | Path to use later |
|---|---|
| Windows | `build\PushBlock.exe` |
| macOS | `build/PushBlock.app` |
| Linux | `build/PushBlock.x86_64` |

**Linux only:**

```
chmod +x build/PushBlock.x86_64
```

**macOS only**, to stop the "app is damaged" warning:

```
xattr -dr com.apple.quarantine build/PushBlock.app
```

You can close Unity now. It must stay closed while training runs.

## Step 6: Add the two scripts

Copy `run_hardware_check.py` and `extract_telemetry_v2.py` into the `ml-agents` folder, next to the `config` and `build` folders:

```
ml-agents/
├── build/
├── config/
├── run_hardware_check.py
├── extract_telemetry_v2.py
└── venv/
```

## Step 7: Prepare the laptop

Run times are part of our data, so this step matters.

- **Plug in the charger.**
- **Turn off power saving:**
  - Windows: **Settings → System → Power & battery → Power mode → Best performance**
  - macOS: **System Settings → Battery → Low Power Mode → Never**
  - Linux: set the power profile to **Performance** or **Balanced**, never Power Saver
- **Stop the laptop from sleeping** for the next 2 hours, and don't close the lid.
- **Close other apps**, especially browsers, games and Unity.
- Don't use the laptop for heavy work while the runs are going.

## Step 8: Run the 9 trainings

Activate the virtual environment if needed (Step 3). Check the list first:

**Windows**

```
python run_hardware_check.py --env build\PushBlock.exe --dry-run
```

**macOS**

```
python run_hardware_check.py --env build/PushBlock.app --dry-run
```

**Linux**

```
python run_hardware_check.py --env build/PushBlock.x86_64 --dry-run
```

You should see 9 lines, from `[1/9]` to `[9/9]`, each ending in `would run`.

Start the trainings with the same command, without `--dry-run`:

**Windows**

```
python run_hardware_check.py --env build\PushBlock.exe
```

**macOS**

```
python run_hardware_check.py --env build/PushBlock.app
```

**Linux**

```
python run_hardware_check.py --env build/PushBlock.x86_64
```

A lot of text will scroll past. That's normal. Each run ends with a line like:

```
[3/9] hw_lr0.0001_bs256_buf2048_s42 finished
```

When all 9 are done, you'll see `All done.`

**If it stops**, for example because the laptop slept, run the same command again. Finished runs are skipped, and an unfinished one starts over.

A message like `finished (exit code 1, check it in the extraction)` is usually fine. It means the training finished but a last export step failed. Step 9 shows whether the run is really OK.

## Step 9: Make your CSV file

Replace `yourname` with your first name in lowercase, for example `anna`:

```
python extract_telemetry_v2.py --results results_hw --source yourname --out runs_yourname_v2.csv
```

The last line should be like:

```
Extracted 9 runs to runs_yourname_v2.csv; failed: 0; incomplete: 0; ...
```

`unseeded` and `not converged` can be any number.

If you see `torch not importable`, add `--no-gpu` at the end of the command and run it again.

If `failed` or `incomplete` is not 0, send Savvas a screenshot.

## Step 10: Upload your CSV to the group repository

You do this on the GitHub website, so you don't need any Git commands. Your name will appear on the change.

1. Open our group repository on GitHub: https://github.com/Group-14-AI-ML/ai-ml-project
2. Go to the folder `group14/data/raw/`.
3. Click **Add file → Upload files** and drag in `runs_yourname_v2.csv`.
4. Under **Commit changes**, write: `Add hardware runs from yourname`
5. Choose **Create a new branch for this commit and start a pull request**, then click **Propose changes**.
6. In the pull request description, write:
   - your laptop model, for example "Lenovo ThinkPad T14"
   - your operating system, for example "Windows 11" or "macOS 14"
   - that the laptop was plugged in and not in power saving mode
7. Click **Create pull request**. Savvas will check it and merge it.

If uploading doesn't work for you, send the CSV and the three details above to Savvas instead.

**Keep the `results_hw` folder. Don't delete it**, in case we need to extract the runs again.

## Problems and fixes

| What you see | What to do |
|---|---|
| `'python' is not recognized` or `command not found` | Activate the virtual environment (Step 3) |
| `mlagents-learn not found` | Activate the virtual environment. If it still fails, repeat Step 4 |
| `module 'numpy' has no attribute 'float'` | Run the last command of Step 4 again |
| `Descriptors cannot be created directly` | Run the last command of Step 4 again |
| `Build not found` | Check that the `--env` path matches your build (Step 5) |
| `Permission denied` on Linux | Run the `chmod` command from Step 5 |
| macOS says the app is damaged | Run the `xattr` command from Step 5 |
| Error about `onnx` or `onnxscript` at the end of a run | Usually fine. Continue to Step 9 |
| `Couldn't launch` or a port is already in use | Close Unity and any other training, then run the command again |
| A run seems frozen for more than 30 minutes | Press `Ctrl+C`, then run the same command again |
| Anything else | Screenshot the terminal and send it to Savvas |

## What happens next

After your pull request is merged, Savvas adds your CSV to the group dataset with `merge_and_audit.py`. You don't need to run this yourself.
