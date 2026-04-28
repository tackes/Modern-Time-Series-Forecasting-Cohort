# Workshop Environment Setup Guide

**Packt Live Workshop — Modern Time Series Forecasting with Python**

This guide walks you through every step to get your environment ready before the workshop. Follow the path that matches your situation. If something looks different than described, check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or reach out before the session.

**Start here at least 24 hours before the workshop.** Do not do this for the first time the morning of.

---

## Step 0 — Which Path Should I Follow?

| My situation | My path |
|---|---|
| I'm on Windows | [Path B: Windows Local (VS Code + Conda)](#path-b-windows-local-requires-anaconda-or-miniconda) ← recommended |
| I'm on Mac or Linux | [Path C: Mac / Linux Local (VS Code)](#path-c-mac--linux-local) ← recommended |
| I can't install software on my machine | [Path A: Google Colab](#path-a-google-colab-fallback-only) |

**We recommend VS Code local (Path B or C) over Google Colab.** Once your environment is set up, VS Code is faster, more reliable, and closer to how real forecasting work is done. You open a notebook, pick your kernel once, and everything just works — no re-installing packages every session.

**Use Colab (Path A) only if you cannot install software on your machine** — e.g., a locked-down work laptop. It works, but requires re-running a setup cell at the start of every session.

---

## Path A: Google Colab (Fallback Only)

Google Colab runs entirely in your browser. You do not need to install Python, Git, or anything else on your computer.

**What you need:** A Google account (Gmail works).

> **Important — Colab sessions are temporary.** When you close your browser or the session times out, Colab resets completely. All installed packages and cloned files are gone. You must run the setup cell (A.3) at the start of **every** new Colab session, including on workshop day. It takes about 5 minutes and is the same every time.

---

### A.1 — Open the Environment Check Notebook from GitHub

Notebooks in Colab are opened directly from GitHub — not by cloning first and then navigating.

1. Open your browser and go to **https://colab.research.google.com**
2. Sign in with your Google account if prompted.
3. A dialog box appears titled **"Open notebook"**. If it does not appear, click **File → Open notebook** in the top menu.
4. Click the **GitHub** tab at the top of the dialog.
5. In the search box, paste:
   ```
   tackes/Modern-Time-Series-Forecasting-Cohort
   ```
   and press **Enter**.
6. A list of notebooks appears. Click **`00_env_check.ipynb`**.
7. The notebook opens in a new tab.

> **On workshop day**, use this same method to open each session's notebook (e.g., `student_notebooks/02_framing_and_config.ipynb`). You will do this for each notebook during the session.

---

### A.2 — Understand the Setup Cell

The first code cell in every notebook is a setup cell. It clones the repository (which brings the data and precomputed artifacts) and installs all packages. It looks like this:

```python
!git clone https://github.com/tackes/Modern-Time-Series-Forecasting-Cohort.git packt-modern-time-series
%cd packt-modern-time-series
!pip install -q torch --index-url https://download.pytorch.org/whl/cpu
!pip install -q -r requirements.txt
```

**You must run this cell every time you open a notebook in a new Colab session.** The clone brings all the data and artifact files — you do not need to upload anything separately.

---

### A.3 — Run the Setup Cell

1. Click inside the first code cell of `00_env_check.ipynb`.
2. Press **Shift + Enter** to run it.
3. You will see a lot of text scroll by. This is normal and takes **3–5 minutes**.
4. When it finishes, you will see `Successfully installed ...` near the bottom. There should be no red `ERROR` lines.

> **If you see a red error line** mentioning a package name and "failed to install", copy the full error text and check TROUBLESHOOTING.md.

---

### A.4 — Run the Environment Check

1. Once the setup cell finishes, click **Runtime** in the top menu.
2. Click **Run all**.
3. Wait for all cells to finish (about 30–60 seconds).
4. Scroll to the bottom of the notebook. You should see something like:

```
✓ Python version OK
✓ All packages import OK
✓ Data file OK
✓ Artifacts OK
✓ Environment ready
```

**All five checks must show ✓ before the workshop.** If any show ✗, follow the fix instruction printed next to it, or check TROUBLESHOOTING.md.

---

## Path B: Windows Local (Requires Anaconda or Miniconda)

> **Important:** Windows requires the conda package manager. A plain `pip install` will not work and will cause crashes. If you do not have Anaconda or Miniconda, install one first (instructions in B.0).

---

### B.0 — Install Miniconda (skip if you already have Anaconda or Miniconda)

1. Go to **https://docs.anaconda.com/miniconda/** in your browser.
2. Under the Windows section, click the link for **Miniconda3 Windows 64-bit**.
3. The installer file (`.exe`) will download. When it finishes, double-click it.
4. Click **Next**, then **I Agree**, then **Next** on the Install Type screen (keep "Just Me" selected).
5. On the installation folder screen, leave the default path and click **Next**.
6. On the Advanced Options screen:
   - Check **"Add Miniconda3 to my PATH environment variable"** (you may see a warning — it is OK to proceed).
   - Check **"Register Miniconda3 as my default Python"**.
   - Click **Install**.
7. Click **Finish** when done.

---

### B.1 — Open a Terminal (Anaconda Prompt)

After installing Miniconda/Anaconda, you will use the **Anaconda Prompt** — not regular PowerShell or Command Prompt.

1. Click the **Windows Start button** (bottom-left of your screen).
2. Type `anaconda prompt` in the search bar.
3. Click **Anaconda Prompt** in the results.
4. A black window opens. You will see something like:
   ```
   (base) C:\Users\YourName>
   ```
   The `(base)` part means conda is active. Leave this window open.

> **All commands in this section are typed into this Anaconda Prompt window**, then pressed with **Enter** to run.

---

### B.2 — Install Git (skip if you already have it)

To check if Git is already installed, type this and press Enter:
```
git --version
```

- If you see `git version 2.X.X` — Git is installed. Skip to B.3.
- If you see `'git' is not recognized` — install Git:

1. Go to **https://git-scm.com/download/win** in your browser.
2. The download starts automatically. Run the installer.
3. Click **Next** on every screen, keeping all defaults. Click **Finish**.
4. Close and reopen your Anaconda Prompt (to reload the PATH), then re-type `git --version` to confirm.

---

### B.3 — Clone the Repository

In your Anaconda Prompt, navigate to where you want the workshop folder to live. For example, to put it on your Desktop:

```
cd %USERPROFILE%\Desktop
```

You will see the prompt change to show `Desktop`. Now clone the repo:

```
git clone https://github.com/tackes/Modern-Time-Series-Forecasting-Cohort.git packt-modern-time-series
```

Then move into the new folder:

```
cd packt-modern-time-series
```

Your prompt will now show `...\packt-modern-time-series`. This means you are inside the folder.

---

### B.4 — Create the Conda Environment

Run these commands **one at a time**, pressing **Enter** after each. When asked `Proceed ([y]/n)?`, type `y` and press Enter.

**Create the environment:**
```
conda create -n packt_timeseries_cohort python=3.12 -y
```
This takes 1–2 minutes.

**Activate it:**
```
conda activate packt_timeseries_cohort
```

Your prompt changes from `(base)` to `(packt_timeseries_cohort)`. This is important — you must see this prefix for the next steps to work correctly.

**Install the binary packages via conda:**
```
conda install -c conda-forge lightgbm scikit-learn scipy numpy pandas pyarrow -y
```
This takes 2–5 minutes.

```
conda install -c conda-forge pytorch pytorch-lightning -y
```
This takes 2–5 minutes.

---

### B.5 — Install the Python Packages

With the conda environment still active (you should see `(packt_timeseries_cohort)` in your prompt):

```
pip install -r requirements.txt
```

This takes 2–4 minutes. You will see a lot of text. Wait for it to finish and confirm there are no red `ERROR` lines.

---

### B.6 — Set the OpenMP Variable (One Time Only)

Run this command exactly as written:

```
setx KMP_DUPLICATE_LIB_OK TRUE
```

You will see `SUCCESS: Specified value was saved.`

**Close your Anaconda Prompt and reopen it** before continuing. This variable takes effect only after a restart.

After reopening, re-activate your environment:
```
conda activate packt_timeseries_cohort
```

---

### B.7 — Open the Environment Check Notebook in VS Code

1. Open **VS Code** (if not installed, download it from https://code.visualstudio.com).
2. Click **File → Open Folder** and select your `packt-modern-time-series` folder.
3. In the left sidebar (Explorer), click `00_env_check.ipynb`.
4. The notebook opens. In the top-right corner of the notebook, you will see a kernel selector — it will say something like **"Select Kernel"** or show a Python version.
5. Click it. A dropdown appears at the top of the screen.
6. Look for an option that includes **`packt_timeseries_cohort`** in the path (something like `anaconda3\envs\packt_timeseries_cohort\python.exe`). Click it.

> **If you don't see `packt_timeseries_cohort` in the list:**
> - Click **"Select Another Kernel..."**
> - Click **"Python Environments..."**
> - It should appear there. If not, close VS Code, reopen your Anaconda Prompt, run `conda activate packt_timeseries_cohort`, then type `code .` to launch VS Code from within the environment.

7. With the correct kernel selected, click the **"Run All"** button (the ▶▶ icon at the top of the notebook, or go to **Run → Run All Cells**).
8. Scroll to the bottom. All five checks must show ✓.

---

## Path C: Mac / Linux Local

---

### C.1 — Open a Terminal

**On Mac:**
- Press **Command + Space** to open Spotlight Search.
- Type `terminal` and press **Enter**.
- A window opens with a prompt like `YourName@MacBook ~ %`.

**On Linux:**
- Press **Ctrl + Alt + T** (works on most distributions).
- Or search your applications for "Terminal".

Leave this window open. All commands in this section are typed here.

---

### C.2 — Check Prerequisites

Check that Git is installed:
```bash
git --version
```
You should see `git version 2.X.X`. If not, install Git:
- Mac: Run `xcode-select --install` and follow the prompts.
- Linux (Ubuntu/Debian): Run `sudo apt install git -y`.

Check that Python 3.12 is available:
```bash
python3 --version
```
You should see `Python 3.12.X`. If not, download it from **https://www.python.org/downloads/**.

---

### C.3 — Clone the Repository

Navigate to where you want the workshop folder:
```bash
cd ~/Desktop
```

Clone the repo:
```bash
git clone https://github.com/tackes/Modern-Time-Series-Forecasting-Cohort.git packt-modern-time-series
cd packt-modern-time-series
```

---

### C.4 — Create a Virtual Environment

```bash
python3 -m venv packt_timeseries_cohort
source packt_timeseries_cohort/bin/activate
```

Your prompt will now show `(packt_timeseries_cohort)` at the start. Keep this active for all remaining steps.

---

### C.5 — Install the Packages

Install PyTorch first (CPU-only version):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install everything else:
```bash
pip install -r requirements.txt
```

Each command takes 2–5 minutes. Wait for each to finish before running the next.

---

### C.6 — Run the Environment Check Notebook

From your terminal (with the virtual environment active):
```bash
jupyter notebook 00_env_check.ipynb
```

Your browser will open automatically to the notebook.

- Click **Cell → Run All** in the menu bar.
- Wait for all cells to finish.
- Scroll to the bottom. All five checks must show ✓.

> **If your browser doesn't open automatically**, look at the terminal output for a line like `http://localhost:8888/...` and paste that URL into your browser manually.

---

## Final Verification — All Paths

Before the workshop session begins, confirm the following:

- [ ] `00_env_check.ipynb` shows **5 out of 5 checks passing** (all ✓)
- [ ] You can open `student_notebooks/01_welcome.ipynb` and the kernel connects without error

If any of these are not true, see TROUBLESHOOTING.md or contact your instructor before the session.

---

## Quick Reference: How to Open a Terminal

| Platform | How to open |
|---|---|
| Windows (conda) | Start menu → search "Anaconda Prompt" → click it |
| Mac | Command + Space → type "terminal" → Enter |
| Linux | Ctrl + Alt + T |
| Google Colab | Use the `!` prefix in a code cell to run shell commands |

## Quick Reference: How to Activate Your Environment

| Platform | Command |
|---|---|
| Windows (conda) | `conda activate packt_timeseries_cohort` |
| Mac / Linux | `source packt_timeseries_cohort/bin/activate` |
| Google Colab | Not needed — environment is per-session |

**Every time you open a new terminal before the workshop, re-activate your environment first.**
