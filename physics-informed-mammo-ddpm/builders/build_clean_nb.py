# build_clean_nb.py - assembles pinn_clean_baseline_kaggle.ipynb (clean patient-level data + baseline retraining)
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

# Phase A setup cell is reused unchanged (Python-3.13-safe install, flash attention off)
pa = json.load(open("/mnt/user-data/outputs/pinn_phaseA_kaggle.ipynb"))
SETUP = "".join(pa["cells"][2]["source"])
assert "CELL 2" in SETUP

MD = r"""# Clean data + baseline retraining (step 2 of the timeline)
Phase A (A1) found patient leakage and duplicates; the data audit measured the clean split. This notebook
1. builds the **clean patient-level CBIS-DDSM cache** (rule B, decided 19 Oct 2026): one record per unique image,
   33 label-conflict images removed, the 282 skipped test images recovered, the 31 patients that were in both official
   splits moved to test → expected **train 2336 / test 663** images, no patient in both;
2. retrains the baseline **from scratch with the exact Experiment-2 recipe** (`train_arm2.py`, λ = 0, 40 epochs),
   **two seeds at the same time** (seed 42 on GPU 0, seed 43 on GPU 1 — same session time as one);
3. does a quick sample check (100 images/class, DDIM 50, guidance 4) of each finished baseline.

Base model: U24_synthesis (Linardos, Thakur, Gastounioti, Makris & Bakas, *Scientific Reports* 2026) — not our work.

**How to run**
1. Settings → Accelerator **GPU T4 ×2** · Internet **ON**.
2. Add Input → *CBIS-DDSM Breast Cancer Image Dataset*. Optional: *oracle-checkpoint-final* (oracle accuracy in the sample check).
3. **Save Version → Save & Run All (Commit)**. Nothing to edit (unless your weekly GPU quota is below 11.8 h: then set `BUDGET_H` in Cell 1).

**Time:** clean cache ~12 min + training ~7.5 h (11.1 min/epoch, measured in Phase A for this recipe) + check ~15 min ≈ **8 h**.
If the session stops early, add this notebook's latest output as input (Add Input → Your Work) and Save & Run All again:
training resumes exactly where it stopped (optimizer state included), also from a friend's account.

**Send back:** the printed block of the last cell, `clean_data/clean_summary.json` and the two `eval/grid_*.png` images.
"""

CELL1 = r'''# ===================== CELL 1 — CONFIG =====================
SEEDS = [42, 43]          # two independent baselines (GPU 0, GPU 1) - same session time as one
EPOCHS = 40               # = Experiment 2 (base2_w_ep040.pt)
BUDGET_H = 11.0           # hours this session may use = min(11.0, your remaining weekly GPU quota - 0.8)
EVAL_GS = 4.0             # guidance scale for the quick sample check (taken from the base paper, not tuned)
EVAL_N = 100              # samples per class in the quick check
import os, time
if not os.path.exists("/tmp/pg_session_start"):
    open("/tmp/pg_session_start", "w").write(str(time.time()))
def used_h():
    return (time.time() - float(open("/tmp/pg_session_start").read())) / 3600
ARMS = [f"base_clean_s{s}" for s in SEEDS]
print("arms:", ARMS, "| epochs", EPOCHS, "| budget", BUDGET_H, "h")
'''

CELL3 = r'''# ===================== CELL 3 — code + CLEAN data cache + carry-forward + job runner =====================
import os, sys, glob, json, shutil, subprocess, threading, time
from pathlib import Path
WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
os.chdir(WD)
Path("pinn_physics.py").write_text(r''' + "'''" + rd("ref/pinn_physics.py") + "'''" + r''')
Path("train_arm2.py").write_text(r''' + "'''" + rd("ref/train_arm2.py") + "'''" + r''')
Path("eval_arm.py").write_text(r''' + "'''" + rd("ref/eval_arm.py") + "'''" + r''')
Path("/tmp/prepare_clean.py").write_text(r''' + "'''" + rd("src/prepare_clean.py") + "'''" + r''')
print("wrote pinn_physics.py, train_arm2.py (Experiment-2 recipe, unchanged), eval_arm.py, prepare_clean.py")
subprocess.run([sys.executable, "/tmp/prepare_clean.py"], check=True)

# ---------- carry forward earlier sessions' checkpoints of THESE arms (so every output version is complete) ----------
CK = "/kaggle/working/ckpt"
os.makedirs(CK, exist_ok=True)
copied = []
for arm in ARMS:
    for f in glob.glob(f"/kaggle/input/**/ckpt/{arm}_*", recursive=True):
        dst = os.path.join(CK, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy(f, dst); copied.append(os.path.basename(f))
print(f"carried forward {len(copied)} files from earlier sessions" + (": " + ", ".join(sorted(copied)) if copied else ""))
print(subprocess.run("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv",
                     shell=True, capture_output=True, text=True).stdout)
r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True)
try:
    NGPU = int(r.stdout.split()[-1])
except Exception:
    NGPU = 0; print("!! torch check failed:", r.stderr[-300:])
print("GPUs visible:", NGPU, "" if NGPU >= 2 else "!! choose GPU T4 x2 (with 1 GPU the seeds run one after the other)")
LOGS = "/kaggle/working/logs"
os.makedirs(LOGS, exist_ok=True)

def run_jobs(jobs, poll_min=30):
    """jobs = [(name, gpu, [script, args...])] -> run together (one per GPU); returns {name: (return_code, output)}"""
    if NGPU < 2 and len(jobs) > 1:
        res = {}
        for j in jobs:
            res.update(run_jobs([j], poll_min))
        return res
    procs = []
    for name, gpu, argv in jobs:
        env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu if NGPU >= 2 else 0)
        fh = open(f"{LOGS}/{name}.log", "a")
        p = subprocess.Popen([sys.executable, "-u"] + argv, cwd=WD, env=env, text=True, bufsize=1,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((name, p, fh, []))
        print(f">>> started {name} on GPU {env['CUDA_VISIBLE_DEVICES']} | session {used_h():.2f} h", flush=True)
    def pump(name, p, fh, buf):
        for line in p.stdout:
            fh.write(line); fh.flush(); buf.append(line)
            print(line if line.startswith("[") else f"[{name}] {line}", end="", flush=True)
    def poll():
        while any(p.poll() is None for _, p, _, _ in procs):
            time.sleep(poll_min * 60)
            if any(p.poll() is None for _, p, _, _ in procs):
                q = subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader",
                                   shell=True, capture_output=True, text=True).stdout.strip().replace("\n", " | ")
                print(f"[nvidia-smi] {q} | session {used_h():.2f} h", flush=True)
    ths = [threading.Thread(target=pump, args=(n, p, fh, b)) for n, p, fh, b in procs]
    for th in ths: th.start()
    threading.Thread(target=poll, daemon=True).start()
    for th in ths: th.join()
    out = {}
    for name, p, fh, buf in procs:
        rc = p.wait(); fh.close(); out[name] = (rc, "".join(buf))
        print(f"<<< {name} finished with return code {rc}" + ("" if rc == 0 else f"  !! FAILED -> see {LOGS}/{name}.log"))
    return out
print("Cell 3 done.")
'''

CELL4 = r'''# ===================== CELL 4 — SMOKE TEST (~5 min, skipped when training already started in an earlier session) =====================
# train 20 batches -> resume to epoch 2 on both GPUs; checks resume, optimizer restore, file writing, data size, seeds.
import re
started = any(glob.glob(f"{CK}/{arm}_last.pt") for arm in ARMS)
if started:
    SMOKE_OK = True
    print("training already started in an earlier session -> smoke test skipped")
else:
    SM = "/kaggle/working/ckpt_smoke"
    shutil.rmtree(SM, ignore_errors=True)
    common = ["--smoke_batches", "20", "--ckpt_dir", SM, "--snap_every", "1", "--log_every", "10", "--lam", "0.0"]
    jobs = lambda ep: [(f"smoke_ep{ep}_{arm}", g, ["train_arm2.py", "--arm", arm, "--seed", str(s), "--epochs", str(ep)] + common)
                       for g, (arm, s) in enumerate(zip(ARMS, SEEDS))]
    r1 = run_jobs(jobs(1), poll_min=5)
    r2 = run_jobs(jobs(2), poll_min=5)
    ntr = int(json.load(open("/kaggle/working/clean_data/clean_summary.json"))["train"]["images"])
    S = r"START epoch=(\d+) step=(\d+) opt_states=(\d+) lam=\S+ checksum=(\S+)"
    def g(pat, s):
        m = re.search(pat, s); return m.groups() if m else None
    checks = []
    for arm in ARMS:
        o1, o2 = r1[f"smoke_ep1_{arm}"][1], r2[f"smoke_ep2_{arm}"][1]
        checks += [(f"{arm}: run 1 from scratch (epoch 0, step 0)", lambda o1=o1: g(S, o1)[:2] == ("0", "0")),
                   (f"{arm}: run 2 RESUMED at epoch 1, step 20", lambda o2=o2: g(S, o2)[:2] == ("1", "20")),
                   (f"{arm}: optimizer state restored", lambda o2=o2: int(g(S, o2)[2]) > 0),
                   (f"{arm}: trains on the CLEAN cache ({ntr} images)", lambda o1=o1: f"data {ntr} images" in o1),
                   (f"{arm}: files written atomically", lambda arm=arm: all(os.path.exists(f"{SM}/{arm}_{x}") for x in
                       ["last.pt", "w_latest.pt", "w_ep001.pt", "w_ep002.pt", "history.json"]) and not glob.glob(f"{SM}/*.tmp"))]
    if len(ARMS) == 2:
        checks.append(("the two seeds start from DIFFERENT weights",
                       lambda: g(S, r1[f"smoke_ep1_{ARMS[0]}"][1])[3] != g(S, r1[f"smoke_ep1_{ARMS[1]}"][1])[3]))
    def ok(f):
        try: return bool(f())
        except Exception: return False
    print("\n" + "=" * 70)
    for name, f in checks:
        print(("PASS  " if ok(f) else "FAIL  ") + name)
    SMOKE_OK = all(ok(f) for _, f in checks)
    if SMOKE_OK:
        shutil.rmtree(SM, ignore_errors=True)
        print("ALL PASS -> smoke files deleted, Cell 5 starts the real training.")
    else:
        print("SMOKE TEST FAILED -> Cell 5 will NOT train. Send Claude this block + logs/smoke_*.log")
'''

CELL5 = r'''# ===================== CELL 5 — TRAINING: baseline seed 42 on GPU 0 + seed 43 on GPU 1 (resumable) =====================
if not SMOKE_OK:
    print("smoke test did not pass -> not training")
else:
    jobs = []
    for g, (arm, s) in enumerate(zip(ARMS, SEEDS)):
        hist = f"{CK}/{arm}_history.json"
        done = os.path.exists(hist) and len(json.load(open(hist))) >= EPOCHS
        if done:
            print(f"{arm}: already finished {EPOCHS} epochs")
            continue
        jobs.append((arm, g, ["train_arm2.py", "--arm", arm, "--lam", "0.0", "--seed", str(s), "--epochs", str(EPOCHS),
                              "--budget_h", str(BUDGET_H), "--snap_every", "5", "--ckpt_dir", CK]))
    if jobs:
        TR = run_jobs(jobs, poll_min=30)
    for arm in ARMS:
        hist = f"{CK}/{arm}_history.json"
        n = len(json.load(open(hist))) if os.path.exists(hist) else 0
        print(f"{arm}: {n}/{EPOCHS} epochs" + (" -> FINISHED" if n >= EPOCHS else " -> NOT finished: next session, attach "
              "this notebook's latest output (Add Input -> Your Work) and Save & Run All again"))
        if n >= EPOCHS:
            json.dump({"arm": arm, "epochs": n}, open(f"{CK}/{arm}_done.json", "w"))
    print(subprocess.run(f"ls -lh {CK}", shell=True, capture_output=True, text=True).stdout)
'''

CELL6 = r'''# ===================== CELL 6 — QUICK SAMPLE CHECK + SUMMARY (send this block to Claude) =====================
EV = "/kaggle/working/eval"
todo = [(arm, g) for g, arm in enumerate(ARMS)
        if os.path.exists(f"{CK}/{arm}_w_ep{EPOCHS:03d}.pt") and not os.path.exists(f"{EV}/results_{arm}_ep{EPOCHS:03d}.json")]
if todo and used_h() < 11.2:
    run_jobs([(f"eval_{arm}", g, ["eval_arm.py", "--arm", arm, "--epoch", str(EPOCHS), "--n_per_class", str(EVAL_N),
                                  "--gs", str(EVAL_GS), "--ckpt_dir", CK, "--out", EV]) for arm, g in todo], poll_min=10)
elif todo:
    print(f"{used_h():.1f} h used -> sample check skipped this session (it runs automatically next session)")
cs = json.load(open("/kaggle/working/clean_data/clean_summary.json"))
print("\n" + "=" * 90)
print("CLEAN DATA:", {k: cs[k] for k in ("train", "test")})
print("rejected by full-mammogram check:", cs["rejected_files"], "| label-conflict images excluded:",
      cs["label_conflict_images_excluded"], "| row resolution:", cs["row_resolution"])
summ = {"clean_data": {k: cs[k] for k in ("train", "test")}, "arms": {}}
for arm in ARMS:
    hist = f"{CK}/{arm}_history.json"
    h = json.load(open(hist)) if os.path.exists(hist) else []
    r = f"{EV}/results_{arm}_ep{EPOCHS:03d}.json"
    res = json.load(open(r)) if os.path.exists(r) else {}
    row = {"epochs": len(h)}
    if h:
        row.update({k: round(h[-1][k], 5) for k in ("loss", "mse", "ms", "hf", "phys")})
        row["min_per_epoch"] = round(sum(x["minutes"] for x in h[-5:]) / len(h[-5:]), 2)
        row["mse_epochs_1_10_20_30_40"] = [round(h[i - 1]["mse"], 5) for i in (1, 10, 20, 30, 40) if i <= len(h)]
    for k in ("KID_mean", "KID_std", "FID", "oracle_acc", "oracle_real_acc", "physics_residual_generated",
              "physics_residual_real_test"):
        if k in res:
            row[k] = res[k]
    summ["arms"][arm] = row
    print(f"{arm}: {row}")
json.dump(summ, open("/kaggle/working/baseline_clean_summary.json", "w"), indent=1)
print("grids:", sorted(glob.glob(f"{EV}/grid_*.png")))
print(f"session time used: {used_h():.2f} h")
'''

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.strip("\n").splitlines(keepends=True)},
                code(CELL1), code(SETUP), code(CELL3), code(CELL4), code(CELL5), code(CELL6)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
OUTP = "/mnt/user-data/outputs/pinn_clean_baseline_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1)
print("wrote", OUTP)
