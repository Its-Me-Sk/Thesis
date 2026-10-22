# build_nb.py - assembles pinn_phaseA_kaggle.ipynb from the scripts in ./src (+ the unchanged pinn_physics.py and
# prepare.py of Experiment 2 in ./ref)
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

MD = r"""# Phase A — pilot diagnostics (no training) · Fokker–Planck-consistency thesis
Base model: the class-conditional MONAI DDPM of **U24_synthesis** (Linardos, Thakur, Gastounioti, Makris & Bakas,
*Evaluating the utility of AI-generated mammography images in breast cancer analysis*, Scientific Reports 2026), retrained
by us on CBIS-DDSM in Experiment 2. Nothing is trained here: every test uses the Experiment-2 checkpoint
`base2_w_ep040.pt` (100 % data, epoch 40) or the real images.

| test | what it decides | device |
|---|---|---|
| **A1** leakage / duplicates (HARD GATE) | patient or image overlap between train and test → FAIL | CPU |
| **A2** conditional-mean ("Jensen") test | is the old physics loss on x̂0 measuring blur? → sets **t\*** for the NLF arm | GPU 1 |
| **A3** noise vs texture + lesion size | is the "noise curve" noise or tissue texture at 256 px? | CPU |
| **A4** Fokker–Planck residual estimator | is the FP residual measurable, and is fp16 finite-difference precise enough? | GPU 0 |
| **A5** compute benchmark | cost of the FP term → number of seeds (≤2× → 5, ≤3× → 4, >3× → reconsider) | GPU 0 (after A4) |

Every rule is written in the script header **before** the run. The report only applies those rules; the LOCK is written
afterwards with the RA.

**How to run**
1. Settings → Accelerator **GPU T4 ×2** · Internet **ON**.
2. Add Input → *CBIS-DDSM Breast Cancer Image Dataset*.
3. Add Input → *Your Work* → **notebook64a7f79592** (the Experiment 2 notebook — it holds `base2_w_ep040.pt` and `base2_last.pt`).
4. Optional: Add Input → *Your Work* → **notebook00f3867662** (Experiment 3). Then A2 is also run on a 25 %-data model (+ ~1 h).
5. **Save Version → Save & Run All (Commit)**. Nothing to edit.

**Time:** probably 2–4 h (+ ~1 h if the 25 % model is attached). This is an estimate from Experiment 2's measured training speed
(1.67 it/s), not a measurement of these scripts; the logs print the real time of every step. Kaggle's 12 h limit is not at risk.

**Send back:** `phaseA/phaseA_report.txt` from the Output tab (or the whole `phaseA_results.zip`). If a cell fails, also send its log
from `phaseA/logs/`.
"""

CELL1 = r'''# ===================== CELL 1 — CONFIG (nothing needs to be changed) =====================
RUN = {"A1": True, "A2": True, "A3": True, "A4": True, "A5": True}
RUN_A2_25 = True          # A2 also on a 25 %-data model - only if the Experiment 3 or 4 output is attached (+ ~1 h)
import os, time
if not os.path.exists("/tmp/pg_session_start"):
    open("/tmp/pg_session_start", "w").write(str(time.time()))
def used_h():
    return (time.time() - float(open("/tmp/pg_session_start").read())) / 3600
print("Phase A | tests:", [k for k, v in RUN.items() if v], "| A2 on 25 % model:", RUN_A2_25)
'''

CELL2 = r'''# ===================== CELL 2 — setup (needs Internet ON) =====================
import os, re, sys, subprocess
os.chdir("/kaggle/working")
if not os.path.isdir("/kaggle/working/U24_synthesis/.git"):
    subprocess.run("git clone -q https://github.com/Its-Me-Sk/U24_synthesis.git", shell=True, check=True)
if not os.path.exists("/tmp/pip_done"):
    # Kaggle runs Python 3.13: install the repo requirements once, never force-reinstall numpy / pandas
    subprocess.run("cd /kaggle/working/U24_synthesis && grep -v xformers requirements.txt > /tmp/req.txt && "
                   "pip install -q -r /tmp/req.txt", shell=True, check=True)
    subprocess.run("pip install -q --no-deps torch-fidelity", shell=True, check=True)
    open("/tmp/pip_done", "w").write("ok")
probe = ("import sys, numpy, pandas, scipy, torch, torchmetrics, monai, generative, PIL; "
         "print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, "
         "'| torch', torch.__version__, '| torchmetrics', torchmetrics.__version__, '| monai', monai.__version__)")
env_chk = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
if env_chk.returncode != 0:
    print(env_chk.stderr[-2500:])
    raise SystemExit("!! environment check failed (see the error above) -> send Claude this block")
print("environment OK:", env_chk.stdout.strip())
p = "/kaggle/working/U24_synthesis/lightning_synthesis/model_architectures/monai_ddpm.py"
s = open(p).read()
s2 = re.sub(r"use_flash_attention\s*=\s*True", "use_flash_attention=False", s)   # xformers is not installed
if s2 != s:
    open(p, "w").write(s2)
print("flash attention ->", "patched off" if s2 != s else "already off")
m = "/kaggle/working/U24_synthesis/models.py"          # oracle loader (used later in A6): hard-coded config path
s = open(m).read()
s2 = re.sub(r"open\(\s*['\"]/home/[^'\"]*config\.yaml['\"]", "open('/kaggle/working/U24_synthesis/config.yaml'", s)
if s2 != s:
    open(m, "w").write(s2)
print("Cell 2 done. Kernel restart NOT needed (all GPU work runs in subprocesses).")
'''

CELL3 = r'''# ===================== CELL 3 — code + data cache + input checks + job runner =====================
import os, sys, glob, json, subprocess, threading, time
from pathlib import Path
WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
os.chdir(WD)
Path("phaseA_common.py").write_text(r''' + "'''" + rd("src/phaseA_common.py") + "'''" + r''')
Path("pinn_physics.py").write_text(r''' + "'''" + rd("ref/pinn_physics.py") + "'''" + r''')
Path("a1_leakage.py").write_text(r''' + "'''" + rd("src/a1_leakage.py") + "'''" + r''')
Path("a2_jensen.py").write_text(r''' + "'''" + rd("src/a2_jensen.py") + "'''" + r''')
Path("a3_texture.py").write_text(r''' + "'''" + rd("src/a3_texture.py") + "'''" + r''')
Path("a4_fp.py").write_text(r''' + "'''" + rd("src/a4_fp.py") + "'''" + r''')
Path("a5_bench.py").write_text(r''' + "'''" + rd("src/a5_bench.py") + "'''" + r''')
Path("report.py").write_text(r''' + "'''" + rd("src/report.py") + "'''" + r''')
print("wrote phaseA_common.py, pinn_physics.py, a1_leakage.py, a2_jensen.py, a3_texture.py, a4_fp.py, a5_bench.py, report.py")

# ---------- data cache (identical to Experiment 2: 100 % CBIS-DDSM train, 1300 benign + all malignant; official test) ----------
Path("/tmp/prepare.py").write_text(r''' + "'''" + rd("ref/prepare.py") + "'''" + r''')
subprocess.run([sys.executable, "/tmp/prepare.py"], check=True)

# ---------- inputs ----------
_IDX = {}
for dp, dns, fns in os.walk("/kaggle/input"):                    # one walk; the image folders are skipped
    dns[:] = [d for d in dns if d != "jpeg"]
    for fn in fns:
        _IDX.setdefault(fn, []).append(os.path.join(dp, fn))
def find_in(name):
    return sorted(_IDX.get(name, []))
MAIN = find_in("base2_w_ep040.pt")
LAST = find_in("base2_last.pt")
C25 = [f for s in range(42, 49) for f in find_in(f"base3_s{s}_w_ep045.pt")]
print("main checkpoint  base2_w_ep040.pt :", MAIN[0] if MAIN else
      "!! NOT FOUND -> Add Input -> Your Work -> notebook64a7f79592 (Experiment 2). A2, A4, A5 will FAIL without it.")
print("warm-start file  base2_last.pt    :", LAST[0] if LAST else "not found (only reported by A5)")
print("25 % checkpoint (optional)        :", C25[0] if C25 else "not attached -> A2-25 skipped")
print(subprocess.run("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv",
                     shell=True, capture_output=True, text=True).stdout)
r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True)
try:
    NGPU = int(r.stdout.split()[-1])
except Exception:
    NGPU = 0
    print("!! torch check failed:", r.stderr[-300:])
print("GPUs visible:", NGPU, "" if NGPU >= 2 else "!! choose GPU T4 x2 (with 1 GPU the GPU tests run one after the other)")
LOGS = "/kaggle/working/phaseA/logs"
os.makedirs(LOGS, exist_ok=True)
RC = {}

def run_jobs(jobs, poll_min=20):
    """jobs = [(name, gpu_or_None, [script, args...])] -> all run at the same time; output on screen + logs/<name>.log.
    CPU jobs: gpu = None. With fewer than 2 GPUs the GPU jobs run one after the other."""
    if NGPU < 2:
        gpu_jobs = [j for j in jobs if j[1] is not None]
        if len(gpu_jobs) > 1:
            first = [j for j in jobs if j[1] is None] + gpu_jobs[:1]
            run_jobs(first, poll_min)
            for j in gpu_jobs[1:]:
                run_jobs([j], poll_min)
            return
    procs = []
    for name, gpu, argv in jobs:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if gpu is None:
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["OMP_NUM_THREADS"] = "2"
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu if NGPU >= 2 else 0)
        fh = open(f"{LOGS}/{name}.log", "w")
        p = subprocess.Popen([sys.executable, "-u"] + argv, cwd=WD, env=env, text=True, bufsize=1,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((name, p, fh))
        print(f">>> started {name} on {'CPU' if gpu is None else 'GPU ' + env['CUDA_VISIBLE_DEVICES']} | session {used_h():.2f} h")
    def pump(name, p, fh):
        for line in p.stdout:
            fh.write(line); fh.flush()
            print(line if line.startswith("[") else f"[{name}] {line}", end="", flush=True)
    def poll():
        while any(p.poll() is None for _, p, _ in procs):
            time.sleep(poll_min * 60)
            if any(p.poll() is None for _, p, _ in procs):
                q = subprocess.run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader",
                                   shell=True, capture_output=True, text=True).stdout.strip().replace("\n", " | ")
                run = [n for n, p, _ in procs if p.poll() is None]
                print(f"[nvidia-smi] {q} | still running: {run} | session {used_h():.2f} h", flush=True)
    ths = [threading.Thread(target=pump, args=(n, p, fh)) for n, p, fh in procs]
    for th in ths: th.start()
    threading.Thread(target=poll, daemon=True).start()
    for th in ths: th.join()
    for name, p, fh in procs:
        RC[name] = p.wait(); fh.close()
        print(f"<<< {name} finished with return code {RC[name]}" + ("" if RC[name] == 0 else "  !! FAILED -> see " + f"{LOGS}/{name}.log"))
print("Cell 3 done.")
'''

CELL4 = r'''# ===================== CELL 4 — RUN Phase A (stage 1: A1 + A3 on CPU, A4 on GPU 0, A2 on GPU 1; stage 2: A5 + A2-25) =====================
# A failure of one test does not stop the others. Each test saves its own JSON in /kaggle/working/phaseA.
stage1 = []
if RUN["A1"]: stage1.append(("A1", None, ["a1_leakage.py"]))
if RUN["A3"]: stage1.append(("A3", None, ["a3_texture.py"]))
if RUN["A4"]: stage1.append(("A4", 0, ["a4_fp.py"]))
if RUN["A2"]: stage1.append(("A2-main", 1, ["a2_jensen.py", "--which", "main"]))
t0 = time.time()
run_jobs(stage1)
print(f"===== stage 1 done in {(time.time() - t0) / 3600:.2f} h =====")
stage2 = []
if RUN["A5"]: stage2.append(("A5", 0, ["a5_bench.py"]))          # after A4: uses the precision / step chosen by A4
if RUN["A2"] and RUN_A2_25 and C25: stage2.append(("A2-25", 1, ["a2_jensen.py", "--which", "25"]))
t0 = time.time()
run_jobs(stage2)
print(f"===== stage 2 done in {(time.time() - t0) / 3600:.2f} h | return codes {RC} | session {used_h():.2f} h =====")
'''

CELL5 = r'''# ===================== CELL 5 — REPORT + zip (send phaseA/phaseA_report.txt or phaseA_results.zip to Claude) =====================
import shutil
r = subprocess.run([sys.executable, "report.py"], cwd=WD, capture_output=True, text=True)
print(r.stdout[-30000:])
if r.returncode != 0:
    print("!! report.py failed:", r.stderr[-1500:])
with open("/kaggle/working/phaseA/return_codes.json", "w") as fh:
    json.dump(RC, fh, indent=1)
z = shutil.make_archive("/kaggle/working/phaseA_results", "zip", "/kaggle/working/phaseA")
print("return codes:", RC)
print("ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB) | report: /kaggle/working/phaseA/phaseA_report.txt | session {used_h():.2f} h")
'''

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.strip("\n").splitlines(keepends=True)},
                code(CELL1), code(CELL2), code(CELL3), code(CELL4), code(CELL5)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
OUTP = "/mnt/user-data/outputs/pinn_phaseA_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1)
print("wrote", OUTP)
