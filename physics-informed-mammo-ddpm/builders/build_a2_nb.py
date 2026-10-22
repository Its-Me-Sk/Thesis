# build_a2_nb.py - assembles pinn_S2_baseline_audit_kaggle.ipynb (S2: baseline audit on the clean seed-42 baseline)
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

pa = json.load(open("/mnt/user-data/outputs/pinn_phaseA_kaggle.ipynb"))
SETUP = "".join(pa["cells"][2]["source"])
assert "CELL 2" in SETUP

MD = r"""# S2 — baseline audit of the clean baseline (seed 42) before the LOCK. No diffusion training. Test set NOT used.
Validation = 15 % of the clean TRAIN patients (fixed seed), used only for classifiers, gates and the guidance choice.

| test | what | rule (pre-stated in the script) |
|---|---|---|
| A10 | real data by class: view, laterality, density, digitiser (CPU) | reported with patient-bootstrap CI |
| A6 | pathology classifier (ConvNeXt-Tiny, fixed recipe, 3 seeds) | judge usable if mean **validation** AUC ≥ 0.65 |
| A6b | view classifier CC vs MLO | usable if validation AUC ≥ 0.95 |
| A7 | samples at guidance 4, 1 (200/class), 2, 3 (100/class); guidance 0 same-noise **identity test** | guidance 0 must give identical benign/malignant images |
| A9 | conditioning gate, view share, brightness/saturation, metric sanity, calibration KID vs train, memorisation, guidance rule | gate: AUC_gen − 0.5 ≥ 0.5·(AUC_val − 0.5) |
| A8 | Fokker–Planck residual, step-table spot check, S_ref(t), FP-vs-base gradients → λ candidates | spot check ≤ 0.10 |

**How to run**
1. Settings → Accelerator **GPU T4 ×2** · Internet **ON**.
2. Add Input → *Your Work* → **clean-data-baseline-retraining** (latest version) and Add Input → *CBIS-DDSM Breast Cancer Image Dataset*.
3. **Save Version → Save & Run All (Commit)**. Then **stop the interactive session** (it also uses GPU quota).

**Time:** expected ≈ 3 h; hard limit `BUDGET_H = 6.0` h in Cell 1 (every job is skipped if it would pass the limit).
**Send back:** `S2/S2_report.txt` + the `S2/grid_*.png` and `S2/memorisation_top8_*.png` images.
"""

CELL1 = r'''# ===================== CELL 1 — CONFIG =====================
BUDGET_H = 6.0            # hard limit for this session (your quota showed 7 h left -> 7 - 0.8 margin, rounded down)
import os, time
os.environ["PA_OUT"] = "/kaggle/working/S2"
os.makedirs(os.environ["PA_OUT"], exist_ok=True)
if not os.path.exists("/tmp/pg_session_start"):
    open("/tmp/pg_session_start", "w").write(str(time.time()))
def used_h():
    return (time.time() - float(open("/tmp/pg_session_start").read())) / 3600
print("S2 baseline audit | budget", BUDGET_H, "h | output", os.environ["PA_OUT"])
'''

CELL3 = r'''# ===================== CELL 3 — code + clean data (reused) + input checks + job queues =====================
import os, sys, glob, json, shutil, subprocess, threading, time
from pathlib import Path
WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
os.chdir(WD)
FILES = {"phaseA_common.py": r''' + "'''" + rd("src2/phaseA_common.py") + "'''" + r''',
         "fp_core.py": r''' + "'''" + rd("src2/fp_core.py") + "'''" + r''',
         "a6_classifier.py": r''' + "'''" + rd("src2/a6_classifier.py") + "'''" + r''',
         "a6b_view.py": r''' + "'''" + rd("src2/a6b_view.py") + "'''" + r''',
         "a7_generate.py": r''' + "'''" + rd("src2/a7_generate.py") + "'''" + r''',
         "a8_fp_clean.py": r''' + "'''" + rd("src2/a8_fp_clean.py") + "'''" + r''',
         "a9_metrics.py": r''' + "'''" + rd("src2/a9_metrics.py") + "'''" + r''',
         "a10_metadata.py": r''' + "'''" + rd("src2/a10_metadata.py") + "'''" + r''',
         "report2.py": r''' + "'''" + rd("src2/report2.py") + "'''" + r'''}
for n, s in FILES.items():
    Path(n).write_text(s)
Path("/tmp/prepare_clean.py").write_text(r''' + "'''" + rd("src/prepare_clean.py") + "'''" + r''')
print("wrote", ", ".join(FILES))
subprocess.run([sys.executable, "/tmp/prepare_clean.py"], check=True)      # reuses clean_data/ from the input (identical bytes)
ck = sorted(glob.glob("/kaggle/input/**/base_clean_s42_w_ep040.pt", recursive=True))
print("clean baseline seed 42:", ck[0] if ck else "!! NOT FOUND -> Add Input -> Your Work -> clean-data-baseline-retraining")
cb = subprocess.run("find /kaggle/input -maxdepth 6 -name dicom_info.csv", shell=True, capture_output=True, text=True).stdout.strip()
print("CBIS-DDSM csv:", cb or "!! NOT FOUND -> Add Input -> CBIS-DDSM (needed by A10 metadata and A6b view labels)")
r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True)
NGPU = int(r.stdout.split()[-1]) if r.returncode == 0 else 0
print("GPUs visible:", NGPU)
LOGS = "/kaggle/working/S2/logs"; os.makedirs(LOGS, exist_ok=True)
RC = {}

def run_one(name, gpu, argv, est_h):
    """one job; skipped (rc = 'skipped') if it would pass BUDGET_H (0.6 h kept for A9 + report)"""
    if used_h() + est_h > BUDGET_H - 0.6 and name != "A9":
        print(f"--- {name} SKIPPED: {used_h():.2f} h used + {est_h} h estimate would pass the budget", flush=True)
        RC[name] = "skipped"; return
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu if NGPU >= 2 else 0)
    if gpu is None:
        env["OMP_NUM_THREADS"] = "2"
    print(f">>> {name} started on {'CPU' if gpu is None else 'GPU ' + env['CUDA_VISIBLE_DEVICES']} | session {used_h():.2f} h", flush=True)
    with open(f"{LOGS}/{name}.log", "w") as fh:
        p = subprocess.Popen([sys.executable, "-u"] + argv, cwd=WD, env=env, text=True, bufsize=1,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            fh.write(line); fh.flush()
            print(line if line.startswith("[") else f"[{name}] {line}", end="", flush=True)
        RC[name] = p.wait()
    print(f"<<< {name} return code {RC[name]} | session {used_h():.2f} h" + ("" if RC[name] == 0 else f"  !! see {LOGS}/{name}.log"), flush=True)

def run_queues(queues):
    """queues = {gpu: [(name, [script, args], est_h), ...]} -> each GPU runs its list in order, GPUs in parallel"""
    if NGPU < 2:
        queues = {0: [j for q in queues.values() for j in q]}
    ths = [threading.Thread(target=lambda g=g, q=q: [run_one(n, g, a, e) for n, a, e in q]) for g, q in queues.items()]
    for t in ths: t.start()
    for t in ths: t.join()
print("Cell 3 done.")
'''

CELL4 = r'''# ===================== CELL 4 — RUN (A10 on CPU; then GPU 0 and GPU 1 queues in parallel; then A9) =====================
t0 = time.time()
run_one("A10", None, ["a10_metadata.py"], 0.1)
run_queues({
    0: [("A7-gen-a", ["a7_generate.py", "--seed_model", "42", "--spec", "4:200,1:200,0:50s", "--max_min", "130"], 1.7),
        ("A7-gen-b", ["a7_generate.py", "--seed_model", "42", "--spec", "2:100", "--tag", "_b", "--max_min", "45"], 0.5)],
    1: [("A6", ["a6_classifier.py"], 0.5),
        ("A6b", ["a6b_view.py"], 0.25),
        ("A8", ["a8_fp_clean.py", "--seed_model", "42"], 0.7),
        ("A7-gen-c", ["a7_generate.py", "--seed_model", "42", "--spec", "3:100", "--tag", "_c", "--max_min", "45"], 0.5)]})
print(f"===== queues done in {(time.time() - t0) / 3600:.2f} h | session {used_h():.2f} h =====")
if used_h() + 0.5 <= BUDGET_H:
    run_one("A9", 0, ["a9_metrics.py"], 0.5)
else:
    print("!! A9 skipped: not enough budget left"); RC["A9"] = "skipped"
print(f"===== all done | return codes {RC} | session {used_h():.2f} h =====")
'''

CELL5 = r'''# ===================== CELL 5 — REPORT + zip =====================
r = subprocess.run([sys.executable, "report2.py"], cwd=WD, capture_output=True, text=True)
print(r.stdout[-40000:])
if r.returncode != 0:
    print("!! report2.py failed:", r.stderr[-1500:])
json.dump(RC, open("/kaggle/working/S2/return_codes.json", "w"), indent=1)
z = shutil.make_archive("/kaggle/working/S2_results", "zip", "/kaggle/working/S2")
print("return codes:", RC, "| ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB) | session {used_h():.2f} h")
'''

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.strip("\n").splitlines(keepends=True)},
                code(CELL1), code(SETUP), code(CELL3), code(CELL4), code(CELL5)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
OUTP = "/mnt/user-data/outputs/pinn_S2_baseline_audit_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1)
print("wrote", OUTP)
