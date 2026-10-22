# build_s11_nb.py - assembles pinn_S11_dose_augmentation_kaggle.ipynb (S11: physics dose augmentation for the classifier).
# No diffusion model, no repo clone, no pip installs: plain Kaggle GPU image.
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

MD = r"""# S11 — physics-based dose augmentation for the benign/malignant classifier (no generative model)
Rules: `S11_preregistration.txt` (fixed before any S11 code ran). ConvNeXt-Tiny, fixed S2 recipe, 5 seeds × 3 arms ×
2 training datasets (CMMD, CBIS-DDSM):
**R0** standard augmentation · **R1** + generic white noise (matched power) · **R2** + physics dose-reduction noise.
Primary endpoints P1–P3 on CMMD (clean and simulated 25 %-dose test images), secondary cross-dataset tests.

**Input:** Add Input → the **S8 replication** notebook output (latest version; it contains `cmmd_data/` and `clean_data/`).
GPU T4 ×2 · Internet ON (ImageNet weights) · Save & Run All (Commit). **≈ 1.5 h.** Send back `S11_results.zip`.
"""

CELL1 = r'''# ===================== CELL 1 — config + code + inputs =====================
import os, sys, glob, json, shutil, subprocess, threading, time
from pathlib import Path
OUT = "/kaggle/working/S11"; WD = "/kaggle/working/s11_code"
os.makedirs(OUT, exist_ok=True); os.makedirs(WD, exist_ok=True)
FILES = {"phaseA_common.py": r''' + "'''" + rd("src8/phaseA_common.py") + "'''" + r''',
         "pinn_physics.py": r''' + "'''" + rd("ref/pinn_physics.py") + "'''" + r''',
         "a6_classifier.py": r''' + "'''" + rd("src2/a6_classifier.py") + "'''" + r''',
         "s11_clf.py": r''' + "'''" + rd("src11/s11_clf.py") + "'''" + r''',
         "s11_stats.py": r''' + "'''" + rd("src11/s11_stats.py") + "'''" + r'''}
for n, s in FILES.items():
    Path(f"{WD}/{n}").write_text(s)
def walk_find(name, root="/kaggle/input"):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in ("CMMD", "jpeg", "U24_synthesis", ".git", "ckpt")]
        if name in fns:
            out.append(os.path.join(dp, name))
    return sorted(out)
def need_dir(summary, parent):
    h = [os.path.dirname(p) for p in walk_find(summary) if os.path.basename(os.path.dirname(p)) == parent]
    if not h:
        raise SystemExit(f"!! {parent}/{summary} not found -> Add Input -> the S8 replication notebook output")
    print(parent, "->", h[0]); return h[0]
DATA_CMMD = need_dir("cmmd_summary.json", "cmmd_data"); DATA_CBIS = need_dir("clean_summary.json", "clean_data")
r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True)
NGPU = int(r.stdout.split()[-1]) if r.returncode == 0 else 0
print("GPUs visible:", NGPU)
LOGS = f"{OUT}/logs"; os.makedirs(LOGS, exist_ok=True); RC = {}
def run_one(name, gpu, argv):
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"; env["CUDA_VISIBLE_DEVICES"] = str(gpu if NGPU >= 2 else 0)
    env["PA_OUT"] = f"{OUT}/tmp"
    print(f">>> {name} started on GPU {env['CUDA_VISIBLE_DEVICES']}", flush=True)
    with open(f"{LOGS}/{name}.log", "w") as fh:
        p = subprocess.Popen([sys.executable, "-u"] + argv, cwd=WD, env=env, text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            fh.write(line); fh.flush(); print(line if line.startswith("[") else f"[{name}] {line}", end="", flush=True)
        RC[name] = p.wait()
    print(f"<<< {name} return code {RC[name]}" + ("" if RC[name] == 0 else f"  !! see {LOGS}/{name}.log"), flush=True)
COMMON = ["--data_cmmd", DATA_CMMD, "--data_cbis", DATA_CBIS]
print("Cell 1 done.")
'''

CELL2 = r'''# ===================== CELL 2 — smoke test (~5 min) =====================
SM = "/kaggle/working/s11_smoke"
run_one("smoke", 0, ["s11_clf.py", "--train_ds", "cmmd", "--arms", "R0,R1,R2", "--seeds", "0", "--smoke", "--out", SM] + COMMON)
if RC.get("smoke") != 0 or len(glob.glob(f"{SM}/pred_*.npz")) != 3:
    raise SystemExit("SMOKE TEST FAILED -> send Claude this block + S11/logs/smoke.log")
shutil.rmtree(SM, ignore_errors=True); print("SMOKE TEST PASSED")
'''

CELL3 = r'''# ===================== CELL 3 — training: CMMD-trained on GPU 0, CBIS-trained on GPU 1 (~1-1.5 h) =====================
ths = [threading.Thread(target=run_one, args=(f"train_{ds}", g, ["s11_clf.py", "--train_ds", ds, "--out", OUT] + COMMON))
       for g, ds in enumerate(("cmmd", "cbis"))]
for t in ths: t.start()
for t in ths: t.join()
'''

CELL4 = r'''# ===================== CELL 4 — endpoints + zip =====================
r = subprocess.run([sys.executable, "s11_stats.py", OUT], cwd=WD, capture_output=True, text=True)
print(r.stdout[-20000:]); print(r.stderr[-3000:] if r.returncode else "")
json.dump(RC, open(f"{OUT}/return_codes.json", "w"), indent=1)
shutil.rmtree(f"{OUT}/tmp", ignore_errors=True)
z = shutil.make_archive("/kaggle/working/S11_results", "zip", OUT)
print("ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB) | return codes {RC}")
'''

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.strip("\n").splitlines(keepends=True)},
                code(CELL1), code(CELL2), code(CELL3), code(CELL4)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
OUTP = "/mnt/user-data/outputs/pinn_S11_dose_augmentation_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1); print("wrote", OUTP)
