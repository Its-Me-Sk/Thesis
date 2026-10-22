# build_s9_nb.py - assembles pinn_S9_diagnostics_kaggle.ipynb (S9+: exploratory CPU diagnostics, no GPU, no training).
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

MD = r"""# S9+ — exploratory diagnostics (CPU only, ≈ 15–25 min)
Nothing here trains a model or changes a pre-registered result. It explains *why* the physics terms did not help:
**B** CMMD noise-level function at original resolution → ½ → ¼ → 256 px · **C** measured noise curves, real vs generated ·
**D** radial power spectra · **E** intensity distributions · **F** KID stability vs number of images · **G** model table ·
**I** FP-gradient conditioning by timestep (from the S8 training logs).

**Inputs (Add Input):** 1) the **S8 replication** notebook output (latest version) · 2) the dataset
**"The Chinese Mammography Database (CMMD) 2022"** (tommyngx), the same one S8 used.
**Settings:** Accelerator **None** (CPU) · Internet not needed · Save & Run All (Commit). Send back `S9_results.zip`.
"""

CELL1 = r'''# ===================== CELL 1 — code + inputs =====================
import os, sys, glob, json, shutil, subprocess, time
from pathlib import Path
OUT = "/kaggle/working/S9"; WD = "/kaggle/working/s9_code"
os.makedirs(OUT, exist_ok=True); os.makedirs(WD, exist_ok=True)
Path(f"{WD}/s9_analysis.py").write_text(r''' + "'''" + rd("src9/s9_analysis.py") + "'''" + r''')
def walk_find(name, root="/kaggle/input"):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in ("CMMD", "jpeg", "U24_synthesis", ".git", "ckpt", "train_logs", "logs")]
        if name in fns:
            out.append(os.path.join(dp, name))
    return sorted(out)
def need_dir(summary, parent):
    h = [os.path.dirname(p) for p in walk_find(summary) if os.path.basename(os.path.dirname(p)) == parent]
    if not h:
        raise SystemExit(f"!! {parent}/{summary} not found -> Add Input -> the S8 replication notebook output")
    print(f"{parent:10s} -> {h[0]}"); return h[0]
DATA_CMMD = need_dir("cmmd_summary.json", "cmmd_data"); DATA_CBIS = need_dir("clean_summary.json", "clean_data")
S8_DIR = need_dir("S8_state.json", "S8")
for f in ("S8_cmmd_main_val_features.npz", "S8_cbis_long_val_features.npz", "S8_cbis_sep_val_features.npz", "gen_c40.npz", "gen_b60.npz"):
    print(f"  {'ok     ' if os.path.exists(os.path.join(S8_DIR, f)) else 'MISSING'} {f}")
print("  training logs:", sorted(os.path.basename(p) for p in glob.glob(f"{S8_DIR}/train_logs/*_fp_log.jsonl")))
xl = walk_find("CMMD_clinicaldata_revision.xlsx")
CMMD_ROOT = os.path.dirname(xl[0]) if xl else ""
print("CMMD DICOM root ->", CMMD_ROOT or "NOT FOUND (part B will be skipped: Add Input -> the CMMD dataset)")
print("Cell 1 done.")
'''

CELL2 = r'''# ===================== CELL 2 — run all parts (~15-25 min on CPU) =====================
t0 = time.time()
p = subprocess.Popen([sys.executable, "-u", "s9_analysis.py", S8_DIR, DATA_CMMD, DATA_CBIS, OUT, "150", CMMD_ROOT], cwd=WD,
                     text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
with open(f"{OUT}/S9_run.log", "w") as fh:
    for line in p.stdout:
        fh.write(line); fh.flush(); print(line, end="", flush=True)
rc = p.wait(); print(f"return code {rc} | {(time.time() - t0) / 60:.1f} min" + ("" if rc == 0 else "  !! send Claude S9/S9_run.log"))
'''

CELL3 = r'''# ===================== CELL 3 — zip =====================
from IPython.display import Image, display
for f in sorted(glob.glob(f"{OUT}/*.png")):
    print(os.path.basename(f)); display(Image(filename=f, width=900))
z = shutil.make_archive("/kaggle/working/S9_results", "zip", OUT)
print("ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB)")
'''

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.strip("\n").splitlines(keepends=True)},
                code(CELL1), code(CELL2), code(CELL3)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "none", "isInternetEnabled": False, "isGpuEnabled": False}},
      "nbformat": 4, "nbformat_minor": 4}
OUTP = "/mnt/user-data/outputs/pinn_S9_diagnostics_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1); print("wrote", OUTP)
