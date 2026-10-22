# build_s12_nb.py - assembles pinn_S12_quantum_classifier_kaggle.ipynb (S12 v2: VQC classifier as in the supervisor's poster,
# CMMD + CBIS-DDSM, with fair classical controls and literature controls).
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
Q3 = "'" * 3
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert Q3 not in s, path
    return s
def embed(path):
    return "r" + Q3 + rd(path) + Q3

MD = """# S12 — quantum (VQC) classifier for benign vs malignant mammograms, as in the supervisor's poster (CMMD + CBIS-DDSM)
Rules: `S12_preregistration.txt` (v2, fixed before any S12 code ran). Exact quantum-circuit **simulation** (no quantum hardware).
**Poster pipeline:** image → 4×4 patches → PCA + angle encoding → VQC (Rx Ry Rz + CNOT ring, 8 qubits × 4 layers) → ⟨Z⟩ → dense → softmax.
**Arms:** VQC on patches · Hybrid CNN+VQC · fair controls on identical inputs (same-size MLP, logistic regression, RBF-SVM) ·
literature controls (no-entanglement VQC, data re-uploading, quantum-kernel SVM) · poster's other encodings (amplitude, basis) ·
poster baselines Classical CNN / ResNet50 / EfficientNet-B0 / ConvNeXt-Tiny. Primary Q1–Q3 on BOTH datasets.

**Inputs (Add Input):** 1) the **S8 replication** notebook output (`cmmd_data/`, `clean_data/`, `cmmd_clf_s0.pt`) ·
2) the **S2 baseline-audit** notebook output (`a6_clf_s0.pt`, the CBIS-DDSM ConvNeXt; S8 used the same one).
GPU T4 ×2 · Internet ON · Run All (editor) or Save & Run All. **≈ 2–3 h.** Send back `S12_results.zip`.
"""

CELL1 = """# ===================== CELL 1 — code + inputs =====================
import os, sys, glob, json, shutil, subprocess, threading, time
from pathlib import Path
OUT = "/kaggle/working/S12"; WD = "/kaggle/working/s12_code"; PL = "/kaggle/working/pl_pkgs"
QUANTUM_BUDGET_H = 2.5            # secondary quantum configurations are skipped after this (main ones always run)
os.makedirs(OUT, exist_ok=True); os.makedirs(WD, exist_ok=True)
FILES = {"phaseA_common.py": @phaseA_common@,
         "a6_classifier.py": @a6_classifier@,
         "qsim.py": @qsim@,
         "s12_gpu.py": @s12_gpu@,
         "s12_quantum.py": @s12_quantum@,
         "s12_plcheck.py": @s12_plcheck@,
         "s12_stats.py": @s12_stats@}
for n, s in FILES.items():
    Path(f"{WD}/{n}").write_text(s)
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
        raise SystemExit(f"!! {parent}/ not found -> Add Input -> the S8 replication notebook output")
    print(f"{parent:14s} -> {h[0]}"); return h[0]
def need_file(name, where):
    h = walk_find(name)
    if not h:
        raise SystemExit(f"!! {name} not found -> Add Input -> {where}")
    print(f"{name:14s} -> {h[0]}"); return h[0]
DATA = {"cmmd": need_dir("cmmd_summary.json", "cmmd_data"), "cbis": need_dir("clean_summary.json", "clean_data")}
CLF = {"cmmd": need_file("cmmd_clf_s0.pt", "the S8 replication notebook output"),
       "cbis": need_file("a6_clf_s0.pt", "the S2 baseline-audit notebook output (the one S8 also used)")}
r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True)
NGPU = int(r.stdout.split()[-1]) if r.returncode == 0 else 0
print("GPUs visible:", NGPU)
# PennyLane into a SEPARATE folder (used only by the cross-check process; the notebook's own packages are untouched)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", PL, "pennylane"], capture_output=True, text=True)
print("PennyLane install:", "ok" if r.returncode == 0 else "FAILED (cross-check will be skipped) " + r.stderr[-300:])
LOGS = f"{OUT}/logs"; os.makedirs(LOGS, exist_ok=True); RC = {}
def run_one(name, gpu, argv, extra_env=None):
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"; env["PA_OUT"] = f"{OUT}/tmp"
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu if NGPU >= 2 else 0)
    env.update(extra_env or {})
    print(f">>> {name} started on {'CPU' if gpu is None else 'GPU ' + env['CUDA_VISIBLE_DEVICES']}", flush=True)
    with open(f"{LOGS}/{name}.log", "w") as fh:
        p = subprocess.Popen([sys.executable, "-u"] + argv, cwd=WD, env=env, text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            fh.write(line); fh.flush(); print(line if line.startswith("[") else f"[{name}] {line}", end="", flush=True)
        RC[name] = p.wait()
    print(f"<<< {name} return code {RC[name]}" + ("" if RC[name] == 0 else f"  !! see {LOGS}/{name}.log"), flush=True)
print("Cell 1 done.")
"""
for key, path in (("phaseA_common", "src8/phaseA_common.py"), ("a6_classifier", "src2/a6_classifier.py"), ("qsim", "src12/qsim.py"),
                  ("s12_gpu", "src12/s12_gpu.py"), ("s12_quantum", "src12/s12_quantum.py"), ("s12_plcheck", "src12/s12_plcheck.py"),
                  ("s12_stats", "src12/s12_stats.py")):
    CELL1 = CELL1.replace(f"@{key}@", embed(path))

CELL2 = """# ===================== CELL 2 — smoke test (~8-10 min) =====================
SM = "/kaggle/working/s12_smoke"; shutil.rmtree(SM, ignore_errors=True)
if os.path.isdir(PL):
    run_one("pennylane_check", None, ["s12_plcheck.py", OUT], {"PYTHONPATH": PL})
    if RC["pennylane_check"] != 0:
        raise SystemExit("!! simulator disagrees with PennyLane (or the check crashed) -> send Claude S12/logs/pennylane_check.log")
for ds in ("cmmd", "cbis"):
    run_one(f"smoke_feats_{ds}", 0, ["s12_gpu.py", "--task", "feats", "--ds", ds, "--data", DATA[ds], "--clf", CLF[ds], "--out", SM, "--smoke"])
    run_one(f"smoke_quantum_{ds}", None, ["s12_quantum.py", "--ds", ds, "--data", DATA[ds], "--feats", f"{SM}/cnn_features_{ds}.npz",
                                          "--out", SM, "--seeds", "0", "--smoke"])
run_one("smoke_cnn", 0, ["s12_gpu.py", "--task", "cnn", "--ds", "cmmd", "--data", DATA["cmmd"], "--out", SM, "--seeds", "0", "--smoke"])
ok = all(v == 0 for k, v in RC.items() if k.startswith("smoke")) and len(glob.glob(f"{SM}/pred_cmmd_cnn_*.npz")) == 4 and \\
     all(len(glob.glob(f"{SM}/pred_{ds}_*_VQC_s0.npz")) == 5 and len(glob.glob(f"{SM}/pred_{ds}_*qkernel_s0.npz")) == 2 for ds in ("cmmd", "cbis"))
if not ok:
    raise SystemExit("SMOKE TEST FAILED -> send Claude this block + the S12/logs/smoke_*.log files")
shutil.rmtree(SM, ignore_errors=True); print("SMOKE TEST PASSED")
"""

CELL3 = """# ===================== CELL 3 — full run (~2-3 h): features, then CNNs (GPU 0 = CMMD, GPU 1 = CBIS) + quantum (CPU) =====================
t0 = time.time()
for ds in ("cmmd", "cbis"):
    run_one(f"feats_{ds}", 0, ["s12_gpu.py", "--task", "feats", "--ds", ds, "--data", DATA[ds], "--clf", CLF[ds], "--out", OUT])
    if RC[f"feats_{ds}"] != 0:
        raise SystemExit(f"!! feature extraction failed -> send Claude S12/logs/feats_{ds}.log")
jobs = [(f"cnn_{ds}", g, ["s12_gpu.py", "--task", "cnn", "--ds", ds, "--data", DATA[ds], "--out", OUT]) for g, ds in enumerate(("cmmd", "cbis"))]
jobs += [(f"quantum_{ds}", None, ["s12_quantum.py", "--ds", ds, "--data", DATA[ds], "--feats", f"{OUT}/cnn_features_{ds}.npz", "--out", OUT,
                                  "--budget_h", str(QUANTUM_BUDGET_H)]) for ds in ("cmmd", "cbis")]
ths = [threading.Thread(target=run_one, args=j) for j in jobs]
for t in ths: t.start()
for t in ths: t.join()
print(f"all jobs finished in {(time.time() - t0) / 60:.0f} min | return codes {RC}")
"""

CELL4 = """# ===================== CELL 4 — endpoints, tables, figures, zip =====================
r = subprocess.run([sys.executable, "s12_stats.py", OUT, DATA["cmmd"]], cwd=WD, capture_output=True, text=True)
print(r.stdout[-40000:]); print(r.stderr[-3000:] if r.returncode else "")
from IPython.display import Image, display
for f in sorted(glob.glob(f"{OUT}/S12_*.png")):
    print(os.path.basename(f)); display(Image(filename=f, width=950))
json.dump(RC, open(f"{OUT}/return_codes.json", "w"), indent=1)
shutil.rmtree(f"{OUT}/tmp", ignore_errors=True)
z = shutil.make_archive("/kaggle/working/S12_results", "zip", OUT)
print("ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB) | return codes {RC}")
"""

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.strip("\n").splitlines(keepends=True)},
                code(CELL1), code(CELL2), code(CELL3), code(CELL4)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
OUTP = "/mnt/user-data/outputs/pinn_S12_quantum_classifier_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1); print("wrote", OUTP)
