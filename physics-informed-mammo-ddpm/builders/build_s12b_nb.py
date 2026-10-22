# build_s12b_nb.py - assembles pinn_S12b_complete_secondary_kaggle.ipynb: CPU-only completion of the 3 secondary S12
# quantum configurations per dataset that did not run because of a code error (pure amplitude encoding with 48 patch numbers
# < 64 amplitudes). Fix = zero-padding. Everything that already exists is reused unchanged; then the full S12 report is rebuilt.
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
Q3 = "'" * 3
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert Q3 not in s, path
    return s
def embed(path):
    return "r" + Q3 + rd(path) + Q3

MD = """# S12b — complete the S12 secondary quantum arms (CPU only, ≈ 1 h, no GPU quota)
In S12, the *patch* version of amplitude encoding crashed (48 patch numbers < 64 amplitudes), so 3 secondary configurations
per dataset never ran (patch amplitude encoding, basis encoding, circuit sizes). **Bug fix: zero-pad 48 → 64** (standard
amplitude encoding). Nothing else changes; all existing S12 results are reused as they are, then the full S12 report is rebuilt.
Primary endpoints Q1–Q3 are not affected (they had already finished).

**Inputs (Add Input):** 1) the **S12 — quantum** notebook output · 2) **S8 replication** output · 3) **S2 baseline-audit** output.
**Accelerator: None (CPU)** · Internet not needed · Save & Run All (Commit). Send back `S12_results_complete.zip`.
"""

CELL1 = """# ===================== CELL 1 — code + inputs, copy the existing S12 results =====================
import os, sys, glob, json, shutil, subprocess, threading, time
from pathlib import Path
OUT = "/kaggle/working/S12"; WD = "/kaggle/working/s12_code"
os.makedirs(WD, exist_ok=True)
FILES = {"qsim.py": @qsim@, "s12_quantum.py": @s12_quantum@, "s12_stats.py": @s12_stats@}
for n, s in FILES.items():
    Path(f"{WD}/{n}").write_text(s)
def walk_find(name, root="/kaggle/input"):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in ("CMMD", "jpeg", "U24_synthesis", ".git", "ckpt", "train_logs", "logs")]
        if name in fns:
            out.append(os.path.join(dp, name))
    return sorted(out)
def need_dir(marker, parent, where):
    h = [os.path.dirname(p) for p in walk_find(marker) if os.path.basename(os.path.dirname(p)) == parent]
    if not h:
        raise SystemExit(f"!! {parent}/{marker} not found -> Add Input -> {where}")
    print(f"{parent:10s} -> {h[0]}"); return h[0]
S12_IN = need_dir("quantum_meta_cmmd.json", "S12", "the S12 — quantum notebook output")
DATA = {"cmmd": need_dir("cmmd_summary.json", "cmmd_data", "the S8 replication notebook output"),
        "cbis": need_dir("clean_summary.json", "clean_data", "the S8 replication or S2 baseline-audit output")}
shutil.rmtree(OUT, ignore_errors=True); shutil.copytree(S12_IN, OUT, copy_function=shutil.copyfile)   # writable copies
print("copied", len(glob.glob(f"{OUT}/pred_*.npz")), "existing prediction files")
for ds in ("cmmd", "cbis"):
    print(ds, "configurations already done:", list(json.load(open(f"{OUT}/quantum_meta_{ds}.json"))))
LOGS = f"{OUT}/logs"; os.makedirs(LOGS, exist_ok=True); RC = {}
def run_one(name, argv):
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"; env["CUDA_VISIBLE_DEVICES"] = ""
    print(f">>> {name} started", flush=True)
    with open(f"{LOGS}/{name}.log", "w") as fh:
        p = subprocess.Popen([sys.executable, "-u"] + argv, cwd=WD, env=env, text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            fh.write(line); fh.flush(); print(line if line.startswith("[") else f"[{name}] {line}", end="", flush=True)
        RC[name] = p.wait()
    print(f"<<< {name} return code {RC[name]}" + ("" if RC[name] == 0 else f"  !! see {LOGS}/{name}.log"), flush=True)
print("Cell 1 done.")
"""
for key, path in (("qsim", "src12/qsim.py"), ("s12_quantum", "src12/s12_quantum.py"), ("s12_stats", "src12/s12_stats.py")):
    CELL1 = CELL1.replace(f"@{key}@", embed(path))

CELL2 = """# ===================== CELL 2 — run the missing configurations (existing files are skipped) (~1 h) =====================
t0 = time.time()
ths = [threading.Thread(target=run_one, args=(f"quantum_complete_{ds}", ["s12_quantum.py", "--ds", ds, "--data", DATA[ds],
       "--feats", f"{OUT}/cnn_features_{ds}.npz", "--out", OUT, "--budget_h", "6"])) for ds in ("cmmd", "cbis")]
for t in ths: t.start()
for t in ths: t.join()
print(f"finished in {(time.time() - t0) / 60:.0f} min | return codes {RC}")
for ds in ("cmmd", "cbis"):
    print(ds, "configurations now:", list(json.load(open(f"{OUT}/quantum_meta_{ds}.json"))))
"""

CELL3 = """# ===================== CELL 3 — rebuild the full S12 report + zip =====================
r = subprocess.run([sys.executable, "s12_stats.py", OUT, DATA["cmmd"]], cwd=WD, capture_output=True, text=True)
note = ("NOTE: S12 report rebuilt after S12b (2026-10-23). S12b ran ONLY the secondary configurations that had not run in S12 "
        "because of a code error (patch amplitude encoding: 48 numbers < 64 amplitudes; fixed by zero-padding). All other results "
        "are the original S12 files, unchanged.\\n\\n")
rep = f"{OUT}/S12_report.txt"
if os.path.exists(rep):
    body = open(rep).read(); open(rep, "w").write(note + body)        # read first, then write (fix 2026-10-23)
print(note + r.stdout[-40000:]); print(r.stderr[-3000:] if r.returncode else "")
json.dump(RC, open(f"{OUT}/return_codes_S12b.json", "w"), indent=1)
z = shutil.make_archive("/kaggle/working/S12_results_complete", "zip", OUT)
print("ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB) | return codes {RC}")
"""

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.strip("\n").splitlines(keepends=True)},
                code(CELL1), code(CELL2), code(CELL3)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "none", "isInternetEnabled": False, "isGpuEnabled": False}},
      "nbformat": 4, "nbformat_minor": 4}
OUTP = "/mnt/user-data/outputs/pinn_S12b_complete_secondary_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1); print("wrote", OUTP)
