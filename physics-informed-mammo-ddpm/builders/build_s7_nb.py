# build_s7_nb.py - assembles pinn_S7_acquisition_kaggle.ipynb (S7: physics-based acquisition layer = Poisson-Gaussian
# noise-level function fitted on real train-fit images, added to the generated baseline images). No diffusion training.
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

pa = json.load(open("/mnt/user-data/outputs/pinn_phaseA_kaggle.ipynb"))
SETUP = "".join(pa["cells"][2]["source"])
assert "CELL 2" in SETUP

MD = r"""# S7 (last experiment) — physics-based acquisition layer
No training. Rules: `S7_preregistration.txt` (+ Addendum 1), fixed before any S7 code ran.

The diffusion model draws a clean image; a real mammogram is that image **plus acquisition noise**.
S7 fits the Poisson–Gaussian noise-level function σ²(I) = a·I + b on the **real train-fit images only** and adds that
noise to the generated baseline images (`v_base`, epoch 40, because S6 had no winner).

* **Validation:** `v_base` vs `v_base + noise`, reference = train-fit images, 95 % CI.
* **Test (only if validation PASSES, run once):** `t_base` vs `t_base + noise`, fresh noise seed, reference = TEST images, **97.5 % CI**.
* PASS = KID-mammo ≤ 0.90× with paired CI < 0, KID-DINOv2 ≤ 1.00×, recall ≥ baseline − 0.05 (same as S6).

**How to run:** Add Input → **clean-data-baseline-retraining**, **S2** and **S6 (final)** outputs · GPU T4 ×2 · Internet ON ·
Save & Run All (Commit) · stop the session. **≈ 30–40 min.** Send back `S7_results.zip`.
"""

CELL1 = r'''# ===================== CELL 1 — CONFIG =====================
BUDGET_H = 3.0
VAL_SEED, TEST_SEED = 7007, 9707      # noise seeds (pre-registered, never used before)
TEST_ALPHA = 0.025                    # 97.5 % CI at the test stage (2nd confirmation in the project)
RUN_SMOKE = True
import os, time
os.environ["PA_OUT"] = "/kaggle/working/S7"
os.makedirs(os.environ["PA_OUT"], exist_ok=True)
if not os.path.exists("/tmp/pg_session_start"):
    open("/tmp/pg_session_start", "w").write(str(time.time()))
def used_h():
    return (time.time() - float(open("/tmp/pg_session_start").read())) / 3600
print("S7 acquisition layer | budget", BUDGET_H, "h | output", os.environ["PA_OUT"])
'''

CELL3 = r'''# ===================== CELL 3 — code + clean data + inputs + job runner =====================
import os, sys, glob, json, shutil, subprocess, time
from pathlib import Path
WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
os.chdir(WD)
FILES = {"phaseA_common.py": r''' + "'''" + rd("src2/phaseA_common.py") + "'''" + r''',
         "a6_classifier.py": r''' + "'''" + rd("src2/a6_classifier.py") + "'''" + r''',
         "pinn_physics.py": r''' + "'''" + rd("ref/pinn_physics.py") + "'''" + r''',
         "s7_noise.py": r''' + "'''" + rd("src7/s7_noise.py") + "'''" + r''',
         "s7_stats.py": r''' + "'''" + rd("src7/s7_stats.py") + "'''" + r''',
         "s7_eval.py": r''' + "'''" + rd("src7/s7_eval.py") + "'''" + r'''}
for n, s in FILES.items():
    Path(n).write_text(s)
Path("/tmp/prepare_clean.py").write_text(r''' + "'''" + rd("src/prepare_clean.py") + "'''" + r''')
subprocess.run([sys.executable, "/tmp/prepare_clean.py"], check=True)
def find_in(name):
    h = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    print(f"{name}: {h[0] if h else '-- not found'}"); return h[0] if h else None
def need(name, hint):
    p = find_in(name)
    if p is None:
        raise SystemExit(f"!! {name} not found -> attach the {hint} notebook output (Add Input)")
    return p
OUT = os.environ["PA_OUT"]
need("a6_clf_s0.pt", "S2")
for nm in ("v_base", "t_base"):
    shutil.copy(need(f"gen_{nm}.npz", "S6 (final)"), f"{OUT}/gen_{nm}.npz")
LOGS = f"{OUT}/logs"; os.makedirs(LOGS, exist_ok=True)
RC = {}
def run_one(name, argv, est_h, out_dir=None):
    if used_h() + est_h > BUDGET_H - 0.2:
        print(f"--- {name} SKIPPED (budget)", flush=True); RC[name] = "skipped"; return
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"; env["CUDA_VISIBLE_DEVICES"] = "0"
    if out_dir:
        env["PA_OUT"] = out_dir
    print(f">>> {name} started | session {used_h():.2f} h", flush=True)
    with open(f"{LOGS}/{name}.log", "w") as fh:
        p = subprocess.Popen([sys.executable, "-u"] + argv, cwd=WD, env=env, text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            fh.write(line); fh.flush(); print(line if line.startswith("[") else f"[{name}] {line}", end="", flush=True)
        RC[name] = p.wait()
    print(f"<<< {name} return code {RC[name]} | session {used_h():.2f} h" + ("" if RC[name] == 0 else f"  !! see {LOGS}/{name}.log"), flush=True)
print("Cell 3 done.")
'''

CELL4 = r'''# ===================== CELL 4 — SMOKE TEST (~5 min) =====================
SM = "/kaggle/working/smoke"
if RUN_SMOKE and not os.path.exists(f"{OUT}/S7_decision.json"):
    os.makedirs(SM, exist_ok=True)
    subprocess.run([sys.executable, "-c", "import numpy as np; d = np.load('" + OUT + "/gen_v_base.npz'); "
                    "np.savez('" + SM + "/gen_sm.npz', x=d['x'][::25], y=d['y'][::25])"], check=True)
    run_one("smoke_noise", ["s7_noise.py", "--sets", "sm", "--seed", "1"], 0.05, SM)
    run_one("smoke_eval", ["s7_eval.py", "--stage", "val", "--sets", "sm,sm_nl", "--n_boot", "20"], 0.1, SM)
    if not (RC.get("smoke_noise") == 0 and RC.get("smoke_eval") == 0 and os.path.exists(f"{SM}/S7_decision.json")):
        raise SystemExit("SMOKE TEST FAILED -> nothing long was started. Send Claude this block + S7/logs/smoke_*.log")
    shutil.rmtree(SM, ignore_errors=True); RC.clear(); print(f"SMOKE TEST PASSED | session {used_h():.2f} h")
'''

CELL5 = r'''# ===================== CELL 5 — VALIDATION (reference = train-fit images) =====================
run_one("noise_val", ["s7_noise.py", "--sets", "v_base", "--seed", str(VAL_SEED)], 0.1)
run_one("eval_val", ["s7_eval.py", "--stage", "val", "--sets", "v_base,v_base_nl"], 0.2)
DEC = json.load(open(f"{OUT}/S7_decision.json")) if os.path.exists(f"{OUT}/S7_decision.json") else {"winner": None}
print("S7 VALIDATION:", "PASS -> one test confirmation" if DEC["winner"] == "v_base_nl" else "FAIL -> no test stage (pre-stated). S7 is negative.")
'''

CELL6 = r'''# ===================== CELL 6 — TEST (only if validation passed) + report + zip =====================
if DEC["winner"] == "v_base_nl":
    run_one("noise_test", ["s7_noise.py", "--sets", "t_base", "--seed", str(TEST_SEED)], 0.1)
    run_one("eval_test", ["s7_eval.py", "--stage", "test", "--sets", "t_base,t_base_nl", "--alpha", str(TEST_ALPHA)], 0.3)
rep = "\n\n".join(open(f"{OUT}/{f}").read() for f in ("S7_val_report.txt", "S7_test_report.txt") if os.path.exists(f"{OUT}/{f}"))
open(f"{OUT}/S7_report.txt", "w").write(rep or "!! no report produced"); print(rep)
json.dump(RC, open(f"{OUT}/return_codes.json", "w"), indent=1)
z = shutil.make_archive("/kaggle/working/S7_results", "zip", OUT)
print("ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB) | session {used_h():.2f} h | return codes {RC}")
'''

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.strip("\n").splitlines(keepends=True)},
                code(CELL1), code(SETUP), code(CELL3), code(CELL4), code(CELL5), code(CELL6)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
OUTP = "/mnt/user-data/outputs/pinn_S7_acquisition_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1); print("wrote", OUTP)
