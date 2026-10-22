# build_s3_nb.py - assembles pinn_S3_fp_feasibility_kaggle.ipynb (S3: FP feasibility + lambda selection, 2 epochs from ep 40)
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

pa = json.load(open("/mnt/user-data/outputs/pinn_phaseA_kaggle.ipynb"))
SETUP = "".join(pa["cells"][2]["source"])
assert "CELL 2" in SETUP

MD = r"""# S3 — Fokker–Planck feasibility and λ selection (validation only; the TEST SET is not used)
All arms start from the **same** clean baseline (seed 42, epoch 40: weights + Adam state + loss scaler) and train **2 more
epochs** (→ epoch 42) with the exact Experiment-2 recipe and the same data order. The only difference between arms is λ.

| arm | λ | what |
|---|---|---|
| `s3_m0` | 0 | control: plain continuation of the baseline |
| `s3_l016` | 0.016 | FP term, small |
| `s3_l054` | 0.054 | FP term, medium |
| `s3_l107` | 0.107 | FP term, large |

FP term (once per optimizer step): `L_FP = r(u,v1)·r(u,v2) / S_ref(t)²`, t ∈ [10, 300], exact gradient by manual recompute,
added to the base gradient **before** the single clip at 1.0. Every step logs gradient norms, cos(base, FP), clipping and the
predicted change of both losses.

**Pre-stated decision** (`s3_decide.py`): R(λ) = 1 − H(λ)/H(M0) on 32 fixed validation images (32 patients), paired.
λ eligible if R ≥ 0.10, CI lower bound > 0, R_jvp > 0, validation loss and MSE ≤ 1.02× control, saturation ≤ 1.25× control
and ≤ 0.10, KID-mammo ≤ 1.5× control. Smallest eligible λ → LOCK. None eligible but some CI > 0 → INCONCLUSIVE (S3b).
Otherwise → H1 FAIL.

**How to run (on the account with the free quota)**
1. On the account that owns them, share the notebooks **clean-data-baseline-retraining** and the **S2 baseline-audit** notebook with
   this account (Share → add the account → *Can view*).
2. Here: Add Input → *Your Work* / *Shared with you* → **clean-data-baseline-retraining** (latest version) **and** the **S2 baseline-audit** notebook output.
3. Settings → Accelerator **GPU T4 ×2** · Internet **ON**.
4. **Save Version → Save & Run All (Commit)**. Then **stop the interactive session** (it also uses GPU quota).

**Time:** setup ≈ 15 min + smoke test ≈ 10 min + ≈ 4.9 h of the two GPU queues (FP arms ≈ 57 min/epoch, control ≈ 11 min/epoch, each evaluation ≈ 45 min incl. 200 samples) + report ≈ 2 min → **≈ 5.5 h of GPU quota**.
Hard limit `BUDGET_H = 10.5` h (Cell 1): a job is skipped if it would pass it; training stops cleanly at an epoch boundary.
**Send back:** `S3/S3_report.txt` + the `S3/grid_*.png` images (or the whole `S3_results.zip`).
"""

CELL1 = r'''# ===================== CELL 1 — CONFIG =====================
BUDGET_H = 10.5           # hard limit for this session (12 h Kaggle limit - margin); the run itself needs ~5 h
ARMS = {"s3_m0": 0.0, "s3_l016": 0.016, "s3_l054": 0.054, "s3_l107": 0.107}   # control first
TO_EPOCH = 42             # warm start at epoch 40 -> 2 more epochs
RUN_SMOKE = True          # ~10 min end-to-end test of training + evaluation before the long jobs
import os, time
os.environ["PA_OUT"] = "/kaggle/working/S3"
os.makedirs(os.environ["PA_OUT"], exist_ok=True)
if not os.path.exists("/tmp/pg_session_start"):
    open("/tmp/pg_session_start", "w").write(str(time.time()))
def used_h():
    return (time.time() - float(open("/tmp/pg_session_start").read())) / 3600
print("S3 FP feasibility | budget", BUDGET_H, "h | arms", ARMS, "| output", os.environ["PA_OUT"])
'''

CELL3 = r'''# ===================== CELL 3 — code + clean data (reused) + input checks + job runner =====================
import os, sys, glob, json, shutil, subprocess, threading, time
from pathlib import Path
WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
os.chdir(WD)
FILES = {"phaseA_common.py": r''' + "'''" + rd("src2/phaseA_common.py") + "'''" + r''',
         "fp_core.py": r''' + "'''" + rd("src2/fp_core.py") + "'''" + r''',
         "a6_classifier.py": r''' + "'''" + rd("src2/a6_classifier.py") + "'''" + r''',
         "train_fp.py": r''' + "'''" + rd("src3/train_fp.py") + "'''" + r''',
         "s3_eval.py": r''' + "'''" + rd("src3/s3_eval.py") + "'''" + r''',
         "s3_decide.py": r''' + "'''" + rd("src3/s3_decide.py") + "'''" + r'''}
for n, s in FILES.items():
    Path(n).write_text(s)
Path("/tmp/prepare_clean.py").write_text(r''' + "'''" + rd("src/prepare_clean.py") + "'''" + r''')
print("wrote", ", ".join(FILES))
subprocess.run([sys.executable, "/tmp/prepare_clean.py"], check=True)      # reuses clean_data/ from the input (identical bytes)

def need(name, hint):
    h = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    print(f"{name}: {h[0] if h else '!! NOT FOUND'}")
    if not h:
        raise SystemExit(f"!! {name} not found -> {hint}")
    return h[0]
INIT = need("base_clean_s42_last.pt", "Add Input -> clean-data-baseline-retraining (latest version)")
BASE40 = need("base_clean_s42_w_ep040.pt", "Add Input -> clean-data-baseline-retraining (latest version)")
A8 = need("A8_fp_clean.json", "Add Input -> the S2 baseline-audit notebook output")
need("a6_clf_s0.pt", "Add Input -> the S2 baseline-audit notebook output")
sref = json.load(open(A8))["S_ref_table"]
print("S_ref(t) from S2:", {k: float(f"{v:.3g}") for k, v in sref.items()})
r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True)
NGPU = int(r.stdout.split()[-1]) if r.returncode == 0 else 0
print("GPUs visible:", NGPU, "" if NGPU >= 2 else "!! expected 2 (Settings -> Accelerator -> GPU T4 x2)")
CK = "/kaggle/working/ckpt"; os.makedirs(CK, exist_ok=True)
LOGS = "/kaggle/working/S3/logs"; os.makedirs(LOGS, exist_ok=True)
RC = {}

def run_one(name, gpu, argv, est_h, out_dir=None):
    """one job in a subprocess; skipped (rc = 'skipped') if it would pass BUDGET_H (0.4 h kept for the report)"""
    if used_h() + est_h > BUDGET_H - 0.4:
        print(f"--- {name} SKIPPED: {used_h():.2f} h used + {est_h} h estimate would pass the budget", flush=True)
        RC[name] = "skipped"; return
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"; env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu if NGPU >= 2 else 0)
    if out_dir:
        env["PA_OUT"] = out_dir
    print(f">>> {name} started on GPU {env['CUDA_VISIBLE_DEVICES']} | session {used_h():.2f} h", flush=True)
    with open(f"{LOGS}/{name}.log", "w") as fh:
        p = subprocess.Popen([sys.executable, "-u"] + argv, cwd=WD, env=env, text=True, bufsize=1,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            fh.write(line); fh.flush()
            print(line if line.startswith("[") else f"[{name}] {line}", end="", flush=True)
        RC[name] = p.wait()
    print(f"<<< {name} return code {RC[name]} | session {used_h():.2f} h" + ("" if RC[name] == 0 else f"  !! see {LOGS}/{name}.log"), flush=True)

def run_queues(queues):
    """queues = {gpu: [(name, argv, est_h), ...]} -> each GPU runs its list in order, the GPUs in parallel"""
    if NGPU < 2:
        queues = {0: [j for q in queues.values() for j in q]}
    ths = [threading.Thread(target=lambda g=g, q=q: [run_one(n, g, a, e) for n, a, e in q]) for g, q in queues.items()]
    for t in ths: t.start()
    for t in ths: t.join()

def train_job(arm, ck_dir=CK, extra=()):
    return ["train_fp.py", "--arm", arm, "--lam", str(ARMS.get(arm, 0.054)), "--epochs", str(TO_EPOCH), "--init", INIT,
            "--sref_json", A8, "--budget_h", str(BUDGET_H - 0.3), "--ckpt_dir", ck_dir] + list(extra)
def eval_job(arm, extra=()):
    return ["s3_eval.py", "--arm", arm, "--epochs", ",".join(str(e) for e in range(41, TO_EPOCH + 1)), "--ckpt_dir", CK,
            "--final"] + list(extra)
print("Cell 3 done.")
'''

CELL4 = r'''# ===================== CELL 4 — SMOKE TEST (8 micro-batches of FP training + a tiny evaluation, ~10 min) =====================
SM = "/kaggle/working/smoke"
started = any(os.path.exists(f"{CK}/{arm}_last.pt") or glob.glob(f"/kaggle/input/**/{arm}_last.pt", recursive=True)
              for arm in ARMS)
if RUN_SMOKE and not started:
    os.makedirs(f"{SM}/eval", exist_ok=True)
    run_one("smoke_train", 0, ["train_fp.py", "--arm", "smoke", "--lam", "0.054", "--epochs", "41", "--init", INIT,
                               "--sref_json", A8, "--ckpt_dir", SM, "--smoke_batches", "8", "--log_every", "4"], 0.2)
    if RC.get("smoke_train") == 0:
        run_one("smoke_eval", 0, ["s3_eval.py", "--arm", "smoke", "--epochs", "41", "--ckpt_dir", SM, "--final", "--smoke"],
                0.2, out_dir=f"{SM}/eval")
    lines = [json.loads(l) for l in open(f"{SM}/smoke_fp_log.jsonl")] if os.path.exists(f"{SM}/smoke_fp_log.jsonl") else []
    ok = RC.get("smoke_train") == 0 and RC.get("smoke_eval") == 0 and len(lines) == 2 and all(l.get("fp_ok") for l in lines)
    for l in lines:
        print("smoke optimizer step:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in l.items()})
    if not ok:
        raise SystemExit("SMOKE TEST FAILED -> nothing long was started. Send Claude this block + S3/logs/smoke_*.log")
    shutil.rmtree(SM, ignore_errors=True)
    print(f"SMOKE TEST PASSED | session {used_h():.2f} h")
else:
    print("smoke test skipped", "(training already started in an earlier session)" if started else "(RUN_SMOKE = False)")
'''

CELL5 = r'''# ===================== CELL 5 — RUN: two GPU queues in parallel (each arm resumes from its own *_last.pt) =====================
t0 = time.time()
run_queues({
    0: [("train_s3_l016", train_job("s3_l016"), 2.0),
        ("train_s3_l107", train_job("s3_l107"), 2.0),
        ("eval_s3_l107", eval_job("s3_l107"), 0.75)],
    1: [("train_s3_l054", train_job("s3_l054"), 2.0),
        ("train_s3_m0", train_job("s3_m0"), 0.45),
        ("eval_s3_m0", eval_job("s3_m0"), 0.75),
        ("eval_s3_l054", eval_job("s3_l054"), 0.75),
        ("eval_s3_l016", eval_job("s3_l016"), 0.75),
        ("eval_base", ["s3_eval.py", "--arm", "base", "--epochs", "40", "--ckpt", BASE40], 0.25)]})
print(f"===== queues done in {(time.time() - t0) / 3600:.2f} h | return codes {RC} | session {used_h():.2f} h =====")
'''

CELL6 = r'''# ===================== CELL 6 — DECISION + REPORT + zip =====================
arms_arg = ",".join(f"{k}:{v}" for k, v in ARMS.items())
r = subprocess.run([sys.executable, "s3_decide.py", "--arms", arms_arg, "--final_epoch", str(TO_EPOCH),
                    "--epochs", ",".join(str(e) for e in range(41, TO_EPOCH + 1))], cwd=WD, capture_output=True, text=True)
print(r.stdout[-40000:])
if r.returncode != 0:
    print("!! s3_decide.py failed:", r.stderr[-2000:])
TL = "/kaggle/working/S3/train_logs"; os.makedirs(TL, exist_ok=True)
for f in glob.glob(f"{CK}/*_fp_log.jsonl") + glob.glob(f"{CK}/*_history.json"):
    shutil.copy(f, TL)
json.dump(RC, open("/kaggle/working/S3/return_codes.json", "w"), indent=1)
z = shutil.make_archive("/kaggle/working/S3_results", "zip", "/kaggle/working/S3")
print("return codes:", RC, "| ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB) | session {used_h():.2f} h")
print("checkpoints kept in /kaggle/working/ckpt (needed for S3b / Phase C):",
      sorted(os.path.basename(f) for f in glob.glob(f"{CK}/*.pt")))
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
OUTP = "/mnt/user-data/outputs/pinn_S3_fp_feasibility_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1)
print("wrote", OUTP)
