# build_s4a_nb.py - assembles pinn_S4a_screening_kaggle.ipynb (S4a: FP gradient signal test + cheap FP-derived regularisers)
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

pa = json.load(open("/mnt/user-data/outputs/pinn_phaseA_kaggle.ipynb"))
SETUP = "".join(pa["cells"][2]["source"])
assert "CELL 2" in SETUP

MD = r"""# S4a — FP gradient signal test + screening of cheap Fokker–Planck-derived regularisers (validation only; TEST SET NOT USED)
**Part B (no training, ≈1 h):** on the epoch-40 baseline: Fokker–Planck residual profile for t = 1…990, exact-JVP check of the
finite-difference step at small t, and the signal-to-noise ratio of the FP gradient vs the ordinary training gradient
(how many FP samples per step a usable FP gradient needs → settings of the stabilised FP arm R4 in S4b).

**Screening:** all arms continue the same clean baseline (seed 42) to **epoch 44** with the exact Experiment-2 recipe and the same
data order; the control is `s3_m0` (from S3) continued from epoch 42 to 44.

| arm | regulariser | where | strength ρ = median ‖λ g_reg‖ / ‖g_base‖ |
|---|---|---|---|
| `s3_m0` | none (control) | — | — |
| `s4_r1a` / `s4_r1b` | score norm σ²‖s‖² (= mean ε̂²) | all t | 0.25 / 0.10 |
| `s4_r2a` / `s4_r2b` | score norm | t ≤ 50 | 0.25 / 0.10 |
| `s4_r3` | divergence (div ε̂)² (2-probe Hutchinson, fp32 FD) | all t | 0.25 |

λ is calibrated once at epoch 40 to the stated ρ; every step the regulariser gradient is capped at 2ρ‖g_base‖ (the S3 failure
cause). **Pre-stated pass rule** (`s4_decide.py`): KID-mammo ≤ 0.90× control with the paired-bootstrap CI below 0, validation
MSE and loss ≤ 1.02× control, saturation ≤ max(1.25× control, control + 0.005), memorisation ≤ 1 %. Winner = largest KID
reduction among passing arms → LOCK / Phase C. The FP residual of every arm is reported (physics-consistent vs generic).

**How to run (account with the S3 notebook)**
1. Add Input → **clean-data-baseline-retraining**, the **S2 baseline-audit** output and the **S3** output (it holds `s3_m0_last.pt`).
2. Settings → Accelerator **GPU T4 ×2** · Internet **ON**.
3. **Save Version → Save & Run All (Commit)**, then stop the interactive session.

**Time:** setup ≈ 10 min + smoke test ≈ 12 min + ≈ 7 h of the two GPU queues + report ≈ 2 min → **≈ 7.5 h of GPU quota**
(hard limit `BUDGET_H = 10.5` h). **Send back:** `S4/S4_report.txt` + `S4/grid_*.png` (or `S4_results.zip`).
"""

CELL1 = r'''# ===================== CELL 1 — CONFIG =====================
BUDGET_H = 10.5           # hard limit for this session; the run needs ~7.5 h
TO_EPOCH = 44
CONTROL = "s3_m0"
ARMS = {                  # name: (regulariser, rho, t_min, t_max)
    "s4_r1a": ("sn", 0.25, 0, 999),
    "s4_r1b": ("sn", 0.10, 0, 999),
    "s4_r2a": ("sn", 0.25, 0, 50),
    "s4_r2b": ("sn", 0.10, 0, 50),
    "s4_r3":  ("div", 0.25, 1, 999)}
RUN_SMOKE = True
import os, time
os.environ["PA_OUT"] = "/kaggle/working/S4"
os.makedirs(os.environ["PA_OUT"], exist_ok=True)
if not os.path.exists("/tmp/pg_session_start"):
    open("/tmp/pg_session_start", "w").write(str(time.time()))
def used_h():
    return (time.time() - float(open("/tmp/pg_session_start").read())) / 3600
print("S4a screening | budget", BUDGET_H, "h | control", CONTROL, "| arms", ARMS)
'''

CELL3 = r'''# ===================== CELL 3 — code + clean data (reused) + input checks + job runner =====================
import os, sys, glob, json, shutil, subprocess, threading, time
from pathlib import Path
WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
os.chdir(WD)
FILES = {"phaseA_common.py": r''' + "'''" + rd("src2/phaseA_common.py") + "'''" + r''',
         "fp_core.py": r''' + "'''" + rd("src2/fp_core.py") + "'''" + r''',
         "a6_classifier.py": r''' + "'''" + rd("src2/a6_classifier.py") + "'''" + r''',
         "train_reg.py": r''' + "'''" + rd("src4/train_reg.py") + "'''" + r''',
         "s4_gradsnr.py": r''' + "'''" + rd("src4/s4_gradsnr.py") + "'''" + r''',
         "s4_eval.py": r''' + "'''" + rd("src4/s4_eval.py") + "'''" + r''',
         "s4_decide.py": r''' + "'''" + rd("src4/s4_decide.py") + "'''" + r'''}
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
need("A8_fp_clean.json", "Add Input -> the S2 baseline-audit notebook output")
need("a6_clf_s0.pt", "Add Input -> the S2 baseline-audit notebook output")
need(f"{CONTROL}_last.pt", "Add Input -> the S3 notebook output (holds the control checkpoint)")
r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True)
NGPU = int(r.stdout.split()[-1]) if r.returncode == 0 else 0
print("GPUs visible:", NGPU, "" if NGPU >= 2 else "!! expected 2 (Settings -> Accelerator -> GPU T4 x2)")
CK = "/kaggle/working/ckpt"; os.makedirs(CK, exist_ok=True)
LOGS = "/kaggle/working/S4/logs"; os.makedirs(LOGS, exist_ok=True)
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
    """queues = {gpu: [(name, argv, est_h[, out_dir]), ...]} -> each GPU runs its list in order, the GPUs in parallel"""
    if NGPU < 2:
        queues = {0: [j for q in queues.values() for j in q]}
    ths = [threading.Thread(target=lambda g=g, q=q: [run_one(n, g, a, e, *rest) for n, a, e, *rest in q]) for g, q in queues.items()]
    for t in ths: t.start()
    for t in ths: t.join()

def train_job(arm, ck_dir=CK, extra=()):
    if arm == CONTROL:
        reg, rho, tmin, tmax = "none", 0.0, 0, 999
    else:
        reg, rho, tmin, tmax = ARMS[arm]
    return ["train_reg.py", "--arm", arm, "--reg", reg, "--rho", str(rho), "--reg_tmin", str(tmin), "--reg_tmax", str(tmax),
            "--epochs", str(TO_EPOCH), "--init", INIT, "--budget_h", str(BUDGET_H - 0.3), "--ckpt_dir", ck_dir] + list(extra)
def eval_job(arm):
    return ["s4_eval.py", "--arm", arm, "--epochs", str(TO_EPOCH), "--ckpt_dir", CK, "--final"]
print("Cell 3 done.")
'''

CELL4 = r'''# ===================== CELL 4 — SMOKE TEST (both new regularisers, part B and the evaluation on tiny settings, ~12 min) =====================
SM = "/kaggle/working/smoke"
started = any(os.path.exists(f"{CK}/{arm}_last.pt") or glob.glob(f"/kaggle/input/**/{arm}_last.pt", recursive=True) for arm in ARMS)
if RUN_SMOKE and not started:
    os.makedirs(f"{SM}/eval", exist_ok=True)
    sm = lambda arm, reg, tmin: ["train_reg.py", "--arm", arm, "--reg", reg, "--rho", "0.25", "--reg_tmin", str(tmin), "--reg_tmax", "999",
                                 "--epochs", "41", "--init", INIT, "--ckpt_dir", SM, "--smoke_batches", "8", "--log_every", "4"]
    run_queues({0: [("smoke_train_sn", sm("smoke_sn", "sn", 0), 0.1), ("smoke_B", ["s4_gradsnr.py", "--init", INIT, "--smoke"], 0.1, f"{SM}/eval")],
                1: [("smoke_train_div", sm("smoke_div", "div", 1), 0.1)]})
    if RC.get("smoke_train_div") == 0:
        run_one("smoke_eval", 1, ["s4_eval.py", "--arm", "smoke_div", "--epochs", "41", "--ckpt_dir", SM, "--final", "--smoke"], 0.1,
                out_dir=f"{SM}/eval")
    ok = all(RC.get(k) == 0 for k in ("smoke_train_sn", "smoke_train_div", "smoke_B", "smoke_eval"))
    for arm in ("smoke_sn", "smoke_div"):
        f = f"{SM}/{arm}_reg_log.jsonl"
        lines = [json.loads(l) for l in open(f)] if os.path.exists(f) else []
        ok = ok and len(lines) == 2 and all(l.get("reg_ok") for l in lines)
        for l in lines:
            print(f"smoke {arm} step:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in l.items()})
    ok = ok and os.path.exists(f"{SM}/eval/feat_smoke_div_ep041_gs2.npy") and os.path.exists(f"{SM}/eval/S4_B_gradsnr.json")
    if not ok:
        raise SystemExit("SMOKE TEST FAILED -> nothing long was started. Send Claude this block + S4/logs/smoke_*.log")
    shutil.rmtree(SM, ignore_errors=True)
    print(f"SMOKE TEST PASSED | session {used_h():.2f} h")
else:
    print("smoke test skipped", "(training already started in an earlier session)" if started else "(RUN_SMOKE = False)")
'''

CELL5 = r'''# ===================== CELL 5 — RUN: two GPU queues in parallel (each arm resumes from its own *_last.pt) =====================
t0 = time.time()
run_queues({
    0: [("B_gradsnr", ["s4_gradsnr.py", "--init", INIT], 1.0),
        ("train_s4_r3", train_job("s4_r3"), 2.0),
        ("train_s4_r1b", train_job("s4_r1b"), 1.5),
        ("eval_s4_r3", eval_job("s4_r3"), 0.8),
        ("eval_s4_r1b", eval_job("s4_r1b"), 0.8),
        ("eval_s4_r2b", eval_job("s4_r2b"), 0.8)],
    1: [("train_s3_m0", train_job(CONTROL), 0.5),
        ("train_s4_r1a", train_job("s4_r1a"), 1.5),
        ("train_s4_r2a", train_job("s4_r2a"), 1.1),
        ("train_s4_r2b", train_job("s4_r2b"), 1.1),
        ("eval_s3_m0", eval_job(CONTROL), 0.8),
        ("eval_s4_r1a", eval_job("s4_r1a"), 0.8),
        ("eval_s4_r2a", eval_job("s4_r2a"), 0.8)]})
print(f"===== queues done in {(time.time() - t0) / 3600:.2f} h | return codes {RC} | session {used_h():.2f} h =====")
'''

CELL6 = r'''# ===================== CELL 6 — DECISION + REPORT + zip =====================
arms_arg = ",".join([f"{CONTROL}:none:0"] + [f"{k}:{v[0]}:{v[1]}" for k, v in ARMS.items()])
r = subprocess.run([sys.executable, "s4_decide.py", "--arms", arms_arg, "--epoch", str(TO_EPOCH)], cwd=WD, capture_output=True, text=True)
print(r.stdout[-40000:])
if r.returncode != 0:
    print("!! s4_decide.py failed:", r.stderr[-2000:])
TL = "/kaggle/working/S4/train_logs"; os.makedirs(TL, exist_ok=True)
for f in glob.glob(f"{CK}/*_reg_log.jsonl") + glob.glob(f"{CK}/*_history.json"):
    shutil.copy(f, TL)
json.dump(RC, open("/kaggle/working/S4/return_codes.json", "w"), indent=1)
z = shutil.make_archive("/kaggle/working/S4_results", "zip", "/kaggle/working/S4")
print("return codes:", RC, "| ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB) | session {used_h():.2f} h")
print("checkpoints kept in /kaggle/working/ckpt:", sorted(os.path.basename(f) for f in glob.glob(f"{CK}/*.pt")))
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
OUTP = "/mnt/user-data/outputs/pinn_S4a_screening_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1)
print("wrote", OUTP)
