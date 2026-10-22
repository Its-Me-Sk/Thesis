# build_s5_nb.py - assembles pinn_S5_physics_scoring_kaggle.ipynb (S5: physics-consistency scoring + failure investigation)
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

pa = json.load(open("/mnt/user-data/outputs/pinn_phaseA_kaggle.ipynb"))
SETUP = "".join(pa["cells"][2]["source"])
assert "CELL 2" in SETUP

MD = r"""# S5 — physics-consistency scoring of synthetic mammograms + failure investigation (no training; TEST SET NOT USED)
Model: the epoch-40 clean baseline (seed 42). Every per-image score has a FIXED direction (lower = better), fixed before the run.

| part | what |
|---|---|
| pool | 300 generated images / class (guidance 2, DDIM-50, the usual noise bank) |
| F1 | **physics**: score Fokker–Planck residual of the model at the noised image (t = 10, 50), validated FD estimator |
| F2 | **physics**: film/digitiser noise (Poisson–Gaussian noise-level function fitted on real train-fit images) |
| F3 | model's own denoising error (t = 50, 200, 500), normalised by real validation images |
| F4 | trivial: brightness / saturation / HF energy / breast area distance to real |
| F5 | reference only: nearest-real distance in the KID feature space (never a pass criterion) |
| T1 | reliability of F1 (independent re-scoring of 80 images; Spearman ≥ 0.5) |
| T2 | filter: keep best 50 % per class → KID-mammo ≤ 0.90× median of 1000 random 50 % subsets and below their 2.5th percentile |
| T3 / E2 | real vs generated detector: AUC ≥ 0.65 with CI > 0.5 (per t for F1) |
| E3 | KID of checkpoints 35 / 40 / 41 / 42 / 44 and of their weight average (descriptive: checkpoint-selection variability) |
| E4 | Epsilon Scaling at sampling, s = 1.005 / 1.01 / 1.02 (helps if KID ≤ 0.90× with paired CI < 0) |

**How to run (account with the S3 and S4a notebooks)**
1. Add Input → **clean-data-baseline-retraining**, **S2 baseline audit**, **S3** output and **S4a** output.
2. Settings → Accelerator **GPU T4 ×2** · Internet **ON**.
3. **Save Version → Save & Run All (Commit)**, then stop the interactive session.

**Time:** setup ≈ 10 min + smoke test ≈ 10 min + ≈ 3.5 h of the two GPU queues + analysis ≈ 10 min → **≈ 4 h of GPU quota**
(hard limit `BUDGET_H = 9.0` h). **Send back:** `S5/S5_report.txt` + `S5/grid_*.png` (or `S5_results.zip`).
"""

CELL1 = r'''# ===================== CELL 1 — CONFIG =====================
BUDGET_H = 9.0            # hard limit for this session; the run needs ~4 h
POOL_PER_CLASS = 300
E3_PER_CLASS = 100        # same size as the S3 / S4 evaluation sets (their epoch-42 / 44 samples are reused)
EPS_SCALES = [1.005, 1.01, 1.02]
RUN_SMOKE = True
import os, time
os.environ["PA_OUT"] = "/kaggle/working/S5"
os.makedirs(os.environ["PA_OUT"], exist_ok=True)
if not os.path.exists("/tmp/pg_session_start"):
    open("/tmp/pg_session_start", "w").write(str(time.time()))
def used_h():
    return (time.time() - float(open("/tmp/pg_session_start").read())) / 3600
print("S5 physics scoring | budget", BUDGET_H, "h | output", os.environ["PA_OUT"])
'''

CELL3 = r'''# ===================== CELL 3 — code + clean data (reused) + input checks + job runner =====================
import os, sys, glob, json, shutil, subprocess, threading, time
from pathlib import Path
WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
os.chdir(WD)
FILES = {"phaseA_common.py": r''' + "'''" + rd("src2/phaseA_common.py") + "'''" + r''',
         "fp_core.py": r''' + "'''" + rd("src2/fp_core.py") + "'''" + r''',
         "a6_classifier.py": r''' + "'''" + rd("src2/a6_classifier.py") + "'''" + r''',
         "pinn_physics.py": r''' + "'''" + rd("ref/pinn_physics.py") + "'''" + r''',
         "s5_generate.py": r''' + "'''" + rd("src5/s5_generate.py") + "'''" + r''',
         "s5_score.py": r''' + "'''" + rd("src5/s5_score.py") + "'''" + r''',
         "s5_stats.py": r''' + "'''" + rd("src5/s5_stats.py") + "'''" + r''',
         "s5_analyze.py": r''' + "'''" + rd("src5/s5_analyze.py") + "'''" + r'''}
for n, s in FILES.items():
    Path(n).write_text(s)
Path("/tmp/prepare_clean.py").write_text(r''' + "'''" + rd("src/prepare_clean.py") + "'''" + r''')
print("wrote", ", ".join(FILES))
subprocess.run([sys.executable, "/tmp/prepare_clean.py"], check=True)      # reuses clean_data/ from the input (identical bytes)

def find_in(name):
    h = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    print(f"{name}: {h[0] if h else '-- not found'}")
    return h[0] if h else None
def need(name, hint):
    p = find_in(name)
    if p is None:
        raise SystemExit(f"!! {name} not found -> {hint}")
    return p
OUT = os.environ["PA_OUT"]
BASE40 = need("base_clean_s42_w_ep040.pt", "Add Input -> clean-data-baseline-retraining")
need("A8_fp_clean.json", "Add Input -> S2 baseline audit"); need("a6_clf_s0.pt", "Add Input -> S2 baseline audit")
CK41 = need("s3_m0_w_ep041.pt", "Add Input -> the S3 notebook output")
CK42 = need("s3_m0_w_ep042.pt", "Add Input -> the S3 notebook output")
CK44 = need("s3_m0_w_ep044.pt", "Add Input -> the S4a notebook output")
CK35 = find_in("base_clean_s42_w_ep035.pt")
REUSE = {}
for ep, fn in ((42, "gen_s3_m0_ep042_gs2.npz"), (44, "gen_s3_m0_ep044_gs2.npz")):
    p = find_in(fn)
    if p:
        shutil.copy(p, f"{OUT}/gen_e3_ep{ep:03d}.npz"); REUSE[ep] = p
print("reused epoch-42/44 sample sets (same sampler and noise bank):", list(REUSE))
r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True)
NGPU = int(r.stdout.split()[-1]) if r.returncode == 0 else 0
print("GPUs visible:", NGPU, "" if NGPU >= 2 else "!! expected 2 (Settings -> Accelerator -> GPU T4 x2)")
LOGS = f"{OUT}/logs"; os.makedirs(LOGS, exist_ok=True)
RC = {}

def run_one(name, gpu, argv, est_h, out_dir=None):
    """one job in a subprocess; skipped (rc = 'skipped') if it would pass BUDGET_H (0.4 h kept for the analysis)"""
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

gen = lambda name, ckpts, n, extra=(): ["s5_generate.py", "--name", name, "--ckpts", ",".join(ckpts), "--n_per_class", str(n)] + list(extra)
print("Cell 3 done.")
'''

CELL4 = r'''# ===================== CELL 4 — SMOKE TEST (tiny versions of every script, ~10 min) =====================
SM = "/kaggle/working/smoke"
done_before = os.path.exists(f"{OUT}/S5_report.txt")
if RUN_SMOKE and not done_before:
    os.makedirs(SM, exist_ok=True)
    run_queues({0: [("smoke_gen_pool", gen("pool_ep040", [BASE40], 2, ["--steps", "5"]), 0.1, SM),
                    ("smoke_score_pool", ["s5_score.py", "--set", "pool", "--gen", f"{SM}/gen_pool_ep040.npz", "--ckpt", BASE40, "--limit", "4"], 0.1, SM),
                    ("smoke_score_retest", ["s5_score.py", "--set", "retest", "--gen", f"{SM}/gen_pool_ep040.npz", "--ckpt", BASE40,
                                            "--n_retest", "2"], 0.1, SM)],
                1: [("smoke_gen_avg", gen("e3_avg", [BASE40, CK41], 2, ["--steps", "5", "--eps_scale", "1.01"]), 0.1, SM),
                    ("smoke_score_real", ["s5_score.py", "--set", "real", "--ckpt", BASE40, "--limit", "4"], 0.1, SM)]})
    if all(RC.get(k) == 0 for k in ("smoke_gen_pool", "smoke_score_pool", "smoke_score_retest", "smoke_gen_avg", "smoke_score_real")):
        run_one("smoke_analyze", 0, ["s5_analyze.py", "--n_random", "50", "--n_boot", "50"], 0.1, SM)
    ok = RC.get("smoke_analyze") == 0 and os.path.exists(f"{SM}/S5_report.txt") and os.path.exists(f"{SM}/S5_data.npz")
    if not ok:
        raise SystemExit("SMOKE TEST FAILED -> nothing long was started. Send Claude this block + S5/logs/smoke_*.log")
    shutil.rmtree(SM, ignore_errors=True)
    print(f"SMOKE TEST PASSED | session {used_h():.2f} h")
else:
    print("smoke test skipped")
'''

CELL5 = r'''# ===================== CELL 5 — RUN: two GPU queues in parallel =====================
t0 = time.time()
POOL = f"{OUT}/gen_pool_ep040.npz"
q0 = [("gen_pool", gen("pool_ep040", [BASE40], POOL_PER_CLASS), 1.4),
      ("score_pool", ["s5_score.py", "--set", "pool", "--gen", POOL, "--ckpt", BASE40], 1.3),
      (f"gen_e4_s{EPS_SCALES[2]:g}", gen(f"e4_s{EPS_SCALES[2]:g}", [BASE40], E3_PER_CLASS, ["--eps_scale", str(EPS_SCALES[2])]), 0.45),
      ("score_retest", ["s5_score.py", "--set", "retest", "--gen", POOL, "--ckpt", BASE40, "--n_retest", "80"], 0.2)]
q1 = [("gen_e3_ep040", gen("e3_ep040", [BASE40], E3_PER_CLASS), 0.45),
      ("gen_e3_ep041", gen("e3_ep041", [CK41], E3_PER_CLASS), 0.45),
      ("gen_e3_avg", gen("e3_avg", [BASE40, CK41, CK42, CK44], E3_PER_CLASS), 0.45)]
if CK35:
    q1.append(("gen_e3_ep035", gen("e3_ep035", [CK35], E3_PER_CLASS), 0.45))
for ep, ck in ((42, CK42), (44, CK44)):
    if ep not in REUSE:
        q1.append((f"gen_e3_ep{ep:03d}", gen(f"e3_ep{ep:03d}", [ck], E3_PER_CLASS), 0.45))
q1 += [(f"gen_e4_s{s:g}", gen(f"e4_s{s:g}", [BASE40], E3_PER_CLASS, ["--eps_scale", str(s)]), 0.45) for s in EPS_SCALES[:2]]
q1.append(("score_real", ["s5_score.py", "--set", "real", "--ckpt", BASE40], 0.8))
run_queues({0: q0, 1: q1})
print(f"===== queues done in {(time.time() - t0) / 3600:.2f} h | return codes {RC} | session {used_h():.2f} h =====")
'''

CELL6 = r'''# ===================== CELL 6 — ANALYSIS + REPORT + zip =====================
run_one("analyze", 0, ["s5_analyze.py"], 0.3)
rp = f"{OUT}/S5_report.txt"
print(open(rp).read() if os.path.exists(rp) else "!! no report - see S5/logs/analyze.log")
json.dump(RC, open(f"{OUT}/return_codes.json", "w"), indent=1)
z = shutil.make_archive("/kaggle/working/S5_results", "zip", OUT)
print("return codes:", RC, "| ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB) | session {used_h():.2f} h")
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
OUTP = "/mnt/user-data/outputs/pinn_S5_physics_scoring_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1)
print("wrote", OUTP)
