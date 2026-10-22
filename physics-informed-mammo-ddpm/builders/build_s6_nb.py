# build_s6_nb.py - assembles pinn_S6_final_kaggle.ipynb (S6: validation screening -> frozen winner -> one test confirmation
# -> downstream classifier). No diffusion training.
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

pa = json.load(open("/mnt/user-data/outputs/pinn_phaseA_kaggle.ipynb"))
SETUP = "".join(pa["cells"][2]["source"])
assert "CELL 2" in SETUP

MD = r"""# S6 (final) — weight averaging, dynamic thresholding, denoising-confidence selection → one test confirmation
No diffusion training. Pre-stated rule in `s6_stats.py` (fixed before the run).

**Stage 1 — validation screening** (reference = train-fit images, 100 / class, same noise bank, paired):

| arm | model | sampler |
|---|---|---|
| `v_base` | epoch 40 | standard (reused from S5) |
| `v_avgA` | weight average of epochs 40/41/42/44 | standard (reused from S5) |
| `v_avgB` | weight average of epochs 35/40/41/42/44 | standard |
| `v_dt` | epoch 40 | dynamic thresholding (99.5th percentile, Imagen) |
| `v_avgA_dt` | average 40–44 | dynamic thresholding |
| `v_bo2` | epoch 40 | best of 2 candidates by denoising error (F3) |
| `v_avgA_dt_bo2` | average 40–44 | dynamic thresholding + best of 2 |

PASS: KID-mammo ≤ 0.90× baseline with paired CI < 0, KID-DINOv2 ≤ 1.00× baseline, recall ≥ baseline − 0.05.
Winner = passing arm with the lowest KID-mammo → **frozen automatically**.

**Stage 2 — ONE test confirmation:** fresh noise seeds, 150 / class, winner vs baseline, reference = the untouched TEST images, same rule.
**Stage 3 — downstream (secondary):** ConvNeXt classifier on real train-fit ± 300 synthetic images, TEST AUC, 3 seeds.

**How to run:** Add Input → clean-data-baseline-retraining, S2, S3, S4a and **S5** outputs · GPU T4 ×2 · Internet ON ·
Save & Run All (Commit) · stop the session. **≈ 4.5 h.** Send back `S6_results.zip`.
"""

CELL1 = r'''# ===================== CELL 1 — CONFIG =====================
BUDGET_H = 9.0
VAL_PER_CLASS, TEST_PER_CLASS = 100, 150
DT = 99.5                   # dynamic-thresholding percentile (Imagen default)
TEST_SEED = 9025            # fresh noise seeds for the test stage (never used before)
RUN_SMOKE = True
import os, time
os.environ["PA_OUT"] = "/kaggle/working/S6"
os.makedirs(os.environ["PA_OUT"], exist_ok=True)
if not os.path.exists("/tmp/pg_session_start"):
    open("/tmp/pg_session_start", "w").write(str(time.time()))
def used_h():
    return (time.time() - float(open("/tmp/pg_session_start").read())) / 3600
print("S6 final | budget", BUDGET_H, "h | output", os.environ["PA_OUT"])
'''

CELL3 = r'''# ===================== CELL 3 — code + clean data + inputs + job runner =====================
import os, sys, glob, json, shutil, subprocess, threading, time
from pathlib import Path
WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
os.chdir(WD)
FILES = {"phaseA_common.py": r''' + "'''" + rd("src2/phaseA_common.py") + "'''" + r''',
         "a6_classifier.py": r''' + "'''" + rd("src2/a6_classifier.py") + "'''" + r''',
         "s6_generate.py": r''' + "'''" + rd("src6/s6_generate.py") + "'''" + r''',
         "s6_f3.py": r''' + "'''" + rd("src6/s6_f3.py") + "'''" + r''',
         "s6_stats.py": r''' + "'''" + rd("src6/s6_stats.py") + "'''" + r''',
         "s6_eval.py": r''' + "'''" + rd("src6/s6_eval.py") + "'''" + r''',
         "s6_downstream.py": r''' + "'''" + rd("src6/s6_downstream.py") + "'''" + r'''}
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
        raise SystemExit(f"!! {name} not found -> {hint}")
    return p
OUT = os.environ["PA_OUT"]
CK = {40: need("base_clean_s42_w_ep040.pt", "clean-data-baseline-retraining"), 35: need("base_clean_s42_w_ep035.pt", "clean-data-baseline-retraining"),
      41: need("s3_m0_w_ep041.pt", "S3 output"), 42: need("s3_m0_w_ep042.pt", "S3 output"), 44: need("s3_m0_w_ep044.pt", "S4a output")}
need("a6_clf_s0.pt", "S2 output")
AVG_A = [CK[e] for e in (40, 41, 42, 44)]; AVG_B = [CK[e] for e in (35, 40, 41, 42, 44)]
REUSED = []
for src, dst in (("gen_e3_ep040.npz", "v_base"), ("gen_e3_avg.npz", "v_avgA")):
    p = find_in(src)
    if p:
        shutil.copy(p, f"{OUT}/gen_{dst}.npz"); REUSED.append(dst)
print("reused from S5 (same sampler, same noise bank):", REUSED)
# arm -> how to generate it (model checkpoints, dynamic threshold, best-of-N)
ARMS = {"v_base": (["40"], 0, 1), "v_avgA": (["40", "41", "42", "44"], 0, 1), "v_avgB": (["35", "40", "41", "42", "44"], 0, 1),
        "v_dt": (["40"], DT, 1), "v_avgA_dt": (["40", "41", "42", "44"], DT, 1), "v_bo2": (["40"], 0, 2),
        "v_avgA_dt_bo2": (["40", "41", "42", "44"], DT, 2)}
r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True)
NGPU = int(r.stdout.split()[-1]) if r.returncode == 0 else 0
print("GPUs visible:", NGPU)
LOGS = f"{OUT}/logs"; os.makedirs(LOGS, exist_ok=True)
RC = {}
def run_one(name, gpu, argv, est_h, out_dir=None):
    if used_h() + est_h > BUDGET_H - 0.3:
        print(f"--- {name} SKIPPED (budget)", flush=True); RC[name] = "skipped"; return
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"; env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu if NGPU >= 2 else 0)
    if out_dir:
        env["PA_OUT"] = out_dir
    print(f">>> {name} started on GPU {env['CUDA_VISIBLE_DEVICES']} | session {used_h():.2f} h", flush=True)
    with open(f"{LOGS}/{name}.log", "w") as fh:
        p = subprocess.Popen([sys.executable, "-u"] + argv, cwd=WD, env=env, text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            fh.write(line); fh.flush(); print(line if line.startswith("[") else f"[{name}] {line}", end="", flush=True)
        RC[name] = p.wait()
    print(f"<<< {name} return code {RC[name]} | session {used_h():.2f} h" + ("" if RC[name] == 0 else f"  !! see {LOGS}/{name}.log"), flush=True)
def run_queues(queues):
    if NGPU < 2:
        queues = {0: [j for q in queues.values() for j in q]}
    ths = [threading.Thread(target=lambda g=g, q=q: [run_one(n, g, a, e, *rest) for n, a, e, *rest in q]) for g, q in queues.items()]
    for t in ths: t.start()
    for t in ths: t.join()
def gen(name, eps, dt, n, seed, extra=()):
    return ["s6_generate.py", "--name", name, "--ckpts", ",".join(CK[int(e)] for e in eps), "--n_per_class", str(n),
            "--dt", str(dt), "--noise_seed", str(seed)] + list(extra)
def jobs_for(arm, name, n, seed, extra=()):
    """generation jobs for one arm (best-of-2 -> two candidate sets) + the selection job"""
    eps, dt, bo = ARMS[arm]
    g = [(f"gen_{name}", gen(name, eps, dt, n, seed, extra), 0.45 * n / 100)]
    if bo == 2:
        g.append((f"gen_{name}_c2", gen(f"{name}_c2", eps, dt, n, seed + 1000, extra), 0.45 * n / 100))
    return g
print("Cell 3 done.")
'''

CELL4 = r'''# ===================== CELL 4 — SMOKE TEST (~10 min) =====================
SM = "/kaggle/working/smoke"
if RUN_SMOKE and not os.path.exists(f"{OUT}/S6_decision.json"):
    os.makedirs(SM, exist_ok=True)
    fast = ["--steps", "5"]
    run_queues({0: [("smoke_gen_a", gen("v_base", ["40"], 0, 2, 2025, fast), 0.1, SM), ("smoke_gen_b", gen("v_x", ["40", "41"], DT, 2, 3025, fast), 0.1, SM),
                    ("smoke_f3", ["s6_f3.py", "--ckpt", CK[40], "--sets", "v_base,v_x", "--select_out", "v_sel"], 0.1, SM),
                    ("smoke_eval", ["s6_eval.py", "--stage", "val", "--sets", "v_base,v_x,v_sel", "--n_boot", "20"], 0.1, SM)],
                1: [("smoke_down", ["s6_downstream.py", "--arm", "smoke", "--seeds", "0", "--smoke"], 0.1, SM)]})
    ok = all(RC.get(k) == 0 for k in ("smoke_gen_a", "smoke_gen_b", "smoke_f3", "smoke_eval", "smoke_down")) and os.path.exists(f"{SM}/S6_decision.json")
    if not ok:
        raise SystemExit("SMOKE TEST FAILED -> nothing long was started. Send Claude this block + S6/logs/smoke_*.log")
    shutil.rmtree(SM, ignore_errors=True); RC.clear(); print(f"SMOKE TEST PASSED | session {used_h():.2f} h")
'''

CELL5 = r'''# ===================== CELL 5 — STAGE 1: validation screening (+ real-only downstream meanwhile) =====================
need_gen = []
for arm, (eps, dt, bo) in ARMS.items():
    if arm in REUSED:
        continue
    if bo == 2:
        base_arm = "v_base" if eps == ["40"] else "v_avgA_dt"      # candidate 1 = the matching single-candidate arm
        need_gen.append((f"gen_{arm}_c2", gen(f"{arm}_c2", eps, dt, VAL_PER_CLASS, 3025), 0.45))
    else:
        need_gen.append((f"gen_{arm}", gen(arm, eps, dt, VAL_PER_CLASS, 2025), 0.45))
q0, q1 = need_gen[0::2], need_gen[1::2]
q1.append(("down_real_only", ["s6_downstream.py", "--arm", "real_only"], 0.6))
run_queues({0: q0, 1: q1})
for arm, c1 in (("v_bo2", "v_base"), ("v_avgA_dt_bo2", "v_avgA_dt")):
    run_one(f"select_{arm}", 0, ["s6_f3.py", "--ckpt", CK[40], "--sets", f"{c1},{arm}_c2", "--select_out", arm], 0.1)
run_one("eval_val", 0, ["s6_eval.py", "--stage", "val", "--sets", ",".join(ARMS)], 0.3)
DEC = json.load(open(f"{OUT}/S6_decision.json")) if os.path.exists(f"{OUT}/S6_decision.json") else {"winner": None}
print(open(f"{OUT}/S6_val_report.txt").read() if os.path.exists(f"{OUT}/S6_val_report.txt") else "!! no validation report")
print("FROZEN WINNER:", DEC["winner"])
'''

CELL6 = r'''# ===================== CELL 6 — STAGE 2: ONE test confirmation + STAGE 3: downstream =====================
W = DEC["winner"]
tb = jobs_for("v_base", "t_base", TEST_PER_CLASS, TEST_SEED)
tw = jobs_for(W, "t_win", TEST_PER_CLASS, TEST_SEED) if W else []
run_queues({0: tb + tw[1:], 1: tw[:1]})
if W and ARMS[W][2] == 2:
    os.replace(f"{OUT}/gen_t_win.npz", f"{OUT}/gen_t_win_c1.npz")
    run_one("select_t_win", 0, ["s6_f3.py", "--ckpt", CK[40], "--sets", "t_win_c1,t_win_c2", "--select_out", "t_win"], 0.1)
if W:
    run_one("eval_test", 0, ["s6_eval.py", "--stage", "test", "--sets", "t_base,t_win"], 0.3)
    print(open(f"{OUT}/S6_test_report.txt").read() if os.path.exists(f"{OUT}/S6_test_report.txt") else "!! no test report")
else:
    print("no validation winner -> no test confirmation (pre-stated)")
run_queues({0: [("down_real_base", ["s6_downstream.py", "--arm", "real_plus_base", "--synthetic", "t_base"], 0.6)],
            1: [("down_real_win", ["s6_downstream.py", "--arm", "real_plus_winner", "--synthetic", "t_win"], 0.6)] if W else []})
sys.path.insert(0, WD)
import numpy as np
from s6_stats import downstream_summary
D = {a: dict(np.load(f"{OUT}/down_{a}.npz")) for a in ("real_only", "real_plus_base", "real_plus_winner") if os.path.exists(f"{OUT}/down_{a}.npz")}
lines = ["[DOWNSTREAM] (secondary) ConvNeXt classifier, TEST AUC"] + (downstream_summary(D) if "real_only" in D else ["  missing"])
rep = "\n\n".join(open(f"{OUT}/{f}").read() for f in ("S6_val_report.txt", "S6_test_report.txt") if os.path.exists(f"{OUT}/{f}")) + "\n\n" + "\n".join(lines)
open(f"{OUT}/S6_report.txt", "w").write(rep); print(rep)
json.dump(RC, open(f"{OUT}/return_codes.json", "w"), indent=1)
z = shutil.make_archive("/kaggle/working/S6_results", "zip", OUT)
print("ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB) | session {used_h():.2f} h | return codes {RC}")
'''

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.strip("\n").splitlines(keepends=True)},
                code(CELL1), code(SETUP), code(CELL3), code(CELL4), code(CELL5), code(CELL6)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
OUTP = "/mnt/user-data/outputs/pinn_S6_final_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1); print("wrote", OUTP)
