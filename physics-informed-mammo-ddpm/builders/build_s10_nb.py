# build_s10_nb.py - assembles pinn_S10_ema_physics_kaggle.ipynb (S10: EMA baseline vs EMA + physics, CMMD, 2 seeds).
# Multi-session like S8: the same notebook is re-run (attaching its own latest output) until "ALL S10 JOBS DONE".
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

pa = json.load(open("/mnt/user-data/outputs/pinn_phaseA_kaggle.ipynb"))
SETUP = "".join(pa["cells"][2]["source"])
assert "CELL 2" in SETUP

MD = r"""# S10 — does physics help BEYOND an improved (EMA, 2-seed) baseline?
Rules: `S10_preregistration.txt` (fixed before any S10 code ran). CMMD, trained from scratch, 50 epochs, EMA, seeds 42 + 43:
**E0** = baseline (λ 0) · **E1** = baseline + noise-physics loss on x̂₀ (λ 0.01).
**K1** EMA vs raw · **K2 (primary)** E1-EMA vs E0-EMA · **K3** E1-raw vs E0-raw · D2 epoch 25 vs 50 (descriptive).

**Session 1 input:** the **S8 replication** notebook output (latest version; provides `cmmd_data/` and `cmmd_clf_s0.pt`).
**Session 2 (and 3):** the same input **plus this notebook's own latest output**. GPU T4 ×2 · Internet ON ·
Save & Run All (Commit). ≈ 11 h + 11 h + ~2 h. Send `S10_results.zip` (or the log) after each session.
"""

CELL1 = r'''# ===================== CELL 1 — CONFIG =====================
BUDGET_H = 11.0
EPOCHS = 50
VAL_N, TEST_N = 100, 150
VAL_SEED, TEST_SEED = 2025, 9025
TEST_ALPHA = 0.05 / 3              # 98.33 % CI = Bonferroni over K1-K3
RUN_SMOKE = True
import os, time
os.environ["PA_OUT"] = "/kaggle/working/S10"
os.makedirs(os.environ["PA_OUT"], exist_ok=True)
if not os.path.exists("/tmp/pg_session_start"):
    open("/tmp/pg_session_start", "w").write(str(time.time()))
def used_h():
    return (time.time() - float(open("/tmp/pg_session_start").read())) / 3600
print("S10 | budget", BUDGET_H, "h | output", os.environ["PA_OUT"])
'''

CELL3 = r'''# ===================== CELL 3 — code + CMMD cache (reused from S8) + carry-forward + runner =====================
import os, sys, glob, json, shutil, subprocess, threading, time
from pathlib import Path
WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
os.chdir(WD)
FILES = {"phaseA_common.py": r''' + "'''" + rd("src8/phaseA_common.py") + "'''" + r''',
         "pinn_physics.py": r''' + "'''" + rd("ref/pinn_physics.py") + "'''" + r''',
         "a6_classifier.py": r''' + "'''" + rd("src2/a6_classifier.py") + "'''" + r''',
         "train_ema.py": r''' + "'''" + rd("src10/train_ema.py") + "'''" + r''',
         "s6_generate.py": r''' + "'''" + rd("src6/s6_generate.py") + "'''" + r''',
         "s7_stats.py": r''' + "'''" + rd("src7/s7_stats.py") + "'''" + r''',
         "s10_eval.py": r''' + "'''" + rd("src10/s10_eval.py") + "'''" + r''',
         "s10_pool.py": r''' + "'''" + rd("src10/s10_pool.py") + "'''" + r'''}
for n, s in FILES.items():
    Path(n).write_text(s)
Path("/tmp/prepare_cmmd.py").write_text(r''' + "'''" + rd("src8/prepare_cmmd.py") + "'''" + r''')
OUT = os.environ["PA_OUT"]; CK = "/kaggle/working/ckpt"; os.makedirs(CK, exist_ok=True)
ENV = {"PA_TMP": "/tmp/cmmd", "PA_DATA": "/kaggle/working/cmmd_data"}
def walk_find(name, root="/kaggle/input"):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in ("CMMD", "jpeg", "U24_synthesis", ".git")]
        if name in fns:
            out.append(os.path.join(dp, name))
    return sorted(out)
h = walk_find("cmmd_clf_s0.pt")
if not h:
    raise SystemExit("!! cmmd_clf_s0.pt not found -> Add Input -> the S8 replication notebook output (latest version)")
CLF = h[0]; print("classifier:", CLF)
SESSION, prev = 1, None
for f in walk_find("S10_state.json"):
    s = json.load(open(f))
    if s.get("session", 0) + 1 > SESSION:
        SESSION, prev = s["session"] + 1, os.path.dirname(os.path.dirname(f))
if prev:
    n = 0
    for sub, dst in (("S10", OUT), ("ckpt", CK)):
        for f in glob.glob(f"{prev}/{sub}/*"):
            t = os.path.join(dst, os.path.basename(f))
            if os.path.isfile(f) and not os.path.exists(t):
                shutil.copy2(f, t); n += 1
    print(f"SESSION {SESSION}: carried forward {n} files from {prev}")
else:
    print("SESSION 1")
subprocess.run([sys.executable, "/tmp/prepare_cmmd.py"], check=True, env={**os.environ, **ENV})   # reuses S8's cmmd_data
r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True)
NGPU = int(r.stdout.split()[-1]) if r.returncode == 0 else 0
print("GPUs visible:", NGPU)
LOGS = f"{OUT}/logs"; os.makedirs(LOGS, exist_ok=True); RC = {}
def run_one(name, gpu, argv, out_dir=None):
    env = os.environ.copy(); env.update(ENV); env["PYTHONUNBUFFERED"] = "1"; env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu if NGPU >= 2 else 0)
    if out_dir:
        env["PA_OUT"] = out_dir
    print(f">>> {name} started on GPU {env['CUDA_VISIBLE_DEVICES']} | session {used_h():.2f} h", flush=True)
    with open(f"{LOGS}/{name}.log", "a") as fh:
        p = subprocess.Popen([sys.executable, "-u"] + argv, cwd=WD, env=env, text=True, bufsize=1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in p.stdout:
            fh.write(line); fh.flush(); print(line if line.startswith("[") else f"[{name}] {line}", end="", flush=True)
        RC[name] = p.wait()
    print(f"<<< {name} return code {RC[name]} | session {used_h():.2f} h" + ("" if RC[name] == 0 else f"  !! see {LOGS}/{name}.log"), flush=True)
    return RC[name]
print("Cell 3 done.")
'''

CELL4 = r'''# ===================== CELL 4 — SMOKE TEST (first session only, ~15 min) =====================
SM = "/kaggle/working/smoke"; CKS = f"{SM}/ckpt"
if RUN_SMOKE and SESSION == 1 and not glob.glob(f"{CK}/E*_last.pt"):
    shutil.rmtree(SM, ignore_errors=True); os.makedirs(CKS, exist_ok=True)
    steps = [("smoke_train_e1", ["train_ema.py", "--arm", "smk", "--lam", "0.01", "--epochs", "1", "--smoke_batches", "16", "--snap_every", "1000",
                                 "--snap_epochs", "1", "--log_every", "8", "--ckpt_dir", CKS]),
             ("smoke_resume", ["train_ema.py", "--arm", "smk", "--lam", "0.01", "--epochs", "2", "--smoke_batches", "16", "--snap_every", "1000",
                               "--snap_epochs", "2", "--log_every", "8", "--ckpt_dir", CKS]),
             ("smoke_gen_raw", ["s6_generate.py", "--name", "a1", "--ckpts", f"{CKS}/smk_w_ep002.pt", "--n_per_class", "2", "--steps", "5"]),
             ("smoke_gen_ema", ["s6_generate.py", "--name", "b1", "--ckpts", f"{CKS}/smk_ema_ep002.pt", "--n_per_class", "2", "--steps", "5"]),
             ("smoke_pool_a", ["s10_pool.py", "a", "a1", "a1"]), ("smoke_pool_b", ["s10_pool.py", "b", "b1", "b1"]),
             ("smoke_eval", ["s10_eval.py", "--stage", "val", "--sets", "a,b", "--clf", CLF, "--tag", "smk", "--n_boot", "20"])]
    for name, argv in steps:
        if run_one(name, 0, argv, SM) != 0:
            raise SystemExit(f"SMOKE TEST FAILED at {name} -> send Claude this block + S10/logs/{name}.log")
    log2 = open(f"{LOGS}/smoke_resume.log").read()
    if "EMA restored from checkpoint" not in log2 or not os.path.exists(f"{SM}/smk_val_report.txt"):
        raise SystemExit("SMOKE TEST FAILED: EMA was not restored on resume, or no report")
    print(open(f"{SM}/smk_val_report.txt").read())
    shutil.rmtree(SM, ignore_errors=True); RC.clear(); print(f"SMOKE TEST PASSED | session {used_h():.2f} h")
else:
    print("smoke test skipped (session", SESSION, ")")
'''

CELL5 = r'''# ===================== CELL 5 — JOB LIST (fixed by the pre-registration) + two-GPU scheduler =====================
def ck(n): return f"{CK}/{n}"
TB = str(BUDGET_H - 0.3)
def gen(name, cks, n, seed):
    return ["s6_generate.py", "--name", name, "--ckpts", ",".join(cks), "--n_per_class", str(n), "--dt", "0", "--noise_seed", str(seed)]
RUNS = [("E0_s42", "0.0", 42), ("E1_s42", "0.01", 42), ("E0_s43", "0.0", 43), ("E1_s43", "0.01", 43)]
JOBS = []
for arm, lam, seed in RUNS:
    se = "25,50" if arm == "E0_s42" else "50"
    JOBS.append((f"train_{arm}", ["train_ema.py", "--arm", arm, "--lam", lam, "--seed", str(seed), "--epochs", str(EPOCHS), "--snap_every", "1000",
                                  "--snap_epochs", se, "--budget_h", TB, "--ckpt_dir", CK], [], ck(f"{arm}_ema_ep050.pt"), 8.5, 0.25))
G = lambda *n: [f"{OUT}/gen_{x}.npz" for x in n]
for pre, n, seed in (("v", VAL_N, VAL_SEED), ("t", TEST_N, TEST_SEED)):
    for arm, _, _ in RUNS:
        for kind, f in (("raw", f"{arm}_w_ep050.pt"), ("ema", f"{arm}_ema_ep050.pt")):
            needs = [ck(f)] + ([f"{OUT}/S10_K2_val.json"] if pre == "t" else [])
            JOBS.append((f"gen_{pre}_{arm}_{kind}", gen(f"{pre}_{arm}_{kind}", [ck(f)], n, seed), needs, f"{OUT}/gen_{pre}_{arm}_{kind}.npz", 0.45 if pre == "v" else 0.65, None))
    if pre == "v":
        for kind, f in (("raw25", "E0_s42_w_ep025.pt"), ("ema25", "E0_s42_ema_ep025.pt")):
            JOBS.append((f"gen_v_E0_s42_{kind}", gen(f"v_E0_s42_{kind}", [ck(f)], n, seed), [ck(f)], f"{OUT}/gen_v_E0_s42_{kind}.npz", 0.45, None))
    for e in ("E0", "E1"):
        for kind in ("raw", "ema"):
            JOBS.append((f"pool_{pre}_{e}_{kind}", ["s10_pool.py", f"{pre}_{e}_{kind}", f"{pre}_{e}_s42_{kind}", f"{pre}_{e}_s43_{kind}"],
                         G(f"{pre}_{e}_s42_{kind}", f"{pre}_{e}_s43_{kind}"), f"{OUT}/gen_{pre}_{e}_{kind}.npz", 0.02, None))
CMP = [("K1", "E0_raw,E0_ema"), ("K2", "E0_ema,E1_ema"), ("K3", "E0_raw,E1_raw")]
for k, sets in CMP:
    v = ",".join(f"v_{s}" for s in sets.split(",")); t = ",".join(f"t_{s}" for s in sets.split(","))
    JOBS.append((f"eval_{k}_val", ["s10_eval.py", "--stage", "val", "--sets", v, "--clf", CLF, "--tag", f"S10_{k}"], G(*v.split(",")),
                 f"{OUT}/S10_{k}_val.json", 0.4, None))
    JOBS.append((f"eval_{k}_test", ["s10_eval.py", "--stage", "test", "--sets", t, "--clf", CLF, "--tag", f"S10_{k}", "--alpha", str(TEST_ALPHA)],
                 G(*t.split(",")) + [f"{OUT}/S10_{k}_val.json"], f"{OUT}/S10_{k}_test.json", 0.45, None))
d2 = "v_E0_s42_raw25,v_E0_s42_ema25,v_E0_s42_raw,v_E0_s42_ema"
JOBS.append(("eval_D2_val", ["s10_eval.py", "--stage", "val", "--sets", d2, "--clf", CLF, "--tag", "S10_D2", "--halves", "0"],
             G(*d2.split(",")), f"{OUT}/S10_D2_val.json", 0.4, None))
NAMES = [j[0] for j in JOBS]; assert len(set(NAMES)) == len(NAMES)
print(f"{len(JOBS)} jobs | already done: {sum(os.path.exists(j[3]) for j in JOBS)}")
lock = threading.Lock(); RUNNING, TRIED, FAILED = set(), set(), set()
def pick():
    left = BUDGET_H - used_h()
    for name, argv, needs, done, est, rmin in JOBS:
        if name in RUNNING or name in TRIED or os.path.exists(done) or not all(os.path.exists(f) for f in needs):
            continue
        if (rmin is not None and left >= rmin + 0.3) or (rmin is None and left >= est + 0.25):
            return name, argv, done
    return None
def worker(gpu):
    while True:
        with lock:
            j = pick()
            if j:
                RUNNING.add(j[0]); TRIED.add(j[0])
            busy = bool(RUNNING)
        if j is None:
            if not busy or BUDGET_H - used_h() < 0.35:
                return
            time.sleep(60); continue
        name, argv, done = j
        try:
            rc = run_one(name, gpu, argv)
        except Exception as e:
            print(f"!! {name} could not run: {e!r}", flush=True); rc = -1
        with lock:
            RUNNING.discard(name)
            if rc != 0:
                FAILED.add(name)
            elif not os.path.exists(done):
                print(f"... {name} stopped before finishing (time budget) -> continues next session", flush=True)
        time.sleep(5)
ths = [threading.Thread(target=worker, args=(g,)) for g in range(max(1, min(NGPU, 2)))]
for t in ths: t.start()
for t in ths: t.join()
print(f"scheduler finished | session {used_h():.2f} h | failed {sorted(FAILED)}")
'''

CELL6 = r'''# ===================== CELL 6 — cleanup + state + report + zip =====================
ex = os.path.exists
for f in glob.glob(f"{CK}/*_w_latest.pt"):
    os.remove(f)
for arm, _, _ in RUNS:
    if ex(ck(f"{arm}_ema_ep050.pt")) and ex(ck(f"{arm}_last.pt")):
        os.remove(ck(f"{arm}_last.pt"))                          # finished -> free 2.5 GB
TL = f"{OUT}/train_logs"; os.makedirs(TL, exist_ok=True)
for f in glob.glob(f"{CK}/*_history.json"):
    shutil.copy(f, TL)
todo = [j[0] for j in JOBS if not ex(j[3])]
json.dump({"session": SESSION, "n_jobs": len(JOBS), "todo": todo, "failed": sorted(FAILED), "return_codes": RC},
          open(f"{OUT}/S10_state.json", "w"), indent=1)
rep = ["S10 REPORT (rules: S10_preregistration.txt)", "=" * 90, f"session {SESSION} | jobs left {len(todo)} | failed {sorted(FAILED)}"]
for arm, _, _ in RUNS:
    f = f"{CK}/{arm}_history.json"
    if ex(f):
        h = json.load(open(f)); rep.append(f"  {arm}: {len(h)}/{EPOCHS} epochs | last mse {h[-1]['mse']:.5f} | {h[-1]['minutes']:.1f} min/epoch")
for k in ("K1", "K2", "K3", "D2"):
    for st in ("val", "test"):
        f = f"{OUT}/S10_{k}_{st}_report.txt"
        if ex(f):
            rep.append(open(f).read())
open(f"{OUT}/S10_report.txt", "w").write("\n".join(rep) + "\n"); print("\n".join(rep))
z = shutil.make_archive("/kaggle/working/S10_results", "zip", OUT)
print("ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB)")
print(subprocess.run(f"du -sh {CK} {OUT}", shell=True, capture_output=True, text=True).stdout)
print("=" * 90)
print("ALL S10 JOBS DONE -> send S10_results.zip (or this log) to Claude" if not todo else
      f"S10 NOT FINISHED ({len(todo)} jobs left) -> NEXT SESSION: same input + this notebook's latest output -> Save & Run All")
if FAILED:
    print("!! FAILED jobs:", sorted(FAILED))
'''

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.strip("\n").splitlines(keepends=True)},
                code(CELL1), code(SETUP), code(CELL3), code(CELL4), code(CELL5), code(CELL6)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
OUTP = "/mnt/user-data/outputs/pinn_S10_ema_physics_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1); print("wrote", OUTP)
