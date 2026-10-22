# build_s8_nb.py - assembles pinn_S8_replication_kaggle.ipynb (S8: CMMD replication + longer CBIS-DDSM training).
# Multi-session: the same notebook is run 2-3 times (each run attaches the previous run's output); a file-based job
# scheduler runs whatever is not done yet on both GPUs, in a fixed priority order, until the session budget.
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(path):
    s = open(os.path.join(HERE, path)).read()
    assert "'''" not in s, path
    return s

pa = json.load(open("/mnt/user-data/outputs/pinn_phaseA_kaggle.ipynb"))
SETUP = "".join(pa["cells"][2]["source"])
assert "CELL 2" in SETUP

MD = r"""# S8 — replication on a second dataset (CMMD, digital) + longer CBIS-DDSM training
Rules: `S8_preregistration.txt` (fixed before any CMMD model was trained). Comparisons:
**C1** FP vs control on CMMD · **C2** FP vs control on the CBIS model trained to epoch 90 · **C3** weight averaging on CMMD ·
**C4** acquisition noise layer on CMMD · **D1** CBIS epoch 40 vs 60 vs 90 (descriptive). Validation first, then ONE test stage for C1–C4.

**This notebook runs in 2 sessions (maybe a short 3rd).** Every session continues where the last one stopped.

**Session 1 inputs (Add Input):** `clean-data-baseline-retraining` · `S2` (baseline audit) · dataset **The Chinese Mammography
Database (CMMD) 2022** (tommyngx). GPU T4 ×2 · Internet ON · Save & Run All (Commit) · stop the interactive session.

**Session 2 (and 3):** same inputs **plus this notebook's own latest output** (Add Input → Your Work → S8, latest version) →
Save & Run All again. The last lines of the log say `ALL S8 JOBS DONE` or what is still to do.

Session 1 ≈ 11 h (two trainings in parallel), session 2 ≈ 6 h. Send `S8_results.zip` after each session.
"""

CELL1 = r'''# ===================== CELL 1 — CONFIG =====================
BUDGET_H = 11.0                 # per session (Kaggle hard limit 12 h; output saving needs time)
RUN_SMOKE = True                # ~20 min end-to-end test, only in the very first session
VAL_N, TEST_N = 100, 150        # images per class (validation / test stage)
VAL_SEED, TEST_SEED = 2025, 9025
NOISE_VAL_SEED, NOISE_TEST_SEED = 7007, 9707
TEST_ALPHA = 0.01               # 99 % CI = Bonferroni over the 5 test comparisons (C1-C5)
import os, time
os.environ["PA_OUT"] = "/kaggle/working/S8"
os.makedirs(os.environ["PA_OUT"], exist_ok=True)
if not os.path.exists("/tmp/pg_session_start"):
    open("/tmp/pg_session_start", "w").write(str(time.time()))
def used_h():
    return (time.time() - float(open("/tmp/pg_session_start").read())) / 3600
print("S8 replication | budget", BUDGET_H, "h | output", os.environ["PA_OUT"])
'''

CELL3 = r'''# ===================== CELL 3 — code + data (CBIS-DDSM reused, CMMD built once) + carry-forward + inputs =====================
import os, sys, glob, json, shutil, subprocess, threading, time
from pathlib import Path
WD = "/kaggle/working/U24_synthesis/lightning_synthesis"
os.chdir(WD)
FILES = {"phaseA_common.py": r''' + "'''" + rd("src8/phaseA_common.py") + "'''" + r''',
         "fp_core.py": r''' + "'''" + rd("src2/fp_core.py") + "'''" + r''',
         "pinn_physics.py": r''' + "'''" + rd("ref/pinn_physics.py") + "'''" + r''',
         "a6_classifier.py": r''' + "'''" + rd("src2/a6_classifier.py") + "'''" + r''',
         "train_arm2.py": r''' + "'''" + rd("src8/train_arm2.py") + "'''" + r''',
         "train_fp.py": r''' + "'''" + rd("src8/train_fp.py") + "'''" + r''',
         "s6_generate.py": r''' + "'''" + rd("src6/s6_generate.py") + "'''" + r''',
         "s7_noise.py": r''' + "'''" + rd("src7/s7_noise.py") + "'''" + r''',
         "s7_stats.py": r''' + "'''" + rd("src7/s7_stats.py") + "'''" + r''',
         "s8_sref.py": r''' + "'''" + rd("src8/s8_sref.py") + "'''" + r''',
         "s8_clf.py": r''' + "'''" + rd("src8/s8_clf.py") + "'''" + r''',
         "s8_eval.py": r''' + "'''" + rd("src8/s8_eval.py") + "'''" + r''',
         "s8_report.py": r''' + "'''" + rd("src8/s8_report.py") + "'''" + r'''}
for n, s in FILES.items():
    Path(n).write_text(s)
Path("/tmp/prepare_clean.py").write_text(r''' + "'''" + rd("src8/prepare_clean.py") + "'''" + r''')
Path("/tmp/prepare_cmmd.py").write_text(r''' + "'''" + rd("src8/prepare_cmmd.py") + "'''" + r''')
OUT = os.environ["PA_OUT"]; CK = "/kaggle/working/ckpt"; os.makedirs(CK, exist_ok=True)
ENV_CMMD = {"PA_TMP": "/tmp/cmmd", "PA_DATA": "/kaggle/working/cmmd_data"}
def walk_find(name, root="/kaggle/input"):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in ("CMMD", "jpeg", "U24_synthesis", ".git")]
        if name in fns:
            out.append(os.path.join(dp, name))
    return sorted(out)
def need(name, hint):
    h = walk_find(name)
    print(f"{name}: {h[0] if h else '!! NOT FOUND'}")
    if not h:
        raise SystemExit(f"!! {name} not found -> Add Input -> {hint}")
    return h[0]
B40 = need("base_clean_s42_w_ep040.pt", "clean-data-baseline-retraining (latest version)")
need("base_clean_s42_last.pt", "clean-data-baseline-retraining (latest version)")
CLF_B = need("a6_clf_s0.pt", "the S2 baseline-audit notebook output")
need("CMMD_clinicaldata_revision.xlsx", "dataset 'The Chinese Mammography Database (CMMD) 2022' (tommyngx)")
# ---- carry forward the previous S8 session (this notebook's own earlier output) ----
SESSION, prev = 1, None
for f in walk_find("S8_state.json"):
    s = json.load(open(f))
    if s.get("session", 0) + 1 > SESSION:
        SESSION, prev = s["session"] + 1, os.path.dirname(os.path.dirname(f))
if prev:
    n = 0
    for sub, dst in (("S8", OUT), ("ckpt", CK)):
        for f in glob.glob(f"{prev}/{sub}/*"):
            t = os.path.join(dst, os.path.basename(f))
            if os.path.isfile(f) and not os.path.exists(t):
                shutil.copy2(f, t); n += 1
    print(f"SESSION {SESSION}: carried forward {n} files from {prev}")
else:
    print("SESSION 1 (no earlier S8 output attached)")
subprocess.run([sys.executable, "/tmp/prepare_clean.py"], check=True)                                  # CBIS-DDSM, reused bytes
subprocess.run([sys.executable, "/tmp/prepare_cmmd.py"], check=True, env={**os.environ, **ENV_CMMD})   # CMMD (built once, ~15-25 min)
print(open("/kaggle/working/cmmd_data/cmmd_summary.json").read())
r = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"], capture_output=True, text=True)
NGPU = int(r.stdout.split()[-1]) if r.returncode == 0 else 0
print("GPUs visible:", NGPU, "" if NGPU >= 2 else "!! choose GPU T4 x2")
LOGS = f"{OUT}/logs"; os.makedirs(LOGS, exist_ok=True)
RC = {}
def run_one(name, gpu, argv, env_extra=None, out_dir=None):
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"; env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu if NGPU >= 2 else 0)
    env.update(env_extra or {})
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

CELL4 = r'''# ===================== CELL 4 — SMOKE TEST (first session only, ~20 min) =====================
SM = "/kaggle/working/smoke"; CKS = f"{SM}/ckpt"
if RUN_SMOKE and SESSION == 1 and not os.path.exists(f"{CK}/cmmd_s42_last.pt"):
    shutil.rmtree(SM, ignore_errors=True); os.makedirs(CKS, exist_ok=True)
    steps = [("smoke_train_cmmd", ["train_arm2.py", "--arm", "smk", "--lam", "0.0", "--epochs", "1", "--smoke_batches", "10", "--snap_every", "1",
                                   "--log_every", "5", "--ckpt_dir", CKS], ENV_CMMD),
             ("smoke_train_cbis", ["train_arm2.py", "--arm", "smkb", "--lam", "0.0", "--epochs", "1", "--smoke_batches", "4", "--snap_every", "1",
                                   "--log_every", "2", "--ckpt_dir", CKS], {}),
             ("smoke_sref", ["s8_sref.py", "--ckpt", f"{CKS}/smk_w_ep001.pt", "--out", "sref_smk.json"], ENV_CMMD),
             ("smoke_fp", ["train_fp.py", "--arm", "smkfp", "--lam", "0.016", "--epochs", "2", "--init", f"{CKS}/smk_last.pt",
                           "--sref_json", f"{SM}/sref_smk.json", "--smoke_batches", "8", "--log_every", "4", "--ckpt_dir", CKS], ENV_CMMD),
             ("smoke_clf", ["s8_clf.py", "--name", "smk_clf", "--smoke"], ENV_CMMD),
             ("smoke_gen", ["s6_generate.py", "--name", "sg", "--ckpts", f"{CKS}/smkfp_w_ep002.pt", "--n_per_class", "2", "--steps", "5"], {}),
             ("smoke_gen_avg", ["s6_generate.py", "--name", "sg2", "--ckpts", f"{CKS}/smk_w_ep001.pt,{CKS}/smkfp_w_ep002.pt", "--n_per_class", "2", "--steps", "5"], {}),
             ("smoke_noise", ["s7_noise.py", "--sets", "sg", "--seed", "1"], ENV_CMMD),
             ("smoke_eval_val", ["s8_eval.py", "--stage", "val", "--sets", "sg,sg2,sg_nl", "--clf", f"{SM}/smk_clf.pt", "--tag", "smk", "--n_boot", "20"], ENV_CMMD),
             ("smoke_eval_test", ["s8_eval.py", "--stage", "test", "--sets", "sg,sg2", "--clf", CLF_B, "--tag", "smkb", "--n_boot", "20",
                                  "--alpha", str(TEST_ALPHA)], {})]
    for name, argv, ex in steps:
        if run_one(name, 0, argv, ex, SM) != 0:
            raise SystemExit(f"SMOKE TEST FAILED at {name} -> nothing long was started. Send Claude this block + S8/logs/{name}.log")
    need_files = [f"{CKS}/smkfp_w_ep002.pt", f"{CKS}/smkfp_fp_log.jsonl", f"{SM}/smk_val_report.txt", f"{SM}/smkb_test_report.txt", f"{SM}/gen_sg_nl.npz"]
    miss = [f for f in need_files if not os.path.exists(f)]
    if miss:
        raise SystemExit(f"SMOKE TEST FAILED: missing {miss}")
    print(open(f"{SM}/smk_val_report.txt").read())
    shutil.rmtree(SM, ignore_errors=True); RC.clear(); print(f"SMOKE TEST PASSED | session {used_h():.2f} h")
else:
    print("smoke test skipped (session", SESSION, ")")
'''

CELL5 = r'''# ===================== CELL 5 — JOB LIST (fixed by the pre-registration) + two-GPU scheduler =====================
def ck(n): return f"{CK}/{n}"
def gen(name, cks, n, seed):
    return ["s6_generate.py", "--name", name, "--ckpts", ",".join(cks), "--n_per_class", str(n), "--dt", "0", "--noise_seed", str(seed)]
TB = str(BUDGET_H - 0.3)
C40, M0 = ck("cmmd_s42_w_ep040.pt"), [ck(f"cmmd_m0_w_ep0{e}.pt") for e in (41, 42, 44)]
CLF_C = f"{OUT}/cmmd_clf_s0.pt"
# (name, argv, env, needs, done, est_h, resumable_min_h or None)
JOBS = [
 ("train_cbis_to90", ["train_arm2.py", "--arm", "base_clean_s42", "--lam", "0.0", "--seed", "42", "--epochs", "90", "--snap_every", "30",
                      "--budget_h", TB, "--ckpt_dir", CK], {}, [], ck("base_clean_s42_w_ep090.pt"), 9.5, 0.25),
 ("clf_cmmd", ["s8_clf.py", "--name", "cmmd_clf_s0"], ENV_CMMD, [], CLF_C, 0.25, None),
 ("train_cmmd", ["train_arm2.py", "--arm", "cmmd_s42", "--lam", "0.0", "--seed", "42", "--epochs", "40", "--snap_every", "40",
                 "--budget_h", TB, "--ckpt_dir", CK], ENV_CMMD, [], C40, 6.0, 0.2),
 ("sref_cmmd", ["s8_sref.py", "--ckpt", C40, "--out", "sref_cmmd40.json"], ENV_CMMD, [C40], f"{OUT}/sref_cmmd40.json", 0.1, None),
 ("fp_cmmd", ["train_fp.py", "--arm", "cmmd_l016", "--lam", "0.016", "--epochs", "42", "--init", ck("cmmd_s42_last.pt"),
              "--sref_json", f"{OUT}/sref_cmmd40.json", "--budget_h", TB, "--ckpt_dir", CK], ENV_CMMD,
  [C40, f"{OUT}/sref_cmmd40.json"], ck("cmmd_l016_w_ep042.pt"), 1.7, 1.0),
 ("m0_cmmd", ["train_fp.py", "--arm", "cmmd_m0", "--lam", "0.0", "--epochs", "44", "--init", ck("cmmd_s42_last.pt"),
              "--sref_json", f"{OUT}/sref_cmmd40.json", "--budget_h", TB, "--ckpt_dir", CK], ENV_CMMD,
  [C40, f"{OUT}/sref_cmmd40.json"], M0[-1], 0.7, 0.25),
 ("sref_cbis90", ["s8_sref.py", "--ckpt", ck("base_clean_s42_w_ep090.pt"), "--out", "sref_cbis90.json"], {},
  [ck("base_clean_s42_w_ep090.pt")], f"{OUT}/sref_cbis90.json", 0.1, None),
 ("fp_cbis90", ["train_fp.py", "--arm", "cl_l016", "--lam", "0.016", "--epochs", "92", "--init", ck("base_clean_s42_last.pt"),
                "--sref_json", f"{OUT}/sref_cbis90.json", "--budget_h", TB, "--ckpt_dir", CK], {},
  [ck("base_clean_s42_w_ep090.pt"), f"{OUT}/sref_cbis90.json"], ck("cl_l016_w_ep092.pt"), 2.1, 1.2),
 ("m0_cbis90", ["train_fp.py", "--arm", "cl_m0", "--lam", "0.0", "--epochs", "92", "--init", ck("base_clean_s42_last.pt"),
                "--sref_json", f"{OUT}/sref_cbis90.json", "--budget_h", TB, "--ckpt_dir", CK], {},
  [ck("base_clean_s42_w_ep090.pt"), f"{OUT}/sref_cbis90.json"], ck("cl_m0_w_ep092.pt"), 0.5, 0.3),
]
VAL = {"c40": [C40], "cm0": [M0[1]], "cfp": [ck("cmmd_l016_w_ep042.pt")], "cavg": [C40] + M0,
       "bm0": [ck("cl_m0_w_ep092.pt")], "bfp": [ck("cl_l016_w_ep092.pt")], "b40": [B40],
       "b60": [ck("base_clean_s42_w_ep060.pt")], "b90": [ck("base_clean_s42_w_ep090.pt")]}
for nm in ("c40", "cm0", "cfp", "cavg", "bm0", "bfp", "b40", "b60", "b90"):
    JOBS.append((f"gen_{nm}", gen(nm, VAL[nm], VAL_N, VAL_SEED), {}, VAL[nm], f"{OUT}/gen_{nm}.npz", 0.45, None))
    if nm == "c40":
        JOBS.append(("noise_c40", ["s7_noise.py", "--sets", "c40", "--seed", str(NOISE_VAL_SEED)], ENV_CMMD,
                     [f"{OUT}/gen_c40.npz"], f"{OUT}/gen_c40_nl.npz", 0.1, None))
G = lambda *n: [f"{OUT}/gen_{x}.npz" for x in n]
EVALS = [("S8_cmmd_fp", "cm0,cfp", CLF_C, ENV_CMMD), ("S8_cmmd_main", "c40,cavg,c40_nl", CLF_C, ENV_CMMD),
         ("S8_cbis_fp", "bm0,bfp", CLF_B, {}), ("S8_cbis_long", "b40,b60,b90", CLF_B, {})]
for tag, sets, clf, ev in EVALS:
    JOBS.append((f"eval_{tag}_val", ["s8_eval.py", "--stage", "val", "--sets", sets, "--clf", clf, "--tag", tag], ev,
                 G(*sets.split(",")) + [clf], f"{OUT}/{tag}_val.json", 0.35, None))
# ---- ONE test stage for C1-C4 (fresh noise seeds; after the matching validation evaluation) ----
TEST = {"tc40": VAL["c40"], "tcm0": VAL["cm0"], "tcfp": VAL["cfp"], "tcavg": VAL["cavg"], "tbm0": VAL["bm0"], "tbfp": VAL["bfp"]}
for nm, cks in TEST.items():
    VJ = {"tc40": "S8_cmmd_main", "tcavg": "S8_cmmd_main", "tcm0": "S8_cmmd_fp", "tcfp": "S8_cmmd_fp", "tbm0": "S8_cbis_fp", "tbfp": "S8_cbis_fp"}[nm]
    JOBS.append((f"gen_{nm}", gen(nm, cks, TEST_N, TEST_SEED), {}, cks + [f"{OUT}/{VJ}_val.json"],
                 f"{OUT}/gen_{nm}.npz", 0.65, None))
JOBS.append(("noise_tc40", ["s7_noise.py", "--sets", "tc40", "--seed", str(NOISE_TEST_SEED)], ENV_CMMD, [f"{OUT}/gen_tc40.npz"],
             f"{OUT}/gen_tc40_nl.npz", 0.1, None))
for tag, sets, clf, ev, vj in (("S8_cmmd_fp", "tcm0,tcfp", CLF_C, ENV_CMMD, "S8_cmmd_fp"), ("S8_cmmd_main", "tc40,tcavg,tc40_nl", CLF_C, ENV_CMMD, "S8_cmmd_main"),
                               ("S8_cbis_fp", "tbm0,tbfp", CLF_B, {}, "S8_cbis_fp")):
    JOBS.append((f"eval_{tag}_test", ["s8_eval.py", "--stage", "test", "--sets", sets, "--clf", clf, "--tag", tag, "--alpha", str(TEST_ALPHA)], ev,
                 G(*sets.split(",")) + [clf, f"{OUT}/{vj}_val.json"], f"{OUT}/{tag}_test.json", 0.35, None))
# ---- C5 (S8 Addendum 1): clean re-test of the September noise-physics loss (pinn2: lambda 0.01) vs lambda 0 ----
SEP40 = os.path.join(os.path.dirname(B40), "base_clean_s42_last.pt")      # clean epoch-40 baseline (weights + Adam state)
assert os.path.exists(SEP40), f"!! {SEP40} missing -> attach clean-data-baseline-retraining (latest version)"
for arm in ("sep_m0", "sep_nlf"):
    if os.path.exists(ck(f"{arm}_w_ep042.pt")):
        if os.path.exists(ck(f"{arm}_last.pt")): os.remove(ck(f"{arm}_last.pt"))      # finished -> free disk
    elif not os.path.exists(ck(f"{arm}_last.pt")):
        shutil.copy(SEP40, ck(f"{arm}_last.pt"))                                       # warm start = epoch 40
for arm, lam in (("sep_m0", "0.0"), ("sep_nlf", "0.01")):
    JOBS.append((f"train_{arm}", ["train_arm2.py", "--arm", arm, "--lam", lam, "--seed", "42", "--epochs", "42", "--snap_every", "1",
                                  "--budget_h", TB, "--ckpt_dir", CK], {}, [], ck(f"{arm}_w_ep042.pt"), 0.5, 0.3))
for nm, arm, n, seed in (("sm0", "sep_m0", VAL_N, VAL_SEED), ("snlf", "sep_nlf", VAL_N, VAL_SEED),
                         ("tsm0", "sep_m0", TEST_N, TEST_SEED), ("tsnlf", "sep_nlf", TEST_N, TEST_SEED)):
    JOBS.append((f"gen_{nm}", gen(nm, [ck(f"{arm}_w_ep042.pt")], n, seed), {},
                 [ck(f"{arm}_w_ep042.pt")] + ([f"{OUT}/S8_cbis_sep_val.json"] if nm.startswith("t") else []), f"{OUT}/gen_{nm}.npz",
                 0.45 if n == VAL_N else 0.65, None))
JOBS.append(("eval_S8_cbis_sep_val", ["s8_eval.py", "--stage", "val", "--sets", "sm0,snlf", "--clf", CLF_B, "--tag", "S8_cbis_sep"], {},
             G("sm0", "snlf") + [CLF_B], f"{OUT}/S8_cbis_sep_val.json", 0.35, None))
JOBS.append(("eval_S8_cbis_sep_test", ["s8_eval.py", "--stage", "test", "--sets", "tsm0,tsnlf", "--clf", CLF_B, "--tag", "S8_cbis_sep",
              "--alpha", str(TEST_ALPHA)], {}, G("tsm0", "tsnlf") + [CLF_B, f"{OUT}/S8_cbis_sep_val.json"], f"{OUT}/S8_cbis_sep_test.json", 0.35, None))
NAMES = [j[0] for j in JOBS]; assert len(set(NAMES)) == len(NAMES)
print(f"{len(JOBS)} jobs | already done: {sum(os.path.exists(j[4]) for j in JOBS)}")

lock = threading.Lock(); RUNNING, TRIED, FAILED = set(), set(), set()
def pick():
    left = BUDGET_H - used_h()
    for name, argv, ev, needs, done, est, rmin in JOBS:
        if name in RUNNING or name in TRIED or os.path.exists(done):
            continue
        if not all(os.path.exists(f) for f in needs):
            continue
        if (rmin is not None and left >= rmin + 0.3) or (rmin is None and left >= est + 0.25):
            return name, argv, ev, done
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
        name, argv, ev, done = j
        try:
            rc = run_one(name, gpu, argv, ev)
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
ex = lambda f: os.path.exists(f)
for f in glob.glob(f"{CK}/*_w_latest.pt") + [ck("cmmd_m0_w_ep043.pt"), ck("cmmd_l016_w_ep041.pt"), ck("cl_m0_w_ep091.pt"), ck("cl_l016_w_ep091.pt")]:
    if ex(f): os.remove(f)
for arm, done in (("cmmd_m0", M0[-1]), ("cmmd_l016", ck("cmmd_l016_w_ep042.pt")), ("cl_m0", ck("cl_m0_w_ep092.pt")), ("cl_l016", ck("cl_l016_w_ep092.pt"))):
    if ex(done) and ex(ck(f"{arm}_last.pt")): os.remove(ck(f"{arm}_last.pt"))
if ex(M0[-1]) and ex(ck("cmmd_l016_w_ep042.pt")) and ex(ck("cmmd_s42_last.pt")): os.remove(ck("cmmd_s42_last.pt"))
if ex(ck("cl_m0_w_ep092.pt")) and ex(ck("cl_l016_w_ep092.pt")) and ex(ck("base_clean_s42_last.pt")): os.remove(ck("base_clean_s42_last.pt"))
TL = f"{OUT}/train_logs"; os.makedirs(TL, exist_ok=True)
for f in glob.glob(f"{CK}/*_history.json") + glob.glob(f"{CK}/*_fp_log.jsonl"):
    shutil.copy(f, TL)
todo = [j[0] for j in JOBS if not ex(j[4])]
json.dump({"session": SESSION, "n_jobs": len(JOBS), "done": [j[0] for j in JOBS if ex(j[4])], "todo": todo, "failed": sorted(FAILED),
           "return_codes": RC}, open(f"{OUT}/S8_state.json", "w"), indent=1)
subprocess.run([sys.executable, "s8_report.py"], cwd=WD, env={**os.environ, "PA_OUT": OUT})
z = shutil.make_archive("/kaggle/working/S8_results", "zip", OUT)
print("ZIP:", z, f"({os.path.getsize(z) / 1e6:.1f} MB)")
print(subprocess.run(f"du -sh {CK} {OUT} /kaggle/working", shell=True, capture_output=True, text=True).stdout)
print("=" * 90)
if not todo:
    print("ALL S8 JOBS DONE -> send S8_results.zip to Claude")
else:
    print(f"S8 NOT FINISHED ({len(todo)} jobs left: {todo[:6]}{' ...' if len(todo) > 6 else ''})")
    print("NEXT SESSION: Add Input -> Your Work -> this notebook (latest version) + the same inputs -> Save & Run All. "
          "Send S8_results.zip to Claude now as well.")
if FAILED:
    print("!! FAILED jobs (send their logs to Claude):", sorted(FAILED))
'''

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.strip("\n").splitlines(keepends=True)},
                code(CELL1), code(SETUP), code(CELL3), code(CELL4), code(CELL5), code(CELL6)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
OUTP = "/mnt/user-data/outputs/pinn_S8_replication_kaggle.ipynb"
json.dump(nb, open(OUTP, "w"), indent=1); print("wrote", OUTP)
