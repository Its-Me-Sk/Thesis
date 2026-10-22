# build_handover.py - builds HANDOVER.ipynb (full project context for the next researcher) and the GitHub-ready folder.
import json, os, shutil, glob

REPO = "/home/claude/handover_repo/physics-informed-mammo-ddpm"
OUTS = "/mnt/user-data/outputs"
SCR = "/tmp/claude-0/-home-claude/3f2b8356-4646-5718-a62c-c0e93328cc98/scratchpad"

MD = []
MD.append(r"""# Physics-informed mammogram synthesis + quantum classifier — HANDOVER
*Prepared 23 October 2026 by Z.M. Zulkar Nayeen for the researcher taking over. Read this notebook top to bottom once;
every later section assumes the vocabulary of §1.*

**Base model credit (always state this):** the class-conditional DDPM is **U24_synthesis** — Linardos, Thakur, Gastounioti,
Makris & Bakas, *"Evaluating the utility of AI-generated mammography images in breast cancer analysis"*, Scientific Reports
2026 (public GitHub repo). It is **not our work**. Our contribution is everything we added and tested on top of it.

## 0. One-paragraph summary
Since September 2026 we tested whether **physics** makes synthetic mammograms from this DDPM better. Across **Experiments 1–5, Phase A and studies
S2–S12** the answer is consistently **no inside the generative model**: noise-physics losses,
a genuine Fokker–Planck PINN residual, cheap FP-derived regularisers, physics-based sample filtering and an acquisition
noise layer never improved image quality, and several made it worse — replicated on two datasets (CBIS-DDSM film,
CMMD digital) and confirmed on held-out test sets (S8). S9 explains why (the generator's real errors are brightness
saturation and missing tissue texture, not noise; the FP gradient is orthogonal to the training gradient). Physics **did**
help where it belongs — as **low-dose augmentation for a classifier** (S11, P1 passed, +0.049 AUC), although generic noise
gave most of that gain. At the supervisor's request we also tested a **hybrid quantum (VQC) classifier** (S12): under a
fair, pre-registered test on both datasets it never beat a same-size classical model, and entanglement made no difference.

**Plan agreed so far:** two papers (Paper 1: physics in diffusion + S11; Paper 2: quantum classifier) and two thesis chapters.
Writing is done in **Prism**. The experiments are finished unless the supervisor asks for more.
""")

MD.append(r"""## 1. Vocabulary and fixed conventions (used everywhere)
| term | meaning in this project |
|---|---|
| base model | MONAI `DiffusionModelUNet` (157.5 M params), ε-prediction, T = 1000, class-conditional (benign/malignant), 256×256, U24_synthesis recipe |
| "Experiment-2 recipe" | the exact published training recipe: loss 0.7·MSE + 0.2·(1−MS-SSIM) + 0.05·HF on x̂₀, per-image min-max to [−1,1], flip/shear/scale aug, 15 % label dropout, batch 2 × accumulation 4, Adam 1e-5, clip 1.0, fp16 (`train_arm2.py`) |
| sampler | DDIM-50, classifier-free guidance 2 (chosen in S2); noise bank seed 2025 = validation, 9025 = test (same noise for every arm → image *i* is paired across arms) |
| splits | patient-level. CBIS-DDSM **clean** split (rule B): train 2336 / test 663. CMMD cohort D1: train 1775 / test 438 (5202 images in CMMD in total; D1 = 2214 used). Validation = 15 % of train patients, `val_mask(seed 0)`, identical in every script |
| KID-mammo | unbiased KID (cubic kernel) in the feature space of a ConvNeXt-Tiny benign/malignant classifier trained on that dataset (CBIS: `a6_clf_s0.pt` from S2; CMMD: `cmmd_clf_s0.pt` from S8). **Primary image-quality metric.** Lower = better |
| KID-DINOv2 | the same in DINOv2 ViT-B/14 space — an independent "second eye"; plus precision/recall (k = 3) and Inception FID/KID (descriptive) |
| decision rule (S6–S10) | PASS if KID-mammo ≤ 0.90× control **and** paired-bootstrap CI of the difference < 0 **and** KID-DINOv2 ≤ 1.00× **and** recall ≥ control − 0.05 |
| empirical floor | real-vs-real KID-mammo under this protocol ≈ 0.0055; our generators are ≈ 0.28 (CMMD) – 0.49 (CBIS) |
| checkpoint noise | KID varies ≈ 12 % between neighbouring checkpoints of ONE run (≈ 3–6× the sampling uncertainty) → effects smaller than ≈ 20 % are not trustworthy |
| pre-registration | rules written and time-stamped **before** a run; S8 (document), **S10, S11, S12 (OSF, embargoed to ≈ Oct 2027)**. S3–S7: rules fixed in the script header before each run ("pre-specified"), not formally registered |
| NLF | noise-level function: high-pass variance vs local intensity, `PhysicsNLF` in `pinn_physics.py` (7-tap Gaussian high-pass, 16 intensity bins) |
| FP residual | score Fokker–Planck residual of the model's score, exact-gradient finite differences in fp32 (`fp_core.py`) |
""")

MD.append(r"""## 2. Timeline — every study, its question and its result
Dates are approximate (2026). "Notebook" = file in `notebooks/` (each is self-contained: the code is embedded in Cell 1).

### Phase 1 — physics-informed loss on the published model (September – mid October)
| # | question | design | result | notebook |
|---|---|---|---|---|
| Exp 1 | does a Poisson–Gaussian NLF loss help? | baseline vs physics, 1 seed | no difference (KID 0.135 vs 0.142, FID 177 vs 182). **Mistake:** loss simplified to MSE → both far below the published model; lesson: keep the full recipe | `pinn_ablation_kaggle.ipynb` |
| Exp 2 | same, with the exact published recipe | one-sided NLF hinge on x̂₀ (τ = 2.21), 40 epochs, 1 seed | no difference (KID 0.082 vs 0.084, p = 0.81); physics over-corrected the noise (residual 0.46 vs real 1.13). Checkpoint `base2_w_ep040.pt` (Kaggle notebook *notebook64a7f79592*) is the "main" model of Phase A | `pinn_experiment2_kaggle.ipynb` |
| Exp 3 | low data (25 %), where a prior should matter | two-sided NLF band, 3 seeds (42–44), downstream ResNet-18 | primary (downstream AUC) not met; exploratory signal: KID-mammo 1.23 → 0.88 in 3/3 seeds (p = 0.16) | `pinn_experiment3_final_kaggle.ipynb` |
| Exp 4 | replicate the Exp-3 signal | 4 new seeds (45–48), pre-registered | **not confirmed** (p = 0.086); Inception FID worse with physics 4/4 (p = 0.044); downstream AUC 0.676 vs 0.676 | `pinn_experiment4_replication_kaggle.ipynb` |
| Exp 5 | is it architecture-specific? (RA's advice) | ADM U-Net, DiT-B/8, WGAN-GP, each ± physics, 3 seeds, 3 Kaggle accounts | **primary not met**: pooled KID-mammo median **+21.3 % worse** with physics (p = 0.92); NPS −17.9 % (p = 0.41) | `pinn_experiment5_three_generators_kaggle.ipynb` |

### Phase 2 — audit, clean data, a genuine PINN (Fokker–Planck) term (18 – 21 October)
| # | question | result | notebook |
|---|---|---|---|
| Phase A | diagnostics before a real PINN | **A1 FAIL — data leakage**: 22 patients in both CBIS splits, 18 identical images, label conflicts, 282 test rows skipped by the repo's `prepare.py`. A3: at 256 px the "noise curve" is texture-dominated. A4: FP residual measurable in fp32 (not fp16). A5: FP cost → seeds | `pinn_phaseA_kaggle.ipynb` |
| Data audit + clean baseline | build a leak-free split and retrain | clean rule B (train 2336 / test 663, 33 label-conflict images removed, 31 overlapping patients moved to test); baseline retrained from scratch, seeds 42 and 43, 40 epochs (Kaggle notebook *clean-data-baseline-retraining*) | `data_audit_kaggle.ipynb`, `pinn_clean_baseline_kaggle.ipynb` |
| S2 | audit the clean baseline, build the evaluation tools | ConvNeXt judge `a6_clf_s0.pt` (KID-mammo features), view classifier, guidance choice (→ 2), FP reference S_ref(t). Kaggle notebook *s2-baseline-audit* (also holds `clean_data/`) | `pinn_S2_baseline_audit_kaggle.ipynb` |
| S3 | train with the FP residual (λ 0.016 / 0.054 / 0.107) | **H1 FAIL**: KID-mammo ×1.54–2.58, MSE +17–52 %; FP gradient ≈ orthogonal to the data gradient (cos ≈ 0), clipping 77 % → 89–97 % of steps | `pinn_S3_fp_feasibility_kaggle.ipynb` |
| S4a | cheaper FP-derived regularisers + FP gradient SNR | FP gradient SNR 0.04–0.12 (usable FP gradient would need 6–50 h/epoch); score-norm and divergence arms: none pass (KID ×1.04–1.36) | `pinn_S4a_screening_kaggle.ipynb` |
| S5 | physics as a quality **score** / filter instead of a loss | FP score unreliable (re-test Spearman 0.26); noise-physics filter worse; across 10 models FP residual vs KID ρ = 0.18 but validation MSE vs KID ρ = 0.89; checkpoint SD 6.4× sampling SD; ε-scaling worse | `pinn_S5_physics_scoring_kaggle.ipynb` |
| S6 | training-free fixes | weight averaging ×0.914 and best-of-2 ×0.911 (CI < 0 but not ≤ 0.90 → fail); dynamic thresholding ×1.35 → no winner | `pinn_S6_final_kaggle.ipynb` |
| S7 | physics acquisition-noise layer (30 h account) | ×1.415 KID-mammo (harm) → fail | `pinn_S7_acquisition_kaggle.ipynb` |

### Phase 3 — replication, diagnosis, where physics does help, quantum (21 – 23 October)
| # | question | result | notebook |
|---|---|---|---|
| S8 | replicate on a 2nd dataset (CMMD, digital) + test sets | **C1–C5 all FAIL; every harm repeated on the held-out test sets (99 % CI)**. C1 FP CMMD ×1.82/×1.94 · C2 FP CBIS-90 ×1.64/×1.67 · C3 weight averaging ×1.00/×1.06 · C4 noise layer ×1.13/×1.12 · C5 Sept. noise loss ×1.13/×1.15 (it *satisfied* its physics constraint, images still worse). Descriptive D1: longer training helps (epoch 60 ×0.81). Kaggle notebook *s8-replication* holds `cmmd_data/`, `clean_data/`, `cmmd_clf_s0.pt`, all generated sets | `pinn_S8_data_check.ipynb`, `pinn_S8_replication_kaggle.ipynb` |
| S9 | why did physics fail? (exploratory, CPU) | NLF shape identical at full DICOM resolution and at 256 px → **not** a resolution artefact. Generated CMMD lacks mid/high-frequency texture (high band ×0.38) and has too many saturated pixels (1.4–3.1 % vs 0.19 % real); CBIS has excess HF (×2). Physics loss did not even fix the noise curve. FP gradient cos = 0.000, raises clipping | `pinn_S9_diagnostics_kaggle.ipynb` |
| S10 | does physics help beyond an improved (EMA, 2-seed) baseline? | **designed, pre-registered on OSF, NOT RUN** (decision 22 Oct). Either run it (≈ 25 GPU-h) or withdraw / disclose it on OSF | `pinn_S10_ema_physics_kaggle.ipynb` |
| S11 | physics where it belongs: dose-reduction **augmentation** for a classifier | **P1 PASS**: 25 %-dose CMMD test AUC 0.648 → 0.696 (+0.049, 98.33 % CI +0.016..+0.084). **P2 fail**: physics vs generic matched noise +0.012 (CI −0.008..+0.031). **P3 fail**: clean AUC −0.025. Generic noise alone +0.037 (significant). CBIS-trained physics aug hurt transfer to CMMD (−0.052) | `pinn_S11_dose_augmentation_kaggle.ipynb` |
| S12 (+S12b) | supervisor's poster: hybrid quantum VQC classifier | **Q1–Q3 NOT supported on either dataset.** Hybrid CNN+VQC vs same-size MLP: CMMD 0.712 vs 0.729, CBIS 0.738 vs 0.757 (CBIS significant); vs fine-tuned ConvNeXt significantly worse. No entanglement = same AUC (±0.003). 9 VQC variants (angle/amplitude/basis encoding, 4–8 qubits, 2–4 layers) never beat their classical twin. Quantum ⟨Z⟩ features *look* more separated (silhouette 0.12 vs 0.02) yet classify worse | `pinn_S12_quantum_classifier_kaggle.ipynb`, `pinn_S12b_complete_secondary_kaggle.ipynb` |
""")

MD.append(r"""## 3. Where everything lives
### Kaggle (notebook outputs are the "databases" of this project)
| what | Kaggle location | needed by |
|---|---|---|
| CBIS-DDSM images | dataset *CBIS-DDSM Breast Cancer Image Dataset* (awsaf49) | Exp 1–5, Phase A, clean baseline |
| CMMD images (DICOM) | dataset *The Chinese Mammography Database (CMMD) 2022* (tommyngx) | S8 data build, S9 part B |
| oracle classifier | dataset *oracle-checkpoint-final* | Exp 1–5 |
| Experiment-2 checkpoint `base2_w_ep040.pt` | notebook *notebook64a7f79592* | Phase A |
| clean baseline checkpoints | notebook *clean-data-baseline-retraining* | S2–S8 |
| `clean_data/`, `a6_clf_s0.pt` (CBIS judge) | notebook *s2-baseline-audit* (account *asefahmedshimanto*) | S3–S12 |
| `cmmd_data/`, `clean_data/`, `cmmd_clf_s0.pt`, all S8 generated sets and features, `S8/train_logs` | notebook *s8-replication* (account *twn7y2srym2ds*) | S9, S10, S11, S12 |
| S11, S12, S12b outputs | notebooks *S11 — physics-based dose*, *S12 — quantum*, *S12b — complete* (account *twn7y2srym2ds*) | paper figures |

**Access:** the new researcher needs these notebooks shared with their Kaggle account (Share → add account → *Can view*),
or the owner publishes the needed outputs as Kaggle datasets. Ask Zulkar for access; nothing else is required.

### This repository
```
HANDOVER.ipynb              <- this file
README.md
notebooks/                  <- every Kaggle notebook (self-contained; upload one to Kaggle and run)
src/                        <- the same code as plain .py files (src = Phase A, src2..src12 = S2..S12, ref = original repo files we patched)
builders/                   <- build_*.py: regenerate a notebook from src/ (python builders/build_s12_nb.py)
preregistrations/           <- S4a amendment, S7, S8, S10, S11, S12 (v2) rule documents
reports/                    <- final text reports: S3, S4, S5, S7, S8 summary, S9, S11 (corrected), S12 (complete), Exp 5, Phase A, analysis pack
figures/                    <- S9, S12, analysis-pack and architecture figures
tests/                      <- test_qsim.py: verifies the quantum simulator against dense matrices (numpy only)
```
""")

MD.append(r"""## 4. How to run anything on Kaggle (lessons learned the hard way)
1. **Upload** the notebook from `notebooks/`, then **Add Input** exactly as written in its first markdown cell.
2. **Settings:** GPU T4 ×2 + Internet ON for training/eval notebooks; **Accelerator None** for CPU-only ones (S9, S12b, data audit). A GPU session burns quota even when idle — stop it after downloading.
3. **Commit (Save Version → Save & Run All)** is preferred: it survives a closed browser and keeps outputs. Logs sometimes appear only after several minutes. If a commit shows **no log at all after 20–25 min** (happened with S7, S9, S11 once), cancel and use **Run All in the editor** (keep the tab open and the laptop awake, then download the zip from Output before closing).
4. Every notebook starts with a **smoke test** (a few minutes on tiny data). If it prints `SMOKE TEST FAILED`, send the named log — do not continue.
5. Multi-session notebooks (S8, S10) **carry forward**: in session 2+, add the notebook's *own previous output* as an extra input.
6. Weekly GPU quota is 30 h per account and resets on Saturday. Kaggle's rules allow one account per person.
7. Known pitfalls already handled in the code: never force-reinstall numpy (Python 3.13 image); statistics run in a subprocess (scipy/numpy clash in the kernel); `ConcurrencyViolation` on commit = two open tabs; input folders are read-only (copy with `shutil.copyfile` before overwriting); PennyLane is installed into a separate folder so it cannot break torch.
""")

MD.append(r"""## 5. Code map (src/)
| folder | study | key files |
|---|---|---|
| `ref/` | original U24_synthesis files we use/patched | `train_arm2.py` (Experiment-2 recipe), `pinn_physics.py` (`PhysicsNLF`), `prepare.py`, `eval_arm.py` |
| `src/` | Phase A + data audit + clean split | `a1_leakage.py` … `a5_bench.py`, `data_audit.py`, `prepare_clean.py`, `phaseA_common.py` (shared helpers: `val_mask`, `find`, env switches `PA_OUT`, `PA_DATA`, `PA_TMP`) |
| `src2/` | S2 audit | `a6_classifier.py` (fixed ConvNeXt recipe: `build`, `prep`, `augment`, `predict`, `auc`), `fp_core.py` (FP residual + exact gradient), `a7_generate.py`, `a9_metrics.py` |
| `src3/`–`src7/` | S3–S7 | `train_fp.py` (FP training), `train_reg.py`, `s5_score.py`, `s6_generate.py` (sampler incl. weight averaging, dynamic thresholding), `s7_stats.py` (KID / bootstrap / decision rule — reused by S8–S10) |
| `src8/` | S8 | `prepare_cmmd.py` (CMMD DICOM → 256 px cache, patient split), `s8_clf.py`, `s8_eval.py`, `s8_report.py` |
| `src9/` | S9 | `s9_analysis.py` (all parts B–I) |
| `src10/` | S10 (not run) | `train_ema.py` (EMA on CPU), `s10_eval.py`, `s10_pool.py` |
| `src11/` | S11 | `s11_clf.py` (R0/R1/R2 augmentation), `s11_stats.py` |
| `src12/` | S12 | `qsim.py` (exact batched state-vector simulator, numpy+torch), `s12_quantum.py`, `s12_gpu.py`, `s12_plcheck.py`, `s12_stats.py` |
""")

MD.append(r"""## 6. Corrections and disclosures (must appear in the papers)
* Exp 1 used a simplified loss (our error) → superseded by Exp 2.
* Phase A found **patient leakage** in the CBIS-DDSM split used by the original repo and by Exp 1–5 → everything from the clean baseline onwards uses the clean patient-level split. Exp 1–5 results are therefore on the leaky split.
* S8 is the first study with a formal pre-registration document; S10/S11/S12 are on OSF. "Pre-registered" is used only for these.
* The CMMD and CBIS test images were used more than once for KID of generated images (S6–S8, S11, S12 classification); never to train or select a model.
* S11: the stats script mislabelled one **secondary** result (S3a "CI includes 0" while the CI lies below 0). Fixed and recomputed; numbers unchanged, primary endpoints unaffected (`reports/S11_report_corrected.txt` + the original).
* S12: a code error (patch amplitude encoding: 48 numbers < 64 amplitudes) stopped 3 secondary configurations per dataset; fixed by zero-padding and run in S12b (CPU). The S12b in-notebook report rebuild failed; the report was rebuilt from the prediction files with the same script; parameter counts restored from S12 metadata. Primary endpoints unaffected.
* S12 v1 pre-registration (CMMD only) was superseded by v2 (both datasets) before any S12 code ran; v2 was written after S9 results were seen (S9 is unrelated to classification).
* S10 was pre-registered but not run → withdraw on OSF or state it in the paper.
* Citations proposed during the project (some via ChatGPT) are **not all verified** — check every reference before submission.
""")

MD.append(r"""## 7. What is left (in priority order)
1. **Write Paper 1** — *physics priors in a clinical-scale conditional DDPM: replicated negative results, mechanisms, and where physics does help (augmentation)*. Material: S3–S9, S11 (+ Exp 1–5 as history, Phase A leakage finding). Targets discussed: Computers in Biology and Medicine, Physics in Medicine & Biology, Medical Physics; Scientific Reports / PLOS ONE as safer options. Honest self-assessment ≈ 7/10.
2. **Write Paper 2** — *a fair, pre-registered test of hybrid quantum classifiers for mammography* (S12/S12b): conference / IEEE Access / specialised QML journal. ≈ 6/10. Main messages: no quantum advantage vs parameter-matched classical heads on two datasets; entanglement irrelevant; "separated clusters" pictures are not evidence.
3. **Thesis**: Chapter A = physics-informed synthesis, Chapter B = quantum classification; common theme: *where domain knowledge actually helps in mammography AI under fair tests*. Thesis due ≈ April/May 2027.
4. Decide S10 (run ≈ 25 GPU-h, or withdraw on OSF). Without it, state the limitation "no EMA / multi-seed baseline".
5. Verify all citations; release the code on GitHub (this repo) with the U24_synthesis credit; make paper numbers match the report files exactly.
6. Optional figures: the mammogram-over-t (denoising trajectory) figure; S9 and S12 figures are ready in `figures/`.
7. Limitations to state: one base generator for the main study, 256 px, simulated low-dose test (no real low-dose pairs exist publicly), no reader study, quantum = exact simulation (no hardware), single ConvNeXt feature extractor per dataset.
""")

MD.append(r"""## 8. Quick checks you can run from this repository
The cells below need only Python with numpy, scipy and matplotlib (no GPU, no Kaggle).
""")

CODE1 = r'''# list the repository and print the final reports' verdict lines
import os, glob, re
for d in ("notebooks", "preregistrations", "reports", "figures"):
    print(f"{d}/ ({len(os.listdir(d))} files)")
for f in ("reports/S8_FINAL_DECISION_SUMMARY.txt", "reports/S11_report_corrected.txt", "reports/S12_report_complete.txt"):
    print("\n=====", f)
    for line in open(f, encoding="utf-8"):
        if re.search(r"FAIL|PASS|fail|SUPPORTED|^C[1-5] ", line):
            print(line.rstrip()[:160])
'''
CODE2 = r'''# verify the quantum simulator used in S12 against brute-force dense matrices (numpy only, ~5 s)
import subprocess, sys
print(subprocess.run([sys.executable, "tests/test_qsim.py"], capture_output=True, text=True).stdout)
'''
CODE3 = r'''# show the key S12 and S9 figures
from IPython.display import Image, display
for f in ("figures/S12_cmmd_box9_results.png", "figures/S12_cbis_box9_results.png", "figures/S9_C_measured_noise_curves.png",
          "figures/S9_D_power_spectra.png"):
    print(f); display(Image(filename=f, width=900))
'''

def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}
def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}
nb = {"cells": [md(m) for m in MD] + [code(CODE1), code(CODE2), code(CODE3)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}

# ------------------------------- assemble the repository -------------------------------
shutil.rmtree(REPO, ignore_errors=True)
for d in ("notebooks", "src", "builders", "preregistrations", "reports", "figures", "tests"):
    os.makedirs(f"{REPO}/{d}", exist_ok=True)
json.dump(nb, open(f"{REPO}/HANDOVER.ipynb", "w"), indent=1)
for f in sorted(glob.glob(f"{OUTS}/*.ipynb")):
    shutil.copy(f, f"{REPO}/notebooks/")
P = "/home/claude/phaseA"
for d in ["ref", "src"] + [f"src{i}" for i in range(2, 13)]:
    for f in glob.glob(f"{P}/{d}/*.py"):
        os.makedirs(f"{REPO}/src/{d}", exist_ok=True); shutil.copy(f, f"{REPO}/src/{d}/")
for f in glob.glob(f"{P}/build_*.py"):
    shutil.copy(f, f"{REPO}/builders/")
for f in ("S4a_rule_amendment.txt", "S7_preregistration.txt", "S8_preregistration.txt", "S10_preregistration.txt",
          "S11_preregistration.txt", "S12_preregistration.txt"):
    shutil.copy(f"{OUTS}/{f}", f"{REPO}/preregistrations/")
rep = {f"{OUTS}/S8_FINAL_DECISION_SUMMARY.txt": "S8_FINAL_DECISION_SUMMARY.txt", f"{OUTS}/S11_report_corrected.txt": "S11_report_corrected.txt",
       f"{SCR}/s11res/S11_report_original_kaggle.txt": "S11_report_original_kaggle.txt", f"{OUTS}/S12_report_complete.txt": "S12_report_complete.txt",
       f"{SCR}/s9res/S9_PLUS_report.txt": "S9_PLUS_report.txt", f"{SCR}/s3res/S3_report.txt": "S3_report.txt",
       f"{SCR}/s4res/amended/S4_report.txt": "S4a_report.txt", f"{SCR}/s5res/S5_report.txt": "S5_report.txt",
       f"{SCR}/s7res/S7_report.txt": "S7_report.txt", f"{SCR}/s8res1/S8_report.txt": "S8_session1_report.txt",
       f"{SCR}/pa_res/phaseA_report.txt": "PhaseA_report.txt", f"{SCR}/final5/final5_ALL/summary_exp5.txt": "Exp5_summary.txt",
       f"{OUTS}/analysis_pack/analysis_results.txt": "analysis_pack_results.txt", f"{OUTS}/analysis_pack/weakness_response_plan.txt": "weakness_response_plan.txt"}
for s, d in rep.items():
    if os.path.exists(s):
        shutil.copy(s, f"{REPO}/reports/{d}")
    else:
        print("missing report", s)
for f in glob.glob(f"{SCR}/s9res/*.png") + glob.glob(f"{SCR}/s12final/*.png") + glob.glob(f"{OUTS}/analysis_pack/*.png") + \
         glob.glob(f"{OUTS}/architecture_figures/*.png") + glob.glob(f"{SCR}/final5/final5_ALL/*.png"):
    shutil.copy(f, f"{REPO}/figures/")
shutil.copy(f"{OUTS}/analysis_pack/make_analysis.py", f"{REPO}/src/analysis_pack_make_analysis.py")
# simulator test, made path-independent
t = open(f"{SCR}/s12t/test_qsim.py").read().replace('sys.path.insert(0, "/home/claude/phaseA/src12")',
    'import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "src12"))')
open(f"{REPO}/tests/test_qsim.py", "w").write(t)
open(f"{REPO}/README.md", "w").write("""# Physics-informed mammogram synthesis (U24_synthesis DDPM) + quantum classifier study

Thesis project, September-October 2026. **Start with `HANDOVER.ipynb`** - full context, every study, every result,
how to run, what is left.

Base generative model: **U24_synthesis** - Linardos, Thakur, Gastounioti, Makris & Bakas, "Evaluating the utility of
AI-generated mammography images in breast cancer analysis", Scientific Reports 2026. Not our work; we build on it.

* `notebooks/` - Kaggle notebooks, each self-contained (code embedded). Upload to Kaggle, add the inputs listed in its first cell, run.
* `src/` - the same code as .py files; `builders/` regenerate the notebooks from `src/`.
* `preregistrations/`, `reports/`, `figures/` - rules written before each run, final reports, figures.
* `tests/test_qsim.py` - checks the quantum simulator against dense matrices (numpy only).

Datasets are NOT included (CBIS-DDSM and CMMD are on Kaggle/TCIA under their own licences).
""")
print("repo built at", REPO)
