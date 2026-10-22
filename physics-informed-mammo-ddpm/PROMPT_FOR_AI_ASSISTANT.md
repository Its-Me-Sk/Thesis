# Prompt to paste into an AI assistant (Claude / ChatGPT) together with this repository

Copy everything below the line, then attach or upload `HANDOVER.ipynb`, `LINKS.md` and the report you are working on
(from `reports/`). For code work, also attach the notebook from `notebooks/` you want to change.

---
I am taking over a thesis project from a teammate. Attached are HANDOVER.ipynb (full project history, conventions,
results, open tasks) and LINKS.md (Kaggle inputs). Read both completely before answering.

Context in one line: we tested physics-informed regularisation of a class-conditional mammogram DDPM (U24_synthesis,
Linardos et al., Scientific Reports 2026 - NOT our work, always credit it), a dose-reduction augmentation for a
classifier (S11) and a hybrid quantum VQC classifier (S12), on CBIS-DDSM and CMMD, with pre-registered rules.

Rules for helping me:
1. Never change a pre-registered rule, metric, threshold, seed or test set after seeing results; anything new is
   "exploratory" or needs a new pre-registration written BEFORE running.
2. Do not invent numbers: every number I put in a paper must come from a file in reports/ or a results zip.
3. Keep the existing conventions (splits, val_mask seed 0, sampler DDIM-50 guidance 2, KID-mammo feature spaces,
   decision rule) unless I explicitly start a new study.
4. Code runs on Kaggle (2x T4 or CPU, 30 GPU-h/week). Give complete, copy-paste-ready notebooks or cells, a smoke test
   first, and the exact Kaggle inputs to attach.
5. Tell me honestly when a result is negative or weak; do not oversell.

My first task: <write your task here, e.g. "draft the Methods section of Paper 1 from S3-S9 and S11" or
"decide with me whether to run S10 or withdraw it on OSF">.
