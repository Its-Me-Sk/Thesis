# Links and inputs

## Kaggle notebooks whose OUTPUTS are the project data (ask Zulkar to share them with your Kaggle account)
| notebook | link | what its output contains |
|---|---|---|
| S8 replication | https://www.kaggle.com/code/twn7y2srym2ds/s8-replication | `cmmd_data/` (CMMD 256 px caches + patient split), `clean_data/`, `S8/cmmd_clf_s0.pt` (CMMD judge), all S8 generated sets, features, training logs |
| S2 baseline audit | https://www.kaggle.com/code/asefahmedshimanto/s2-baseline-audit | `clean_data/` (clean CBIS-DDSM split), `S2/a6_clf_s0.pt` (CBIS judge) |
| Clean baseline retraining | https://www.kaggle.com/code/asefahmedshimanto/clean-data-baseline-retraining (owner not confirmed - Zulkar to check) | clean-split DDPM checkpoints (seeds 42, 43; epoch 40) |
| Experiment 2 | https://www.kaggle.com/code/asefahmedshimanto/notebook64a7f79592 (owner not confirmed - Zulkar to check) | `base2_w_ep040.pt` (Experiment-2 checkpoint, leaky split; Phase A only) |
| S11 physics-based dose | https://www.kaggle.com/code/twn7y2srym2ds/s11-physics-based-dose | S11 predictions and report |
| S12 quantum / S12b complete | on account twn7y2srym2ds (ask Zulkar for the exact links) | S12 predictions, figures, report |

Inside a Kaggle notebook these appear as `/kaggle/input/notebooks/<owner>/<notebook-slug>/...`
(for example `/kaggle/input/notebooks/twn7y2srym2ds/s8-replication/cmmd_data`). Our notebooks FIND these folders
automatically by file name, so you only have to attach them (Add Input -> Notebooks).

## Datasets (attach on Kaggle with Add Input -> Datasets -> search the exact title)
| dataset | Kaggle search title (owner) | original source |
|---|---|---|
| CBIS-DDSM | "CBIS-DDSM Breast Cancer Image Dataset" (awsaf49) | https://www.cancerimagingarchive.net/collection/cbis-ddsm/ |
| CMMD | "The Chinese Mammography Database (CMMD) 2022" (tommyngx) | https://www.cancerimagingarchive.net/collection/cmmd/ |
| oracle classifier (Exp 1-5 only) | "oracle-checkpoint-final" (project dataset; ask Zulkar) | - |

## Base model
U24_synthesis - Linardos, Thakur, Gastounioti, Makris & Bakas, Scientific Reports 2026 (public GitHub repository of the paper).

## Pre-registrations (OSF, embargoed until ~Oct 2027)
S10, S11, S12 (v1 and v2 amendment) are registered under Zulkar's OSF account - ask him to add you as a contributor or send the links.
The text of every registration is in `preregistrations/`.
