# s12_stats.py - S12 endpoints, poster-style results tables and figures for CMMD and CBIS-DDSM (CPU).
# usage: python s12_stats.py <S12 output folder> <cmmd_data folder>
import os, sys, glob, json
import numpy as np
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, DATA = sys.argv[1], sys.argv[2]
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#c79a1e", "#3aa6b9", "#8a8a8a", "#b0559b", "#4d7f2e"]
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "font.size": 9, "savefig.dpi": 170, "savefig.bbox": "tight"})
DSN = {"cmmd": "CMMD", "cbis": "CBIS-DDSM"}
def auc(p, y):
    r = rankdata(p); n1 = (y == 1).sum(); n0 = (y == 0).sum(); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def runs(ds, pattern):
    return [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(D, f"pred_{ds}_{pattern}")))]
ens = lambda rs, s: np.mean([r[s] for r in rs], 0)
def metrics(p, y):
    yh = (p >= 0.5).astype(int); tp = ((yh == 1) & (y == 1)).sum(); fp = ((yh == 1) & (y == 0)).sum(); fn = ((yh == 0) & (y == 1)).sum()
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
    return {"accuracy": float((yh == y).mean()), "precision": float(pr), "recall": float(rc), "f1": float(2 * pr * rc / max(1e-9, pr + rc))}
M8 = "q8_l4_angle"
ARMS = {  # label -> pattern; poster box 9 models + fair controls + literature controls
    "Classical CNN (small, scratch)": "cnn_smallcnn_s*.npz", "ResNet50": "cnn_resnet50_s*.npz",
    "EfficientNet-B0": "cnn_efficientnet_b0_s*.npz", "ConvNeXt-Tiny": "cnn_convnext_tiny_s*.npz",
    "CNN-PCA8 + logistic regression": f"hybrid_{M8}_LR_s*.npz", "CNN-PCA8 + MLP (matched size)": f"hybrid_{M8}_MLP_s*.npz",
    "CNN-PCA8 + RBF-kernel SVM": "hybrid_rbfsvm_s*.npz", "CNN-PCA8 + quantum-kernel SVM": "hybrid_qkernel_s*.npz",
    "Hybrid CNN+VQC": f"hybrid_{M8}_VQC_s*.npz", "Hybrid CNN+VQC, no entanglement": f"hybrid_{M8}_noent_VQC_s*.npz",
    "Hybrid CNN+VQC, data re-uploading": f"hybrid_{M8}_reup_VQC_s*.npz",
    "patch-PCA8 + logistic regression": f"pure_{M8}_LR_s*.npz", "patch-PCA8 + MLP (matched size)": f"pure_{M8}_MLP_s*.npz",
    "patch-PCA8 + RBF-kernel SVM": "pure_rbfsvm_s*.npz", "patch-PCA8 + quantum-kernel SVM": "pure_qkernel_s*.npz",
    "VQC on patches": f"pure_{M8}_VQC_s*.npz", "VQC on patches, no entanglement": f"pure_{M8}_noent_VQC_s*.npz",
    "VQC on patches, data re-uploading": f"pure_{M8}_reup_VQC_s*.npz"}
L = ["S12 REPORT - quantum (VQC) classifier as in the supervisor's poster, benign vs malignant, CMMD + CBIS-DDSM",
     "(rules: S12_preregistration.txt)", "=" * 120,
     "Simulation: exact state-vector simulator (verified against dense matrices; PennyLane cross-check below). No quantum hardware.",
     "8 qubits x 4 layers = 96 circuit + 18 dense parameters; VQC = 5 seeds, MLP = 5 seeds, CNNs = 3 seeds, LR / SVMs deterministic."]
pc = os.path.join(D, "pennylane_check.json")
if os.path.exists(pc):
    j = json.load(open(pc)); L.append(f"PennyLane {j['pennylane']} cross-check: max |<Z> difference| {j['max_abs_diff']:.1e}")
rng = np.random.default_rng(0)
def boot(pa, pb, y, alpha, n=2000):
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]; d = []
    for _ in range(n):
        b = np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]); d.append(auc(pa[b], y[b]) - auc(pb[b], y[b]))
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
OUT = {}

def do_dataset(ds):
    R = {k: runs(ds, v) for k, v in ARMS.items()}; O = {"arms": {}, "endpoints": {}}
    META = json.load(open(os.path.join(D, f"quantum_meta_{ds}.json"))) if os.path.exists(os.path.join(D, f"quantum_meta_{ds}.json")) else {}
    miss = [k for k, v in R.items() if not v]
    L.extend(["", "#" * 120, f"DATASET: {DSN[ds]}" + (f"   (missing arms: {miss})" if miss else "   (all arms present)"), "#" * 120,
              f"RESULTS TABLE - {DSN[ds]} TEST (ensemble = mean probability over seeds; threshold 0.5 for accuracy / precision / recall / F1)",
              f"  {'model':40s} {'params':>9s} {'seeds':>5s} {'AUC':>6s} {'seed AUC mean+-SD':>18s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'F1':>6s}"])
    par = {"Hybrid CNN+VQC": META.get(f"hybrid_{M8}", {}).get("params_VQC"), "CNN-PCA8 + MLP (matched size)": META.get(f"hybrid_{M8}", {}).get("params_MLP"),
           "VQC on patches": META.get(f"pure_{M8}", {}).get("params_VQC"), "patch-PCA8 + MLP (matched size)": META.get(f"pure_{M8}", {}).get("params_MLP"),
           "CNN-PCA8 + logistic regression": 9, "patch-PCA8 + logistic regression": 9}
    for k, rs in R.items():
        if not rs:
            continue
        y = rs[0]["y_test"]; p = ens(rs, "test"); s = [auc(r["test"], y) for r in rs]; m = metrics(p, y)
        O["arms"][k] = {"auc": auc(p, y), "seed_aucs": s, "val_auc": auc(ens(rs, "val"), rs[0]["y_val"]), **m}
        sd = f"{np.mean(s):.3f}+-{np.std(s, ddof=1):.3f}" if len(s) > 1 else f"{s[0]:.3f}"
        pp = par.get(k) or ("full net" if k in ("Classical CNN (small, scratch)", "ResNet50", "EfficientNet-B0", "ConvNeXt-Tiny") else "-")
        L.append(f"  {k:40s} {str(pp):>9s} "
                 f"{len(rs):>5d} {auc(p, y):6.3f} {sd:>18s} {m['accuracy']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")
    cf = os.path.join(D, f"cnn_features_{ds}.npz")
    if os.path.exists(cf):
        C = np.load(cf); L.append(f"  (frozen ConvNeXt that supplies the hybrid features, its own head: test AUC {auc(C['p_test'], C['y_test']):.3f})")
    def endpoint(name, A, B, alpha, primary):
        if not (R.get(A) and R.get(B)):
            L.append(f"  {name}: arm missing"); return None
        y = R[A][0]["y_test"]; pa, pb = ens(R[A], "test"), ens(R[B], "test"); dd = auc(pa, y) - auc(pb, y); lo, hi = boot(pa, pb, y, alpha)
        eq = (lo > -0.02) and (hi < 0.02)
        verdict = ("PASS" if lo > 0 else "fail") if primary else ("CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0")
        L.append(f"  {name:4s} AUC[{A}] - AUC[{B}] = {dd:+.4f}  CI{100 * (1 - alpha):.2f}% {lo:+.4f}..{hi:+.4f} -> {verdict}"
                 + (" | within +-0.02 (practically equivalent)" if eq else ""))
        return {"diff": dd, "ci": [lo, hi], "verdict": verdict, "passed": bool(lo > 0), "equivalent_0.02": bool(eq)}
    L.extend(["", f"PRIMARY ENDPOINTS on {DSN[ds]} (98.33 % CI = Bonferroni over Q1-Q3; PASS = CI lower bound > 0)"])
    for q, A, B in (("Q1", "Hybrid CNN+VQC", "CNN-PCA8 + MLP (matched size)"), ("Q2", "VQC on patches", "patch-PCA8 + MLP (matched size)"),
                    ("Q3", "Hybrid CNN+VQC", "ConvNeXt-Tiny")):
        O["endpoints"][q] = endpoint(q, A, B, 0.05 / 3, True)
    L.extend(["", "SECONDARY (95 % CI, descriptive)"])
    for nm, A, B in (("S1", "Hybrid CNN+VQC", "CNN-PCA8 + logistic regression"), ("S2", "VQC on patches", "patch-PCA8 + logistic regression"),
                     ("S3", "Hybrid CNN+VQC", "ResNet50"), ("S4", "Hybrid CNN+VQC", "EfficientNet-B0"),
                     ("S5", "Hybrid CNN+VQC", "Classical CNN (small, scratch)"), ("S6", "Hybrid CNN+VQC", "VQC on patches"),
                     ("S7", "Hybrid CNN+VQC", "Hybrid CNN+VQC, no entanglement"), ("S8", "VQC on patches", "VQC on patches, no entanglement"),
                     ("S9", "Hybrid CNN+VQC, data re-uploading", "CNN-PCA8 + MLP (matched size)"),
                     ("S10", "VQC on patches, data re-uploading", "patch-PCA8 + MLP (matched size)"),
                     ("S11", "CNN-PCA8 + quantum-kernel SVM", "CNN-PCA8 + RBF-kernel SVM"),
                     ("S12", "patch-PCA8 + quantum-kernel SVM", "patch-PCA8 + RBF-kernel SVM")):
        O["endpoints"][nm] = endpoint(nm, A, B, 0.05, False)
    L.extend(["", "ENCODINGS AND CIRCUIT SIZE (descriptive): each VQC vs its size-matched MLP and logistic regression on identical inputs"])
    for cfg, mt in META.items():
        v, m_, lr_ = runs(ds, f"{cfg}_VQC_s*.npz"), runs(ds, f"{cfg}_MLP_s*.npz"), runs(ds, f"{cfg}_LR_s*.npz")
        if v and m_ and lr_:
            y = v[0]["y_test"]
            L.append(f"  {cfg:28s} ({mt['role']:8s}; {mt['pca_components']:2d} PCA numbers keep {100 * mt['pca_explained_variance']:4.1f} % variance): "
                     f"VQC {auc(ens(v, 'test'), y):.3f} ({mt.get('params_VQC')} p) | MLP {auc(ens(m_, 'test'), y):.3f} ({mt.get('params_MLP')} p) | "
                     f"LR {auc(ens(lr_, 'test'), y):.3f}")
    # ---- figures: box 4 (Bloch), box 7 (feature space), box 9 (ROC + AUC bars) ----
    vq = runs(ds, f"hybrid_{M8}_VQC_s0.npz")
    if vq and "bloch_test" in vq[0]:
        v = vq[0]; yb = v["y_test"]; B = v["bloch_test"]
        fig = plt.figure(figsize=(11, 3.8))
        for j, q in enumerate((0, 1, 2)):
            a_ = fig.add_subplot(1, 3, j + 1, projection="3d"); u, w = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
            a_.plot_wireframe(np.cos(u) * np.sin(w), np.sin(u) * np.sin(w), np.cos(w), color="#dddddd", lw=0.3)
            for c, col, lab in ((0, PAL[0], "benign"), (1, PAL[4], "malignant")):
                P = B[yb == c, q]; a_.scatter(P[:, 0], P[:, 1], P[:, 2], s=5, color=col, label=lab, alpha=0.6)
            sep = np.linalg.norm(B[yb == 1, q].mean(0) - B[yb == 0, q].mean(0))
            a_.set_title(f"{DSN[ds]} qubit {q} Bloch vectors\n(class-mean distance {sep:.2f})", fontsize=8); a_.set_box_aspect((1, 1, 1))
            if j == 0:
                a_.legend(fontsize=7, loc="upper left")
        fig.savefig(f"{D}/S12_{ds}_box4_bloch.png"); plt.close(fig)
        L.extend(["", "BLOCH VECTORS (hybrid VQC, seed 0, test images): distance between class-mean Bloch vectors per qubit (max possible 2)",
                  "  " + " | ".join(f"q{q} {np.linalg.norm(B[yb == 1, q].mean(0) - B[yb == 0, q].mean(0)):.3f}" for q in range(B.shape[1]))])
        from sklearn.metrics import silhouette_score
        from sklearn.manifold import TSNE
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.7)); sil = {}
        for a_, (nm, F_) in zip(ax[:2], (("classical PCA input (8 numbers)", v["pca_test"]), ("quantum features <Z> (8 numbers)", v["zvec_test"]))):
            Fc = F_ - F_.mean(0); _, _, Vt_ = np.linalg.svd(Fc, full_matrices=False); P2 = Fc @ Vt_[:2].T; sil[nm] = float(silhouette_score(F_, yb))
            for c, col, lab, mk in ((0, PAL[0], "benign", "o"), (1, PAL[4], "malignant", "^")):
                a_.scatter(P2[yb == c, 0], P2[yb == c, 1], s=9, color=col, marker=mk, label=lab, alpha=0.7)
            a_.set_title(f"{DSN[ds]}: PCA of {nm}\nsilhouette {sil[nm]:+.3f}", loc="left", fontsize=8.5); a_.set_xlabel("PC 1"); a_.set_ylabel("PC 2")
        ax[0].legend(frameon=False, fontsize=7.5)
        T = TSNE(2, random_state=0, perplexity=30, init="pca").fit_transform(v["zvec_test"])
        for c, col, lab, mk in ((0, PAL[0], "benign", "o"), (1, PAL[4], "malignant", "^")):
            ax[2].scatter(T[yb == c, 0], T[yb == c, 1], s=9, color=col, marker=mk, label=lab, alpha=0.7)
        ax[2].set_title(f"{DSN[ds]}: t-SNE of quantum features <Z>", loc="left", fontsize=8.5)
        fig.savefig(f"{D}/S12_{ds}_box7_feature_space.png"); plt.close(fig)
        L.extend(["", "FEATURE-SPACE SEPARATION (silhouette score, test images; higher = classes more separated; -1..1)"]
                 + [f"  {k:34s}: {s:+.3f}" for k, s in sil.items()]); O["silhouette"] = sil
    names = [k for k in ARMS if k in O["arms"]]
    if names:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.1]}); fig.subplots_adjust(wspace=0.9)
        KEY = ["Classical CNN (small, scratch)", "ResNet50", "EfficientNet-B0", "ConvNeXt-Tiny", "CNN-PCA8 + MLP (matched size)",
               "Hybrid CNN+VQC", "patch-PCA8 + MLP (matched size)", "VQC on patches"]
        for j, k in enumerate([k for k in KEY if k in names]):
            rs = R[k]; y = rs[0]["y_test"]; p = ens(rs, "test"); o = np.argsort(-p)
            tp = np.cumsum(y[o] == 1) / (y == 1).sum(); fp = np.cumsum(y[o] == 0) / (y == 0).sum(); qk = "VQC" in k or "quantum" in k
            ax[0].plot(np.r_[0, fp], np.r_[0, tp], color=PAL[j % len(PAL)], lw=1.8 if qk else 1.0, ls="-" if qk else "--",
                       label=f"{k} ({O['arms'][k]['auc']:.3f})")
        ax[0].plot([0, 1], [0, 1], color="#999", lw=0.8); ax[0].set_xlabel("false-positive rate"); ax[0].set_ylabel("true-positive rate")
        ax[0].set_title(f"ROC, {DSN[ds]} test (ensembles)", loc="left"); ax[0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
        au = [O["arms"][k]["auc"] for k in names]
        ax[1].barh(range(len(names)), au, color=[PAL[4] if ("VQC" in k or "quantum" in k) else PAL[0] for k in names])
        ax[1].set_yticks(range(len(names)), names, fontsize=6.5); ax[1].set_xlim(0.45, max(au) + 0.06); ax[1].set_xlabel("test AUC")
        ax[1].invert_yaxis(); ax[1].set_title("AUC by model (red = quantum)", loc="left")
        for i, v_ in enumerate(au):
            ax[1].text(v_ + 0.003, i, f"{v_:.3f}", va="center", fontsize=6.5)
        fig.savefig(f"{D}/S12_{ds}_box9_results.png"); plt.close(fig)
    O["meta"] = META
    return O

for ds in ("cmmd", "cbis"):
    OUT[ds] = do_dataset(ds)
L.extend(["", "=" * 120, "OVERALL VERDICT (pre-registered): a question is SUPPORTED only if it PASSES on BOTH datasets"])
for q, txt in (("Q1", "Hybrid CNN+VQC beats the same-size classical head on the same CNN features"),
               ("Q2", "VQC on patches beats the same-size classical head on the same patch features"),
               ("Q3", "Hybrid CNN+VQC beats a fully fine-tuned ConvNeXt-Tiny")):
    r = [OUT[ds]["endpoints"].get(q) for ds in ("cmmd", "cbis")]
    st = ["missing" if x is None else ("PASS" if x["passed"] else "fail") for x in r]
    L.append(f"  {q}: {txt}: CMMD {st[0]} | CBIS-DDSM {st[1]} -> {'SUPPORTED' if st == ['PASS', 'PASS'] else 'NOT supported'}")

# ---- box 2 (patches) and box 5 (circuit), once ----
X = np.load(os.path.join(DATA, "test_cache.npz")); xt, yt = X["x"], X["y"]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4)); i0 = int(np.where(yt == 0)[0][0]); i1 = int(np.where(yt == 1)[0][0])
for a_, i, t in ((ax[0], i0, "benign"), (ax[1], i1, "malignant")):
    a_.imshow(xt[i], cmap="gray"); a_.set_title(f"CMMD test image ({t})", loc="left")
    for k in range(1, 4):
        a_.axhline(64 * k, color="#eb6834", lw=0.8); a_.axvline(64 * k, color="#eb6834", lw=0.8)
    a_.set_xticks([]); a_.set_yticks([]); a_.grid(False)
small = xt[i1].astype(np.float32).reshape(64, 4, 64, 4).mean((1, 3)); ax[2].imshow(small, cmap="gray"); ax[2].grid(False)
ax[2].set_title("64x64 input, 4x4 patches -> 48 numbers -> PCA 8", loc="left"); ax[2].set_xticks([]); ax[2].set_yticks([])
fig.savefig(f"{D}/S12_box2_patches.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 3.3)); n = 8; ax.set_xlim(-0.5, 11); ax.set_ylim(-0.8, n); ax.axis("off")
for q in range(n):
    ax.plot([0.3, 10.6], [n - 1 - q] * 2, color="#555", lw=0.8); ax.text(-0.3, n - 1 - q, f"q{q} |0>", va="center", ha="right", fontsize=8)
    ax.add_patch(plt.Rectangle((0.5, n - 1.35 - q), 0.9, 0.7, color="#d9e8fb")); ax.text(0.95, n - 1 - q, "RY(x)", ha="center", va="center", fontsize=6.5)
for l in range(4):
    x0 = 2 + 2.1 * l
    for q in range(n):
        ax.add_patch(plt.Rectangle((x0, n - 1.35 - q), 1.0, 0.7, color="#fde4d8")); ax.text(x0 + 0.5, n - 1 - q, "Rx Ry Rz", ha="center", va="center", fontsize=6)
    for q in range(n):
        t = (q + 1) % n; xc = x0 + 1.2 + 0.08 * q
        ax.plot([xc, xc], [n - 1 - q, n - 1 - t], color="#1baf7a", lw=0.8); ax.plot(xc, n - 1 - q, "o", color="#1baf7a", ms=3)
    ax.text(x0 + 0.8, -0.6, f"layer {l + 1}", ha="center", fontsize=8)
for q in range(n):
    ax.text(10.75, n - 1 - q, "<Z>", va="center", fontsize=7)
ax.set_title("Main VQC: angle encoding, 4 layers of trainable Rx-Ry-Rz + CNOT ring, <Z> readout -> dense -> softmax", loc="left")
fig.savefig(f"{D}/S12_box5_circuit.png"); plt.close(fig)

txt = "\n".join(L); print(txt)
open(f"{D}/S12_report.txt", "w").write(txt + "\n"); json.dump(OUT, open(f"{D}/S12_results.json", "w"), indent=1, default=float)
