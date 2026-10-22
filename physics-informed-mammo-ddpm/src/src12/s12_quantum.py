# s12_quantum.py - S12 quantum part for ONE dataset (CPU, exact state-vector simulation; rules: S12_preregistration.txt).
# Poster pipeline: image -> preprocessing & patch extraction -> encoding (PCA + angle = main; amplitude / basis = poster's
# other schemes) -> VQC (Rx Ry Rz + CNOT ring) x L -> measure <Z_i> -> classical dense layer -> softmax.
#   "pure"  : VQC on 4x4 patch statistics of the image (poster "VQC (Ours)")
#   "hybrid": VQC on frozen ConvNeXt features (poster "Hybrid CNN+VQC")
# Fair controls on IDENTICAL inputs: logistic regression; MLP with the SAME number of trainable parameters as the VQC head.
# Literature controls: no-entanglement VQC (Bowles et al. 2024), data re-uploading (Perez-Salinas et al. 2020),
# quantum-kernel SVM (Havlicek et al. 2019) vs classical RBF-kernel SVM.
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qsim as Q

ap = argparse.ArgumentParser()
ap.add_argument("--ds", choices=["cmmd", "cbis"], required=True)
ap.add_argument("--data", required=True); ap.add_argument("--feats", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--budget_h", type=float, default=3.0)        # secondary configurations are skipped once this is used up
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
torch.set_num_threads(2); T0 = time.time()
def log(*m): print(f"[S12-quantum-{a.ds}]", *m, flush=True)
bk = Q.Backend(torch)

def val_mask(patients, frac=0.15, seed=0):
    ups = sorted(set(patients)); rng = np.random.default_rng(seed)
    val = set(rng.choice(ups, int(round(frac * len(ups))), replace=False).tolist())
    return np.array([p in val for p in patients])
d = np.load(os.path.join(a.data, "train_cache.npz")); vm = val_mask(d["patient"].astype(str))
X = {"train": d["x"][~vm], "val": d["x"][vm], "test": np.load(os.path.join(a.data, "test_cache.npz"))["x"]}
C = np.load(a.feats); Y = {k: C[f"y_{k}"].astype(np.int64) for k in ("train", "val", "test")}
if a.smoke:
    X = {k: v[:len(Y[k])] for k, v in X.items()}
assert all(len(X[k]) == len(Y[k]) for k in X), {k: (len(X[k]), len(Y[k])) for k in X}

def patch_feats(U8):
    """poster step 2: resize 256 -> 64, per-image min-max, 4x4 grid of 16x16 patches, per patch (mean, std, high-pass energy)"""
    x = U8.astype(np.float32).reshape(len(U8), 64, 4, 64, 4).mean((2, 4))
    mn = x.min((1, 2), keepdims=True); mx = x.max((1, 2), keepdims=True); x = (x - mn) / (mx - mn + 1e-8)
    pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    box = sum(pad[:, i:i + 64, j:j + 64] for i in range(3) for j in range(3)) / 9.0; hp = (x - box) ** 2
    P = lambda z: z.reshape(len(z), 4, 16, 4, 16).transpose(0, 1, 3, 2, 4).reshape(len(z), 16, 256)
    return np.concatenate([P(x).mean(2), P(x).std(2), P(hp).mean(2)], 1)          # (N, 48)
RAW = {"pure": {k: patch_feats(v) for k, v in X.items()}, "hybrid": {k: C[f"f_{k}"] for k in X}}

def pca(kind, n):
    """standardise (train-fit stats) -> PCA to n components (fitted on train-fit) -> z-score"""
    R = RAW[kind]; mu, sd = R["train"].mean(0), R["train"].std(0) + 1e-8
    Z = {k: (v - mu) / sd for k, v in R.items()}
    _, S, Vt = np.linalg.svd(Z["train"], full_matrices=False); V = Vt[:n].T
    P = {k: v @ V for k, v in Z.items()}; m2, s2 = P["train"].mean(0), P["train"].std(0) + 1e-8
    return {k: ((v - m2) / s2).astype(np.float32) for k, v in P.items()}, float((S[:n] ** 2).sum() / (S ** 2).sum())
def encode(kind, n, enc):
    """returns (classical input Z, quantum input) ; angle: 2*arctan(z) | amplitude: 2^n PCA numbers, unit norm, on n qubits |
    basis: sign(z) bits -> computational basis state"""
    if enc == "amp":
        Z, ev = pca(kind, 2 ** n)
        # fix 2026-10-23: patch features have only 48 numbers < 2^6 = 64 amplitudes -> zero-pad (standard amplitude encoding)
        pad = lambda v: np.pad(v, ((0, 0), (0, 2 ** n - v.shape[1]))) if v.shape[1] < 2 ** n else v
        A = {k: pad(v) / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8) for k, v in Z.items()}
        return Z, {k: v.astype(np.float32) for k, v in A.items()}, ev
    Z, ev = pca(kind, n)
    if enc == "basis":
        w = 2 ** np.arange(n - 1, -1, -1)
        return Z, {k: ((v > 0).astype(np.int64) * w).sum(1) for k, v in Z.items()}, ev
    return Z, {k: (2 * np.arctan(v)).astype(np.float32) for k, v in Z.items()}, ev

class VQCHead(nn.Module):
    def __init__(self, n, L, enc="angle", reupload=False, entangle=True):
        super().__init__(); self.n, self.enc, self.reupload, self.entangle = n, enc, reupload, entangle
        self.W = nn.Parameter(0.1 * torch.randn(L, n, 3))                          # near-identity start (pre-stated)
        self.dense = nn.Linear(n, 2); self.perm = Q.cnot_ring_perm(n); self.zs = torch.tensor(Q.zsign(n))
    def state(self, xin):
        B = xin.shape[0]
        if self.enc == "amp":
            init = torch.complex(xin, torch.zeros_like(xin)); ang = torch.zeros(B, self.n)
        elif self.enc == "basis":
            re = torch.zeros(B, 2 ** self.n); re[torch.arange(B), xin] = 1.0
            init = torch.complex(re, torch.zeros_like(re)); ang = torch.zeros(B, self.n)
        else:
            init, ang = None, xin
        return Q.vqc_state(bk, ang, self.W, self.perm, init=init, reupload=self.reupload, entangle=self.entangle)
    def forward(self, xin):
        return self.dense(Q.expz(bk, self.state(xin), self.zs))
class MLPHead(nn.Module):
    def __init__(self, n_in, h):
        super().__init__(); self.net = nn.Sequential(nn.Linear(n_in, h), nn.Tanh(), nn.Linear(h, 2))
    def forward(self, z):
        return self.net(z)
nparams = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)

def fit(model, Xtr, ytr, seed, epochs, bs=64, lr=0.01):
    w = torch.tensor([len(ytr) / (2 * max(1, (ytr == c).sum())) for c in (0, 1)], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr); Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    g = torch.Generator().manual_seed(seed); hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(Yt), generator=g); tl = 0.0
        for i in range(0, len(Yt), bs):
            idx = perm[i:i + bs]; loss = F.cross_entropy(model(Xt[idx]), Yt[idx], weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); tl += float(loss) * len(idx)
        hist.append(tl / len(Yt))
    return model, hist
@torch.no_grad()
def proba(model, Xn):
    return torch.softmax(model(torch.from_numpy(Xn)), 1)[:, 1].numpy()
def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1, n0 = (y == 1).sum(), (y == 0).sum(); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def save(name, pv, pt, **kw):
    np.savez(os.path.join(a.out, f"pred_{a.ds}_{name}.npz"), val=pv, test=pt, y_val=Y["val"], y_test=Y["test"], **kw)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# (feature set, qubits, layers, encoding, re-upload, entangle, role) - in priority order
CONFIGS = [("hybrid", 8, 4, "angle", False, True, "main"), ("pure", 8, 4, "angle", False, True, "main"),
           ("hybrid", 8, 4, "angle", False, False, "noent"), ("pure", 8, 4, "angle", False, False, "noent"),
           ("hybrid", 8, 4, "angle", True, True, "reupload"), ("pure", 8, 4, "angle", True, True, "reupload"),
           ("hybrid", 6, 4, "amp", False, True, "encoding"), ("pure", 6, 4, "amp", False, True, "encoding"),
           ("hybrid", 8, 4, "basis", False, True, "encoding"),
           ("hybrid", 4, 2, "angle", False, True, "size"), ("hybrid", 4, 4, "angle", False, True, "size"),
           ("hybrid", 8, 2, "angle", False, True, "size")]
if a.smoke:
    CONFIGS = [CONFIGS[0], CONFIGS[1], CONFIGS[4], CONFIGS[6], CONFIGS[8]]; a.epochs = 2
SEEDS = [int(s) for s in a.seeds.split(",")]; META = {}
mpath = os.path.join(a.out, f"quantum_meta_{a.ds}.json")
if os.path.exists(mpath):
    META = json.load(open(mpath))

# ---------------- quantum kernel SVM vs classical RBF SVM (fixed feature map, no training of the circuit) ----------------
for kind in ("hybrid", "pure"):
    Z, ANG, _ = encode(kind, 8, "angle")
    with torch.no_grad():
        S = {k: Q.feature_map_state(Q.Backend(np), v.astype(np.float64), reps=2) for k, v in ANG.items()}
    Ktr = np.abs(S["train"].conj() @ S["train"].T) ** 2
    qs = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(Ktr, Y["train"])
    sig = lambda z: 1 / (1 + np.exp(-z))
    pv = sig(qs.decision_function(np.abs(S["val"].conj() @ S["train"].T) ** 2))
    pt = sig(qs.decision_function(np.abs(S["test"].conj() @ S["train"].T) ** 2)); save(f"{kind}_qkernel_s0", pv, pt)
    rb = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced").fit(Z["train"], Y["train"])
    save(f"{kind}_rbfsvm_s0", sig(rb.decision_function(Z["val"])), sig(rb.decision_function(Z["test"])))
    log(f"{kind}: quantum-kernel SVM test AUC {auc(pt, Y['test']):.3f} | RBF SVM test AUC {auc(sig(rb.decision_function(Z['test'])), Y['test']):.3f}")

for kind, n, L, enc, reup, ent, role in CONFIGS:
    if role != "main" and (time.time() - T0) / 3600 > a.budget_h:
        log(f"time budget {a.budget_h} h used -> {role} config {kind} {n}q {L}l {enc} skipped"); continue
    cfg = f"{kind}_q{n}_l{L}_{enc}" + ("_reup" if reup else "") + ("" if ent else "_noent")
    Z, QIN, ev = encode(kind, n, enc)
    vq_params = Q.n_params(n, L) + 2 * n + 2; n_in = Z["train"].shape[1]
    h = max(1, round((vq_params - 2) / (n_in + 3)))                                  # MLP n_in -> h -> 2 with ~ the same parameters
    META[cfg] = {**META.get(cfg, {}), "kind": kind, "qubits": n, "layers": L, "encoding": enc, "reupload": reup, "entangle": ent, "role": role,
                 "pca_components": n_in, "pca_explained_variance": ev, "mlp_hidden": h}
    lr_ = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Z["train"], Y["train"])
    save(f"{cfg}_LR_s0", lr_.predict_proba(Z["val"])[:, 1], lr_.predict_proba(Z["test"])[:, 1])
    for seed in SEEDS:
        for head in ("VQC", "MLP"):
            f = os.path.join(a.out, f"pred_{a.ds}_{cfg}_{head}_s{seed}.npz")
            if os.path.exists(f):
                continue
            torch.manual_seed(seed); t0 = time.time()
            m = VQCHead(n, L, enc, reup, ent) if head == "VQC" else MLPHead(n_in, h)
            Xin = QIN if head == "VQC" else Z
            m, hist = fit(m, Xin["train"], Y["train"], seed, a.epochs)
            pv, pt = proba(m, Xin["val"]), proba(m, Xin["test"]); META[cfg][f"params_{head}"] = nparams(m)
            extra = {}
            if head == "VQC" and role == "main" and seed == SEEDS[0]:
                with torch.no_grad():
                    st = m.state(torch.from_numpy(Xin["test"]))
                    extra = {"zvec_test": Q.expz(bk, st, m.zs).numpy(), "pca_test": Z["test"],
                             "bloch_test": np.stack([Q.bloch(bk, st, q, n).numpy() for q in range(n)], 1), "W": m.W.detach().numpy()}
            save(f"{cfg}_{head}_s{seed}", pv, pt, loss=np.array(hist), **extra)
            log(f"{cfg} {head} seed {seed}: {nparams(m)} params | val AUC {auc(pv, Y['val']):.3f} | test AUC {auc(pt, Y['test']):.3f} | "
                f"final loss {hist[-1]:.4f} | {time.time() - t0:.0f} s | elapsed {(time.time() - T0) / 60:.0f} min")
    json.dump(META, open(mpath, "w"), indent=1)
log(f"done in {(time.time() - T0) / 60:.0f} min")
