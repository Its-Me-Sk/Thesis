# s12_gpu.py - S12 GPU part (rules: S12_preregistration.txt).
#   --task feats : frozen ConvNeXt-Tiny of the dataset (CMMD: cmmd_clf_s0.pt from S8; CBIS-DDSM: a6_clf_s0.pt from S2)
#                  -> 768-d penultimate features of train-fit / val / test
#                  (the "CNN" of the poster's Hybrid CNN+VQC; same layers as the KID-mammo features)
#   --task cnn   : the poster's classical baselines, fine-tuned end to end with the fixed S2 recipe:
#                  smallcnn (4-block CNN from scratch, lr 1e-3), resnet50, efficientnet_b0, convnext_tiny (lr 1e-4)
import os, sys, json, math, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a6_classifier import build, prep, augment, predict, auc, dev

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["feats", "cnn"], required=True)
ap.add_argument("--ds", choices=["cmmd", "cbis"], required=True)
ap.add_argument("--data", required=True)                 # cmmd_data/ or clean_data/ folder
ap.add_argument("--clf", default="")                     # cmmd_clf_s0.pt (CMMD) or a6_clf_s0.pt (CBIS-DDSM)
ap.add_argument("--archs", default="smallcnn,resnet50,efficientnet_b0,convnext_tiny")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--out", required=True)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
def log(*m): print(f"[S12-{a.task}-{a.ds}]", *m, flush=True)

def val_mask(patients, frac=0.15, seed=0):
    ups = sorted(set(patients)); rng = np.random.default_rng(seed)
    val = set(rng.choice(ups, int(round(frac * len(ups))), replace=False).tolist())
    return np.array([p in val for p in patients])
d = np.load(os.path.join(a.data, "train_cache.npz")); xtr, ytr, ptr = d["x"], d["y"].astype(np.int64), d["patient"].astype(str)
vm = val_mask(ptr); XF, YF, XV, YV = xtr[~vm], ytr[~vm], xtr[vm], ytr[vm]
t = np.load(os.path.join(a.data, "test_cache.npz")); XT, YT = t["x"], t["y"].astype(np.int64)
if a.smoke:
    XF, YF, XV, YV, XT, YT, a.epochs = XF[:64], YF[:64], XV[:32], YV[:32], XT[:40], YT[:40], 1

if a.task == "feats":
    m = build(); m.load_state_dict({k: v.float() for k, v in torch.load(a.clf, map_location="cpu").items()}); m = m.to(dev).eval()
    @torch.no_grad()
    def feats(X, bs=50):
        out = []
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(dev).float().div(255).unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(m.classifier[1](m.classifier[0](m.avgpool(m.features(prep(xb))))).float().cpu())
        return torch.cat(out).numpy()
    P = {"f_train": feats(XF), "f_val": feats(XV), "f_test": feats(XT), "y_train": YF, "y_val": YV, "y_test": YT,
         "p_val": predict(m, XV), "p_test": predict(m, XT)}
    np.savez(os.path.join(a.out, f"cnn_features_{a.ds}.npz"), **P)
    log(f"features train {P['f_train'].shape} val {P['f_val'].shape} test {P['f_test'].shape} | frozen classifier itself: "
        f"val AUC {auc(P['p_val'], YV):.3f} test AUC {auc(P['p_test'], YT):.3f}")
    raise SystemExit(0)

class SmallCNN(nn.Module):                                 # the poster's "Classical CNN"
    def __init__(self):
        super().__init__(); L = []; c = 3
        for o in (32, 64, 128, 256):
            L += [nn.Conv2d(c, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1),
                  nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2)]; c = o
        self.f = nn.Sequential(*L); self.h = nn.Linear(256, 2)
    def forward(self, x):
        return self.h(F.adaptive_avg_pool2d(self.f(x), 1).flatten(1))
def make(arch):
    import torchvision
    if arch == "smallcnn":
        return SmallCNN(), 1e-3
    if arch == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2"); m.fc = nn.Linear(m.fc.in_features, 2); return m, 1e-4
    if arch == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1"); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2); return m, 1e-4
    return build(), 1e-4

def train(arch, seed, epochs, bs=32):                      # = a6_classifier.train_clf with a selectable architecture
    Xg, Yg = torch.from_numpy(XF).to(dev), torch.from_numpy(YF).long().to(dev)
    w = torch.tensor([len(YF) / (2 * max(1, (YF == c).sum())) for c in (0, 1)], device=dev, dtype=torch.float32)
    torch.manual_seed(seed); m, lr = make(arch); m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.05)
    spe = math.ceil(len(YF) / bs); total = spe * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / spe) * 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler("cuda"); g = torch.Generator(device=dev).manual_seed(seed)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(YF), generator=torch.Generator().manual_seed(seed * 100 + ep)).to(dev); tl = 0.0
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            xb = augment(Xg[idx].float().div(255).unsqueeze(1), g)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(m(prep(xb)).float(), Yg[idx], weight=w)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += float(loss.detach())
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"{arch} seed {seed} epoch {ep + 1}/{epochs} loss {tl / spe:.4f}")
    return m

for arch in a.archs.split(","):
    for seed in [int(s) for s in a.seeds.split(",")]:
        f = os.path.join(a.out, f"pred_{a.ds}_cnn_{arch}_s{seed}.npz")
        if os.path.exists(f):
            log(f"{f} exists -> skipped"); continue
        t0 = time.time(); m = train(arch, seed, a.epochs).eval()
        pv, pt = predict(m, XV), predict(m, XT); np.savez(f, val=pv, test=pt, y_val=YV, y_test=YT)
        log(f"{arch} seed {seed}: params {sum(p.numel() for p in m.parameters()) / 1e6:.1f} M | val AUC {auc(pv, YV):.3f} | "
            f"test AUC {auc(pt, YT):.3f} | {(time.time() - t0) / 60:.1f} min")
        del m; torch.cuda.empty_cache()
