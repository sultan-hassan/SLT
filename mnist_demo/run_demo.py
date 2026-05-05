#!/usr/bin/env python3
"""
MNIST Demo: Mechanistic Interpretability vs Singular Learning Theory
Talk: "Interpretability: Should we cache activations or monitor the loss landscape?"

Two training regimes are compared:
  MEMORIZER  — tiny dataset (200 samples), no regularisation
               → train acc ≈ 100%, test acc ≈ 60–70%  (classic overfit)
  GENERALIZER — full dataset (3 000 samples), weight-decay
               → train acc ≈ 98%,  test acc ≈ 95%

Run:
    python run_demo.py              # full analysis (~25 min on CPU)
    python run_demo.py --quick      # fast preview  (~6 min on CPU / MPS)
"""

import copy, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════

class SmallMLP(nn.Module):
    """
    784 → 512 → 256 → 10.
    No spatial inductive bias → memorises small MNIST cleanly,
    giving a sharp, narrow loss basin that contrasts with the generaliser.
    """
    def __init__(self, h1: int = 512, h2: int = 256):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

    def penultimate(self, x: torch.Tensor):
        """Returns (logits, fc2_hidden) for mech interp."""
        x = self.flatten(x)
        h = self.relu(self.fc2(self.relu(self.fc1(x))))
        return self.fc3(h), h


class SmallCNN(nn.Module):
    """Small 2-conv CNN — used only for filter visualisation."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc2(self.relu(self.fc1(x.flatten(1))))

    def penultimate(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        h = self.relu(self.fc1(x.flatten(1)))
        return self.fc2(h), h


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

def load_mnist(n_train: int, n_test: int, data_dir: str):
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        raise ImportError("pip install torchvision")

    tf = T.ToTensor()
    full_train = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    full_test  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=tf)

    train_idx = _balanced_idx(np.array(full_train.targets), n_train)
    test_idx  = _balanced_idx(np.array(full_test.targets),  n_test)

    tl = DataLoader(Subset(full_train, train_idx), batch_size=64, shuffle=True)
    vl = DataLoader(Subset(full_test,  test_idx),  batch_size=256, shuffle=False)
    return tl, vl


def _balanced_idx(targets: np.ndarray, n: int) -> list:
    per = n // 10
    idx = []
    for c in range(10):
        ci = np.where(targets == c)[0]
        idx.extend(np.random.choice(ci, min(per, len(ci)), replace=False).tolist())
    return idx


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def _eval(model, loader, crit, device):
    model.eval()
    loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            lg = model(x)
            loss    += crit(lg, y).item() * len(x)
            correct += (lg.argmax(1) == y).sum().item()
            n       += len(x)
    return loss / n, correct / n


def train_model(model, train_loader, test_loader, epochs, lr, wd, device, ckpt_epochs):
    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    metrics = dict(epochs=[], train_loss=[], test_loss=[], train_acc=[], test_acc=[])
    ckpts   = {}

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()

        tr_l, tr_a = _eval(model, train_loader, crit, device)
        te_l, te_a = _eval(model, test_loader,  crit, device)
        metrics["epochs"].append(ep)
        metrics["train_loss"].append(tr_l); metrics["test_loss"].append(te_l)
        metrics["train_acc"].append(tr_a);  metrics["test_acc"].append(te_a)

        if ep in ckpt_epochs:
            ckpts[ep] = copy.deepcopy(model.state_dict())
            print(f"    ep {ep:3d} | train {tr_a:.1%} / test {te_a:.1%} | "
                  f"loss {tr_l:.3f}/{te_l:.3f}")
    return metrics, ckpts


# ═══════════════════════════════════════════════════════════════════════════════
# SLT — LLC estimator (localised SGLD)
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_llc(model, train_loader, n_train, device,
                 n_steps=200, step_size=1e-4, localization=5.0, burnin=50):
    """
    Localised SGLD LLC estimator.

    Tuning for MLPs on small datasets (n≈200–3000):
      step_size   1e-4   — large enough that the chain explores the basin
      localization  5.0  — light spring; keeps chain in basin without trapping it

    Baseline is estimated per-step on the same mini-batch at w* (not once over
    the full dataset). This avoids spurious LLC inflation at late training epochs
    when the full-dataset loss → 0 but mini-batch variance remains positive: with
    a shared-batch baseline, energy = L_batch(w_t) − L_batch(w*) measures genuine
    curvature rather than sampling noise.
    """
    beta = 1.0 / math.log(n_train)
    crit = nn.CrossEntropyLoss()

    sgld   = copy.deepcopy(model).to(device)
    w_star = {n: p.data.clone() for n, p in sgld.named_parameters()}
    scale  = beta * n_train
    energies = []
    it = _cycle(train_loader)

    for step in range(n_steps + burnin):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        sgld.train(); sgld.zero_grad(); crit(sgld(x), y).backward()
        with torch.no_grad():
            for name, p in sgld.named_parameters():
                if p.grad is None: continue
                p.data -= step_size * (scale / len(x) * p.grad +
                                       localization * (p.data - w_star[name]))
                p.data += torch.randn_like(p) * math.sqrt(2.0 * step_size)
        if step >= burnin:
            with torch.no_grad():
                sgld.eval()
                # Per-step baseline: L(w*) on the same batch — cancels mini-batch
                # variance so the energy measures curvature, not sampling noise.
                w_t = {n: p.data.clone() for n, p in sgld.named_parameters()}
                for name, p in sgld.named_parameters():
                    p.data.copy_(w_star[name])
                baseline_batch = crit(sgld(x), y).item()
                for name, p in sgld.named_parameters():
                    p.data.copy_(w_t[name])
                e = max(0.0, crit(sgld(x), y).item() - baseline_batch)
                energies.append(e)

    return beta * n_train * float(np.mean(energies)), energies


def _cycle(loader):
    while True:
        yield from loader


# ═══════════════════════════════════════════════════════════════════════════════
# SLT — Loss landscape
# ═══════════════════════════════════════════════════════════════════════════════

def _fndir(model):
    """Filter-normalised random direction (Li et al. 2018)."""
    dirs = []
    for p in model.parameters():
        d = torch.randn_like(p)
        if p.dim() >= 2:
            norms_p = p.reshape(p.shape[0], -1).norm(dim=1)
            norms_d = d.reshape(d.shape[0], -1).norm(dim=1) + 1e-10
            s = (norms_p / norms_d).reshape(p.shape[0], *([1] * (p.dim() - 1)))
            d = d * s
        dirs.append(d)
    return dirs


def _perturb(m, p0, d1, d2, a, b):
    for p, v0, v1, v2 in zip(m.parameters(), p0, d1, d2):
        p.data.copy_(v0 + a * v1 + b * v2)


def _perturb1d(m, p0, d, a):
    for p, v0, v in zip(m.parameters(), p0, d):
        p.data.copy_(v0 + a * v)


def _restore(m, p0):
    for p, v0 in zip(m.parameters(), p0):
        p.data.copy_(v0)


def land2d(model, x_ref, y_ref, d1, d2, n_grid=25, span=1.0):
    """2D loss surface using pre-computed filter-normalised directions."""
    crit   = nn.CrossEntropyLoss()
    p0     = [p.data.clone() for p in model.parameters()]
    alphas = np.linspace(-span, span, n_grid)
    betas  = np.linspace(-span, span, n_grid)
    Z      = np.zeros((n_grid, n_grid))
    model.eval()
    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                _perturb(model, p0, d1, d2, a, b)
                Z[i, j] = crit(model(x_ref), y_ref).item()
    _restore(model, p0)
    return alphas, betas, Z


def land1d(model, x_ref, y_ref, n_pts=50, span=1.5, n_dirs=4):
    """1D ΔL scans along n_dirs filter-normalised random directions.

    Returns (alphas, {dir_k: delta_loss_array}) where delta_loss = L(w*+αd) − L(w*).
    Using ΔL makes the two models directly comparable regardless of their different
    baseline loss values (memoriser has ~0.85 test loss; generaliser ~0.25).
    """
    crit   = nn.CrossEntropyLoss()
    p0     = [p.data.clone() for p in model.parameters()]
    alphas = np.linspace(-span, span, n_pts)
    # baseline loss at w* (α=0)
    model.eval()
    with torch.no_grad():
        baseline = crit(model(x_ref), y_ref).item()
    res    = {}
    with torch.no_grad():
        for k in range(n_dirs):
            d = _fndir(model)
            losses = []
            for a in alphas:
                _perturb1d(model, p0, d, a)
                losses.append(crit(model(x_ref), y_ref).item())
            res[f"dir{k+1}"] = np.array(losses) - baseline
    _restore(model, p0)
    return alphas, res


def ref_batch(loader, device, n=512):
    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)
        if sum(t.shape[0] for t in xs) >= n: break
    return torch.cat(xs)[:n].to(device), torch.cat(ys)[:n].to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# Mech Interp — activations
# ═══════════════════════════════════════════════════════════════════════════════

def get_acts(model, loader, device, max_n=600):
    model.eval()
    acts, labs, total = [], [], 0
    with torch.no_grad():
        for x, y in loader:
            if total >= max_n: break
            _, h = model.penultimate(x.to(device))
            acts.append(h.cpu().numpy()); labs.append(y.numpy()); total += len(x)
    return np.concatenate(acts)[:max_n], np.concatenate(labs)[:max_n]


def pca2d(X):
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, s[:2] ** 2 / (s ** 2).sum()


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

_S = dict(**{"axes.spines.top": False, "axes.spines.right": False},
          **{"axes.grid": True, "grid.alpha": 0.2},
          **{"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
_C10 = plt.cm.tab10(np.linspace(0, 1, 10))


def _st():
    plt.rcParams.update(_S)


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Dual training dynamics (the gap)
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_dual_training(m_memo, m_gen, out):
    """
    Two training curves on the same axes.
    The gap between memoriser and generaliser test accuracy is the central point.
    """
    _st()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Two Training Regimes: Memorisation vs Generalisation\n"
        "Same architecture — different dataset size and regularisation",
        fontsize=13, fontweight="bold",
    )

    for ax, k_tr, k_te, ylabel, title in [
        (axes[0], "train_acc",  "test_acc",  "Accuracy", "Accuracy"),
        (axes[1], "train_loss", "test_loss", "Loss",      "Loss"),
    ]:
        ax.plot(m_memo["epochs"], m_memo[k_tr], color="#E91E63", lw=2.0, ls="-",
                label="Memoriser train", alpha=0.9)
        ax.plot(m_memo["epochs"], m_memo[k_te], color="#E91E63", lw=2.0, ls="--",
                label="Memoriser test", alpha=0.9)
        ax.plot(m_gen["epochs"],  m_gen[k_tr],  color="#2196F3", lw=2.0, ls="-",
                label="Generaliser train", alpha=0.9)
        ax.plot(m_gen["epochs"],  m_gen[k_te],  color="#2196F3", lw=2.0, ls="--",
                label="Generaliser test", alpha=0.9)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=9)

    # Annotate final gap
    ax = axes[0]
    final_memo_test = m_memo["test_acc"][-1]
    final_gen_test  = m_gen["test_acc"][-1]
    x_gap = max(m_memo["epochs"][-1], m_gen["epochs"][-1])
    ax.annotate(
        f"Test gap ≈ {(final_gen_test - final_memo_test):.0%}",
        xy=(x_gap * 0.7, (final_memo_test + final_gen_test) / 2),
        fontsize=10, color="#555", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", alpha=0.85),
    )

    plt.tight_layout()
    _save(fig, out / "fig1_dual_training.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Activation PCA: 3-phase temporal + cross-model comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_activation_pca(panels, out):
    """
    panels: list of (title, subtitle, acts, labs)
    Shows cluster evolution — both models form digit structure,
    but loss-landscape quality differs dramatically.
    """
    _st()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.0))
    fig.suptitle(
        "Mech Interp — Penultimate Layer Activations (PCA)\n"
        "Both regimes form digit clusters — activations alone can't tell memoriser from generaliser",
        fontsize=12, fontweight="bold",
    )

    for ax, panel in zip(axes, panels):
        title, subtitle, acts, labs, acc = panel
        proj, var = pca2d(acts)
        for c in range(10):
            m = labs == c
            ax.scatter(proj[m, 0], proj[m, 1], c=[_C10[c]], s=14, alpha=0.75)
        ax.set_title(f"{title}\n{subtitle}  ·  test {acc:.1%}", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_xticks([]); ax.set_yticks([])
        var_txt = f"PC1 {var[0]:.1%}  PC2 {var[1]:.1%}"
        ax.text(0.97, 0.02, var_txt, transform=ax.transAxes,
                fontsize=8, ha="right", color="#777")

    handles = [plt.scatter([], [], c=[_C10[c]], s=22) for c in range(10)]
    fig.legend(handles, [str(c) for c in range(10)],
               title="Digit", loc="center right",
               bbox_to_anchor=(1.0, 0.5), fontsize=9)
    plt.tight_layout()
    _save(fig, out / "fig2_activation_pca.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Mech Interp: Conv1 filter evolution (CNN bonus)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_filters(filt_by_ep, out):
    _st()
    eps = sorted(filt_by_ep)
    fig, axes = plt.subplots(len(eps), 16, figsize=(16, len(eps) * 1.3 + 0.8))
    fig.suptitle(
        "Mech Interp (CNN) — Conv1 Filter Evolution\n"
        "Filters grow structured, but WHEN exactly is ambiguous from activations alone",
        fontsize=12, fontweight="bold",
    )
    for row, ep in enumerate(eps):
        w = filt_by_ep[ep]
        for col in range(16):
            ax = axes[row, col] if len(eps) > 1 else axes[col]
            wi = w[col, 0].numpy()
            wi = (wi - wi.min()) / (wi.max() - wi.min() + 1e-8)
            ax.imshow(wi, cmap="RdBu_r", vmin=0, vmax=1); ax.axis("off")
        ax0 = axes[row, 0] if len(eps) > 1 else axes[0]
        ax0.set_ylabel(f"Ep {ep}", fontsize=9, rotation=90, va="center", labelpad=38)
    plt.tight_layout()
    _save(fig, out / "fig3_cnn_filters.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — SLT: LLC trajectories for both models
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_llc_dual(llc_memo, metrics_memo, n_memo,
                   llc_gen,  metrics_gen,  n_gen, out):
    """
    3-panel LLC figure.
    Left/centre: each model's LLC trajectory vs test accuracy.
    Right: normalised LLC = λ̂ / (β·n) = mean energy gap E[ΔL], directly comparable
           across different training-set sizes because β·n cancels out.
    """
    _st()
    import math as _math
    beta_m = 1.0 / _math.log(n_memo)
    beta_g = 1.0 / _math.log(n_gen)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    fig.suptitle(
        "SLT — Local Learning Coefficient (LLC) Over Training\n"
        "Raw LLC scales with n (right panel normalises to allow cross-model comparison)",
        fontsize=12, fontweight="bold",
    )

    for ax, llc_data, metrics, label, c_llc, c_acc in [
        (axes[0], llc_memo, metrics_memo, "Memoriser (n=200)",   "#E91E63", "#FF8F00"),
        (axes[1], llc_gen,  metrics_gen,  "Generaliser (n=3000)", "#1565C0", "#2E7D32"),
    ]:
        eps_arr  = np.array(metrics["epochs"])
        test_acc = np.array(metrics["test_acc"])
        le    = [e for e, _ in llc_data]
        lv    = [v for _, v in llc_data]
        acc_i = np.interp(le, eps_arr, test_acc)

        # Raw estimates as dots; LOWESS trend so the signal isn't buried in SGLD noise
        ax.scatter(le, lv, color=c_llc, s=28, zorder=3, label="LLC λ̂ (raw)")
        if len(le) >= 4:
            from scipy.signal import savgol_filter as _sg
            win = min(len(le) if len(le) % 2 == 1 else len(le) - 1, 5)
            trend = _sg(lv, window_length=win, polyorder=min(2, win - 1))
            ax.plot(le, trend, color=c_llc, lw=2.5, label="LLC λ̂ (trend)")
        ax.fill_between(le, lv, alpha=0.10, color=c_llc)
        ax.set_xlabel("Epoch"); ax.set_ylabel("LLC λ̂", color=c_llc)
        ax.tick_params(axis="y", labelcolor=c_llc)

        ax2 = ax.twinx()
        ax2.plot(le, acc_i, color=c_acc, lw=2, ls="--", marker="s", ms=4,
                 label="Test acc")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Test accuracy", color=c_acc)
        ax2.tick_params(axis="y", labelcolor=c_acc)
        ax2.spines["right"].set_visible(True)

        l1, la1 = ax.get_legend_handles_labels()
        l2, la2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=9)
        ax.set_title(label)

    # ── Right panel: normalised LLC = E[ΔL] ──────────────────────────────────
    ax = axes[2]
    le_m  = [e for e, _ in llc_memo]; lv_m = [v for _, v in llc_memo]
    le_g  = [e for e, _ in llc_gen];  lv_g = [v for _, v in llc_gen]
    norm_m = [v / (beta_m * n_memo) for v in lv_m]
    norm_g = [v / (beta_g * n_gen)  for v in lv_g]

    ax.scatter(le_m, norm_m, color="#E91E63", s=28, zorder=3)
    ax.scatter(le_g, norm_g, color="#1565C0", marker="s", s=28, zorder=3)
    if len(le_m) >= 4:
        from scipy.signal import savgol_filter as _sg
        win_m = min(len(le_m) if len(le_m) % 2 == 1 else len(le_m) - 1, 5)
        ax.plot(le_m, _sg(norm_m, win_m, min(2, win_m-1)), color="#E91E63", lw=2.5, label="Memoriser")
    else:
        ax.plot(le_m, norm_m, color="#E91E63", lw=2.5, label="Memoriser")
    if len(le_g) >= 4:
        from scipy.signal import savgol_filter as _sg
        win_g = min(len(le_g) if len(le_g) % 2 == 1 else len(le_g) - 1, 5)
        ax.plot(le_g, _sg(norm_g, win_g, min(2, win_g-1)), color="#1565C0", lw=2.5, label="Generaliser")
    else:
        ax.plot(le_g, norm_g, color="#1565C0", lw=2.5, label="Generaliser")
    ax.set_xlabel("Epoch"); ax.set_ylabel("E[ΔL]  =  LLC / (β·n)")
    ax.set_title("Normalised LLC — mean energy gap\n(comparable across training-set sizes)")
    ax.legend(fontsize=9)
    ax.text(0.5, 0.96,
            "Higher E[ΔL] = more loss increase per perturbation = sharper basin",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, style="italic",
            color="#555")

    plt.tight_layout()
    _save(fig, out / "fig4_llc_dual.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — SLT: 1D landscape scans — sharp vs flat
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_1d(memo_ep, gen_ep, memo_1d, gen_1d, out):
    _st()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle(
        "SLT — 1D Loss Landscape Scans (same test set, ΔL from w*): Sharp Basin vs Flat Basin\n"
        "Each curve: ΔL(w* + αd) along one filter-normalised direction  |  α=0 → w*  |  ΔL=0 at origin by construction",
        fontsize=11, fontweight="bold",
    )
    pal = ["#E91E63", "#9C27B0", "#3F51B5", "#00BCD4"]
    for ax, (alphas, res), ep, title, bg, note in [
        (axes[0], memo_1d, memo_ep,
         f"Memoriser  (epoch {memo_ep})", "#FFF0F0",
         "⚠ Tall narrow walls\n→ fragile, brittle solution\nAny perturbation sharply\nincreases test loss"),
        (axes[1], gen_1d, gen_ep,
         f"Generaliser (epoch {gen_ep})", "#F0FFF4",
         "✓ Shallow wide basin\n→ robust invariant circuit\nPerturbations cause only\nsmall test loss increase"),
    ]:
        ax.set_facecolor(bg)
        for (k, v), c in zip(res.items(), pal):
            ax.plot(alphas, v, color=c, lw=2.0, alpha=0.9)
        ax.axvline(0, color="black", lw=1.5, ls="--", alpha=0.6, label="w*")
        ax.axhline(0, color="gray",  lw=0.8, ls=":",  alpha=0.5)
        ax.set_xlabel("Perturbation α  (filter-normalised)")
        ax.set_ylabel("ΔL = L(w* + αd) − L(w*)  [test set]")
        ax.set_title(title)
        ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=10,
                va="top", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    plt.tight_layout()
    _save(fig, out / "fig5_1d_landscapes.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — SLT: 2D landscape temporal evolution (memoriser trajectory)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_2d_evolution(panels, out):
    """panels: list of (label, epoch, test_acc, alphas, betas, Z)"""
    _st()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 5))
    if n == 1: axes = [axes]
    fig.suptitle(
        "SLT — 2D Loss Landscape Evolution (Memoriser Trajectory)\n"
        "Fixed filter-normalised directions — geometry is directly comparable",
        fontsize=12, fontweight="bold",
    )
    all_Z  = np.concatenate([Z.ravel() for _, _, _, _, _, Z in panels])
    vmin, vmax = np.percentile(all_Z, 2), np.percentile(all_Z, 96)
    im_ref = None
    for ax, (label, ep, acc, alphas, betas, Z) in zip(axes, panels):
        im = ax.contourf(alphas, betas, Z.T, levels=30, cmap="RdYlGn_r",
                         vmin=vmin, vmax=vmax)
        ax.contour(alphas, betas, Z.T, levels=15, colors="white",
                   linewidths=0.5, alpha=0.5)
        ax.plot(0, 0, "w*", ms=13, zorder=5)
        ax.set_xlabel("α  (dir₁)"); ax.set_ylabel("β  (dir₂)")
        ax.set_title(f"{label}\nEpoch {ep}  ·  test {acc:.1%}", fontsize=10)
        ax.set_aspect("equal"); im_ref = im
    if im_ref: fig.colorbar(im_ref, ax=axes, shrink=0.75, label="Loss L(w)")
    plt.tight_layout()
    _save(fig, out / "fig6_2d_landscapes.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — SLT: 3D landscape — THE SHOWSTOPPER
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_3d(memo_panel, gen_panel, out):
    """
    memo_panel, gen_panel: (label, epoch, test_acc, alphas, betas, Z)
    The single most compelling visual for the talk.
    """
    _st()
    fig = plt.figure(figsize=(16, 6.5))
    fig.suptitle(
        "SLT — 3D Loss Landscape: Sharp Basin (Memoriser) vs Flat Basin (Generaliser)\n"
        "The geometry of parameter space reveals what activations hide",
        fontsize=13, fontweight="bold",
    )

    for idx, (panel, cmap, note, elev, azim) in enumerate([
        (memo_panel, "YlOrRd",
         "Sharp narrow well\nSmall Δw → large loss spike\nFragile — not a real circuit",
         32, -50),
        (gen_panel,  "YlGn",
         "Flat wide valley\nPerturbations absorbed\nRobust invariant circuit found",
         32, -50),
    ]):
        label, ep, acc, alphas, betas, Z = panel
        A, B = np.meshgrid(alphas, betas)
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        surf = ax.plot_surface(A, B, Z.T, cmap=cmap, alpha=0.88,
                               linewidth=0, antialiased=True,
                               rcount=60, ccount=60)
        z_centre = Z[Z.shape[0] // 2, Z.shape[1] // 2]
        ax.scatter([0], [0], [z_centre], color="red", s=90, zorder=5, label="w*")
        ax.set_xlabel("α", labelpad=6); ax.set_ylabel("β", labelpad=6)
        ax.set_zlabel("Loss", labelpad=6)
        ax.set_title(f"{label}\nEpoch {ep}  ·  Test acc {acc:.1%}\n\n{note}",
                     fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.10, label="Loss")

    plt.tight_layout()
    _save(fig, out / "fig7_3d_landscapes.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 — The Key Comparison: same activations, different landscapes
# ═══════════════════════════════════════════════════════════════════════════════

def fig8_comparison(memo_acts_panel, gen_acts_panel,
                    memo_land_panel, gen_land_panel,
                    memo_edl, gen_edl, out):
    """
    2×2 layout: (Mech Interp row) × (Memoriser | Generaliser column)
                (SLT row)         × (Memoriser | Generaliser column)

    Core argument: PCA looks similar for both — the landscape is completely different.
    """
    _st()
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "The Core Argument — Two Lenses on the Same Phenomenon\n"
        "Activations say 'both learned'; the loss landscape says 'only one learned robustly'",
        fontsize=13, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.35,
                           top=0.94, bottom=0.06)

    all_Z  = np.concatenate([memo_land_panel[5].ravel(), gen_land_panel[5].ravel()])
    vmin   = np.percentile(all_Z, 2)
    vmax   = np.percentile(all_Z, 96)

    # ── Row 0: Activation PCA ─────────────────────────────────────────────
    for col, (label, sub, acts, labs, acc) in enumerate([
        memo_acts_panel, gen_acts_panel,
    ]):
        ax   = fig.add_subplot(gs[0, col])
        proj, var = pca2d(acts)
        for c in range(10):
            m = labs == c
            ax.scatter(proj[m, 0], proj[m, 1], c=[_C10[c]], s=13, alpha=0.75)
        ax.set_title(f"Mech Interp — {label}\n{sub} · test acc {acc:.1%}", fontsize=10.5)
        ax.set_xticks([]); ax.set_yticks([])
        note = ("Digit clusters look structured.\n"
                '"Model seems to have learned."\n'
                "⚠ Activations can't assess robustness.")
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=8.5,
                va="bottom", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", alpha=0.9))

    # ── Row 1: 2D loss landscape ──────────────────────────────────────────
    for col, (panel, edl_val, note, fc) in enumerate([
        (memo_land_panel, memo_edl,
         f"E[ΔL] = {memo_edl:.2f}  (normalised LLC)\n⚠ Sharp narrow basin\n"
         "Small perturbation = catastrophic loss\n→ Memorised, not understood",
         "#FFEBEE"),
        (gen_land_panel, gen_edl,
         f"E[ΔL] = {gen_edl:.2f}  (normalised LLC)\n✓ Flat wide basin\n"
         "Wide valley = solution is invariant\n→ Generalised circuit found",
         "#E8F5E9"),
    ]):
        label, ep, acc, alphas, betas, Z = panel
        ax = fig.add_subplot(gs[1, col])
        ax.contourf(alphas, betas, Z.T, levels=28, cmap="RdYlGn_r",
                    vmin=vmin, vmax=vmax)
        ax.contour(alphas, betas, Z.T, levels=14, colors="white",
                   linewidths=0.5, alpha=0.55)
        ax.plot(0, 0, "w*", ms=14, zorder=5)
        ax.set_xlabel("α  (dir₁)"); ax.set_ylabel("β  (dir₂)")
        ax.set_title(f"SLT — {label}\nEpoch {ep}  ·  test acc {acc:.1%}",
                     fontsize=10.5)
        ax.set_aspect("equal")
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=8.5,
                va="bottom", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=fc, alpha=0.92))

    handles = [plt.scatter([], [], c=[_C10[c]], s=22) for c in range(10)]
    fig.legend(handles, [str(c) for c in range(10)], title="Digit",
               loc="lower center", ncol=10, bbox_to_anchor=(0.5, 0.0), fontsize=9)
    _save(fig, out / "fig8_key_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Phase portrait + summary table
# ═══════════════════════════════════════════════════════════════════════════════

def fig9_portrait(llc_memo, metrics_memo, llc_gen, metrics_gen, out):
    _st()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "SLT — Phase Portrait and What Activation Caching Misses\n"
        "Trajectory in (test_loss, LLC) space reveals structure invisible to activation analysis",
        fontsize=12, fontweight="bold",
    )

    # ── Left: phase portrait for both models ──────────────────────────────
    ax = axes[0]
    for llc_data, metrics, label, cmap in [
        (llc_memo, metrics_memo, "Memoriser",   "Reds"),
        (llc_gen,  metrics_gen,  "Generaliser", "Blues"),
    ]:
        eps_a  = np.array(metrics["epochs"])
        tl_a   = np.array(metrics["test_loss"])
        le   = np.array([e for e, _ in llc_data])
        lv   = np.array([v for _, v in llc_data])
        tl_i = np.interp(le, eps_a, tl_a)

        colors = plt.get_cmap(cmap)(np.linspace(0.4, 0.9, len(le)))
        sc = ax.scatter(tl_i, lv, c=le, cmap=cmap, s=60,
                        edgecolors="white", lw=0.5, zorder=3, vmin=le[0], vmax=le[-1])
        ax.plot(tl_i, lv, color=colors[len(colors)//2], lw=0.8, alpha=0.5, zorder=2)

    ax.set_xlabel("Test loss"); ax.set_ylabel("LLC λ̂")
    ax.set_title("Phase Portrait  (coloured by epoch)")
    ax.text(0.5, 0.95, "← Memoriser trajectory\n← Generaliser trajectory",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            style="italic", color="#555")

    # ── Right: What each approach reveals ────────────────────────────────
    ax = axes[1]
    ax.axis("off")
    ax.set_title("What Each Lens Reveals", fontsize=12, fontweight="bold", pad=14)

    rows = [
        ("Method",                "Reveals",                               "Misses"),
        ("Mech Interp\n(activations)",
         "• Feature clusters / filters\n• Representation geometry\n• Which neurons activate",
         "• Basin sharpness / width\n• LLC / singularity depth\n• WHEN phase change happened"),
        ("SLT\n(loss landscape)",
         "• Sharp → flat transition\n• LLC trajectory signal\n• Phase portrait structure",
         "• Which neurons are active\n• Input→output feature map\n• Attention patterns"),
        ("Combined",
         "• Complete picture:\n  HOW + WHEN + WHY\n  structure emerges",
         "—  (complementary, not competing)"),
    ]
    xs   = [0.02, 0.35, 0.68]
    ys   = [0.94, 0.73, 0.44, 0.10]
    cols = ["#263238", "#1565C0", "#6A1B9A", "#2E7D32"]

    for ri, row in enumerate(rows):
        for ci, (val, x) in enumerate(zip(row, xs)):
            ax.text(x, ys[ri], val, transform=ax.transAxes,
                    fontsize=9, va="top", fontweight="bold" if ri == 0 else "normal",
                    color=cols[ri])

    ax.plot([0, 1], [0.67, 0.67], color="#CFD8DC", lw=1.0, transform=ax.transAxes,
            clip_on=False)

    plt.tight_layout()
    _save(fig, out / "fig9_phase_portrait.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def _ckpt_schedule(N):
    raw = sorted({1, 2, 5, N//5, N//3, N//2, 2*N//3, 4*N//5, N})
    return [e for e in raw if 1 <= e <= N]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",        default="./figures")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--llc_steps",  type=int, default=250)
    ap.add_argument("--land_grid",  type=int, default=28)
    ap.add_argument("--land_span",  type=float, default=1.0)
    ap.add_argument("--quick",      action="store_true",
                    help="Fewer epochs, coarser grid — preview in ~6 min on MPS/GPU.")
    args = ap.parse_args()

    if args.quick:
        args.llc_steps = 200
        args.land_grid = 18

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda"  if torch.cuda.is_available() else
              "mps"   if torch.backends.mps.is_available() else "cpu")
    out  = Path(args.out)
    data = str(Path(__file__).parent / "data")

    # ── Training configurations ──────────────────────────────────────────────
    # Memoriser: tiny dataset, no regularisation → classic overfit
    # Generaliser: full dataset, weight decay → genuine generalisation
    if args.quick:
        CFG = dict(
            memo = dict(n_train=200,  n_test=2000, epochs=70,  lr=5e-3, wd=0.0),
            gen  = dict(n_train=3000, n_test=2000, epochs=35,  lr=3e-3, wd=5e-3),
        )
    else:
        CFG = dict(
            memo = dict(n_train=200,  n_test=2000, epochs=150, lr=5e-3, wd=0.0),
            gen  = dict(n_train=3000, n_test=2000, epochs=70,  lr=3e-3, wd=5e-3),
        )

    # ── Train Memoriser ──────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f" MEMORISER:  {CFG['memo']['n_train']} samples · no weight-decay · "
          f"{CFG['memo']['epochs']} epochs")
    print(f"{'='*62}")
    tl_m, vl_m = load_mnist(CFG["memo"]["n_train"], CFG["memo"]["n_test"], data)
    ckpts_m_sched = set(_ckpt_schedule(CFG["memo"]["epochs"]))
    mdl_m = SmallMLP().to(device)
    metrics_m, ckpts_m = train_model(mdl_m, tl_m, vl_m,
                                     CFG["memo"]["epochs"], CFG["memo"]["lr"],
                                     CFG["memo"]["wd"], device, ckpts_m_sched)

    # ── Train Generaliser ────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f" GENERALISER: {CFG['gen']['n_train']} samples · wd={CFG['gen']['wd']} · "
          f"{CFG['gen']['epochs']} epochs")
    print(f"{'='*62}")
    tl_g, vl_g = load_mnist(CFG["gen"]["n_train"], CFG["gen"]["n_test"], data)
    ckpts_g_sched = set(_ckpt_schedule(CFG["gen"]["epochs"]))
    mdl_g = SmallMLP().to(device)
    metrics_g, ckpts_g = train_model(mdl_g, tl_g, vl_g,
                                     CFG["gen"]["epochs"], CFG["gen"]["lr"],
                                     CFG["gen"]["wd"], device, ckpts_g_sched)

    # Final state summaries
    m_tr_f = metrics_m["train_acc"][-1]; m_te_f = metrics_m["test_acc"][-1]
    g_tr_f = metrics_g["train_acc"][-1]; g_te_f = metrics_g["test_acc"][-1]
    print(f"\n  Memoriser  final:  train {m_tr_f:.1%} | test {m_te_f:.1%}")
    print(f"  Generaliser final: train {g_tr_f:.1%} | test {g_te_f:.1%}")

    # ── Figure 1: Dual training dynamics ────────────────────────────────────
    print("\n[3/7] Mech interp figures...")
    fig1_dual_training(metrics_m, metrics_g, out)

    # ── Activation PCA: 5 panels ─────────────────────────────────────────────
    # Memo early (epoch 1) | Memo mid (overfitting) | Memo final | Gen final
    eps_arr_m = np.array(metrics_m["epochs"])
    test_m    = np.array(metrics_m["test_acc"])
    train_m   = np.array(metrics_m["train_acc"])

    # Memoriser "peak overfitting" epoch: first where train > 90% AND test < 85%
    # (gap of >5pp signals clear overfitting; thresholds are lenient to handle easy MNIST)
    overfit_mask = np.where((train_m > 0.90) & (test_m < 0.85))[0]
    overfit_ep   = int(eps_arr_m[overfit_mask[0]]) if len(overfit_mask) else CFG["memo"]["epochs"] // 5
    overfit_ckpt = min(ckpts_m.keys(), key=lambda e: abs(e - overfit_ep))
    final_m_ckpt = max(ckpts_m.keys())
    final_g_ckpt = max(ckpts_g.keys())
    first_ckpt   = min(ckpts_m.keys())

    def _load_m(ckpts, ep, model_class=SmallMLP):
        m = model_class().to(device); m.load_state_dict(ckpts[ep]); return m

    pca_panels = []
    # Panel 1: Memo — init
    m_tmp = _load_m(ckpts_m, first_ckpt)
    a, l  = get_acts(m_tmp, vl_m, device)
    acc_  = float(np.interp(first_ckpt, eps_arr_m, test_m))
    pca_panels.append(("Memoriser — Init",
                        f"epoch {first_ckpt}", a, l, acc_))

    # Panel 2: Memo — overfitting peak
    m_tmp = _load_m(ckpts_m, overfit_ckpt)
    a, l  = get_acts(m_tmp, vl_m, device)
    acc_  = float(np.interp(overfit_ckpt, eps_arr_m, test_m))
    pca_panels.append(("Memoriser — Overfitting",
                        f"epoch {overfit_ckpt}  ← high train, low test", a, l, acc_))

    # Panel 3: Memo — final
    m_tmp = _load_m(ckpts_m, final_m_ckpt)
    a, l  = get_acts(m_tmp, vl_m, device)
    acc_  = m_te_f
    pca_panels.append(("Memoriser — Final",
                        f"epoch {final_m_ckpt}", a, l, acc_))

    # Panel 4: Gen — final
    m_tmp = _load_m(ckpts_g, final_g_ckpt)
    a, l  = get_acts(m_tmp, vl_g, device)
    acc_  = g_te_f
    pca_panels.append(("Generaliser — Final",
                        f"epoch {final_g_ckpt}", a, l, acc_))

    fig2_activation_pca(pca_panels, out)

    # ── CNN filter bonus ─────────────────────────────────────────────────────
    print("  Training small CNN for filter visualisation...")
    cnn = SmallCNN().to(device)
    crit_cnn = nn.CrossEntropyLoss()
    opt_cnn  = torch.optim.AdamW(cnn.parameters(), lr=3e-3, weight_decay=1e-3)
    filt_eps = {1: None, 5: None, 30: None}
    for ep in range(1, 31):
        cnn.train()
        for x, y in tl_g:
            x, y = x.to(device), y.to(device)
            opt_cnn.zero_grad(); crit_cnn(cnn(x), y).backward(); opt_cnn.step()
        if ep in filt_eps:
            filt_eps[ep] = cnn.conv1.weight.detach().cpu()
    fig3_filters({ep: w for ep, w in filt_eps.items() if w is not None}, out)

    # ── Reference batches for landscape ─────────────────────────────────────
    print("\n[4/7] Building reference batches...")
    x_ref_m, y_ref_m = ref_batch(tl_m, device, n=512)
    x_ref_g, y_ref_g = ref_batch(tl_g, device, n=512)

    # Fixed filter-normalised directions (computed from gen final for consistency)
    mdl_g.eval(); mdl_g.load_state_dict(ckpts_g[final_g_ckpt])
    fd1 = _fndir(mdl_g); fd2 = _fndir(mdl_g)

    # ── 2D landscapes for memoriser at 3 stages ──────────────────────────────
    print(f"\n[5/7] 2D landscapes (grid {args.land_grid}²)...")

    def _l2d(ckpts, ep, x_r, y_r, metrics_ref):
        m = _load_m(ckpts, ep)
        alphas, betas, Z = land2d(m, x_r, y_r, fd1, fd2,
                                   n_grid=args.land_grid, span=args.land_span)
        acc = float(np.interp(ep,
                               np.array(metrics_ref["epochs"]),
                               np.array(metrics_ref["test_acc"])))
        return acc, alphas, betas, Z

    # Memoriser trajectory panels
    memo_ep1  = first_ckpt
    memo_ep2  = overfit_ckpt
    memo_ep3  = final_m_ckpt
    land_panels_evo = []
    for ep, lbl in [(memo_ep1, "Init"), (memo_ep2, "Overfitting"), (memo_ep3, "Memorised")]:
        print(f"  Memoriser epoch {ep}...", end=" ", flush=True)
        acc, alphas, betas, Z = _l2d(ckpts_m, ep, x_ref_m, y_ref_m, metrics_m)
        land_panels_evo.append((f"Memoriser — {lbl}", ep, acc, alphas, betas, Z))
        print("done")

    # Generaliser final
    print(f"  Generaliser epoch {final_g_ckpt}...", end=" ", flush=True)
    acc_gf, alphas_gf, betas_gf, Z_gf = _l2d(ckpts_g, final_g_ckpt, x_ref_g, y_ref_g,
                                               metrics_g)
    land_gen_final = ("Generaliser — Final", final_g_ckpt, acc_gf, alphas_gf, betas_gf, Z_gf)
    print("done")

    sharp_panel_2d = land_panels_evo[-1]      # Memoriser final (overfit)
    flat_panel_2d  = land_gen_final           # Generaliser final

    # ── 1D landscapes — use a SHARED test batch so ΔL curves are comparable ──
    print("  1D scans...", end=" ", flush=True)
    x_ref_test, y_ref_test = ref_batch(vl_m, device, n=512)   # vl_m / vl_g share the same test set
    m_tmp = _load_m(ckpts_m, final_m_ckpt)
    als_m, res_m = land1d(m_tmp, x_ref_test, y_ref_test, n_pts=50, span=1.5)
    m_tmp = _load_m(ckpts_g, final_g_ckpt)
    als_g, res_g = land1d(m_tmp, x_ref_test, y_ref_test, n_pts=50, span=1.5)
    print("done")

    fig5_1d(final_m_ckpt, final_g_ckpt, (als_m, res_m), (als_g, res_g), out)
    fig6_2d_evolution(land_panels_evo + [land_gen_final], out)
    fig7_3d(sharp_panel_2d, flat_panel_2d, out)

    # ── LLC trajectories ─────────────────────────────────────────────────────
    print(f"\n[6/7] LLC estimation ({args.llc_steps} SGLD steps per checkpoint)...")
    llc_m_list, llc_g_list = [], []

    LLC_KW = dict(n_steps=args.llc_steps)   # step_size / localization use function defaults

    print("  Memoriser:")
    for ep in sorted(ckpts_m):
        m = _load_m(ckpts_m, ep)
        v, _ = estimate_llc(m, tl_m, CFG["memo"]["n_train"], device, **LLC_KW)
        llc_m_list.append((ep, v))
        print(f"    epoch {ep:3d}  LLC = {v:.2f}")

    print("  Generaliser:")
    for ep in sorted(ckpts_g):
        m = _load_m(ckpts_g, ep)
        v, _ = estimate_llc(m, tl_g, CFG["gen"]["n_train"], device, **LLC_KW)
        llc_g_list.append((ep, v))
        print(f"    epoch {ep:3d}  LLC = {v:.2f}")

    memo_llc_final = dict(llc_m_list).get(final_m_ckpt, float("nan"))
    gen_llc_final  = dict(llc_g_list).get(final_g_ckpt, float("nan"))

    # Normalised LLC = E[ΔL] = LLC / (β·n) — directly comparable across training sizes
    import math as _math
    beta_m = 1.0 / _math.log(CFG["memo"]["n_train"])
    beta_g = 1.0 / _math.log(CFG["gen"]["n_train"])
    memo_edl = memo_llc_final / (beta_m * CFG["memo"]["n_train"])
    gen_edl  = gen_llc_final  / (beta_g * CFG["gen"]["n_train"])

    fig4_llc_dual(llc_m_list, metrics_m, CFG["memo"]["n_train"],
                  llc_g_list, metrics_g, CFG["gen"]["n_train"], out)

    # ── Key comparison & summary ─────────────────────────────────────────────
    print("\n[7/7] Comparison and summary figures...")

    # Activation panels for comparison figure
    m_tmp = _load_m(ckpts_m, final_m_ckpt)
    a_m, l_m = get_acts(m_tmp, vl_m, device)
    m_tmp = _load_m(ckpts_g, final_g_ckpt)
    a_g, l_g = get_acts(m_tmp, vl_g, device)

    memo_acts_for_cmp = ("Memoriser", f"epoch {final_m_ckpt}", a_m, l_m, m_te_f)
    gen_acts_for_cmp  = ("Generaliser", f"epoch {final_g_ckpt}", a_g, l_g, g_te_f)

    fig8_comparison(memo_acts_for_cmp, gen_acts_for_cmp,
                    sharp_panel_2d, flat_panel_2d,
                    memo_edl, gen_edl, out)
    fig9_portrait(llc_m_list, metrics_m, llc_g_list, metrics_g, out)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"All figures → {out}/")
    print(f"{'='*62}")
    for f in sorted(out.glob("fig*.png")):
        print(f"  {f.name}")
    print()
    print("Key slides:")
    print("  fig7_3d_landscapes.png   ← THE SHOWSTOPPER (sharp vs flat 3D)")
    print("  fig8_key_comparison.png  ← main argument")
    print("  fig4_llc_dual.png        ← SLT signal")
    print("  fig1_dual_training.png   ← setup / the gap")
    print("  fig2_activation_pca.png  ← mech interp baseline")
    print()
    print(f"  Memoriser:   train {m_tr_f:.1%}  test {m_te_f:.1%}  "
          f"LLC={memo_llc_final:.1f}  E[ΔL]={memo_edl:.2f}")
    print(f"  Generaliser: train {g_tr_f:.1%}  test {g_te_f:.1%}  "
          f"LLC={gen_llc_final:.1f}  E[ΔL]={gen_edl:.2f}")


if __name__ == "__main__":
    main()
