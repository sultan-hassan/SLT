"""
Main experiment: track LLC across the grokking phase transition.

Usage:
    python train.py                     # full run (p=97, ~25 min on CPU)
    python train.py --quick             # fast demo (p=23, ~3 min on CPU)
    python train.py --p 113 --steps 8000

Produces figures/ with four publication-quality plots.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.model import ModularTransformer, count_params
from src.data import make_dataloaders
from src.llc import estimate_llc
from src.viz import (
    plot_training_dynamics,
    plot_llc_trajectory,
    plot_phase_portrait,
    plot_model_comparison,
    plot_sgld_diagnostics,
)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for a, b, c in loader:
        a, b, c = a.to(device), b.to(device), c.to(device)
        logits = model(a, b)
        total_loss += criterion(logits, c).item() * len(a)
        correct += (logits.argmax(-1) == c).sum().item()
        count += len(a)
    model.train()
    return total_loss / count, correct / count


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"\n{'='*60}")
    print(f"SLT × Grokking experiment")
    print(f"  p={cfg.p}, d={cfg.d_model}, heads={cfg.n_heads}, layers={cfg.n_layers}")
    print(f"  device={device}, steps={cfg.steps}")
    print(f"{'='*60}\n")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader, test_loader, full_loader = make_dataloaders(
        p=cfg.p, train_frac=cfg.train_frac, batch_size=cfg.batch_size, seed=cfg.seed
    )
    n_train = len(train_loader.dataset)

    model = ModularTransformer(
        p=cfg.p, d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers
    ).to(device)
    print(f"Model parameters: {count_params(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98)
    )
    criterion = nn.CrossEntropyLoss()

    metrics = {
        "eval_steps":  [],
        "train_loss":  [], "test_loss":  [],
        "train_acc":   [], "test_acc":   [],
        "llc_steps":   [],
        "llc":         [],
    }

    figures_dir = Path("figures")
    train_iter = _cycle(train_loader)
    t0 = time.time()

    for step in range(cfg.steps + 1):
        # ── evaluate ─────────────────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            tr_loss, tr_acc = evaluate(model, train_loader, criterion, device)
            te_loss, te_acc = evaluate(model, test_loader,  criterion, device)
            metrics["eval_steps"].append(step)
            metrics["train_loss"].append(tr_loss)
            metrics["test_loss"].append(te_loss)
            metrics["train_acc"].append(tr_acc)
            metrics["test_acc"].append(te_acc)
            elapsed = time.time() - t0
            print(f"step {step:>5d} | train {tr_loss:.3f}/{tr_acc:.2%} | "
                  f"test {te_loss:.3f}/{te_acc:.2%} | {elapsed:.0f}s")

        # ── LLC estimation ────────────────────────────────────────────────────
        if step % cfg.llc_interval == 0:
            print(f"  → estimating LLC at step {step} ...", end="", flush=True)
            llc, energies = estimate_llc(
                model, criterion, train_loader,
                n_steps=cfg.llc_steps, step_size=cfg.llc_step_size,
                localization=cfg.llc_localization,
                device=device, burnin=cfg.llc_burnin,
            )
            metrics["llc_steps"].append(step)
            metrics["llc"].append(llc)
            print(f"  λ̂ = {llc:.4f}")

            # Save SGLD diagnostics for a few checkpoints
            if step in {0, cfg.steps // 4, cfg.steps // 2, cfg.steps}:
                plot_sgld_diagnostics(energies, llc, step, save_path=str(figures_dir))

        if step == cfg.steps:
            break

        # ── training step ─────────────────────────────────────────────────────
        model.train()
        a, b, c = next(train_iter)
        a, b, c = a.to(device), b.to(device), c.to(device)

        optimizer.zero_grad()
        loss = criterion(model(a, b), c)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # ── Save metrics ──────────────────────────────────────────────────────────
    figures_dir.mkdir(exist_ok=True)
    with open(figures_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Figures 1–3 ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_training_dynamics(metrics, save_path=str(figures_dir))
    plot_llc_trajectory(metrics,    save_path=str(figures_dir))
    plot_phase_portrait(metrics,    save_path=str(figures_dir))

    # ── Figure 4: model comparison ────────────────────────────────────────────
    if cfg.model_sweep:
        print("Running model sweep for Figure 4...")
        sweep_results = []
        for d, layers in [(32, 1), (64, 1), (64, 2), (128, 2), (128, 4)]:
            print(f"  d_model={d}, n_layers={layers}")
            m = ModularTransformer(cfg.p, d_model=d, n_heads=min(4, d // 16 or 1),
                                   n_layers=layers).to(device)
            _quick_train(m, train_loader, criterion, device, n_steps=cfg.steps)
            te_loss, te_acc = evaluate(m, test_loader, criterion, device)
            llc_val, _ = estimate_llc(m, criterion, train_loader,
                                      n_steps=cfg.llc_steps, device=device)
            sweep_results.append({
                "d_model": d, "n_layers": layers,
                "llc_final": llc_val, "test_acc_final": te_acc,
            })
        plot_model_comparison(sweep_results, save_path=str(figures_dir))

    print(f"\nDone! All figures saved to {figures_dir}/")
    return metrics


def _quick_train(model, loader, criterion, device, n_steps=3000, lr=1e-3, wd=1.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    it = _cycle(loader)
    model.train()
    for _ in range(n_steps):
        a, b, c = next(it)
        a, b, c = a.to(device), b.to(device), c.to(device)
        opt.zero_grad()
        criterion(model(a, b), c).backward()
        opt.step()


def _cycle(loader):
    while True:
        yield from loader


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SLT × Grokking experiment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--quick",   action="store_true",
                   help="fast demo (p=23, 1-layer, 2000 steps)")
    p.add_argument("--delayed", action="store_true",
                   help=(
                       "classic delayed-grokking setup (p=97, 1-layer, 30%% train,\n"
                       "30 000 steps) — shows the memorisation plateau before\n"
                       "the sudden generalisation jump that SLT predicts.\n"
                       "~10 min on MPS."
                   ))
    p.add_argument("--p",            type=int,   default=97,    help="prime modulus")
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--n_heads",      type=int,   default=4)
    p.add_argument("--n_layers",     type=int,   default=2)
    p.add_argument("--train_frac",   type=float, default=0.5)
    p.add_argument("--steps",        type=int,   default=6000)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0,
                   help="high weight decay is essential for grokking")
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--eval_interval",type=int,   default=100)
    p.add_argument("--llc_interval", type=int,   default=300)
    p.add_argument("--llc_steps",    type=int,   default=200,
                   help="SGLD steps per LLC estimate")
    p.add_argument("--llc_step_size",type=float, default=3e-6)
    p.add_argument("--llc_burnin",       type=int,   default=50)
    p.add_argument("--llc_localization", type=float, default=10_000.0,
                   help="spring constant keeping SGLD near w* (0=global, >0=localised)")
    p.add_argument("--model_sweep",  action="store_true",
                   help="also run Figure 4 model comparison sweep")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    cfg = parse_args()

    if cfg.quick:
        cfg.p             = 23
        cfg.d_model       = 64
        cfg.n_heads       = 4
        cfg.n_layers      = 1
        cfg.steps         = 2000
        cfg.llc_interval  = 200
        cfg.llc_steps     = 100
        cfg.llc_burnin    = 20

    if cfg.delayed:
        # Classic delayed-grokking regime (Power et al. 2022 style):
        #   - 1-layer transformer — simpler, memorises easily but generalises slowly
        #   - 30% train fraction  — less data makes the generalising algorithm harder to find
        #   - 30 000 steps        — the memorisation plateau lasts thousands of steps
        #   - LLC estimated every 2000 steps (20 checkpoints total)
        cfg.p             = 97
        cfg.d_model       = 128
        cfg.n_heads       = 4
        cfg.n_layers      = 1
        cfg.train_frac    = 0.30
        cfg.steps         = 30_000
        cfg.eval_interval = 200
        cfg.llc_interval  = 500        # fine enough to see the transition
        cfg.llc_steps     = 200
        cfg.llc_burnin    = 50

    train(cfg)
