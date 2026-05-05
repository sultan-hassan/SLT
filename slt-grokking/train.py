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
from src.hessian import hessian_trace, top_eigenvalue, loss_surface_2d
from src.mechinterp import get_readout_activations, pca2d
from src.viz import (
    plot_training_dynamics,
    plot_llc_trajectory,
    plot_phase_portrait,
    plot_model_comparison,
    plot_sgld_diagnostics,
    plot_flatness_trajectory,
    plot_loss_surfaces,
    plot_activation_phases,
    plot_two_lenses,
    plot_hessian_phases,
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
        "eval_steps":   [],
        "train_loss":   [], "test_loss":   [],
        "train_acc":    [], "test_acc":    [],
        "llc_steps":    [],
        "llc":          [],
        "htrace":       [],   # Hessian trace  (∑ eigenvalues = total curvature)
        "lambda_max":   [],   # largest Hessian eigenvalue (sharpest direction)
    }

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    # Shared random directions for loss surface panels (fixed seed → comparable plots)
    torch.manual_seed(99)
    _d = count_params(model)
    _surf_dir1 = torch.randn(_d)
    _surf_dir2 = torch.randn(_d)

    # Checkpoints saved for loss surface visualisation
    # Keys: training step → (state_dict, test_acc at that step)
    surface_ckpts: dict[int, dict] = {}
    # Save at 4 evenly-spaced steps across the run
    save_at = {0,
               cfg.steps // 3,
               2 * cfg.steps // 3,
               cfg.steps}

    # Phase checkpoints for mech interp figure:
    #   init (step 0), memorising, plateau, post-grokking
    # Detected dynamically; filled in as training progresses.
    phase_ckpts: dict[str, dict] = {}   # label → {state, step, train_acc, test_acc}
    _phase_flags = {"memorising": False, "plateau": False, "grokking": False}
    _memo_step: int = 0   # training step when Memorising was first detected

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

        # ── LLC + Hessian estimation ──────────────────────────────────────────
        if step % cfg.llc_interval == 0:
            print(f"  → step {step}: LLC ...", end="", flush=True)
            llc, energies = estimate_llc(
                model, criterion, train_loader,
                n_steps=cfg.llc_steps, step_size=cfg.llc_step_size,
                localization=cfg.llc_localization,
                device=device, burnin=cfg.llc_burnin,
            )
            metrics["llc_steps"].append(step)
            metrics["llc"].append(llc)
            print(f"  λ̂={llc:.1f}", end="", flush=True)

            print(f"  | Hessian ...", end="", flush=True)
            ht = hessian_trace(model, criterion, train_loader,
                               n_samples=cfg.htrace_samples,
                               delta=cfg.htrace_delta, device=device)
            lm = top_eigenvalue(model, criterion, train_loader,
                                n_iter=cfg.eig_iter,
                                delta=cfg.htrace_delta, device=device)
            metrics["htrace"].append(ht)
            metrics["lambda_max"].append(lm)
            print(f"  tr(H)={ht:.1f}  λ_max={lm:.2f}")

            if step in {0, cfg.steps // 4, cfg.steps // 2, cfg.steps}:
                plot_sgld_diagnostics(energies, llc, step, save_path=str(figures_dir))

        # ── Phase checkpoints for mech interp ────────────────────────────────
        if metrics["eval_steps"] and metrics["eval_steps"][-1] == step:
            tr_a = metrics["train_acc"][-1]
            te_a = metrics["test_acc"][-1]
            _snap = lambda lbl: {
                "state": {k: v.cpu() for k, v in model.state_dict().items()},
                "step": step, "train_acc": tr_a, "test_acc": te_a,
            }
            if step == 0:
                phase_ckpts["Init"] = _snap("Init")
            if not _phase_flags["memorising"] and tr_a > 0.95:
                phase_ckpts["Memorising"] = _snap("Memorising")
                _phase_flags["memorising"] = True
                _memo_step = step
            # Plateau: captured well into the plateau (≥ llc_interval steps after
            # Memorising) so it's a genuinely different checkpoint from Memorising.
            if (_phase_flags["memorising"] and not _phase_flags["plateau"]
                    and step >= _memo_step + cfg.llc_interval
                    and tr_a > 0.98 and te_a < 0.40):
                phase_ckpts["Plateau"] = _snap("Plateau")
                _phase_flags["plateau"] = True
            if not _phase_flags["grokking"] and te_a > 0.85:
                phase_ckpts["Post-grokking"] = _snap("Post-grokking")
                _phase_flags["grokking"] = True

        # ── Checkpoint for loss surface ───────────────────────────────────────
        if step in save_at:
            # record test acc at this step (interpolate from last eval)
            te_acc_now = metrics["test_acc"][-1] if metrics["test_acc"] else 0.0
            surface_ckpts[step] = {
                "state": {k: v.cpu() for k, v in model.state_dict().items()},
                "test_acc": te_acc_now,
            }

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
    with open(figures_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Figures 1–3 ───────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_training_dynamics(metrics, save_path=str(figures_dir))
    plot_llc_trajectory(metrics,    save_path=str(figures_dir))
    plot_phase_portrait(metrics,    save_path=str(figures_dir))

    # ── Figure 4 / 9: Hessian flatness figures ───────────────────────────────
    if metrics["htrace"]:
        plot_flatness_trajectory(metrics, save_path=str(figures_dir))
        plot_hessian_phases(metrics, save_path=str(figures_dir))

    # ── Figure 6: 2-D loss surface at 4 training stages ──────────────────────
    if surface_ckpts:
        print("  Computing loss surfaces (this takes ~30 s) ...", flush=True)
        panels = []
        for step_key in sorted(surface_ckpts):
            info = surface_ckpts[step_key]
            model.load_state_dict({k: v.to(device) for k, v in info["state"].items()})
            Z, alphas, betas = loss_surface_2d(
                model, criterion, train_loader,
                dir1=_surf_dir1.clone(), dir2=_surf_dir2.clone(),
                grid_size=cfg.surface_grid, extent=cfg.surface_extent,
                device=device, seed=99,
            )
            panels.append({
                "step": step_key,
                "test_acc": info["test_acc"],
                "Z": Z, "alphas": alphas, "betas": betas,
            })
        # Restore final weights
        model.load_state_dict(
            {k: v.to(device) for k, v in surface_ckpts[cfg.steps]["state"].items()}
        )
        plot_loss_surfaces(panels, save_path=str(figures_dir))

    # ── Figures 7–8: Mech Interp activation PCA ──────────────────────────────
    if len(phase_ckpts) >= 2:
        print("  Computing activation PCA for mech interp figures...")
        # Use the full dataset loader so both memorised and generalised examples appear
        _, _, full_loader = make_dataloaders(
            p=cfg.p, train_frac=cfg.train_frac, batch_size=cfg.batch_size, seed=cfg.seed
        )
        act_panels = []
        for label in ["Init", "Memorising", "Plateau", "Post-grokking"]:
            if label not in phase_ckpts:
                continue
            ckpt = phase_ckpts[label]
            model.load_state_dict({k: v.to(device) for k, v in ckpt["state"].items()})
            acts, _, _, labs = get_readout_activations(model, full_loader, device)
            proj, var = pca2d(acts)
            act_panels.append({
                "label": label,
                "step":      ckpt["step"],
                "train_acc": ckpt["train_acc"],
                "test_acc":  ckpt["test_acc"],
                "proj": proj, "labels": labs, "var": var,
            })
        # Restore final weights
        model.load_state_dict(
            {k: v.to(device) for k, v in phase_ckpts.get(
                "Post-grokking", list(phase_ckpts.values())[-1])["state"].items()}
        )
        if act_panels:
            plot_activation_phases(act_panels, cfg.p, save_path=str(figures_dir))
            plot_two_lenses(act_panels, metrics, metrics, cfg.p,
                            save_path=str(figures_dir))

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
                       "3 000 steps) — shows the full memorise → plateau → grokk arc.\n"
                       "~30 s on MPS."
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
    # Hessian / flatness options
    p.add_argument("--htrace_samples", type=int,   default=20,
                   help="Hutchinson samples for trace(H) estimate")
    p.add_argument("--htrace_delta",   type=float, default=1e-3,
                   help="finite-difference step for Hessian estimates")
    p.add_argument("--eig_iter",       type=int,   default=20,
                   help="power-iteration steps for top Hessian eigenvalue")
    # Loss surface options
    p.add_argument("--surface_grid",   type=int,   default=41,
                   help="grid resolution for 2-D loss surface (NxN points)")
    p.add_argument("--surface_extent", type=float, default=1.0,
                   help="±extent of the loss surface grid (filter-normalised units)")
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
        #   - 5 000 steps covers the full arc: memorise → plateau → grokk
        #   - LLC every 250 steps → ~20 estimates across the arc
        #   - localization=100 (σ=0.1/param) so the SGLD chain probes the full basin
        #     geometry and can detect the LLC drop during grokking.  The previous
        #     default of 10 000 (σ=0.01) only sampled local Hessian curvature.
        cfg.p                = 97
        cfg.d_model          = 128
        cfg.n_heads          = 4
        cfg.n_layers         = 1
        cfg.train_frac       = 0.30
        cfg.steps            = 5_000
        cfg.eval_interval    = 100
        cfg.llc_interval     = 250
        cfg.llc_steps        = 200
        cfg.llc_burnin       = 50
        cfg.llc_localization = 100.0
        cfg.llc_step_size    = 3e-5

    train(cfg)
