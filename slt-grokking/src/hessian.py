"""
Loss-surface geometry utilities: Hessian trace, top eigenvalue, 2D surface slices.

Connection to SLT:
  The Hessian H = ∇²L(w*) is the matrix of second derivatives of the loss.
  Its eigenvalue spectrum directly characterises the local geometry at w*:

    - Large eigenvalue   → high curvature, sharp direction.
    - Near-zero eigenvalue → flat direction, a degeneracy.

  In the REGULAR case (all eigenvalues > 0):  LLC λ = d/2  (maximum, all params effective).
  Each flat direction reduces λ below d/2.  So:

        fewer sharp Hessian eigenvalues  ←→  lower LLC  ←→  more singular minimum

  The LLC goes beyond the Hessian by capturing higher-order degeneracies (when the Hessian
  has a zero eigenvalue, the precise value of λ depends on 3rd/4th-order terms), but
  for practical networks `trace(H)` and `λ_max(H)` are reliable leading indicators.

All methods use finite differences or first-order gradients — fully MPS-safe.
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# Parameter helpers
# ──────────────────────────────────────────────────────────────────────────────

def _flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.cpu().flatten() for p in model.parameters()])


def _load_flat(model: nn.Module, flat: torch.Tensor):
    device = next(model.parameters()).device
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].view(p.shape).to(device))
        offset += n


@torch.no_grad()
def _full_loss(model: nn.Module, criterion: nn.Module,
               loader: DataLoader, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for a, b, c in loader:
        a, b, c = a.to(device), b.to(device), c.to(device)
        total += criterion(model(a, b), c).item() * len(a)
        count += len(a)
    return total / count


# ──────────────────────────────────────────────────────────────────────────────
# Hessian trace — Hutchinson estimator via finite differences
# ──────────────────────────────────────────────────────────────────────────────

def hessian_trace(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    n_samples: int = 30,
    delta: float = 1e-3,
    device: str = "cpu",
) -> float:
    """
    Estimate trace(H) using the Hutchinson finite-difference trick.

    trace(H) ≈ E_v [ (L(w+δv) - 2L(w) + L(w-δv)) / δ² ]   v ~ N(0, I)

    Each Rademacher/Gaussian probe gives an unbiased estimate of v^T H v.
    Taking the expectation over v gives trace(H) = Σ_i λ_i.

    Only forward passes required — MPS-safe.
    """
    w0 = _flat(model)
    L0 = _full_loss(model, criterion, loader, device)

    estimates = []
    for _ in range(n_samples):
        v = torch.randn_like(w0)
        _load_flat(model, w0 + delta * v)
        Lp = _full_loss(model, criterion, loader, device)
        _load_flat(model, w0 - delta * v)
        Lm = _full_loss(model, criterion, loader, device)
        estimates.append((Lp + Lm - 2.0 * L0) / (delta ** 2))

    _load_flat(model, w0)
    return float(np.mean(estimates))


# ──────────────────────────────────────────────────────────────────────────────
# Top Hessian eigenvalue — power iteration via gradient differences
# ──────────────────────────────────────────────────────────────────────────────

def _grad_flat(model: nn.Module, criterion: nn.Module,
               loader: DataLoader, device: str) -> torch.Tensor:
    """Full-dataset gradient, returned as a flat CPU tensor."""
    model.train()
    total_grad = None
    n_batches = 0
    for a, b, c in loader:
        a, b, c = a.to(device), b.to(device), c.to(device)
        model.zero_grad()
        criterion(model(a, b), c).backward()
        g = torch.cat([p.grad.cpu().flatten() for p in model.parameters()
                       if p.grad is not None])
        total_grad = g if total_grad is None else total_grad + g
        n_batches += 1
    model.zero_grad()
    return total_grad / n_batches


def top_eigenvalue(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    n_iter: int = 20,
    delta: float = 1e-3,
    device: str = "cpu",
) -> float:
    """
    Estimate the largest Hessian eigenvalue λ_max via power iteration.

    Each iteration computes the Hessian-vector product using gradient differences:
        H v ≈ (∇L(w + δv) − ∇L(w − δv)) / (2δ)

    Then normalises: v ← Hv / ‖Hv‖.  The eigenvalue estimate is ‖Hv‖.

    Requires first-order backward passes only — MPS-safe.
    """
    w0 = _flat(model)
    d = w0.numel()

    v = torch.randn(d)
    v = v / v.norm()
    eigenvalue = 0.0

    for _ in range(n_iter):
        _load_flat(model, w0 + delta * v)
        gp = _grad_flat(model, criterion, loader, device)
        _load_flat(model, w0 - delta * v)
        gm = _grad_flat(model, criterion, loader, device)

        Hv = (gp - gm) / (2.0 * delta)
        eigenvalue = Hv.norm().item()
        v = Hv / (eigenvalue + 1e-12)

    _load_flat(model, w0)
    return eigenvalue


# ──────────────────────────────────────────────────────────────────────────────
# 2-D loss surface slice
# ──────────────────────────────────────────────────────────────────────────────

def _filter_normalise(direction: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Scale a flat direction so each parameter-block has the same L2 norm as that
    parameter block in the model (Li et al. 2018 filter normalisation).
    Makes loss surfaces comparable across different weight scales.
    """
    d = direction.clone()
    offset = 0
    for p in model.parameters():
        n = p.numel()
        w_norm = p.data.cpu().norm().item() + 1e-12
        d_norm = d[offset:offset + n].norm().item() + 1e-12
        d[offset:offset + n] *= w_norm / d_norm
        offset += n
    return d


def loss_surface_2d(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    dir1: torch.Tensor | None = None,
    dir2: torch.Tensor | None = None,
    grid_size: int = 41,
    extent: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate loss on a 2-D grid around w* in two weight-space directions.

    Uses filter normalisation so the extent is scale-invariant across
    checkpoints and model sizes.  dir1/dir2 are shared across calls when
    you pass the same tensors — enabling direct panel-to-panel comparison.

    Returns
    -------
    Z      : (grid_size, grid_size) loss values
    alphas : (grid_size,) x-axis coordinates
    betas  : (grid_size,) y-axis coordinates
    """
    torch.manual_seed(seed)
    w0 = _flat(model)
    d = w0.numel()

    if dir1 is None:
        dir1 = torch.randn(d)
    if dir2 is None:
        dir2 = torch.randn(d)

    # Filter-normalise then orthogonalise
    dir1 = _filter_normalise(dir1.clone(), model)
    dir2 = _filter_normalise(dir2.clone(), model)
    dir1 = dir1 / dir1.norm()
    dir2 = dir2 - (dir2 @ dir1) * dir1
    dir2 = dir2 / (dir2.norm() + 1e-12)

    alphas = np.linspace(-extent, extent, grid_size)
    betas  = np.linspace(-extent, extent, grid_size)
    Z = np.zeros((grid_size, grid_size))

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                _load_flat(model, w0 + alpha * dir1 + beta * dir2)
                Z[i, j] = _full_loss(model, criterion, loader, device)
        _load_flat(model, w0)

    return Z, alphas, betas
