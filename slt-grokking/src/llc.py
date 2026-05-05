"""
Local Learning Coefficient (LLC) estimator via Stochastic Gradient Langevin Dynamics.

Theory (Watanabe 2009):
  The free energy near a singular point w* satisfies:
      nF_n(w*) ≈ nL_n(w*) + λ log(n) - (m-1) log(log(n)) + O(1)

  where λ is the Real Log Canonical Threshold (RLCT), the Local Learning Coefficient.
  A lower λ indicates a more degenerate (singular) loss landscape — a simpler effective
  model. SLT explains why overparameterised networks generalise: they find singular optima
  with small λ, conferring an implicit complexity penalty.

Estimator (SGLD-based, following devinterp conventions):
  We sample from the tempered posterior p_β(w) ∝ exp(-βnL_n(w)) using SGLD at
  inverse temperature β = 1/log(n), then estimate:

      λ ≈ β · n · E_{w~p_β}[L_n(w) - L_n(w*)]

Localisation:
  A fixed step_size works well during early training but, once the model is in a sharp
  post-grokking minimum, the SGLD can escape the local basin and probe the global loss
  landscape rather than the local singular structure — biasing λ̂ upward.

  The remedy (following devinterp) is a *localisation* quadratic prior centred at w*:

      U_loc(w) = β·n·L_n(w)  +  (γ/2) · ‖w − w*‖²

  The extra spring force  γ·(w − w*)  keeps the chain near w*.
  This gives the *localised LLC* — the singularity depth of the local basin at w*.
  Set γ=0 to recover the original estimator.

Per-batch baseline:
  Both L(w_t) and L(w*) are evaluated on the same mini-batch so that mini-batch
  sampling variance cancels out of the energy difference.  Using a fixed full-dataset
  baseline causes the measured energies to track mini-batch variance rather than
  curvature once training loss → 0.
"""

import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


def estimate_llc(
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    n_steps: int = 200,
    step_size: float = 3e-5,
    beta: Optional[float] = None,
    localization: float = 100.0,
    device: str = "cpu",
    burnin: int = 50,
) -> tuple[float, list[float]]:
    """
    Estimate the Local Learning Coefficient at the current weights via localised SGLD.

    Key parameter choices for transformers on modular arithmetic:
      step_size    3e-5   — larger than before so the chain explores a meaningful radius
      localization  100   — stationary std σ = 1/√loc ≈ 0.1 per parameter, large enough
                            for the chain to sense the difference between the memorisation
                            basin (sharp → high LLC) and the grokking basin (flat → low LLC).
                            The previous default of 10,000 gave σ=0.01, which only probed
                            the local Hessian and missed the basin-level geometry change.

    Parameters
    ----------
    model         : network at its current weights w*
    criterion     : loss function (reduction='mean')
    train_loader  : dataloader for the training set
    n_steps       : SGLD steps to collect after burnin
    step_size     : SGLD step size ε
    beta          : inverse temperature; defaults to 1/log(n_train) — the WBIC choice
    localization  : γ ≥ 0, spring constant keeping SGLD near w*.
                    Stationary std per parameter = 1/√γ.
                    Typical range 10–1000; tune so SGLD energy trace is stationary.
    burnin        : initial SGLD steps discarded to allow mixing

    Returns
    -------
    llc     : scalar LLC estimate λ̂
    energies: per-step (L_batch(w_t) − L_batch(w*)) values for diagnostics
    """
    n = len(train_loader.dataset)
    if beta is None:
        beta = 1.0 / math.log(n)

    sgld_model = copy.deepcopy(model).to(device)
    w_star = {name: p.data.clone() for name, p in sgld_model.named_parameters()}

    sgld_model.train()
    train_iter = _infinite(train_loader)
    energies: list[float] = []
    scale = beta * n          # gradient scale factor (per full dataset)

    for step in range(n_steps + burnin):
        a, b, c = next(train_iter)
        a, b, c = a.to(device), b.to(device), c.to(device)

        sgld_model.zero_grad()
        loss = criterion(sgld_model(a, b), c)
        loss.backward()

        batch_scale = scale / len(a)
        with torch.no_grad():
            for name, param in sgld_model.named_parameters():
                if param.grad is None:
                    continue
                spring = localization * (param.data - w_star[name])
                noise  = torch.randn_like(param) * math.sqrt(2.0 * step_size)
                param.data -= step_size * (batch_scale * param.grad + spring)
                param.data += noise

        if step >= burnin:
            with torch.no_grad():
                sgld_model.eval()
                # Per-batch baseline: evaluate L(w*) on the same batch so that
                # mini-batch variance cancels out of the energy difference.
                w_t = {n: p.data.clone() for n, p in sgld_model.named_parameters()}
                for nm, p in sgld_model.named_parameters():
                    p.data.copy_(w_star[nm])
                baseline_batch = criterion(sgld_model(a, b), c).item()
                for nm, p in sgld_model.named_parameters():
                    p.data.copy_(w_t[nm])
                perturbed = criterion(sgld_model(a, b), c).item()
                sgld_model.train()
                energies.append(max(0.0, perturbed - baseline_batch))

    llc = beta * n * float(np.mean(energies))
    return llc, energies


def _infinite(loader: DataLoader):
    while True:
        yield from loader
