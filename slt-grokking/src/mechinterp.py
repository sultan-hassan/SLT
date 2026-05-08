"""
Mechanistic interpretability helpers for the grokking transformer.

Extracts readout-slot activations (h[:, -1, :]) across the full dataset at
different training phases, then projects to 2D via PCA for visualisation.

The key argument: activation clusters look structurally similar at the
memorisation plateau AND post-grokking, even though test accuracy jumps from
~20% to ~100% between the two phases.  The loss landscape (LLC) discriminates;
activation geometry alone cannot.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def get_readout_activations(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full loader through the model, collect the final-layer residual-stream
    vector at the readout (=) position before the unembedding head.

    Returns
    -------
    acts   : (N, d_model)  readout activations
    a_vals : (N,)          first operand
    b_vals : (N,)          second operand
    labels : (N,)          correct answer (a+b) mod p
    """
    model.eval()
    acts_list, a_list, b_list, c_list = [], [], [], []

    # Hook the norm → head input (h[:, -1, :] after LayerNorm, before Linear)
    readout_buf: list[torch.Tensor] = []

    def _hook(module, input, output):
        # input[0] is the tensor going into the Linear head; shape (B, d_model)
        readout_buf.append(input[0].detach().cpu())

    handle = model.head.register_forward_hook(_hook)

    for a, b, c in loader:
        a, b, c = a.to(device), b.to(device), c.to(device)
        model(a, b)   # forward pass triggers hook
        a_list.append(a.cpu())
        b_list.append(b.cpu())
        c_list.append(c.cpu())

    handle.remove()
    model.train()

    acts   = torch.cat(readout_buf, dim=0).numpy()
    a_vals = torch.cat(a_list).numpy()
    b_vals = torch.cat(b_list).numpy()
    labels = torch.cat(c_list).numpy()
    return acts, a_vals, b_vals, labels


@torch.no_grad()
def fourier_frequency_amplitudes(model: nn.Module) -> np.ndarray:
    """
    Compute the Fourier frequency amplitudes of the token embedding matrix.

    For modular arithmetic (a+b) mod p, the model learns to represent tokens
    using Fourier features.  Post-grokking, only ~10 frequencies remain active
    in the embedding (Nanda et al. 2023).  Tracking this over training shows
    which frequencies emerge and when — a purely mechanistic-interpretability
    signal that we compare against the SLT Hessian-LLC.

    Returns
    -------
    amplitudes : (p//2,) float array
        Amplitude of each frequency k=1..p//2.  Frequency k is "active" when
        its amplitude stands significantly above the rest.
    """
    p = model.p
    # Only the first p rows are token embeddings (row p is the readout token)
    W = model.embed.weight[:p].detach().cpu().float().numpy()   # (p, d_model)

    # DFT along the token axis for each embedding dimension
    W_hat = np.fft.fft(W, axis=0)  # (p, d_model), complex

    # Amplitude of frequency k = RMS magnitude across all d_model dims, normalised
    amps = (np.abs(W_hat) ** 2).sum(axis=1) ** 0.5 / p  # (p,)
    return amps[1 : p // 2 + 1].astype(float)            # (p//2,), drop DC and mirror


def pca2d(acts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean-centre and project activations to the top-2 PCs.

    Returns
    -------
    proj     : (N, 2)  2-D projection
    var_frac : (2,)    fraction of variance explained by each PC
    """
    X = acts - acts.mean(axis=0)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    var = s ** 2
    proj = X @ Vt[:2].T
    return proj, var[:2] / var.sum()
