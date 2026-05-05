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
