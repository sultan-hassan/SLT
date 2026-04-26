"""
Visualization utilities for the SLT grokking experiment.
Produces publication-quality figures with a clean, minimal style.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

_STYLE = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
}


def _apply_style():
    plt.rcParams.update(_STYLE)


def plot_training_dynamics(metrics: dict, save_path: str | None = None):
    """
    Figure 1: Training and test loss/accuracy curves.
    Highlights the grokking transition where test accuracy suddenly rises.
    """
    _apply_style()
    steps = metrics["eval_steps"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Grokking: Memorisation → Generalisation", fontsize=13, fontweight="bold")

    # Detect phase boundaries from test accuracy
    test_acc_arr = np.array(metrics["test_acc"])
    train_acc_arr = np.array(metrics["train_acc"])
    steps_arr = np.array(steps)

    # Memorisation: first step where train acc > 95%
    memo_idx = np.where(train_acc_arr > 0.95)[0]
    memo_step = int(steps_arr[memo_idx[0]]) if len(memo_idx) else None
    # Grokking: first step where test acc > 50%
    grokk_idx = np.where(test_acc_arr > 0.50)[0]
    grokk_step = int(steps_arr[grokk_idx[0]]) if len(grokk_idx) else None
    xmax = steps_arr[-1]

    for ax, ylabel, train_y, test_y, title, ylim in [
        (axes[0], "Cross-entropy loss", metrics["train_loss"], metrics["test_loss"],
         "Loss curves", None),
        (axes[1], "Accuracy", metrics["train_acc"], metrics["test_acc"],
         "Accuracy curves", (-0.05, 1.05)),
    ]:
        # Phase shading
        alpha = 0.08
        if memo_step:
            ax.axvspan(0, memo_step, alpha=alpha, color="#2196F3", zorder=0,
                       label="_memo")
        if memo_step and grokk_step and grokk_step > memo_step:
            ax.axvspan(memo_step, grokk_step, alpha=alpha, color="#FF9800", zorder=0)
        if grokk_step:
            ax.axvspan(grokk_step, xmax, alpha=alpha, color="#4CAF50", zorder=0)

        ax.plot(steps, train_y, label="Train", color="#2196F3", lw=1.5)
        ax.plot(steps, test_y,  label="Test",  color="#F44336", lw=1.5, ls="--")
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend()

    # Phase labels on loss panel
    if memo_step:
        axes[0].text(memo_step * 0.45, 0.97, "Memorise",
                     transform=axes[0].get_xaxis_transform(),
                     ha="center", fontsize=8, color="#1565C0", style="italic")
    if memo_step and grokk_step:
        axes[0].text((memo_step + grokk_step) / 2, 0.97, "Plateau",
                     transform=axes[0].get_xaxis_transform(),
                     ha="center", fontsize=8, color="#E65100", style="italic")
    if grokk_step:
        axes[0].text((grokk_step + xmax) / 2, 0.97, "Generalise",
                     transform=axes[0].get_xaxis_transform(),
                     ha="center", fontsize=8, color="#1B5E20", style="italic")

    plt.tight_layout()
    _save(fig, save_path, "fig1_training_dynamics.png")
    return fig


def plot_llc_trajectory(metrics: dict, save_path: str | None = None):
    """
    Figure 2: LLC trajectory overlaid with test accuracy.
    The key SLT plot — shows LLC as a signature of the phase transition.
    """
    _apply_style()
    llc_steps = metrics["llc_steps"]
    llcs      = metrics["llc"]

    # Interpolate accuracy at LLC checkpoints
    eval_steps = np.array(metrics["eval_steps"])
    test_acc   = np.array(metrics["test_acc"])
    acc_at_llc = np.interp(llc_steps, eval_steps, test_acc)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        "Local Learning Coefficient (LLC) During Training\n"
        "Lower λ = more singular = simpler effective model",
        fontsize=12, fontweight="bold"
    )

    color_llc = "#9C27B0"
    color_acc = "#4CAF50"

    ax1.plot(llc_steps, llcs, color=color_llc, lw=2, marker="o", ms=4, label="LLC λ")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("LLC λ̂", color=color_llc)
    ax1.tick_params(axis="y", labelcolor=color_llc)

    ax2 = ax1.twinx()
    ax2.plot(llc_steps, acc_at_llc, color=color_acc, lw=2, ls="--",
             marker="s", ms=4, label="Test accuracy")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Test accuracy", color=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_acc)
    ax2.spines["right"].set_visible(True)

    # Annotate phase boundaries from accuracy curve
    _annotate_phases(ax1, ax2, llc_steps, np.array(llcs), acc_at_llc)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

    plt.tight_layout()
    _save(fig, save_path, "fig2_llc_trajectory.png")
    return fig


def plot_phase_portrait(metrics: dict, save_path: str | None = None):
    """
    Figure 3: Phase portrait in (test_loss, LLC) space coloured by training step.
    Reveals distinct phases: high-loss/high-λ → low-loss/low-λ.
    """
    _apply_style()
    llc_steps = np.array(metrics["llc_steps"])
    llcs      = np.array(metrics["llc"])

    eval_steps = np.array(metrics["eval_steps"])
    test_loss  = np.array(metrics["test_loss"])
    test_acc   = np.array(metrics["test_acc"])

    loss_at_llc = np.interp(llc_steps, eval_steps, test_loss)
    acc_at_llc  = np.interp(llc_steps, eval_steps, test_acc)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Phase Portrait: LLC vs Test Loss", fontsize=12, fontweight="bold")

    sc = ax.scatter(loss_at_llc, llcs, c=llc_steps, cmap="viridis",
                    s=60, zorder=3, edgecolors="white", lw=0.5)
    ax.plot(loss_at_llc, llcs, color="grey", lw=0.8, alpha=0.5, zorder=2)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Training step")

    ax.set_xlabel("Test loss")
    ax.set_ylabel("LLC λ̂")

    # Annotate phases
    if len(llcs) > 4:
        early = len(llcs) // 6
        late  = -len(llcs) // 6
        ax.annotate("Memorisation\n(high λ, poor test)", xy=(loss_at_llc[early], llcs[early]),
                    xytext=(0.5, 0.85), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)
        ax.annotate("Generalisation\n(low λ, good test)", xy=(loss_at_llc[late], llcs[late]),
                    xytext=(0.15, 0.15), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="green"), color="green", fontsize=9)

    plt.tight_layout()
    _save(fig, save_path, "fig3_phase_portrait.png")
    return fig


def plot_model_comparison(results: list[dict], save_path: str | None = None):
    """
    Figure 4: LLC at convergence for different model sizes.
    Illustrates SLT's prediction: larger models can be more singular (lower λ).

    results : list of dicts with keys 'd_model', 'n_layers', 'llc_final', 'test_acc_final'
    """
    _apply_style()
    labels = [f"d={r['d_model']}\nL={r['n_layers']}" for r in results]
    llcs   = [r["llc_final"] for r in results]
    accs   = [r["test_acc_final"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("LLC at Convergence Across Model Sizes", fontsize=12, fontweight="bold")

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(results)))

    ax = axes[0]
    bars = ax.bar(labels, llcs, color=colors)
    ax.set_ylabel("LLC λ̂")
    ax.set_title("Local Learning Coefficient")

    ax = axes[1]
    ax.bar(labels, accs, color=colors)
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test Accuracy")

    plt.tight_layout()
    _save(fig, save_path, "fig4_model_comparison.png")
    return fig


def plot_sgld_diagnostics(energies: list[float], llc: float,
                          step: int, save_path: str | None = None):
    """Diagnostic plot of SGLD energy trace for a single LLC estimate."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.suptitle(f"SGLD Diagnostics  (step {step},  λ̂ = {llc:.3f})",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(energies, color="#9C27B0", lw=1)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("SGLD step")
    ax.set_ylabel("L(w_t) - L(w*)")
    ax.set_title("Energy trace")

    ax = axes[1]
    ax.hist(energies, bins=30, color="#9C27B0", alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(energies), color="red", ls="--", lw=1.5,
               label=f"mean = {np.mean(energies):.4f}")
    ax.set_xlabel("L(w_t) - L(w*)")
    ax.set_title("Energy distribution")
    ax.legend()

    plt.tight_layout()
    name = f"diag_sgld_step{step}.png"
    _save(fig, save_path, name)
    return fig


def _annotate_phases(ax_main, ax_acc, steps, llcs, accs):
    """
    Detect memorisation and grokking boundaries from the accuracy curve and
    draw vertical shaded bands + text labels on ax_main.
    """
    steps = np.array(steps)
    accs  = np.array(accs)

    # Memorisation: first step where train-side is high but test is still low
    # We infer from the LLC jump: find where LLC first exceeds 10% of its max
    llc_threshold = 0.10 * llcs.max()
    memo_candidates = np.where(llcs > llc_threshold)[0]
    memo_step = int(steps[memo_candidates[0]]) if len(memo_candidates) else None

    # Grokking: first step where test acc exceeds 50%
    grokk_candidates = np.where(accs > 0.50)[0]
    grokk_step = int(steps[grokk_candidates[0]]) if len(grokk_candidates) else None

    xmax = steps[-1]
    ymin, ymax = ax_main.get_ylim() if ax_main.get_ylim() != (0, 1) else (0, llcs.max() * 1.1)

    phase_alpha = 0.08
    if memo_step is not None:
        ax_main.axvspan(0, memo_step, alpha=phase_alpha, color="#2196F3", zorder=0)
        ax_main.text(memo_step * 0.45, llcs.max() * 0.95,
                     "Memorisation", ha="center", va="top",
                     fontsize=8, color="#1565C0", style="italic")

    if memo_step is not None and grokk_step is not None and grokk_step > memo_step:
        ax_main.axvspan(memo_step, grokk_step, alpha=phase_alpha, color="#FF9800", zorder=0)
        mid = (memo_step + grokk_step) / 2
        ax_main.text(mid, llcs.max() * 0.95,
                     "Plateau", ha="center", va="top",
                     fontsize=8, color="#E65100", style="italic")

    if grokk_step is not None:
        ax_main.axvspan(grokk_step, xmax, alpha=phase_alpha, color="#4CAF50", zorder=0)
        mid = (grokk_step + xmax) / 2
        ax_main.text(mid, llcs.max() * 0.95,
                     "Generalisation", ha="center", va="top",
                     fontsize=8, color="#1B5E20", style="italic")


def plot_flatness_trajectory(metrics: dict, save_path: str | None = None):
    """
    Figure 5: Hessian trace and top eigenvalue alongside LLC.

    All three are proxies for loss-surface geometry:
      trace(H) = sum of ALL curvatures  (total sharpness)
      λ_max(H) = sharpest single direction
      LLC λ̂   = effective singular dimension (SLT)

    SLT prediction: all three should rise during memorisation and
    decrease (or stabilise) after grokking as the Fourier solution's
    flat directions become dominant.
    """
    _apply_style()
    steps      = np.array(metrics["llc_steps"])
    llcs       = np.array(metrics["llc"])
    htraces    = np.array(metrics["htrace"])
    lambda_max = np.array(metrics["lambda_max"])

    eval_steps = np.array(metrics["eval_steps"])
    test_acc   = np.array(metrics["test_acc"])
    acc_at_llc = np.interp(steps, eval_steps, test_acc)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Loss-Surface Flatness During Training\n"
        "Flatness = more zero/near-zero Hessian eigenvalues = lower LLC (SLT)",
        fontsize=12, fontweight="bold",
    )

    def _shade(ax, ymax_scale=1.0):
        """Draw memorisation/grokking phase bands."""
        memo_c = np.where(acc_at_llc > 0.95)[0]
        grokk_c = np.where(acc_at_llc > 0.50)[0]
        xmax = steps[-1]
        memo_step  = int(steps[memo_c[0]])  if len(memo_c)  else None
        grokk_step = int(steps[grokk_c[0]]) if len(grokk_c) else None
        if memo_step:
            ax.axvspan(0, memo_step, alpha=0.07, color="#2196F3", zorder=0)
        if memo_step and grokk_step and grokk_step > memo_step:
            ax.axvspan(memo_step, grokk_step, alpha=0.07, color="#FF9800", zorder=0)
        if grokk_step:
            ax.axvspan(grokk_step, xmax, alpha=0.07, color="#4CAF50", zorder=0)

    # Panel 1: Hessian trace
    ax = axes[0]
    ax.plot(steps, htraces, color="#E91E63", lw=2, marker="o", ms=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("trace(H) — total curvature")
    ax.set_title("Hessian Trace  tr(H) = Σᵢ λᵢ")
    _shade(ax)

    # Panel 2: Top eigenvalue
    ax = axes[1]
    ax.plot(steps, lambda_max, color="#FF5722", lw=2, marker="o", ms=4)
    ax.set_xlabel("Training step")
    ax.set_ylabel("λ_max(H) — sharpest direction")
    ax.set_title("Top Hessian Eigenvalue  λ_max")
    _shade(ax)

    # Panel 3: LLC alongside normalised trace for comparison
    ax = axes[2]
    color_llc   = "#9C27B0"
    color_trace = "#E91E63"
    ax2 = ax.twinx()

    # Normalise both to [0,1] for visual overlay
    llc_n   = (llcs  - llcs.min())  / (llcs.max()  - llcs.min()  + 1e-9)
    trace_n = (htraces - htraces.min()) / (htraces.max() - htraces.min() + 1e-9)

    ax.plot(steps, llc_n,   color=color_llc,   lw=2, label="LLC λ̂ (norm.)")
    ax2.plot(steps, trace_n, color=color_trace, lw=2, ls="--", label="tr(H) (norm.)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Normalised LLC", color=color_llc)
    ax2.set_ylabel("Normalised tr(H)", color=color_trace)
    ax.tick_params(axis="y", labelcolor=color_llc)
    ax2.tick_params(axis="y", labelcolor=color_trace)
    ax2.spines["right"].set_visible(True)
    ax.set_title("LLC vs Hessian Trace (both normalised)")
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, l1 + l2, fontsize=9)
    _shade(ax)

    plt.tight_layout()
    _save(fig, save_path, "fig5_flatness_trajectory.png")
    return fig


def plot_loss_surfaces(panels: list[dict], save_path: str | None = None):
    """
    Figure 6: 2-D loss surface slices at 4 training stages.

    Each panel shows L(w* + α·d₁ + β·d₂) on a grid, with the same
    two filter-normalised directions d₁, d₂ used for all panels so
    the geometry is directly comparable.

    Colour scale is shared across all panels so sharpness is visible.

    panels : list of dicts with keys 'step', 'test_acc', 'Z', 'alphas', 'betas'
    """
    _apply_style()
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "2-D Loss Surface Slices at Four Training Stages\n"
        "Same weight-space directions used across all panels (filter-normalised)",
        fontsize=12, fontweight="bold",
    )

    # Shared colour scale: clip at 95th percentile to avoid outliers dominating
    all_Z = np.concatenate([p["Z"].ravel() for p in panels])
    vmin, vmax = np.percentile(all_Z, 2), np.percentile(all_Z, 95)

    for ax, panel in zip(axes, panels):
        Z = panel["Z"]
        alphas = panel["alphas"]
        betas  = panel["betas"]
        step   = panel["step"]
        acc    = panel["test_acc"]

        im = ax.contourf(alphas, betas, Z.T,
                         levels=30, cmap="RdYlGn_r",
                         vmin=vmin, vmax=vmax)
        ax.contour(alphas, betas, Z.T,
                   levels=15, colors="white", linewidths=0.4, alpha=0.5)
        ax.plot(0, 0, "w*", ms=10, zorder=5, label="w*")  # mark the minimum
        ax.set_xlabel("α  (direction d₁)")
        ax.set_ylabel("β  (direction d₂)")

        # Infer phase label from test accuracy
        if acc < 0.10:
            phase = "Init / learning"
        elif acc < 0.40:
            phase = "Memorised (plateau)"
        elif acc < 0.90:
            phase = "Grokking transition"
        else:
            phase = "Post-grokking"

        ax.set_title(f"Step {step}\n{phase}  (test {acc:.0%})", fontsize=10)
        ax.set_aspect("equal")

    # Single shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="Loss L(w)")
    plt.tight_layout()
    _save(fig, save_path, "fig6_loss_surfaces.png")
    return fig


def _save(fig, save_path, name):
    if save_path is not None:
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {p / name}")
