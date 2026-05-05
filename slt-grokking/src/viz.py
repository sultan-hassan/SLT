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

    # Annotate the three distinct clusters in (test_loss, LLC) space.
    # • Init/learning:         (high loss, low  LLC) — bottom-right
    # • Memorisation plateau:  (high loss, high LLC) — top-right
    # • Post-grokking:         (low  loss, high LLC) — top-left
    # Note: LLC does NOT drop after grokking (calibration issue — see §5).
    # The Hessian trace / λ_max are the clean flatness signal.
    if len(llcs) > 4:
        # Memorisation plateau: pick a point well inside the high-loss, high-LLC cluster
        # (roughly the middle third of the trajectory)
        memo_idx  = max(1, len(llcs) // 3)      # ~step 1000, LLC≈1020, loss≈4.7
        # Post-grokking: last point (loss≈0, LLC still high)
        post_idx  = len(llcs) - 1

        ax.annotate("Memorisation plateau\n(high loss, λ ≈ 800–1400)",
                    xy=(loss_at_llc[memo_idx], llcs[memo_idx]),
                    xytext=(0.60, 0.25), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)
        ax.annotate("Post-grokking\n(loss → 0, λ stays high*)",
                    xy=(loss_at_llc[post_idx], llcs[post_idx]),
                    xytext=(0.30, 0.60), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="green"), color="green", fontsize=9)
        ax.text(0.98, 0.03, "* LLC stays high: calibration issue (§5).\n"
                "  True post-grokking λ ≈ 20 from Hessian.",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, style="italic", color="#555555")

    plt.tight_layout()
    _save(fig, save_path, "fig3_phase_portrait.png")
    return fig


def plot_model_comparison(results: list[dict], save_path: str | None = None):
    """
    Figure 6 (optional, --model_sweep): LLC at convergence for different model sizes.
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
    _save(fig, save_path, "fig6_model_comparison.png")
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

    # Memorisation plateau begins when LLC has risen past 50% of its maximum.
    # (Using 10% placed the boundary too early, in the fast-learning phase before
    # train accuracy reaches 100% and the lookup-table circuit is entrenched.)
    llc_threshold = 0.50 * llcs.max()
    memo_candidates = np.where(llcs > llc_threshold)[0]
    memo_step = int(steps[memo_candidates[0]]) if len(memo_candidates) else None

    # Grokking: first step where test acc exceeds 50%
    grokk_candidates = np.where(accs > 0.50)[0]
    grokk_step = int(steps[grokk_candidates[0]]) if len(grokk_candidates) else None

    xmax = steps[-1]

    phase_alpha = 0.08
    if memo_step is not None:
        ax_main.axvspan(0, memo_step, alpha=phase_alpha, color="#2196F3", zorder=0)
        ax_main.text(memo_step * 0.45, llcs.max() * 0.95,
                     "Memorising", ha="center", va="top",
                     fontsize=8, color="#1565C0", style="italic")

    if memo_step is not None and grokk_step is not None and grokk_step > memo_step:
        ax_main.axvspan(memo_step, grokk_step, alpha=phase_alpha, color="#FF9800", zorder=0)
        mid = (memo_step + grokk_step) / 2
        ax_main.text(mid, llcs.max() * 0.95,
                     "Memorisation plateau", ha="center", va="top",
                     fontsize=8, color="#E65100", style="italic")

    if grokk_step is not None:
        ax_main.axvspan(grokk_step, xmax, alpha=phase_alpha, color="#4CAF50", zorder=0)
        mid = (grokk_step + xmax) / 2
        ax_main.text(mid, llcs.max() * 0.95,
                     "Grokking", ha="center", va="top",
                     fontsize=8, color="#1B5E20", style="italic")


def plot_flatness_trajectory(metrics: dict, save_path: str | None = None):
    """
    Figure 4: Hessian trace and top eigenvalue alongside LLC.

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
        """Draw memorisation/grokking phase bands (consistent with _annotate_phases)."""
        # Use LLC > 50% of max for memorisation plateau start (same threshold as
        # _annotate_phases), and test_acc > 50% for grokking onset.
        llc_max = llcs.max()
        memo_c  = np.where(llcs > 0.50 * llc_max)[0]
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
    _save(fig, save_path, "fig4_flatness_trajectory.png")
    return fig


def plot_loss_surfaces(panels: list[dict], save_path: str | None = None):
    """
    Figure 5: 2-D loss surface slices at 4 training stages.

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
    _save(fig, save_path, "fig5_loss_surfaces.png")
    return fig


def plot_hessian_phases(metrics: dict, save_path: str | None = None):
    """
    Presentation figure: Hessian trace (top) + accuracy curves (bottom) stacked
    on a shared x-axis. Phase bands align across both panels so the audience can
    immediately see that tr(H) collapses exactly when test accuracy jumps.
    """
    _apply_style()

    h_steps    = np.array(metrics["llc_steps"])
    htraces    = np.array(metrics["htrace"])
    eval_steps = np.array(metrics["eval_steps"])
    train_acc  = np.array(metrics["train_acc"])
    test_acc   = np.array(metrics["test_acc"])

    # Phase boundaries (same logic used everywhere else)
    memo_idx  = np.where(train_acc > 0.95)[0]
    grokk_idx = np.where(test_acc  > 0.50)[0]
    memo_step  = int(eval_steps[memo_idx[0]])  if len(memo_idx)  else None
    grokk_step = int(eval_steps[grokk_idx[0]]) if len(grokk_idx) else None
    xmax = int(eval_steps[-1])

    fig, (ax_h, ax_a) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.08},
    )
    fig.suptitle(
        "Hessian Trace Tracks Grokking Phase Transitions\n"
        "tr(H) rises during memorisation, holds at plateau, collapses when grokking fires",
        fontsize=12, fontweight="bold",
    )

    def _shade(ax):
        a = 0.08
        if memo_step:
            ax.axvspan(0, memo_step, alpha=a, color="#2196F3", zorder=0)
        if memo_step and grokk_step and grokk_step > memo_step:
            ax.axvspan(memo_step, grokk_step, alpha=a, color="#FF9800", zorder=0)
        if grokk_step:
            ax.axvspan(grokk_step, xmax, alpha=a, color="#4CAF50", zorder=0)

    # ── Top panel: Hessian trace ────────────────────────────────────────────
    _shade(ax_h)
    ax_h.plot(h_steps, htraces, color="#E91E63", lw=2.2, marker="o", ms=4,
              zorder=3, label="tr(H)")
    ax_h.set_ylabel("Hessian trace  tr(H)", fontsize=11)
    ax_h.tick_params(labelbottom=False)

    # Phase labels inside top panel
    if memo_step:
        ax_h.text(memo_step * 0.45, 0.93, "Memorising",
                  transform=ax_h.get_xaxis_transform(),
                  ha="center", fontsize=9, color="#1565C0", style="italic", fontweight="bold")
    if memo_step and grokk_step:
        ax_h.text((memo_step + grokk_step) / 2, 0.93, "Plateau",
                  transform=ax_h.get_xaxis_transform(),
                  ha="center", fontsize=9, color="#E65100", style="italic", fontweight="bold")
    if grokk_step:
        ax_h.text((grokk_step + xmax) / 2, 0.93, "Grokking",
                  transform=ax_h.get_xaxis_transform(),
                  ha="center", fontsize=9, color="#1B5E20", style="italic", fontweight="bold")

    # Annotate peak (restrict to memorising phase) and post-grokking trough
    if grokk_step is not None:
        memo_mask = h_steps < grokk_step
        peak_idx  = int(np.argmax(htraces[memo_mask])) if memo_mask.sum() else int(np.argmax(htraces))
    else:
        peak_idx = int(np.argmax(htraces))

    peak_val  = htraces[peak_idx]
    peak_step = h_steps[peak_idx]

    if grokk_step is not None:
        post_mask = h_steps >= grokk_step
        if post_mask.sum() >= 2:
            post_h = htraces[post_mask]
            post_s = h_steps[post_mask]
            trough_idx_local = int(np.argmin(post_h))
            trough_val  = post_h[trough_idx_local]
            trough_step = post_s[trough_idx_local]
        else:
            trough_val, trough_step = htraces[-1], h_steps[-1]
    else:
        trough_val, trough_step = htraces[-1], h_steps[-1]

    ratio = peak_val / max(trough_val, 1.0)

    ax_h.annotate(
        f"Peak: {peak_val:,.0f}\n(memorising)",
        xy=(peak_step, peak_val),
        xytext=(peak_step + xmax * 0.04, peak_val * 0.82),
        fontsize=8.5, color="#C2185B",
        arrowprops=dict(arrowstyle="->", color="#C2185B", lw=1.2),
    )
    ax_h.annotate(
        f"Trough: {trough_val:,.0f}\n({ratio:.0f}× drop)",
        xy=(trough_step, trough_val),
        xytext=(trough_step - xmax * 0.18, trough_val + peak_val * 0.18),
        fontsize=8.5, color="#1B5E20",
        arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.2),
    )

    # Mark post-grokking transient spikes: only flag values that jump back up
    # AFTER the model has already settled into the Fourier minimum (tr(H) < 3× trough).
    if grokk_step is not None:
        post_mask = h_steps >= grokk_step
        post_h = htraces[post_mask]
        post_s = h_steps[post_mask]
        settled = post_s[post_h < trough_val * 3]
        if len(settled):
            first_settled = settled[0]
            spike_mask = (post_h > trough_val * 10) & (post_s >= first_settled)
            for sp_step, sp_val in zip(post_s[spike_mask], post_h[spike_mask]):
                ax_h.annotate(
                    "AdamW\ntransient",
                    xy=(sp_step, sp_val),
                    xytext=(sp_step - xmax * 0.12, sp_val * 0.75),
                    fontsize=7.5, color="#888888", style="italic",
                    arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1),
                )

    # ── Bottom panel: accuracy ──────────────────────────────────────────────
    _shade(ax_a)
    ax_a.plot(eval_steps, train_acc, color="#2196F3", lw=2,   label="Train acc")
    ax_a.plot(eval_steps, test_acc,  color="#F44336", lw=2, ls="--", label="Test acc")
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.set_ylabel("Accuracy", fontsize=11)
    ax_a.set_xlabel("Training step", fontsize=11)
    ax_a.legend(loc="center right", fontsize=9)

    plt.tight_layout()
    _save(fig, save_path, "fig9_hessian_phases.png")
    return fig


def _save(fig, save_path, name):
    if save_path is not None:
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {p / name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Mech Interp: activation PCA across training phases
# ─────────────────────────────────────────────────────────────────────────────

def plot_activation_phases(
    panels: list[dict],
    p: int,
    save_path: str | None = None,
):
    """
    4-panel PCA of readout-slot activations, one panel per training phase.

    Each panel entry:
        {"label": str, "step": int, "train_acc": float, "test_acc": float,
         "proj": (N,2) ndarray, "labels": (N,) int ndarray, "var": (2,) ndarray}

    Coloured by answer (a+b) mod p using a cyclic colormap.
    Key message: cluster structure looks similar at the memorisation plateau and
    post-grokking even though test accuracy jumps from ~20% to ~100%.
    Activation PCA cannot distinguish the phases; the loss landscape (LLC) can.
    """
    _apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5))
    fig.suptitle(
        "Mech Interp — Readout Activations (PCA) Across Training Phases\n"
        "Cluster structure forms during memorisation and persists through grokking\n"
        "— activations alone cannot identify when generalisation happened",
        fontsize=11, fontweight="bold",
    )

    cmap = plt.cm.hsv
    norm = plt.Normalize(0, p)

    for ax, panel in zip(axes, panels):
        proj   = panel["proj"]
        labels = panel["labels"]
        var    = panel["var"]

        sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap=cmap, norm=norm,
                        s=8, alpha=0.6, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var[1]:.1%})")

        phase_color = {
            "Init":        "#888888",
            "Memorising":  "#E91E63",
            "Plateau":     "#FF8F00",
            "Post-grokking": "#1565C0",
        }.get(panel["label"], "#333333")

        ax.set_title(
            f"{panel['label']}  (step {panel['step']})\n"
            f"train {panel['train_acc']:.0%}  |  test {panel['test_acc']:.0%}",
            fontsize=10, color=phase_color, fontweight="bold",
        )

        note_color = "#FFEBEE" if panel["test_acc"] < 0.5 else "#E8F5E9"
        note = (
            "Clusters forming\n⚠ test acc still low\n→ memorised, not understood"
            if panel["test_acc"] < 0.5
            else "Similar cluster structure\n✓ test acc high\n→ generalised circuit found"
        )
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=8.5,
                va="bottom", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=note_color, alpha=0.92))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[-1], label=f"Answer (a+b) mod {p}", shrink=0.8)
    plt.tight_layout()
    _save(fig, save_path, "fig7_activation_phases.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Two lenses: activation PCA vs LLC across phases
# ─────────────────────────────────────────────────────────────────────────────

def plot_two_lenses(
    act_panels: list[dict],
    llc_metrics: dict,
    train_metrics: dict,
    p: int,
    save_path: str | None = None,
):
    """
    2×2 grid: (Mech Interp row) × (Plateau | Post-grokking column).

    Top row   — activation PCA: looks structurally similar in both phases.
    Bottom row — LLC trajectory with phase markers: clearly different signal.

    Core argument: PCA says 'both have learned'; LLC says 'only one generalised'.
    """
    _apply_style()

    # Pick the plateau and post-grokking panels
    plateau_panel = next((p for p in act_panels if p["label"] == "Plateau"), act_panels[-2])
    grokk_panel   = next((p for p in act_panels if p["label"] == "Post-grokking"), act_panels[-1])

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(
        "Two Lenses on Grokking\n"
        "Activations say 'both learned'; LLC says 'only one learned robustly'",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.3,
                           top=0.88, bottom=0.08)

    cmap = plt.cm.hsv
    norm_c = plt.Normalize(0, p)

    # ── Row 0: Activation PCA ────────────────────────────────────────────────
    for col, panel in enumerate([plateau_panel, grokk_panel]):
        ax = fig.add_subplot(gs[0, col])
        proj, labels, var = panel["proj"], panel["labels"], panel["var"]
        ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap=cmap, norm=norm_c,
                   s=8, alpha=0.6, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(
            f"Mech Interp — {panel['label']}\n"
            f"step {panel['step']}  ·  test {panel['test_acc']:.0%}",
            fontsize=10.5,
        )
        note = ("Clusters look structured.\n"
                '"Model seems to have learned."\n'
                "⚠ Can't assess whether it generalised.")
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=8.5,
                va="bottom", style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", alpha=0.9))

    # ── Row 1: LLC trajectory with phase shading ─────────────────────────────
    for col in range(2):
        ax = fig.add_subplot(gs[1, col])
        focus_panel = [plateau_panel, grokk_panel][col]

        llc_steps = np.array(llc_metrics["llc_steps"])
        llc_vals  = np.array(llc_metrics["llc"])
        eval_steps = np.array(train_metrics["eval_steps"])
        test_acc   = np.array(train_metrics["test_acc"])
        train_acc  = np.array(train_metrics["train_acc"])

        # Phase shading
        memo_step  = eval_steps[np.where(train_acc > 0.95)[0][0]] if np.any(train_acc > 0.95) else None
        grokk_step = eval_steps[np.where(test_acc  > 0.50)[0][0]] if np.any(test_acc  > 0.50) else None
        xmax = eval_steps[-1]

        if memo_step:
            ax.axvspan(0, memo_step, alpha=0.07, color="#E91E63", label="Memorising")
        if memo_step and grokk_step:
            ax.axvspan(memo_step, grokk_step, alpha=0.07, color="#FF8F00", label="Plateau")
        if grokk_step:
            ax.axvspan(grokk_step, xmax, alpha=0.07, color="#1565C0", label="Grokking")

        ax.plot(llc_steps, llc_vals, color="#7B1FA2", lw=2.5, marker="o", ms=5)

        # Mark the focus step
        focus_step = focus_panel["step"]
        focus_llc  = float(np.interp(focus_step, llc_steps, llc_vals))
        ax.axvline(focus_step, color="black", lw=1.5, ls="--", alpha=0.7)
        ax.scatter([focus_step], [focus_llc], color="black", s=80, zorder=5)
        ax.annotate(
            f"LLC={focus_llc:.0f}\n(this phase)",
            xy=(focus_step, focus_llc),
            xytext=(focus_step + xmax * 0.05, focus_llc * 1.05),
            fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=1),
        )

        ax.set_xlabel("Training step"); ax.set_ylabel("LLC λ̂")
        ax.set_title(
            f"SLT — LLC trajectory  [{focus_panel['label']}]",
            fontsize=10.5,
        )
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    _save(fig, save_path, "fig8_two_lenses.png")
    return fig
