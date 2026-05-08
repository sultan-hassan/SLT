"""
Generate the three paper figures from saved results.

Loads all results/run_*.json files, computes mean ± std across seeds,
and saves three publication-quality figures to figures/.

Usage:
    python plot.py
    python plot.py --results_dir results --figures_dir figures
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        13,
    "axes.labelsize":   15,
    "axes.titlesize":   15,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  12,
    "legend.framealpha": 0.85,
    "lines.linewidth":  2.2,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── colour palette ────────────────────────────────────────────────────────────
C_TRAIN    = "#2471A3"   # blue
C_TEST     = "#C0392B"   # red
C_LLC      = "#7D3C98"   # purple  (SGLD-LLC)
C_HLLC     = "#17A589"   # teal    (Hessian-LLC)
C_FREQ     = "#D35400"   # orange  (active frequency count)
C_RATIO    = "#5D6D7E"   # slate

# Phase background shading — saturated enough to be visible in print
PH_MEM  = "#85C1E9"   # medium blue   — memorisation
PH_PLAT = "#F9E79F"   # medium yellow — plateau
PH_GROK = "#82E0AA"   # medium green  — grokking


# ---------------------------------------------------------------------------
# Data loading & alignment
# ---------------------------------------------------------------------------

def load_runs(results_dir: Path) -> list[dict]:
    paths = sorted(results_dir.glob("run_*.json"))
    if not paths:
        raise FileNotFoundError(f"No run_*.json files found in {results_dir}")
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))
    print(f"Loaded {len(runs)} run(s): seeds {[r['seed'] for r in runs]}")
    return runs


def align(runs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack values for `key` across runs, return (steps, mean, std).
    All runs must share the same checkpoint steps.
    """
    steps = np.array(runs[0]["steps"])
    matrix = np.array([r[key] for r in runs])   # (n_seeds, n_steps)
    return steps, matrix.mean(axis=0), matrix.std(axis=0)


def fourier_matrix(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (steps, mean_amp_matrix) where mean_amp_matrix has shape
    (n_steps, p//2) — the mean Fourier amplitude spectrum across seeds.
    """
    steps = np.array(runs[0]["steps"])
    matrices = np.array([r["fourier_amps"] for r in runs])  # (n_seeds, n_steps, p//2)
    return steps, matrices.mean(axis=0)                     # (n_steps, p//2)


def detect_transition(runs: list[dict], threshold: float = 0.5) -> tuple[float, float]:
    """
    Per-run: first step where test_acc exceeds threshold.
    Returns (mean_step, std_step) across seeds.
    """
    transition_steps = []
    for r in runs:
        steps = np.array(r["steps"])
        accs  = np.array(r["test_acc"])
        idx = np.where(accs >= threshold)[0]
        if len(idx):
            transition_steps.append(steps[idx[0]])
    arr = np.array(transition_steps, dtype=float)
    return float(arr.mean()), float(arr.std()) if len(arr) > 1 else 0.0


# ---------------------------------------------------------------------------
# Shared phase-shading helper
# ---------------------------------------------------------------------------

def add_phase_shading(ax, mem_end: float, grok_start: float, grok_end: float,
                      xmax: float):
    """Add colour-coded background bands for the three training phases."""
    ax.axvspan(0,          mem_end,    alpha=0.30, color=PH_MEM,  zorder=0)
    ax.axvspan(mem_end,    grok_start, alpha=0.30, color=PH_PLAT, zorder=0)
    ax.axvspan(grok_start, grok_end,   alpha=0.42, color=PH_GROK, zorder=0)


def phase_legend_handles():
    return [
        mpatches.Patch(color=PH_MEM,  alpha=0.85, label="Memorisation"),
        mpatches.Patch(color=PH_PLAT, alpha=0.85, label="Plateau"),
        mpatches.Patch(color=PH_GROK, alpha=0.85, label="Grokking"),
    ]


# ---------------------------------------------------------------------------
# Figure 1 — Training dynamics
# ---------------------------------------------------------------------------

def fig_training(runs, figures_dir, phase_steps):
    mem_end, grok_start, grok_end = phase_steps
    steps, mean_tr_loss, std_tr_loss = align(runs, "train_loss")
    _,     mean_te_loss, std_te_loss = align(runs, "test_loss")
    _,     mean_tr_acc,  std_tr_acc  = align(runs, "train_acc")
    _,     mean_te_acc,  std_te_acc  = align(runs, "test_acc")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    xmax = steps[-1]

    # — Loss panel ——————————————————————————————————————————————————————
    add_phase_shading(ax1, mem_end, grok_start, grok_end, xmax)

    ax1.semilogy(steps, mean_tr_loss, color=C_TRAIN, label="Train loss")
    ax1.fill_between(steps,
                     np.maximum(mean_tr_loss - std_tr_loss, 1e-6),
                     mean_tr_loss + std_tr_loss,
                     color=C_TRAIN, alpha=0.18)

    ax1.semilogy(steps, mean_te_loss, color=C_TEST, linestyle="--", label="Test loss")
    ax1.fill_between(steps,
                     np.maximum(mean_te_loss - std_te_loss, 1e-6),
                     mean_te_loss + std_te_loss,
                     color=C_TEST, alpha=0.18)

    # Random-chance baseline
    n_classes = runs[0]["fourier_amps"][0].__len__() * 2 + 1  # ≈ p
    ax1.axhline(np.log(97), color="gray", linestyle=":", linewidth=1.5,
                label=f"Random chance (log 97)")

    ax1.set_ylim(bottom=1e-3)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss")
    ax1.legend(loc="upper right")

    # Phase labels — placed near y-axis top after ylim is set
    ylo, yhi = ax1.get_ylim()
    label_y = 10 ** (0.82 * (np.log10(yhi) - np.log10(ylo)) + np.log10(ylo))
    ax1.text(mem_end / 2, label_y,
             "Memorisation", ha="center", fontsize=11, color="#1A5276", alpha=0.9)
    ax1.text((mem_end + grok_start) / 2, label_y,
             "Plateau", ha="center", fontsize=11, color="#7D6608", alpha=0.9)
    ax1.text((grok_start + min(grok_end + 200, xmax)) / 2, label_y,
             "Grokking", ha="center", fontsize=11, color="#145A32", alpha=0.9)

    # — Accuracy panel ———————————————————————————————————————————————————
    add_phase_shading(ax2, mem_end, grok_start, grok_end, xmax)

    ax2.plot(steps, 100 * mean_tr_acc, color=C_TRAIN, label="Train accuracy")
    ax2.fill_between(steps,
                     100 * (mean_tr_acc - std_tr_acc),
                     100 * (mean_tr_acc + std_tr_acc),
                     color=C_TRAIN, alpha=0.18)

    ax2.plot(steps, 100 * mean_te_acc, color=C_TEST, linestyle="--",
             label="Test accuracy")
    ax2.fill_between(steps,
                     100 * (mean_te_acc - std_te_acc),
                     100 * (mean_te_acc + std_te_acc),
                     color=C_TEST, alpha=0.18)

    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.set_ylim(-5, 105)
    ax2.axhline(100, color=C_TRAIN, linestyle=":", linewidth=1, alpha=0.5)
    ax2.legend(loc="center right")

    n = len(runs)
    fig.suptitle(
        f"Grokking arc: memorisation  →  plateau  →  generalisation "
        f"(mean ± std, n={n} seeds)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    out = figures_dir / "fig1_training.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Local vs. global geometry diverge
# ---------------------------------------------------------------------------

def fig_dissociation(runs, figures_dir, phase_steps):
    mem_end, grok_start, grok_end = phase_steps
    steps, mean_llc,  std_llc  = align(runs, "llc")
    _,     mean_hllc, std_hllc = align(runs, "hessian_llc")
    _,     mean_te_acc, _      = align(runs, "test_acc")

    ratio_all = np.array([np.array(r["llc"]) / np.array(r["hessian_llc"])
                          for r in runs])
    mean_ratio = ratio_all.mean(axis=0)
    std_ratio  = ratio_all.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    xmax = steps[-1]

    # — Left: both LLCs on log scale ─────────────────────────────────────
    add_phase_shading(ax1, mem_end, grok_start, grok_end, xmax)

    ax1.semilogy(steps, mean_llc, color=C_LLC, label="SGLD-LLC  (global multiplicity)")
    ax1.fill_between(steps,
                     np.maximum(mean_llc - std_llc, 1),
                     mean_llc + std_llc,
                     color=C_LLC, alpha=0.18)

    ax1.semilogy(steps, mean_hllc, color=C_HLLC,
                 label=r"Hessian-LLC $\hat\lambda_H$  (local flatness)")
    ax1.fill_between(steps,
                     np.maximum(mean_hllc - std_hllc, 0.1),
                     mean_hllc + std_hllc,
                     color=C_HLLC, alpha=0.18)

    # Reference line: circuit prediction (~12 freq × 2 params = 24, λ_H ≈ 19)
    ax1.axhline(19, color=C_HLLC, linestyle=":", linewidth=1.8, alpha=0.7)
    ax1.text(xmax * 0.97, 19 * 1.35,
             r"$\hat\lambda_H \approx 19$" + "\n(~12 freq × 2 params)",
             ha="right", fontsize=10.5, color=C_HLLC)

    # Test accuracy on secondary axis
    ax1b = ax1.twinx()
    ax1b.plot(steps, 100 * mean_te_acc, color=C_TEST,
              linestyle="--", linewidth=1.5, alpha=0.7, label="Test acc (%)")
    ax1b.set_ylabel("Test accuracy (%)", fontsize=13, color=C_TEST)
    ax1b.tick_params(axis="y", labelcolor=C_TEST)
    ax1b.set_ylim(-5, 115)
    ax1b.spines["right"].set_visible(True)

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("LLC  (log scale)")
    ax1.set_title("Two orders of magnitude diverge at grokking")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               loc="lower right", fontsize=11)

    # — Right: ratio on log scale ─────────────────────────────────────────
    add_phase_shading(ax2, mem_end, grok_start, grok_end, xmax)

    ax2.semilogy(steps, mean_ratio, color=C_RATIO,
                 label="SGLD-LLC / Hessian-LLC")
    ax2.fill_between(steps,
                     np.maximum(mean_ratio - std_ratio, 0.1),
                     mean_ratio + std_ratio,
                     color=C_RATIO, alpha=0.22)

    # Annotate the post-grokking plateau
    post_idx = np.where(steps >= grok_end)[0]
    if len(post_idx):
        post_ratio = mean_ratio[post_idx[0]]
        ax2.annotate(
            "~100× gap\n(geometry-based\ntransition detector)",
            xy=(steps[post_idx[0]], post_ratio),
            xytext=(steps[post_idx[0]] - (xmax * 0.3), post_ratio * 3),
            fontsize=11, color=C_RATIO,
            arrowprops=dict(arrowstyle="->", color=C_RATIO, lw=1.5),
        )

    ax2.set_xlabel("Training step")
    ax2.set_ylabel("SGLD-LLC / Hessian-LLC  (log scale)")
    ax2.set_title("Ratio jumps at the grokking transition")
    ax2.legend(loc="upper left")

    n = len(runs)
    fig.suptitle(
        "Local flatness (Hessian) collapses at grokking; "
        f"global multiplicity (SGLD) does not  (n={n} seeds)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    out = figures_dir / "fig2_dissociation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — Circuit Imprint
# ---------------------------------------------------------------------------

def fig_circuit_imprint(runs, figures_dir, phase_steps):
    mem_end, grok_start, grok_end = phase_steps
    # Use a single representative seed for the heatmap — crisper than the mean
    # (averaging across seeds blurs individual frequency activations).
    # Mean across seeds is used for all line plots on the right panel.
    ref_run = runs[0]
    amp_matrix = np.array(ref_run["fourier_amps"])    # (n_steps, p//2)
    steps      = np.array(ref_run["steps"])
    _,  mean_hllc, std_hllc = align(runs, "hessian_llc")
    _,  mean_te_acc, _      = align(runs, "test_acc")

    threshold_mult = 2.0
    active_all = []
    for r in runs:
        amps = np.array(r["fourier_amps"])             # (n_steps, p//2)
        mean_amp = amps.mean(axis=1, keepdims=True)
        active_all.append((amps > threshold_mult * mean_amp).sum(axis=1))
    active_all   = np.array(active_all)               # (n_seeds, n_steps)
    mean_active  = active_all.mean(axis=0)
    std_active   = active_all.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    xmax = steps[-1]

    # — Left: Fourier spectrum heatmap (mean across seeds) ────────────────
    # amp_matrix: (n_steps, p//2) → transpose to (p//2, n_steps) for imshow
    n_freq = amp_matrix.shape[1]
    im = ax1.imshow(
        amp_matrix.T,                                  # (p//2, n_steps)
        aspect="auto",
        origin="lower",
        extent=[steps[0], steps[-1], 0.5, n_freq + 0.5],
        cmap="hot",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax1, label="Amplitude", fraction=0.04, pad=0.02)

    # Phase lines
    ax1.axvline(mem_end,    color="dodgerblue", linewidth=1.5,
                linestyle="--", alpha=0.8)
    ax1.axvline(grok_start, color="limegreen",  linewidth=2.0,
                linestyle="--", alpha=0.9, label="Grokking onset")

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Fourier frequency  k")
    ax1.set_title("Token-embedding frequency spectrum\n"
                  "(bright = high amplitude; single seed)")
    ax1.legend(loc="upper left", fontsize=11)

    # — Right: active freq count + Hessian-LLC ───────────────────────────
    add_phase_shading(ax2, mem_end, grok_start, grok_end, xmax)

    # Active frequency count (left y-axis)
    ax2.plot(steps, mean_active, color=C_FREQ, linewidth=2.5,
             label="Active Fourier frequencies")
    ax2.fill_between(steps,
                     np.maximum(mean_active - std_active, 0),
                     mean_active + std_active,
                     color=C_FREQ, alpha=0.22)
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Active frequency count", color=C_FREQ, fontsize=14)
    ax2.tick_params(axis="y", labelcolor=C_FREQ)
    ax2.set_ylim(-1, max(mean_active.max() * 1.5, 20))

    # Hessian-LLC (right y-axis)
    ax2b = ax2.twinx()
    ax2b.semilogy(steps, mean_hllc, color=C_HLLC, linewidth=2.5,
                  linestyle="-", label=r"Hessian-LLC $\hat\lambda_H$")
    ax2b.fill_between(steps,
                      np.maximum(mean_hllc - std_hllc, 0.1),
                      mean_hllc + std_hllc,
                      color=C_HLLC, alpha=0.18)
    ax2b.axhline(19, color=C_HLLC, linestyle=":", linewidth=1.8, alpha=0.7)
    ax2b.set_ylabel(r"Hessian-LLC $\hat\lambda_H$  (log scale)",
                    color=C_HLLC, fontsize=14)
    ax2b.tick_params(axis="y", labelcolor=C_HLLC)
    ax2b.spines["right"].set_visible(True)

    # Test accuracy for reference
    ax2c_line, = ax2b.plot(steps, mean_te_acc * max(mean_hllc) * 1.1,
                           color=C_TEST, linestyle="--",
                           linewidth=1.5, alpha=0.6)

    # Convergence annotation
    ax2.annotate(
        "Both signals\ntransition here",
        xy=(grok_start, mean_active[np.where(steps >= grok_start)[0][0]]),
        xytext=(grok_start + (xmax - grok_start) * 0.25,
                mean_active.max() * 0.55),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    )

    # Legend
    lines  = [Line2D([0], [0], color=C_FREQ, lw=2.5),
              Line2D([0], [0], color=C_HLLC, lw=2.5),
              Line2D([0], [0], color=C_TEST, lw=1.5, linestyle="--", alpha=0.7)]
    labels = ["Active Fourier freq.",
              r"Hessian-LLC $\hat\lambda_H$",
              "Test accuracy (scaled)"]
    ax2.legend(lines, labels, loc="lower right", fontsize=11)

    ax2.set_title("Circuit Imprint: independent signals converge\n"
                  r"at grokking onset  ($\sim\!12$ freq $\times\,2 \approx 24$; $\hat\lambda_H \approx 19$)")

    n = len(runs)
    fig.suptitle(
        "Fourier amplitude analysis and Hessian curvature agree "
        f"with no shared information  (n={n} seeds)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = figures_dir / "fig3_circuit_imprint.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate the three paper figures from saved results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--figures_dir", default="figures")
    # Phase boundary steps (used for background shading)
    # These are approximate; the actual transition is visible in the curves.
    parser.add_argument("--mem_end",    type=int, default=500,
                        help="end of memorisation phase (step)")
    parser.add_argument("--grok_start", type=int, default=1900,
                        help="start of grokking transition (step)")
    parser.add_argument("--grok_end",   type=int, default=2600,
                        help="end of grokking transition (step)")
    cfg = parser.parse_args()

    results_dir = Path(cfg.results_dir)
    figures_dir = Path(cfg.figures_dir)
    figures_dir.mkdir(exist_ok=True)

    runs = load_runs(results_dir)
    phase_steps = (cfg.mem_end, cfg.grok_start, cfg.grok_end)

    # Override phase boundaries from the data if only one seed loaded
    if len(runs) == 1:
        t_mean, _ = detect_transition(runs, threshold=0.50)
        phase_steps = (500, int(t_mean) - 100, int(t_mean) + 300)

    print(f"\nPhase boundaries: mem_end={phase_steps[0]}, "
          f"grok_start={phase_steps[1]}, grok_end={phase_steps[2]}")
    print(f"Generating figures → {figures_dir}/\n")

    fig_training(runs,         figures_dir, phase_steps)
    fig_dissociation(runs,     figures_dir, phase_steps)
    fig_circuit_imprint(runs,  figures_dir, phase_steps)

    print("\nDone!  Three figures saved:")
    print("  figures/fig1_training.png")
    print("  figures/fig2_dissociation.png")
    print("  figures/fig3_circuit_imprint.png")


if __name__ == "__main__":
    main()
