# MNIST Demo — Mechanistic Interpretability vs Singular Learning Theory

**Talk:** *"Interpretability: Should we cache activations or monitor the loss landscape?"*

---

## Central Argument

Both a memorising model and a generalising model can show similar activation structure (digit clusters in PCA space). Yet their *loss landscapes* are completely different — one has a sharp, narrow basin; the other sits in a wide, flat valley. **SLT detects this geometric difference; activation caching alone cannot.**

---

## Experimental Design

Two training regimes are compared side-by-side:

| Model | Training samples | Weight decay | Final train acc | Final test acc |
|-------|-----------------|--------------|-----------------|----------------|
| **Memoriser** | 200 (tiny) | 0 | 100% | ~83% |
| **Generaliser** | 3 000 | 0.005 | 100% | ~95% |

Both use the same `SmallMLP` architecture (784 → 512 → 256 → 10, no convolutions).  
Using an MLP — rather than a CNN — removes the spatial inductive bias that would otherwise help the memoriser generalise spuriously on MNIST.

The memoriser goes through three observable phases:

| Phase | Epoch | Train acc | Test acc | Loss landscape |
|-------|-------|-----------|----------|----------------|
| Init / random | 1 | ~50% | ~47% | unstructured |
| **Memorisation** | ~5 | ~98% | ~79% | **sharp narrow basin** |
| Fully memorised | 70 | 100% | ~83% | very sharp, frozen |

---

## Figures

### Mech Interp lens

| File | What it shows |
|------|---------------|
| `fig1_dual_training.png` | Training curves for both models — the ~12pp test-acc gap |
| `fig2_activation_pca.png` | PCA of penultimate activations at 4 stages (init / overfitting / memorised final / generalised final). Both models form digit clusters — activations *alone* cannot tell the regimes apart. |
| `fig3_cnn_filters.png` | Conv1 filter evolution for a small CNN trained on the full dataset. Filters grow into structured edge detectors, but the *timing* is opaque from filter images alone. |

### SLT lens

| File | What it shows |
|------|---------------|
| `fig4_llc_dual.png` | LLC λ̂ over training for each model (left/centre) + **normalised LLC = E[ΔL] = λ̂/(β·n)** (right panel) which is directly comparable across different training-set sizes. Memoriser E[ΔL] stays elevated; generaliser's decreases as it finds a simpler solution. |
| `fig5_1d_landscapes.png` | 1D loss scans along 4 filter-normalised random directions. Memoriser: tall narrow walls. Generaliser: shallow wide valley. |
| `fig6_2d_landscapes.png` | 2D loss surface at 4 epochs (memoriser init → overfitting → final, generaliser final). Fixed filter-normalised directions — geometry is *directly comparable* across panels. Shows the basin sharpening during memorisation, then the qualitative difference with the generaliser. |
| **`fig7_3d_landscapes.png`** | **THE SHOWSTOPPER.** Side-by-side 3D surfaces: sharp narrow well (memoriser) vs wide flat valley (generaliser). |

### Intersection / comparison

| File | What it shows |
|------|---------------|
| **`fig8_key_comparison.png`** | 2×2 grid: (activation PCA | loss landscape) × (memoriser | generaliser). The PCA rows look *similar*; the landscape rows look *completely different*. This is the main slide. |
| `fig9_phase_portrait.png` | Phase portrait in (test loss, LLC) space for both models, coloured by epoch. Reveals distinct trajectory clusters. Right panel: summary table of what each approach reveals vs misses. |

---

## How to Run

```bash
cd mnist_demo

# First time — create the venv
uv sync

# Quick preview (~6 min on MPS/GPU)
uv run python run_demo.py --quick --out ./figures

# Full quality (~25 min on CPU, ~8 min on MPS/GPU)
uv run python run_demo.py --out ./figures
```

Figures are written to `./figures/`. MNIST data is downloaded automatically to `./data/` on first run.

---

## Key SLT Concepts Illustrated

### Local Learning Coefficient (LLC)

The LLC λ̂ is the Real Log Canonical Threshold (RLCT) from Watanabe's singular learning theory. It estimates the *effective dimensionality* of the loss basin near a given weight `w*`.

```
λ̂ = β · n · E_{w ~ SGLD}[L(w) − L(w*)]
```

where `β = 1/log(n)` is the inverse temperature. A higher LLC at a given epoch indicates the model is in a *sharper* basin — small perturbations cause larger loss increases.

**Important:** raw LLC scales with `n`, so comparing λ̂ across models trained on different data amounts requires normalisation. The right panel of `fig4_llc_dual.png` shows `E[ΔL] = λ̂/(β·n)`, which is directly comparable.

### Filter-Normalised Loss Landscape

Following Li et al. (2018), perturbation directions are normalised so that each filter (or weight-matrix row) maintains its original norm. This removes the trivial scale ambiguity and makes landscape visualisations meaningful across different model sizes.

### What Activation Caching Misses

- **Basin sharpness**: an activation snapshot at epoch 5 (memorising) and at epoch 70 (fully memorised) looks nearly identical — both show digit clusters. But the loss landscape has *changed* (it became sharper and more entrenched as the model overfitted harder).
- **Fragility**: a memorising model is one small perturbation away from catastrophic loss. You cannot see this from activations; you can see it immediately from the 1D/2D/3D landscape scans.
- **Phase transition timing**: the LLC (SLT) starts changing as soon as the model enters the memorisation basin — *before* the test accuracy plateaus visibly. Activation PCA clusters already look structured at that point, offering no phase-change signal.

---

## Architecture

```
SmallMLP:  784 → ReLU → 512 → ReLU → 256 → 10
           ≈ 540 k parameters
SmallCNN:  Conv(16,5) → Pool → Conv(32,5) → Pool → FC(128) → 10
           ≈ 110 k parameters  (used only for filter visualisation)
```

---

## References

- Watanabe, S. (2009). *Algebraic Geometry and Statistical Learning Theory.*
- Li, H. et al. (2018). *Visualizing the Loss Landscape of Neural Nets.* NeurIPS.
- Hoogland, J. et al. (2023). *The Local Learning Coefficient.* [devinterp]
- Zhang, C. et al. (2017). *Understanding Deep Learning Requires Rethinking Generalisation.* ICLR.
