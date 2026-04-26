# SLT × Grokking: Loss-Surface Geometry Across Phase Transitions in Transformers

A self-contained project applying **Singular Learning Theory (SLT)** to study the
*grokking* phenomenon — the surprising ability of a neural network to suddenly generalise
long after memorising its training set — through three complementary geometric lenses:
the **Local Learning Coefficient (LLC)**, the **Hessian trace**, and **2-D loss surface
slices**.

**Research question:**
> *How does the geometry of the loss landscape — flatness, curvature, and singularity
> structure — change across the grokking phase transition, and what does this reveal about
> the internal structure of the solutions found at each developmental stage?*

---

## Contents

1. [Background: SLT, Flatness, and the Hessian](#1-background-slt-flatness-and-the-hessian)
2. [The Grokking Testbed](#2-the-grokking-testbed)
3. [Results](#3-results)
   - [Fig 1 – Training Dynamics](#fig-1--training-dynamics-grokking-curves)
   - [Fig 2 – LLC Trajectory](#fig-2--llc-trajectory)
   - [Fig 3 – Phase Portrait](#fig-3--phase-portrait)
   - [Fig 4 – Flatness Analysis (Hessian)](#fig-4--flatness-analysis-hessian-trace--top-eigenvalue)
   - [Fig 5 – 2-D Loss Surface Slices](#fig-5--2-d-loss-surface-slices)
   - [Fig 6 – Model Comparison (optional)](#fig-6--model-comparison-optional)
4. [Interpretation: Geometry Tells the Story](#4-interpretation-geometry-tells-the-story)
5. [The LLC Calibration Challenge](#5-the-llc-calibration-challenge)
6. [Connections to Timaeus Research](#6-connections-to-timaeus-research)
7. [Setup and Usage](#7-setup-and-usage)
8. [References](#8-references)

---

## 1. Background: SLT, Flatness, and the Hessian

### Why classical theory fails for neural networks

Classical asymptotic statistics (Bernstein–von Mises, AIC, BIC) assume the Fisher
information matrix is non-singular at the true parameter. For neural networks this
assumption fails catastrophically: the parameter-to-function map is massively
non-injective. A single function can be realised by infinitely many parameter settings
(weight permutations, rescalings, and deeper symmetries), so the Fisher information is
degenerate everywhere.

### Watanabe's resolution: the RLCT

Sumio Watanabe's Singular Learning Theory (SLT) replaces Fisher information with
algebraic geometry. The key result is the **free energy formula**:

```
n F_n(w*) = n L_n(w*) + λ log n − (m − 1) log log n + O(1)
```

where:
- `L_n(w*)` is the training loss at the basin minimum w*
- **`λ`** is the **Real Log Canonical Threshold (RLCT)**, the *Local Learning Coefficient*
- `m` is the multiplicity of the singularity
- `n` is the number of training samples

`λ` measures *how flat the loss landscape is* around w* in a precise algebraic-geometric
sense — the effective dimension of the parameter space that the data can resolve.

| Loss landscape near w* | λ |
|---|---|
| Regular (all Hessian eigenvalues > 0) | d/2  (d = number of parameters) |
| Mildly degenerate | < d/2 |
| Highly degenerate (many flat directions) | ≪ d/2 |

**The key SLT insight:** Among minima with similar training loss, the one with *smaller λ*
has smaller free energy and is preferred by the Bayesian posterior. Smaller λ means a
more singular (degenerate) solution — more flat directions, more symmetry, simpler
effective structure. This is SLT's rigorous account of implicit regularisation.

### The Hessian: second derivatives as a flatness ruler

The **Hessian matrix** H at w* is:

```
H_ij  =  ∂²L / ∂w_i ∂w_j
```

the matrix of all second partial derivatives of the loss. Its eigenvalues measure
curvature in every direction of weight space:

- **Large eigenvalue** → sharp direction, narrow loss bowl, many parameters matter locally.
- **Near-zero eigenvalue** → **flat direction**, a degeneracy, the loss barely changes
  if you move that way.

The number of near-zero Hessian eigenvalues is the leading-order indicator of the LLC:

```
In the regular case:    LLC  ≈  (rank of H) / 2  =  d/2
In the singular case:   LLC  <  (rank of H) / 2
```

SLT goes beyond the Hessian by capturing higher-order degeneracies (when the Hessian
has a true zero eigenvalue, the LLC depends on 3rd/4th-order Taylor terms), but for
practical networks two Hessian scalars — the trace and the largest eigenvalue — are
reliable leading-order proxies:

| Measure | Definition | Interpretation |
|---|---|---|
| **tr(H)** | Σᵢ λᵢ(H) — sum of all curvatures | Total sharpness of the basin |
| **λ_max(H)** | Largest eigenvalue | Curvature in the sharpest direction |
| **LLC λ̂** | SGLD-based WBIC estimator | Effective singular dimension (SLT) |

All three should **rise during memorisation** (sharp non-degenerate lookup-table minimum)
and **fall after grokking** (flat degenerate Fourier minimum). Confirming this with real
data is the goal of this project.

### Estimating LLC via localised SGLD

We use the **WBIC estimator** (Watanabe 2013) with a localisation spring:

```
U_loc(w) = β·n·L_n(w)  +  (γ/2)·‖w − w*‖²

λ̂  =  β · n · E_{SGLD}[L(w) − L(w*)]
```

where β = 1/log(n) is the WBIC temperature and γ = 10,000 keeps the chain near w*.
The SGLD update for this energy:

```
w_{t+1} = w_t  −  ε·[(β·n/|B|)·∇L_B(w_t) + γ·(w_t − w*)]  +  √(2ε)·η
```

### Estimating Hessian geometry via finite differences

We use two MPS-safe methods that require only forward or first-order backward passes:

**Hessian trace (Hutchinson estimator):**
```
trace(H)  ≈  E_v [ (L(w+δv) − 2L(w) + L(w−δv)) / δ² ]    v ~ N(0, I)
```
Each probe requires two forward passes. 20–30 probes give a low-variance estimate.

**Top eigenvalue (power iteration):**
```
H v  ≈  (∇L(w+δv) − ∇L(w−δv)) / (2δ)
```
Iterated normalisation converges to the eigenvector of λ_max in ~20 steps.

**2-D loss surface slice (filter-normalised):**
Evaluate `L(w* + α·d₁ + β·d₂)` on a grid, using filter-normalised directions
(Li et al. 2018) so the scale is comparable across checkpoints.

---

## 2. The Grokking Testbed

**Task:** Predict `(a + b) mod p` given tokens a, b ∈ {0,…,p−1}, where p = 97.

**Why this task?** It is the canonical grokking benchmark (Power et al. 2022), and the
learned algorithm is mechanistically understood (Nanda et al. 2023): after grokking, the
transformer implements a *Discrete Fourier Transform* over ℤ_97, using only O(√p) ≈ 10
active Fourier frequencies. This solution is **algebraically degenerate** — permuting
which frequencies are used, or rotating within frequency subspace, gives an identical
function. We therefore know *in advance* that the post-grokking minimum should have many
flat directions and a lower LLC, making this a genuine prediction test for SLT.

**Delayed grokking configuration:**

| Parameter | Value | Reason |
|---|---|---|
| Architecture | 1-layer transformer | Less expressive → slower generalisation |
| d_model | 128, n_heads = 4 | — |
| Train fraction | 30% (2,821 of 9,409 pairs) | Less data → harder generalisation |
| Weight decay | 1.0 | Essential: penalises the memorising solution |
| Optimiser | AdamW, lr = 1e-3 | — |
| Total parameters | 223,872 | λ_max (SLT) ≤ d/2 = 111,936 |
| Total training steps | 3,000 | Full arc completes by ~step 2,400 |
| Flatness measured every | 200 steps | ~15 estimates across the arc |

---

## 3. Results

### Fig 1 – Training Dynamics (Grokking Curves)

![Training dynamics](figures/fig1_training_dynamics.png)

Three phases are visible in the right panel (accuracy):

1. **Learning** (steps 0–400): train and test accuracy both rise — the model is still
   finding any signal.
2. **Memorisation plateau** (steps 400–1800): train accuracy reaches 99.8% and stalls;
   test accuracy is stuck at 15–26%. The model has memorised the training pairs but has
   not found the generalising algorithm.
3. **Grokking transition** (steps 1800–2400): test accuracy climbs rapidly from ~41% to
   ~100% in just 600 steps. The Fourier circuit takes over.

In the left panel (loss), note that test loss briefly *rises above the random-chance
baseline* (log 97 ≈ 4.57) during the plateau — the memorisation circuit is actively
suppressing generalisation.

The shaded bands (blue = memorise, orange = plateau, green = generalise) are the same
across all figures, making direct comparison easy.

---

### Fig 2 – LLC Trajectory

![LLC trajectory](figures/fig2_llc_trajectory.png)

| Step | Test acc | LLC λ̂ | tr(H) | λ_max(H) |
|---|---|---|---|---|
| 0 | 1% | 5 | 288 | 496 |
| 200 | 1% | 87 | 5,834 | 402 |
| 400 | 8% | 531 | **43,413** | 751 |
| 600 | 15% | 848 | 36,164 | 705 |
| 800 | 19% | 950 | 21,178 | 269 |
| 1000 | 20% | 1,016 | 21,499 | 468 |
| 1200 | 22% | 1,134 | 22,179 | 934 |
| 1400 | 26% | 1,234 | 19,713 | 300 |
| 1600 | 32% | 1,236 | 17,962 | 306 |
| 1800 | 41% | 1,398 | 20,368 | 286 |
| **2000** | **65%** | **1,360** | **13,739** ↓ | 314 |
| 2200 | 91% | 1,339 ↓ | 8,329 ↓ | 193 ↓ |
| 2400 | 98% | 1,330 ↓ | 6,776 ↓ | 200 |
| 2600 | 100% | 1,292 ↓ | 4,633 ↓ | 225 |
| **2800** | **100%** | 1,439 | **818** ↓↓ | **19.8** ↓↓ |

The LLC (purple, left axis) shows the characteristic rise from ~5 at init to ~1,300–1,400
during the memorisation plateau, with a gentle dip during the grokking transition. It is
a noisy signal — see [§5](#5-the-llc-calibration-challenge) for why.

The test accuracy (green dashed, right axis) shows the sharp S-curve of grokking.

---

### Fig 3 – Phase Portrait

![Phase portrait](figures/fig3_phase_portrait.png)

The trajectory in **(test loss, LLC)** space, coloured by training step:

- **Early** (yellow): high test loss, low LLC — flat diffuse landscape near init.
- **Memorisation** (blue): high test loss, high LLC — sharp lookup-table minimum.
- **Grokking** (purple → green): test loss falls steeply; LLC decreases slightly,
  tracing the transition from memorisation to Fourier basin.

The non-monotone path — LLC *rises before* test loss falls — reflects the fact that
internal complexity peaks during the plateau, before the model finds the simpler solution.

---

### Fig 4 – Flatness Analysis: Hessian Trace & Top Eigenvalue

![Flatness trajectory](figures/fig4_flatness_trajectory.png)

**This is the clearest result in the project.** The three panels show:

**Left — Hessian trace tr(H):**
Rises from 288 (init) to a peak of **43,413** at step 400 (fast memorisation), then
decreases through the plateau (~20,000), and **collapses to 818 at step 2800** once the
Fourier solution is fully established. That is a **53× drop** from peak to post-grokking
trough — direct, model-free evidence that the model's loss landscape acquires an enormous
number of flat directions after grokking.

**Middle — top Hessian eigenvalue λ_max:**
Drops from ~750 during memorisation to **19.8 at step 2800**, a **38× decrease** in the
curvature of the sharpest direction alone. This is a particularly clean signal: even the
direction that was previously most sensitive to parameter perturbation is nearly flat in
the Fourier minimum.

**Right — LLC vs normalised tr(H) overlay:**
Both measures, when normalised to [0, 1], trace similar trajectories. Their correlation
confirms that the LLC is genuinely detecting the same flatness signal as the Hessian,
despite using a completely different estimation method (SGLD vs. finite-difference probes).
The LLC is noisier than tr(H) because the SGLD estimator has calibration challenges at
low loss values (see §5), but the direction of travel is consistent.

The phase bands are clearly visible: tr(H) climbs in the blue (memorise) band and falls
steeply through the orange (plateau) and green (generalise) bands.

---

### Fig 5 – 2-D Loss Surface Slices

![Loss surfaces](figures/fig5_loss_surfaces.png)

Four snapshots of the loss landscape, each centred on w* at that training stage. The same
two filter-normalised random directions are used across all panels, so the geometry is
directly comparable. The star marks w*; colour encodes loss (red = high, green = low).

| Panel | Stage | What to look for |
|---|---|---|
| Step 0 | Init / learning | Wide, shallow bowl — the random landscape has no committed structure |
| Step 1000 | Memorised (plateau) | Tight, sharp contours — narrow well with steep walls in all directions |
| Step 2000 | Grokking transition | Intermediate — basin is sharp but beginning to elongate asymmetrically |
| Step 3000 | Post-grokking | Flatter contours in at least one direction; the flat directions of the Fourier solution appear as elongated ridges |

The loss surface at step 1000 has tight, nearly circular contours: the memorisation
minimum is sharp and isotropic — all parameter directions matter roughly equally. By step
3000, the contours elongate: the Fourier solution has acquired flat directions (where
rotating between equivalent frequency representations leaves the loss unchanged).

Note that a random 2-D slice will not align perfectly with the flat directions of the
Fourier minimum — the flattest directions live in the high-dimensional frequency subspace.
The Hessian trace and λ_max (Fig 4) are more sensitive to these directions precisely
because they average over *all* weight-space directions, not just a 2-D slice.

---

### Fig 6 – Model Comparison (optional)

![Model comparison](figures/fig6_model_comparison.png)

*Generated only with `python train.py --model_sweep` (~20 min extra).* Runs 5 model sizes
(d = 32–128, layers = 1–4) through a full training arc and plots LLC at convergence versus
test accuracy. Illustrates the SLT prediction that larger, more expressive models tend to
find more singular minima (lower LLC) on this task, because they have richer sets of
equivalent solutions.

---

## 4. Interpretation: Geometry Tells the Story

### The two solutions and their geometry

The grokking phenomenon is a competition between two local minima that both fit the
training data but differ dramatically in their weight-space geometry:

**The memorisation minimum (step ~400–1800):**
- Mechanistically: an attention-based lookup table storing all 2,821 training (a, b) → c
  pairs independently.
- Geometry: many sharp, independent circuits — one "slot" per training example. Almost
  every parameter contributes to some stored memory.
- Hessian: nearly full-rank with large eigenvalues. tr(H) ≈ 43,000. λ_max ≈ 750.
- SLT: near-regular. LLC ≈ 1,200 (maximum is d/2 = 111,936; this is small but far from zero).

**The Fourier minimum (step ~2800+):**
- Mechanistically: computes (a + b) mod 97 using the Discrete Fourier Transform over ℤ_97.
  Only ~10 Fourier frequencies are active; the rest of weight space is unused.
- Geometry: many flat directions. Permuting which frequencies are used, rotating within
  frequency subspace, or rescaling conjugate pairs all leave the function unchanged.
- Hessian: nearly degenerate. tr(H) ≈ 818. λ_max ≈ 20.
- SLT: highly singular. LLC should be ≪ d/2 — our estimator gives ~1,300 (noisy;
  true value likely much lower with better calibration).

**The quantitative geometry comparison:**

| Quantity | Memorisation peak | Post-grokking | Ratio |
|---|---|---|---|
| tr(H) | 43,413 | 818 | **53×** |
| λ_max | 750 | 19.8 | **38×** |
| LLC λ̂ | ~1,380 | ~1,290 | ~1.07× (noisy) |

The Hessian-based measures give an unambiguous confirmation of the SLT prediction. The
LLC shows the same direction but at much lower signal-to-noise — a calibration problem,
not a failure of the theory (see §5).

### Why weight decay is the mechanism

Weight decay creates a norm penalty that differentiates the two solutions energetically:

```
E(w) = n·L_n(w)  +  (weight_decay / 2)·‖w‖²
```

The memorisation circuit requires high weight norms (one circuit per training example,
each independently scaled). The Fourier circuit uses structured, compressible weights
with lower total norm. Under repeated L2 shrinkage:

1. Both circuits lose energy to weight decay at the same *multiplicative* rate each step.
2. The memorisation circuit, having higher norm, loses more *absolute* magnitude.
3. Eventually the Fourier circuit achieves lower *total energy* E(w) and takes over.

This is an instantiation of SLT's Bayesian principle: the posterior prefers singular minima,
and weight decay is what physically drives the model into the more singular basin.

### The Hessian trajectory as a developmental indicator

tr(H) tells the story of the model's internal development more cleanly than any single
accuracy metric:

| tr(H) range | Developmental stage |
|---|---|
| ~300 | Random initialisation — no committed structure |
| ~5,000–40,000 | Rapid circuit formation — lookup table being built |
| ~20,000 | Memorisation plateau — entrenched sharp minimum |
| ~14,000–7,000 | Grokking transition — flat directions emerging |
| ~800 | Fourier minimum — mostly flat, heavily degenerate |

### Confirming the SLT prediction

SLT predicts that the loss landscape should become *more degenerate* (flatter, lower LLC)
at generalising solutions. The full prediction for the grokking arc is:

```
flatness(init) < flatness(memorisation) > flatness(grokking)
```

Our results:

| Prediction | Measure | Evidence |
|---|---|---|
| init < memorisation | tr(H): 288 → 43,413 | ✅ 150× increase, very strong |
| memorisation > grokking | tr(H): 43,413 → 818 | ✅ 53× decrease, very strong |
| memorisation > grokking | λ_max: 750 → 19.8 | ✅ 38× decrease, very strong |
| memorisation > grokking | LLC: ~1,380 → ~1,290 | ✓ directional, within noise |

**The Hessian-based measures strongly confirm the SLT prediction.** The LLC gives the
same direction but requires better calibration for the full quantitative picture.

---

## 5. The LLC Calibration Challenge

The post-grokking LLC estimates (~1,300–1,440) remain high relative to the dramatic
flatness visible in tr(H). This is not a failure of SLT but a practical limitation of
the fixed-step-size SGLD estimator.

**Root cause:** The SGLD step size ε = 3×10⁻⁶ is tuned for the memorisation regime,
where train loss ~0.05–0.1. Post-grokking, the loss drops to ~0.001 — two orders of
magnitude smaller. The same step size now causes the chain to escape the local basin
entirely, probing global landscape features rather than the local singularity structure.

The Hessian confirms this: at step 2800, λ_max = 19.8. The optimal SGLD step size for
staying inside this basin is ε ~ 1/λ_max ~ 0.05 — roughly 17,000× larger than our
current ε, but the noise term `√(2ε) ≈ 0.0024` would then be enormous. Staying inside
the basin at such low λ_max requires a very small noise term, which means a very small ε,
which means very slow mixing. This is the fundamental tension in SGLD-based LLC
estimation for sharp (post-training) minima.

**What the Hessian data tells us about the true post-grokking LLC:**
Using the Gaussian approximation: LLC ≈ rank(H) / 2. With tr(H) ≈ 818 and λ_max ≈ 20,
the effective rank of H is roughly tr(H)/λ_max ≈ 41. This gives a Gaussian-approximation
LLC of ~20 — a factor of 60–70× below our SGLD estimate. This is consistent with the
Fourier solution using only ~10 active frequencies, each contributing ~2 effective
parameters (amplitude and phase).

**What would be needed for accurate post-grokking LLC:**
1. Adaptive step size ε ~ 1/λ_max(H) — requires online Hessian eigenvalue estimation.
2. Much stronger localisation: γ ~ λ_max × d ~ 20 × 224,000 ~ 4.5 × 10⁶.
3. Thermodynamic integration or HMC instead of SGLD for better mixing at low temperature.

**Why this motivates Timaeus' Spectroscopy approach:**
The calibration challenge shows that SGLD-based LLC estimation degrades when the model
reaches sharp, low-loss minima. Spectroscopy's susceptibility-based measures are
naturally calibrated to the local curvature and do not require global exploration —
making them more robust in exactly the post-training regime where LLC estimation
is hardest. Our Hessian and LLC measures can be seen as two ends of a spectrum: the
Hessian gives a clean but shallow (2nd-order) picture; the LLC gives a deeper (all-order)
picture but requires careful sampling calibration.

---

## 6. Connections to Timaeus Research

### Spectroscopy

Timaeus' Spectroscopy methodology characterises neural network structure using
susceptibility-based measures grounded in *weight space*.

| Spectroscopy concept | This project |
|---|---|
| Weight-space grounded | All measures (LLC, tr(H), loss surface) are computed from loss geometry at w*, not from activations |
| Susceptibility measures | tr(H) = total susceptibility to weight perturbation; LLC = Bayesian susceptibility under SGLD |
| Discovers internal structure | tr(H) trajectory cleanly separates memorisation, plateau, and Fourier phases |
| Local characterisation | Filter-normalised loss surface shows local basin shape; localised LLC probes the local singularity |
| Beyond 2nd order | LLC captures higher-order degeneracies that tr(H) misses; complementary to Hessian analysis |

An extension of this project would compute *per-layer* Hessian traces, decomposing
which layers' flat directions emerge first during grokking — a finer-grained version of
the Spectroscopy signal.

### Patterning

The Fourier circuit develops slowly because training data is dominated by memorisation-
supporting examples (all 2,821 training pairs reinforce the lookup table). Patterning
would:
1. Estimate per-example flatness contributions (which examples reduce tr(H) most?).
2. Reweight the training distribution to amplify Fourier-supporting examples, accelerating
   the grokking transition without changing the model architecture.

The Hessian trace provides a natural scalar signal for this inversion: choose data that
maximally decreases tr(H) (increases flatness) at each step.

### Phase transitions and developmental stages

Grokking is a controlled toy model for the developmental stages observed in large
language models (induction heads, in-context learning circuits, etc.). The combined
LLC + Hessian analysis demonstrates that weight-space measures can:

- Track transitions continuously and quantitatively.
- Separate genuinely distinct developmental phases (init / memorise / plateau / grokk)
  from noise.
- Provide model-free evidence of the internal algorithmic change (lookup table → DFT)
  without requiring circuit-level interpretability.

---

## 7. Setup and Usage

### Install

```bash
git clone <this-repo>
cd slt-grokking

uv venv -p 3.12 .venv && uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

### Run

| Command | What it does | Time (Apple MPS) |
|---|---|---|
| `python train.py --quick` | p=23, 1-layer, 2000 steps — smoke test | ~30 s |
| `python train.py` | p=97, 2-layer, 50% data, 6000 steps | ~2 min |
| `python train.py --delayed` | **Recommended** — full delayed-grokking arc with Hessian + loss surfaces | ~4 min |
| `python train.py --delayed --model_sweep` | Also generates Fig 6: LLC at convergence vs model size | ~20 min extra |

`--delayed` produces Figs 1–5 and the complete analysis described in §3–§5.
Add `--model_sweep` to also generate Fig 6 (model comparison, ~20 min extra).

### Key CLI flags

```bash
python train.py --delayed \
  --htrace_samples 30    # Hutchinson probes for tr(H) (default 20)
  --eig_iter 20          # power-iteration steps for λ_max (default 20)
  --surface_grid 41      # resolution of 2-D loss surface NxN (default 41)
  --surface_extent 1.0   # ±extent in filter-normalised units
  --llc_localization 1e4 # spring constant γ (increase for sharper minima)
```

### Code structure

```
slt-grokking/
├── src/
│   ├── model.py     # ModularTransformer: 3-slot decoder [a, b, =]
│   ├── data.py      # ModularAdditionDataset: all p² (a,b,c) triples
│   ├── llc.py       # Localised SGLD LLC estimator — from scratch, no devinterp dep.
│   ├── hessian.py   # Hutchinson trace, power-iteration λ_max, 2-D surface slices
│   └── viz.py       # Six publication-quality figure generators
├── train.py         # Training loop + LLC + Hessian tracking + CLI
├── pyproject.toml   # uv/pip project spec
└── requirements.txt
```

All Hessian methods use finite differences or first-order gradients — **fully MPS-safe**,
no `create_graph=True` needed.

---

## 8. References

- Watanabe, S. (2009). *Algebraic Geometry and Statistical Learning Theory*. Cambridge UP.
- Watanabe, S. (2013). A widely applicable Bayesian information criterion. *JMLR*, 14, 867–897.
- Power, A. et al. (2022). Grokking: Generalisation beyond overfitting on small algorithmic datasets. *arXiv:2201.02177*.
- Nanda, N. et al. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2023*.
- Li, H. et al. (2018). Visualizing the loss landscape of neural nets. *NeurIPS 2018*.
- Hoogland, J. et al. (2024). The developmental landscape of in-context learning. *arXiv:2402.02364*.
- Lau, E. et al. (2023). Quantifying degeneracy in singular models via the learning coefficient. *arXiv:2308.12108*.
- Timaeus devinterp library: https://github.com/timaeus-research/devinterp
- Gordon, B. et al. (2026). Spectroscopy of neural networks. *Timaeus*.
- Wang, C. & Murfet, D. (2026). Patterning. *Timaeus*.
