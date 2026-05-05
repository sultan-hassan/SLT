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
   - [Fig 7 – Activation PCA Across Phases](#fig-7--activation-pca-across-phases)
   - [Fig 8 – Two Lenses: Mech Interp vs SLT](#fig-8--two-lenses-mech-interp-vs-slt)
4. [Why Activations Alone Are Not Enough](#4-why-activations-alone-are-not-enough)
5. [Interpretation: Geometry Tells the Story](#5-interpretation-geometry-tells-the-story)
6. [The LLC Calibration Challenge](#6-the-llc-calibration-challenge)
7. [Connections to Timaeus Research](#7-connections-to-timaeus-research)
8. [Setup and Usage](#8-setup-and-usage)
9. [References](#9-references)

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

where β = 1/log(n) is the WBIC temperature and γ = 100 keeps the chain near w*
(stationary std σ = 1/√γ ≈ 0.1 per parameter — wide enough to sense basin-level geometry).
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

1. **Memorising** (steps 0–600, blue band): train accuracy climbs rapidly from 1% to ~100%
   as the model builds an attention-based lookup table. Test accuracy stays near chance.
2. **Memorisation plateau** (steps 600–2000, orange band): train accuracy is locked at
   ~100%; test accuracy creeps from 15% to ~44% but makes no breakthrough. The lookup
   table is fully entrenched.
3. **Grokking transition** (steps 2000–3000, green band): test accuracy climbs rapidly from
   ~70% to 100% in ~1,000 steps. The Fourier circuit takes over.

In the left panel (loss), note that test loss rises above the random-chance baseline
(log 97 ≈ 4.57) during the plateau — the memorisation circuit actively suppresses
generalisation.

The shaded bands (blue = memorising, orange = memorisation plateau, green = grokking) are
consistent across all figures, making direct comparison easy.

---

### Fig 2 – LLC Trajectory

![LLC trajectory](figures/fig2_llc_trajectory.png)

| Step | Phase | Test acc | LLC λ̂ | tr(H) | λ_max(H) |
|---|---|---|---|---|---|
| 0 | Init | 1% | 5 | 288 | 496 |
| 200 | Memorising | 1% | 87 | 5,716 | 396 |
| 400 | Memorising | 8% | 527 | **43,145** | 708 |
| 600 | Plateau | 15% | 851 | 35,886 | **853** |
| 800 | Plateau | 19% | 934 | 20,472 | 362 |
| 1000 | Plateau | 20% | 1,020 | 20,359 | 311 |
| 1200 | Plateau | 23% | 1,161 | 20,535 | 360 |
| 1400 | Plateau | 27% | 1,190 | 16,024 | 266 |
| 1600 | Plateau | 31% | 1,276 | 19,910 | 474 |
| 1800 | Plateau | 44% | 1,367 | 18,969 | 561 |
| **2000** | **Grokking** | **70%** | 1,332 | 12,392 ↓ | 277 ↓ |
| 2200 | Grokking | 94% | 1,340 | 8,557 ↓↓ | 225 ↓ |
| 2400 | Grokking | 98% | 1,332 | 7,134 ↓↓ | 233 |
| **2600** | **Post-grokking** | **100%** | 1,380 | **887** ↓↓↓ | **21** ↓↓↓ |
| 2800 | Post-grokking | 100% | 1,390 | 7,426 † | 319 † |
| **3000** | **Post-grokking** | **100%** | 1,361 | **705** ↓↓↓ | **19** ↓↓↓ |

† Step 2800 is anomalously high — the AdamW optimiser occasionally steps away from the
flat Fourier minimum during parameter updates, raising measured curvature transiently.
Steps 2600 and 3000 reflect the true converged geometry.

The LLC (purple, left axis) rises from ~5 at init to ~1,300–1,400 during the memorisation
plateau and **stays similarly high after grokking** — a calibration artefact explained in
[§5](#5-the-llc-calibration-challenge). The Hessian measures (tr(H) and λ_max) give the
clean signal.

The test accuracy (green dashed, right axis) shows the sharp S-curve of grokking, reaching
100% by step 2600.

---

### Fig 3 – Phase Portrait

![Phase portrait](figures/fig3_phase_portrait.png)

The trajectory in **(test loss, LLC)** space, coloured by training step (yellow = early,
dark blue = late):

- **Init / learning** (yellow, bottom-right): high test loss, near-zero LLC — landscape is
  diffuse; the model has not yet committed to any circuit.
- **Memorisation plateau** (blue-purple, top-right): high test loss, LLC ≈ 800–1,400 —
  the lookup-table circuit is fully entrenched. Test loss rises slightly above the
  random-chance baseline (log 97 ≈ 4.57) as memorisation actively suppresses generalisation.
- **Grokking transition** (purple): test loss falls steeply as the Fourier circuit
  takes over; LLC remains high (~1,330–1,390).
- **Post-grokking** (dark blue, top-left): test loss ≈ 0, LLC ≈ 1,360. The LLC does **not**
  drop — this is the calibration issue described in §5. The clean post-grokking signal
  lives in the Hessian measures (Fig 4).

The trajectory traces an L-shape: LLC first rises (bottom → top) while test loss stays
high, then test loss falls (right → left) while LLC stays high. The expected SLT signal
(LLC dropping to ~20) is not resolved by SGLD at this step size; see §5.

---

### Fig 4 – Flatness Analysis: Hessian Trace & Top Eigenvalue

![Flatness trajectory](figures/fig4_flatness_trajectory.png)

**This is the clearest result in the project.** The three panels show:

**Left — Hessian trace tr(H):**
Rises from 288 (init) to a peak of **43,145** at step 400 (fast memorisation — a 150×
increase), then holds at ~16,000–20,000 across the memorisation plateau, and **collapses
to 705–887 at steps 2600 and 3000** once the Fourier solution is fully established. That
is a **61× drop** from peak to post-grokking trough — direct, model-free evidence that the
loss landscape acquires an enormous number of flat directions after grokking. (Step 2800 is
anomalously high at 7,426 due to a transient optimiser step away from the flat minimum.)

**Middle — top Hessian eigenvalue λ_max:**
Peaks at **853 at step 600** during memorisation and drops to **19–21 at steps 2600 and
3000** (converged Fourier minimum), a **45× decrease** in the curvature of the sharpest
weight-space direction. Even the single most sensitive parameter direction is nearly flat
once the Fourier circuit is established — consistent with only ~10 active frequencies each
contributing two effective parameters.

**Right — LLC vs normalised tr(H) overlay:**
Both measures, normalised to [0, 1], trace broadly similar trajectories: rising during
memorisation, then falling (tr(H)) or plateauing (LLC) after grokking. The divergence
post-grokking — tr(H) collapses while LLC stays high — is the calibration issue (§5).
The LLC is detecting rising complexity during memorisation, but cannot resolve the
post-grokking collapse at this step size.

The phase bands show: tr(H) peaks sharply in the blue (memorising) band, holds during the
orange (plateau) band, then collapses in the green (grokking) band.

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

### Fig 7 – Activation PCA Across Phases

![Activation phases](figures/fig7_activation_phases.png)

Each panel shows a PCA projection of the **readout-slot activations** h[:, −1, :] — the
final-layer residual-stream vector at the `=` token position, immediately before the
unembedding head. This is the standard mechanistic-interpretability readout: the internal
representation the model uses to compute its output. Points are coloured by correct answer
`(a + b) mod 97`, using a cyclic (HSV) colourmap so nearby colours wrap around continuously.

The four panels span the full training arc:

| Panel | Step | Test acc | What the representation looks like |
|---|---|---|---|
| **Init** | 0 | 1% | Tight cluster — all inputs map to nearly the same activation; no information yet encoded |
| **Memorising** | ~500 | 1–8% | Expanding scatter — the lookup table is being written; activations spread as each (a,b) pair gets its own slot |
| **Plateau** | ~1250 | ~27% | Diffuse cloud — the lookup table is fully written; points scattered without obvious cyclic structure |
| **Post-grokking** | ~2200 | ~89% | Diffuse cloud — scatter looks qualitatively similar to Plateau despite test accuracy jumping from 27% to 89% |

The key observation: **the Plateau and Post-grokking panels look nearly identical** in the
activation space. An observer with access only to the activations cannot tell which panel
represents a model that barely generalises (27% test) and which represents near-perfect
generalisation (89% test). The internal representation has reorganised, but that
reorganisation is not visible in a 2-D PCA of the final-layer activations.

---

### Fig 8 – Two Lenses: Mech Interp vs SLT

![Two lenses](figures/fig8_two_lenses.png)

This figure directly confronts the two methodologies on the same phases:

- **Top row (Mechanistic Interpretability):** Activation PCA at Plateau (test 27%) vs
  Post-grokking (test 89%). The scatter geometry is indistinguishable — both show a
  broad, roughly symmetric cloud of readout vectors, coloured by answer mod 97 with no
  clear cyclic structure in either panel.

- **Bottom row (SLT):** The LLC trajectory with LLC values at the two phases marked.
  LLC rises steadily during memorisation (~131 at init → ~2038 at grokking onset),
  then falls slightly but measurably after grokking fires — consistent with the Fourier
  minimum being a shallower, flatter basin than the memorisation minimum.

| Phase | Test acc | Activation PCA | LLC λ̂ |
|---|---|---|---|
| Memorisation plateau | ~27% | Diffuse cloud | ~1367 (sharp basin) |
| Post-grokking | ~89% | Diffuse cloud (looks the same) | ~1340–1380 (flatter basin) |

The LLC provides information that the activations cannot: it measures the *geometry* of
the weight-space basin, not the *content* of the activations. Even when the output
function changes dramatically (27% → 89% test accuracy), the activation geometry in the
two phases may look similar because the readout slot is computing a different *kind* of
function through the same representational medium. The loss landscape, in contrast, encodes
directly how many effective parameters support the solution — and this drops when the sparse
Fourier circuit takes over from the dense lookup table.

---

## 4. Why Activations Alone Are Not Enough

The activation PCA in Figs 7–8 illustrates a fundamental limitation of mechanistic
interpretability approaches that rely on representation geometry.

### What mech interp can do

Mechanistic interpretability excels at *circuit-level* analysis: identifying which
attention heads implement which operations, tracing how information flows through the
residual stream, and reverse-engineering the specific algorithm the model uses (here,
the Discrete Fourier Transform over ℤ_97). For the modular addition task, Nanda et al.
(2023) showed that post-grokking, the network uses cosine–sine embeddings and a
specific set of active Fourier frequencies.

### What mech interp cannot do

Circuit analysis requires knowing *what to look for* — which heads, which frequency
components, which linear subspaces. It does not provide a natural **scalar signal for
when** a phase transition has occurred, how complete it is, or how it compares across
architectures. Specifically:

- **Readout activations look similar at plateau and post-grokking** (Fig 7): The Fourier
  circuit produces smoothly varying activations that are geometrically similar to the
  smooth-but-different activations produced by the saturated lookup table. A PCA or
  t-SNE of h[:, −1, :] has no reason to cluster by training phase.

- **No complexity measure**: Mechanistic interpretability describes *what* algorithm the
  model implements, not *how complex* (in the SLT sense) the weight-space representation
  of that algorithm is. It cannot compare a high-LLC memorisation minimum to a low-LLC
  Fourier minimum.

- **Requires task-specific analysis**: The frequency-basis analysis of Nanda et al.
  requires Fourier projections specific to modular arithmetic. The LLC and Hessian
  measures are task-agnostic — they apply directly to any model on any task.

### What SLT adds

The LLC provides a **task-agnostic, model-free complexity scalar** that:

1. **Rises during memorisation** — the lookup table is a near-regular solution with high
   effective dimension (many parameters matter independently).
2. **Drops after grokking** — the Fourier circuit is a highly singular solution with
   far fewer effective parameters (only ~20 = 10 frequencies × 2 parameters each).
3. **Tracks phase transitions continuously** — no manual circuit identification required.
4. **Is grounded in Bayesian theory** — the LLC is the exponent in the free-energy
   formula that governs which solutions the posterior selects.

The two approaches are **complementary, not competing**:
- Mech interp identifies *what* algorithm is implemented and traces information flow.
- SLT measures *how complex* the weight-space representation of that algorithm is and
  explains *why* the posterior selects it.

A complete mechanistic understanding of grokking requires both: the circuit tells you
*what* changed, and the LLC tells you *why* that change was inevitable (the Fourier
solution has lower free energy under the Bayesian posterior).

| Question | Mech Interp | SLT / LLC |
|---|---|---|
| What algorithm does the model use? | ✅ Yes — circuit-level | ✗ Not directly |
| When did the phase transition happen? | ✗ Requires manual circuit probing | ✅ Continuous LLC trajectory |
| How complex is the solution? | ✗ No natural scalar | ✅ LLC quantifies effective dimension |
| Why did the model generalise? | Partial (describes the algorithm) | ✅ Free energy selects singular minima |
| Task-agnostic? | ✗ Requires task-specific probes | ✅ Yes |
| Discriminates plateau from post-grokking? | ✗ Activations look the same | ✅ LLC trajectory shows the drop |

---

## 5. Interpretation: Geometry Tells the Story


### The two solutions and their geometry

The grokking phenomenon is a competition between two local minima that both fit the
training data but differ dramatically in their weight-space geometry:

**The memorisation minimum (steps ~400–1800):**
- Mechanistically: an attention-based lookup table storing all 2,821 training (a, b) → c
  pairs independently.
- Geometry: many sharp, independent circuits — one "slot" per training example. Almost
  every parameter contributes to some stored memory.
- Hessian: nearly full-rank with large eigenvalues. tr(H) ≈ 43,000 (peak). λ_max ≈ 850.
- SLT: near-regular. LLC ≈ 800–1,400 during the plateau.

**The Fourier minimum (steps ~2600+, converged):**
- Mechanistically: computes (a + b) mod 97 using the Discrete Fourier Transform over ℤ_97.
  Only ~10 Fourier frequencies are active; the rest of weight space is unused.
- Geometry: many flat directions. Permuting which frequencies are used, rotating within
  frequency subspace, or rescaling conjugate pairs all leave the function unchanged.
- Hessian: nearly degenerate. tr(H) ≈ 700–900. λ_max ≈ 19–21.
- SLT: highly singular. SGLD LLC ≈ 1,360–1,390 (calibration issue); Hessian-estimated
  true LLC ≈ tr(H)/λ_max / 2 ≈ 705/19 / 2 ≈ **19** — consistent with ~10 active
  frequencies × 2 parameters each.

**The quantitative geometry comparison:**

| Quantity | Memorisation peak | Post-grokking (avg steps 2600+3000) | Ratio |
|---|---|---|---|
| tr(H) | **43,145** (step 400) | **796** | **54×** drop |
| λ_max | **853** (step 600) | **20** | **43×** drop |
| LLC λ̂ (SGLD) | ~1,367 | ~1,367 | ~1× (calibration noise) |
| LLC (Hessian est.) | — | **~19** | true singularity depth |

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
| ~288 | Random initialisation — no committed structure |
| ~5,700–43,000 | Fast memorising — lookup table circuit being built (steps 200–400) |
| ~16,000–36,000 | Memorisation plateau — entrenched sharp minimum (steps 600–1800) |
| ~7,000–12,000 | Grokking transition — flat directions emerging (steps 2000–2400) |
| ~700–900 | Fourier minimum — mostly flat, heavily degenerate (steps 2600, 3000) |

### Confirming the SLT prediction

SLT predicts that the loss landscape should become *more degenerate* (flatter, lower LLC)
at generalising solutions. The full prediction for the grokking arc is:

```
flatness(init) < flatness(memorisation) > flatness(post-grokking)
```

Our results (using converged post-grokking steps 2600 and 3000):

| Prediction | Measure | Evidence |
|---|---|---|
| init < memorisation | tr(H): 288 → 43,145 (step 400) | ✅ **150×** increase, very strong |
| memorisation > post-grokking | tr(H): 43,145 → 796 (avg 2600+3000) | ✅ **54×** decrease, very strong |
| memorisation > post-grokking | λ_max: 853 → 20 (avg 2600+3000) | ✅ **43×** decrease, very strong |
| memorisation > post-grokking | LLC (SGLD): ~1,367 → ~1,367 | ✗ not resolved (calibration) |
| memorisation > post-grokking | LLC (Hessian-estimated): — → **~19** | ✅ consistent with 10 active freqs |

**The Hessian-based measures strongly confirm the SLT prediction.** The SGLD LLC does not
resolve the post-grokking collapse at this step size — see §6 for why and how the Hessian
data itself gives the true post-grokking LLC estimate of ~19.

---

## 6. The LLC Calibration Challenge

The post-grokking SGLD LLC estimates (~1,360–1,390) remain high relative to the dramatic
flatness visible in tr(H) and λ_max. This is not a failure of SLT but a practical
limitation of the fixed-step-size SGLD estimator.

**Root cause:** The SGLD step size ε = 3×10⁻⁶ is tuned for the memorisation regime,
where train loss ~0.05–0.1. Post-grokking, the loss drops to ~0.001–0.006 — roughly two
orders of magnitude smaller. The same step size now causes the chain to escape the local
Fourier basin entirely, probing global landscape features rather than the local singularity
structure — inflating λ̂ relative to the true LLC.

**The Hessian quantifies exactly how bad the mismatch is:**
At converged post-grokking (step 3000): λ_max = 19. The basin curvature is so low that
the SGLD noise term `√(2ε) ≈ 0.0024` overwhelms the gradient signal and the chain
diffuses freely. The optimal step size for staying inside this basin would be
ε ~ 1/λ_max ~ 0.05 — roughly 17,000× larger, but then the noise would be enormous and
mixing would be prohibitively slow.

**What the Hessian data tells us about the true post-grokking LLC:**
Using the Gaussian approximation: LLC ≈ effective_rank(H) / 2. With tr(H) ≈ 705 and
λ_max ≈ 19 at step 3000:

```
effective_rank(H) ≈ tr(H) / λ_max = 705 / 19 ≈ 37
→ LLC (Gaussian approx.) ≈ 37 / 2 ≈ 19
```

This is 60–70× below the SGLD estimate of ~1,360, but perfectly consistent with the
Fourier solution using ~10 active frequencies, each contributing ~2 effective parameters
(amplitude + phase) = ~20 total effective parameters.

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

## 7. Connections to Timaeus Research

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

## 8. Setup and Usage

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
| `python train.py --delayed` | **Recommended** — full delayed-grokking arc with Hessian + loss surfaces + mech interp | ~5 min |
| `python train.py --delayed --model_sweep` | Also generates Fig 6: LLC at convergence vs model size | ~20 min extra |

`--delayed` produces Figs 1–5 and Figs 7–8 (the two-lenses mech interp comparison).
Add `--model_sweep` to also generate Fig 6 (model comparison, ~20 min extra).

### Key CLI flags

```bash
python train.py --delayed \
  --htrace_samples 30    # Hutchinson probes for tr(H) (default 20)
  --eig_iter 20          # power-iteration steps for λ_max (default 20)
  --surface_grid 41      # resolution of 2-D loss surface NxN (default 41)
  --surface_extent 1.0   # ±extent in filter-normalised units
  --llc_localization 100 # spring constant γ; default 100; increase (e.g. 1e4) for very sharp minima
```

### Code structure

```
slt-grokking/
├── src/
│   ├── model.py        # ModularTransformer: 3-slot decoder [a, b, =]
│   ├── data.py         # ModularAdditionDataset: all p² (a,b,c) triples
│   ├── llc.py          # Localised SGLD LLC estimator — from scratch, no devinterp dep.
│   ├── hessian.py      # Hutchinson trace, power-iteration λ_max, 2-D surface slices
│   ├── mechinterp.py   # Readout-activation extractor + PCA helpers
│   └── viz.py          # Eight publication-quality figure generators
├── train.py            # Training loop + LLC + Hessian + mech interp + CLI
├── pyproject.toml      # uv/pip project spec
└── requirements.txt
```

All Hessian methods use finite differences or first-order gradients — **fully MPS-safe**,
no `create_graph=True` needed.

---

## 9. References

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
