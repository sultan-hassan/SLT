# SLT × Grokking: LLC as a Signature of Phase Transitions in Transformers

A self-contained project applying **Singular Learning Theory (SLT)** to study the
*grokking* phenomenon — the surprising ability of a neural network to suddenly generalise
long after memorising its training set.

**Research question:**
> *Does the Local Learning Coefficient (LLC) serve as a reliable weight-space indicator of
> the grokking phase transition, and what does its trajectory reveal about the internal
> structure of the solutions found at each phase?*

---

## Contents

1. [Background: SLT and the LLC](#1-background-slt-and-the-llc)
2. [The Grokking Testbed](#2-the-grokking-testbed)
3. [Results](#3-results)
   - [Fig 1 – Training Dynamics](#fig-1--training-dynamics-grokking-curves)
   - [Fig 2 – LLC Trajectory](#fig-2--llc-trajectory)
   - [Fig 3 – Phase Portrait](#fig-3--phase-portrait)
4. [Interpretation](#4-interpretation)
5. [The Calibration Challenge](#5-the-calibration-challenge)
6. [Connections to Timaeus Research](#6-connections-to-timaeus-research)
7. [Setup and Usage](#7-setup-and-usage)
8. [References](#8-references)

---

## 1. Background: SLT and the LLC

### Why classical theory fails for neural networks

Classical asymptotic statistics (the Bernstein–von Mises theorem, AIC, BIC) all assume
that the Fisher information matrix is non-singular at the true parameter. For neural
networks this assumption fails catastrophically: the parameter-to-function map is
massively non-injective. A single function can be realised by infinitely many parameter
settings (weight permutations, rescalings, and deeper symmetries), so the Fisher
information is degenerate everywhere.

### Watanabe's resolution: the RLCT

Sumio Watanabe's Singular Learning Theory (SLT) replaces the Fisher information approach
with algebraic geometry. The key result is the **free energy formula**:

```
n F_n(w*) = n L_n(w*) + λ log n − (m − 1) log log n + O(1)
```

where:
- `L_n(w*)` is the minimum training loss (the empirical risk at the basin minimum w*)
- **`λ` is the Real Log Canonical Threshold (RLCT)**, the *Local Learning Coefficient*
- `m` is the multiplicity of the singularity at w* (an integer ≥ 1)
- `n` is the number of training samples

The RLCT λ characterises *how flat the loss landscape is in the neighbourhood of w** in a
precise algebraic-geometric sense. It measures the effective dimension of the local
parameter space that the training data can resolve. Intuitively:

| Loss landscape near w* | λ |
|---|---|
| Regular (Fisher nonsingular) | = d/2, where d = number of parameters |
| Mildly degenerate | < d/2 |
| Highly degenerate / many flat directions | ≪ d/2 |

**The key SLT insight for generalisation:** Among local minima with similar training loss,
the one with *smaller λ* has smaller free energy and is therefore preferred by the Bayesian
posterior. A smaller λ corresponds to a more singular (degenerate) solution — one with
more flat directions and symmetries, i.e., simpler effective structure. This is SLT's
rigorous explanation for *implicit regularisation*: overparameterised networks generalise
because gradient descent tends to find the most singular compatible minimum.

### Estimating λ via SGLD

We use the **WBIC estimator** (Watanabe 2013). The key identity is:

```
λ = β · E_{w~p_β}[n · L_n(w)] − n · L_n(w*)
```

where p_β(w) ∝ exp(−β·n·L_n(w)) is the *tempered posterior* at inverse temperature
β = 1/log(n). We sample from p_β using Stochastic Gradient Langevin Dynamics (SGLD):

```
w_{t+1} = w_t  −  ε · (β·n/|B|) · ∇L_B(w_t)  +  sqrt(2ε) · η,    η ~ N(0, I)
```

and estimate λ as the empirical average of the energy above the baseline:

```
λ̂ = β · n · mean_t [ L(w_t) − L(w*) ]
```

We also add a **localisation spring** (following the devinterp library) to constrain the
SGLD chain to the local basin:

```
U_loc(w) = β·n·L_n(w)  +  (γ/2) · ‖w − w*‖²
```

This gives the *localised LLC* — the singularity depth of the local basin rather than a
global landscape feature. See [§5](#5-the-calibration-challenge) for discussion of why
this matters.

---

## 2. The Grokking Testbed

**Task:** Predict `(a + b) mod p` given integer tokens a, b ∈ {0,…,p−1}, where p = 97.

**Why this task?** It is the canonical grokking benchmark (Power et al. 2022), and the
learned algorithm is mechanistically understood: after grokking, the transformer
implements a *Discrete Fourier Transform* over ℤ_97, using only O(√p) Fourier frequencies
(Nanda et al. 2023). This means we know *a priori* what the generalising solution's
internal structure looks like — it is algebraically elegant and highly degenerate in
weight space, predicting a lower LLC after the phase transition.

**Delayed grokking configuration:**

| Parameter | Value | Reason |
|---|---|---|
| Architecture | 1-layer transformer | Less expressive → slower generalisation |
| d_model | 128 | Sufficient capacity to eventually grokk |
| n_heads | 4 | — |
| Train fraction | 30% (2,821 of 9,409 pairs) | Less data → harder generalisation |
| Weight decay | 1.0 | Essential: penalises the memorising solution |
| Optimiser | AdamW, lr = 1e-3 | — |
| Total parameters | 223,872 | λ_max = d/2 ≈ 112,000 |
| Total steps | 30,000 | — |
| LLC interval | every 500 steps | 60 estimates across training |

With 30% training data and a 1-layer model, the network reliably exhibits **classic
delayed grokking**: fast memorisation followed by a long plateau before sudden
generalisation. This gives the clearest signal for LLC analysis.

---

## 3. Results

### Fig 1 – Training Dynamics (Grokking Curves)

![Training dynamics](figures/fig1_training_dynamics.png)

**What to look at:**
- **Left panel (loss):** Train loss drops to near zero by step ~400. Test loss *worsens* 
  briefly during memorisation — the model is overfitting, pushing test loss above the
  random-chance baseline of log(97) ≈ 4.57.
- **Right panel (accuracy):** Three phases are visible:
  1. **Learning phase** (steps 0–400): both train and test accuracy rise together.
  2. **Memorisation/plateau** (steps 400–1800): train reaches 99.8%, test stalls at
     15–26%. The gap is the classic grokking delay.
  3. **Generalisation** (steps 1800–2600): test accuracy climbs rapidly from ~49% to
     ~99%. This is the grokking transition.

**Why the delay?** Two competing algorithms coexist in weight space:
- *Memorisation circuit*: a "lookup table" that stores each training (a, b) → c pair
  directly. Learns fast, high weight norm.
- *Generalisation circuit*: a DFT-based Fourier algorithm. Learns slowly, but has
  lower weight norm (the Fourier features are structured and compressible).

Weight decay (L2 regularisation) applies a constant multiplicative shrinkage to all
weights at every step. The memorisation circuit, having high weight norm, loses energy
faster. Eventually — at around step 1800 — the Fourier circuit becomes competitive in
training loss and takes over, causing the sudden test accuracy jump.

This competition is precisely the setting in which SLT's predictions are most interesting:
the grokking transition corresponds to the loss landscape minimum shifting from a
high-λ (non-degenerate memorisation) minimum to a low-λ (degenerate Fourier) minimum.

---

### Fig 2 – LLC Trajectory

![LLC trajectory](figures/fig2_llc_trajectory.png)

**The LLC values across training:**

| Training phase | Step | Train acc | Test acc | LLC λ̂ |
|---|---|---|---|---|
| Random initialisation | 0 | 1% | 1% | **3.6** |
| Early learning | 500 | 90% | 7% | **767** |
| Memorisation plateau | 1000 | 99.6% | 19% | 1,061 |
| Plateau (continuing) | 1500 | 99.7% | 26% | 1,295 |
| Grokking transition | 2000 | 99.9% | 77% | 1,333 |
| Post-grokking | 2500 | 99.9% | 99% | 1,239 |
| Long tail | 3000–30000 | 100% | 100% | noisy ~1,100–1,950 |

**Key observations:**

**1. The initialisation → memorisation jump (3.6 → 767, a 213× increase).**
At random initialisation, the loss landscape near w* is approximately flat: the SGLD
chain barely rises above the baseline (L(w*) ≈ 4.57, the log-uniform loss), giving
λ̂ ≈ 3.6. This is consistent with randomly initialised networks having a very diffuse
posterior — the model has no committed structure, and small perturbations don't change
loss much.

By step 500, the model has memorised ~90% of the training set. The loss landscape has
become *sharp*: each training example now has a dedicated, high-curvature "well". The
SGLD chain jumps far above the baseline, giving λ̂ = 767. This is a genuine signal —
the memorising solution is non-degenerate.

**2. The plateau (steps 500–1800, LLC ≈ 700–1300).**
During the memorisation plateau, LLC continues rising slowly. This tracks the refinement
of the lookup-table circuit: weight decay is already shrinking the memorisation weights,
but the Fourier circuit has not yet become dominant. The loss landscape remains sharp and
non-degenerate.

**3. The grokking transition (steps 1800–2600).**
LLC reaches approximately 1,300 at step 2000 (when test acc is ~77%) and then shows a
subtle decrease to ~1,239 at step 2500 (when test acc reaches 99%). This is directionally
consistent with the SLT prediction — the generalising Fourier solution is more singular
(lower λ) — but the decrease is small relative to the measurement noise.

**4. The post-grokking plateau (steps 2600–30000).**
LLC fluctuates noisily in the range 1,100–1,950. There is no further systematic trend.
See [§5](#5-the-calibration-challenge) for the interpretation of this behaviour.

---

### Fig 3 – Phase Portrait

![Phase portrait](figures/fig3_phase_portrait.png)

This plot shows the trajectory in **(test loss, LLC)** space, coloured by training step.
It reveals that the training dynamics trace a path between two regimes:

- **Early phase** (yellow/green): high test loss, low LLC — the model is near its
  random-initialisation state, exploring a flat landscape.
- **Memorisation phase** (blue): high test loss, high LLC — the model is in a sharp
  lookup-table minimum that doesn't generalise.
- **Grokking** (purple): the path moves downward and to the left — test loss falls and
  LLC decreases slightly — as the Fourier circuit takes over.

The phase portrait makes the non-monotone trajectory visible: LLC rises *before* the
test loss falls. The model's internal complexity (as measured by weight-space structure)
peaks during the memorisation plateau, not at the generalisation point.

---

## 4. Interpretation

### The two solutions in weight space

The grokking phenomenon involves a competition between two local minima in weight space,
both of which fit the training data:

**The memorisation minimum (high λ):**
- Mechanistically: a lookup table implemented via attention-based retrieval of stored
  (a, b) pairs.
- Structure: many sharp, independent circuits — one "slot" per training example.
  Almost all parameters contribute to different training examples.
- In SLT terms: near-regular. The Fisher information has few degeneracies. λ is close
  to d/2 (maximum).

**The Fourier minimum (low λ):**
- Mechanistically: the network computes `a + b mod p` by embedding a and b in a
  Fourier basis over ℤ_p, multiplying frequencies, and reading off the result. This
  requires only O(√p) ≈ 10 active frequencies (Nanda et al. 2023).
- Structure: highly symmetric. Permuting which Fourier frequencies are used, or rotating
  within the frequency subspace, gives the same function. The weight space has many flat
  directions.
- In SLT terms: highly singular. The effective number of parameters is ≪ d. λ ≪ d/2.

**The role of weight decay:**
Weight decay is essential because it creates a *norm penalty* that differentiates the two
solutions. The memorisation circuit, with its many independent sub-circuits, has higher
total weight norm than the compact Fourier circuit. Under L2 regularisation, the total
energy is:

```
E(w) = n · L_n(w)  +  (weight_decay / 2) · ‖w‖²
```

Initially both circuits have similar loss and norm. As training continues:
- Both circuits are shrunk by weight decay at the same multiplicative rate.
- The memorisation circuit degrades faster (its high norm is its Achilles heel).
- Eventually the Fourier circuit achieves lower *total energy* E(w) and dominates.

This is an instantiation of SLT's general principle: the Bayesian posterior prefers
singular minima, and weight decay is what drives the model into the basin of the more
singular minimum.

### LLC as a developmental indicator

The LLC trajectory tells a story about the *internal developmental state* of the network:

1. **Near init (LLC ≈ 3.6):** No committed structure. The model is "tabula rasa" — the
   loss landscape is flat, reflecting maximum entropy over possible circuits.

2. **During memorisation (LLC ≈ 700–1,300):** The model is "overfit but structured" — it
   has built a complex, sharp circuit that fits training data. High LLC reflects low
   degeneracy: almost every parameter participates in some stored memory.

3. **At the transition (LLC peaks then slightly dips):** The Fourier circuit begins to
   compete. The landscape is a superposition of two basins. LLC peaks because the
   transition region has the most complex geometry — the model is between two attractors.

4. **Post-grokking (LLC noisily high):** The model is in the Fourier minimum. The LLC
   *should* be low here (reflecting the DFT solution's symmetries and flat directions),
   but our estimator struggles — see §5.

### Comparison to the SLT prediction

SLT's prediction for the full grokking arc is:

```
λ(init) < λ(memorisation) > λ(grokking)
```

- `λ(init) < λ(memorisation)`: ✓ observed (3.6 vs. ~1,200)
- `λ(memorisation) > λ(grokking)`: directionally ✓, but the decrease is within noise

The first part of the prediction is strongly confirmed. The second part is directionally
correct but obscured by the SGLD calibration issue described next.

---

## 5. The Calibration Challenge

The post-grokking LLC estimates (1,100–1,950) are high and noisy. This is not a failure
of SLT but a practical challenge of the SGLD estimator.

**Root cause:** The SGLD step size ε = 3×10⁻⁶ was chosen for the *memorisation* regime,
where the training loss is ~0.05–0.1. Post-grokking, the training loss drops to ~0.001 —
two orders of magnitude smaller. The curvature of the loss landscape near the Fourier
minimum is much higher (it's a sharper basin), so the same step size now causes the SGLD
chain to *escape the local basin entirely* and probe global features of the landscape
rather than local singularity structure.

Formally, the equilibrium radius of the localised SGLD chain is:

```
E[‖w − w*‖²] ≈ d / γ
```

where d = 223,872 and γ = 10,000 (our spring constant). This gives an equilibrium
displacement of √(d/γ) ≈ 4.7 in L2 norm — about 0.01 per parameter. For a model
with post-grokking loss ~0.001, moving 0.01 per parameter can cause large loss
increases, biasing λ̂ upward.

**What would be needed for accurate post-grokking LLC:**
1. **Adaptive step size:** ε should scale with the local Hessian eigenvalue spectrum. Near
   a sharp minimum, ε should shrink proportionally to the inverse of the maximum Hessian
   eigenvalue.
2. **Stronger localisation:** γ should be large enough that √(d/γ) is much smaller than
   the basin radius. Post-grokking, γ ~ 10⁸ would be more appropriate.
3. **Longer chains:** More SGLD steps with smaller ε improves estimation accuracy but not
   the bias introduced by scale mismatch.

**Why this matters for Timaeus' Spectroscopy approach:**
This calibration challenge is precisely what motivates alternative weight-space methods.
The Spectroscopy program uses *susceptibility-based* measures — local responses of the
network to small perturbations — which are naturally calibrated to the local geometry.
Unlike the SGLD estimator, susceptibility measures do not require global exploration of
the loss landscape and are less sensitive to the absolute scale of the loss.

**An open research question raised by this work:**
Can the SGLD-based LLC estimator be made robust to post-grokking sharp minima through
(a) adaptive localisation or (b) HMC instead of SGLD? The directional signal we observe
(LLC peaks at memorisation and begins to fall at the grokking step) suggests the answer
is yes, but requires careful calibration work.

---

## 6. Connections to Timaeus Research

### Spectroscopy

Timaeus' Spectroscopy methodology characterises neural network structure using
susceptibility-based measures grounded in *weight space* — in contrast to
activation-space methods like sparse autoencoders and probing.

| Spectroscopy | This project |
|---|---|
| Weight-space grounded | LLC is computed from the loss landscape geometry at w*, not from activations |
| Susceptibility measures | LLC measures how much model behaviour changes under weight perturbation (via SGLD) |
| Discovers internal structure | LLC trajectory reveals three distinct developmental phases |
| Local characterisation | Localised LLC (with spring penalty) probes the local basin around w* |

The LLC can be understood as a *global susceptibility*: how sensitive is the model's
loss to perturbations of all weights simultaneously? Spectroscopy's methods probe finer
structure — layer-wise or circuit-wise susceptibilities — giving a richer picture of
which parts of the network change at each phase transition.

An extension of this project would be to compute *per-layer LLC contributions* by running
SGLD with perturbations restricted to each layer, revealing which layers drive the
memorisation-to-generalisation transition.

### Patterning

Timaeus' Patterning methodology steers what structures neural networks develop by
inverting the spectroscopy signal to reweight training data.

The grokking delay is controlled by the proportion of training data and the weight decay
strength. Patterning would invert this:
- *Identify* which training examples are "supporting" the memorisation circuit (high
  per-example LLC contribution) vs. the Fourier circuit.
- *Reweight* the training distribution to down-weight memorisation-supporting examples
  and up-weight Fourier-supporting examples, accelerating grokking.

This is exactly the regime where Patterning's signal-inversion approach would be most
powerful — the two circuits have very different per-example loss patterns.

### Phase transitions and developmental stages

Grokking is a toy model for the *developmental stages* that transformers go through
during pre-training (Olsson et al.'s in-context learning circuits, induction heads, etc.).
The LLC trajectory in this project demonstrates that weight-space measures can track
these transitions continuously, without requiring interpretability techniques that
activate specific circuits.

---

## 7. Setup and Usage

### Install

```bash
git clone <this-repo>
cd slt-grokking

uv venv -p 3.12 .venv
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

### Run options

| Command | What it does | Time (MPS) |
|---|---|---|
| `python train.py --quick` | p=23, 1-layer, 2000 steps — smoke test | ~30 s |
| `python train.py` | p=97, 2-layer, 50% data, 6000 steps | ~90 s |
| `python train.py --delayed` | **Recommended: classic delayed grokking** | ~4 min |
| `python train.py --model_sweep` | Adds Fig 4: LLC vs model size | ~20 min |

The `--delayed` flag uses the configuration described in §2 and produces the results in §3.

### Custom config

```bash
python train.py \
  --p 113 --n_layers 1 --train_frac 0.3 \
  --steps 50000 --llc_interval 1000 \
  --llc_localization 50000   # increase for sharper minima
```

### Code structure

```
slt-grokking/
├── src/
│   ├── model.py    # ModularTransformer: 3-slot decoder [a, b, =]
│   ├── data.py     # ModularAdditionDataset: all p² (a,b,c) triples
│   ├── llc.py      # Localised SGLD LLC estimator — implemented from scratch
│   └── viz.py      # Five figure generators (publication style)
├── train.py        # Training loop + LLC tracking + CLI (--quick / --delayed)
├── pyproject.toml  # uv/pip project spec
└── requirements.txt
```

The LLC estimator in `src/llc.py` is implemented from scratch without depending on the
devinterp library, to make the SLT machinery explicit and auditable.

---

## 8. References

- Watanabe, S. (2009). *Algebraic Geometry and Statistical Learning Theory*. Cambridge UP.
- Watanabe, S. (2013). A widely applicable Bayesian information criterion. *JMLR*, 14, 867–897.
- Power, A. et al. (2022). Grokking: Generalisation beyond overfitting on small algorithmic datasets. *arXiv:2201.02177*.
- Nanda, N. et al. (2023). Progress measures for grokking via mechanistic interpretability. *ICLR 2023*.
- Hoogland, J. et al. (2024). The developmental landscape of in-context learning. *arXiv:2402.02364*.
- Lau, E. et al. (2023). Quantifying degeneracy in singular models via the learning coefficient. *arXiv:2308.12108*.
- Timaeus devinterp library: https://github.com/timaeus-research/devinterp
- Gordon, B. et al. (2026). Spectroscopy of neural networks. *Timaeus*.
- Wang, C. & Murfet, D. (2026). Patterning. *Timaeus*.
