# Toggle-specific attribution — the physics, and a path past the M3 NO-GO

**Status: researched, not yet implemented.** This is the literature-grounded answer to
the open question the N-D saddle work surfaced (`scripts/vv/FINDINGS.md`, "N-D saddle";
the M3 go/no-go). It records *why* the 1-D saddle gain gate does not extend to a 2-node
toggle and *what* physically-distinguishable signature would — so a future thrust can
build it deliberately instead of rediscovering the wall. Produced by an adversarially-
verified `/deep-research` sweep (110 agents; 27 primary sources; 24/25 claims confirmed,
1 refuted) drawing on non-equilibrium statistical mechanics, not just comp-bio.

## The question

The saddle transition-mode gain gate makes attribution fail-safe on a **1-species**
self-activation switch: a gain (Hill-`n`) reduction collapses the switch onto its saddle,
piling graded cells there, and the transition-mode weight `w_trans` cleanly detects it
(≈0.9 gain vs ≈0.01 else). On the **2-node mutual-inhibition toggle** the same `w_trans`
probe measured 0.00–0.25 across gain/threshold/ceiling/no-effect — no separation (a
measured NO-GO). We kept the gate guarded to `n_species == 1` and abstain on toggles.
Open question: *how do threshold / gain / ceiling manifest in a multi-attractor snapshot,
and what signature distinguishes them?*

## Why `w_trans` was the wrong probe for a toggle (the verified physical reason)

The transition-mode/`w_trans` idea attributes gain via **basin occupancy** — how much
mass sits in the intermediate (saddle) mode. That works in 1-D because the saddle is
exactly where graded cells accumulate. It fails in the toggle because, in a bistable
*stochastic* system, **mixture weights are not set by the deterministic saddle geometry
at all**:

- The stochastic basin boundary sits where the two inter-basin **escape rates balance**,
  which can be far from the deterministic saddle (Perez-Carrasco 2016: boundary at M≈0.3
  vs deterministic saddle-node at M=1.0). Occupancy follows a WKB/large-deviation law
  `P_ss(x) ∝ exp(−φ_QP(x)/ε)` where the **global quasi-potential depths** at the
  attractors set the weights (Zhou & Li 2016) — *not* local barrier heights, and *not*
  the separatrix location.
- The toggle vector field is **non-gradient** (a non-equilibrium steady state): the drift
  is `−D∇U` **plus a curl-flux** term that is not divergence-free and not orthogonal to
  the gradient (Zhou et al. 2012; Li & Wang 2016). So there is no scalar potential to fit,
  and energy-landscape intuition about "which basin is deeper" is misleading — weights
  require an action-minimizing (gMAM/MAP) quasi-potential, which is expensive.
- Directly on point: the claim that *a gain reduction cleanly eliminates a lobe (drives a
  basin weight to zero) as a snapshot-visible signature* was the **one refuted claim**
  (1–2 vote). Gain reduction reshapes the lobes and eventually annihilates bistability at
  a saddle-node, but does not give a clean weight-based gain readout — exactly the NO-GO.

**Takeaway:** occupancy/weight is a *noisy, expensive, non-gradient* channel. The gain
signal for a toggle lives elsewhere — in the **shape of each lobe**, not its weight.

## The signature that should work: per-mode covariance (linear-noise / Lyapunov)

Near each **stable** fixed point the stationary distribution is locally Gaussian with a
covariance `Σ_k` given by the **Lyapunov equation**

```
A_k · Σ_k + Σ_k · A_kᵀ + D_k = 0
```

where `A_k = ∂(drift)/∂x` is the drift Jacobian at mode `k` (available by autodiff — we
already compute exactly this Jacobian in `_nd_kernel` for the eigenvalue classification)
and `D_k` is the diffusion matrix from the birth–death propensities (Elf & Ehrenberg
2003; van Kampen LNA). No stochastic simulation needed.

The discriminating fact (Paulsson 2004): the **kinetic parameters enter `A_k` through
different channels**, so gain and ceiling reshape `Σ_k` differently:

| Mechanism | How it enters the mode | Covariance signature |
|---|---|---|
| **Gain** (Hill `m`) | via the repression **elasticity / logarithmic gain** `H_ij = d ln R/d ln x_j`, whose magnitude **scales with `m`** — steepens off-diagonal coupling in `A` | **rotates / stretches** each lobe's ellipse (off-diagonal / anisotropy) |
| **Ceiling** (`v_max`) | mainly rescales the **mean copy number** ⟨n⟩ → the intrinsic `1/⟨n⟩` noise floor | shifts the **mean** and scales overall variance, weakly changes ellipse *shape* |
| **Threshold** (`K`) | **asymmetrically translates** the nullclines → moves the means and the saddle off-diagonal | most **distinct**: shifts mode *positions* and separatrix, not just shape |

A second, possibly cleaner discriminator: the **separatrix orientation** — the stable
manifold of the index-1 saddle, locally the saddle Jacobian's eigenvector (we already have
the saddle and its Jacobian). Gain vs ceiling reorient it differently.

## The residual degeneracy (and how to break it)

**gain ⇄ ceiling are partially confounded** from a steady-state snapshot: both can
simultaneously shrink a lobe's weight and narrow its covariance, so along one direction of
the snapshot they look alike (a "sloppy" direction; the DSGRN coherence result is the
structural root — ceiling/threshold set *where* the bistability boundary is, gain sets
*whether* it is reached). **Threshold `K` is the identifiable one.**

The single most valuable extra observable to break gain⇄ceiling — **a co-measured
constitutive control**: a ceiling (`v_max`) change rescales absolute production, so it
**shifts an uninhibited/constitutive reporter's level**; a gain (`m`) change alters only
coupling steepness and **leaves the constitutive level unchanged**. This is the *same*
constitutive-reporter control NUDGE already validated for NUDGE-LIM-006 (nonlinear
readout) — see [`CONSTITUTIVE_CONTROL.md`](CONSTITUTIVE_CONTROL.md). One experimental-
design suggestion resolves two identifiability problems. (A second time point is an
alternative: relaxation exposes the Jacobian eigenvalue *timescales*, which `m` and `v`
affect differently.)

## Concrete recommended loss (if/when a toggle attribution thrust is built)

Fit a **2-component Gaussian mixture** whose parameters are *derived from the circuit*,
not free:

1. **Means** `μ_k(θ)` = the numerically-solved deterministic stable fixed points
   (already produced by `Circuit.fixed_points()`; note the symmetric FP is a quintic — no
   closed form, so the numeric solve is load-bearing).
2. **Covariances** `Σ_k(θ)` = Lyapunov solve with `A_k` from autodiff + `D_k` from
   propensities. **This term carries the gain-vs-ceiling information** — the highest-value
   piece to add.
3. **Weights** `π_k` — treat as **nuisance / free** parameters, *not* a fitted scalar
   potential (the non-gradient quasi-potential proxy is expensive and the refuted claim
   says weights are a poor gain channel anyway).
4. *(optional)* a **separatrix-orientation** term from the saddle Jacobian eigenvector.

Attribution then reads gain vs ceiling off the **covariance shape** (2), threshold off the
**mode/saddle positions** (1), and breaks the residual gain⇄ceiling tie with a
**constitutive control** channel.

## Caveats (honesty rule — do not overclaim from this)

- The LNA/Lyapunov Gaussian is **local and approximate**: second-moment-exact only for
  unimolecular networks, and it **degrades precisely where the perturbation pushes toward
  monostability** — near the saddle, near the bifurcation, and at low copy number (Grima
  2013). So covariance attribution is weakest exactly in the regime a large gain reduction
  creates. This must be measured, not assumed.
- Weights are genuinely hard (non-gradient, expensive); the recommendation is to *not*
  lean on them.
- The gain⇄ceiling degeneracy is a **reasoned** consequence of verified physics, not a
  directly-measured snapshot indistinguishability. The natural first build is a **Fisher-
  information / sloppiness eigen-analysis of the LNA mixture** to turn the asserted
  confound into a *measured* degeneracy magnitude — the kind of honest, quantitative
  result this project values — **before** committing to the full loss.

## Key sources

- Elf & Ehrenberg 2003, *Genome Res.* 13:2475 — LNA stationary covariance (Lyapunov).
- Paulsson 2004, *Nature* 428:415 — logarithmic gains `H_ij`; kinetic order = Hill coeff.
- Grima et al. 2013, PMC3849541 — LNA breakdown regimes (low copy, bimolecular).
- Perez-Carrasco et al. 2016, *PLoS Comput Biol* 10.1371/journal.pcbi.1005154 — stochastic
  boundary ≠ deterministic saddle; Minimum Action Path.
- Zhou, Aliyu, Aurell, Huang 2012, arXiv:1206.2311; Li & Wang 2016, *JCP* 144:094109 —
  non-gradient quasi-potential + curl-flux; global QP depth sets occupancy.
- Gardner, Cantor, Collins 2000, *Nature* 403:339 — the canonical toggle drift.
- DSGRN toggle analysis, arXiv:2204.13739 — ceiling/threshold set boundary *location*,
  gain sets *whether* it is reached (root of the gain⇄ceiling confound).
