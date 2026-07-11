# Run 000000004 — blind_design (nonlinear INVERSION / bifurcation) · WITH vs WITHOUT NUDGE

*Immutable record (append-only; never edit). The "physics money-shot": a task an LLM was expected to
be unable to compute. The result overturned that expectation — and is more valuable for it.*

- **Case:** `blind_design` (surface `design`) · **model:** `claude-opus-4-8` (headless `claude -p`)
- **Web:** DENIED (both arms) · **Sandbox:** outside the repo · both arms had `Bash` (a code interpreter).
- **Agent-facing:** `blind_toggle.h5ad` (bimodal single-cell toggle population) +
  `switch_characterization.json` (topology + Hill kinetics n=4, K=1, vmax=2, basal=0.5).
- **Task:** the exact % reduction in Gene A's basal rate to collapse the HIGH-A state to the
  alternative (high-B) stable state, + the safety (bifurcation?) assessment.

## Ground truth — and a correction the RUN forced

- **My held-out key said 88.6%** (basal_A ×0.114) — that is what NUDGE's `design()` returns.
- **The true minimum-collapse threshold is 78.6%** (the saddle-node fold, basal_A = 0.10695), which
  BOTH agents computed independently and rigorously (root-count bisection + the fold condition
  det(J)=0, h'(A)h'(B)=1; eigenvalues {0,-2}). **NUDGE's 88.6% is an OVERSHOOT** — `design()`
  optimizes to land *deep in* the target basin, not to the minimal fold crossing. So the eval's own
  "ground truth" was the less-accurate value; the agents found the better one.

## Result — both arms reached the EXACT answer; no WITH>WITHOUT contrast

| arm | answer | safety | grade | turns | cost |
|---|---|---|---|---|---|
| **with-nudge** | **78.6%** (own bifurcation analysis) — flagged NUDGE's 88.6% as an overshoot | crosses fold (bifurcation), monostable HIGH-B | correct-call | 9 | $0.482 |
| **without-nudge** | **78.6%** (own root-finding + saddle-node) — noted noise lowers it | crosses fold, loss of bistability | correct-call | 11 | $0.596 |

**Opus 4.8 can do exact nonlinear-dynamics computation — by WRITING CODE, not "mentally."** The
premise "LLMs cannot invert nonlinear ODEs" is false for a code-capable frontier agent: both arms
implemented the toggle ODE, enumerated fixed points, located the saddle-node, and returned the exact
threshold + a correct stability analysis. Neither needed NUDGE for the computation.

## Two REAL NUDGE `design()` issues the with-nudge agent surfaced

1. **Non-physical `predicted_state` (a bug).** `design()` reported `predicted_state = [0.734, 12.70]`
   — B=12.70 exceeds the model's production ceiling (basal+vmax = 2.5), so it is NOT a fixed point.
   A gradient-descent artifact; `design()` should validate its predicted state is a real steady
   state (or flag it). The agent caught it independently.
2. **Overshoot vs threshold (a capability gap).** `design()` answers "how do I robustly FLIP to the
   other basin" (lands deep in high-B, 88.6%), not "what is the MINIMUM intervention to destabilize
   the current state" (the fold, 78.6%). For "collapse/threshold" questions those differ; NUDGE has
   no minimal-fold-crossing mode. (Its Cap-5 safety gate DID correctly flag crosses_fold=True,
   independently confirmed by both agents — that part worked.)

## The honest ablation read (across runs 000000001-4 — 8 arms, 4 case types)

**0 confident-wrong; every arm correct (abstain or exact call); on NO case did WITHOUT-NUDGE fail
where WITH-NUDGE succeeded.** Opus 4.8 is a strikingly capable computational scientist: it abstains
correctly on statistics, decodes confounds, AND does exact bifurcation analysis with code. So a
capability-gap money-shot does not exist for this model on these tasks — and on the inversion task
NUDGE was *less* precise than the agent (overshoot + a predicted-state artifact the agent caught).

**What this means for NUDGE's value (honest):** not "capability the agent lacks," but
**validation, reproducibility, and trustworthy guarantees** — a documented, pre-verified mechanistic
engine + a calibrated safety layer, vs the agent's correct-but-ad-hoc, per-task, unvalidated code
("trust my throwaway ODE solver" is not a scientific foundation). And the collaboration cuts both
ways: the agent caught NUDGE's artifact; NUDGE's safety gate corroborated the agent's fold. Neither
is a black box. To see NUDGE change an OUTCOME, the differentiator must be something the agent cannot
reconstruct: a **less-capable agent** (smaller model, or no code interpreter — remove `Bash`), or a
step that needs NUDGE's **validated differentiable FIT from noisy data** (don't hand over the
kinetics — make the agent recover the model first). Flagged for the human.

---

## Verbatim `REPORT.md` — WITH NUDGE (9 turns)

```markdown
# REPORT — Collapsing the HIGH-A state of a bistable toggle switch

## Task
Find the % reduction in Gene A's **basal transcription rate** required to collapse the HIGH-A
stable state into the HIGH-B stable state, and state whether that intervention is dynamically
**safe** or crosses a **bifurcation** (loss of bistability).

## Model (from `switch_characterization.json`)
Symmetric 2-gene mutual-repression toggle (A ⊣ B, B ⊣ A), Hill kinetics, unit degradation:

```
dA/dt = basal_A + vmax / (1 + (B/K)^n) − A
dB/dt = basal_B + vmax / (1 + (A/K)^n) − B
n = 4,  K = 1,  vmax = 2,  basal_A = basal_B = 0.5
```

Baseline fixed points (my root-finding): low-A/high-B node A=0.5676, saddle A=1.1802,
high-A/low-B node A=2.3119. These **exactly match** the NUDGE `design` tool's reported
`stable_nodes = [[0.5676, 2.3119],[2.3119, 0.5676]]` and `saddle = [1.1802, 1.1802]`, confirming
the same model.

## Tool call — NUDGE `design` (inverse verb, circuit mode)
```
design(topology='toggle', n=4, K=1, vmax=2, basal=0.5,
       free='species0.basal', start='high', to='low')
```
Output (read literally):
- `deltas`: scale `species[0].basal` by **factor = 0.1139** (log_delta −2.172) → basal_A → 0.0570.
- `reason`: "closes 96% of the target gap — HIGH RISK OF INSTABILITY: the intervention CROSSES
  THE FOLD; switch loses bistability (NUDGE-LIM-013)."
- `safety`: `crosses_fold = true`, `high_risk_of_instability = true`, `proximity_before = 0.184`
  (critical-slowing channel proximity 0.369).
- `predicted_state = [0.734, 12.70]` — I flag B=12.70 as a gradient-descent artifact: it exceeds
  the model's production ceiling (basal+vmax = 2.5), so it is NOT a true fixed point. The real
  destination basin, from direct root-finding, is (A≈0.157, B≈2.499) = the HIGH-B state.

The tool's factor (0.1139 → **88.6% reduction**) is a *sufficient, deliberately overshooting*
intervention: it drives basal_A well past the fold to land robustly deep in the HIGH-B basin
("closes 96% of the gap"). It is not the minimal threshold.

## Independent verification — locating the exact fold (saddle-node)
Collapsing the high-A fixed point means destroying it. As basal_A is lowered, the high-A node and
the saddle move together and annihilate in a saddle-node (fold) bifurcation. I bracketed the root
count (3 fixed points → 1) and bisected to the fold, confirming with the fold condition
det(J)=0 ⇔ h'(A)·h'(B)=1:

- **Critical basal_A = 0.10695** (high-A node and saddle collide at A≈1.641, B≈0.743; det→0).
- Just below it, only the HIGH-B state survives (A≈0.157, B≈2.499) → the switch has fallen.

Threshold as a reduction from basal_A = 0.5:
- **factor to fold = 0.2139  →  minimum reduction = 78.61%.**

Any reduction ≥ 78.6% collapses the high-A state; the tool's 88.6% is one such (larger) choice.

## Safety assessment
The intervention is **NOT dynamically safe — it crosses a bifurcation.** This is intrinsic, not
incidental: collapsing (destroying) the high-A fixed point *is* a saddle-node/fold bifurcation, so
bistability is necessarily lost — the system becomes **monostable at HIGH-B**. There is no
reduction that both eliminates the high-A state and preserves bistability. The NUDGE Cap-5 safety
gate independently flags this (`crosses_fold=true`, `high_risk_of_instability=true`,
NUDGE-LIM-013).

## FINAL ANSWER
- **Minimum required reduction in Gene A's basal rate to collapse HIGH-A → HIGH-B: ≈ 78.6%**
  (basal_A: 0.5 → 0.107; the saddle-node fold). Verified two ways (root-count bisection and the
  det(J)=0 fold condition); agrees with the model the tool used.
- **NUDGE `design` proposes ≈ 88.6% reduction** (factor 0.114, basal_A → 0.057) — a sufficient
  overshoot that lands deep in the HIGH-B basin.
- **Safety: crosses a bifurcation (NOT safe).** The collapse is itself a saddle-node/fold event;
  bistability is lost and the system ends up monostable at HIGH-B. Confirmed by NUDGE's safety gate
  and by direct eigenvalue/determinant analysis.
```

---

## Verbatim `REPORT.md` — WITHOUT NUDGE (control, 11 turns)

```markdown
# REPORT — Collapsing the HIGH-A state of a bistable toggle switch

## Goal
Determine the **percentage reduction in Gene A's basal transcription rate** required to
destabilize the HIGH-A fixed point so the system falls into the alternative HIGH-B stable
state, and assess whether that intervention is dynamically **safe** or crosses a
**bifurcation** (loss of bistability).

---

## 1. Inputs

`switch_characterization.json`:
- Topology: 2-gene mutual repression (A ⊣ B, B ⊣ A), **symmetric** kinetics.
- Hill kinetics: `n_gain = 4`, `K_threshold = 1`, `vmax_ceiling = 2`, `basal_rate = 0.5`.

`blind_toggle.h5ad` (tool: anndata, 4000 cells × 2 genes A,B):
- Two clusters. HIGH-A: A median ≈ 349, B median ≈ 84. HIGH-B: mirror image (A≈86, B≈353).
- **State ratio high/low ≈ 349/84 ≈ 4.1** (symmetric between the two states).

## 2. Model

Standard toggle-switch ODEs (degradation rate γ, protein = gene product):

```
dA/dt = b_A + v·Kⁿ/(Kⁿ + Bⁿ) − γ·A
dB/dt = b_B + v·Kⁿ/(Kⁿ + Aⁿ) − γ·B
```
with v=2, K=1, n=4, b_A=b_B=0.5. Degradation is not given in the JSON; the natural
nondimensionalization sets **γ = 1** (time in units of protein lifetime, concentration in
units of K). This is validated below by the data.

**Fixed points at the baseline (b_A=b_B=0.5), computed by root-finding + Jacobian eigenvalues:**
- HIGH-A: (A,B) = (2.3119, 0.5676) — **stable**
- Saddle: (A,B) = (1.1802, 1.1802) — unstable
- HIGH-B: (A,B) = (0.5676, 2.3119) — **stable**

Model state ratio = 2.3119/0.5676 = **4.07**, matching the measured data ratio ≈ 4.1.
→ This confirms the parameter set and the γ=1 assumption: the dimensionless model reproduces
the observed two-state structure. (Absolute counts are just a linear scale factor; the
**percentage** change in a rate is scale-invariant, so it is read directly from the model.)

## 3. Intervention: reduce only b_A

Lowering b_A weakens gene A's floor. Tracing the fixed points as b_A decreases (fsolve over a
grid + eigenvalue classification), the HIGH-A stable node and the saddle move toward each
other and **annihilate in a saddle-node bifurcation**. Below that value only HIGH-B remains,
so the system necessarily collapses into HIGH-B.

**Exact saddle-node location** — solving simultaneously the two steady-state equations plus
the degeneracy condition det(J)=0, i.e. `1 − h'(A)·h'(B) = 0`:

```
b_A,crit = 0.10694854
at (A*, B*) = (1.64094, 0.74241)
Jacobian eigenvalues there = {0, −2}  → zero eigenvalue = saddle-node (confirmed)
```

**Percentage reduction required:**
```
(0.5 − 0.10695) / 0.5 × 100 = 78.61 %
```

### Verification
- Fixed-point count vs b_A: at b_A=0.107 → 3 fixed points (HIGH-A node (1.647,0.739) and
  saddle (1.635,0.745) almost merged); at b_A=0.1069 and below → **only HIGH-B** (0.157,2.499).
- Direct ODE integration from the HIGH-A initial state (2.312,0.568):
  - b_A=0.107 → settles back to HIGH-A (A=1.65,B=0.74)
  - b_A=0.105 → collapses to HIGH-B (A=0.155, B=2.499)
  - b_A=0.100 → collapses to HIGH-B
  Threshold sits between 0.105 and 0.107, bracketing the analytic 0.10695. ✓

## 4. Safety assessment

**Not "safe" — the collapse IS a bifurcation.** Destroying the HIGH-A fixed point is only
possible by driving it through the saddle-node at b_A,crit. Any reduction smaller than 78.61%
leaves HIGH-A stable (no collapse, i.e. "safe" but ineffective); the minimal reduction that
achieves collapse lands exactly on the saddle-node and beyond it the system is **monostable**
(HIGH-B only) — bistability is lost while b_A is held down. The bifurcation is a real-eigenvalue
saddle-node (eigenvalues {0,−2}), so the transition is a smooth, non-oscillatory slide to
HIGH-B (no spiral/overshoot), but it does cross the loss-of-bistability boundary by construction.
Restoring b_A to 0.5 re-creates the two fixed points, so the switch hardware is not permanently
damaged — but the act of collapsing HIGH-A is inherently a bifurcation crossing, not a
within-basin nudge.

---

## FINAL ANSWER

- **Percentage reduction of Gene A basal rate to collapse HIGH-A → HIGH-B: ≈ 78.6 %**
  (b_A: 0.5 → 0.1069; exact critical value 0.10694854).
- **Safety: NOT dynamically safe.** The collapse occurs precisely at a **saddle-node
  bifurcation** where the HIGH-A fixed point is annihilated; crossing it produces a
  **loss of bistability** (system becomes monostable HIGH-B). Eigenvalues {0, −2} → smooth
  non-oscillatory saddle-node.

### Assumptions / caveats
- Degradation γ=1 (standard nondimensionalization); justified because the resulting fixed-point
  ratio (4.07) matches the measured state ratio (~4.1). The % reduction is independent of the
  count-to-concentration scale factor.
- Symmetric Hill kinetics and Hill form as given in the JSON. Deterministic ODE (no stochastic
  noise); noise would let cells hop the barrier at reductions somewhat below 78.6%, so this is
  the deterministic upper bound for a guaranteed collapse.
```
