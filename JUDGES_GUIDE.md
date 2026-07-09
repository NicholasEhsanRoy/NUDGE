# NUDGE — a guide for judges

Thanks for reading. This is a short, honest tour of NUDGE against the four judging
criteria, with pointers to the exact files, commits, and results behind each claim —
and a straight account of what is proven versus prototype-stage.

**NUDGE in one paragraph.** Perturb-seq screens tell you *that* a gene matters; they
can't tell you *how* — whether a perturbation moves a genetic switch's **threshold**,
changes its **gain** (steepness), or lowers its **ceiling**. NUDGE fits a
differentiable gene-regulatory *circuit* to single-cell counts and attributes each
perturbation to one of those mechanisms — and its defining property is that it
**abstains, loudly, rather than guess wrong**. That fail-safe guarantee is *measured*,
not asserted: **0% misclassification across 852 synthetic datasets.** Built on
MADDENING (a differentiable JAX physics engine) for *Built with Claude: Life Sciences*.

The single best artifact to read is **[`scripts/vv/FINDINGS.md`](scripts/vv/FINDINGS.md)** —
the measured results, including the multi-agent research arc (§T0.5-3 → §T0.5-5).
**[`design/STATE.md`](design/STATE.md)** is the engineering source of truth.

---

## 1. Impact (25%) — mechanism, not just a hit list

The question NUDGE answers is one a linear screen analysis *structurally cannot*:
two perturbations can move the same readout by the same amount while doing opposite
things to the underlying switch. That distinction is exactly what decides a
follow-up: a **threshold** mover is a sensitizer, a **gain** mover reshapes the
decision, a **ceiling** mover caps output. Getting it wrong wastes a wet-lab cycle —
which is why the product is built to **abstain** instead.

- The value framing: [`design/PITCH.md`](design/PITCH.md) (plain language) and
  [`design/WORKING_BACKWARDS.md`](design/WORKING_BACKWARDS.md) (the PR/FAQ).
- Scope discipline is stated up front (README "Capabilities NOT provided"): it is a
  hypothesis-prioritizer for a powered screen, not a clinical tool or a hit-caller.
- **Honest status — the fail-safe guarantee, demonstrated on real data.** NUDGE ran
  end-to-end on the **real** 150 GB Gladstone CD4+ T-cell screen (2.79M cells; a
  pointer-based loader reads only the relevant cells/genes on any laptop) and **abstained**:
  the BIC topology gate found the 8-h activation readout is a graded single population, not a
  bistable switch (5,884/6,000 cells in one mode), so it declined to attribute mechanism
  rather than fabricate one. That is the *defining* property — abstain rather than overclaim —
  shown on real biology, not a synthetic (`scripts/vv/FINDINGS.md` "Phase 4 — real data").
  Focused higher-powered screens / other timepoints are the natural next validation; the
  point stands that the tool does the honest thing when the data doesn't support a switch.

## 2. Claude Use (25%) — an async, multi-agent R&D lab

This project was built *with* Claude Code as an active engineering participant, and
the git history is deliberately an auditable record of that (see `CLAUDE.md`: commits
credit the Claude work explicitly).

- **Multi-agent async R&D — the part that surprised us.** When we hit a hard research
  question — *can the fit represent emergent bistability?* — the main agent (Opus 4.8)
  spawned an **autonomous subagent** to run an isolated JAX/Optax numerical spike *in
  parallel* while the main build continued, then folded the verdict back in. It did this
  **twice**: once to scope multi-basin feasibility (**§T0.5-3**), and once to prototype
  the saddle fix (**§T0.5-5**). The second spike produced a genuinely non-obvious
  scientific diagnosis — *the optimizer instability lives in the free mode locations, not
  the basin mixture* — that reframed the whole approach. A separate agent context,
  running unsupervised in the background, scoped a core architecture decision cheaply.
- **`/deep-research` literature synthesis.** Adversarially-verified research workflows
  resolved hard cruxes: the two count-model / near-bifurcation design questions
  ([`design/GENERATOR_DESIGN.md`](design/GENERATOR_DESIGN.md)), and — when a measured
  NO-GO left an open question (how mechanism manifests in a *toggle* snapshot) — a
  cross-disciplinary sweep (non-equilibrium stat-mech: linear-noise covariance,
  Freidlin–Wentzell quasi-potential) that diagnosed *why* the saddle gain gate can't
  extend and named the signature that would, tying it back to the constitutive control we
  already validated ([`design/TOGGLE_ATTRIBUTION_RESEARCH.md`](design/TOGGLE_ATTRIBUTION_RESEARCH.md)).
  Each ran 25+ sources through a 2/3-refute verifier (one headline claim was *refuted*,
  and we kept it out).
- **Honest AI collaboration.** Claude's output was *independently verified*, not
  trusted: we audited the subagent's spike code for ground-truth leakage and
  reproduced its headline result before integrating it (§T0.5-5).
- **The full harness:** custom Agent Skills (`.claude/skills/`), the co-authorship
  policy, and a documented decision to *bypass* a framework primitive when it didn't
  fit (`../plans/NUDGE_deterministic_solve_vs_graphmanager.md`).

## 3. Depth & Execution (20%) — we wrestled with it

The spine of the project is a single arc where each step was forced by the previous
one's honest failure — the opposite of a quick hack. It is all in the git history and
in `FINDINGS.md`:

1. **Single-basin fit** attributes mechanism and abstains — fail-safe, but everything
   so far is an *inverse crime* (generator and fitter share a model).
2. **Tier-0.5 independent stochastic simulator** (`20cf3e0`) breaks the inverse crime.
   Fail-safe *survives* — and we found a real **boundary**: it can be wrong under
   *topology* misspecification (**§T0.5-2**). We documented the boundary instead of
   hiding it.
3. **Multi-basin fit** (`b5348f1`) *represents* emergent bistability (10× lower loss)
   but its attribution **degenerates** — a confident wrong call. We shipped this as a
   **documented negative** (§T0.5-4), marked EXPERIMENTAL, rather than force a green test.
4. **Saddle transition-mode gain gate** (`453eabf`) **fixes** it: fail-safe gain
   attribution on emergent-bistable data, recovering the case the previous step got
   wrong (§T0.5-5).

Engineering craft in that last step: the fail-safe fix was designed against **six
named failure modes** (NaN-at-the-bifurcation via a covariance-free transition sample +
a validity mask; N-dimensional saddles via a *decoupled* `Circuit.fixed_points` and an
`n_species == 1` gate; off-model / no-effect ordering; probe selection; a tunable
threshold). We also caught and fixed a subtle integration bug (restricted fits must
start from the nominal, not the distorted-WT, circuit). Nothing was asserted that
wasn't measured.

**A decoy battery that earns the "fails safely" claim, and a bound we surface rather
than hide.** Five adversarial negatives (`data/decoys.py`) span three gates —
noise-induced bimodality, cell-type/doublet mixtures, dropout zero-peaks (parsimony
gate); a dead-guide null (no-effect gate); a marginal Hill (margin calibration) — each a
green CI test where NUDGE must *decline*, verified on both fit paths and at the default
budget. The sixth case is the honest one: `NUDGE-LIM-006` — a **nonlinear reporter** on a
linear circuit *does* fool NUDGE (it fixes the readout as affine, so it can't tell circuit
ultrasensitivity from measurement ultrasensitivity). We keep it as a strict-`xfail`
witness, not a hidden blind spot, precisely bounding the guarantee: *fails safely under an
approximately-affine readout.* Its discovery is also a **case study in verifying an AI
collaborator** — an autonomous spike claimed the failure; independent reproduction
reconciled it *twice* (a fit-budget confound, then confirmed a residual structural
failure) before we believed and documented it. Then we turned the bound into a
**validated contribution**: an identifiability study (audited + reproduced) showed the
readout/circuit confound is a fundamental degeneracy (a single population cannot even tell
a circuit switch exists), and that a **constitutive-reporter control** breaks it — letting
the fit reject "no switch." That is both a candidate NUDGE capability and a concrete
experimental-design suggestion to the field for making mechanism attribution survive a
nonlinear readout (`FINDINGS.md`, NUDGE-LIM-006 + mitigation).

**Generalizing beyond one gene — and knowing when to stop.** We took the saddle gain gate
toward multi-species switches (the canonical 2-node toggle) in bite-size milestones, each
ending with the full slow lane green *and identical to baseline* (the decoy battery is the
regression harness). The N-D **saddle finder** and **multi-basin representation** landed and
are verified (multi-start Newton + Jacobian-index classification; the toggle is represented
at 56× lower loss than naive seeding) — engineered against real numerical traps (f32 Newton
cancellation → a local x64 context; the XLA dynamic-shape cliff → static padded arrays;
monostable excursions and root-order slot-swaps that would thrash the optimizer). But a
measured **go/no-go** showed the gain *signature* is 1-D-specific and does not extend to the
toggle — so we **kept the gate guarded to one species and shipped abstention** (NUDGE stays
fail-safe on toggles, `test_toggle_nd_safety.py`) rather than ship an unreliable N-D gate.
Reusable infrastructure + an honest boundary, not overreach (`FINDINGS.md`, "N-D saddle").

**Then we came back and solved it — theory → measurement → working attribution.** The NO-GO
left a real question: *how does mechanism manifest in a toggle snapshot?* A cross-disciplinary
`/deep-research` (non-equilibrium stat-mech) said the signal lives in each lobe's **covariance**,
not its basin weight. We turned that into a **measured** result: a Fisher-information /
sloppiness analysis (`scripts/vv/fisher_sloppiness.py`) proved — and *corrected the
literature's medium-confidence guess* — that the confounded pair is **gain⇄threshold** (not
gain⇄ceiling), analytically because a snapshot constrains only `n·ln(K/B)`; it is robust to
extrinsic noise; and the degeneracy-breaker is a **second operating point** (FIM stiffening
×16). Then we **built** the covariance-structured attribution (`nudge.inference.lyapunov`, an
additive/opt-in/guarded path, milestones M0–M4): a differentiable linear-noise Gaussian-mixture
fit whose single-condition call *abstains between gain and threshold* (reproducing the confound
honestly) and whose **multi-operating-point joint fit resolves it** — the gain↔threshold NLL
gap widens 0.005→0.098 (~20×, matching the FIM) and attribution flips from *abstain* to the
correct mechanism. It **abstains loudly** where the Gaussian breaks (low counts, near a
bifurcation, monostability; `lna_reliable`). Along the way the fit *rediscovered* why single-
cell pipelines normalize by sequencing depth (the `scale⇄vmax` degeneracy → depth pinned from
a WT/housekeeping reference). A full arc — literature → first-principles measurement that
overturned it → shipped, guarded capability — validated on synthetic ground truth and staged
for the real multi-target Gladstone screen (`FINDINGS.md`, "Covariance attribution").

## 4. Demo (30%) — reproducible science you can run

The demo today is **reproducible, trustworthy findings** rather than a polished UI —
and every headline number is a command away:

```bash
uv venv && uv pip install -e ".[dev]"

# Fail-safe attribution + abstention, end to end (the proof of concept):
uv run pytest tests/inference/test_fit_end_to_end.py -m slow -q

# The honest arc — independent stochastic data + the saddle gain-gate fix:
uv run pytest tests/verification/test_stochastic_inverse_crime.py -m "slow or verification" -q

# The calibration behind "0% misclassification" (figures + CSVs land in results/):
uv run python scripts/vv/overnight_sweep.py --smoke
```

**And now it's a tool you can drive — from a terminal or from Claude itself:**

```bash
# A working CLI over the tested engine (no data needed for these two):
uv run nudge mechanisms                     # the mechanism library + its cards
uv run nudge explain unresolved             # WHY an abstention was the honest answer
uv run nudge attribute screen.h5ad -t SOS1  # attribute a perturbation, honestly

# Or hand NUDGE to Claude as a custom MCP server (verified — see the memo):
uv pip install -e ".[mcp]"
claude mcp add --scope project nudge -- uv run nudge-mcp
#   then, in Claude: "Use nudge to attribute SOS1, and explain any abstention."
```

The MCP server exposes `attribute`, `explain_abstention`, `list_mechanisms`, and
`get_mechanism_card` — so a scientist asks for a mechanism map *in one sentence*
and gets back the same honest, abstaining answer a human gets, with the decoy +
limitation + Mechanism Card that explains any "I can't tell." The exact connection
recipe for Claude Code / Desktop / the **Claude Science** workbench is verified in
[`design/INTEGRATION_FEASIBILITY.md`](design/INTEGRATION_FEASIBILITY.md); the "why
abstain?" traversal is `nudge.knowledge` today and a SPARQL graph in
[`design/ONTOLOGY.md`](design/ONTOLOGY.md).

- Read alongside [`scripts/vv/FINDINGS.md`](scripts/vv/FINDINGS.md) and the figures in
  `scripts/vv/results/` (calibration curve, identifiability heatmaps).
- **Honest status:** the science holds up today — green tests, measured guarantees,
  reproducible results — and there is now a **runnable tool surface** (CLI + a live
  Claude MCP integration) rather than only a test suite. The remaining road is a
  guided notebook / visual walkthrough on *real* T-cell data (pending the download);
  the integration itself is built and tested, not vapor.

---

### The one thing to take away

NUDGE's core promise is a **fail-safe guarantee** — never confidently wrong, abstain
loudly — and this week we did the hard thing: we *attacked* that guarantee with
independent stochastic data and a genuinely harder problem (emergent bistability), found
where it bent, and **extended it rather than quietly breaking it** — with a multi-agent
Claude workflow doing real, verified R&D in the loop.
