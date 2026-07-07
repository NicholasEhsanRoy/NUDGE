# NUDGE — Working Backwards (PR/FAQ)

> **Method.** This document follows **Amazon's "Working Backwards" process** and
> its central artifact, the **PR/FAQ** (Press Release + Frequently Asked
> Questions) — written *before* the code, as if the product had already
> launched, to force clarity on end-user value first. Per the Amazon convention
> the FAQ mixes *external* questions (what a user/judge asks) with *internal*
> ones (viability, risk, competition); Parts 0–2 and 5–6 are the internal
> working-backwards analysis that a strict Amazon PR/FAQ would keep to the FAQ,
> broken out here because this is a build document, not a one-pager.

**Drug-discovery framing.** This document reframes NUDGE from "can MADDENING
generalize?" (a framework-validation curiosity) into "NUDGE is a mechanism-
attribution tool for perturbation screens, and MADDENING happens to be the
right substrate to build it on." The press release and FAQ are written for that
audience — target-discovery and computational-biology teams — not for a physics
crowd.

Companion to `../brief.md` and the plain-language `PITCH.md`. Written 2026-07-07.

---

## Part 0 — What already exists (the honest inventory)

The point of the Working Backwards exercise is to not build what you already
have. Here is the map from *what NUDGE needs* to *what MADDENING already ships*,
graded by how much transfers.

| NUDGE need (from `brief.md`) | MADDENING asset today | Transfer grade |
|---|---|---|
| One differentiable object mapping perturbation → outcome; linear baseline is the *same* graph with edges swapped | `GraphManager` + `SimulationNode` functional-state pattern; the whole graph step JIT-compiles to one pure `state → state` function, differentiable with `jax.grad` end-to-end (verified through 1000-step rollouts) | **Direct.** This is the core architecture, unchanged. |
| `ZeroOrderIntegrator` for ERK: Goldbeter–Koshland steady state solved by root-find, differentiated via IFT-through-converged-solve — "same trick as the physics adjoints" | `maddening.core.solver_utils.ift_linear_solve`, `@stability(STABLE)`. Thin lineax wrapper with the GMRES `restart = min(N,50)` clamp that guards the *silent low-rank adjoint* bug. Regression-tested. | **Direct.** The single biggest borrow-don't-build. The hardest numerical requirement in the brief is already a hardened public primitive. |
| Fit stability *near a bifurcation* — the brief's day-one worry: "as unstable as backprop through a near-singular solve" | `AdaptiveNode` base class: a **blindness diagnostic** that detects when a gradient is transverse-blind at a symmetric (Palais) fixed point, plus an **anisotropic `symmetry_break`** escape. Grounded in Palais 1979 + Chen-Ziyin 2023 (isotropic noise *cannot* escape Type-II saddles). Seven-round derisking spike behind it. | **Conceptual, not drop-in.** The class is built for *basis-coefficient PDE solves*, not ODE circuit fits. The `ift_linear_solve` primitive underneath transfers directly; the blindness/symmetry-break idea is exactly the right lens for bistable-switch fitting but needs re-expression for a small ODE state. Do not claim the class works out of the box. |
| Stage 2: invert the fitted circuit by gradient descent over perturbation-space | Same JIT'd differentiable step, run backwards. `jax.grad` already flows through initial conditions and external inputs (`add_external_input`, differentiable — designed for learning control policies). | **Direct.** Inversion is the forward machinery with the objective flipped. |
| Cheap uncertainty on proposals (Laplace / Hessian at optimum) | `jax.hessian` over the same loss; MADDENING's compliance layer already carries a UQ module and anomaly tracking | **Direct** for the Hessian; the UQ scaffolding is a bonus. |
| `Perturbation` as continuous strength fit from CRISPRi dose | `add_external_input()` injects differentiable per-node/per-edge modifiers into `boundary_inputs` | **Direct.** |
| Timescale separation (fast signaling vs slow transcriptional readout) | Multi-rate scheduler: each node at its own timestep, GCD base rate, always-compute/conditionally-apply, fully differentiable | **Available, optional.** Nice-to-have, not core-week scope. |
| Reusable package a biologist points at their own dataset + hypothesis | Node-authoring contract, serialization (`to_dict`/`from_dict`), USD graph I/O, FastAPI/WS server for a demo layer | **Direct** for the library; the server is stretch/demo. |
| Provenance / traceability if this ever nears a regulated pipeline | IEC 62304 SOUP packaging, EU MDR guideline docs, `@stability` API audit, SBOM, `known_anomalies.yaml` | **Inherited scaffolding.** Unusual for a hackathon-grade bio tool; a genuine differentiator downstream. |

**What genuinely has to be built new (small, and already scoped in the brief):**
the bio node library — `Species`, `RegulatoryEffect` (`LinearEffect` /
`HillActivationEffect` / `HillRepressionEffect`), the three `IntegratorModel`
variants, `Readout`, and the CRISPRi-dose `Perturbation` calibration. These are
thin configuration on top of the existing contract, not new framework.

**What is genuinely risky (name it now):** (1) the `Readout` — mapping a smooth
internal state to a discrete, noisy single-cell signature and calibrating it
against real outcomes is where the modeling honesty lives; (2) *identifiability*
— whether threshold-vs-gain is actually recoverable from a handful of
perturbations rather than a fitting artifact (see FAQ Q2); (3) the bifurcation-
region fit stability is a *conceptual* transfer from AdaptiveNode, not a free
ride.

---

## Part 1 — MADDENING's unique strong points *for drug discovery*

Five things NUDGE gets from MADDENING that its competitors (CellBox, GEARS, CPA,
RACIPE) structurally do not have:

1. **Per-edge / per-node mechanism composition, not one global nonlinearity.**
   CellBox — the closest prior art — fits *one* shared nonlinearity across the
   whole network. In drug discovery you care *which* mechanism, because a drug
   acts on a mechanism, not on a network-average. MADDENING's typed-edge graph
   with swappable effect models makes "is this edge a Hill activation or a
   linear coupling?" a first-class, attributable question. **This is the whole
   product.**

2. **The hard adjoint is already solved and hardened.** `ift_linear_solve`
   turns the brief's scariest line ("decide this on day one, don't retrofit
   later") into an import. The zero-order ERK integrator differentiates through
   its converged steady state via the implicit function theorem, with a shipped
   guard against the low-rank-adjoint failure mode.

3. **A named theory of *why gradient fits stall near switches*.** No cell-
   biology fitting tool has this. Bistable/ultrasensitive fitting stalls at
   symmetry-induced saddles precisely in the interesting regime. MADDENING has a
   *diagnostic* for it (blindness ratio) and a principled *escape* (anisotropic
   perturbation), from the microrobotics adaptive-solver work. Even used only as
   a diagnostic, it tells you when to *distrust* a fit near a bifurcation —
   which is exactly where naïve tools quietly give you a wrong answer.

4. **Fit and design are the same machinery run in two directions.** Because the
   graph step is one differentiable pure function, Stage 2 (propose untested
   interventions) is not a second system — it is `jax.grad` of the same object
   with the objective flipped, and Laplace uncertainty is `jax.hessian` of the
   same loss. Combination-therapy hypotheses with error bars, nearly free.

5. **A provenance posture built in.** For anything that will eventually sit near
   a decision that matters, the IEC 62304 / EU-MDR / `@stability` / SBOM
   scaffolding is inherited, not bolted on. Not a medical-device claim — a
   traceability *posture* that competitors don't start with.

> **The unfair advantage, stated plainly.** A team building this from scratch
> spends its first week writing — and debugging — a differentiable ODE solver
> that stays stable through a converged steady state and near a bifurcation.
> That is the single hardest, most error-prone part, and it is *not the science*.
> NUDGE doesn't spend that week. The hard kinetics/adjoint engine is already
> built, hardened, and regression-tested in MADDENING, so every hour goes into
> the biology — the mechanism library, the readout calibration, the falsifiable
> prediction. In a one-week build, "the hard infrastructure is already done" is
> not a nice-to-have; it is the entire reason NUDGE is buildable at all.

---

## Part 2 — Distribution & the Claude Science integration

**The original plan** was a CLI plus a minimal localhost app. That still ships —
it's the core. But NUDGE's natural home is inside **Claude Science** (Anthropic's
research workbench, launched 30 June 2026), and the fit is close enough that it
is worth making a first-class deliverable rather than an afterthought.

### What Claude Science actually is (and why it matters here)

Claude Science is deliberately **not a new model** — it is a *workflow layer*: a
generalist coordinating agent that spins up sub-agents, a library of 60+
domain **connectors** (MCP servers) and **Agent Skills**, cluster-compute
management, and — the two features that matter most for NUDGE — **per-artifact
provenance** (every figure records the exact code, environment, and full message
history that produced it) and a **reviewer agent that runs in parallel and flags
any number or citation it cannot trace**.

### How NUDGE plugs in — two standard mechanisms, both cheap

Claude Science exposes exactly two extension points, and NUDGE uses both:

1. **An MCP server.** NUDGE wraps its two verbs — `fit(adata, circuit) →
   mechanism map` and `design(target_outcome) → ranked interventions` — as MCP
   tools. The coordinating agent can then call them in plain language:
   *"pull the T-cell Perturb-seq dataset, fit a switch model, and tell me which
   regulators move the threshold versus the gain."* This is the CLI's core
   re-exposed through a thin adapter — same engine, no rewrite.
2. **An Agent Skill** (`SKILL.md`). A folder documenting the NUDGE pipeline —
   when to reach for it, how to specify a circuit hypothesis, how to read the
   output — so the workbench invokes it consistently, the same way the built-in
   `single-cell-rna-qc` skill wraps scverse QC.

Data arrives through connectors that already exist (**10x Genomics** for
single-cell, **Synapse.org** / **Benchling** for datasets and notebooks);
literature grounding via **PubMed**. NUDGE consumes an `adata` object and returns
a mechanism map — it does not need to own the data-ingestion problem.

### Why the integration is *on-thesis*, not decoration

Three alignments make this more than "we added an MCP wrapper":

- **NUDGE fills a gap Claude Science visibly has.** The workbench connects data,
  runs QC, and calls structure/sequence models (BioNeMo — Evo 2, Boltz-2,
  OpenFold3). None of those is a *dynamical-systems mechanism-attribution* layer.
  Claude Science can tell you a protein's structure and a screen's hit list; it
  cannot tell you whether a hit moves a switch's threshold or its gain. NUDGE is
  precisely the reasoning layer the orchestration is missing (see FAQ Q6).
- **The provenance stories compose.** Claude Science records code + environment +
  history per figure; NUDGE inherits MADDENING's IEC 62304 / `@stability` / SBOM
  traceability. A NUDGE mechanism claim generated *inside* Claude Science is
  reproducible at two layers at once — an unusually strong position for a
  research artifact.
- **NUDGE is built for the reviewer agent.** The reviewer flags numbers it can't
  trace. NUDGE's entire design ethic (Part 4, Q5) is to emit uncertainty bounds
  and explicit *"unresolved"* flags rather than confident point claims — exactly
  the traceable, self-doubting output the reviewer rewards. The tool that fails
  loudly is the tool the reviewer agent lets through.

### Honest scope for the week

The CLI is the core and ships first. The MCP server is a thin adapter over the
same two verbs — realistic as a hackathon stretch and a strong live demo
("watch a scientist ask Claude Science for a mechanism map in one sentence").
The full `SKILL.md` polish and multi-connector data plumbing are post-hackathon.
Do not let the integration demo compromise the falsifiable-prediction result —
that result is the substance; Claude Science is the distribution.

---

## Part 3 — The Press Release

**FOR IMMEDIATE RELEASE — July 2026**

### NUDGE tells drug-discovery teams which targets are worth a year of work — before they spend the year finding out the hard way

*Most drug programs fail late and expensively, chasing targets that looked
promising in a screen and fell apart in the clinic. NUDGE, a new open-source
tool, turns a single genetic screen into a mechanism map — separating targets a
team can safely tune like a dial from ones that behave like an unpredictable
cliff. The payoff is blunt: fewer dead-end programs, less wasted lab time, and
life-saving drugs reaching patients sooner.*

Today the Microrobotics Simulation Framework team released **NUDGE**
(Node/edge Ultrasensitivity Diagnostic for Gene-regulatory Effects), a tool
that fits a mechanistic, compositional circuit model to Perturb-seq data and
returns, for every perturbation, *what kind* of effect it has — does it shift
where a switch trips (threshold), how sharply it commits (gain), or how far it
can go (ceiling)? Standard screen analysis, including the analysis published on
the Gladstone genome-scale T-cell dataset, collapses every perturbation to a
single linear coefficient. That number tells you a gene *matters*; it cannot
tell you *how it acts* — and two genes with identical effect sizes can have
opposite therapeutic profiles. NUDGE recovers the mechanism, then inverts the
fitted circuit to propose untested interventions, including gene combinations
never present in the original screen, each with an uncertainty estimate.

"Our screens are very good at ranking genes and very bad at telling us what to
do next," said a target-discovery lead at a collaborating lab. "A threshold
mover is a titratable sensitivity dial — you can dose it. A gain mover sitting
next to a bistable switch is a cliff — small dose does nothing, slightly more
flips the whole cell, and it can get *stuck* on. Same effect size, completely
different program. NUDGE is the first thing we've run that draws that line for
us, on our own data, before we spend a year of medicinal chemistry finding out
the hard way."

NUDGE is built on **MADDENING**, a differentiable graph-based simulation engine
originally developed for magnetically-actuated microrobots. The two hardest
parts of fitting nonlinear biological switches — differentiating cleanly through
a solver's converged steady state, and keeping the fit stable near a bifurcation
where naïve gradient descent silently fails — are domain-general numerical
problems MADDENING already solved and battle-tested in physics. Because that
engine was already built and hardened, the NUDGE team spent its week on the
biology instead of burning it writing a differentiable ODE solver from scratch —
the hard kinetics came for free. NUDGE supplies the biology as configuration: a
small library of composable node and edge mechanisms. Its first validation is a
falsifiable prediction on the T-cell Ras/SOS activation switch — knocking down
the feedback driver should collapse the digital activation signature toward
graded, while knocking down the graded arm should not. NUDGE is open source and
designed as a reusable instrument: a biologist points it at their own dataset
and circuit hypothesis, not at a pre-baked analysis. It ships as a command-line
tool, a local app, and — for the teams already living there — as a connector
inside **Claude Science**, where a scientist can ask for a mechanism map in
plain language and get one back with its uncertainty attached and its
provenance recorded. `pip install nudge`.

---

## Part 4 — The FAQ (the hardest questions)

### Q1. "This is just CellBox with extra steps — or worse, why not use a black-box predictor like GEARS/CPA that already validates on held-out perturbations?"

Different deliverable, and the difference is the point. GEARS/CPA answer *"what
will happen if I perturb X?"* with high accuracy and zero mechanism — they
cannot tell you *why*, so they cannot extrapolate to a mechanism you didn't
sample, and they give a medicinal chemist nothing to act on. CellBox is closer —
it fits a differentiable ODE network — but it uses **one global nonlinearity
shared across the whole network**, so its output is a predictive model, not a
per-target mechanism label. NUDGE's deliverable is **mechanism attribution**:
this edge is a cooperative Hill feedback, that node is a zero-order
ultrasensitive integrator, this perturbation moves a *threshold* and that one
moves a *gain*. We validate against the black boxes on prediction where we can,
but we are not competing on leave-one-drug-out accuracy — we are answering a
question they structurally cannot.

### Q2. "You are fitting a stiff nonlinear circuit to noisy, discrete single-cell data with a handful of perturbations. How do you know threshold-vs-gain is *identifiable* and not an artifact of an over-flexible model?"

This is the real risk and we treat it as the gating question, not a footnote.
Three defenses, in order of strength:

1. **A pre-registered falsifiable prediction, not a fit statistic.** SOS
   knockdown should collapse the digital activation signature toward graded
   (it removes the positive-feedback loop that creates bistability); RasGRP1
   knockdown should not (it drives the graded arm, no feedback). This is a
   *directional, mechanism-specific* prediction fixed before fitting. If the
   fit "attributes" mechanism but gets this backwards, the attribution is
   wrong and we say so.
2. **Model comparison through the *same* machinery.** The linear baseline is
   the identical graph with every edge swapped to `LinearEffect`. We only claim
   a nonlinear mechanism where the mechanistic fit beats linear by a margin
   that survives the parameter uncertainty — apples-to-apples, one codebase, no
   two-model confound.
3. **Uncertainty on every claim.** The Laplace approximation (Hessian of the
   fit loss at the optimum, nearly free here) gives each attributed parameter an
   error bar. A threshold-vs-gain call with overlapping posteriors is reported
   as *unresolved*, not as a finding. We would rather return "can't tell" than a
   confident artifact.

We do not claim identifiability in general. We claim it *conditionally* — where
the data, the model-comparison margin, and the posterior all agree — and we make
the conditions visible.

### Q3. "A microrobotics physics engine has no business modeling gene circuits. Isn't this a solution looking for a problem?"

The engine doesn't know it's doing biology, and that's the argument, not against
it. Strip a bistable gene switch and a magnetically-actuated microrobot near
step-out down to their numerics and you get the same two hard problems: (a)
differentiate an objective through a solver's *converged* solution without
backpropagating through the unrolled iteration, and (b) keep gradient descent
stable when the system sits near a bifurcation / symmetric fixed point where the
adjoint goes singular and naïve optimizers silently stall. MADDENING solved both
as *domain-general* infrastructure — the IFT linear-solve primitive is
`@stability(STABLE)` with a regression guard, and the Palais/Chen-Ziyin
gradient-blindness diagnostic came out of a seven-round derisking spike. NUDGE
inherits the hard numerics and supplies only the biology, as a few hundred lines
of node/edge configuration. Building the same adjoint infrastructure from
scratch in a bespoke bio tool is how you get the day-one bug the brief warns
about. Reusing hardened infrastructure across domains is the *opposite* of a
solution looking for a problem — it's the problem finding infrastructure that
already fits it.

### Q4. "Stage 2 proposes untested gene combinations by gradient descent. Why would a bench scientist burn a screen slot on a suggestion from an optimizer?"

Because it's a *ranked, uncertainty-quantified, mechanistic* hypothesis
generator, not an oracle — and that's exactly what target teams already lack.
Three reasons it earns a screen slot over the status quo (a biologist's
intuition plus a linear hit list):

- **It extrapolates on mechanism, not correlation.** Because the circuit is
  mechanistic, a proposed combination that "collapses the pathological
  bistable-on state to monostable-off" is a claim about *how* the intervention
  works, which is testable and falsifiable — unlike a black-box prediction that
  can't survive outside its training distribution.
- **It ranks and de-risks.** Each proposal carries a Laplace error bar; the tool
  surfaces the combinations that are both high-effect *and* well-constrained,
  and flags the ones that are high-effect but sitting on a fitting cliff. That
  triage is the value even before a single wet experiment.
- **The first prediction is already cheap to test.** The SOS/RasGRP1 call
  (Q2) is runnable in the existing T-cell system today. A tool that makes one
  correct, non-obvious, mechanistically-grounded prediction on known biology
  earns the right to have its novel proposals taken seriously.

We position Stage 2 as *hypothesis prioritization for the next screen*, never as
a substitute for it.

### Q5. "What is the minimum evidence that this works by end of week — and what happens if the SOS prediction fails?"

**NUDGE is engineered to fail safely and loudly — and in drug discovery, that is
a headline feature, not a hedge.** The catastrophic, expensive failure in this
field is a *confident wrong answer*: a tool that hands you a clean point estimate
on an artifact, and a team burns a year of medicinal chemistry chasing it. NUDGE
is built specifically to not do that. Every mechanism call comes with uncertainty
bounds; calls with overlapping posteriors are reported as **unresolved**, not as
findings; a fit that stalls in a bifurcation-region saddle is caught by the
blindness diagnostic and flagged as *distrust this*. The failure mode that
sinks programs — a silent false positive — is the one thing NUDGE's borrowed
MADDENING machinery is purpose-built to prevent. **Loud, bounded, early failure
is the product.** It saves the year.

Minimum ship bar: the core library (`Species`, `LinearEffect`,
`HillActivationEffect`, `LinearIntegrator`, `SaturatingIntegrator`,
`Perturbation`, `Readout`, `LinearBaseline`) fits the T-cell circuit, the
mechanistic fit beats the linear baseline on the activation signature by a
margin that survives Laplace uncertainty, and the pipeline runs from
`fit(adata, circuit) → classification table` on a biologist's own data. That is
a shippable, reusable tool independent of the biology result.

And if the SOS-collapses-digital / RasGRP1-doesn't prediction itself **fails**?
That is a *clean, cheap, week-one kill* of a specific circuit hypothesis — which
is exactly the kind of result NUDGE exists to produce fast. Killing a wrong
hypothesis in week one instead of year one is where the money and the lives are
saved; the tool that produced an uncertainty-quantified falsification did its
job, and the hypothesis was the thing under test, not the software. "The switch
hypothesis was wrong" is a publishable, valuable outcome. "The tool lied
confidently" is the only real failure — and NUDGE is designed so that one can't
happen quietly.

### Q6. "You're integrating with Claude Science, which already has BioNeMo, 10x Genomics, and CRISPR-screen skills. Doesn't the platform already do this?"

No — and the gap is structural, not a matter of coverage. Claude Science is an
*orchestration and workflow* layer: it connects data (10x Genomics, Benchling,
Synapse), runs QC (`single-cell-rna-qc`), retrieves literature (PubMed), and
calls *structure and sequence* models through BioNeMo — Evo 2 for genomics,
Boltz-2 and OpenFold3 for biomolecular structure. Every one of those answers a
*static* question: what is this sequence, what is this structure, what is the
hit list. **None of them is a differentiable dynamical-systems model of a
regulatory circuit**, and none can answer NUDGE's question — does this
perturbation move a switch's *threshold* or its *gain*? That is a question about
the *dynamics* of a feedback circuit, and it requires fitting a mechanistic ODE
model and inverting it. NUDGE is not competing with the platform's connectors; it
is the **mechanism-attribution layer the orchestration is currently missing**,
and it becomes callable by the coordinating agent the moment it registers as an
MCP server. The integration is complementary by construction: Claude Science
brings the data and the provenance; NUDGE brings the dynamical reasoning.

---

## Part 5 — The data problem (and how the constraint becomes the thesis)

This is the hard, fun part, and it deserves to be reasoned through before a line
of fitting code. **MADDENING was built for large, deterministic, time-resolved
physical state. Perturb-seq is the opposite: a stochastic, sparse, single-
timepoint snapshot of a *population* of cells, each measured once and destroyed.**
Adapting one to the other is the actual research contribution — and the
adaptation, done right, is not a compromise; it is the reason the mechanism
signal is visible at all.

### The one pivot everything else hangs on: heterogeneity is the signal

The instinct from physics is to treat cell-to-cell variation as noise and fit
the mean. That instinct throws away the entire result. **Threshold-vs-gain lives
in the *shape of the population distribution*, not in its mean:**

- A **threshold** shift moves *what fraction* of cells have crossed the switch —
  it slides the split point of a bimodal (off/on) population.
- A **gain** change alters *how sharply* cells commit — it sharpens or blurs the
  separation between the off and on modes, and steepens the dose–response
  measured *across* the population.
- A **ceiling** change moves the on-mode's location without touching the split.

Two perturbations with identical mean effect can have opposite distributional
signatures. So the fitting target is a **distribution of single cells per
condition**, and the natural MADDENING primitive is already there:
`run_sweep` (vmap over initial conditions / per-cell parameters). **An ensemble
of deterministic circuit solves, one per cell, *is* a population model.** That is
the whole adaptation in one sentence — MADDENING becomes a single-cell simulator
by vmapping its deterministic solve over a distribution of per-cell parameters
and reading out counts.

A corollary that saves us: Perturb-seq is a **steady-state snapshot**, not a
time course. We never observe dynamics, so we never need time-resolved data — we
fit the steady-state *distribution*. The `ZeroOrderIntegrator`'s converged-
steady-state solve (differentiated via `ift_linear_solve`) is exactly the right
object. Bistability shows up as **bimodality in the snapshot**; the one honest
limit is that *hysteresis* proper needs a dose series or history to observe, so
we claim bimodality-collapse (testable from a snapshot), not hysteresis-loop
recovery (not testable from one).

### The five concrete adaptations physics→wet-lab

1. **Population model via `vmap` — the elegant core of the whole adaptation.**
   This deserves to be stated loudly, because it is the single prettiest idea in
   the project. NUDGE's circuit solver is a *deterministic* function
   `θ_cell → steady-state → readout`. A single cell is one evaluation. **JAX's
   `jax.vmap` turns that one deterministic solver into a population simulator for
   free:** draw a batch of per-cell parameter vectors `θ_cell` from a distribution
   (extrinsic cell-to-cell variation — different cells have different expression
   levels, ribosome counts, basal rates), `vmap` the solver across the batch, and
   the **empirical distribution of the batch's readouts *is* the predicted
   single-cell distribution**. No stochastic solver, no SDE, no per-cell Python
   loop — one vectorized XLA program that runs thousands of cells in parallel on
   a GPU. Three properties make this more than a convenience:

   - **It stays fully differentiable.** The randomness lives in the *sampled
     parameters*, not inside the ODE solve, so the reparameterization is clean:
     gradients flow from a population-distribution loss all the way back to the
     shared kinetic parameters *and* to the hyperparameters of the per-cell
     distribution (its mean and spread). You are fitting "what is the circuit,
     and how much do cells vary" in one differentiable object.
   - **The deterministic→stochastic bridge is distributional-over-inputs, not
     stochastic-in-the-solver.** That is exactly the regime MADDENING was built
     for — `run_sweep` already `vmap`s the graph over initial conditions for
     parameter sweeps. NUDGE reuses that machinery; the "sweep" is now "the cell
     population."
   - **It matches the biology.** A Perturb-seq condition is not one cell, it is a
     *cloud* of cells sharing a genotype but differing in state. `vmap`-over-θ is
     the literal computational image of that cloud. The thing MADDENING does
     anyway (batch a deterministic solve) is the thing single-cell biology needs.

   In one line: **MADDENING becomes a single-cell simulator not by adding
   stochasticity to the solver, but by `vmap`-ing the deterministic solver over a
   distribution of cells — and because that keeps everything differentiable, the
   fit and the design inversion come along unchanged.**
2. **A real observation/`Readout` model — the biggest honesty point.** The
   switch variable (Ras-GTP, ppERK) is *protein/phospho state and is never
   measured*. Perturb-seq measures **mRNA counts of downstream targets**. The
   `Readout` must therefore be a genuine latent→observed link: latent switch
   state → expected expression → **negative-binomial counts with dropout and
   library-size variation**. We fit to downstream transcriptional reporters of a
   latent switch, not to the switch itself. Getting this layer right is where the
   project's credibility is won or lost.
3. **Stochastic optimization over cells.** Datasets are 10⁵–10⁶ cells. The fit is
   a **minibatch SGD loop (optax) over cell minibatches**, not one deterministic
   loss — new plumbing around MADDENING's per-step `jax.grad`, but standard.
4. **Perturbation strength is latent and variable.** CRISPRi knockdown is
   *partial and cell-to-cell variable*; Perturb-seq cells can carry multiple
   guides (MOI); guide-to-cell assignment is itself noisy. `Perturbation` must be
   a fitted continuous latent (possibly per-cell), not a binary flag — which the
   brief already anticipated, but the *variability* and *assignment noise* are
   the parts to design for.
5. **A distributional loss, not MSE-on-means.** Fit simulated vs observed cell
   distributions with an **energy distance / MMD** (pertpy already ships
   "E-distance" perturbation metrics we can borrow) or explicit distributional
   moments (fraction-activated, bimodality coefficient). This is what makes the
   threshold/gain signal identifiable at all.

### The dataset ladder — synthetic-first, real-validated

Deliberately tiered from *ground-truth-known* (build & CI) to *ground-truth-
unknown but real* (validation). Develop up the ladder; never skip to the top.

| Tier | Data | Ground truth? | Role |
|---|---|---|---|
| **0** | **Synthetic from NUDGE's own circuit models** | Yes, exact | CI backbone, self-consistency, day-one dev |
| **0.5** | **Synthetic from an *independent* simulator** — SERGIO (chemical-Langevin GRN) or **BoolODE** (which generates *bifurcating* topologies = bistable switches with known ground truth) | Yes, from a different model | **Inverse-crime guard** — robustness to model misspecification |
| **1** | **Small real curated perturbation data** via **pertpy / scPerturb** (44 harmonized datasets: Adamson 2016 UPR is small & clean, Dixit 2016, Norman 2019) | No | Tune against *real* noise at manageable scale; `sc.datasets.pbmc3k` for pure AnnData-I/O smoke tests |
| **2** | **Gladstone genome-scale T-cell Perturb-seq** | No | Headline validation + the SOS/RasGRP1 falsifiable prediction |

(GEO and the Broad **Single Cell Portal** are the fishing grounds if we want an
additional small, documented switch-like system between Tiers 1 and 2.)

### The synthetic generator — the CI backbone, specified

The single most valuable piece of infrastructure to build first. Because NUDGE's
circuit *is* a generative model, we get a ground-truth simulator nearly for free:

```
generate_synthetic_perturbseq(
    circuit,               # a NUDGE circuit with known-true parameters
    conditions,            # WT/control + perturbations, each tagged with its
                           #   TRUE mechanism: threshold | gain | ceiling | combo
    n_cells_per_condition,
    noise_model,           # NB dispersion, dropout rate, library-size dist, batch
    realism_level,         # 0..3 difficulty dial (see below)
    seed,                  # deterministic → reproducible CI
) -> AnnData               # .X counts; .obs perturbation/guide labels;
                           # .uns['ground_truth'] = true params + mechanism labels
```

Pipeline: per-cell params (extrinsic noise) → vmapped steady-state circuit solve
→ latent switch state → `Readout` link → NB/dropout/library-size counts → AnnData
with **ground truth stashed in `.uns`** so tests can assert recovery.

A **realism dial** turns this into a difficulty ladder (and, not incidentally, a
robustness *curve* for the paper): `0` exact model · `1` +observation/count noise
· `2` +extrinsic cell heterogeneity · `3` +**misspecification** (an unmodeled
latent species or a wrong edge) — level 3 is the honest test, because levels 0–1
alone are the *inverse crime* (fitting data your own model generated is too easy
and proves nothing about real data).

### What synthetic ground truth lets CI actually assert

This is the point of the whole exercise — turning "we think it distinguishes
threshold from gain" into a green check:

- **Mechanism recovery.** Confusion matrix over {threshold, gain, ceiling}
  must beat a target accuracy on Tier-0/0.5 data.
- **Parameter recovery** within tolerance of the known-true values.
- **Uncertainty calibration — this is what makes the "fails safely and loudly"
  claim *tested* rather than asserted.** The Laplace 90% intervals must cover the
  truth ≈90% of the time. If coverage is off, the fail-loud promise is a lie, and
  CI catches it.
- **The false-positive guard.** On data generated by a *linear* circuit, the
  mechanistic fit must **not** claim a mechanism — it must tie the linear
  baseline. This is the artifact-chasing failure mode from FAQ Q5, made into a
  regression test.
- **The falsifiable prediction, dry-run.** Build a synthetic SOS-like feedback
  circuit; confirm feedback-knockdown collapses bimodality and graded-arm
  knockdown does not — proving the *logic* on ground truth before staking the
  claim on the real T-cell data.
- **The blindness diagnostic fires** on a synthetic bifurcation-region case.

### The decoy battery: synthetic negatives engineered to look positive

The positive-recovery tests above prove NUDGE can find a mechanism when one
exists. **The decoy battery proves the harder, more valuable thing: that NUDGE
*refuses* to find one when it shouldn't — loudly.** In drug discovery the
expensive failure is the confident false positive, so this is the CI suite that
directly earns the "fails safely and loudly" claim. Each decoy is a synthetic
dataset engineered so a *naive* method (fit a Hill, report the steepest one)
returns a confident positive; the **pass condition is that NUDGE returns the
correct negative with wide/abstaining uncertainty**, not a clean mechanism call.

| Decoy (true generator) | Why it *looks* like a switch | NUDGE's required response |
|---|---|---|
| **Linear circuit + saturating readout** | The sigmoidal *observation* function manufactures apparent ultrasensitivity that isn't in the circuit | Attribute to the `Readout`, **tie the linear baseline** — no circuit mechanism |
| **Two-cell-type / doublet mixture** | Population structure produces bimodality that mimics a bistable on/off split | **"technical / population structure"**, not a switch |
| **Dropout-driven zero peak** | Technical zero-inflation looks like a biological "off" mode; a capture-efficiency shift fakes a threshold move | Absorbed by the NB/dropout model → **no biological threshold call** |
| **Confounded threshold≈gain regime** | In the available #cells/#perturbations, a threshold shift and a gain change give near-identical distributions | **"unresolved"** (overlapping posteriors) — *not* a coin-flip pick |
| **Batch aligned with perturbation** | A batch/library shift rides along with one perturbation and masquerades as its effect | **Flag confounding**; refuse to attribute mechanism to the batch axis |
| **Off-target / off-model effect** | The perturbation moves the readout via a path outside the modeled circuit (off-target, global stress) | **Poor global fit / high residual → "distrust"**, not a mechanism |
| **Dead guide (≈0 efficiency)** | Near-WT data with a real perturbation label invites a spurious weak-effect call | **"no effect"** with the perturbation-strength latent pinned near zero |
| **Underpowered / pure sampling noise** | Too few cells; sampling fluctuation reads as an effect | Widen uncertainty and **abstain** |
| **Marginal-overfit Hill** | Noise tuned so a Hill fits *slightly* better than linear by chance | Margin must **not survive Laplace uncertainty** → no call |

Two design consequences fall directly out of this battery, and both are load-
bearing:

1. **NUDGE needs an output vocabulary richer than {threshold, gain, ceiling}.**
   The fail-loud promise has nowhere to land unless the tool can *say*
   `no-effect`, `unresolved`, `technical/population artifact`, and
   `off-model / poor-fit`. Designing these abstention-and-attribution categories
   is a first-class requirement, not an afterthought — the decoy battery is what
   forces them into the API.
2. **Each decoy has one *correct* answer, so each is a green-or-red CI test.**
   Build the battery early; it is the fastest way to develop the uncertainty
   gates and the readout/technical-noise model, because it fails informatively
   the moment a gate is too permissive.

A subtle payoff: the decoys are also the **best driver for the `Readout` and
noise-model work** (adaptation #2 above). You cannot pass "linear + saturating
readout" or "dropout zero-peak" without a genuinely good observation model, so
the battery pulls the hardest, most credibility-critical component forward in
the schedule instead of letting it slip.

### The thing most easily missed: use synthetic data to *pre-answer* identifiability

FAQ Q2 (is threshold-vs-gain even identifiable from a handful of perturbations?)
should not be argued rhetorically — it should be **measured**. Sweep the
synthetic generator over #perturbations, cells-per-condition, and noise level,
and map the boundary of the regime where mechanism recovery succeeds. That gives
us three things at once: (a) a hard paper figure ("identifiable above N cells and
K perturbations"), (b) a **pre-flight check** we can run on any *real* dataset to
say "your screen is/ isn't powered to resolve this" *before* fitting, and (c) an
honest scope statement instead of a hope. The data constraint, turned into a
power analysis, becomes a feature of the tool.

### Data hygiene: the standard single-cell pipeline is *hostile* to NUDGE

This is the counterpart to "don't feed it log-normalized data," generalized —
and it is a genuine adoption risk, because a computational biologist's muscle
memory is the **scanpy / Seurat clustering-and-DE pipeline, which is optimized to
*suppress* single-cell variation** (so clusters separate cleanly). NUDGE needs
the opposite: the raw single-cell *distribution* preserved, because that
distribution is the entire signal. Most standard preprocessing steps are
therefore not neutral — they are actively destructive, and they fail *silently*
(the fit runs, the answer is just wrong). NUDGE **owns the count model** — it
wants **raw integer counts** in and applies its own negative-binomial + dropout
observation model. The `fit()` boundary must refuse, or loudly warn on, anything
else. The offenders, worst first:

| Standard step | What it does to NUDGE | Verdict |
|---|---|---|
| **Imputation / denoising** (MAGIC, DCA, ALRA, scVI-denoised, kNN smoothing) | Smooths away the **bimodality that IS the switch signal** — manufactures a unimodal blob | **Catastrophic. Never.** |
| **Pseudobulk aggregation** | Collapses each condition to a mean, deleting the entire distribution NUDGE fits | **Catastrophic. Defeats the method.** |
| **Batch integration into a corrected space** (Harmony, scVI/scANVI embeddings, Combat, Scanorama) | Replaces counts with corrected values/embeddings NUDGE's count model can't consume; can erase *or invent* perturbation effects | **Reject.** Model batch *inside* NUDGE instead |
| **log1p / CPM / TPM / size-factor normalization; Seurat `data`/`scale.data` slots** | Not counts; breaks the NB likelihood and warps the distribution | **Reject — raw counts only** |
| **Nuisance regression** (`regress_out` cell-cycle, %mito, total counts) | Can remove *real* signal if the switch covaries with cell state; distorts residual distribution | **Avoid; pass covariates to NUDGE, don't pre-remove** |
| **HVG selection / target gene-panel subsetting** | May drop the very downstream reporter genes the `Readout` needs | **Ensure the readout genes survive** |
| **Ambient-RNA decontamination** (SoupX, CellBender) and **doublet removal** (Scrublet, DoubletFinder) | Removes genuine technical artifacts that would otherwise fake an "off" mode or a false bimodality (see decoy battery) | **Do this — on counts, upstream. The one class of preprocessing that helps.** |
| **Unequal cells-per-condition / sequencing depth** | Confounds a distributional comparison — a depth difference mimics an effect | **Model or match; never ignore** |
| **Guide-assignment thresholds / MOI filtering** | Define the effective perturbation labels; sloppy calls = garbage-in on the `Perturbation` latent | **Document the calling rule; treat as part of the data contract** |

The design consequence: NUDGE's ingestion layer should **inspect the AnnData and
fail loudly** — check that `.X` is integer-valued (not floats from log1p), warn
if a known corrected layer/embedding is being passed, and confirm the readout
genes are present — rather than trust the biologist to have preserved the raw
counts. "Fails safely and loudly" applies to the *input*, not just the output.
The only preprocessing NUDGE *wants* upstream is genuine-artifact removal
(ambient RNA, doublets, empty droplets) performed on counts; everything else it
does itself.

---

## Part 6 — Project structure, adapted from MADDENING's documentation architecture

MADDENING's `DOCUMENTATION_ARCHITECTURE.md` was built to make an open-source
research tool **auditable enough to sit underneath a Class III medical device** —
IEC 62304 SOUP packaging, ISO 14971 risk hooks, EU-MDR boundary language, the
works. NUDGE does not need that ceiling and should not carry that weight: **it is
research software, and dragging the full medical-device apparatus into a
hackathon-scale project would be cargo-culting.** But the *discipline* underneath
that apparatus — explicit scope, documented assumptions, known-failure registries,
provenance, a verification/validation split — is precisely what turns a clever
demo into a tool a biologist trusts and a reviewer (human or Claude Science's) can
cite. The right move is to **inherit the posture, not the paperwork.** Here is the
adapted, right-sized subset.

### The one idea to steal wholesale: the **Mechanism Card**

MADDENING's algorithm-documentation standard (its §3) gives every physics node a
structured doc with a fixed skeleton: governing equations, discretization,
assumptions & simplifications, **validated regimes**, **known limitations and
failure modes**, references, and verification evidence. **This is the single
highest-leverage thing to port**, because NUDGE's whole product is
mechanism attribution — so every mechanism in the library should ship with a
card:

> **Mechanism Card** (per `RegulatoryEffect` / `IntegratorModel` in the library)
> - **Governing equation** — the exact functional form (e.g. Hill activation
>   `v = V·xⁿ/(Kⁿ+xⁿ)`), with each parameter's biological meaning.
> - **What it can represent** — the regime this mechanism is *for*.
> - **Assumptions & simplifications** — e.g. "quasi-steady-state on the fast
>   variable," "no explicit cooperativity binding intermediate."
> - **Known failure modes** — where it mis-fits, and *which decoy in the battery
>   exercises that failure* (direct cross-reference).
> - **Identifiability regime** — from the synthetic power sweep: the data regime
>   under which this mechanism's parameters are recoverable.
> - **Verification evidence** — the synthetic-recovery test IDs that prove it.
> - **References** — the primary literature (Das 2009, Huang-Ferrell 1996, …).

The Mechanism Card is not documentation *about* the code — it is the artifact
that makes the abstain vocabulary and the decoy battery legible. When NUDGE
returns `unresolved` or `off-model`, the card is *where the user learns why that
was the honest answer*.

### The verification/validation split (its §4), translated

MADDENING separates **verification** ("does the code compute what we claim,
correctly") from **validation** ("does it match reality for a given context of
use"), and states plainly *where the framework's responsibility ends and the
downstream user's begins*. NUDGE's analog is clean and worth adopting verbatim in
spirit:

- **Verification** = the synthetic ground-truth suite (Part 5): mechanism
  recovery, parameter recovery, uncertainty calibration, the decoy battery. These
  prove NUDGE *does what it says* and *fails when it should*. They run in CI.
- **Validation** = the T-cell SOS/RasGRP1 falsifiable prediction against real
  data. This proves NUDGE *answers a real biological question*.
- **The boundary statement** (steal this framing directly): *NUDGE attributes
  mechanism **given a circuit hypothesis and a context of use** (raw-count
  Perturb-seq, steady-state snapshot, a powered screen). It does not prove the
  hypothesis, certify the data quality, or own the biological validity of the
  conclusion — the biologist does.* This is the same layered-responsibility model
  as MADDENING's, one layer down: NUDGE provides the auditable engine; the user
  owns their COU.

### Five lighter hooks worth the small cost

Each is cheap, each buys disproportionate credibility, and each has a direct
MADDENING ancestor:

1. **A `known_limitations.yaml` registry** (from MADDENING's machine-checked
   anomaly registry, §9.7). A living, CI-validated list of known failure modes —
   most of them *already enumerated by the decoy battery*. It is the honest
   counterweight to the press release, and CI can assert the docs haven't drifted
   from reality.
2. **Provenance on every result** (from §9.2). Stamp each fit/design output with:
   input data hash, circuit hypothesis, NUDGE + MADDENING + JAX versions, random
   seed, and the loss/metric values. This is nearly free, makes results
   reproducible, and **composes directly with Claude Science's per-figure
   provenance** (Part 2) — the two provenance layers stack.
3. **A `fit()`/`design()` stability contract + changelog** (from §5). Mark the
   public two-verb surface `stable`, everything else `experimental`; keep a real
   CHANGELOG. This is what makes NUDGE a *reusable tool* (the brief's stated goal)
   rather than a one-off script, and it is the surface the MCP server exposes.
4. **A "New Mechanism" contributor checklist** (from the New Node Checklist, §7).
   To add a mechanism to the library you must supply: its Mechanism Card, a
   synthetic-recovery test, and at least one decoy it must correctly resist. This
   keeps the abstain-and-attribute guarantees from rotting as the library grows.
5. **A "Capabilities NOT provided" section** in the README (from §10). Explicit
   non-goals: NUDGE is not a Perturb-seq hit-caller, not a black-box response
   predictor, not a clinical/diagnostic tool, not a substitute for a wet-lab
   screen. Scope discipline stated up front is the same move as MADDENING's
   intended-use statement, and it pre-empts half the skeptical questions.

### What to explicitly *leave out* (so the discipline doesn't become theatre)

Named so nobody reflexively ports them: the full **IEC 62304 SOUP package**,
**ISO 14971 risk-management file**, **EU-MDR / FDA boundary documents**, **safety
classification**, **Notified-Body / QMS apparatus**, and **health-check watchdog
nodes**. These exist in MADDENING because *something downstream might be a
device*. Nothing downstream of NUDGE is a device; NUDGE proposes hypotheses for a
wet lab. Keep the **spirit** these encode — auditability, honest limitations,
provenance, reproducibility — and drop the regulatory machinery. A single
lightweight **dependency/provenance note** (versions of JAX, MADDENING, scanpy,
optax) is the one trace of the SOUP idea worth keeping, because reproducibility
needs it anyway.

> **The through-line.** MADDENING's doc architecture answers "how does a solo
> researcher make software credible enough that others stake real decisions on
> it?" NUDGE inherits the *answer* — explicit scope, documented mechanisms, known
> failure modes, verification-vs-validation, provenance — at one-tenth the
> weight, because its decisions are "which hypothesis to test next," not "which
> patient to treat." Same discipline, proportionate to the stakes.

---

## Appendix — one-line strategic read

NUDGE's defensibility is not "we fit ODEs to Perturb-seq" (CellBox did, in
2021). It is the *conjunction* of (a) per-mechanism attribution instead of a
global nonlinearity, (b) a hardened differentiable-through-converged-solve
adjoint borrowed rather than reinvented — the unfair advantage that makes a
one-week build possible at all, (c) a named theory of when to distrust the fit
near a switch, expressed as a product that fails loudly and cheaply, and (d) a
distribution wedge into Claude Science as the mechanism-attribution layer its
orchestration is missing. The T-cell SOS/RasGRP1 prediction is the entry wedge —
one cheap, falsifiable, mechanistically-grounded call on known biology that
earns the tool the right to propose novel combinations on unknown biology.
