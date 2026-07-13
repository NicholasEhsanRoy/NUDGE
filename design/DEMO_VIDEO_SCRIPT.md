# NUDGE — 3-minute demo/pitch video script

*Built with Claude: Life Sciences. This doc is the shooting script for a ~3-minute
demo/pitch video. It contains (1) a brief cited research synthesis, (2) the visible
iteration trail from v1 to the final, (3) the final timestamped script, (4) a shot/asset
list, and (5) a one-paragraph mapping to the four judging criteria.*

**Honesty north star (non-negotiable, carried through every beat).** NUDGE is the
"honesty engine": it reports **measured, not asserted**, and its one hard failure is a
confident-wrong. The video's defensible edges are **scale (the OOM wall), speed, rigor,
reproducibility, and provenance** — *never* "raw Claude can't do this." Every headline
number traces to the repo (`scripts/vv/FINDINGS.md`, `design/STATE.md`, `CHANGELOG.md`,
the two `scripts/demo_*` scripts). The Alzheimer's demo uses a **real published model
topology + rate-law forms with demo-scaled constants and a synthetic cohort**
(`NUDGE-LIM-026`) — it is *never* framed as a clinical finding or real patient data.

---

## 1. Research synthesis — what the evidence says (brief, cited)

Five angles were researched (developer-tool demo structure; hackathon judging + the hook;
honest on-camera A/B; AI-for-science positioning; the close). The actionable takeaways:

1. **The first ~15 seconds decide retention — hook with the problem, not the architecture.**
   Judges (and viewers) tune out on jargon; open like a movie trailer and make them *feel*
   the pain or see the result before any tech. Lead with a surprising line or a visible
   failure. ([Devpost best practices](https://help.devpost.com/hc/en-us/articles/360021816952-Video-making-best-practices),
   [Spiel: demo structure](https://www.spielcreative.com/blog/demo-video-structure/),
   [BizThon: pitch like a pro](https://medium.com/@BizthonOfficial/pitch-perfect-how-to-present-your-hack-like-a-pro-1104430a5d93))

2. **Show, don't tell; cut anything that doesn't move the story; ~3 s/screen, not 10.**
   "Instruction is often mistaken for value" — a benefit has to be *visible inside the
   workflow*, not narrated as a claim. A feature-list demo is documentation, not a demo.
   The single strongest device is a split-screen of the clunky/failing way vs. the new way.
   ([SmoothCapture: software demo](https://www.smoothcapture.app/blog/software-demo-video),
   [Spiel](https://www.spielcreative.com/blog/demo-video-structure/))

3. **An honest on-camera A/B keeps *everything* equal except the one variable.** Same
   hardware, same lighting/quality, no sandbagging the "before" — otherwise the improvement
   looks like it came from production, not the product. For us that means: the *same laptop,
   same memory cap, same real model + synthetic cohort* on both sides of every WITH/WITHOUT
   split, so the OOM and the ellipse-collapse are self-evidently real.
   ([Wideframe: before/after](https://try.wideframe.com/blog/how-to-create-before-and-after-comparison-videos/),
   [Motion: demo methods](https://www.motiontheagency.com/blog/ways-to-make-a-product-demo-video))

4. **AI-for-science tools now earn trust by making the *whole computation* inspectable and
   by acknowledging uncertainty instead of hiding it.** Credibility is shifting from the
   output to "can I reconstruct, inspect, and verify every step?" — and confident-toned AI
   that obscures epistemic uncertainty is the failure mode the field is worried about. This
   is *exactly* NUDGE's differentiator: the abstention (uncertainty made loud) + the
   `fig.py` provenance grain (every figure reconstructible). We should aim the whole video
   at that gap. ([Royal Society: opaque AI tools](https://royalsociety.org/news/2024/05/ai-research-tools-could-undermine-trust-accuracy-scientific-findings/),
   [npj Digital Medicine: TREAT/regulatory AI](https://www.nature.com/articles/s41746-025-01596-0),
   [Computational provenance & adversarial reproducibility](https://medium.com/@y.marutani/computational-provenance-and-adversarial-reproducibility-in-ai-driven-scientific-research-3dd396e410f9))

5. **Close with a structured recap + one specific, low-friction CTA + a verifiable number.**
   Endings are remembered disproportionately; a concrete, checkable fact ("on PyPI as
   `nudge-bio`", "0% misclassification across hundreds of synthetic datasets") is social proof judges can repeat.
   ([Vidyard: strong CTA](https://www.vidyard.com/blog/end-video-strong-call-action-examples/),
   [PitchAvatar: ending for impact](https://pitchavatar.com/the-last-step-the-best-ending-for-a-presentation/))

**Net design implication.** Open on a *visible* confident-wrong (angle 1), state the honesty
thesis fast, then spend the body on two WITH/WITHOUT money-shots — the OOM wall and the
ellipse-collapse (angles 2–3) — land the `fig.py`/abstention **provenance** beat as the
thematic peak (angle 4), and close on measured social proof + `pip install` (angle 5).

---

## 2. Iteration trail (v1 → final)

### v1 — first draft beat sheet (~3 min)

1. **0:00–0:20 Hook.** VO: "This is NUDGE, a mechanism-attribution tool for biology." Show
   the logo and a repo tour.
2. **0:20–1:00 What it does.** Explain threshold vs. gain vs. ceiling, the circuit model,
   MADDENING, the fit engine. Diagram of the pipeline.
3. **1:00–1:40 Identifiability.** Talk through matrix-free vs. dense; show the scaling
   script output.
4. **1:40–2:20 OED.** Explain Fisher information and gradient OED; show the CRLB numbers.
5. **2:20–2:50 Claude.** Mention it was built with Claude Code and multi-agent loops.
6. **2:50–3:00 Close.** "Thanks — check out the repo."

**Self-critique of v1.**
- *Research angle 1 (hook):* fails badly — opens on a logo and a definition, exactly the
  "jargon / feature-list" anti-pattern. No emotional/visible hook in the first 15 s.
- *Angle 2 (show don't tell):* it's all *telling* — threshold/gain/ceiling exposition,
  pipeline diagrams. No workflow visible, no money-shot.
- *Angle 4 (provenance):* the single most on-thesis asset (`fig.py` + abstention) is absent.
- *Criteria:* Demo (30%) is weak (nothing running on camera); Claude Use (25%) is an
  afterthought; the honesty thesis — the whole differentiator — never appears.
- *Honesty:* no A/B, so "faster/scales" would be an unbacked claim.
- **Verdict:** reorder around a visible failure, cut the exposition, make the two
  capabilities WITH/WITHOUT splits, and promote provenance to a named beat.

### v2 — reordered around the honesty thesis + the two money-shots

1. **0:00–0:15 Hook.** A naive least-squares fit prints 12 confident per-subject rate
   constants; a red stamp drops: "unidentifiable." VO: "A model that's confidently wrong is
   worse than one that admits it can't tell."
2. **0:15–0:35 Thesis.** NUDGE = the honesty engine; abstains rather than guess; 0%
   misclassification across hundreds of synthetic datasets (300 linear + 120 switch measured at every margin_k); on PyPI as `nudge-bio`.
3. **0:35–0:55 Setup.** Real published Alzheimer's amyloid-β QSP model (Proctor 2013, CC0) +
   synthetic cohort; caveat on-screen; driven from Claude Science via the `nudge` MCP
   connector.
4. **0:55–1:35 Capability 1 — matrix-free identifiability, WITH/WITHOUT.** Dense Jacobian
   OOM-kills; NUDGE stays flat ~0.57 GB and certifies the verdict.
5. **1:35–2:15 Capability 2 — gradient OED.** k_on ⇄ k_gl confounded (corr ≈ 1.000); the
   ellipse-collapse GIF; CRLB ×259 measured.
6. **2:15–2:40 Provenance.** Figures/GIFs come back inline carrying their `fig.py` +
   data sidecar; pixel-identical replay. "Measured, not asserted."
7. **2:40–3:00 Claude + close.** Built with Claude Code; multi-agent hardening; `pip install
   nudge-bio`.

**Self-critique of v2.**
- *Angle 1 & 3:* much better — visible failure hook, honest A/B framing. Good.
- *Angle 3 (honest A/B) risk:* I must *state on camera that both sides run on the same
  laptop under the same 2.5 GB cap*, or a skeptic reads the OOM as staged. Not yet explicit.
- *Angle 4:* provenance is now a beat, but at 25 s it competes with Claude Use for the
  ending and both feel rushed. The provenance beat *is* the Claude-Science-integration beat —
  they should merge, freeing time.
- *Criteria — Claude Use (25%):* still thin. The strongest Claude story is the **multi-agent
  hardening loop that found and closed a real confident-wrong (P7) with an independent
  audit** — that's the "surprising capability" judges reward, and it *is* the honesty thesis
  applied to the tool itself. Name it concretely, don't hand-wave "multi-agent loops."
- *Pacing:* the setup (0:35–0:55) risks slowing right after the hook; tighten so momentum
  carries into Capability 1.
- *Honesty:* "CRLB ×259" must be attributed to the exact asset (the `oed` MCP tool on
  `ad_qsp`), and the demo-scaled/synthetic caveat must ride *with* the AD model on-screen,
  not just once.
- **Verdict:** make the same-hardware point explicit; fold provenance into the
  Claude-Science moment as the thematic peak; upgrade Claude Use to the concrete
  red-team→fix→audit story with P7; tighten the setup.

### v3 — sharpen honesty guardrails, elevate Claude Use, merge provenance

Changes from v2: (a) the OOM beat now *says* "same laptop, same 2.5 GB cap, same model" and
shows both memory traces side-by-side; (b) the provenance beat and the Claude-Science beat
are one moment — the GIF *arriving inline with its `fig.py`* is the visual, and the VO ties
it to "reconstruct and verify every step"; (c) Claude Use becomes the concrete story: an
auditable git history + a red-team → uq-fixer → independent-audit loop that found a
confident-wrong (P7) and *closed* it; (d) the AD caveat is a persistent lower-third whenever
the model is on screen; (e) the setup is cut to ~15 s.

**Self-critique of v3.**
- *Angle 2 (pacing):* the two capability beats are ~35–40 s each — right for a money-shot,
  but I should pre-commit to *hard cuts* (no lingering on terminal scrollback) and keep the
  memory/ellipse visuals moving. Add explicit "hold ≤3 s" notes to the shot list.
- *Angle 5 (close):* the CTA is present (`pip install`) but the *final line* should restate
  the thesis in one breath and leave the measured social-proof number on screen. Tighten the
  last 8 seconds to a single memorable sentence.
- *Angle 4:* strong now — provenance is the peak and doubles as Claude Use scaffolding.
- *Honesty:* one more guard — when we say the k_on⇄k_gl pair is "confounded," show corr ≈
  1.000 as the *reason* (not just an assertion), and keep "local OED at θ₀ (NUDGE-LIM-024)"
  in the lower-third so the ×259 isn't read as a global guarantee.
- *Criteria:* Demo 30% (two live money-shots) ✓; Claude Use 25% (auditable + self-red-team)
  ✓; Impact 25% (a real QSP identifiability problem + a general MCP tool over a model
  registry) — could be one sentence stronger in the thesis; Depth 20% (the confounded-pair
  resolution + the measured OOM wall) ✓.
- **Verdict:** minor timing/wording polish → **final**. Add the shot-list hold-times, tighten
  the close to one sentence, put one Impact sentence in the thesis, keep every caveat in a
  lower-third.

---

## 3. FINAL shootable script (~3:05)

**Format:** `[timestamp] VOICEOVER` then *on-screen action / B-roll*. Persistent
lower-thirds are called out where load-bearing. Target runtime 3:00–3:10. All numbers are
measured — see the trace table in §4.

---

**[0:00–0:13] — HOOK (the visible confident-wrong)**

> VO: "Ask a normal fitting tool for these twelve rate constants, and it will hand you
> twelve confident numbers. Every one of them is meaningless — the data can't identify them.
> A model that's *confidently wrong* is worse than one that says: I can't tell."

*On screen:* a terminal least-squares fit prints a tidy 12×N table of confident-looking
parameter values (from the `scripts/demo_ab/` raw-agent path). Beat. A red stamp drops over
the table: **"UNIDENTIFIABLE."** Hard cut. Hold the stamp ≤2 s.

---

**[0:13–0:31] — THESIS (the honesty engine)**

> VO: "This is NUDGE. It's a mechanism-attribution and experimental-design tool for
> biology, and its defining property is that it reports what it *measured*, never what it
> guessed. Across hundreds of synthetic ground-truth datasets, it misclassified a mechanism zero
> times — because when it can't be sure, it abstains, loudly. It's on PyPI, today, as
> `nudge-bio`."

*On screen:* NUDGE wordmark → quick cut to the FINDINGS table showing **0% misclassification
/ hundreds of datasets** (highlight the "0%" cell) → a `pip install nudge-bio` line auto-typing in a
terminal. Keep each ≤3 s.

---

**[0:31–0:46] — SETUP (real model, honest framing, driven from Claude)**

> VO: "Here's a real one. A published Alzheimer's amyloid-beta model — Proctor 2013, open
> and public-domain — with a synthetic patient cohort. We're driving NUDGE entirely from
> Claude, through a custom connector, inside Claude Science."

*On screen:* the Claude Science chat panel; a one-line prompt being typed ("Use nudge:
is this cohort identifiable? then design the schedule that resolves the antibody's effect").
The `nudge` MCP tools list flashes.
**Lower-third (persists whenever the AD model is on screen):** *"Proctor 2013 QSP
(BioModels BIOMD0000000488, CC0). Real topology + rate-law forms; demo-scaled constants +
synthetic cohort — not real patients, not a clinical finding. NUDGE-LIM-026."*

---

**[0:46–1:24] — CAPABILITY 1: matrix-free identifiability (WITH / WITHOUT, the OOM wall)**

> VO: "Population-scale, this is thousands of parameters. The textbook way builds a dense
> Jacobian. Same laptop, same memory budget — watch it. The dense build climbs, and dies:
> out of memory. NUDGE never forms that matrix. It works through matrix-vector products
> only, stays flat at about half a gigabyte, and returns the honest verdict:
> unidentifiable — with the plaque-growth gain and threshold named as the sloppiest knobs."

*On screen:* a split terminal. **LEFT ("dense jacfwd"):** a live memory meter rising, rising,
then a kernel **`Killed` / OOM** message (real, under the 2.5 GB `systemd` cap). **RIGHT
("NUDGE matrix-free"):** memory trace pinned flat (~0.57 GB), finishing with a green
**`unidentifiable`** verdict + the named sloppy directions (`k_pg`, `K_pg`). A small callout:
**"same machine · same 2.5 GB cap · same model."** Hold the OOM-vs-flat comparison — this is
money-shot #1 — but keep the meters animating so it's never a static hold.

---

**[1:24–2:04] — CAPABILITY 2: gradient OED (the ellipse collapse)**

> VO: "Now the design question. Two parameters — how fast the antibody binds, and how fast
> microglia clear the plaque — both lower amyloid, so a naive baseline-and-end scan can't
> separate them. Their correlation is basically one. NUDGE differentiates the information
> criterion through the whole ODE and *moves the measurement times* to break the tie. Watch
> the uncertainty ellipse collapse. Measured, on this model: the antibody parameter gets two
> hundred and fifty-nine times more identifiable."

*On screen:* the confounded (k_on, k_gl) scatter with a huge diagonal 95% ellipse; a
**corr ≈ 1.000** label. Then the **ellipse-collapse GIF** plays: sample times slide off the
baseline/end cluster into the dosing transient while the ellipse shrinks to a tight blob.
End card: **"CRLB ×259 · corr 1.000 → cond 22."** Money-shot #2.
**Lower-third:** *"`oed` MCP tool on `ad_qsp`. Local OED at θ₀ — measured, not extrapolated
(NUDGE-LIM-024)."*

---

**[2:04–2:32] — PROVENANCE (the thematic peak: measured, not asserted)**

> VO: "And here's the part that makes it science you can trust. That figure didn't just
> arrive as a picture. It came back carrying the exact code that regenerates it and the data
> behind it — so anyone can reconstruct and verify every step, no re-fit, pixel for pixel.
> The abstention, the one-sided bound, the collapse — all of it is reproducible provenance,
> not a claim."

*On screen:* the inline GIF in the Claude Science chat, then expand the attached **`fig.py`**
and **`fig.data.json`** provenance; run `fig.py` in a fresh shell → the *same* figure
renders. Overlay: **"inline artifact → `fig.py` + data sidecar → pixel-identical replay."**

---

**[2:32–2:57] — CLAUDE USE (built with Claude, and it red-teamed itself)**

> VO: "NUDGE was built with Claude Code — deliberately, with an auditable git history of
> every step. More than that: Claude agents adversarially attacked NUDGE's own honesty. One
> loop — a red-team, a fixer, and an *independent* auditor — found a case where NUDGE gave a
> confident, wrong answer, and closed it. Reporting a hole in your own tool as a win *is* the
> thesis, applied to itself."

*On screen:* a fast scroll of the git log with `Co-Authored-By: Claude` trailers; a diagram
of the **red-team → uq-fixer → audit** loop; the P7 record (confident-wrong → `NUDGE-LIM-025`
→ closed, audit PASS). Keep it moving; ≤3 s per element.

---

**[2:57–3:05] — CLOSE (recap + CTA, one breath)**

> VO: "NUDGE. It scales past the wall, it designs the experiment that resolves the
> confound, and it never claims more than it measured. `pip install nudge-bio`."

*On screen:* the wordmark; three tight text lines — **"Scales. Resolves. Abstains."** — then
the CTA card: **`pip install nudge-bio` · Built with Claude: Life Sciences**, with the
**0%** number still visible in the corner. Hold 3 s. End.

---

## 4. Shot / asset list

**Money-shots (capture these first — the video lives or dies on them):**

| # | Shot | Source / how to capture | Hold |
|---|---|---|---|
| M1 | **Dense-Jacobian OOM vs. matrix-free flat**, split screen with live memory meters | `uv run python scripts/demo_matrix_free_scale.py` (dense worker runs under the `systemd` 2.5 GB cap → real OOM-kill; matrix-free flat ~0.57 GB, same verdict). Screen-record both workers; overlay memory meters. | animate through |
| M2 | **95%-confidence-ellipse collapse GIF** (k_on, k_gl) | `uv run python scripts/demo_gradient_oed.py` writes the GIF (`nudge.viz.oed` animator); *or* the `oed` MCP tool on `ad_qsp` returns it inline (the ×259 figure). | play full GIF |
| M3 | **`fig.py` + `fig.data.json` inline → pixel-identical replay** | The `render_figure` / `oed` MCP tool under `NUDGE_ENV=cloud` returns `image_base64` + `code` + `data`; run the emitted `fig.py` in a fresh shell to reproduce. | ≤6 s |

**Supporting shots / existing assets to reuse:**

- **Confident-wrong hook table** — the raw-agent least-squares fit from `scripts/demo_ab/`
  (12×N confident constants) + a red "UNIDENTIFIABLE" stamp (added in edit).
- **FINDINGS "0%" table** — `scripts/vv/FINDINGS.md` §1 (highlight the 0% cell).
- **`pip install nudge-bio`** auto-typing terminal (PyPI 0.3.0).
- **Claude Science chat + `nudge` connector** — the prompt + the MCP tools list; recipe in
  `docs/user_guide/claude_science.md` (use `NUDGE_ENV=cloud` for inline figures).
- **FIM / sloppiness spectrum figure** — the `identifiability` tool's inline FIM-spectrum
  figure (names `k_pg`/`K_pg`/`k_dis` as sloppiest); optional insert over Capability 1.
- **Naive-schedule confound scatter** — corr ≈ 1.000 diagonal ellipse (from
  `demo_gradient_oed.py` `_confound`), the "before" of M2.
- **Git-log scroll** — `git log` showing `Co-Authored-By: Claude Opus 4.8` trailers.
- **Hardening-loop diagram** — red-team → uq-fixer → audit; the P7 record
  (`NUDGE-LIM-025`, `CHANGELOG` 0.3.0 Fixed; `design/FAILSAFE_REDTEAM*.md`).
- **Persistent lower-thirds:** the `NUDGE-LIM-026` caveat (whenever the AD model shows) and
  the `NUDGE-LIM-024` local-OED caveat (over Capability 2).

**Number-trace table (every headline claim → source; nothing asserted beyond these):**

| Claim in VO | Value | Source |
|---|---|---|
| never misclassified / abstains | **0% across hundreds of synthetic datasets (300 linear + 120 switch measured at every margin_k)** | `scripts/vv/FINDINGS.md` §1; `JUDGES_GUIDE.md` |
| on PyPI | **`nudge-bio` 0.3.0** | `CHANGELOG.md`; `docs/user_guide/claude_science.md` |
| dense build dies | **dense jacfwd OOM-killed at n_free ≥ 1000, 2.5 GB cap** | `scripts/demo_matrix_free_scale.py`; STATE.md "AD QSP" |
| NUDGE stays flat | **~0.57 GB (1.01×), same `unidentifiable` verdict** | same |
| sloppiest knobs named | **plaque-growth gain `k_pg` / threshold `K_pg`** (single-subject sloppy-but-predictive, cond ≈ 7e8, span 8.8 decades) | `scripts/demo_matrix_free_scale.py`; `NUDGE-LIM-026` |
| the two params are confounded | **corr(k_on, k_gl) ≈ 1.000** under a naive baseline+end schedule | `scripts/demo_gradient_oed.py`; STATE.md |
| identifiability gain | **CRLB ×259, corr 1.000 → cond 22** (`oed` MCP tool on `ad_qsp`) | `CHANGELOG.md` 0.3.0; STATE.md ("ad_qsp ×259 CRLB") |
| provenance replay | **inline `fig.py` + `fig.data.json`, pixel-identical** | `nudge.viz` provenance; `docs/user_guide/claude_science.md` |
| self-red-team found + closed a confident-wrong | **P7 → `NUDGE-LIM-025`, independent audit PASS** | `CHANGELOG.md` 0.3.0 Fixed; `scripts/vv/FINDINGS.md` §P7 |

> Note on the ×259: the **standalone script** `scripts/demo_gradient_oed.py` measures ×220
> CRLB / ×205 min-eig for the k_on⇄k_gl pair; the **general `oed` MCP tool** on `ad_qsp`
> (the on-camera Claude-Science path) measures **×259 CRLB / corr 1.000→cond 22**. The VO
> uses ×259 because that is the asset shown. Either is honest; keep the number matched to
> the footage on screen.

---

## 5. Rationale — how the structure maps to the four criteria

The script is deliberately weighted toward **Demo (30%)**: two-thirds of its runtime is live,
on-camera WITH/WITHOUT money-shots — a real out-of-memory kill vs. a flat memory trace, and a
confidence ellipse visibly collapsing — the two things "genuinely cool to watch" that a
feature-list narration can't buy. **Claude Use (25%)** is carried by the two most surprising
beats — the entire demo is driven *through Claude* via a custom MCP connector in Claude
Science, and Claude's own multi-agent loop adversarially found and closed a confident-wrong
in the tool (the auditable git history makes that checkable). **Impact (25%)** rides on the
choice of a real, published QSP identifiability/experimental-design problem exposed as a
*general* model-registry tool, not a toy — the kind of thing a modeler could actually use
tomorrow. **Depth & Execution (20%)** shows in the substance of the two capabilities (a
matrix-free FIM that beats the dense OOM wall; a differentiable OED that resolves a corr≈1.000
confound) and in the honesty guardrails visible on screen. And the **provenance beat is the
spine that ties all four together**: "measured, not asserted," reconstructible `fig.py`,
loud abstention — the differentiator the AI-for-science evidence says the field now rewards,
and the one thing a polished-but-inflated competitor can't fake.
