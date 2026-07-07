# NUDGE — the pitch

*Node/edge Ultrasensitivity Diagnostic for Gene-regulatory Effects*
Built for **Built with Claude: Life Sciences** Hackathon, July 2026

*(The plain-language version. The full engineering reasoning lives in
`WORKING_BACKWARDS.md`; this is the one you can read in four minutes.)*

---

## The problem, in one breath

When you run a big genetic screen, you learn *which* genes matter. You almost
never learn *how* they matter — and "how" is the part that decides whether a drug
works.

Here's the bit that catches people out. Two genes can look identical in a screen —
same effect size, same ranking — and behave completely differently in a patient.
One is a **dial**: turn it and the response moves smoothly, so you can find a safe
dose. The other is a **cliff**: nothing happens, nothing happens, then one small
step tips the whole cell over — and sometimes it stays stuck there. Same number on
the screen. Totally different drug program. Teams routinely find out which one
they've got a year and a few million dollars too late.

NUDGE tells you at the start.

## What NUDGE actually does

Point it at a perturbation screen (Perturb-seq data) and a rough sketch of the
circuit, and for each gene you knocked down it tells you whether the knockdown:

- moved the **threshold** (where the switch trips),
- changed the **gain** (how sharply it flips), or
- shifted the **ceiling** (how far it can go).

Then — because the whole model is differentiable — it can run backwards: *given a
result you want, what should I try next?* Including gene combinations that were
never in the original screen. Every suggestion comes with an honest error bar.

And when it can't tell? It says so, out loud. That turns out to be one of the best
parts — more below.

## Why we can build this in a week (the honest version)

We didn't write a differentiable ODE solver from scratch this week. We
didn't have to.

The genuinely hard part of this project — doing calculus cleanly through a solver
that's already settled into a steady state, and staying stable right at the switch
point where naive optimizers quietly fall apart — was already built, tested, and
hardened in **MADDENING**, a differentiable physics engine originally made for
magnetically-steered microrobots. It turns out a bistable gene switch and a
microrobot near its tipping point are the *same math problem* underneath. So we
borrowed the engine and spent our actual time on the biology.

That's the moat, and we'll say it plainly: **we're not spending the week
building math — we're using a hardened physics engine to do biology.**

## The clever bit: heterogeneity is the signal

Most single-cell tools treat cell-to-cell variation as noise and average it away.
For us, that variation *is* the answer.

Whether a gene moved a threshold or a gain shows up in the **shape** of the cell
population — how the cells split into "off" and "on" groups, and how sharp that
split is. Average it, and you erase the exact thing you're trying to measure.

There's a tidy trick that makes this cheap. Our model runs one cell at a time,
deterministically. JAX's `vmap` lets us run *thousands* of cells at once — each
with slightly different settings, just like real cells — in one fast computation
on a GPU. The spread of results across that crowd *is* the predicted population.
One clean idea does all the work, and it stays differentiable, so the "what should
I try next" part comes for free.

## Fail safely, fail loudly

This is the part we're proudest of. In drug discovery the expensive mistake isn't
missing a hit — it's a **confident wrong answer** that sends a team chasing a
mirage for a year.

So we built NUDGE to fail *loudly*. We test it against a whole battery of
**decoys** — datasets rigged to look like a real switch but aren't. A boring
linear response dressed up by a saturating measurement. Two cell types faking a
clean on/off split. Technical dropout mimicking an "off" state. A naive tool calls
every one of these a confident hit. NUDGE is built to look at each and say *"nope,
that's an artifact,"* or *"I genuinely can't tell from this data."*

When NUDGE says **"unresolved,"** that isn't a failure. That's the tool doing the
most valuable thing it can do: saving you the year.

## Proving it's real, right now

We're not waving our hands about "generic hits." We picked a switch the field
already understands — T-cell activation, driven by the SOS feedback loop (Das et
al., *Cell* 2009) — and made a falsifiable call *before* looking:

> Knock down **SOS** (the feedback that *creates* the switch) and the sharp,
> digital activation signature should collapse toward a smooth, graded one. Knock
> down **RasGRP1** (the smooth arm, no feedback) and it shouldn't.

If we're right, that's a crisp mechanistic prediction a black-box model
structurally can't make. If we're wrong, that's a real, publishable result about
the biology — not a broken tool. Either way we learn something true, this week.

## Where it lives

NUDGE ships as a command-line tool and a small local app — but its natural home is
**Claude Science**, Anthropic's new research workbench. Claude Science is already
great at pulling data, running quality control, and predicting protein structures.
What it *doesn't* have is a layer that reasons about the **dynamics** of a
regulatory circuit — thresholds, gains, feedback. That's exactly the gap NUDGE
fills. It plugs in as an MCP tool, so a scientist can just ask, in plain English,
*"which of these regulators moves the switch's threshold?"* and get an answer back
with its uncertainty attached and a full record of how it was produced.

**NUDGE isn't a one-off script. It's the missing mechanism-attribution layer for
modern AI research environments.**

## Why you can trust the "I'm not sure"

A tool that says "unresolved" is only useful if you actually believe it. So NUDGE
borrows a *provenance posture* from medical-grade software (via MADDENING's
documentation architecture, right-sized — no regulatory theatre). Every mechanism
in the library ships with a plain-language card saying what it assumes and where it
breaks. Every result is stamped with the data, versions, and settings that made it.
The known failure modes are written down, not buried.

That's what lets a researcher trust an *"unresolved"* label instead of
second-guessing it — and it stacks neatly on top of Claude Science's own
reproducibility tracking.

## What NUDGE is *not*

Being upfront saves everyone time:

- It's **not** a general hit-caller — it answers a sharper question than "is this
  gene a hit?"
- It's **not** a black-box predictor — the whole point is the mechanism, not just
  the number.
- It's **not** a clinical or diagnostic tool.
- It **won't** replace a wet-lab screen — it tells you which experiment is worth
  running next.

## The one-liner

**NUDGE reads a genetic screen and tells you which targets you can safely dose and
which are cliffs — borrowing a hardened physics engine to do the hard math,
proving it on a known biological switch, saying "I'm not sure" out loud when it
isn't, and plugging straight into Claude Science.**
