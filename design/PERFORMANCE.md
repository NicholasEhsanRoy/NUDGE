# NUDGE — performance profiling report

*Measured, not guessed.* Every number below comes from a benchmark script under
`scripts/perf/` run on this machine (CPU-only `jaxlib==0.5.1`, few-thousand-cell / ≤6-
species problem sizes). Reproduce: `uv run python scripts/perf/<script>.py`. Inspiration:
Astral (ruff/uv) — treat performance as a measured, first-class feature.

The motivating pain is **demo latency**: big-data loads take minutes and the first JAX
fit stalls. This report breaks each hot path into *one-time* vs *per-call* cost, because
for a live demo the first-call latency is what hurts.

---

## TL;DR — top wins

| # | Win | Where | Measured | Effort/Risk | Status |
|---|-----|-------|----------|-------------|--------|
| 1 | Coalesced-runs CSR gather | loader | **4.6–5.4× / 1.7–2.0× gzip** | S / low | **IMPLEMENTED** |
| 2 | Warmup call for cached jits | dose-resp, `_nd_kernel` | first fit **405→55 ms** | S / low | recommended |
| 3 | `n_boot` 500→200 default | dose-response | **2.66→1.09 s (2.4×)**, CI same | S / low-med | recommended |
| 4 | Hoist lyapunov `step` to cached jit | lyapunov | **~500 ms recompile ×3–4/attr** | L / med | recommended |
| 5 | vmap-batched JAX bootstrap | dose-response | ceiling **0.14 µs** vs 5.2 ms/boot | L / med | recommended |

(All wins ✅ help demo latency — the first-call/load stall is the motivating pain.)

GPU verdict (one line): **stay CPU** — the workload is many *tiny sequential* ODE solves
+ scipy least-squares + compile-dominated latency; none of it is GPU-favorable, and an
8 GB CUDA jaxlib would add launch overhead + longer compiles for ~zero gain.

---

## 1. Loader — the CSR row-gather (IMPLEMENTED)

**Hot path (prior profiling: ~99% of loader time).** `_read_h5ad_rows` selects only the
wanted cells × genes from a backed `.h5ad`. For CSR it reads the tiny `indptr`, then
gathers the selected rows' nonzero ranges from `data`/`indices`. The old code did one
big **h5py fancy index** `node["data"][flat]` where `flat` is the concatenation of the
selected rows' `indptr` ranges. On the *scattered* selections NUDGE actually produces
(per-condition subsampling → thousands of non-adjacent rows), that fancy index is slow:
h5py pays a large per-element selection overhead.

**Fix (implemented, `src/nudge/data/loaders/perturbseq.py::_coalesced_gather`).** Coalesce
the sorted selected ranges into maximal contiguous `[lo, hi)` runs and read each run as a
**slice** `dset[lo:hi]` (a cheap h5py hyperslab), then concatenate. Same bytes, same
order → **byte-identical output** (proven: 48-case fuzz over random shapes + gzip, 0
mismatches; `tests/data/test_perturbseq_loader.py` green incl. the slow peak-RSS test).
Crucially it stays **O(selection), not O(file)** — it never reads unselected rows, so it
holds at 150 GB (unlike a naive read-the-whole-span-then-mask, benchmarked below as
`coalesced-span`, which reads the entire min→max span — the whole file for a scattered
selection).

**Measured** (`scripts/perf/bench_loader_gather.py`), 40k–200k-cell synthetic CSR,
scattered selection (~3–4% of the matrix):

| strategy | uncompressed | gzip | notes |
|---|---|---|---|
| current (fancy-index) | 1.0× (baseline) | 1.0× | h5py per-element overhead |
| **coalesced-runs (slice)** | **4.6–5.4×** | **1.7–2.0×** | ✅ shipped, I/O-proportional |
| coalesced-span (mask) | 4.2–6.8× | 1.7–1.9× | ❌ reads whole span → all 150 GB |
| rdcc-256 MB (fancy) | 1.0× | 0.9× | chunk cache doesn't help scattered |
| coalesced + threads | 1.3–1.7× | 1.0× | thread overhead > gain on tiny reads |

**Why the others lose.** `rdcc` (bigger chunk cache) does nothing because scattered rows
rarely re-hit a chunk. Threads lose because each run read is small and the concatenate +
pool overhead dominates (h5py does release the GIL, but there's not enough work per run).
The span/mask variant is fast *here* only because the synthetic selection already spans
the file; at 150 GB it would read the whole matrix — exactly the constraint we must not
violate.

**Scaling to 150 GB.** The gather is now fast *per selected byte* and reads only selected
bytes. For the real Gladstone read (6,367 of 2.79 M cells, an IEG panel), selected nnz is
~0.2% of the matrix, so wall-time tracks the selection (seconds), not the file. The gzip
speedup is smaller (decompression of touched chunks is the floor), but real screens are
often gzip'd, so the honest expected win there is **~1.7–2×**, and **~5×** on uncompressed
files. Either way it removes the fancy-index constant that dominated.

---

## 2. JAX warmup vs compute (`scripts/perf/bench_jax_warmup.py`, `bench_dose_response.py`)

The demo-latency killer is **first-call XLA compilation**. Two distinct regimes, measured:

**(a) Properly cached jits — a warmup call fixes them.**
- `_jax_model` (dose-response Hill predict + jacfwd) is `lru_cache`d. First
  `fit_dose_response` (n_boot=0) = **405 ms**, of which **~350 ms is one-time compile**;
  the warm call is **55 ms**. A single throwaway warmup fit amortizes it.
- `Circuit._nd_kernel` (N-D fixed-point finder) is cached **per topology**: toggle first
  call **512 ms** (trace+compile) → **2.0 ms** warm (**257×**, matching the documented
  ~333× `_nd_kernel` win in FINDINGS). A *new* topology recompiles (expected — the cache
  is keyed on circuit structure, kinetics enter as a traced arg). 1-species switches use
  the closed-form root path and never pay this.

**(b) A jit that recompiles on EVERY call — a warmup does NOT help; needs a refactor.**
`fit_lyapunov_parameters` defines its optax `step` (and the `nll` closure over `circuit`,
`base`, `optimizer`, `free`, `k_modes`) **inside the function**, so each call builds a
fresh closure and XLA re-traces + re-compiles. Measured across 6 back-to-back identical
calls (30 steps, whose *compute* is only ~45 ms): **917, 604, 707, 535, 547, 524 ms** — it
plateaus near **~520 ms and never reaches the compute floor**. The marginal cost is only
**1.46 ms/optax step**; the fixed **~500–570 ms/call is pure compile**. Because
`attribute_lyapunov_single` runs 3 restricted fits (+ `calibrate_from_wt`), a single
attribution pays **~1.5–2 s of recompilation**. `fit_lyapunov_multi` has the same pattern.

**Fix (recommended, win #4).** Hoist the jitted `step`/`nll` to a module-level function
with the circuit topology as a cached/`static` argument and the kinetics/data as traced
args (the same trick `_nd_kernel` already uses successfully). This would turn ~500 ms/call
into a one-time ~500 ms compile per topology, amortized across all restricted fits and
conditions — plausibly **3–10× on end-to-end lyapunov attribution**. Effort L, risk medium
(must preserve exact numerics; guard with the existing `tests/inference/test_lyapunov.py`).
Not implemented here (needs careful numeric-equivalence validation); flagged as the
highest-value *deep* win.

---

## 3. Dose-response bootstrap (`scripts/perf/bench_dose_response.py`)

**The bootstrap dominates.** MLE-only (`n_boot=0`, the two Hill fits) is **55 ms** warm;
the default **`n_boot=500` is 2.66 s** — i.e. **~98% of the cost is the bootstrap loop**,
at **~5.2 ms/resample**.

**Is that JAX?** No. A single jitted `predict`/`jacfwd` eval is **~7 µs**; a resample's
scipy `curve_fit` does only ~tens of those. The 5.2 ms/boot is **scipy `curve_fit` Python
overhead** on a 16-point problem — and the loop runs **3 warm-start seeds per resample**
(`boot_seeds`), so ~1.7 ms per `curve_fit`. JAX is not the bottleneck here.

**`n_boot` sweep — the CI is already stable well below 500** (3 seeds each):

| n_boot | wall (warm) | 95% CI on n | width | seed-to-seed sd (lo/hi) |
|---|---|---|---|---|
| 0 | 56 ms | — | — | — |
| 50 | 386 ms | [4.72, 5.32] | 0.60 | 0.02 / 0.08 |
| 100 | 719 ms | [4.66, 5.44] | 0.78 | 0.04 / 0.03 |
| **200** | **1.09 s** | **[4.67, 5.47]** | **0.80** | **0.03 / 0.03** |
| 500 (default) | 2.66 s | [4.65, 5.45] | 0.79 | 0.02 / 0.02 |
| 1000 | 5.45 s | [4.65, 5.47] | 0.83 | 0.03 / 0.03 |

The percentile CI on `n` is converged by **~200 resamples** (width and seed-sd match
n_boot=500 to ≤0.02). **Recommendation:** drop the default `n_boot` 500→**200** — a **2.4×
latency cut** with statistically indistinguishable CIs, and expose `n_boot` on the demo
path (n_boot=100 → 3.8×, still fine interactively). *Not silently changed* because the CI
gates the `switch/graded/unresolved` verdict, so a default change touches the fail-safe
surface and deserves an explicit review + a re-run of the OCT4/NANOG regression. Two other
cheap levers, same caveat: use **1 warm-start seed** in the bootstrap instead of 3 (≈3×),
and add a warmup (win #2).

**vmap-batched bootstrap (win #5, deep).** The ceiling is real: a vmapped `predict` over
B=500 runs in **70 µs total = 0.14 µs/item** vs 5.2 ms/boot — a ~4-order-of-magnitude
headroom on the *evaluation*. Capturing it needs a JAX-native **batched bounded
Levenberg-Marquardt / Gauss-Newton** over all resamples at once (replacing 500 sequential
scipy solves), validated to reproduce `curve_fit`'s bounded multi-start CIs. Effort L, risk
medium. Recommended if dose-response becomes a hot interactive path; not prototyped here.

---

## 4. GPU verdict — honest: stay on CPU

I cannot run CUDA here (CPU-only jaxlib), but the problem sizes are measured and the
conclusion follows from them, not hype:

- The heaviest single kernel, `solve_population` (vmap of a 500-step semi-implicit scan
  over per-cell tiny ODEs), is **0.9 ms @256 cells, 3.2 ms @1000, 13.7 ms @4000** on CPU.
  It's a **sequential 500-step scan** with a handful of flops per step — GPUs win on large
  dense linear algebra, not long sequential scans over 1–2-element states. vmap gives cell
  parallelism, but per-step work is too small to hide kernel-launch/host-sync latency.
- The actual latency killers are **(a) XLA compilation** (350–570 ms/first-or-per call) and
  **(b) scipy `curve_fit`** (2.6 s bootstrap). GPU makes (a) *worse* (longer compiles) and
  cannot touch (b) (scipy is CPU/LAPACK).
- Individual jitted ops are microseconds (7 µs eval, 2 ms warm `_nd_kernel`); at these
  sizes GPU kernel-launch overhead (~10s of µs/launch + host↔device transfer) typically
  *loses* to CPU.
- **8 GB limit:** irrelevant — these problems are tiny — but a CUDA jaxlib adds install
  fragility and version pinning for no measured benefit.

**Verdict:** do **not** add a CUDA jaxlib. The wins are algorithmic (coalesced I/O, cache
the jit, fewer/batched bootstraps), not hardware. Revisit only if NUDGE moves to
genome-*wide* per-cell fits over 10⁵–10⁶ cells simultaneously (a different regime).

---

## Safe quick wins vs risky/deep

**Safe quick wins** (low risk, do now):
- ✅ **Coalesced-runs gather** — *implemented*, byte-identical, tests green.
- **Warmup call** at CLI/demo/MCP startup: one throwaway `fit_dose_response(..., n_boot=0)`
  + one `Circuit.fixed_points()` per demo topology → first user-visible fit ~7× faster.
  Zero behavior change (pure ordering). Helps only the *cached* jits (§2a), not lyapunov.
- **Expose `n_boot`** on the demo path (default it to 200) — 2.4× with unchanged CIs.

**Risky / deep** (measure-and-review before merging; each changes numerics or is a rewrite):
- **Hoist lyapunov `step` to a cached module-level jit** (§2b, win #4) — biggest attribution
  latency lever (~500 ms/fit → one-time), but must prove numeric equivalence.
- **vmap-batched JAX-native bootstrap** (win #5) — ~orders-of-magnitude headroom, but a
  bounded-LSQ reimplementation that must reproduce `curve_fit` CIs.
- **Change the `n_boot` default / bootstrap seed count** — touches the fail-safe CI gate;
  needs the V&V regression re-run, not a silent flip.

---

## What was implemented on this branch vs recommended

- **Implemented + measured:** win #1 (coalesced-runs CSR gather) in
  `src/nudge/data/loaders/perturbseq.py` — byte-identical (48-case fuzz), loader tests
  green (incl. slow peak-RSS), ruff + pyright clean.
- **Measured + recommended (not implemented):** wins #2–#5 above. All are backed by the
  benchmark scripts in `scripts/perf/`; none were merged because they either change output
  numerics (fail-safe surface) or are L-effort rewrites warranting explicit review.

Benchmark scripts: `scripts/perf/bench_loader_gather.py`, `bench_dose_response.py`,
`bench_jax_warmup.py`.
