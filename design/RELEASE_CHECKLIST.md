# NUDGE — first PyPI release checklist (`nudge-bio` 0.1.0)

Living prep doc for the first `pip install nudge-bio`. **Gated on the post-moat hardening
loop** (see `design/hardening/`): we cut the release from a branch that a full red-team →
uq-fixer → audit sweep has driven to `HOLES_FOUND: 0`, so the first public version carries
the honesty guarantee, not just green tests.

## Verified ready (measured 2026-07-11, on `main` @ post-moat merge)

- [x] **`uv build` succeeds** — `dist/nudge_bio-0.1.0-py3-none-any.whl` + `.tar.gz`.
- [x] **`twine check dist/*` PASSED** — metadata valid, README renders as the PyPI long
      description.
- [x] **The hard dependency resolves from PyPI** — `maddening` is published (latest 0.3.1),
      so `maddening[ift]>=0.3.1,<0.4` is installable. *This was the #1 release blocker; it is
      cleared.*
- [x] **The distribution name is free** — `pypi.org/project/nudge-bio` is a 404 (untaken).
- [x] **Clean-venv install works** — a fresh Python 3.11 venv `pip install`s the wheel
      (pulling maddening/jax/anndata/zarr/… from PyPI), then `import nudge`,
      `from nudge import design`, and the `nudge --help` entry point all succeed.
- [x] Packaging metadata present: MIT `LICENSE`, `py.typed` (force-included in the wheel),
      classifiers, keywords, `project.urls`, console scripts (`nudge`, `nudge-mcp`).

## Blocked-until-hardening — CLEARED (2026-07-12, PR #1 merged as `2a85b88`)

- [x] **Post-moat hardening loop green** — the cloud loop (rounds 6–7, runs `000000018`–
      `000000026`) red-teamed the moat and found TWO confident-wrong holes, both fixed +
      independently audited, ending with `HOLES_FOUND: 0` on a genuine full re-scan (STOP).
      **P6** (matrix-free identifiability certified `well-constrained` on an isolated exact
      Fisher-null because the fail-safe checked eigenpair-ness, not smallest-ness) → fixed by
      routing `auto` through the exact dense-via-matvec reconstruction to `dense_below=2048` +
      a one-sided inverse-iteration null probe above that, abstaining rather than asserting
      (`NUDGE-LIM-023` sharpened to major). Repro `sloppiness_matrixfree_iterative_mislabel.py`
      now reports 0/6 holes.
- [x] **Close the pre-existing open hole P5** (differential small-multiplicative confound,
      `LIM-016`) — fixed by generalising to the whole per-condition affine class: gate 4d adds
      a free `(s,o)` nuisance on the perturbed context and abstains unless the winning knob
      earns ≥6.0 ΔBIC over the affine null (audit PASS `runs/000000020`).
- [x] **Re-verified after folding into main:** ruff / pyright / 5 doc checkers green;
      **349 pytest passed / 0 failed**; the merged tree is byte-identical to the locally-gated
      commit. **The release is UNBLOCKED** — only the release-time steps below remain.

## Release-time steps (once the branch is green, in order)

1. [ ] **Decide the version.** `0.1.0` is correct for a first public pre-alpha (the
       `Development Status :: 2 - Pre-Alpha` classifier matches). Bump only if the hardening
       loop changes the public API.
2. [ ] **Cut the CHANGELOG.** Rename `## [Unreleased]` → `## [0.1.0] — YYYY-MM-DD`, add a
       fresh empty `[Unreleased]`, and confirm every shipped `NUDGE-METHOD-*` capability is
       listed.
3. [ ] **Sync the README "Status"** to "first PyPI release" and add the
       `pip install nudge-bio` line + the extras table (`[bio]`, `[viz]`, `[mcp]`, `[dev]`).
4. [ ] **Rebuild clean:** `rm -rf dist && uv build && uv run --with twine twine check dist/*`.
5. [ ] **Dry-run on TestPyPI:** `twine upload -r testpypi dist/*`, then
       `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nudge-bio`
       in a clean venv and re-run the import + `nudge --help` smoke test. (The extra index is
       needed because maddening/jax live on real PyPI.)
6. [ ] **Tag + GitHub release:** `git tag v0.1.0 && git push --tags`; draft release notes from
       the CHANGELOG `[0.1.0]` section.
7. [ ] **Upload to PyPI:** `twine upload dist/*` (API token, `__token__`). Immediately verify
       `pip install nudge-bio` in a clean venv from real PyPI.

## Nice-to-have (not release-blocking)

- [ ] **Trim the sdist** — it is ~3.2 MB, mostly the outputs-embedded demo notebooks. If we
      want a lean sdist, add a `[tool.hatch.build.targets.sdist] exclude = ["notebooks/*.ipynb"]`
      (the wheel already only packages `src/nudge`, so this only affects the sdist). Weigh
      against wanting the notebooks shipped with the source. **Decision deferred.**
- [ ] Add a minimal quickstart / `pip install` badge block to the top of the README.
- [ ] Confirm `nudge oed`, `nudge fibrillization`, and the other newest subcommands appear in
      `nudge --help` in the installed wheel (spot-checked; do a full pass at release time).

## Notes

- The `[bio]` extra (scanpy/pertpy) is heavy and only needed for the real-data loaders; the
  core install is intentionally lean (jax + maddening + anndata + typer). Keep it that way.
- `jax==0.5.1` / `jaxlib==0.5.1` are pinned exactly (to match maddening's ABI expectations).
  A CPU wheel resolves everywhere; GPU users install a CUDA jaxlib themselves. State this in
  the README install section so nobody expects GPU out of the box.
