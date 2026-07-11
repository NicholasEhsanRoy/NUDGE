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

## Blocked-until-hardening (do NOT release before these)

- [ ] **Post-moat hardening loop green** — the newly-merged moat code (`inference/oed.py`,
      the matrix-free additions in `inference/sloppiness.py`, the identifiability additions
      in `inference/adjoint.py`) has NOT yet been through the red-team loop. Run the cloud
      hardening loop (prompt: `design/hardening/POST_MOAT_LOOP_PROMPT.md`) on a **separate
      branch** and merge only when red-team reports `HOLES_FOUND: 0` after a genuine full
      sweep.
- [ ] **Close the pre-existing open hole P5** (differential small-multiplicative confound,
      `LIM-016`) — already queued in `design/hardening/LEDGER.md`; fold it into the loop.

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
