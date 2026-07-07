# Changelog

All notable changes to NUDGE are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); the project follows
[Semantic Versioning](https://semver.org/). The `fit` / `design` public surface
is the stability contract (see `docs/architecture/verification_vs_validation.md`).

## [Unreleased]

### Added
- Project skeleton (Phase 0): `src/nudge` package, the two-layer circuit API
  surface (`Circuit`, `CircuitBuilder`, `CircuitSpec`), the `MechanismRegistry`,
  the attribution vocabulary (`MechanismClass` with abstention classes), the
  `MechanismMap` output schema, and stubs for the mechanism library, fit engine,
  data subsystem, and MCP adapter.
- Traceability inherited from `maddening.compliance` (`NUDGE-*` ID prefixes):
  `docs/known_limitations.yaml`, the Mechanism-Card template, and the CI
  validator scripts (`check_anomalies`, `check_citations`, `check_impl_mapping`).
- CI (lint + type-check + verification lane), PEP 561 (`py.typed`), and homes for
  the creative-AI-in-the-loop hooks (`scripts/ai/`, `.pre-commit-config.yaml`).

### Verification
- Placeholder `NUDGE-VER-000` benchmark proves the verification lane runs in CI.

### Known Limitations
- See `docs/known_limitations.yaml` (empty until the decoy battery lands, Phase 3).
