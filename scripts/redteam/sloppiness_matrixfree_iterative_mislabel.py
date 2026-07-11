"""Red-team repro (round 6) — the matrix-free identifiability MOAT mislabels a
structurally-UNIDENTIFIABLE model as ``well-constrained``.

Capability: ``nudge.inference.sloppiness.sloppiness_diagnostic_matrixfree`` /
``analyze_model_matrixfree`` (``NUDGE-LIM-023``) — the FIM-eigenspectrum identifiability
diagnostic that touches the Fisher matrix only through ``JᵀJ·v`` matvecs (the scaling moat
that avoids OOM-ing the dense ``jacfwd``).

The confident-wrong
--------------------
On a model with an EXACT structural redundancy (two parameters that enter the observation
map only through their sum ⇒ the ``(1, -1)`` direction is a genuine zero of the Fisher
information ⇒ those two parameters are provably non-recoverable from ANY amount of data),
the **iterative** path (and ``method="auto"`` whenever ``n_params > dense_below=256``, its
raison-d'être large-network regime) returns::

    label = 'well-constrained'
    reason = 'WELL-CONSTRAINED: ... every parameter is individually identifiable ...'
    n_null_dims = 0

while the exact dense oracle (``analyze_model`` / ``method="dense"``) correctly returns
``unidentifiable`` (``n_null_dims = 1``) and names the redundant pair.

Why the shipped gate fails
--------------------------
``sloppiness_diagnostic_matrixfree`` handles the ``n_params <= n_obs`` case (no shape-rank
certificate) by computing the smallest eigenpairs with ``eigsh(which="SA")`` and
Rayleigh-residual-verifying them (``_verified_smallest_eigsh``). ARPACK/Lanczos MISSES the
isolated exact-zero eigenvalue and converges to the well-conditioned cluster; those returned
pairs are GENUINE eigenpairs, so they PASS the Rayleigh residual check — the check verifies
*eigenpair-ness*, not *smallest-ness*. Hence ``smallest_certified=True``, ``lam_min`` is set
to a large (wrong) value, ``cond``/``span`` are understated, ``computed_null=0``, and the
verdict tree lands on ``well-constrained``. The module's own docstring names exactly this
failure ("eigsh/LOBPCG can ... mislabel a rank-deficient model well-constrained") and claims
the Rayleigh verification handles it fail-safe — it does not for this case.

Run (deterministic, ≥2 seeds through the shipped public API)::

    uv run python scripts/redteam/sloppiness_matrixfree_iterative_mislabel.py

x64 is enabled (the intended, correct usage — so this is NOT the documented float32
downcast caveat; the mislabel is a Krylov-solver gap, present at full precision).
"""

from __future__ import annotations

import jax

# The intended, correct precision (the module warns float32 loses smallest-eig resolution);
# enabling x64 rules out the documented float32 caveat — the hole is at full precision.
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from nudge.inference.sloppiness import (  # noqa: E402
    analyze_model,
    analyze_model_matrixfree,
    sloppiness_diagnostic_matrixfree,
)

SIGMA = 0.01


def structural_null_model(n_params: int, n_obs: int, seed: int):
    """A well-conditioned linear map ``y = M·θ`` with ONE exact structural redundancy.

    ``M``'s last column is set EQUAL to its first, so parameters ``p0`` and ``p{last}`` enter
    the observation only through ``p0 + p{last}`` — a genuine zero of the Fisher information
    (the ``(1, 0, …, 0, -1)`` direction). This is the classic structural non-identifiability
    the module's own ``redundant_exponential_predict`` represents (``A·e^{-(k₁+k₂)t}``),
    scaled up so ``method="auto"`` selects the iterative Krylov path.
    """
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n_obs, n_params))
    mat = mat / np.linalg.norm(mat, axis=0, keepdims=True)  # tight non-null spectrum
    mat[:, n_params - 1] = mat[:, 0]  # EXACT duplicate column → exact structural null
    mat_j = jnp.asarray(mat)

    def predict(theta):
        return mat_j @ theta

    theta0 = np.ones(n_params, dtype=np.float64)
    predict.theta0 = theta0  # type: ignore[attr-defined]
    predict.names = tuple(f"p{i}" for i in range(n_params))  # type: ignore[attr-defined]
    return predict, theta0


def _check(label_mf: str, n_null_mf: int, label_dense: str) -> bool:
    """A HOLE = the dense oracle says unidentifiable but the matrix-free path does not."""
    return label_dense == "unidentifiable" and label_mf != "unidentifiable" and n_null_mf == 0


def main() -> int:
    holes = 0

    print("=" * 78)
    print("CASE 1 — method='auto' (the DEFAULT), n_params=300 > dense_below=256")
    print("         → auto selects the iterative Krylov path (the large-network regime)")
    print("=" * 78)
    for seed in (0, 1, 2):
        fn, theta0 = structural_null_model(300, 400, seed)
        auto = analyze_model_matrixfree(fn, sigma=SIGMA)  # default method='auto'
        dense = sloppiness_diagnostic_matrixfree(fn, theta0, SIGMA, method="dense")
        hole = _check(auto.label, auto.n_null_dims, dense.label)
        holes += hole
        print(
            f" seed={seed}: matrix-free(auto)={auto.label!r} n_null={auto.n_null_dims} "
            f"cond={auto.cond_number:.2e} lam_min={auto.smallest_eigenvalue:.2e}  |  "
            f"dense(exact)={dense.label!r} n_null={dense.n_null_dims} "
            f"{'<<< CONFIDENT-WRONG' if hole else ''}"
        )
        if hole:
            print(f"          reason: {auto.reason[:96]}")

    print()
    print("=" * 78)
    print("CASE 2 — method='iterative' (explicit, documented), n_params=40")
    print("         cross-checked against analyze_model (the jacfwd-SVD dense oracle)")
    print("=" * 78)
    for seed in (0, 1, 2):
        fn, theta0 = structural_null_model(40, 100, seed)
        oracle = analyze_model(fn, sigma=SIGMA)  # jacfwd + SVD rank test (the reference)
        itr = sloppiness_diagnostic_matrixfree(fn, theta0, SIGMA, method="iterative")
        hole = _check(itr.label, itr.n_null_dims, oracle.label)
        holes += hole
        print(
            f" seed={seed}: matrix-free(iter)={itr.label!r} n_null={itr.n_null_dims} "
            f"lam_min={itr.smallest_eigenvalue:.2e}  |  "
            f"dense-oracle={oracle.label!r} n_null={oracle.n_null_dims} "
            f"{'<<< CONFIDENT-WRONG' if hole else ''}"
        )

    print()
    print("=" * 78)
    truth = "unidentifiable (params p0 and p{last} enter only via their sum — provably)"
    print(f"GROUND TRUTH: {truth}")
    print(f"HOLES (confident 'well-constrained'/'usable' on that truth): {holes} / 6")
    print("=" * 78)
    return 0 if holes >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
