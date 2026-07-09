"""Synergy / epistasis attribution — fit + classify, and the Norman 2019 lock-in.

The classifier must (a) call a genuinely additive combo ``additive``; (b) call a
super-additive combo ``synergistic`` and a sub-additive one ``buffering``; (c) return
``no-effect`` when neither single arm moves the signature; and (d) **abstain**
(``unresolved``) when a condition is underpowered — an interaction inherits its weakest
single arm. The synthetic tests build per-cell scalar scores with known ground-truth
effects so the additive null and its deviation are exactly known.

The real-data test (``needs_data``) locks in Norman 2019 (GSE133344) genetic-interaction
calls: at least one paper-labelled synergistic pair vs a paper-labelled
additive/neutral pair, using the direction-safe projection extractor.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from nudge.inference.epistasis import (
    attribute_synergy,
    classify_synergy,
    fit_synergy,
)

_NORMAN_H5AD = "/media/nick/Seagate Hub/norman_2019/norman_2019.h5ad"


def _cells(mean: float, *, n: int = 400, sd: float = 1.0, seed: int = 0) -> np.ndarray:
    """A condition's per-cell scalar scores ~ N(mean, sd)."""
    return np.random.default_rng(seed).normal(mean, sd, size=n)


def test_additive_combo_reads_additive() -> None:
    """A+B lands exactly on effect(A)+effect(B): the additive null holds."""
    ctrl = _cells(0.0, seed=1)
    a = _cells(1.0, seed=2)
    b = _cells(1.2, seed=3)
    ab = _cells(2.2, seed=4)  # = 1.0 + 1.2, the additive prediction
    res = attribute_synergy(ctrl, a, b, ab, n_boot=400, seed=0)
    assert res.call == "additive", res.reason
    lo, hi = res.fit.ci_interaction
    assert lo <= 0.0 <= hi


def test_super_additive_combo_reads_synergistic() -> None:
    """A+B overshoots the additive prediction — super-additive / synergistic."""
    ctrl = _cells(0.0, seed=1)
    a = _cells(1.0, seed=2)
    b = _cells(1.0, seed=3)
    ab = _cells(3.6, seed=4)  # additive pred 2.0; +1.6 above → synergy
    res = attribute_synergy(ctrl, a, b, ab, n_boot=400, seed=0)
    assert res.call == "synergistic", res.reason
    assert res.fit.ci_interaction[0] > 0.0
    assert res.fit.bic_free < res.fit.bic_additive  # parsimony prefers the free level


def test_sub_additive_combo_reads_buffering() -> None:
    """A+B undershoots the additive prediction — sub-additive / buffering."""
    ctrl = _cells(0.0, seed=1)
    a = _cells(1.5, seed=2)
    b = _cells(1.5, seed=3)
    ab = _cells(1.6, seed=4)  # additive pred 3.0; combo barely above a single → buffer
    res = attribute_synergy(ctrl, a, b, ab, n_boot=400, seed=0)
    assert res.call == "buffering", res.reason
    assert res.fit.ci_interaction[1] < 0.0


def test_no_effect_when_neither_arm_moves() -> None:
    """Neither single arm clears noise → no-effect (nothing to attribute)."""
    ctrl = _cells(0.0, seed=1)
    a = _cells(0.0, seed=2)
    b = _cells(0.0, seed=3)
    ab = _cells(0.0, seed=4)
    res = attribute_synergy(ctrl, a, b, ab, n_boot=400, seed=0)
    assert res.call == "no-effect", res.reason


def test_underpowered_arm_abstains_unresolved() -> None:
    """A dead/underpowered arm (few cells) makes the combo unresolved, not a call."""
    ctrl = _cells(0.0, n=400, seed=1)
    a = _cells(1.0, n=400, seed=2)
    b = _cells(1.0, n=8, seed=3)  # only 8 cells — cannot trust this arm
    ab = _cells(3.5, n=400, seed=4)
    res = attribute_synergy(ctrl, a, b, ab, n_boot=400, seed=0, min_cells=30)
    assert res.call == "unresolved", res.reason
    assert "underpowered" in res.reason


def test_wide_ci_straddling_zero_abstains() -> None:
    """A very noisy combo whose interaction CI straddles 0 but is wide → unresolved.

    The two single arms are clean (their CIs clear 0), but the A+B arm is few-celled and
    very noisy, so the interaction CI straddles 0 yet is too wide to RULE OUT synergy —
    the honest call is abstain, not a confident 'additive'.
    """
    ctrl = _cells(0.0, n=500, sd=1.0, seed=1)
    a = _cells(2.0, n=500, sd=1.0, seed=2)
    b = _cells(2.0, n=500, sd=1.0, seed=3)
    ab = _cells(4.0, n=35, sd=8.0, seed=4)  # additive pred 4; huge noise widens the CI
    fit = fit_synergy(ctrl, a, b, ab, n_boot=400, seed=0)
    call, reason = classify_synergy(fit, min_cells=30, rel_width=0.3)
    assert call == "unresolved", reason
    assert "underpowered to separate" in reason


def test_empty_condition_raises() -> None:
    with pytest.raises(ValueError, match="no cells"):
        fit_synergy(_cells(0.0), _cells(1.0), _cells(1.0), np.array([]))


def test_interaction_is_control_referenced_and_signed() -> None:
    """The reported effects/interaction match the constructed ground truth."""
    ctrl = _cells(5.0, seed=1)  # nonzero control baseline must cancel out
    a = _cells(6.0, seed=2)
    b = _cells(7.0, seed=3)
    ab = _cells(9.5, seed=4)  # additive pred = (6-5)+(7-5)=3; observed 4.5 → +1.5
    fit = fit_synergy(ctrl, a, b, ab, n_boot=200, seed=0)
    assert abs(fit.effect_a - 1.0) < 0.2
    assert abs(fit.effect_b - 2.0) < 0.2
    assert abs(fit.additive_pred - 3.0) < 0.3
    assert abs(fit.interaction - 1.5) < 0.3


def _norman_call(adata, gene_a, gene_b):  # type: ignore[no-untyped-def]
    from nudge.inference.bridge import combo_effect_scores
    from nudge.inference.epistasis import attribute_synergy

    ctrl, a, b, ab = combo_effect_scores(
        adata,
        control_label="control",
        a_label=gene_a,
        b_label=gene_b,
        ab_label=f"{gene_a}+{gene_b}",
        condition_col="condition",
    )
    return attribute_synergy(ctrl, a, b, ab, n_boot=500, seed=0)


@pytest.mark.needs_data
@pytest.mark.skipif(
    not os.path.exists(_NORMAN_H5AD), reason="Norman 2019 h5ad not present"
)
def test_norman_synergy_lockin_real_data() -> None:
    """Lock in Norman 2019 (GSE133344) GI calls against the paper's taxonomy.

    Three well-characterised CRISPRa pairs, one per interaction class, on K562 cells:

    - **CBL+CNN1** and **CBL+UBASH3B** → **synergistic** (super-additive). The paper's
      flagship synergy: CBL/CNN1/UBASH3B jointly drive an emergent erythroid state far
      beyond the additive sum (Norman 2019).
    - **DUSP9+ETS2** → **buffering** (sub-additive / epistatic). The paper reports the
      DUSP9 phenotype *dominates* and antagonises ETS2, so the combination undershoots
      the additive prediction (the combo lands near DUSP9-alone).
    - **FOXA1+FOXA3** → **additive**. Paralogous factors whose combination sits on the
      additive line (interaction CI straddles 0).
    """
    import anndata as ad

    adata = ad.read_h5ad(_NORMAN_H5AD)
    adata.obs["condition"] = adata.obs["perturbation_name"].astype(str).values

    syn1 = _norman_call(adata, "CBL", "CNN1")
    assert syn1.call == "synergistic", syn1.reason
    assert syn1.fit.ci_interaction[0] > 0.0  # CI clearly above the additive null

    syn2 = _norman_call(adata, "CBL", "UBASH3B")
    assert syn2.call == "synergistic", syn2.reason

    buf = _norman_call(adata, "DUSP9", "ETS2")
    assert buf.call == "buffering", buf.reason
    assert buf.fit.ci_interaction[1] < 0.0  # CI clearly below the additive null

    add = _norman_call(adata, "FOXA1", "FOXA3")
    assert add.call == "additive", add.reason
