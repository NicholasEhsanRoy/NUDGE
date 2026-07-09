"""Cross-modality readout adapter: bouncer refusals, the panel, the Chure lock-in.

Three things must hold. (a) The **modality bouncer** (`check_readout`) accepts a genuine
continuous fluorescence / fold-change signal but **refuses** the ambiguous inputs that
would silently corrupt a fit: raw integer counts, log-normalized counts (zero-inflated),
substantially-negative (scaled/centered) values, an unknown modality — and delegates
`modality="counts"` to the existing integer guard unchanged (NUDGE-LIM-008). (b) The
**panel** localizes a right-shifted-EC50 variant to `threshold` and a leaky variant
to `ceiling` on synthetic ground truth, and abstains on a non-responsive one. (c) The
`needs_data` **Chure-2019 lock-in** pins the real result: inducer mutants Q294K /
Q294V read `threshold` (a large rightward K shift), DNA-binding mutants Y20I / Q21A read
`ceiling` (a raised leakiness floor), the near-non-inducible Q294R abstains — and NUDGE
never manufactures a gain call. This is the author-labelled K-vs-ceiling ground truth.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from nudge.data.ingest import IngestError, check_readout
from nudge.inference.bridge import fluorescence_dose_response
from nudge.inference.cross_modality import (
    attribute_variant_panel,
    classify_knob_shift,
)
from nudge.inference.dose_response import attribute_dose_response, fit_dose_response
from nudge.mechanisms.regulatory import hill_activation

_CHURE_CSV = "/media/nick/Seagate Hub/chure_2019/Chure2019_summarized_data.csv"


# --------------------------------------------------------------------------- #
# (a) the modality bouncer: accept continuous, refuse ambiguous / mislabeled input
# --------------------------------------------------------------------------- #
def _fluor_curve(k: float, n: float, *, floor: float, amp: float, seed: int = 0,
                 doses: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if doses is None:
        doses = np.array([0.0, 1, 5, 10, 25, 50, 100, 250, 500, 1000], dtype=float)
    clean = floor + np.asarray(hill_activation(doses, k, n, amp))
    noisy = clean * rng.lognormal(0.0, 0.03, size=doses.shape)  # multiplicative noise
    return doses, noisy


def test_bouncer_accepts_continuous_fluorescence() -> None:
    _d, y = _fluor_curve(k=50, n=4.0, floor=0.1, amp=1.0)
    # A genuine continuous curve (non-integer, non-negative, no zero-inflation) passes.
    check_readout(y, modality="fluorescence")


def test_bouncer_refuses_raw_integer_counts() -> None:
    counts = np.array([0, 3, 7, 12, 40, 88, 150, 300], dtype=float)  # integer-valued
    with pytest.raises(IngestError, match="all-integer|COUNTS"):
        check_readout(counts, modality="fluorescence")


def test_bouncer_refuses_log_normalized_counts() -> None:
    """log1p(CPM)-style data: ~half exact zeros (dropout) + a continuous tail."""
    rng = np.random.default_rng(0)
    tail = np.log1p(rng.exponential(2.0, size=500))  # continuous, non-integer, >= 0
    zeros = np.zeros(500)  # dropout
    lognorm = np.concatenate([zeros, tail])
    with pytest.raises(IngestError, match="zeros|LOG-NORMALIZED"):
        check_readout(lognorm, modality="fluorescence")


def test_bouncer_refuses_scaled_centered_values() -> None:
    rng = np.random.default_rng(1)
    scaled = rng.normal(0.0, 1.0, size=200)  # ~half negative (centered)
    with pytest.raises(IngestError, match="negative"):
        check_readout(scaled, modality="foldchange")


def test_bouncer_refuses_unknown_modality() -> None:
    with pytest.raises(IngestError, match="unknown readout modality"):
        check_readout(np.array([0.1, 0.5, 1.2]), modality="protein-blot")


def test_bouncer_tolerates_tiny_negative_foldchange_noise() -> None:
    # Background-subtracted fold-change: a couple of tiny negatives near zero are okay.
    y = np.array([-0.01, 0.02, 0.2, 0.5, 0.8, 1.0, 1.1, 0.95])
    check_readout(y, modality="foldchange")


def test_counts_modality_delegates_to_integer_guard() -> None:
    ad = pytest.importorskip("anndata")
    import numpy as _np

    good = ad.AnnData(X=_np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    check_readout(good, modality="counts")  # integer counts pass unchanged
    bad = ad.AnnData(X=_np.array([[0.5, 1.2], [3.1, 4.0]], dtype=float))
    with pytest.raises(IngestError, match="non-integer"):
        check_readout(bad, modality="counts")


# --------------------------------------------------------------------------- #
# the extractor: a continuous curve flows into the shipped dose-response fit
# --------------------------------------------------------------------------- #
def test_fluorescence_curve_reads_switch() -> None:
    """A genuine high-gain continuous fluorescence curve classifies as a switch."""
    doses = np.concatenate([[0.0], np.logspace(0, 3, 23)])  # dense; spans inflection
    rng = np.random.default_rng(3)
    clean = 0.1 + np.asarray(hill_activation(doses, 50, 6.0, 1.0))
    y = clean * rng.lognormal(0.0, 0.02, size=doses.shape)  # multiplicative noise
    check_readout(y, modality="fluorescence")  # a genuine continuous signal passes
    res = attribute_dose_response(doses, y, direction="activate", n_boot=250, seed=0)
    assert res.call == "switch", res.reason
    assert res.fit.ci_n[0] > 2.0


def test_extractor_runs_bouncer_and_builds_curve() -> None:
    pd = pytest.importorskip("pandas")
    d, y = _fluor_curve(k=50, n=3.0, floor=0.1, amp=1.0)
    df = pd.DataFrame({"dose": d, "signal": y, "variant": "v1"})
    dose, resp = fluorescence_dose_response(
        df, dose_col="dose", response_col="signal", variant="v1",
        variant_col="variant", modality="fluorescence",
    )
    assert dose.shape == resp.shape and dose.size == d.size
    assert np.all(np.diff(dose) > 0)  # dose-sorted

    # And a log-normalized-counts column is refused at extraction time.
    rng = np.random.default_rng(0)
    lognorm = np.concatenate([np.zeros(6), np.log1p(rng.exponential(2.0, size=4))])
    df_bad = pd.DataFrame({"dose": d, "signal": lognorm, "variant": "v1"})
    with pytest.raises(IngestError):
        fluorescence_dose_response(
            df_bad, dose_col="dose", response_col="signal", variant="v1",
            variant_col="variant", modality="fluorescence",
        )


# --------------------------------------------------------------------------- #
# (b) the panel: threshold vs ceiling vs non-responsive on synthetic ground truth
# --------------------------------------------------------------------------- #
def _panel_df():
    pd = pytest.importorskip("pandas")
    doses = np.array([0.0, 1, 5, 10, 25, 50, 100, 250, 500, 1000], dtype=float)
    rows = []
    # control: K=50, floor low, full span
    specs = {
        "ctrl": dict(k=50, n=2.0, floor=0.05, amp=1.0),   # reference
        "thr": dict(k=400, n=2.0, floor=0.05, amp=1.0),   # right EC50 -> threshold
        "ceil": dict(k=50, n=2.0, floor=0.55, amp=0.5),   # raised floor -> ceiling
        "dead": dict(k=50, n=2.0, floor=0.5, amp=0.0),    # flat: no response -> dead
    }
    for i, (name, s) in enumerate(specs.items()):
        _d, y = _fluor_curve(s["k"], s["n"], floor=s["floor"], amp=s["amp"],
                             doses=doses, seed=i)
        for di, yi in zip(doses, y, strict=True):
            rows.append({"dose": di, "signal": yi, "variant": name})
    return pd.DataFrame(rows)


def test_panel_localizes_threshold_and_ceiling() -> None:
    df = _panel_df()
    panel = attribute_variant_panel(
        df, dose_col="dose", response_col="signal", variant_col="variant",
        control_variant="ctrl", modality="fluorescence", n_boot=300, seed=0,
    )
    by = {v.variant: v for v in panel}
    assert by["ctrl"].knob == "control"
    assert by["thr"].knob == "threshold", by["thr"].knob_reason
    assert by["ceil"].knob == "ceiling", by["ceil"].knob_reason
    assert by["dead"].knob == "non-responsive", by["dead"].knob_reason


def test_classify_knob_shift_abstains_when_curves_match() -> None:
    """Two near-identical curves -> no knob clears its gate -> inconclusive."""
    d, y = _fluor_curve(k=50, n=2.0, floor=0.05, amp=1.0, seed=7)
    a = fit_dose_response(d, y, direction="activate", n_boot=200, seed=0)
    d2, y2 = _fluor_curve(k=50, n=2.0, floor=0.05, amp=1.0, seed=8)
    b = fit_dose_response(d2, y2, direction="activate", n_boot=200, seed=1)
    knob, _reason = classify_knob_shift(b, a)
    assert knob == "inconclusive", knob


# --------------------------------------------------------------------------- #
# service wiring: the CLI/MCP entry point gives the same panel from a CSV
# --------------------------------------------------------------------------- #
def test_service_cross_modality_panel(tmp_path) -> None:
    from nudge.service import cross_modality_panel_file

    df = _panel_df()
    csv = tmp_path / "panel.csv"
    df.to_csv(csv, index=False)
    out = cross_modality_panel_file(
        str(csv), dose_col="dose", response_col="signal", variant_col="variant",
        control_variant="ctrl", modality="fluorescence", n_boot=200, seed=0,
    )
    assert out["modality"] == "fluorescence"
    knobs = {v["variant"]: v["knob"] for v in out["variants"]}
    assert knobs["thr"] == "threshold"
    assert knobs["ceil"] == "ceiling"


# --------------------------------------------------------------------------- #
# (c) the Chure-2019 LacI benchmark lock-in (needs the external dataset)
# --------------------------------------------------------------------------- #
@pytest.mark.needs_data
@pytest.mark.skipif(
    not os.path.exists(_CHURE_CSV), reason="Chure 2019 summarized_data.csv not present"
)
def test_chure_laci_kn_ground_truth_real_data() -> None:
    """Recover the author decomposition: inducer -> threshold(K); DNA -> ceiling.

    The robust, defensible facts (matched operator O2, copy number 260, vs WT):
    Q294K / Q294V (inducer) shift the dose-EC50 far right (threshold); Y20I / Q21A
    (DNA) raise the leakiness floor (ceiling); Q294R (near-non-inducible) abstains.
    No mutant is *mis*-called, and none manufactures a gain(n) call.
    """
    pd = pytest.importorskip("pandas")

    df = pd.read_csv(_CHURE_CSV, comment="#")
    singles = df[df["mutant"].str.count("-") == 0]
    panel = attribute_variant_panel(
        singles, dose_col="IPTGuM", response_col="mean", variant_col="mutant",
        control_variant="wt", class_col="class",
        filters={"operator": "O2", "repressors": 260.0},
        direction="activate", n_boot=400, seed=0,
    )
    by = {v.variant: v for v in panel}

    # Inducer-binding mutants -> threshold, with a large rightward EC50 shift.
    for mut in ("Q294K", "Q294V"):
        assert by[mut].class_label == "IND"
        assert by[mut].knob == "threshold", (mut, by[mut].knob_reason)
        assert by[mut].log2_k_ratio > 1.5, (mut, by[mut].log2_k_ratio)

    # DNA-binding mutants -> ceiling, with a raised leakiness floor.
    for mut in ("Y20I", "Q21A"):
        assert by[mut].class_label == "DNA"
        assert by[mut].knob == "ceiling", (mut, by[mut].knob_reason)
        assert by[mut].delta_floor > 0.2, (mut, by[mut].delta_floor)

    # The near-non-inducible inducer mutant abstains (span collapses), not a knob.
    assert by["Q294R"].knob == "non-responsive", by["Q294R"].knob_reason

    # No mutant is mis-attributed to gain, and no DNA mutant is called threshold or
    # vice versa (abstentions inconclusive/non-responsive are allowed and honest).
    for v in panel:
        if v.knob == "control":
            continue
        assert v.knob != "gain", (v.variant, v.knob_reason)
        if v.knob == "threshold":
            assert v.class_label != "DNA", (v.variant, "DNA mutant called threshold")
        if v.knob == "ceiling":
            assert v.class_label != "IND", (v.variant, "IND mutant called ceiling")
