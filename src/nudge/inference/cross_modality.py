"""Cross-modality readout attribution — the same K/n/v_max, read from fluorescence.

NUDGE's core attribution speaks one vocabulary — **K** (threshold), **n** (gain),
**v_max** (ceiling) — regardless of *how* the switch is measured. Its ingest, though,
hard-requires raw integer counts. This module is the **cross-modality adapter**: it runs
the *identical* dose-response attribution when the readout is a **continuous single
channel** — flow-cytometry fluorescence, a live-cell activity reporter, or a fold-change
summary — instead of UMI counts. Nothing about the inference changes; only the
observation channel does.

The core is deliberately thin — it **reuses the shipped dose-response path verbatim**
(:func:`~nudge.inference.dose_response.attribute_dose_response`): a variant's
response-vs-dose curve *is* a dose-response, so the fit, the bootstrap CIs and the
fail-safe classifier all carry over. Two new pieces make it modality-aware:

- the **modality bouncer** (:func:`nudge.data.ingest.check_readout`) refuses ambiguous /
  mislabeled input — most sharply, log-normalized or raw counts dressed up as
  fluorescence (NUDGE-LIM-008); NUDGE never *guesses* a modality;
- the **extractor** (:func:`nudge.inference.bridge.fluorescence_dose_response`) turns a
  tidy continuous-readout table into ``(dose, response)`` per variant.

On top of that, :func:`attribute_variant_panel` fits a **panel** of variants against a
shared control and localizes each one's effect to a single knob — **threshold** (a shift
of the dose EC50), **gain** (a change of Hill steepness), or **ceiling / leakiness** (a
change of the response floor / span) — or honestly abstains (**non-responsive** /
**inconclusive**). This is the Chure-2019 LacI benchmark's engine: DNA-binding-domain
mutants (which alter the DNA-binding energy → repression setpoint / leakiness) vs
inducer-binding-domain mutants (which alter the inducer parameters → the inducer-axis
threshold) are the author-labelled K-vs-ceiling ground truth this recovers.

**Honesty (load-bearing).** The knob call is *comparative* (mutant vs control) and —
like all NUDGE dose-response — inherits NUDGE-LIM-006: a nonlinear reporter can
manufacture apparent ultrasensitivity, so it holds only under an approximately-affine
readout. A single operating point cannot always separate the knobs (a raised leakiness
floor repositions the apparent EC50); the honest answer there is **inconclusive**, not a
guess.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nudge.inference.bridge import fluorescence_dose_response
from nudge.inference.dose_response import DoseResponseFit, attribute_dose_response

__all__ = [
    "VariantAttribution",
    "attribute_variant_panel",
    "classify_knob_shift",
]

#: The knob a variant's effect localizes to, relative to the control.
KNOBS = ("threshold", "gain", "ceiling", "non-responsive", "inconclusive")


@dataclass(frozen=True)
class VariantAttribution:
    """One variant's dose-response fit + its knob call relative to the control."""

    variant: str
    class_label: str | None
    call: str  # the shipped dose-response verdict (switch / graded / no-effect / …)
    reason: str
    knob: str  # threshold | gain | ceiling | non-responsive | inconclusive | control
    knob_reason: str
    k_threshold: float
    ci_k: tuple[float, float]
    n: float
    ci_n: tuple[float, float]
    amp: float
    ci_amp: tuple[float, float]
    floor: float
    ci_floor: tuple[float, float]
    r2: float
    n_points: int
    log2_k_ratio: float  # log2(K_variant / K_control); nan for the control itself
    delta_floor: float  # floor_variant − floor_control
    delta_n: float  # n_variant − n_control
    extras: dict[str, Any] = field(default_factory=dict)


def _disjoint(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if two CIs do not overlap (both finite)."""
    lo_a, hi_a = a
    lo_b, hi_b = b
    if not all(np.isfinite(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return hi_a < lo_b or hi_b < lo_a


def classify_knob_shift(
    fit: DoseResponseFit,
    control: DoseResponseFit,
    *,
    k_octaves: float = 1.0,
    floor_rise: float = 0.15,
    amp_drop_frac: float = 0.6,
    min_response_frac: float = 0.3,
    n_shift: float = 0.75,
) -> tuple[str, str]:
    """Localize a variant's effect to one knob vs the ``control`` fit — or abstain.

    The gates, in biophysically-motivated order (fail-safe first). The **sign** of the
    dose-EC50 shift is load-bearing: a *rightward* shift (a higher half-max dose) is the
    signature of a **weakened inducer response** — a threshold change on the dose axis;
    a raised **leakiness floor** (with the apparent EC50 drifting the *other* way) is
    the signature of a changed **repression setpoint** — a ceiling change. NUDGE reads
    two apart rather than collapsing both to "K moved".

    1. **non-responsive** — the variant's response span collapses (``amp`` below
       ``min_response_frac ×`` the control's): the readout barely moves with dose, so
       there is no curve to attribute (e.g. a near-non-inducible mutant).
    2. **threshold** — the dose EC50 shifts **right** by >= ``k_octaves``
       (``log2(K/K_ctrl) >= k_octaves``) with disjoint ``K`` CIs: a weakened response.
    3. **ceiling** — the response **floor rises** by >= ``floor_rise`` (leakier base)
       *or* the span drops below ``amp_drop_frac ×`` the control's: a changed setpoint /
       dynamic range.
    4. **threshold** (leftward) — the EC50 shifts **left** by >= ``k_octaves`` with
       disjoint ``K`` CIs *and* the floor did **not** rise: a sensitized dose response.
    5. **gain** — the Hill steepness changes by >= ``n_shift`` with disjoint ``n`` CIs.
    6. **inconclusive** — no knob clears its gate at this single operating point (a
       raised floor can reposition the apparent EC50, so the knobs are not always
       separable from one curve; the honest answer is to abstain, not guess).
    """
    if control.amp <= 0:
        return "inconclusive", "control has no measurable response span to compare with"
    if fit.amp < min_response_frac * control.amp:
        return "non-responsive", (
            f"response span amp={fit.amp:.3g} collapsed to <{min_response_frac:g}× the "
            f"control's ({control.amp:.3g}) — the readout barely moves with dose, so "
            "there is no curve to attribute a knob to (abstain)"
        )

    k_mut = max(fit.k_threshold, 1e-12)
    k_ctrl = max(control.k_threshold, 1e-12)
    log2_k = float(np.log2(k_mut / k_ctrl))
    k_disjoint = _disjoint(fit.ci_k, control.ci_k)
    d_floor = fit.floor - control.floor
    span_shrunk = fit.amp < amp_drop_frac * control.amp

    if k_disjoint and log2_k >= k_octaves:
        return "threshold", (
            f"dose EC50 shifts right {log2_k:+.2f} octaves "
            f"(K {control.k_threshold:.3g} -> {fit.k_threshold:.3g}, disjoint CIs) — a "
            "weakened dose response / raised threshold, the inducer-domain signature"
        )
    if d_floor >= floor_rise or span_shrunk:
        return "ceiling", (
            f"baseline leaks up Δfloor={d_floor:+.3g} "
            f"(floor {control.floor:.3g} -> {fit.floor:.3g}) and/or the span changes "
            f"(amp {control.amp:.3g} -> {fit.amp:.3g}) — a changed repression setpoint "
            "/ dynamic range (ceiling), the DNA-binding-domain signature; apparent "
            "EC50 drift is a fold-change-space consequence of the raised floor, not a "
            "threshold change"
        )
    if k_disjoint and log2_k <= -k_octaves:
        return "threshold", (
            f"dose EC50 shifts left {log2_k:+.2f} octaves "
            f"(K {control.k_threshold:.3g} -> {fit.k_threshold:.3g}, disjoint CIs) "
            "with no leakiness rise — a sensitized dose response"
        )
    if _disjoint(fit.ci_n, control.ci_n) and abs(fit.n - control.n) >= n_shift:
        return "gain", (
            f"Hill steepness changes Δn={fit.n - control.n:+.2f} "
            f"(n {control.n:.2f} -> {fit.n:.2f}, disjoint CIs) — a gain change"
        )
    return "inconclusive", (
        f"no single knob clears its gate at this operating point "
        f"(log2 K ratio {log2_k:+.2f}, Δfloor {d_floor:+.3g}, "
        f"Δn {fit.n - control.n:+.2f}); a raised floor can reposition the apparent "
        "EC50, so the knobs are not separable from one curve — abstain rather than "
        "guess (a second operating point / copy-number series would resolve it)"
    )


def _to_attr(
    variant: str,
    class_label: str | None,
    result: Any,
    knob: str,
    knob_reason: str,
    control_fit: DoseResponseFit | None,
) -> VariantAttribution:
    f = result.fit
    if control_fit is not None and control_fit.k_threshold > 0:
        log2k = float(np.log2(max(f.k_threshold, 1e-12) / control_fit.k_threshold))
        d_floor = f.floor - control_fit.floor
        d_n = f.n - control_fit.n
    else:
        log2k = float("nan")
        d_floor = float("nan")
        d_n = float("nan")
    return VariantAttribution(
        variant=variant,
        class_label=class_label,
        call=result.call,
        reason=result.reason,
        knob=knob,
        knob_reason=knob_reason,
        k_threshold=f.k_threshold,
        ci_k=f.ci_k,
        n=f.n,
        ci_n=f.ci_n,
        amp=f.amp,
        ci_amp=f.ci_amp,
        floor=f.floor,
        ci_floor=f.ci_floor,
        r2=f.r2,
        n_points=f.n_points,
        log2_k_ratio=log2k,
        delta_floor=d_floor,
        delta_n=d_n,
    )


def attribute_variant_panel(
    df: Any,
    *,
    dose_col: str,
    response_col: str,
    variant_col: str,
    control_variant: str,
    variants: list[str] | None = None,
    class_col: str | None = None,
    filters: Mapping[str, Any] | None = None,
    direction: str = "activate",
    modality: str = "fluorescence",
    autofluor: float = 0.0,
    agg: str = "mean",
    n_boot: int = 400,
    seed: int = 0,
    knob_kwargs: Mapping[str, Any] | None = None,
) -> list[VariantAttribution]:
    """Attribute a **panel** of variants' continuous dose-responses vs a shared control.

    For each variant (``variants``, or every value of ``variant_col`` in ``df`` under
    ``filters``), it extracts the ``(dose, response)`` curve
    (:func:`nudge.inference.bridge.fluorescence_dose_response` — which runs the modality
    bouncer), fits + classifies it with the shipped
    :func:`~nudge.inference.dose_response.attribute_dose_response` (``direction``
    defaults to ``"activate"`` — a readout that *rises* with dose, e.g. induction), and
    localizes its effect to one **knob** vs the control (:func:`classify_knob_shift`).
    ``class_col`` (if given) carries an author/ground-truth label through onto each
    result for the class-agreement comparison. The control variant is returned first
    with ``knob="control"``. Variants that fail to fit (too flat / non-convergent) come
    back with ``knob="non-responsive"`` + the failure reason. Direction ``"activate"``
    suits an induction curve; use ``"repress"`` when the readout falls with dose.
    """
    if variant_col not in getattr(df, "columns", []):
        raise KeyError(f"variant_col {variant_col!r} not in {list(df.columns)}")

    sub = df
    for key, val in (filters or {}).items():
        sub = sub[sub[key] == val]

    if variants is None:
        variants = sorted({str(v) for v in sub[variant_col].astype(str)})
    if control_variant not in variants:
        variants = [control_variant, *variants]

    def _class_of(name: str) -> str | None:
        if class_col is None:
            return None
        rows = sub[sub[variant_col].astype(str) == str(name)]
        return str(rows[class_col].iloc[0]) if not rows.empty else None

    def _fit_variant(name: str) -> Any | None:
        dose, resp = fluorescence_dose_response(
            df,
            dose_col=dose_col,
            response_col=response_col,
            variant=name,
            variant_col=variant_col,
            filters=filters,
            autofluor=autofluor,
            agg=agg,
            modality=modality,
        )
        finite = np.isfinite(dose) & np.isfinite(resp)
        dose, resp = dose[finite], resp[finite]
        if dose.size < 4:
            return None
        try:
            return attribute_dose_response(
                dose, resp, direction=direction, n_boot=n_boot, seed=seed
            )
        except Exception:
            return None

    ctrl_res = _fit_variant(control_variant)
    control_fit = ctrl_res.fit if ctrl_res is not None else None

    out: list[VariantAttribution] = []
    if ctrl_res is not None:
        out.append(
            _to_attr(
                control_variant,
                _class_of(control_variant),
                ctrl_res,
                "control",
                "the shared control / reference curve",
                None,
            )
        )

    for name in variants:
        if name == control_variant:
            continue
        res = _fit_variant(name)
        if res is None:
            # A curve that won't fit (too flat / non-convergent) is an abstention.
            out.append(
                VariantAttribution(
                    variant=name,
                    class_label=_class_of(name),
                    call="no-effect",
                    reason="fewer than 4 usable dose points or the Hill fit did not "
                    "converge — the response is too flat / degenerate to fit",
                    knob="non-responsive",
                    knob_reason="the dose-response fit did not converge (a near-flat / "
                    "non-inducible curve) — NUDGE abstains rather than force a knob",
                    k_threshold=float("nan"),
                    ci_k=(float("nan"), float("nan")),
                    n=float("nan"),
                    ci_n=(float("nan"), float("nan")),
                    amp=float("nan"),
                    ci_amp=(float("nan"), float("nan")),
                    floor=float("nan"),
                    ci_floor=(float("nan"), float("nan")),
                    r2=float("nan"),
                    n_points=0,
                    log2_k_ratio=float("nan"),
                    delta_floor=float("nan"),
                    delta_n=float("nan"),
                )
            )
            continue
        if control_fit is not None:
            knob, knob_reason = classify_knob_shift(
                res.fit, control_fit, **(knob_kwargs or {})
            )
        else:
            knob, knob_reason = "inconclusive", "no control fit to compare against"
        out.append(_to_attr(name, _class_of(name), res, knob, knob_reason, control_fit))

    return out
