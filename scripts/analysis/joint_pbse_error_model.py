#!/usr/bin/env python3
"""Joint deterministic error geometry for the PbSe/Cr common-mode response.

The public PbSe/Cr transport curves report method ceilings but no replicate-level
covariance.  This analysis therefore does *not* assign probabilities and does
not report confidence intervals.  It constructs several positive-semidefinite
correlation shapes for a deterministic ellipsoid and asks a narrower question:

    does an admissible, simultaneously perturbed S/sigma/kappa case reverse
    the sign of the reoptimized common-mode cold-port response?

The error coordinates are physically separated into

1. an equal additive, temperature-dependent Seebeck drift shared by p and n.
   This coordinate represents a common instrument/calibration-map systematic,
   rather than a change of carrier concentration, and is therefore independent
   of sigma and kappa in every scenario; and
2. branch-specific Seebeck-knot residuals together with whole-curve sigma and
   kappa scale errors.  Their correlations are varied through explicit factor
   models, including same-run scale coupling and band-transport coupling.

All Seebeck candidates are intersected with the continuous 5% method ceiling,
must retain the source-curve signs and monotonicity, and are compared with their
own alpha-preserving flattened-common-mode counterfactual.  The case library
is deterministic but not a global optimization over every function compatible
with the public figures.  A negative case proves sign non-identifiability for
that stated geometry; failure to find one does not prove global identifiability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import platform
import sys
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import analyze_pbse_common_mode_contribution as common  # noqa: E402
from scripts.analysis import analyze_pbse_device_forward_constraint as forward  # noqa: E402
from scripts.analysis import analyze_pbse_gamma_identifiability as ident  # noqa: E402
from scripts.tec_1d_solver import (  # noqa: E402
    PchipTemperatureProperty,
    TemperatureDependentNumericalCouple,
)


SCHEMA_VERSION = "joint_pbse_error_model/v1"
ANALYSIS_ID = "SCI-JOINT-PBSE-ERROR-MODEL-20260826"
DEFAULT_JSON = (
    ROOT / "results/scientific_analysis/joint_pbse_error_model_results.json"
)
DEFAULT_FIGURE_PREFIX = (
    ROOT / "results/scientific_analysis/joint_pbse_error_model_figure"
)
DEFAULT_SOURCE_CSV = (
    ROOT / "results/scientific_analysis/joint_pbse_error_model_source_data.csv"
)

SEEBECK_RELATIVE_CEILING = 0.05
SIGMA_RELATIVE_CEILING = 0.05
KAPPA_RELATIVE_CEILING = 0.15
GRADIENT_STEPS = (0.01, 0.005)
SHARED_FRACTIONS = (0.0, 0.5, 1.0)
COMMON_DRIFT_SWEEP = (0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
DENSE_PROPERTY_POINTS = 2001
ELLIPSOID_TOLERANCE = 2.0e-10
PROPERTY_TOLERANCE = 2.0e-12

PARAMETER_NAMES = (
    "shared_additive_Seebeck_drift",
    "p_Seebeck_323K_residual",
    "n_Seebeck_323K_residual",
    "p_Seebeck_373K_residual",
    "n_Seebeck_373K_residual",
    "p_sigma_scale_error",
    "n_sigma_scale_error",
    "p_kappa_scale_error",
    "n_kappa_scale_error",
)

FloatArray = NDArray[np.float64]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_locator(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class ModelContext:
    inputs: dict[str, Any]
    validation: dict[str, Any]
    base: TemperatureDependentNumericalCouple
    p_temperature_k: FloatArray
    p_seebeck_v_per_k: FloatArray
    n_temperature_k: FloatArray
    n_seebeck_v_per_k: FloatArray
    normalization_center_k: float
    normalization_half_range_k: float
    common_additive_bound_v_per_k: float
    nominal_current_a: float


def build_context() -> ModelContext:
    inputs, validation = ident._validate_and_load_inputs()
    target = inputs["target"]
    hot = float(target["hot_side_temperature_k"])
    cold = hot - float(target["delta_t_max_k"])
    base = forward.build_couple(inputs, forward.SCENARIOS["nominal"], cold, hot)
    nominal_capacity = common._optimized_capacity(base)
    nominal_current = float(
        nominal_capacity["contact_corrected_optimum"]["current_a"]
    )

    p_temperature, p_values = forward._fig1_arrays(inputs["p_seebeck"])
    n_temperature, n_values = forward._fig1_arrays(inputs["n_seebeck"])
    center, half_range = ident._full_support_normalization(
        p_temperature, n_temperature
    )
    p_grid = np.linspace(
        base.p_leg.seebeck.minimum_temperature_k,
        base.p_leg.seebeck.maximum_temperature_k,
        DENSE_PROPERTY_POINTS,
    )
    n_grid = np.linspace(
        base.n_leg.seebeck.minimum_temperature_k,
        base.n_leg.seebeck.maximum_temperature_k,
        DENSE_PROPERTY_POINTS,
    )
    minimum_magnitude = min(
        float(np.min(np.abs(base.p_leg.seebeck.evaluate(p_grid)))),
        float(np.min(np.abs(base.n_leg.seebeck.evaluate(n_grid)))),
    )
    common_bound = SEEBECK_RELATIVE_CEILING * minimum_magnitude
    return ModelContext(
        inputs=inputs,
        validation=validation,
        base=base,
        p_temperature_k=p_temperature,
        p_seebeck_v_per_k=p_values,
        n_temperature_k=n_temperature,
        n_seebeck_v_per_k=n_values,
        normalization_center_k=center,
        normalization_half_range_k=half_range,
        common_additive_bound_v_per_k=common_bound,
        nominal_current_a=nominal_current,
    )


def build_perturbed_couple(
    context: ModelContext,
    normalized_coordinates: Iterable[float],
) -> TemperatureDependentNumericalCouple:
    coordinate = np.asarray(tuple(normalized_coordinates), dtype=float)
    if coordinate.shape != (len(PARAMETER_NAMES),):
        raise ValueError("joint error coordinate must contain nine entries")
    if not np.all(np.isfinite(coordinate)):
        raise ValueError("joint error coordinate must be finite")

    p_values = context.p_seebeck_v_per_k.copy()
    n_values = context.n_seebeck_v_per_k.copy()
    p_values[1] *= 1.0 + SEEBECK_RELATIVE_CEILING * coordinate[1]
    n_values[1] *= 1.0 + SEEBECK_RELATIVE_CEILING * coordinate[2]
    p_values[2] *= 1.0 + SEEBECK_RELATIVE_CEILING * coordinate[3]
    n_values[2] *= 1.0 + SEEBECK_RELATIVE_CEILING * coordinate[4]

    p_seebeck = PchipTemperatureProperty(context.p_temperature_k, p_values)
    n_seebeck = PchipTemperatureProperty(context.n_temperature_k, n_values)
    additive_drift = (
        context.common_additive_bound_v_per_k * float(coordinate[0])
    )
    p_seebeck = ident.AffinePerturbedSeebeckProperty(
        p_seebeck,
        normalization_center_k=context.normalization_center_k,
        normalization_half_range_k=context.normalization_half_range_k,
        additive_drift_v_per_k=additive_drift,
    )
    n_seebeck = ident.AffinePerturbedSeebeckProperty(
        n_seebeck,
        normalization_center_k=context.normalization_center_k,
        normalization_half_range_k=context.normalization_half_range_k,
        additive_drift_v_per_k=additive_drift,
    )
    seebeck_couple = ident._build_couple_with_properties(
        context.base, p_seebeck, n_seebeck
    )
    return ident._build_transport_scaled_couple(
        seebeck_couple,
        p_sigma_scale=1.0 + SIGMA_RELATIVE_CEILING * float(coordinate[5]),
        n_sigma_scale=1.0 + SIGMA_RELATIVE_CEILING * float(coordinate[6]),
        p_kappa_scale=1.0 + KAPPA_RELATIVE_CEILING * float(coordinate[7]),
        n_kappa_scale=1.0 + KAPPA_RELATIVE_CEILING * float(coordinate[8]),
    )


def property_admissibility(
    context: ModelContext,
    normalized_coordinates: Iterable[float],
) -> dict[str, Any]:
    coordinate = np.asarray(tuple(normalized_coordinates), dtype=float)
    couple = build_perturbed_couple(context, coordinate)
    p_temperature = np.linspace(
        context.base.p_leg.seebeck.minimum_temperature_k,
        context.base.p_leg.seebeck.maximum_temperature_k,
        DENSE_PROPERTY_POINTS,
    )
    n_temperature = np.linspace(
        context.base.n_leg.seebeck.minimum_temperature_k,
        context.base.n_leg.seebeck.maximum_temperature_k,
        DENSE_PROPERTY_POINTS,
    )
    p_value = couple.p_leg.seebeck.evaluate(p_temperature)
    n_value = couple.n_leg.seebeck.evaluate(n_temperature)
    p_baseline = context.base.p_leg.seebeck.evaluate(p_temperature)
    n_baseline = context.base.n_leg.seebeck.evaluate(n_temperature)
    relative_p = np.abs(p_value - p_baseline) / np.abs(p_baseline)
    relative_n = np.abs(n_value - n_baseline) / np.abs(n_baseline)
    p_sigma_scale = 1.0 + SIGMA_RELATIVE_CEILING * float(coordinate[5])
    n_sigma_scale = 1.0 + SIGMA_RELATIVE_CEILING * float(coordinate[6])
    p_kappa_scale = 1.0 + KAPPA_RELATIVE_CEILING * float(coordinate[7])
    n_kappa_scale = 1.0 + KAPPA_RELATIVE_CEILING * float(coordinate[8])
    maximum_relative = float(max(np.max(relative_p), np.max(relative_n)))
    p_monotone = bool(np.all(np.diff(p_value) > 0.0))
    n_monotone = bool(np.all(np.diff(n_value) < 0.0))
    p_sign = bool(np.all(p_value > 0.0))
    n_sign = bool(np.all(n_value < 0.0))
    transport_inside = bool(
        abs(float(coordinate[5])) <= 1.0 + PROPERTY_TOLERANCE
        and abs(float(coordinate[6])) <= 1.0 + PROPERTY_TOLERANCE
        and abs(float(coordinate[7])) <= 1.0 + PROPERTY_TOLERANCE
        and abs(float(coordinate[8])) <= 1.0 + PROPERTY_TOLERANCE
    )
    admissible = bool(
        maximum_relative <= SEEBECK_RELATIVE_CEILING + PROPERTY_TOLERANCE
        and p_monotone
        and n_monotone
        and p_sign
        and n_sign
        and transport_inside
    )
    return {
        "maximum_relative_p_Seebeck_deviation": float(np.max(relative_p)),
        "maximum_relative_n_Seebeck_deviation": float(np.max(relative_n)),
        "maximum_relative_Seebeck_deviation": maximum_relative,
        "p_Seebeck_monotone_increasing": p_monotone,
        "n_Seebeck_monotone_decreasing": n_monotone,
        "p_Seebeck_sign_preserved": p_sign,
        "n_Seebeck_sign_preserved": n_sign,
        "p_sigma_scale": p_sigma_scale,
        "n_sigma_scale": n_sigma_scale,
        "p_kappa_scale": p_kappa_scale,
        "n_kappa_scale": n_kappa_scale,
        "transport_coordinates_inside_method_ceilings": transport_inside,
        "admissible": admissible,
    }


class ResponseEvaluator:
    """Cache full reoptimized source-minus-flat comparisons."""

    def __init__(self, context: ModelContext) -> None:
        self.context = context
        self.cache: dict[tuple[float, ...], dict[str, Any]] = {}

    @staticmethod
    def _key(coordinate: FloatArray) -> tuple[float, ...]:
        return tuple(float(value) for value in np.round(coordinate, 12))

    def evaluate(self, normalized_coordinates: Iterable[float]) -> dict[str, Any]:
        coordinate = np.asarray(tuple(normalized_coordinates), dtype=float)
        key = self._key(coordinate)
        if key in self.cache:
            return self.cache[key]
        admissibility = property_admissibility(self.context, coordinate)
        if not admissibility["admissible"]:
            raise ValueError(f"inadmissible joint error coordinate: {coordinate}")
        couple = build_perturbed_couple(self.context, coordinate)
        cold = couple.cold_temperature_k
        m_cold = 0.5 * (
            float(couple.p_leg.seebeck.evaluate([cold])[0])
            + float(couple.n_leg.seebeck.evaluate([cold])[0])
        )
        flattened = common.build_flattened_common_mode_couple(couple, m_cold)
        original = common._optimized_capacity(couple)["contact_corrected_optimum"]
        flat = common._optimized_capacity(flattened)["contact_corrected_optimum"]
        effect = float(original["Qc_after_contact_w"] - flat["Qc_after_contact_w"])
        record = {
            "normalized_coordinates": {
                name: float(value)
                for name, value in zip(PARAMETER_NAMES, coordinate)
            },
            "reoptimized_response": {
                "original_Qc_max_w": float(original["Qc_after_contact_w"]),
                "flattened_Qc_max_w": float(flat["Qc_after_contact_w"]),
                "original_minus_flattened_Qc_max_w": effect,
                "original_optimum_current_a": float(original["current_a"]),
                "flattened_optimum_current_a": float(flat["current_a"]),
                "original_energy_residual_w": float(original["energy_residual_w"]),
                "flattened_energy_residual_w": float(flat["energy_residual_w"]),
            },
            "property_admissibility": admissibility,
        }
        self.cache[key] = record
        return record


def _factor_shape(
    *,
    same_temperature_pn_loading: float = 0.0,
    same_branch_cross_temperature_loading: float = 0.0,
    seebeck_transport_loading: float = 0.0,
    sigma_transport_loading: float = 0.0,
    kappa_transport_loading: float = 0.0,
    common_sigma_loading: float = 0.0,
    common_kappa_loading: float = 0.0,
) -> FloatArray:
    """Return an 8x8 PSD correlation matrix from explicit latent loadings."""

    loadings = np.zeros((8, 16), dtype=float)
    # Shared latent factors: G323, G373, Bp, Bn, Hp, Hn, Gsigma, Gkappa.
    seebeck_rows = ((0, 2, 4), (0, 3, 5), (1, 2, 4), (1, 3, 5))
    for row, (same_temperature, branch, transport) in enumerate(seebeck_rows):
        loadings[row, same_temperature] = same_temperature_pn_loading
        loadings[row, branch] = same_branch_cross_temperature_loading
        # Increasing carrier concentration commonly decreases |S| while
        # increasing sigma; the minus sign encodes this scenario direction.
        loadings[row, transport] = -seebeck_transport_loading
    loadings[4, 4] = sigma_transport_loading
    loadings[4, 6] = common_sigma_loading
    loadings[5, 5] = sigma_transport_loading
    loadings[5, 6] = common_sigma_loading
    loadings[6, 4] = kappa_transport_loading
    loadings[6, 7] = common_kappa_loading
    loadings[7, 5] = kappa_transport_loading
    loadings[7, 7] = common_kappa_loading
    for row in range(8):
        used = float(loadings[row] @ loadings[row])
        if used > 1.0 + 1.0e-12:
            raise ValueError("factor loadings exceed unit diagonal")
        loadings[row, 8 + row] = math.sqrt(max(0.0, 1.0 - used))
    correlation = loadings @ loadings.T
    np.testing.assert_allclose(np.diag(correlation), 1.0, atol=1.0e-12)
    if float(np.min(np.linalg.eigvalsh(correlation))) < -1.0e-12:
        raise RuntimeError("correlation shape is not positive semidefinite")
    return correlation


def correlation_scenarios() -> list[dict[str, Any]]:
    root = math.sqrt
    specifications = [
        {
            "scenario_id": "independent_residual_axes",
            "short_label": "independent",
            "description": (
                "Branch Seebeck residuals and transport-scale errors are orthogonal; "
                "the shared additive drift remains a separate instrument axis."
            ),
            "loadings": {},
        },
        {
            "scenario_id": "same_run_scale_dominated",
            "short_label": "same-run",
            "description": (
                "Same-temperature p/n relative Seebeck residuals are dominated by a "
                "common DeltaT scale factor; S is otherwise independent of transport."
            ),
            "loadings": {
                "same_temperature_pn_loading": root(0.85),
                "same_branch_cross_temperature_loading": root(0.05),
                "sigma_transport_loading": root(0.25),
                "kappa_transport_loading": root(0.25),
                "common_sigma_loading": root(0.20),
                "common_kappa_loading": root(0.20),
            },
        },
        {
            "scenario_id": "moderate_band_transport_coupling",
            "short_label": "moderate band",
            "description": (
                "Moderate same-run p/n coupling is combined with same-branch "
                "S-sigma anticorrelation and sigma-kappa positive correlation."
            ),
            "loadings": {
                "same_temperature_pn_loading": root(0.60),
                "same_branch_cross_temperature_loading": root(0.05),
                "seebeck_transport_loading": root(0.20),
                "sigma_transport_loading": root(0.60),
                "kappa_transport_loading": root(0.40),
                "common_sigma_loading": root(0.15),
                "common_kappa_loading": root(0.15),
            },
        },
        {
            "scenario_id": "strong_band_locked_coupling",
            "short_label": "strong lock",
            "description": (
                "A stress-test geometry with strong p/n same-temperature locking and "
                "strong same-branch electronic/thermal transport coupling."
            ),
            "loadings": {
                "same_temperature_pn_loading": root(0.80),
                "seebeck_transport_loading": root(0.15),
                "sigma_transport_loading": root(0.75),
                "kappa_transport_loading": root(0.60),
                "common_sigma_loading": root(0.10),
                "common_kappa_loading": root(0.10),
            },
        },
    ]
    output: list[dict[str, Any]] = []
    for specification in specifications:
        correlation = _factor_shape(**specification["loadings"])
        full = np.eye(len(PARAMETER_NAMES), dtype=float)
        full[1:, 1:] = correlation
        named = {
            "p_n_Seebeck_323K": float(full[1, 2]),
            "p_n_Seebeck_373K": float(full[3, 4]),
            "p_Seebeck_323K_sigma_p": float(full[1, 5]),
            "n_Seebeck_323K_sigma_n": float(full[2, 6]),
            "p_Seebeck_373K_sigma_p": float(full[3, 5]),
            "n_Seebeck_373K_sigma_n": float(full[4, 6]),
            "sigma_p_kappa_p": float(full[5, 7]),
            "sigma_n_kappa_n": float(full[6, 8]),
            "shared_drift_maximum_absolute_cross_correlation": float(
                np.max(np.abs(full[0, 1:]))
            ),
        }
        output.append(
            {
                **specification,
                "correlation_matrix": full,
                "named_correlations": named,
                "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(full))),
            }
        )
    return output


def finite_difference_gradient(
    evaluator: ResponseEvaluator,
    step: float,
) -> FloatArray:
    gradient = np.zeros(len(PARAMETER_NAMES), dtype=float)
    for index in range(len(PARAMETER_NAMES)):
        plus = np.zeros(len(PARAMETER_NAMES), dtype=float)
        minus = np.zeros(len(PARAMETER_NAMES), dtype=float)
        plus[index] = step
        minus[index] = -step
        plus_effect = evaluator.evaluate(plus)["reoptimized_response"][
            "original_minus_flattened_Qc_max_w"
        ]
        minus_effect = evaluator.evaluate(minus)["reoptimized_response"][
            "original_minus_flattened_Qc_max_w"
        ]
        gradient[index] = (float(plus_effect) - float(minus_effect)) / (2.0 * step)
    return gradient


def ellipsoid_shape(
    correlation: FloatArray,
    shared_fraction: float,
) -> tuple[FloatArray, float]:
    if not 0.0 <= shared_fraction <= 1.0:
        raise ValueError("shared fraction must lie in [0,1]")
    branch_fraction = math.sqrt(max(0.0, 1.0 - shared_fraction**2))
    diagonal = np.asarray(
        [shared_fraction, *([branch_fraction] * 4), 1.0, 1.0, 1.0, 1.0],
        dtype=float,
    )
    scaling = np.diag(diagonal)
    return scaling @ correlation @ scaling, branch_fraction


def ellipsoid_norm_squared(coordinate: FloatArray, shape: FloatArray) -> float:
    return float(coordinate @ np.linalg.pinv(shape, rcond=1.0e-12) @ coordinate)


def _maximum_admissible_scale(
    context: ModelContext,
    direction: FloatArray,
) -> float:
    if property_admissibility(context, direction)["admissible"]:
        return 1.0
    lower = 0.0
    upper = 1.0
    for _ in range(55):
        middle = 0.5 * (lower + upper)
        if property_admissibility(context, middle * direction)["admissible"]:
            lower = middle
        else:
            upper = middle
    return lower


def case_directions(shape: FloatArray, gradient: FloatArray) -> list[dict[str, Any]]:
    directions: list[dict[str, Any]] = []
    support_squared = float(gradient @ shape @ gradient)
    if support_squared > 1.0e-30:
        gradient_direction = (shape @ gradient) / math.sqrt(support_squared)
        for polarity, sign in (("minimum", -1.0), ("maximum", 1.0)):
            directions.append(
                {
                    "case_id": f"local_gradient_extremizer__{polarity}",
                    "objective_polarity": polarity,
                    "construction": (
                        f"{polarity} support direction of the local linear response"
                    ),
                    "coordinate": sign * gradient_direction,
                }
            )
    for index, parameter in enumerate(PARAMETER_NAMES):
        diagonal = float(shape[index, index])
        if diagonal <= 1.0e-15:
            continue
        basis = np.zeros(len(PARAMETER_NAMES), dtype=float)
        basis[index] = 1.0
        sign = 1.0 if gradient[index] >= 0.0 else -1.0
        coordinate_direction = sign * (shape @ basis) / math.sqrt(diagonal)
        for polarity, multiplier in (("minimum", -1.0), ("maximum", 1.0)):
            directions.append(
                {
                    "case_id": f"coordinate_extremizer__{parameter}__{polarity}",
                    "objective_polarity": polarity,
                    "construction": (
                        f"ellipsoid direction that gives the local {polarity} excursion "
                        f"along the {parameter} coordinate extremizer"
                    ),
                    "coordinate": multiplier * coordinate_direction,
                }
            )
    return directions


def evaluate_case_library(
    context: ModelContext,
    evaluator: ResponseEvaluator,
    shape: FloatArray,
    gradient: FloatArray,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for candidate in case_directions(shape, gradient):
        direction = np.asarray(candidate["coordinate"], dtype=float)
        scale = _maximum_admissible_scale(context, direction)
        coordinate = scale * direction
        evaluation = evaluator.evaluate(coordinate)
        record = {
            "case_id": candidate["case_id"],
            "objective_polarity": candidate["objective_polarity"],
            "construction": candidate["construction"],
            "boundary_scale_after_continuous_property_check": scale,
            "ellipsoid_norm_squared": ellipsoid_norm_squared(coordinate, shape),
            **evaluation,
        }
        if record["ellipsoid_norm_squared"] > 1.0 + ELLIPSOID_TOLERANCE:
            raise RuntimeError("constructed case lies outside its ellipsoid")
        records.append(record)
    records.sort(
        key=lambda item: item["reoptimized_response"][
            "original_minus_flattened_Qc_max_w"
        ]
    )
    minimum = records[0]
    maximum = records[-1]
    return {
        "deterministic_direction_count": len(records),
        "global_nonlinear_minimum_certified": False,
        "negative_case_found": bool(
            minimum["reoptimized_response"][
                "original_minus_flattened_Qc_max_w"
            ]
            < 0.0
        ),
        "minimum_case": minimum,
        "maximum_case": maximum,
        "evaluated_response_interval_w": [
            float(
                minimum["reoptimized_response"][
                    "original_minus_flattened_Qc_max_w"
                ]
            ),
            float(
                maximum["reoptimized_response"][
                    "original_minus_flattened_Qc_max_w"
                ]
            ),
        ],
        "evaluated_interval_contains_zero": bool(
            minimum["reoptimized_response"][
                "original_minus_flattened_Qc_max_w"
            ]
            <= 0.0
            <= maximum["reoptimized_response"][
                "original_minus_flattened_Qc_max_w"
            ]
        ),
        "records": records,
    }


def _shared_drift_axis_analysis(
    context: ModelContext,
    evaluator: ResponseEvaluator,
) -> dict[str, Any]:
    def record(amplitude: float) -> dict[str, Any]:
        coordinate = np.zeros(len(PARAMETER_NAMES), dtype=float)
        coordinate[0] = -float(amplitude)
        evaluation = evaluator.evaluate(coordinate)
        return {
            "fraction_of_full_negative_drift_axis": float(amplitude),
            "endpoint_additive_drift_v_per_k": float(
                -amplitude * context.common_additive_bound_v_per_k
            ),
            "reoptimized_response_w": float(
                evaluation["reoptimized_response"][
                    "original_minus_flattened_Qc_max_w"
                ]
            ),
            "maximum_relative_Seebeck_deviation": evaluation[
                "property_admissibility"
            ]["maximum_relative_Seebeck_deviation"],
        }

    sweep = [record(amplitude) for amplitude in COMMON_DRIFT_SWEEP]

    def response(amplitude: float) -> float:
        return record(amplitude)["reoptimized_response_w"]

    if response(0.0) * response(1.0) >= 0.0:
        raise RuntimeError("full common-drift axis did not bracket a sign reversal")
    root = float(brentq(response, 0.0, 1.0, xtol=2.0e-7, rtol=2.0e-7))
    root_record = record(root)
    return {
        "physical_interpretation": (
            "Equal additive temperature drift is treated as a shared calibration-map "
            "systematic, not a carrier-concentration perturbation; its correlations "
            "with sigma and kappa are exactly zero."
        ),
        "full_axis_endpoint_amplitude_v_per_k": float(
            context.common_additive_bound_v_per_k
        ),
        "sweep": sweep,
        "zero_crossing_fraction_of_full_negative_axis": root,
        "zero_crossing_endpoint_additive_drift_v_per_k": float(
            -root * context.common_additive_bound_v_per_k
        ),
        "zero_crossing_record": root_record,
        "full_negative_axis_is_sign_reversing_case": bool(response(1.0) < 0.0),
    }


def _single_datum_sign_analysis(
    context: ModelContext,
    gradient: FloatArray,
) -> dict[str, Any]:
    rows = []
    index_map = {
        ("p", 323): 1,
        ("n", 323): 2,
        ("p", 373): 3,
        ("n", 373): 4,
    }
    value_map = {
        ("p", 323): float(context.p_seebeck_v_per_k[1]),
        ("n", 323): float(context.n_seebeck_v_per_k[1]),
        ("p", 373): float(context.p_seebeck_v_per_k[2]),
        ("n", 373): float(context.n_seebeck_v_per_k[2]),
    }
    for temperature in (323, 373):
        for carrier in ("p", "n"):
            index = index_map[(carrier, temperature)]
            source_value = value_map[(carrier, temperature)]
            derivative_per_relative_fraction = float(
                gradient[index] / SEEBECK_RELATIVE_CEILING
            )
            derivative_per_additive = derivative_per_relative_fraction / source_value
            rows.append(
                {
                    "carrier": carrier,
                    "nominal_temperature_label_k": temperature,
                    "source_knot_temperature_k": float(
                        context.p_temperature_k[1 if temperature == 323 else 2]
                        if carrier == "p"
                        else context.n_temperature_k[1 if temperature == 323 else 2]
                    ),
                    "source_Seebeck_v_per_k": source_value,
                    "reoptimized_response_derivative_w_per_unit_relative_error": derivative_per_relative_fraction,
                    "response_change_mw_per_positive_1_uV_per_K_additive_error": float(
                        1.0e-3 * derivative_per_additive
                    ),
                }
            )
    by_temperature: list[dict[str, Any]] = []
    for temperature in (323, 373):
        selected = [
            row for row in rows if row["nominal_temperature_label_k"] == temperature
        ]
        p_row = next(row for row in selected if row["carrier"] == "p")
        n_row = next(row for row in selected if row["carrier"] == "n")
        p_relative_sign = math.copysign(
            1.0,
            p_row["reoptimized_response_derivative_w_per_unit_relative_error"],
        )
        n_relative_sign = math.copysign(
            1.0,
            n_row["reoptimized_response_derivative_w_per_unit_relative_error"],
        )
        p_additive = p_row[
            "response_change_mw_per_positive_1_uV_per_K_additive_error"
        ]
        n_additive = n_row[
            "response_change_mw_per_positive_1_uV_per_K_additive_error"
        ]
        by_temperature.append(
            {
                "nominal_temperature_label_k": temperature,
                "relative_sensitivity_signs_are_opposite": bool(
                    p_relative_sign * n_relative_sign < 0.0
                ),
                "common_additive_contributions_have_same_sign": bool(
                    p_additive * n_additive > 0.0
                ),
                "p_contribution_mw_per_positive_1_uV_per_K": p_additive,
                "n_contribution_mw_per_positive_1_uV_per_K": n_additive,
                "p_plus_n_contribution_mw_per_positive_1_uV_per_K": float(
                    p_additive + n_additive
                ),
            }
        )
    return {
        "coordinate_definition": (
            "The published single-datum derivatives use multiplicative relative "
            "knot errors.  A common additive deltaS produces deltaS/Sp and deltaS/Sn "
            "with opposite signs because Sp>0 and Sn<0."
        ),
        "branch_records": rows,
        "temperature_sums": by_temperature,
        "same_temperature_branch_cancellation": False,
        "cross_temperature_partial_cancellation": bool(
            by_temperature[0][
                "p_plus_n_contribution_mw_per_positive_1_uV_per_K"
            ]
            * by_temperature[1][
                "p_plus_n_contribution_mw_per_positive_1_uV_per_K"
            ]
            < 0.0
        ),
    }


def analyze_joint_model() -> tuple[dict[str, Any], ResponseEvaluator]:
    started = time.time()
    context = build_context()
    evaluator = ResponseEvaluator(context)
    origin = np.zeros(len(PARAMETER_NAMES), dtype=float)
    nominal = evaluator.evaluate(origin)
    nominal_effect = float(
        nominal["reoptimized_response"]["original_minus_flattened_Qc_max_w"]
    )
    if not math.isclose(
        nominal_effect, ident.EXPECTED_NOMINAL_EFFECT_W, rel_tol=0.0, abs_tol=5.0e-13
    ):
        raise RuntimeError("nominal PbSe common-mode response was not reproduced")

    gradients = {
        str(step): finite_difference_gradient(evaluator, step)
        for step in GRADIENT_STEPS
    }
    gradient = gradients[str(GRADIENT_STEPS[-1])]
    gradient_difference = np.abs(
        gradients[str(GRADIENT_STEPS[0])] - gradients[str(GRADIENT_STEPS[-1])]
    )
    shared_axis = _shared_drift_axis_analysis(context, evaluator)
    single_datum = _single_datum_sign_analysis(context, gradient)

    scenario_results: list[dict[str, Any]] = []
    for scenario in correlation_scenarios():
        correlation = np.asarray(scenario["correlation_matrix"], dtype=float)
        allocations = []
        for shared_fraction in SHARED_FRACTIONS:
            shape, branch_fraction = ellipsoid_shape(correlation, shared_fraction)
            linear_half_width = math.sqrt(float(gradient @ shape @ gradient))
            case_library = evaluate_case_library(
                context, evaluator, shape, gradient
            )
            minimum_effect = float(
                case_library["minimum_case"]["reoptimized_response"][
                    "original_minus_flattened_Qc_max_w"
                ]
            )
            maximum_effect = float(
                case_library["maximum_case"]["reoptimized_response"][
                    "original_minus_flattened_Qc_max_w"
                ]
            )
            allocations.append(
                {
                    "shared_additive_drift_fraction": float(shared_fraction),
                    "branch_specific_Seebeck_fraction": float(branch_fraction),
                    "quadratic_Seebeck_budget_partition": True,
                    "ellipsoid_shape_matrix": shape.tolist(),
                    "local_linear_response_interval_w": [
                        float(nominal_effect - linear_half_width),
                        float(nominal_effect + linear_half_width),
                    ],
                    "local_linear_interval_contains_zero": bool(
                        nominal_effect - linear_half_width <= 0.0
                    ),
                    "minimum_evaluated_nonlinear_case_w": minimum_effect,
                    "maximum_evaluated_nonlinear_case_w": maximum_effect,
                    "evaluated_nonlinear_case_interval_w": [
                        minimum_effect,
                        maximum_effect,
                    ],
                    "evaluated_nonlinear_case_interval_contains_zero": bool(
                        minimum_effect <= 0.0 <= maximum_effect
                    ),
                    "constructive_sign_reversal_found": bool(minimum_effect < 0.0),
                    "positive_sign_survives_tested_library": bool(
                        minimum_effect > 0.0
                    ),
                    "positive_sign_globally_certified": False,
                    "case_library": case_library,
                }
            )
        scenario_results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "short_label": scenario["short_label"],
                "description": scenario["description"],
                "factor_loadings": scenario["loadings"],
                "correlation_shape_matrix": correlation.tolist(),
                "named_correlations": scenario["named_correlations"],
                "minimum_correlation_matrix_eigenvalue": scenario[
                    "minimum_eigenvalue"
                ],
                "allocations": allocations,
            }
        )

    all_cases = [
        case
        for scenario in scenario_results
        for allocation in scenario["allocations"]
        for case in allocation["case_library"]["records"]
    ]
    maximum_energy_residual = max(
        abs(
            float(
                case["reoptimized_response"][residual]
            )
        )
        for case in all_cases
        for residual in ("original_energy_residual_w", "flattened_energy_residual_w")
    )
    maximum_ellipsoid_norm = max(
        float(case["ellipsoid_norm_squared"]) for case in all_cases
    )
    maximum_seebeck_deviation = max(
        float(
            case["property_admissibility"][
                "maximum_relative_Seebeck_deviation"
            ]
        )
        for case in all_cases
    )

    analysis = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "title": "Joint deterministic error geometry for the PbSe/Cr common-mode response",
        "central_scientific_result": (
            "The public method ceilings do not select a unique sign conclusion. "
            "A shared additive calibration-drift axis is physically separable from "
            "band transport and reverses the reoptimized response only near its full "
            "conservative 5% ceiling.  Simultaneous S/sigma/kappa ellipsoids retain "
            "negative cases for independent and moderately band-coupled residuals, "
            "whereas strongly same-run-correlated geometries markedly narrow the tested "
            "envelope and retain a positive minimum in the deterministic direction library."
        ),
        "scope": {
            "probability_distribution_assumed": False,
            "confidence_interval_reported": False,
            "correlation_matrices_empirically_estimated": False,
            "correlation_entries_are_deterministic_scenario_parameters": True,
            "continuous_all_function_worst_case_certified": False,
            "global_nonlinear_ellipsoid_minima_certified": False,
            "negative_case_proves_sign_nonidentifiability_for_its_stated_geometry": True,
            "positive_tested_minimum_proves_global_identifiability": False,
            "real_pbse_cr_device_validation": False,
            "figure_derived_candidate_scenario_screen": True,
        },
        "physical_error_model": {
            "parameters": list(PARAMETER_NAMES),
            "shared_additive_drift": {
                "scale_v_per_k": context.common_additive_bound_v_per_k,
                "temperature_basis": "z=(T-Tmid)/Thalf over the common source support",
                "same_additive_deltaS_applied_to_p_and_n": True,
                "interpretation": (
                    "common thermovoltage/DeltaT calibration-map drift; not a change "
                    "of carrier concentration or band structure"
                ),
                "independent_of_sigma_and_kappa_in_all_scenarios": True,
                "public_record_proves_p_n_same_calibration_state": False,
            },
            "branch_specific_Seebeck_residuals": {
                "active_knots_k": [323, 373],
                "axis_scale_relative": SEEBECK_RELATIVE_CEILING,
                "later_knots_omitted_because_local_PCHIP_sensitivity_is_zero": True,
                "300K_and_423K_knots_omitted_from_reduced_joint_model": True,
            },
            "transport_scale_errors": {
                "sigma_axis_scale_relative": SIGMA_RELATIVE_CEILING,
                "kappa_axis_scale_relative": KAPPA_RELATIVE_CEILING,
                "whole_curve_scale_coordinates": True,
            },
            "ellipsoid_definition": (
                "delta = D C^(1/2) u with ||u||_2 <= 1; C is a deterministic "
                "correlation-shape matrix and D partitions the Seebeck error budget"
            ),
            "reported_ranges_are_not_confidence_intervals": True,
        },
        "target_condition": {
            "device_id": forward.TARGET_DEVICE_ID,
            "hot_temperature_k": context.base.hot_temperature_k,
            "cold_temperature_k": context.base.cold_temperature_k,
            "nominal_optimum_current_a": context.nominal_current_a,
            "below_300_k_extrapolation_used": False,
        },
        "nominal_result": nominal,
        "local_reoptimized_gradient": {
            "coordinate_order": list(PARAMETER_NAMES),
            "finite_difference_steps": list(GRADIENT_STEPS),
            "gradient_w_per_normalized_coordinate": gradient.tolist(),
            "coarse_gradient_w_per_normalized_coordinate": gradients[
                str(GRADIENT_STEPS[0])
            ].tolist(),
            "maximum_coarse_fine_absolute_difference_w": float(
                np.max(gradient_difference)
            ),
        },
        "single_datum_sign_structure": single_datum,
        "shared_instrument_drift_axis": shared_axis,
        "joint_correlation_scenarios": scenario_results,
        "what_is_learned": [
            (
                "A common additive instrument drift is not constrained by "
                "Pisarenko or Wiedemann-Franz because it is not a material change."
            ),
            (
                "Opposite p/n relative-knot sensitivity signs do not imply branch "
                "cancellation for an equal additive deltaS; the carrier signs convert "
                "that drift into opposite relative errors and same-sign branch effects."
            ),
            (
                "Joint S/sigma/kappa correlations can narrow the tested response range "
                "enough that some explicit geometries retain a positive minimum."
            ),
            (
                "Other explicit PSD geometries retain admissible negative cases, "
                "so the public curves and marginal method ceilings alone cannot select "
                "the sign conclusion."
            ),
        ],
        "what_is_not_learned": [
            "No scenario matrix is an empirical covariance estimate.",
            "No probability, confidence level, or confidence interval is obtained.",
            "A positive minimum in the tested direction library is not a global proof.",
            "The analysis does not validate the material attribution of the device.",
            "The public record does not establish same-run p/n calibration covariance.",
        ],
        "required_measurement_to_resolve_sign": {
            "data": (
                "paired same-run raw p/n thermovoltage and DeltaT records near 323 and "
                "373 K, calibration/reference records, replicate covariance, matched "
                "sigma and kappa specimen identifiers, and specimen-to-device mapping"
            ),
            "reason": (
                "these data choose the shared-drift allocation and correlation geometry "
                "that the public figures leave unspecified"
            ),
        },
        "verification": {
            "nominal_effect_reproduction_error_w": abs(
                nominal_effect - ident.EXPECTED_NOMINAL_EFFECT_W
            ),
            "maximum_gradient_step_difference_w": float(
                np.max(gradient_difference)
            ),
            "maximum_case_ellipsoid_norm_squared": maximum_ellipsoid_norm,
            "maximum_case_relative_Seebeck_deviation": maximum_seebeck_deviation,
            "all_cases_admissible": bool(
                all(case["property_admissibility"]["admissible"] for case in all_cases)
            ),
            "all_scenario_shapes_positive_semidefinite": bool(
                all(
                    scenario["minimum_correlation_matrix_eigenvalue"] >= -1.0e-12
                    for scenario in scenario_results
                )
            ),
            "shared_drift_cross_correlations_exactly_zero": bool(
                all(
                    scenario["named_correlations"][
                        "shared_drift_maximum_absolute_cross_correlation"
                    ]
                    == 0.0
                    for scenario in scenario_results
                )
            ),
            "maximum_optimized_module_energy_residual_w": maximum_energy_residual,
            "evaluated_unique_full_responses": len(evaluator.cache),
            "elapsed_seconds": float(time.time() - started),
        },
        "input_bindings": context.validation["bindings"],
        "figure_metadata": {
            "core_conclusion": (
                "The sign conclusion changes with the physically stated joint-error "
                "geometry, and the opposite p/n relative sensitivities do not cancel "
                "an equal additive instrument drift."
            ),
            "evidence_chain": {
                "a": "shared-drift threshold and sign crossing",
                "b": "minimum nonlinear cases across correlation geometries",
                "c": "conversion from relative-knot sensitivity to additive-error response",
            },
            "layout": "quantitative grid",
            "emphasis_panel": "b",
            "backend": "Python matplotlib only",
            "statistical_interval": False,
            "source_data_exported": True,
        },
    }
    return analysis, evaluator


def write_source_csv(analysis: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for record in analysis["shared_instrument_drift_axis"]["sweep"]:
        rows.append(
            {
                "panel": "a",
                "series": "negative shared additive drift axis",
                "x": record["fraction_of_full_negative_drift_axis"],
                "y": 1.0e3 * record["reoptimized_response_w"],
                "x_unit": "fraction of deterministic axis",
                "y_unit": "mW",
                "note": "reoptimized original-minus-flattened Qc",
            }
        )
    for scenario in analysis["joint_correlation_scenarios"]:
        for allocation in scenario["allocations"]:
            rows.append(
                {
                    "panel": "b",
                    "series": scenario["short_label"],
                    "x": allocation["shared_additive_drift_fraction"],
                    "y": 1.0e3
                    * allocation["minimum_evaluated_nonlinear_case_w"],
                    "x_unit": "shared Seebeck budget fraction",
                    "y_unit": "mW",
                    "note": "minimum deterministic direction-library case",
                }
            )
    for record in analysis["single_datum_sign_structure"]["branch_records"]:
        rows.append(
            {
                "panel": "c",
                "series": f"{record['carrier']}, {record['nominal_temperature_label_k']} K",
                "x": record["nominal_temperature_label_k"],
                "y": record[
                    "response_change_mw_per_positive_1_uV_per_K_additive_error"
                ],
                "x_unit": "K",
                "y_unit": "mW per (uV/K)",
                "note": "branch contribution for equal positive additive deltaS",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("panel", "series", "x", "y", "x_unit", "y_unit", "note"),
        )
        writer.writeheader()
        writer.writerows(rows)


def make_figure(analysis: dict[str, Any], prefix: Path) -> list[Path]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    navy = "#244A64"
    orange = "#D9822B"
    blue = "#5A8BB0"
    neutral = "#6E7781"
    positive = "#9C3D4A"
    negative = "#356A9A"

    fig = plt.figure(figsize=(3.50, 6.15), constrained_layout=False)
    grid = fig.add_gridspec(
        3, 1, height_ratios=(1.0, 1.45, 1.05), hspace=0.62,
        left=0.27, right=0.96, bottom=0.08, top=0.98
    )

    # a: exact nonlinear shared-instrument axis.
    ax_a = fig.add_subplot(grid[0, 0])
    sweep = analysis["shared_instrument_drift_axis"]["sweep"]
    x = np.asarray(
        [row["fraction_of_full_negative_drift_axis"] for row in sweep], dtype=float
    )
    y = 1.0e3 * np.asarray(
        [row["reoptimized_response_w"] for row in sweep], dtype=float
    )
    ax_a.plot(x, y, color=navy, lw=1.7, marker="o", ms=3.4, mfc="white", mew=0.9)
    ax_a.axhline(0.0, color=neutral, lw=0.8, ls=(0, (3, 2)))
    root = analysis["shared_instrument_drift_axis"][
        "zero_crossing_fraction_of_full_negative_axis"
    ]
    ax_a.axvline(root, color=orange, lw=1.0, ls=(0, (2, 2)))
    ax_a.text(
        root - 0.015,
        0.30,
        f"zero at {root:.3f}",
        color=orange,
        ha="right",
        va="bottom",
        fontsize=6.6,
    )
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_xlabel("fraction of conservative shared-drift axis")
    ax_a.set_ylabel(r"reoptimized $\Delta Q_{c,\mathrm{cm}}$ (mW)")
    ax_a.text(-0.17, 1.05, "a", transform=ax_a.transAxes, fontweight="bold", fontsize=8)

    # b: minimum evaluated nonlinear case by geometry and allocation.
    ax_b = fig.add_subplot(grid[1, 0])
    scenarios = analysis["joint_correlation_scenarios"]
    fractions = [
        row["shared_additive_drift_fraction"] for row in scenarios[0]["allocations"]
    ]
    matrix = np.asarray(
        [
            [
                1.0e3 * allocation["minimum_evaluated_nonlinear_case_w"]
                for allocation in scenario["allocations"]
            ]
            for scenario in scenarios
        ],
        dtype=float,
    )
    span = max(abs(float(np.min(matrix))), abs(float(np.max(matrix))))
    image = ax_b.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-span,
        vmax=span,
        aspect="auto",
        interpolation="nearest",
    )
    ax_b.set_xticks(np.arange(len(fractions)), [f"{value:.1f}" for value in fractions])
    ax_b.set_yticks(
        np.arange(len(scenarios)), [scenario["short_label"] for scenario in scenarios]
    )
    ax_b.set_xlabel("shared fraction of quadratic Seebeck budget")
    ax_b.set_ylabel("correlation-shape scenario")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if abs(value) > 0.52 * span else "black"
            ax_b.text(
                column,
                row,
                f"{value:+.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=6.4,
                fontweight="bold" if value < 0.0 else "normal",
            )
    colorbar = fig.colorbar(image, ax=ax_b, fraction=0.045, pad=0.035)
    colorbar.set_label("minimum evaluated case (mW)", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6)
    ax_b.text(-0.17, 1.05, "b", transform=ax_b.transAxes, fontweight="bold", fontsize=8)

    # c: convert relative single-knot sensitivities to a common additive error.
    ax_c = fig.add_subplot(grid[2, 0])
    branches = analysis["single_datum_sign_structure"]["branch_records"]
    positions = {("p", 323): -0.18, ("n", 323): 0.18, ("p", 373): 0.82, ("n", 373): 1.18}
    colors = {"p": orange, "n": blue}
    for row in branches:
        key = (row["carrier"], row["nominal_temperature_label_k"])
        value = row[
            "response_change_mw_per_positive_1_uV_per_K_additive_error"
        ]
        ax_c.bar(
            positions[key], value, width=0.31, color=colors[row["carrier"]],
            edgecolor="white", linewidth=0.5
        )
        offset = 0.035 if value >= 0 else -0.035
        ax_c.text(
            positions[key], value + offset, f"{value:+.2f}",
            ha="center", va="bottom" if value >= 0 else "top", fontsize=6.3
        )
    sums = analysis["single_datum_sign_structure"]["temperature_sums"]
    for x_position, row in zip((0.0, 1.0), sums):
        total = row["p_plus_n_contribution_mw_per_positive_1_uV_per_K"]
        ax_c.plot(x_position, total, marker="D", ms=4.1, color=navy, zorder=5)
        ax_c.text(
            x_position,
            total + 0.065,
            f"sum {total:+.2f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=navy,
        )
    ax_c.axhline(0.0, color=neutral, lw=0.8)
    ax_c.set_xticks((0.0, 1.0), ("323 K", "373 K"))
    ax_c.set_xlim(-0.55, 1.55)
    ax_c.set_ylim(-1.05, 1.02)
    ax_c.set_ylabel(
        r"response / ($+1\ \mu$V K$^{-1}$ $\delta S$) (mW)",
        fontsize=6.5,
        labelpad=3,
    )
    ax_c.text(
        0.98, 0.04, "bars: branches   points: p+n", transform=ax_c.transAxes,
        ha="right", va="bottom", fontsize=6.3, color=navy
    )
    ax_c.text(0.01, 1.04, "c", transform=ax_c.transAxes, fontweight="bold", fontsize=8)

    # Direct carrier labels avoid a detached legend.
    ax_c.text(-0.18, 0.87, "p", color=orange, ha="center", fontsize=6.5)
    ax_c.text(0.18, 0.87, "n", color=blue, ha="center", fontsize=6.5)

    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [prefix.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf", ".tiff")]
    fig.savefig(outputs[0], dpi=600, facecolor="white")
    fig.savefig(
        outputs[1],
        facecolor="white",
        metadata={"Date": None, "Creator": None},
    )
    fig.savefig(outputs[2], facecolor="white")
    fig.savefig(
        outputs[3], dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close(fig)
    return outputs


def serialize_results(
    analysis: dict[str, Any],
    json_output: Path,
    figure_outputs: list[Path],
    source_csv: Path,
) -> dict[str, Any]:
    script = Path(__file__).resolve()
    analysis["outputs"] = {
        "analysis_script": {
            "locator": output_locator(script),
            "sha256": file_sha256(script),
        },
        "result_json": output_locator(json_output),
        "source_data": {
            "locator": output_locator(source_csv),
            "sha256": file_sha256(source_csv),
        },
        "figures": {
            path.suffix.removeprefix("."): {
                "locator": output_locator(path),
                "sha256": file_sha256(path),
            }
            for path in figure_outputs
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": mpl.__version__,
        },
    }
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return analysis


def run_analysis(
    json_output: Path = DEFAULT_JSON,
    figure_prefix: Path = DEFAULT_FIGURE_PREFIX,
    source_csv: Path = DEFAULT_SOURCE_CSV,
) -> dict[str, Any]:
    analysis, _ = analyze_joint_model()
    write_source_csv(analysis, source_csv)
    figure_outputs = make_figure(analysis, figure_prefix)
    return serialize_results(analysis, json_output, figure_outputs, source_csv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--figure-prefix", type=Path, default=DEFAULT_FIGURE_PREFIX)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    result = run_analysis(
        arguments.json_output, arguments.figure_prefix, arguments.source_csv
    )
    root = result["shared_instrument_drift_axis"][
        "zero_crossing_fraction_of_full_negative_axis"
    ]
    print(f"shared-drift zero crossing: {root:.6f} of the conservative axis")
    for scenario in result["joint_correlation_scenarios"]:
        summary = ", ".join(
            f"f={row['shared_additive_drift_fraction']:.1f}: "
            f"{1e3 * row['minimum_evaluated_nonlinear_case_w']:+.3f} mW"
            for row in scenario["allocations"]
        )
        print(f"{scenario['scenario_id']}: {summary}")


if __name__ == "__main__":
    main()
