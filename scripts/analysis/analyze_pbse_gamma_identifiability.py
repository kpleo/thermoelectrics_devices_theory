#!/usr/bin/env python3
"""Bound the identifiability of the PbSe/Cr common-mode Thomson response.

The nominal public-figure reconstruction gives a 3.066878 mW change in the
reoptimized seven-pair cold-side capacity when the Seebeck common mode

    M(T) = [S_p(T) + S_n(T)] / 2

is flattened while the differential mode alpha(T)=S_p(T)-S_n(T) is retained.
This script asks a different and stricter materials-physics question: is the
sign or order of magnitude of that small counterfactual response identified by
the public Seebeck data and their published ``below 5%'' method statement?

No probability model or p/n covariance matrix was reported.  The analysis
therefore uses deterministic admissible cases rather than Monte Carlo
confidence intervals.  It separates:

* vector-extraction disagreement (digitization only),
* interpolation choice (PCHIP, piecewise linear, and a bound-checked smooth
  quadratic),
* structured Seebeck errors (constant offset, gain, temperature drift, and
  p/n coupling),
* a complete monotone +/-5% corner screen over the PCHIP knots that can affect
  the 309.894--362.998 K operating interval, and
* a separate sigma/kappa branch-transfer sensitivity layer.  The latter never
  gets pooled with the Seebeck layer into a pseudo-confidence interval.

Every perturbed Seebeck pair is decomposed anew into alpha and M.  Its original
candidate is compared with its own alpha-preserving flattened counterfactual,
so the reported delta Qc always isolates the candidate's common-mode gradient
within that candidate rather than comparing two different differential modes.

All source roles and numerical inputs are hash-bound and fail closed.  The
result remains a figure-derived scenario screen, not a calibrated device
validation or a statistical uncertainty statement.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import analyze_pbse_common_mode_contribution as common  # noqa: E402
from scripts.analysis import analyze_pbse_device_forward_constraint as forward  # noqa: E402
from scripts.tec_1d_solver import (  # noqa: E402
    PchipTemperatureProperty,
    TemperatureDependentLeg,
    TemperatureDependentNumericalCouple,
)


SCHEMA_VERSION = "pbse_gamma_identifiability/v1"
ANALYSIS_ID = "SCI-PBSE-GAMMA-IDENTIFIABILITY-20260826"
DEFAULT_JSON = (
    ROOT
    / "results/scientific_analysis/pbse_gamma_identifiability_results.json"
)
DEFAULT_FIGURE_PREFIX = (
    ROOT / "results/scientific_analysis/pbse_gamma_identifiability"
)

VECTOR_CANDIDATE = (
    ROOT / "data/raw/device_transport_vector/vector_reconciled_candidate.json"
)
VECTOR_QC = ROOT / "data/raw/device_transport_vector/vector_reconciliation_qc.json"
SOURCE_RECORDS = ROOT / "data/raw/source_records.csv"

EXPECTED_INPUT_SHA256 = {
    forward.FIG1_CSV: "8de968addb27b5badc54d775cb0d3be1d6c3bc033cdb487cb18ac520711c8938",
    forward.S9_CSV: "dbe455f657e164bb4e1a16909ec996feb9765b40badba9095a0079339518d2da",
    forward.FIG4_CSV: "cdd9a4c693208434e10ca6b79cb29ebb606c16d1d24d494c68fdeed15940ba48",
    forward.CONDITIONS_CSV: "69c511fed9ccb3e4ecd6d2250f0613896d23de3746dd1bee808b188db1e1c243",
    VECTOR_CANDIDATE: "23c00d1d438a68f98633bc7f76715ff77331b8240ac56041cbec127bece373d0",
    VECTOR_QC: "3ba2ed2808fa27e7ddf2615cfed9ab6532f0e759843832a4e067df841d3112e5",
    SOURCE_RECORDS: "80ead0fb33ca377c03ac5c14f2119ef2921eaee08c77d26bfa66223ff0d81f67",
}
EXPECTED_SI_SHA256 = "d633afd285f385a95f5c20655654f6bd014fa32d370630b4e22ca0af32d974a5"
EXPECTED_NOMINAL_EFFECT_W = 0.0030668783059080162

SEEBECK_RELATIVE_METHOD_CEILING = 0.05
SIGMA_RELATIVE_METHOD_CEILING = 0.05
KAPPA_RELATIVE_METHOD_CEILING = 0.15
PROPERTY_GRID_POINTS = 401
FULL_SUPPORT_GRID_POINTS = 1001
LOCALLY_ACTIVE_PCHIP_KNOTS_PER_LEG = 4
SENSITIVITY_RELATIVE_STEP = 1.0e-3

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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _validate_and_load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    observed_hashes: dict[str, str] = {}
    for path, expected in EXPECTED_INPUT_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(f"required bound input is absent: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"bound input hash mismatch for {path}: {observed} != {expected}"
            )
        observed_hashes[output_locator(path)] = observed

    source_rows = _read_csv(SOURCE_RECORDS)
    si_rows = [row for row in source_rows if row["source_id"] == "science_si"]
    if len(si_rows) != 1:
        raise ValueError("science_si source-record binding is not unique")
    si_row = si_rows[0]
    if si_row["data_role"] != "primary_supporting_information":
        raise ValueError("science_si role mismatch")
    if si_row["sha256"] != EXPECTED_SI_SHA256:
        raise ValueError("science_si source-record hash mismatch")
    si_locator = si_row["path"]
    si_path_from_environment = os.environ.get("PBSE_SCIENCE_SI_PDF")
    source_file_verified_in_run = False
    if si_path_from_environment:
        si_path = Path(si_path_from_environment).expanduser()
        if not si_path.is_file() or file_sha256(si_path) != EXPECTED_SI_SHA256:
            raise ValueError("PBSE_SCIENCE_SI_PDF is absent or hash-mismatched")
        source_file_verified_in_run = True

    inputs = forward.load_inputs()
    if any(
        row["data_role"]
        != "primary_article_figure_derived_source_object_candidate"
        for key in ("p_seebeck", "n_seebeck", "p_sigma", "n_sigma")
        for row in inputs[key]
    ):
        raise ValueError("Figure 1 selected-curve role mismatch")
    if any(
        row["data_role"]
        != "figure_derived_candidate_measured_as_described_in_si"
        for key in ("p_kappa", "n_kappa")
        for row in inputs[key]
    ):
        raise ValueError("Figure S9 selected-curve role mismatch")
    if any(
        row["independent_device_validation_eligible"].lower() != "false"
        for key in (
            "p_seebeck",
            "n_seebeck",
            "p_sigma",
            "n_sigma",
            "p_kappa",
            "n_kappa",
        )
        for row in inputs[key]
    ):
        raise ValueError("candidate input unexpectedly passed real-data independent device-validation criterion")

    vector_qc = json.loads(VECTOR_QC.read_text(encoding="utf-8"))
    if vector_qc.get("status") != "pass":
        raise ValueError("independent vector-route reconciliation did not pass")
    if vector_qc["validation_status"]["independent_device_validation_passed"] is not False:
        raise ValueError("vector candidate unexpectedly passed real-data independent device-validation criterion")
    vector = json.loads(VECTOR_CANDIDATE.read_text(encoding="utf-8"))
    if vector.get("schema_version") != "device_transport_vector_reconciled_candidate/v1":
        raise ValueError("unexpected vector-candidate schema")
    if len(vector.get("points", [])) != 28:
        raise ValueError("expected 28 reconciled p/n sigma/Seebeck vector points")

    bindings = {
        "hash_algorithm": "sha256",
        "bound_input_hashes": observed_hashes,
        "supporting_information": {
            "locator": si_locator,
            "sha256": EXPECTED_SI_SHA256,
            "data_role": si_row["data_role"],
            "source_file_verified_in_run": source_file_verified_in_run,
            "method_statement_location": "PDF page 2 / printed S2",
            "method_statement_paraphrase": (
                "ZEM-3 Seebeck coefficient and electrical-conductivity "
                "measurement errors were kept below 5% over 300--673 K"
            ),
            "thermal_statement_location": "PDF page 3 / printed S3",
            "thermal_statement_paraphrase": (
                "total-thermal-conductivity uncertainty is within 15%"
            ),
        },
        "roles": {
            "figure1": "primary_article_figure_derived_source_object_candidate",
            "figure_s9": "figure_derived_candidate_measured_as_described_in_si",
            "figure4_endpoint": "figure_derived_measured_as_described_in_main_text",
            "independent_device_validation_eligible": False,
        },
    }
    return inputs, {"bindings": bindings, "vector": vector}


class PiecewiseLinearTemperatureProperty(PchipTemperatureProperty):
    """Piecewise-linear property with analytic one-sided segment derivative."""

    def __init__(self, temperature_knots_k: Iterable[float], values_si: Iterable[float]):
        super().__init__(temperature_knots_k, values_si)
        object.__setattr__(
            self,
            "_linear_temperature",
            np.asarray(self.temperature_knots_k, dtype=float),
        )
        object.__setattr__(
            self,
            "_linear_values",
            np.asarray(self.values_si, dtype=float),
        )
        object.__setattr__(
            self,
            "_linear_slopes",
            np.diff(self._linear_values) / np.diff(self._linear_temperature),
        )

    def evaluate(self, temperature_k: object) -> FloatArray:
        checked = PchipTemperatureProperty.evaluate(self, temperature_k)
        # The parent call is used only for its strict domain check.
        temperature = np.asarray(temperature_k, dtype=float)
        return np.asarray(
            np.interp(temperature, self._linear_temperature, self._linear_values),
            dtype=float,
        ).reshape(np.shape(checked))

    def derivative(self, temperature_k: object) -> FloatArray:
        checked = PchipTemperatureProperty.derivative(self, temperature_k)
        temperature = np.asarray(temperature_k, dtype=float)
        indices = np.searchsorted(
            self._linear_temperature, temperature, side="right"
        ) - 1
        indices = np.clip(indices, 0, len(self._linear_slopes) - 1)
        return np.asarray(self._linear_slopes[indices], dtype=float).reshape(
            np.shape(checked)
        )


class BoundCheckedQuadraticTemperatureProperty(PchipTemperatureProperty):
    """Smooth quadratic accepted only when it stays physical and within 5% at knots."""

    def __init__(self, temperature_knots_k: Iterable[float], values_si: Iterable[float]):
        temperature = np.asarray(tuple(temperature_knots_k), dtype=float)
        values = np.asarray(tuple(values_si), dtype=float)
        super().__init__(temperature, values)
        center = 0.5 * (float(temperature[0]) + float(temperature[-1]))
        half_range = 0.5 * (float(temperature[-1]) - float(temperature[0]))
        normalized = (temperature - center) / half_range
        coefficients = np.polynomial.polynomial.polyfit(normalized, values, 2)
        fitted = np.polynomial.polynomial.polyval(normalized, coefficients)
        maximum_relative_residual = float(
            np.max(np.abs(fitted - values) / np.abs(values))
        )
        dense_x = np.linspace(-1.0, 1.0, FULL_SUPPORT_GRID_POINTS)
        dense_values = np.polynomial.polynomial.polyval(dense_x, coefficients)
        derivative_coefficients = np.polynomial.polynomial.polyder(coefficients)
        dense_derivative = (
            np.polynomial.polynomial.polyval(dense_x, derivative_coefficients)
            / half_range
        )
        direction = float(np.sign(values[-1] - values[0]))
        if maximum_relative_residual > SEEBECK_RELATIVE_METHOD_CEILING:
            raise ValueError("smooth quadratic exceeds the published 5% knot band")
        if np.any(np.sign(dense_values) != np.sign(values[0])):
            raise ValueError("smooth quadratic changes carrier sign")
        if np.any(direction * dense_derivative <= 0.0):
            raise ValueError("smooth quadratic violates the observed monotone trend")
        object.__setattr__(self, "_quadratic_center_k", center)
        object.__setattr__(self, "_quadratic_half_range_k", half_range)
        object.__setattr__(self, "_quadratic_coefficients", coefficients)
        object.__setattr__(
            self,
            "maximum_relative_knot_residual",
            maximum_relative_residual,
        )

    def evaluate(self, temperature_k: object) -> FloatArray:
        checked = PchipTemperatureProperty.evaluate(self, temperature_k)
        temperature = np.asarray(temperature_k, dtype=float)
        normalized = (
            temperature - self._quadratic_center_k
        ) / self._quadratic_half_range_k
        return np.asarray(
            np.polynomial.polynomial.polyval(
                normalized, self._quadratic_coefficients
            ),
            dtype=float,
        ).reshape(np.shape(checked))

    def derivative(self, temperature_k: object) -> FloatArray:
        checked = PchipTemperatureProperty.derivative(self, temperature_k)
        temperature = np.asarray(temperature_k, dtype=float)
        normalized = (
            temperature - self._quadratic_center_k
        ) / self._quadratic_half_range_k
        derivative_coefficients = np.polynomial.polynomial.polyder(
            self._quadratic_coefficients
        )
        return np.asarray(
            np.polynomial.polynomial.polyval(normalized, derivative_coefficients)
            / self._quadratic_half_range_k,
            dtype=float,
        ).reshape(np.shape(checked))


class AffinePerturbedSeebeckProperty(PchipTemperatureProperty):
    """Apply a bounded affine gain and/or additive drift to a source law.

    S*(T) = [1 + g0 + g1 z(T)] S(T) + b0 + b1 z(T),
    where z=(T-Tmid)/Thalf.  The analytic derivative includes the product rule.
    """

    def __init__(
        self,
        source: PchipTemperatureProperty,
        *,
        normalization_center_k: float,
        normalization_half_range_k: float,
        gain_constant_fraction: float = 0.0,
        gain_drift_fraction: float = 0.0,
        additive_constant_v_per_k: float = 0.0,
        additive_drift_v_per_k: float = 0.0,
    ) -> None:
        super().__init__(
            (source.minimum_temperature_k, source.maximum_temperature_k),
            (0.0, 0.0),
        )
        if normalization_half_range_k <= 0.0:
            raise ValueError("normalization_half_range_k must be positive")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "normalization_center_k", float(normalization_center_k))
        object.__setattr__(
            self, "normalization_half_range_k", float(normalization_half_range_k)
        )
        object.__setattr__(self, "gain_constant_fraction", float(gain_constant_fraction))
        object.__setattr__(self, "gain_drift_fraction", float(gain_drift_fraction))
        object.__setattr__(
            self, "additive_constant_v_per_k", float(additive_constant_v_per_k)
        )
        object.__setattr__(
            self, "additive_drift_v_per_k", float(additive_drift_v_per_k)
        )

    def _z(self, temperature_k: object) -> FloatArray:
        return np.asarray(
            (np.asarray(temperature_k, dtype=float) - self.normalization_center_k)
            / self.normalization_half_range_k,
            dtype=float,
        )

    def evaluate(self, temperature_k: object) -> FloatArray:
        PchipTemperatureProperty.evaluate(self, temperature_k)
        z = self._z(temperature_k)
        gain = 1.0 + self.gain_constant_fraction + self.gain_drift_fraction * z
        additive = self.additive_constant_v_per_k + self.additive_drift_v_per_k * z
        return np.asarray(gain * self.source.evaluate(temperature_k) + additive)

    def derivative(self, temperature_k: object) -> FloatArray:
        PchipTemperatureProperty.derivative(self, temperature_k)
        z = self._z(temperature_k)
        source_value = self.source.evaluate(temperature_k)
        source_derivative = self.source.derivative(temperature_k)
        gain = 1.0 + self.gain_constant_fraction + self.gain_drift_fraction * z
        return np.asarray(
            gain * source_derivative
            + self.gain_drift_fraction
            / self.normalization_half_range_k
            * source_value
            + self.additive_drift_v_per_k / self.normalization_half_range_k,
            dtype=float,
        )


class ScaledPchipProperty(PchipTemperatureProperty):
    """Scale a specified property while retaining its domain and derivative."""

    def __init__(self, source: PchipTemperatureProperty, scale: float) -> None:
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("property scale must be positive and finite")
        super().__init__(
            (source.minimum_temperature_k, source.maximum_temperature_k),
            (1.0, 1.0),
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "scale", float(scale))

    def evaluate(self, temperature_k: object) -> FloatArray:
        PchipTemperatureProperty.evaluate(self, temperature_k)
        return np.asarray(self.scale * self.source.evaluate(temperature_k), dtype=float)

    def derivative(self, temperature_k: object) -> FloatArray:
        PchipTemperatureProperty.derivative(self, temperature_k)
        return np.asarray(self.scale * self.source.derivative(temperature_k), dtype=float)


def _build_couple_with_properties(
    base: TemperatureDependentNumericalCouple,
    p_seebeck: PchipTemperatureProperty,
    n_seebeck: PchipTemperatureProperty,
    *,
    p_rho: PchipTemperatureProperty | None = None,
    n_rho: PchipTemperatureProperty | None = None,
    p_kappa: PchipTemperatureProperty | None = None,
    n_kappa: PchipTemperatureProperty | None = None,
) -> TemperatureDependentNumericalCouple:
    p_leg = TemperatureDependentLeg(
        seebeck=p_seebeck,
        electrical_resistivity=p_rho or base.p_leg.electrical_resistivity,
        thermal_conductivity=p_kappa or base.p_leg.thermal_conductivity,
        length_m=base.p_leg.length_m,
        area_m2=base.p_leg.area_m2,
    )
    n_leg = TemperatureDependentLeg(
        seebeck=n_seebeck,
        electrical_resistivity=n_rho or base.n_leg.electrical_resistivity,
        thermal_conductivity=n_kappa or base.n_leg.thermal_conductivity,
        length_m=base.n_leg.length_m,
        area_m2=base.n_leg.area_m2,
    )
    return TemperatureDependentNumericalCouple(
        p_leg=p_leg,
        n_leg=n_leg,
        cold_temperature_k=base.cold_temperature_k,
        hot_temperature_k=base.hot_temperature_k,
    )


def _build_transport_scaled_couple(
    base: TemperatureDependentNumericalCouple,
    *,
    p_sigma_scale: float,
    n_sigma_scale: float,
    p_kappa_scale: float,
    n_kappa_scale: float,
) -> TemperatureDependentNumericalCouple:
    # rho=1/sigma, hence a conductivity gain maps to the reciprocal rho scale.
    return _build_couple_with_properties(
        base,
        base.p_leg.seebeck,
        base.n_leg.seebeck,
        p_rho=ScaledPchipProperty(
            base.p_leg.electrical_resistivity, 1.0 / p_sigma_scale
        ),
        n_rho=ScaledPchipProperty(
            base.n_leg.electrical_resistivity, 1.0 / n_sigma_scale
        ),
        p_kappa=ScaledPchipProperty(base.p_leg.thermal_conductivity, p_kappa_scale),
        n_kappa=ScaledPchipProperty(base.n_leg.thermal_conductivity, n_kappa_scale),
    )


def _property_summary(
    couple: TemperatureDependentNumericalCouple,
    baseline: TemperatureDependentNumericalCouple,
) -> dict[str, Any]:
    cold = couple.cold_temperature_k
    hot = couple.hot_temperature_k
    target_temperature = np.linspace(cold, hot, PROPERTY_GRID_POINTS)
    support_minimum = max(
        couple.p_leg.seebeck.minimum_temperature_k,
        couple.n_leg.seebeck.minimum_temperature_k,
        baseline.p_leg.seebeck.minimum_temperature_k,
        baseline.n_leg.seebeck.minimum_temperature_k,
    )
    support_maximum = min(
        couple.p_leg.seebeck.maximum_temperature_k,
        couple.n_leg.seebeck.maximum_temperature_k,
        baseline.p_leg.seebeck.maximum_temperature_k,
        baseline.n_leg.seebeck.maximum_temperature_k,
    )
    support_temperature = np.linspace(
        support_minimum, support_maximum, FULL_SUPPORT_GRID_POINTS
    )

    sp = couple.p_leg.seebeck.evaluate(target_temperature)
    sn = couple.n_leg.seebeck.evaluate(target_temperature)
    dsp = couple.p_leg.seebeck.derivative(target_temperature)
    dsn = couple.n_leg.seebeck.derivative(target_temperature)
    common_mode = 0.5 * (sp + sn)
    gamma = target_temperature * 0.5 * (dsp + dsn)
    alpha = sp - sn

    sp_support = couple.p_leg.seebeck.evaluate(support_temperature)
    sn_support = couple.n_leg.seebeck.evaluate(support_temperature)
    base_sp_support = baseline.p_leg.seebeck.evaluate(support_temperature)
    base_sn_support = baseline.n_leg.seebeck.evaluate(support_temperature)
    relative_p = np.abs(sp_support - base_sp_support) / np.abs(base_sp_support)
    relative_n = np.abs(sn_support - base_sn_support) / np.abs(base_sn_support)
    baseline_alpha = baseline.p_leg.seebeck.evaluate(
        target_temperature
    ) - baseline.n_leg.seebeck.evaluate(target_temperature)
    return {
        "temperature_k": target_temperature.tolist(),
        "common_mode_v_per_k": common_mode.tolist(),
        "gamma_v_per_k": gamma.tolist(),
        "alpha_v_per_k": alpha.tolist(),
        "summary": {
            "M_at_cold_v_per_k": float(common_mode[0]),
            "M_at_hot_v_per_k": float(common_mode[-1]),
            "M_change_v_per_k": float(common_mode[-1] - common_mode[0]),
            "Gamma_minimum_v_per_k": float(np.min(gamma)),
            "Gamma_maximum_v_per_k": float(np.max(gamma)),
            "Gamma_temperature_mean_v_per_k": float(
                np.trapezoid(gamma, target_temperature) / (hot - cold)
            ),
            "maximum_alpha_change_from_nominal_v_per_k": float(
                np.max(np.abs(alpha - baseline_alpha))
            ),
            "maximum_relative_p_Seebeck_deviation_on_common_support": float(
                np.max(relative_p)
            ),
            "maximum_relative_n_Seebeck_deviation_on_common_support": float(
                np.max(relative_n)
            ),
            "p_Seebeck_sign_preserved": bool(np.all(sp_support > 0.0)),
            "n_Seebeck_sign_preserved": bool(np.all(sn_support < 0.0)),
            "p_Seebeck_monotone_increasing": bool(
                np.all(np.diff(sp_support) > 0.0)
            ),
            "n_Seebeck_monotone_decreasing": bool(
                np.all(np.diff(sn_support) < 0.0)
            ),
        },
    }


def _effect_record(
    scenario_id: str,
    family: str,
    couple: TemperatureDependentNumericalCouple,
    baseline: TemperatureDependentNumericalCouple,
    nominal_current_a: float,
    *,
    description: str,
    error_coupling: str,
    interpolation: str,
    ceiling_application: str,
    perturbation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    properties = _property_summary(couple, baseline)
    cold = couple.cold_temperature_k
    m_cold = 0.5 * (
        float(couple.p_leg.seebeck.evaluate([cold])[0])
        + float(couple.n_leg.seebeck.evaluate([cold])[0])
    )
    flattened = common.build_flattened_common_mode_couple(couple, m_cold)
    fixed_original = common._solve_terminal(couple, nominal_current_a)
    fixed_flattened = common._solve_terminal(flattened, nominal_current_a)
    original_capacity = common._optimized_capacity(couple)
    flattened_capacity = common._optimized_capacity(flattened)
    original_optimum = original_capacity["contact_corrected_optimum"]
    flattened_optimum = flattened_capacity["contact_corrected_optimum"]
    delta_q = float(
        original_optimum["Qc_after_contact_w"]
        - flattened_optimum["Qc_after_contact_w"]
    )
    return {
        "scenario_id": scenario_id,
        "family": family,
        "description": description,
        "error_coupling": error_coupling,
        "interpolation": interpolation,
        "ceiling_application": ceiling_application,
        "perturbation": perturbation or {},
        "property_summary": properties["summary"],
        "property_curves": properties,
        "fixed_nominal_current": {
            "current_a": nominal_current_a,
            "original_Qc_after_contact_w": float(
                fixed_original["Qc_after_contact_w"]
            ),
            "flattened_Qc_after_contact_w": float(
                fixed_flattened["Qc_after_contact_w"]
            ),
            "original_minus_flattened_Qc_w": float(
                fixed_original["Qc_after_contact_w"]
                - fixed_flattened["Qc_after_contact_w"]
            ),
        },
        "reoptimized": {
            "original_Qc_max_w": float(original_optimum["Qc_after_contact_w"]),
            "flattened_Qc_max_w": float(flattened_optimum["Qc_after_contact_w"]),
            "original_minus_flattened_Qc_max_w": delta_q,
            "original_optimum_current_a": float(original_optimum["current_a"]),
            "flattened_optimum_current_a": float(flattened_optimum["current_a"]),
            "original_energy_residual_w": float(original_optimum["energy_residual_w"]),
            "flattened_energy_residual_w": float(
                flattened_optimum["energy_residual_w"]
            ),
        },
    }


def _strip_property_curves(record: dict[str, Any]) -> dict[str, Any]:
    stripped = copy.deepcopy(record)
    stripped.pop("property_curves", None)
    return stripped


def _vector_b_seebeck_arrays(
    vector: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, tuple[FloatArray, FloatArray]]:
    output: dict[str, tuple[FloatArray, FloatArray]] = {}
    material_by_carrier = {
        "p": "Pb0.996Cu0.0004Se+0.001Cr",
        "n": "Pb0.996Cu0.0004Se+0.005Cr",
    }
    for carrier, material in material_by_carrier.items():
        points = [
            point
            for point in vector["points"]
            if point["key"]["property_id"] == "seebeck_coefficient"
            and point["key"]["carrier"] == carrier
            and point["key"]["material_id"] == material
        ]
        points.sort(key=lambda point: int(point["key"]["point_index"]))
        if len(points) != 7:
            raise ValueError(f"expected seven route-B Seebeck points for {carrier}")
        route_b_t = np.asarray(
            [float(point["extractor_B"]["temperature_K"]) for point in points]
        )
        route_b_s = np.asarray(
            [float(point["extractor_B"]["si_value"]) for point in points]
        )
        route_a_t, route_a_s = forward._fig1_arrays(inputs[f"{carrier}_seebeck"])
        selected_t = np.asarray(
            [
                float(point["source_object_candidate"]["temperature_K"])
                for point in points
            ]
        )
        selected_s = np.asarray(
            [float(point["source_object_candidate"]["si_value"]) for point in points]
        )
        np.testing.assert_allclose(route_a_t, selected_t, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(route_a_s, selected_s, rtol=0.0, atol=1.0e-15)
        output[carrier] = (route_b_t, route_b_s)
    return output


def _source_property(
    interpolation: str,
    temperature: FloatArray,
    values: FloatArray,
) -> PchipTemperatureProperty:
    if interpolation == "pchip":
        return PchipTemperatureProperty(temperature, values)
    if interpolation == "piecewise_linear":
        return PiecewiseLinearTemperatureProperty(temperature, values)
    if interpolation == "bound_checked_smooth_quadratic":
        return BoundCheckedQuadraticTemperatureProperty(temperature, values)
    raise ValueError(f"unsupported interpolation: {interpolation}")


def _full_support_normalization(
    p_temperature: FloatArray,
    n_temperature: FloatArray,
) -> tuple[float, float]:
    minimum = min(float(p_temperature[0]), float(n_temperature[0]))
    maximum = max(float(p_temperature[-1]), float(n_temperature[-1]))
    return 0.5 * (minimum + maximum), 0.5 * (maximum - minimum)


def _structured_seebeck_scenarios(
    base: TemperatureDependentNumericalCouple,
    p_temperature: FloatArray,
    p_values: FloatArray,
    n_temperature: FloatArray,
    n_values: FloatArray,
    nominal_current_a: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p_source = base.p_leg.seebeck
    n_source = base.n_leg.seebeck
    center, half_range = _full_support_normalization(p_temperature, n_temperature)
    dense_p = np.linspace(
        p_source.minimum_temperature_k,
        p_source.maximum_temperature_k,
        FULL_SUPPORT_GRID_POINTS,
    )
    dense_n = np.linspace(
        n_source.minimum_temperature_k,
        n_source.maximum_temperature_k,
        FULL_SUPPORT_GRID_POINTS,
    )
    p_minimum_abs = float(np.min(np.abs(p_source.evaluate(dense_p))))
    n_minimum_abs = float(np.min(np.abs(n_source.evaluate(dense_n))))
    common_additive_bound = SEEBECK_RELATIVE_METHOD_CEILING * min(
        p_minimum_abs, n_minimum_abs
    )
    p_additive_bound = SEEBECK_RELATIVE_METHOD_CEILING * p_minimum_abs
    n_additive_bound = SEEBECK_RELATIVE_METHOD_CEILING * n_minimum_abs

    records: list[dict[str, Any]] = []

    def affine_couple(
        *,
        p_gain: float = 0.0,
        n_gain: float = 0.0,
        p_gain_drift: float = 0.0,
        n_gain_drift: float = 0.0,
        p_offset: float = 0.0,
        n_offset: float = 0.0,
        p_additive_drift: float = 0.0,
        n_additive_drift: float = 0.0,
    ) -> TemperatureDependentNumericalCouple:
        p_property = AffinePerturbedSeebeckProperty(
            p_source,
            normalization_center_k=center,
            normalization_half_range_k=half_range,
            gain_constant_fraction=p_gain,
            gain_drift_fraction=p_gain_drift,
            additive_constant_v_per_k=p_offset,
            additive_drift_v_per_k=p_additive_drift,
        )
        n_property = AffinePerturbedSeebeckProperty(
            n_source,
            normalization_center_k=center,
            normalization_half_range_k=half_range,
            gain_constant_fraction=n_gain,
            gain_drift_fraction=n_gain_drift,
            additive_constant_v_per_k=n_offset,
            additive_drift_v_per_k=n_additive_drift,
        )
        return _build_couple_with_properties(base, p_property, n_property)

    def add(
        scenario_id: str,
        family: str,
        couple: TemperatureDependentNumericalCouple,
        description: str,
        coupling: str,
        perturbation: dict[str, Any],
    ) -> None:
        record = _effect_record(
            scenario_id,
            family,
            couple,
            base,
            nominal_current_a,
            description=description,
            error_coupling=coupling,
            interpolation="PCHIP source law with analytic perturbation wrapper",
            ceiling_application="bounded over the continuous common source support",
            perturbation=perturbation,
        )
        maximum_relative = max(
            record["property_summary"][
                "maximum_relative_p_Seebeck_deviation_on_common_support"
            ],
            record["property_summary"][
                "maximum_relative_n_Seebeck_deviation_on_common_support"
            ],
        )
        if maximum_relative > SEEBECK_RELATIVE_METHOD_CEILING + 2.0e-12:
            raise ValueError(f"structured scenario exceeds the 5% ceiling: {scenario_id}")
        records.append(record)

    for sign, label in ((-1.0, "minus"), (1.0, "plus")):
        add(
            f"common_constant_offset_{label}",
            "constant_offsets",
            affine_couple(
                p_offset=sign * common_additive_bound,
                n_offset=sign * common_additive_bound,
            ),
            "Equal constant additive shift of both Seebeck laws; exact Gamma-null control.",
            "perfectly shared additive offset",
            {"common_offset_v_per_k": sign * common_additive_bound},
        )

    for p_sign, n_sign in itertools.product((-1.0, 1.0), repeat=2):
        code = ("p" if p_sign > 0 else "m") + ("p" if n_sign > 0 else "m")
        add(
            f"leg_constant_offsets_{code}",
            "constant_offsets",
            affine_couple(
                p_offset=p_sign * p_additive_bound,
                n_offset=n_sign * n_additive_bound,
            ),
            "Independent legwise constant Seebeck offsets within each leg's 5% bound.",
            "same-sign" if p_sign == n_sign else "opposite-sign",
            {
                "p_offset_v_per_k": p_sign * p_additive_bound,
                "n_offset_v_per_k": n_sign * n_additive_bound,
            },
        )

    for p_sign, n_sign in itertools.product((-1.0, 1.0), repeat=2):
        code = ("p" if p_sign > 0 else "m") + ("p" if n_sign > 0 else "m")
        add(
            f"leg_gains_{code}",
            "constant_gains",
            affine_couple(
                p_gain=p_sign * SEEBECK_RELATIVE_METHOD_CEILING,
                n_gain=n_sign * SEEBECK_RELATIVE_METHOD_CEILING,
            ),
            "Constant multiplicative p/n Seebeck gains at the published ceiling.",
            "correlated" if p_sign == n_sign else "anticorrelated",
            {
                "p_gain_fraction": p_sign * SEEBECK_RELATIVE_METHOD_CEILING,
                "n_gain_fraction": n_sign * SEEBECK_RELATIVE_METHOD_CEILING,
            },
        )

    for p_sign, n_sign in itertools.product((-1.0, 1.0), repeat=2):
        code = ("p" if p_sign > 0 else "m") + ("p" if n_sign > 0 else "m")
        add(
            f"multiplicative_temperature_drifts_{code}",
            "multiplicative_temperature_drifts",
            affine_couple(
                p_gain_drift=p_sign * SEEBECK_RELATIVE_METHOD_CEILING,
                n_gain_drift=n_sign * SEEBECK_RELATIVE_METHOD_CEILING,
            ),
            "Linear-in-temperature multiplicative calibration drift across full support.",
            "correlated" if p_sign == n_sign else "anticorrelated",
            {
                "p_endpoint_gain_amplitude_fraction": p_sign
                * SEEBECK_RELATIVE_METHOD_CEILING,
                "n_endpoint_gain_amplitude_fraction": n_sign
                * SEEBECK_RELATIVE_METHOD_CEILING,
                "normalization_center_k": center,
                "normalization_half_range_k": half_range,
            },
        )

    for sign, label in ((-1.0, "minus"), (1.0, "plus")):
        add(
            f"common_additive_temperature_drift_{label}",
            "common_additive_temperature_drift",
            affine_couple(
                p_additive_drift=sign * common_additive_bound,
                n_additive_drift=sign * common_additive_bound,
            ),
            (
                "Smooth equal additive temperature drift; preserves alpha(T) exactly "
                "and perturbs only the common mode."
            ),
            "perfectly shared additive temperature drift",
            {
                "endpoint_additive_amplitude_v_per_k": sign
                * common_additive_bound,
                "normalization_center_k": center,
                "normalization_half_range_k": half_range,
            },
        )
        add(
            f"differential_additive_temperature_drift_{label}",
            "differential_additive_temperature_drift",
            affine_couple(
                p_additive_drift=sign * common_additive_bound,
                n_additive_drift=-sign * common_additive_bound,
            ),
            "Equal-and-opposite additive temperature drift; primarily perturbs alpha(T).",
            "perfectly anticorrelated additive temperature drift",
            {
                "p_endpoint_additive_amplitude_v_per_k": sign
                * common_additive_bound,
                "n_endpoint_additive_amplitude_v_per_k": -sign
                * common_additive_bound,
                "normalization_center_k": center,
                "normalization_half_range_k": half_range,
            },
        )

    drift_cache: dict[float, dict[str, Any]] = {}

    def drift_record(amplitude_fraction: float) -> dict[str, Any]:
        key = float(amplitude_fraction)
        if key not in drift_cache:
            couple = affine_couple(
                p_additive_drift=key * common_additive_bound,
                n_additive_drift=key * common_additive_bound,
            )
            drift_cache[key] = _effect_record(
                f"common_additive_drift_sweep_{key:+.6f}",
                "common_additive_drift_sweep",
                couple,
                base,
                nominal_current_a,
                description="One-dimensional alpha-preserving common-drift sweep.",
                error_coupling="perfectly shared additive temperature drift",
                interpolation="PCHIP source law with analytic perturbation wrapper",
                ceiling_application="bounded over the continuous common source support",
                perturbation={
                    "normalized_amplitude": key,
                    "endpoint_additive_amplitude_v_per_k": key
                    * common_additive_bound,
                },
            )
        return drift_cache[key]

    sweep_amplitudes = np.linspace(-1.0, 1.0, 9)
    sweep_records = [drift_record(float(amplitude)) for amplitude in sweep_amplitudes]

    def root_function(amplitude: float) -> float:
        return float(
            drift_record(float(amplitude))["reoptimized"][
                "original_minus_flattened_Qc_max_w"
            ]
        )

    if root_function(-1.0) * root_function(0.0) >= 0.0:
        raise RuntimeError("alpha-preserving drift cases do not bracket zero")
    zero_amplitude = float(
        brentq(root_function, -1.0, 0.0, xtol=2.0e-7, rtol=2.0e-7)
    )
    zero_record = drift_record(zero_amplitude)
    drift_summary = {
        "common_additive_bound_v_per_k": common_additive_bound,
        "normalization_center_k": center,
        "normalization_half_range_k": half_range,
        "sweep": [_strip_property_curves(record) for record in sweep_records],
        "zero_crossing_normalized_amplitude": zero_amplitude,
        "zero_crossing_endpoint_additive_amplitude_v_per_k": zero_amplitude
        * common_additive_bound,
        "zero_crossing_record": _strip_property_curves(zero_record),
    }
    return records, drift_summary


def _interpolation_and_digitization_records(
    base: TemperatureDependentNumericalCouple,
    inputs: dict[str, Any],
    vector: dict[str, Any],
    nominal_current_a: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p_temperature, p_values = forward._fig1_arrays(inputs["p_seebeck"])
    n_temperature, n_values = forward._fig1_arrays(inputs["n_seebeck"])
    records: list[dict[str, Any]] = []
    for interpolation in (
        "piecewise_linear",
        "bound_checked_smooth_quadratic",
    ):
        p_property = _source_property(interpolation, p_temperature, p_values)
        n_property = _source_property(interpolation, n_temperature, n_values)
        couple = _build_couple_with_properties(base, p_property, n_property)
        records.append(
            _effect_record(
                interpolation,
                "interpolation_choice",
                couple,
                base,
                nominal_current_a,
                description=(
                    "Alternative deterministic representation of the same seven "
                    "published Seebeck markers."
                ),
                error_coupling="not applicable",
                interpolation=interpolation,
                ceiling_application=(
                    "linear passes through markers; smooth quadratic is accepted only "
                    "if every marker residual is below 5% and both trends stay monotone"
                ),
            )
        )

    route_b = _vector_b_seebeck_arrays(vector, inputs)
    route_b_couple = _build_couple_with_properties(
        base,
        PchipTemperatureProperty(*route_b["p"]),
        PchipTemperatureProperty(*route_b["n"]),
    )
    route_b_record = _effect_record(
        "independent_vector_extractor_B",
        "digitization_route",
        route_b_couple,
        base,
        nominal_current_a,
        description="Independent SVG vector-route Seebeck coordinates; sigma/kappa unchanged.",
        error_coupling="digitization route, not measurement covariance",
        interpolation="PCHIP",
        ceiling_application="no measurement ceiling applied",
    )
    records.append(route_b_record)
    route_differences = {}
    for carrier in ("p", "n"):
        route_a_t, route_a_s = forward._fig1_arrays(inputs[f"{carrier}_seebeck"])
        route_b_t, route_b_s = route_b[carrier]
        route_differences[carrier] = {
            "maximum_temperature_difference_k": float(
                np.max(np.abs(route_a_t - route_b_t))
            ),
            "maximum_Seebeck_difference_v_per_k": float(
                np.max(np.abs(route_a_s - route_b_s))
            ),
            "maximum_relative_Seebeck_difference": float(
                np.max(np.abs(route_a_s - route_b_s) / np.abs(route_a_s))
            ),
        }
    return records, {
        "route_A_definition": "direct source-object conversion fixed in advance",
        "route_B_definition": "independent Poppler-SVG vector path conversion",
        "route_differences": route_differences,
        "route_B_effect_record": _strip_property_curves(route_b_record),
        "measurement_uncertainty_combined": False,
    }


def _pointwise_corner_screen(
    base: TemperatureDependentNumericalCouple,
    inputs: dict[str, Any],
    nominal_current_a: float,
) -> dict[str, Any]:
    arrays = {
        carrier: forward._fig1_arrays(inputs[f"{carrier}_seebeck"])
        for carrier in ("p", "n")
    }
    valid: dict[str, list[tuple[tuple[int, ...], FloatArray]]] = {
        "p": [],
        "n": [],
    }
    for carrier in ("p", "n"):
        _, nominal = arrays[carrier]
        for signs in itertools.product((-1, 1), repeat=LOCALLY_ACTIVE_PCHIP_KNOTS_PER_LEG):
            candidate = nominal.copy()
            candidate[:LOCALLY_ACTIVE_PCHIP_KNOTS_PER_LEG] *= 1.0 + (
                SEEBECK_RELATIVE_METHOD_CEILING * np.asarray(signs, dtype=float)
            )
            if carrier == "p":
                accepted = bool(
                    np.all(candidate > 0.0) and np.all(np.diff(candidate) > 0.0)
                )
            else:
                accepted = bool(
                    np.all(candidate < 0.0) and np.all(np.diff(candidate) < 0.0)
                )
            if accepted:
                valid[carrier].append((tuple(int(value) for value in signs), candidate))

    records: list[dict[str, Any]] = []
    for p_signs, p_values in valid["p"]:
        for n_signs, n_values in valid["n"]:
            p_property = PchipTemperatureProperty(arrays["p"][0], p_values)
            n_property = PchipTemperatureProperty(arrays["n"][0], n_values)
            couple = _build_couple_with_properties(base, p_property, n_property)
            p_code = "".join("p" if sign > 0 else "m" for sign in p_signs)
            n_code = "".join("p" if sign > 0 else "m" for sign in n_signs)
            record = _effect_record(
                f"monotone_corner_p{p_code}_n{n_code}",
                "monotone_pointwise_corners",
                couple,
                base,
                nominal_current_a,
                description=(
                    "Monotone PCHIP corner over the first four locally active "
                    "Seebeck markers in each leg."
                ),
                error_coupling="unknown point-to-point and p/n coupling",
                interpolation="PCHIP",
                ceiling_application=(
                    "+/-5% at published markers; later three PCHIP knots fixed because "
                    "they cannot affect the 309.894--362.998 K interval"
                ),
                perturbation={
                    "p_first_four_signs": list(p_signs),
                    "n_first_four_signs": list(n_signs),
                },
            )
            if not record["property_summary"]["p_Seebeck_monotone_increasing"]:
                raise RuntimeError("accepted p corner lost monotonicity")
            if not record["property_summary"]["n_Seebeck_monotone_decreasing"]:
                raise RuntimeError("accepted n corner lost monotonicity")
            records.append(record)

    records.sort(
        key=lambda record: record["reoptimized"][
            "original_minus_flattened_Qc_max_w"
        ]
    )
    minimum = records[0]
    maximum = records[-1]
    return {
        "raw_binary_corner_count_before_monotonicity": 2
        ** (2 * LOCALLY_ACTIVE_PCHIP_KNOTS_PER_LEG),
        "valid_p_corner_count": len(valid["p"]),
        "valid_n_corner_count": len(valid["n"]),
        "evaluated_pair_corner_count": len(records),
        "locally_active_knot_indices_one_based": list(
            range(1, LOCALLY_ACTIVE_PCHIP_KNOTS_PER_LEG + 1)
        ),
        "later_knot_rule": (
            "PCHIP locality makes knots 5--7 inactive below the 362.998 K endpoint; "
            "they remain at nominal values"
        ),
        "continuous_interior_global_extrema_certified": False,
        "complete_monotone_binary_corner_enumeration": True,
        "reoptimized_effect_envelope_w": [
            float(
                minimum["reoptimized"]["original_minus_flattened_Qc_max_w"]
            ),
            float(
                maximum["reoptimized"]["original_minus_flattened_Qc_max_w"]
            ),
        ],
        "minimum_case": _strip_property_curves(minimum),
        "maximum_case": _strip_property_curves(maximum),
        "records": [_strip_property_curves(record) for record in records],
    }


def _branch_transfer_condition_screen(
    base: TemperatureDependentNumericalCouple,
    nominal_current_a: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for sigma_p_sign, sigma_n_sign, kappa_p_sign, kappa_n_sign in itertools.product(
        (-1.0, 1.0), repeat=4
    ):
        p_sigma = 1.0 + sigma_p_sign * SIGMA_RELATIVE_METHOD_CEILING
        n_sigma = 1.0 + sigma_n_sign * SIGMA_RELATIVE_METHOD_CEILING
        p_kappa = 1.0 + kappa_p_sign * KAPPA_RELATIVE_METHOD_CEILING
        n_kappa = 1.0 + kappa_n_sign * KAPPA_RELATIVE_METHOD_CEILING
        code = "".join(
            "p" if sign > 0 else "m"
            for sign in (
                sigma_p_sign,
                sigma_n_sign,
                kappa_p_sign,
                kappa_n_sign,
            )
        )
        couple = _build_transport_scaled_couple(
            base,
            p_sigma_scale=p_sigma,
            n_sigma_scale=n_sigma,
            p_kappa_scale=p_kappa,
            n_kappa_scale=n_kappa,
        )
        record = _effect_record(
            f"transport_corner_{code}",
            "branch_transfer_sigma_kappa_corners",
            couple,
            base,
            nominal_current_a,
            description=(
                "Nominal Seebeck laws with a structured sigma/kappa corner; this "
                "tests branch-to-port transfer, not Seebeck identifiability."
            ),
            error_coupling="unknown p/n transport coupling",
            interpolation="nominal PCHIP Seebeck",
            ceiling_application=(
                "sigma +/-5% and kappa +/-15% applied separately; no joint "
                "probability or combination with Seebeck errors"
            ),
            perturbation={
                "p_sigma_scale": p_sigma,
                "n_sigma_scale": n_sigma,
                "p_kappa_scale": p_kappa,
                "n_kappa_scale": n_kappa,
            },
        )
        records.append(record)
    records.sort(
        key=lambda record: record["reoptimized"][
            "original_minus_flattened_Qc_max_w"
        ]
    )
    return {
        "corner_count": len(records),
        "seebeck_perturbed": False,
        "sigma_relative_method_ceiling": SIGMA_RELATIVE_METHOD_CEILING,
        "kappa_relative_method_ceiling": KAPPA_RELATIVE_METHOD_CEILING,
        "pooled_with_seebeck_identifiability": False,
        "statistical_interval": False,
        "reoptimized_effect_envelope_w": [
            float(
                records[0]["reoptimized"]["original_minus_flattened_Qc_max_w"]
            ),
            float(
                records[-1]["reoptimized"]["original_minus_flattened_Qc_max_w"]
            ),
        ],
        "minimum_case": _strip_property_curves(records[0]),
        "maximum_case": _strip_property_curves(records[-1]),
        "records": [_strip_property_curves(record) for record in records],
    }


def _single_knot_sensitivities(
    base: TemperatureDependentNumericalCouple,
    inputs: dict[str, Any],
    nominal_current_a: float,
) -> list[dict[str, Any]]:
    arrays = {
        carrier: forward._fig1_arrays(inputs[f"{carrier}_seebeck"])
        for carrier in ("p", "n")
    }

    def fixed_effect(p_values: FloatArray, n_values: FloatArray) -> float:
        couple = _build_couple_with_properties(
            base,
            PchipTemperatureProperty(arrays["p"][0], p_values),
            PchipTemperatureProperty(arrays["n"][0], n_values),
        )
        cold = couple.cold_temperature_k
        m_cold = 0.5 * (
            float(couple.p_leg.seebeck.evaluate([cold])[0])
            + float(couple.n_leg.seebeck.evaluate([cold])[0])
        )
        flattened = common.build_flattened_common_mode_couple(couple, m_cold)
        return float(
            common._solve_terminal(couple, nominal_current_a)["Qc_after_contact_w"]
            - common._solve_terminal(flattened, nominal_current_a)[
                "Qc_after_contact_w"
            ]
        )

    records = []
    for carrier in ("p", "n"):
        for index in range(7):
            plus_p = arrays["p"][1].copy()
            minus_p = arrays["p"][1].copy()
            plus_n = arrays["n"][1].copy()
            minus_n = arrays["n"][1].copy()
            if carrier == "p":
                plus_p[index] *= 1.0 + SENSITIVITY_RELATIVE_STEP
                minus_p[index] *= 1.0 - SENSITIVITY_RELATIVE_STEP
            else:
                plus_n[index] *= 1.0 + SENSITIVITY_RELATIVE_STEP
                minus_n[index] *= 1.0 - SENSITIVITY_RELATIVE_STEP
            plus_effect = fixed_effect(plus_p, plus_n)
            minus_effect = fixed_effect(minus_p, minus_n)
            derivative_w_per_unit_fraction = (
                plus_effect - minus_effect
            ) / (2.0 * SENSITIVITY_RELATIVE_STEP)
            records.append(
                {
                    "carrier": carrier,
                    "knot_index_one_based": index + 1,
                    "temperature_k": float(arrays[carrier][0][index]),
                    "fixed_current_effect_derivative_w_per_unit_relative_knot_change": float(
                        derivative_w_per_unit_fraction
                    ),
                    "effect_change_mw_per_one_percent_relative_knot_change": float(
                        1.0e3 * 0.01 * derivative_w_per_unit_fraction
                    ),
                    "locally_active_for_target_interval": index
                    < LOCALLY_ACTIVE_PCHIP_KNOTS_PER_LEG,
                }
            )
    records.sort(
        key=lambda record: abs(
            record["effect_change_mw_per_one_percent_relative_knot_change"]
        ),
        reverse=True,
    )
    return records


def analyze_identifiability() -> dict[str, Any]:
    inputs, validation = _validate_and_load_inputs()
    target = inputs["target"]
    hot = float(target["hot_side_temperature_k"])
    cold = hot - float(target["delta_t_max_k"])
    base = forward.build_couple(inputs, forward.SCENARIOS["nominal"], cold, hot)

    nominal_capacity = common._optimized_capacity(base)
    nominal_current = float(
        nominal_capacity["contact_corrected_optimum"]["current_a"]
    )
    nominal = _effect_record(
        "nominal_pchip",
        "nominal",
        base,
        base,
        nominal_current,
        description="Unperturbed figure-derived PCHIP candidate.",
        error_coupling="not applicable",
        interpolation="PCHIP",
        ceiling_application="none",
    )
    nominal_effect = float(
        nominal["reoptimized"]["original_minus_flattened_Qc_max_w"]
    )
    if not math.isclose(
        nominal_effect,
        EXPECTED_NOMINAL_EFFECT_W,
        rel_tol=0.0,
        abs_tol=5.0e-13,
    ):
        raise ValueError("nominal 3.066878 mW result was not reproduced")

    p_temperature, p_values = forward._fig1_arrays(inputs["p_seebeck"])
    n_temperature, n_values = forward._fig1_arrays(inputs["n_seebeck"])
    structured, drift_summary = _structured_seebeck_scenarios(
        base,
        p_temperature,
        p_values,
        n_temperature,
        n_values,
        nominal_current,
    )
    representation_records, digitization = _interpolation_and_digitization_records(
        base, inputs, validation["vector"], nominal_current
    )
    corner_screen = _pointwise_corner_screen(base, inputs, nominal_current)
    branch_transfer = _branch_transfer_condition_screen(base, nominal_current)
    sensitivities = _single_knot_sensitivities(base, inputs, nominal_current)

    all_seebeck_records = [nominal, *structured, *representation_records]
    seebeck_effects = [
        float(record["reoptimized"]["original_minus_flattened_Qc_max_w"])
        for record in all_seebeck_records
    ] + [
        float(record["reoptimized"]["original_minus_flattened_Qc_max_w"])
        for record in corner_screen["records"]
    ]
    structured_case_min = min(seebeck_effects)
    structured_case_max = max(seebeck_effects)

    by_id = {record["scenario_id"]: record for record in structured}
    drift_minus = by_id["common_additive_temperature_drift_minus"]
    drift_plus = by_id["common_additive_temperature_drift_plus"]
    for case in (drift_minus, drift_plus):
        if (
            case["property_summary"][
                "maximum_alpha_change_from_nominal_v_per_k"
            ]
            > 2.0e-15
        ):
            raise RuntimeError("common additive drift did not preserve alpha")
    if not (
        drift_minus["reoptimized"]["original_minus_flattened_Qc_max_w"] < 0.0
        < drift_plus["reoptimized"]["original_minus_flattened_Qc_max_w"]
    ):
        raise RuntimeError("smooth admissible cases did not reverse the sign")

    constant_common_records = [
        record
        for record in structured
        if record["scenario_id"].startswith("common_constant_offset_")
    ]
    maximum_constant_offset_effect_change = max(
        abs(
            float(
                record["reoptimized"]["original_minus_flattened_Qc_max_w"]
            )
            - nominal_effect
        )
        for record in constant_common_records
    )

    raw_data_priorities = [
        {
            "rank": 1,
            "request": (
                "Paired same-run raw p- and n-leg Seebeck voltage/DeltaT records and "
                "replicates at 323 and 373 K, including their cross-leg covariance"
            ),
            "reason": (
                "The four largest local response sensitivities are the p/n 323- and "
                "373-K knots; opposite sensitivity signs make their covariance decisive."
            ),
        },
        {
            "rank": 2,
            "request": (
                "ZEM-3 reference-material, thermocouple, polarity-reversal, and "
                "temperature-dependent calibration-drift records for each run"
            ),
            "reason": (
                "An equal additive drift preserves alpha but changes Gamma and is already "
                "sufficient to reverse the modeled port-effect sign within the method ceiling."
            ),
        },
        {
            "rank": 3,
            "request": (
                "Dense numeric S_p(T), S_n(T) tables over 300--375 K with specimen IDs, "
                "repeat counts, temperature errors, and covariance rather than plotted markers"
            ),
            "reason": (
                "The target temperature field lies entirely in this interval; later knots "
                "do not influence the PCHIP response."
            ),
        },
        {
            "rank": 4,
            "request": (
                "A traceable mapping between the measured transport specimens and the "
                "actual p/n legs used in the seven-pair device"
            ),
            "reason": (
                "Without sample identity the best possible curve-level calibration still "
                "does not validate the material-specific device attribution."
            ),
        },
    ]

    maximum_energy_residual = max(
        abs(float(record["reoptimized"][key]))
        for record in [nominal, *structured, *representation_records]
        for key in ("original_energy_residual_w", "flattened_energy_residual_w")
    )
    maximum_energy_residual = max(
        maximum_energy_residual,
        max(
            abs(float(record["reoptimized"][key]))
            for record in corner_screen["records"]
            for key in ("original_energy_residual_w", "flattened_energy_residual_w")
        ),
        max(
            abs(float(record["reoptimized"][key]))
            for record in branch_transfer["records"]
            for key in ("original_energy_residual_w", "flattened_energy_residual_w")
        ),
    )

    return {
        "_internal": {
            "nominal_record": nominal,
            "structured_records": structured,
            "representation_records": representation_records,
            "drift_minus_record": drift_minus,
            "drift_plus_record": drift_plus,
        },
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "title": "Identifiability of the PbSe/Cr common-mode Thomson response",
        "central_scientific_result": (
            "The public below-5% Seebeck method ceiling does not identify the sign "
            "or order of magnitude of the nominal 3.067 mW common-mode response. "
            "Two smooth, monotone, alpha-preserving equal-additive drift cases "
            "inside that ceiling give opposite-signed reoptimized port responses. "
            "The result must therefore remain a nominal scenario until paired raw "
            "Seebeck slopes and their calibration/covariance records are obtained."
        ),
        "scope": {
            "statistical_confidence_interval": False,
            "probability_distribution_assumed": False,
            "p_n_covariance_assumed": False,
            "structured_admissible_case_envelope": True,
            "continuous_all_function_worst_case_certified": False,
            "real_pbse_cr_device_validation": False,
            "figure_derived_candidate_scenario_screen": True,
            "public_data_identify_nominal_effect_sign": False,
            "public_data_identify_nominal_effect_order_of_magnitude": False,
        },
        "target_condition": {
            "device_id": forward.TARGET_DEVICE_ID,
            "hot_temperature_k": hot,
            "cold_temperature_k": cold,
            "delta_t_k": hot - cold,
            "nominal_optimum_current_a": nominal_current,
            "below_300_k_extrapolation_used": False,
        },
        "measurement_limits": {
            "Seebeck_relative_method_ceiling": SEEBECK_RELATIVE_METHOD_CEILING,
            "electrical_conductivity_relative_method_ceiling": SIGMA_RELATIVE_METHOD_CEILING,
            "thermal_conductivity_relative_method_ceiling": KAPPA_RELATIVE_METHOD_CEILING,
            "ceilings_interpreted_as_standard_deviations": False,
            "ceilings_interpreted_as_confidence_intervals": False,
            "digitization_and_measurement_combined": False,
            "sigma_kappa_and_Seebeck_pooled": False,
        },
        "nominal_result": _strip_property_curves(nominal),
        "seebeck_identifiability": {
            "evaluated_admissible_case_envelope_w": [
                float(structured_case_min),
                float(structured_case_max),
            ],
            "envelope_contains_zero": bool(
                structured_case_min <= 0.0 <= structured_case_max
            ),
            "sign_identified": False,
            "order_of_magnitude_identified": False,
            "decisive_alpha_preserving_cases": {
                "negative": _strip_property_curves(drift_minus),
                "positive": _strip_property_curves(drift_plus),
            },
            "constant_common_offset_maximum_effect_change_w": float(
                maximum_constant_offset_effect_change
            ),
            "interpretation": (
                "A constant common offset leaves Gamma unchanged and is a null control. "
                "The unresolved temperature-dependent common calibration drift, not "
                "the absolute offset alone, destroys material-specific identifiability."
            ),
        },
        "structured_Seebeck_scenarios": [
            _strip_property_curves(record) for record in structured
        ],
        "alpha_preserving_common_drift_sweep": drift_summary,
        "interpolation_and_digitization": {
            "records": [
                _strip_property_curves(record) for record in representation_records
            ],
            "digitization_only": digitization,
        },
        "monotone_pointwise_corner_screen": corner_screen,
        "branch_transfer_condition_sensitivity": branch_transfer,
        "single_knot_local_sensitivity": {
            "finite_difference_relative_step": SENSITIVITY_RELATIVE_STEP,
            "fixed_current_a": nominal_current,
            "records_ranked_by_absolute_effect": sensitivities,
            "top_four_all_at_323_or_373_k": bool(
                all(315.0 < record["temperature_k"] < 380.0 for record in sensitivities[:4])
            ),
        },
        "raw_data_priorities": raw_data_priorities,
        "verification": {
            "nominal_effect_reproduction_error_w": abs(
                nominal_effect - EXPECTED_NOMINAL_EFFECT_W
            ),
            "smooth_negative_case_alpha_change_v_per_k": drift_minus[
                "property_summary"
            ]["maximum_alpha_change_from_nominal_v_per_k"],
            "smooth_positive_case_alpha_change_v_per_k": drift_plus[
                "property_summary"
            ]["maximum_alpha_change_from_nominal_v_per_k"],
            "maximum_constant_common_offset_effect_change_w": maximum_constant_offset_effect_change,
            "maximum_optimized_module_energy_residual_w": maximum_energy_residual,
            "all_smooth_cases_monotone": bool(
                all(
                    record["property_summary"]["p_Seebeck_monotone_increasing"]
                    and record["property_summary"]["n_Seebeck_monotone_decreasing"]
                    for record in (drift_minus, drift_plus)
                )
            ),
            "all_smooth_cases_inside_continuous_5_percent_ceiling": bool(
                all(
                    max(
                        record["property_summary"][
                            "maximum_relative_p_Seebeck_deviation_on_common_support"
                        ],
                        record["property_summary"][
                            "maximum_relative_n_Seebeck_deviation_on_common_support"
                        ],
                    )
                    <= SEEBECK_RELATIVE_METHOD_CEILING + 2.0e-12
                    for record in (drift_minus, drift_plus)
                )
            ),
        },
        "input_bindings": validation["bindings"],
    }


def _family_envelope(records: list[dict[str, Any]]) -> tuple[float, float]:
    values = [
        1.0e3
        * float(record["reoptimized"]["original_minus_flattened_Qc_max_w"])
        for record in records
    ]
    return min(values), max(values)


def make_figure(analysis: dict[str, Any], output_prefix: Path) -> list[Path]:
    """Render the identifiability argument as a double-column quantitative grid.

    Figure design
    -------------
    Core conclusion: the public Seebeck ceiling admits smooth alpha-preserving
    candidates with opposite port-effect signs, so 3.07 mW is not identified.
    Evidence chain: (a) M(T) perturbation curves and pointwise ceiling; (b) Gamma(T)
    and interpolation dependence; (c) structured response envelopes; (d) the
    common-drift zero crossing and broader scenario relation.
    Layout: quantitative grid emphasizing panel c.
    Export: 183-mm double-column SVG/PDF plus 600-dpi TIFF and PNG; all source
    data are embedded in the result JSON. Intervals are deterministic response
    ranges, not probabilities.
    Limitations: no covariance or raw repeat data exist, and the pointwise
    corner range is not a certified global continuous-function extremum.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.1,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
        }
    )
    navy = "#17324D"
    blue = "#4C78A8"
    orange = "#E28E2C"
    red = "#C44E52"
    green = "#4F8C6B"
    purple = "#7A6FA8"
    grey = "#8A9099"
    pale = "#DDE6ED"

    internal = analysis["_internal"]
    nominal = internal["nominal_record"]
    drift_minus = internal["drift_minus_record"]
    drift_plus = internal["drift_plus_record"]
    representation = internal["representation_records"]
    structured = internal["structured_records"]

    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.55))
    fig.subplots_adjust(
        left=0.085,
        right=0.975,
        bottom=0.125,
        top=0.885,
        wspace=0.34,
        hspace=0.46,
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    temperature = np.asarray(
        nominal["property_curves"]["temperature_k"], dtype=float
    )
    nominal_m = 1.0e6 * np.asarray(
        nominal["property_curves"]["common_mode_v_per_k"], dtype=float
    )
    base_sp = 1.0e6 * np.asarray(
        nominal["property_curves"]["alpha_v_per_k"], dtype=float
    )
    # Recover Sp and Sn from alpha and M for an exact pointwise independent band.
    sp = nominal_m + 0.5 * base_sp
    sn = nominal_m - 0.5 * base_sp
    m_half_width = 0.5 * SEEBECK_RELATIVE_METHOD_CEILING * (
        np.abs(sp) + np.abs(sn)
    )
    ax_a.fill_between(
        temperature,
        nominal_m - m_half_width,
        nominal_m + m_half_width,
        color=pale,
        alpha=0.8,
        label="independent pointwise 5% ceiling",
    )
    ax_a.axhline(0.0, color="#B7BCC2", linewidth=0.8)
    ax_a.plot(temperature, nominal_m, color=navy, linewidth=1.8, label="nominal")
    for record, color, label in (
        (drift_minus, red, "negative-response smooth case"),
        (drift_plus, green, "positive-response smooth case"),
    ):
        ax_a.plot(
            temperature,
            1.0e6
            * np.asarray(record["property_curves"]["common_mode_v_per_k"]),
            color=color,
            linewidth=1.5,
            label=label,
        )
    ax_a.set(
        xlabel="temperature (K)",
        ylabel=r"common mode $M$ ($\mu$V K$^{-1}$)",
    )
    ax_a.set_title(r"The reported ceiling is wider than nominal $M(T)$", loc="left")
    ax_a.legend(loc="upper left", ncol=1)

    ax_b.axhline(0.0, color="#B7BCC2", linewidth=0.8)
    ax_b.plot(
        temperature,
        1.0e6 * np.asarray(nominal["property_curves"]["gamma_v_per_k"]),
        color=navy,
        linewidth=1.8,
        label="PCHIP nominal",
    )
    interpolation_colors = {
        "piecewise_linear": blue,
        "bound_checked_smooth_quadratic": purple,
    }
    interpolation_labels = {
        "piecewise_linear": "piecewise linear",
        "bound_checked_smooth_quadratic": "smooth quadratic",
    }
    for record in representation:
        if record["scenario_id"] not in interpolation_colors:
            continue
        ax_b.plot(
            temperature,
            1.0e6 * np.asarray(record["property_curves"]["gamma_v_per_k"]),
            color=interpolation_colors[record["scenario_id"]],
            linewidth=1.2,
            linestyle="--",
            label=interpolation_labels[record["scenario_id"]],
        )
    for record, color, label in (
        (drift_minus, red, "shared drift negative case"),
        (drift_plus, green, "shared drift positive case"),
    ):
        ax_b.plot(
            temperature,
            1.0e6 * np.asarray(record["property_curves"]["gamma_v_per_k"]),
            color=color,
            linewidth=1.5,
            label=label,
        )
    ax_b.set(
        xlabel="temperature (K)",
        ylabel=r"$\Gamma=T\,dM/dT$ ($\mu$V K$^{-1}$)",
    )
    ax_b.set_title(r"An unresolved common drift changes $\Gamma$ itself", loc="left")
    ax_b.legend(loc="upper left", ncol=1)

    nominal_mw = 1.0e3 * float(
        nominal["reoptimized"]["original_minus_flattened_Qc_max_w"]
    )
    family_rows: list[tuple[str, list[dict[str, Any]], str]] = []
    digitization_records = [
        nominal,
        *[
            record
            for record in representation
            if record["family"] == "digitization_route"
        ],
    ]
    family_rows.append(("vector route", digitization_records, grey))
    family_rows.append(
        (
            "interpolation",
            [
                nominal,
                *[
                    record
                    for record in representation
                    if record["family"] == "interpolation_choice"
                ],
            ],
            blue,
        )
    )
    family_rows.append(
        (
            "constant offsets",
            [nominal, *[r for r in structured if r["family"] == "constant_offsets"]],
            purple,
        )
    )
    family_rows.append(
        (
            "gain + drift",
            [
                nominal,
                *[
                    r
                    for r in structured
                    if r["family"]
                    in {
                        "constant_gains",
                        "multiplicative_temperature_drifts",
                        "differential_additive_temperature_drift",
                    }
                ],
            ],
            orange,
        )
    )
    family_rows.append(
        (
            r"shared drift ($\alpha$ fixed)",
            [nominal, drift_minus, drift_plus],
            green,
        )
    )
    corner_records = analysis["monotone_pointwise_corner_screen"]["records"]
    family_rows.append(("monotone 5% knot corners", corner_records, red))
    transport_records = analysis["branch_transfer_condition_sensitivity"]["records"]
    family_rows.append((r"$\sigma,\kappa$ transfer only", transport_records, navy))

    ax_c.axvline(0.0, color="#AEB4BA", linewidth=0.9)
    ax_c.axvline(nominal_mw, color=navy, linewidth=0.9, linestyle=":")
    for y, (label, records, color) in enumerate(family_rows):
        lower, upper = _family_envelope(records)
        ax_c.hlines(y, lower, upper, color=color, linewidth=2.6)
        ax_c.scatter([lower, upper], [y, y], color=color, s=18, zorder=3)
        if lower <= 0.0 <= upper:
            ax_c.scatter([0.0], [y], marker="x", color="#20262C", s=22, zorder=4)
    ax_c.set_yticks(np.arange(len(family_rows)), [row[0] for row in family_rows])
    ax_c.invert_yaxis()
    ax_c.set_xlabel(r"reoptimized $Q_c^{original}-Q_c^{flat}$ (mW)")
    ax_c.set_title("Seebeck shapes alone already reverse the sign", loc="left")

    scatter_groups = [
        (
            "structured",
            structured,
            orange,
            "o",
            17,
        ),
        (
            "monotone corners",
            corner_records,
            red,
            ".",
            19,
        ),
        (
            r"$\sigma,\kappa$ transfer",
            transport_records,
            blue,
            "s",
            12,
        ),
    ]
    ax_d.axhline(0.0, color="#AEB4BA", linewidth=0.8)
    ax_d.axvline(0.0, color="#AEB4BA", linewidth=0.8)
    for label, records, color, marker, size in scatter_groups:
        x = [1.0e6 * float(r["property_summary"]["M_change_v_per_k"]) for r in records]
        y = [
            1.0e3
            * float(r["reoptimized"]["original_minus_flattened_Qc_max_w"])
            for r in records
        ]
        ax_d.scatter(x, y, color=color, marker=marker, s=size, alpha=0.68, label=label)
    for record, color, label, offset, horizontal_alignment in (
        (drift_minus, red, "negative smooth case", (-7, -18), "right"),
        (drift_plus, green, "positive smooth case", (7, 9), "left"),
    ):
        x = 1.0e6 * float(record["property_summary"]["M_change_v_per_k"])
        y = 1.0e3 * float(
            record["reoptimized"]["original_minus_flattened_Qc_max_w"]
        )
        ax_d.scatter(
            [x], [y], s=34, color=color, edgecolor="white", linewidth=0.6, zorder=5
        )
        ax_d.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.8,
            color=color,
            ha=horizontal_alignment,
        )
    ax_d.set(
        xlabel=r"$M(T_h)-M(T_c)$ ($\mu$V K$^{-1}$)",
        ylabel=r"reoptimized $\Delta Q_c$ (mW)",
    )
    ax_d.set_title("The terminal residue follows the unresolved common-mode slope", loc="left")
    ax_d.legend(loc="lower right")

    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        axis.text(
            -0.13,
            1.08,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8.8,
            va="top",
        )
        axis.tick_params(direction="out", length=2.8, width=0.65)

    envelope = analysis["seebeck_identifiability"][
        "evaluated_admissible_case_envelope_w"
    ]
    fig.suptitle(
        "Public Seebeck curves do not resolve the 3.07 mW PbSe/Cr common-mode response",
        x=0.085,
        y=0.965,
        ha="left",
        fontsize=10.3,
        fontweight="bold",
        color=navy,
    )
    fig.text(
        0.085,
        0.035,
        "Deterministic admissible-case analysis, not a confidence interval. "
        f"Evaluated Seebeck envelope: {1.0e3*envelope[0]:+.2f} to "
        f"{1.0e3*envelope[1]:+.2f} mW. Sigma/kappa corners are shown only as a "
        "separate branch-transfer sensitivity and are not pooled with Seebeck errors.",
        ha="left",
        va="bottom",
        fontsize=5.8,
        color="#555B63",
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_prefix.with_suffix(".png"),
        output_prefix.with_suffix(".svg"),
        output_prefix.with_suffix(".pdf"),
        output_prefix.with_suffix(".tiff"),
    ]
    fig.savefig(
        outputs[0],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "matplotlib; PbSe Gamma identifiability"},
    )
    fig.savefig(
        outputs[1],
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": "2026-08-26", "Creator": None},
    )
    fig.savefig(
        outputs[2],
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": "PbSe Gamma identifiability",
            "Author": "research analysis",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        outputs[3],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return outputs


def serialize_results(
    analysis: dict[str, Any],
    figure_paths: list[Path],
) -> dict[str, Any]:
    result = copy.deepcopy(analysis)
    result.pop("_internal")
    result["figure_metadata"] = {
        "core_conclusion": (
            "Smooth alpha-preserving Seebeck candidates inside the published method "
            "ceiling produce opposite port-effect signs; 3.07 mW is not identified."
        ),
        "evidence_chain": [
            "M(T) pointwise ceiling and smooth alpha-preserving cases",
            "Gamma(T) under interpolation and common temperature drift",
            "separated deterministic response envelopes",
            "M(Th)-M(Tc) versus reoptimized terminal residue",
        ],
        "layout": "quantitative grid with an emphasized identifiability envelope",
        "backend": "Python matplotlib only",
        "export": "183-mm double-column SVG/PDF plus 600-dpi PNG/TIFF",
        "source_data": "embedded in this JSON",
        "statistics": (
            "deterministic admissible cases and finite-difference sensitivities; "
            "no confidence interval, p-value, probability, or inferred covariance"
        ),
        "limitation": (
            "public data lack raw repeats and covariance; the monotone corner screen "
            "is complete over binary locally active corners but not a certified global "
            "extremum over all continuous error functions"
        ),
    }
    result["outputs"] = {
        "analysis_script": output_locator(Path(__file__).resolve()),
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
        "figures": {
            path.suffix.lstrip("."): {
                "locator": output_locator(path),
                "sha256": file_sha256(path),
            }
            for path in figure_paths
        },
    }
    result["software_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    return result


def run_analysis(json_output: Path, figure_prefix: Path) -> dict[str, Any]:
    analysis = analyze_identifiability()
    figure_paths = make_figure(analysis, figure_prefix)
    result = serialize_results(analysis, figure_paths)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=DEFAULT_FIGURE_PREFIX,
        help="output path without suffix; PNG, SVG, PDF, and TIFF are written",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    result = run_analysis(arguments.json_output, arguments.figure_prefix)
    nominal = result["nominal_result"]["reoptimized"][
        "original_minus_flattened_Qc_max_w"
    ]
    envelope = result["seebeck_identifiability"][
        "evaluated_admissible_case_envelope_w"
    ]
    print(
        "PbSe Gamma identifiability complete: "
        f"nominal={1.0e3*nominal:.6f} mW; evaluated Seebeck case "
        f"envelope=[{1.0e3*envelope[0]:+.6f}, {1.0e3*envelope[1]:+.6f}] mW; "
        "sign_identified=false"
    )


if __name__ == "__main__":
    main()
