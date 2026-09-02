#!/usr/bin/env python3
"""Forward-constrain the PbSe/Cr 363 K source-Fig.-4a zero-load endpoint.

This analysis asks a direct device-physics question: do the public, selected
PbSe/Cr transport candidates and the nominal seven-couple, 2 x 2 x 6 mm
geometry close the highest-temperature source-Fig.-4a ``DeltaT_max`` point?

The selected point has ``T_h=362.998047 K`` and ``DeltaT=53.103912 K``, hence
``T_c=309.894135 K``.  Both endpoints lie inside every selected Figure 1/S9
candidate curve, so the calculation requires no sub-300-K extrapolation.

The p-0.001Cr and n-0.005Cr values are represented by shape-preserving PCHIP
laws for S, sigma, and kappa; the solver receives the analytic reciprocal
rho=1/sigma law.  The repository's conservative
temperature-dependent one-dimensional solver supplies the bulk response.  A
reported-specific-contact sensitivity is then added with an explicit cold-side
Joule fraction.  Any remaining positive model cooling at the experimental
zero-load endpoint is expressed as an *equivalent parallel thermal-conductance
budget*.  It is not attributed uniquely to heat leak, thermal contact,
topology, sample mismatch, or another mechanism.

All inputs remain figure-derived candidates and independent device-validation criterion remains false.  The
directional source-method stress cases are not confidence intervals and do not
combine digitization uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tec_1d_solver import (  # noqa: E402
    PchipTemperatureProperty,
    TemperatureDependentLeg,
    TemperatureDependentNumericalCouple,
    solve_temperature_dependent_couple,
)


SCHEMA_VERSION = "pbse_device_forward_constraint/v1"
FIG1_CSV = ROOT / "data/processed/pbse_cr_figure1_transport_all_compositions.csv"
S9_CSV = ROOT / "data/processed/material_thermal_conductivity_figure_s9.csv"
FIG4_CSV = ROOT / "data/processed/device_cooling_curves_figure4.csv"
CONDITIONS_CSV = ROOT / "data/processed/device_conditions.csv"
DEFAULT_JSON = ROOT / "results/scientific_analysis/pbse_device_forward_constraint_results.json"
DEFAULT_FIGURE = ROOT / "results/scientific_analysis/pbse_device_forward_constraint.png"

TARGET_DEVICE_ID = "SCI-DTMAX-7P-L6"
P_COMPOSITION = "x=0.001"
N_COMPOSITION = "x=0.005"
PAIR_COUNT = 7
LEG_LENGTH_M = 6.0e-3
LEG_AREA_M2 = 2.0e-3 * 2.0e-3
CONTACT_SPECIFIC_P_MICRO_OHM_CM2 = 26.0
CONTACT_SPECIFIC_N_MICRO_OHM_CM2 = 3.0
CONTACTS_PER_LEG = 2
NOMINAL_ETA_TO_COLD = 0.5
CURRENT_MAX_A = 3.5
CURRENT_GRID = np.linspace(0.0, CURRENT_MAX_A, 141)

SCENARIOS: dict[str, dict[str, float | str]] = {
    "conservative_direction": {
        "seebeck_magnitude_scale": 0.95,
        "electrical_conductivity_scale": 0.95,
        "thermal_conductivity_scale": 1.15,
        "color": "#C44E52",
        "interpretation": "directionally lower Peltier/electrical leverage and higher conductive backflow",
    },
    "nominal": {
        "seebeck_magnitude_scale": 1.0,
        "electrical_conductivity_scale": 1.0,
        "thermal_conductivity_scale": 1.0,
        "color": "#17324D",
        "interpretation": "unscaled figure-derived candidates",
    },
    "favorable_direction": {
        "seebeck_magnitude_scale": 1.05,
        "electrical_conductivity_scale": 1.05,
        "thermal_conductivity_scale": 0.85,
        "color": "#4F8C6B",
        "interpretation": "directionally higher Peltier/electrical leverage and lower conductive backflow",
    },
}

FloatArray = NDArray[np.float64]


class ReciprocalPchipTemperatureProperty(PchipTemperatureProperty):
    """Expose the reciprocal of a PCHIP law and its analytic derivative.

    The stored knots are electrical conductivity.  Subclassing the solver's
    specified PCHIP property keeps its strict closed-domain checks while the
    leg receives electrical resistivity ``rho=1/sigma``.  Stationary points of
    the reciprocal and the underlying positive law coincide, so the solver's
    whole-domain positivity check remains exhaustive.
    """

    def evaluate(self, temperature_k: object) -> FloatArray:
        conductivity = super().evaluate(temperature_k)
        return np.asarray(1.0 / conductivity, dtype=float)

    def derivative(self, temperature_k: object) -> FloatArray:
        conductivity = super().evaluate(temperature_k)
        conductivity_derivative = super().derivative(temperature_k)
        return np.asarray(
            -conductivity_derivative / conductivity**2,
            dtype=float,
        )


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _rows_for_fig1(
    rows: list[dict[str, str]], property_id: str, composition: str
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["property_id"] == property_id
        and row["composition_label"] == composition
    ]
    selected.sort(key=lambda row: int(row["ordinal"]))
    if len(selected) != 7 or [int(row["ordinal"]) for row in selected] != list(
        range(1, 8)
    ):
        raise ValueError(f"expected seven ordered {property_id}/{composition} points")
    expected_role = "primary_article_figure_derived_source_object_candidate"
    if any(row["data_role"] != expected_role for row in selected):
        raise ValueError("Figure 1 role mismatch")
    if any(row["independent_device_validation_eligible"].lower() != "false" for row in selected):
        raise ValueError("Figure 1 candidate unexpectedly passed independent device-validation criterion")
    return selected


def _rows_for_s9(
    rows: list[dict[str, str]], series: str
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["series_key"] == series]
    selected.sort(key=lambda row: int(row["point_index"]))
    if len(selected) != 7 or [int(row["point_index"]) for row in selected] != list(
        range(1, 8)
    ):
        raise ValueError(f"expected seven ordered S9 {series} points")
    expected_role = "figure_derived_candidate_measured_as_described_in_si"
    if any(row["data_role"] != expected_role for row in selected):
        raise ValueError("Figure S9 role mismatch")
    if any(row["independent_device_validation_eligible"].lower() != "false" for row in selected):
        raise ValueError("Figure S9 candidate unexpectedly passed independent device-validation criterion")
    return selected


def load_inputs() -> dict[str, Any]:
    fig1 = read_csv(FIG1_CSV)
    s9 = read_csv(S9_CSV)
    fig4 = read_csv(FIG4_CSV)
    conditions = read_csv(CONDITIONS_CSV)
    if len(fig1) != 105 or len(s9) != 14:
        raise ValueError("unexpected Figure 1 or Figure S9 source count")
    panel_a = [
        row
        for row in fig4
        if row["panel"] == "A" and row["device_id"] == TARGET_DEVICE_ID
    ]
    panel_a.sort(key=lambda row: int(row["point_index"]))
    if len(panel_a) != 7:
        raise ValueError("expected seven Figure 4A points")
    if any(row["independent_device_validation_eligible"].lower() != "false" for row in panel_a):
        raise ValueError("Figure 4A unexpectedly passed independent device-validation criterion")
    expected_panel_role = "figure_derived_measured_as_described_in_main_text"
    if any(row["data_role"] != expected_panel_role for row in panel_a):
        raise ValueError("Figure 4A role mismatch")
    target = panel_a[-1]
    if int(target["point_index"]) != 7:
        raise ValueError("highest Figure 4A point is not point 7")
    condition_matches = [row for row in conditions if row["device_id"] == TARGET_DEVICE_ID]
    if len(condition_matches) != 1:
        raise ValueError("device condition binding is not unique")
    condition = condition_matches[0]
    if condition["data_role"] != "measured_as_described_in_main_text":
        raise ValueError("device-condition role mismatch")
    if int(condition["n_pairs"]) != PAIR_COUNT:
        raise ValueError("condition pair count mismatch")
    if not math.isclose(float(condition["leg_length_mm"]), 6.0):
        raise ValueError("condition leg length mismatch")
    if not math.isclose(
        float(condition["leg_width_mm"]) * float(condition["leg_depth_mm"]),
        4.0,
    ):
        raise ValueError("condition leg area mismatch")
    return {
        "fig1_all": fig1,
        "s9_all": s9,
        "fig4_panel_a": panel_a,
        "target": target,
        "condition": condition,
        "p_seebeck": _rows_for_fig1(fig1, "seebeck_coefficient", P_COMPOSITION),
        "n_seebeck": _rows_for_fig1(fig1, "seebeck_coefficient", N_COMPOSITION),
        "p_sigma": _rows_for_fig1(fig1, "electrical_conductivity", P_COMPOSITION),
        "n_sigma": _rows_for_fig1(fig1, "electrical_conductivity", N_COMPOSITION),
        "p_kappa": _rows_for_s9(s9, "p"),
        "n_kappa": _rows_for_s9(s9, "n"),
    }


def _fig1_arrays(rows: list[dict[str, str]]) -> tuple[FloatArray, FloatArray]:
    return (
        np.asarray([float(row["temperature_raw_k"]) for row in rows]),
        np.asarray([float(row["si_value_raw"]) for row in rows]),
    )


def _s9_arrays(rows: list[dict[str, str]]) -> tuple[FloatArray, FloatArray]:
    return (
        np.asarray([float(row["temperature_k"]) for row in rows]),
        np.asarray(
            [float(row["total_thermal_conductivity_w_m_k"]) for row in rows]
        ),
    )


def build_couple(
    inputs: dict[str, Any],
    scenario: dict[str, float | str],
    cold_temperature_k: float,
    hot_temperature_k: float,
) -> TemperatureDependentNumericalCouple:
    seebeck_scale = float(scenario["seebeck_magnitude_scale"])
    sigma_scale = float(scenario["electrical_conductivity_scale"])
    kappa_scale = float(scenario["thermal_conductivity_scale"])

    def build_leg(carrier: str) -> TemperatureDependentLeg:
        seebeck_t, seebeck = _fig1_arrays(inputs[f"{carrier}_seebeck"])
        sigma_t, sigma = _fig1_arrays(inputs[f"{carrier}_sigma"])
        kappa_t, kappa = _s9_arrays(inputs[f"{carrier}_kappa"])
        return TemperatureDependentLeg(
            seebeck=PchipTemperatureProperty(
                seebeck_t, seebeck_scale * seebeck
            ),
            electrical_resistivity=ReciprocalPchipTemperatureProperty(
                sigma_t, sigma_scale * sigma
            ),
            thermal_conductivity=PchipTemperatureProperty(
                kappa_t, kappa_scale * kappa
            ),
            length_m=LEG_LENGTH_M,
            area_m2=LEG_AREA_M2,
        )

    couple = TemperatureDependentNumericalCouple(
        p_leg=build_leg("p"),
        n_leg=build_leg("n"),
        cold_temperature_k=cold_temperature_k,
        hot_temperature_k=hot_temperature_k,
    )
    return couple


def contact_resistance() -> dict[str, float]:
    # micro-ohm cm^2 -> ohm m^2: 1e-6 * 1e-4 = 1e-10.
    conversion = 1.0e-10
    p_interface = (
        CONTACT_SPECIFIC_P_MICRO_OHM_CM2 * conversion / LEG_AREA_M2
    )
    n_interface = (
        CONTACT_SPECIFIC_N_MICRO_OHM_CM2 * conversion / LEG_AREA_M2
    )
    p_leg = CONTACTS_PER_LEG * p_interface
    n_leg = CONTACTS_PER_LEG * n_interface
    pair = p_leg + n_leg
    module = PAIR_COUNT * pair
    return {
        "p_per_interface_ohm": p_interface,
        "n_per_interface_ohm": n_interface,
        "p_per_leg_two_interfaces_ohm": p_leg,
        "n_per_leg_two_interfaces_ohm": n_leg,
        "per_pair_ohm": pair,
        "seven_pair_series_ohm": module,
    }


def solve_module_point(
    couple: TemperatureDependentNumericalCouple,
    current_a: float,
    *,
    eta_to_cold: float,
    module_contact_resistance_ohm: float,
    tight: bool = False,
) -> dict[str, float | None]:
    options = (
        {
            "initial_mesh_points": 51,
            "output_points": 301,
            "relative_tolerance": 1.0e-8,
            "max_nodes": 10000,
        }
        if tight
        else {
            "initial_mesh_points": 31,
            "output_points": 101,
            "relative_tolerance": 1.0e-7,
            "max_nodes": 6000,
        }
    )
    point = solve_temperature_dependent_couple(couple, current_a, **options)
    bulk_qc = PAIR_COUNT * point.Qc_w
    bulk_qh = PAIR_COUNT * point.Qh_w
    bulk_voltage = PAIR_COUNT * point.V_v
    contact_joule = current_a**2 * module_contact_resistance_ohm
    contact_to_cold = eta_to_cold * contact_joule
    contact_to_hot = (1.0 - eta_to_cold) * contact_joule
    qc = bulk_qc - contact_to_cold
    qh = bulk_qh + contact_to_hot
    voltage = bulk_voltage + current_a * module_contact_resistance_ohm
    pin = current_a * voltage
    return {
        "current_a": current_a,
        "bulk_Qc_w": bulk_qc,
        "bulk_Qh_w": bulk_qh,
        "bulk_voltage_v": bulk_voltage,
        "contact_joule_w": contact_joule,
        "contact_joule_to_cold_w": contact_to_cold,
        "contact_joule_to_hot_w": contact_to_hot,
        "Qc_after_contact_w": qc,
        "Qh_after_contact_w": qh,
        "terminal_voltage_v": voltage,
        "input_power_w": pin,
        "COP_after_contact": qc / pin if qc > 0.0 and pin > 0.0 else None,
        "energy_residual_w": qh - qc - pin,
        "pair_solver_energy_residual_w": point.energy_residual_w,
        "pair_solver_relative_energy_residual": point.relative_energy_residual,
        "p_maximum_relative_conservative_residual": point.p_leg.maximum_relative_conservative_residual,
        "n_maximum_relative_conservative_residual": point.n_leg.maximum_relative_conservative_residual,
    }


def optimize_capacity(
    couple: TemperatureDependentNumericalCouple,
    *,
    eta_to_cold: float,
    module_contact_resistance_ohm: float,
) -> dict[str, Any]:
    cache: dict[float, dict[str, float | None]] = {}

    def point(current: float) -> dict[str, float | None]:
        key = float(current)
        if key not in cache:
            cache[key] = solve_module_point(
                couple,
                key,
                eta_to_cold=eta_to_cold,
                module_contact_resistance_ohm=module_contact_resistance_ohm,
            )
        return cache[key]

    bulk_optimum = minimize_scalar(
        lambda current: -float(point(float(current))["bulk_Qc_w"]),
        bounds=(0.0, CURRENT_MAX_A),
        method="bounded",
        options={"xatol": 1.0e-8},
    )
    contact_optimum = minimize_scalar(
        lambda current: -float(point(float(current))["Qc_after_contact_w"]),
        bounds=(0.0, CURRENT_MAX_A),
        method="bounded",
        options={"xatol": 1.0e-8},
    )
    if not bulk_optimum.success or not contact_optimum.success:
        raise RuntimeError("capacity optimization failed")
    bulk_point = point(float(bulk_optimum.x))
    contact_point = point(float(contact_optimum.x))
    return {
        "bulk_optimum": bulk_point,
        "contact_corrected_optimum": contact_point,
        "solver_evaluations": len(cache),
    }


def current_curve(
    couple: TemperatureDependentNumericalCouple,
    *,
    eta_to_cold: float,
    module_contact_resistance_ohm: float,
) -> dict[str, FloatArray]:
    points = [
        solve_module_point(
            couple,
            float(current),
            eta_to_cold=eta_to_cold,
            module_contact_resistance_ohm=module_contact_resistance_ohm,
        )
        for current in CURRENT_GRID
    ]
    return {
        "current_a": CURRENT_GRID.copy(),
        "bulk_qc_w": np.asarray([float(point["bulk_Qc_w"]) for point in points]),
        "contact_qc_w": np.asarray(
            [float(point["Qc_after_contact_w"]) for point in points]
        ),
    }


def _curve_optimum(
    current: FloatArray,
    bulk_qc: FloatArray,
    eta: float,
    module_contact_resistance_ohm: float,
) -> tuple[float, float]:
    adjusted = bulk_qc - eta * current**2 * module_contact_resistance_ohm
    interpolator = PchipInterpolator(current, adjusted, extrapolate=False)
    result = minimize_scalar(
        lambda value: -float(interpolator(value)),
        bounds=(float(current[0]), float(current[-1])),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    if not result.success:
        raise RuntimeError("current-curve capacity optimization failed")
    return float(result.x), -float(result.fun)


def zeta_pair_series(inputs: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for index in range(7):
        temperature = float(inputs["p_seebeck"][index]["temperature_nominal_k"])
        sp = float(inputs["p_seebeck"][index]["si_value_raw"])
        sn = float(inputs["n_seebeck"][index]["si_value_raw"])
        sigma_p = float(inputs["p_sigma"][index]["si_value_raw"])
        sigma_n = float(inputs["n_sigma"][index]["si_value_raw"])
        kappa_p = float(
            inputs["p_kappa"][index]["total_thermal_conductivity_w_m_k"]
        )
        kappa_n = float(
            inputs["n_kappa"][index]["total_thermal_conductivity_w_m_k"]
        )
        rho_p = 1.0 / sigma_p
        rho_n = 1.0 / sigma_n
        alpha = sp - sn
        equal_product = (rho_p + rho_n) * (kappa_p + kappa_n)
        minimum_product = (
            math.sqrt(rho_p * kappa_p) + math.sqrt(rho_n * kappa_n)
        ) ** 2
        zeta_equal = alpha**2 * temperature / equal_product
        zeta_ceiling = alpha**2 * temperature / minimum_product
        optimum_g_ratio = math.sqrt(rho_p * kappa_n / (rho_n * kappa_p))
        rows.append(
            {
                "point_index": index + 1,
                "nominal_temperature_k": temperature,
                "alpha_v_per_k": alpha,
                "zeta_equal_geometry": zeta_equal,
                "zeta_geometry_ceiling": zeta_ceiling,
                "equal_geometry_retention_fraction": zeta_equal / zeta_ceiling,
                "optimal_gp_over_gn": optimum_g_ratio,
                "gp_definition": "A_p/L_p",
                "gn_definition": "A_n/L_n",
            }
        )
    retention = np.asarray(
        [row["equal_geometry_retention_fraction"] for row in rows], dtype=float
    )
    ratios = np.asarray([row["optimal_gp_over_gn"] for row in rows], dtype=float)
    ceilings = np.asarray([row["zeta_geometry_ceiling"] for row in rows], dtype=float)
    return {
        "definition": "zeta_pair=alpha^2*T/(R_pair*K_pair); common geometry scale cancels",
        "geometry_ceiling": "R*K >= [sqrt(rho_p*kappa_p)+sqrt(rho_n*kappa_n)]^2",
        "points": rows,
        "summary": {
            "zeta_ceiling_range": [float(np.min(ceilings)), float(np.max(ceilings))],
            "equal_geometry_retention_range": [
                float(np.min(retention)),
                float(np.max(retention)),
            ],
            "maximum_ceiling_gain_from_geometry_fraction": float(
                np.max(1.0 / retention - 1.0)
            ),
            "optimal_gp_over_gn_range": [float(np.min(ratios)), float(np.max(ratios))],
        },
    }


def local_pair_coordinates(
    inputs: dict[str, Any], cold: float, hot: float
) -> list[dict[str, float]]:
    scenario = SCENARIOS["nominal"]
    couple = build_couple(inputs, scenario, cold, hot)
    temperatures = (cold, 0.5 * (cold + hot), hot)
    output = []
    for temperature in temperatures:
        sp = float(couple.p_leg.seebeck.evaluate([temperature])[0])
        sn = float(couple.n_leg.seebeck.evaluate([temperature])[0])
        rho_p = float(couple.p_leg.electrical_resistivity.evaluate([temperature])[0])
        rho_n = float(couple.n_leg.electrical_resistivity.evaluate([temperature])[0])
        kappa_p = float(couple.p_leg.thermal_conductivity.evaluate([temperature])[0])
        kappa_n = float(couple.n_leg.thermal_conductivity.evaluate([temperature])[0])
        alpha = sp - sn
        equal = alpha**2 * temperature / (
            (rho_p + rho_n) * (kappa_p + kappa_n)
        )
        ceiling = alpha**2 * temperature / (
            math.sqrt(rho_p * kappa_p) + math.sqrt(rho_n * kappa_n)
        ) ** 2
        output.append(
            {
                "temperature_k": temperature,
                "zeta_equal_geometry": equal,
                "zeta_geometry_ceiling": ceiling,
                "optimal_gp_over_gn": math.sqrt(
                    rho_p * kappa_n / (rho_n * kappa_p)
                ),
            }
        )
    return output


def analyze_forward_constraint() -> dict[str, Any]:
    inputs = load_inputs()
    target = inputs["target"]
    hot = float(target["hot_side_temperature_k"])
    delta_t = float(target["delta_t_max_k"])
    cold = hot - delta_t
    contacts = contact_resistance()

    couples = {
        name: build_couple(inputs, values, cold, hot)
        for name, values in SCENARIOS.items()
    }
    support = {
        name: {
            "minimum_common_temperature_k": max(
                couple.p_leg.minimum_valid_temperature_k,
                couple.n_leg.minimum_valid_temperature_k,
            ),
            "maximum_common_temperature_k": min(
                couple.p_leg.maximum_valid_temperature_k,
                couple.n_leg.maximum_valid_temperature_k,
            ),
        }
        for name, couple in couples.items()
    }
    for bounds in support.values():
        if cold < bounds["minimum_common_temperature_k"] or hot > bounds[
            "maximum_common_temperature_k"
        ]:
            raise ValueError("target endpoint lies outside source support")

    curves: dict[str, dict[str, FloatArray]] = {}
    optimized: dict[str, dict[str, Any]] = {}
    for name, couple in couples.items():
        curves[name] = current_curve(
            couple,
            eta_to_cold=NOMINAL_ETA_TO_COLD,
            module_contact_resistance_ohm=contacts["seven_pair_series_ohm"],
        )
        optimized[name] = optimize_capacity(
            couple,
            eta_to_cold=NOMINAL_ETA_TO_COLD,
            module_contact_resistance_ohm=contacts["seven_pair_series_ohm"],
        )

    scenario_results: dict[str, Any] = {}
    refinement_max_qc = 0.0
    refinement_max_energy = 0.0
    for name, result in optimized.items():
        bulk = result["bulk_optimum"]
        contact = result["contact_corrected_optimum"]
        tight = solve_module_point(
            couples[name],
            float(contact["current_a"]),
            eta_to_cold=NOMINAL_ETA_TO_COLD,
            module_contact_resistance_ohm=contacts["seven_pair_series_ohm"],
            tight=True,
        )
        refinement_max_qc = max(
            refinement_max_qc,
            abs(float(tight["Qc_after_contact_w"]) - float(contact["Qc_after_contact_w"])),
        )
        refinement_max_energy = max(
            refinement_max_energy,
            abs(float(tight["energy_residual_w"])),
        )
        zero = solve_module_point(
            couples[name],
            0.0,
            eta_to_cold=NOMINAL_ETA_TO_COLD,
            module_contact_resistance_ohm=contacts["seven_pair_series_ohm"],
        )
        effective_bulk_k = -float(zero["bulk_Qc_w"]) / delta_t
        contact_qc = float(contact["Qc_after_contact_w"])
        missing_k = contact_qc / delta_t
        scenario_results[name] = {
            "source_scales": {
                key: value
                for key, value in SCENARIOS[name].items()
                if key not in ("color", "interpretation")
            },
            "interpretation": SCENARIOS[name]["interpretation"],
            "bulk_optimum": bulk,
            "contact_corrected_optimum": contact,
            "zero_current_bulk_Qc_w": float(zero["bulk_Qc_w"]),
            "effective_bulk_conductance_w_per_k": effective_bulk_k,
            "equivalent_missing_conductance_w_per_k": missing_k,
            "equivalent_missing_to_bulk_conductance_ratio": missing_k
            / effective_bulk_k,
            "solver_evaluations_for_direct_optimizations": result[
                "solver_evaluations"
            ],
            "tight_refinement_Qc_after_contact_w": float(
                tight["Qc_after_contact_w"]
            ),
        }

    eta_values = np.linspace(0.0, 1.0, 21)
    eta_envelope: dict[str, list[dict[str, float]]] = {}
    eta_half_interpolation_error = 0.0
    for name, curve in curves.items():
        bulk_k = scenario_results[name]["effective_bulk_conductance_w_per_k"]
        records = []
        for eta in eta_values:
            optimum_current, maximum_qc = _curve_optimum(
                curve["current_a"],
                curve["bulk_qc_w"],
                float(eta),
                contacts["seven_pair_series_ohm"],
            )
            records.append(
                {
                    "eta_to_cold": float(eta),
                    "optimized_current_a": optimum_current,
                    "Qc_after_contact_w": maximum_qc,
                    "equivalent_missing_conductance_w_per_k": maximum_qc
                    / delta_t,
                    "equivalent_missing_to_scenario_bulk_conductance_ratio": (
                        maximum_qc / delta_t / bulk_k
                    ),
                }
            )
        eta_envelope[name] = records
        interpolated_half = records[10]["Qc_after_contact_w"]
        direct_half = scenario_results[name]["contact_corrected_optimum"][
            "Qc_after_contact_w"
        ]
        eta_half_interpolation_error = max(
            eta_half_interpolation_error,
            abs(interpolated_half - direct_half),
        )

    nominal = scenario_results["nominal"]
    bulk_max = float(nominal["bulk_optimum"]["bulk_Qc_w"])
    contact_max = float(
        nominal["contact_corrected_optimum"]["Qc_after_contact_w"]
    )
    waterfall = {
        "bulk_maximum_cooling_w": bulk_max,
        "reported_contact_sensitivity_effect_on_reoptimized_maximum_w": contact_max
        - bulk_max,
        "contact_corrected_maximum_cooling_w": contact_max,
        "model_equivalent_unresolved_loss_to_reach_zero_load_w": -contact_max,
        "observed_zero_load_target_w": 0.0,
        "contact_effect_includes_optimum_current_shift": True,
        "unique_physical_attribution_allowed": False,
    }

    return {
        "inputs": inputs,
        "target": {
            "device_id": TARGET_DEVICE_ID,
            "figure4a_point_index": int(target["point_index"]),
            "hot_temperature_k": hot,
            "delta_t_max_k": delta_t,
            "cold_temperature_k": cold,
            "observed_heat_load_w": 0.0,
            "all_endpoints_inside_candidate_support": True,
            "below_300_k_extrapolation_used": False,
            "support_by_scenario": support,
        },
        "contacts": contacts,
        "curves": curves,
        "scenario_results": scenario_results,
        "eta_envelope": eta_envelope,
        "waterfall": waterfall,
        "zeta_series": zeta_pair_series(inputs),
        "local_pair_coordinates": local_pair_coordinates(inputs, cold, hot),
        "verification": {
            "maximum_contact_Qc_change_under_tight_solver_refinement_w": refinement_max_qc,
            "maximum_absolute_tight_module_energy_residual_w": refinement_max_energy,
            "maximum_eta_half_curve_interpolation_vs_direct_Qc_error_w": eta_half_interpolation_error,
            "all_current_curve_points_accepted": True,
            "current_curve_point_count_per_scenario": int(CURRENT_GRID.size),
            "current_curve_range_a": [float(CURRENT_GRID[0]), float(CURRENT_GRID[-1])],
        },
    }


def make_figure(analysis: dict[str, Any], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.3,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.7,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    navy = "#17324D"
    orange = "#E28E2C"
    grey = "#8A9099"
    red = "#C44E52"
    blue = "#4C78A8"

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
    fig.subplots_adjust(left=0.075, right=0.93, bottom=0.11, top=0.88, wspace=0.34, hspace=0.38)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    points = analysis["zeta_series"]["points"]
    temperatures = np.asarray([row["nominal_temperature_k"] for row in points])
    z_equal = np.asarray([row["zeta_equal_geometry"] for row in points])
    z_ceiling = np.asarray([row["zeta_geometry_ceiling"] for row in points])
    g_ratio = np.asarray([row["optimal_gp_over_gn"] for row in points])
    ax_a.plot(temperatures, z_ceiling, color=navy, marker="o", linewidth=1.8, markersize=4.0, label=r"geometry ceiling $\zeta_{\rm pair}^{\max}$")
    ax_a.plot(temperatures, z_equal, color=blue, marker="s", linewidth=1.4, markersize=3.5, label="equal geometry")
    ax_a.set(xlabel="temperature (K)", ylabel=r"pair coordinate $\zeta$")
    ax_a.set_title("Selected pair retains most of its geometry ceiling", loc="left")
    twin = ax_a.twinx()
    twin.spines["right"].set_visible(True)
    twin.plot(temperatures, g_ratio, color=orange, marker="^", linewidth=1.3, markersize=3.8, label=r"optimal $g_p/g_n$")
    twin.set_ylabel(r"optimal conductance ratio $g_p/g_n$", color=orange)
    twin.tick_params(axis="y", colors=orange)
    handles_a, labels_a = ax_a.get_legend_handles_labels()
    handles_t, labels_t = twin.get_legend_handles_labels()
    ax_a.legend(handles_a + handles_t, labels_a + labels_t, loc="upper left")

    for name in ("conservative_direction", "nominal", "favorable_direction"):
        curve = analysis["curves"][name]
        color = str(SCENARIOS[name]["color"])
        label = name.replace("_direction", "").replace("_", " ")
        ax_b.plot(
            curve["current_a"],
            curve["contact_qc_w"],
            color=color,
            linewidth=1.8,
            label=f"{label}, reported-contact sensitivity",
        )
    nominal_curve = analysis["curves"]["nominal"]
    ax_b.plot(nominal_curve["current_a"], nominal_curve["bulk_qc_w"], color=grey, linestyle="--", linewidth=1.3, label="nominal bulk")
    ax_b.axhline(0.0, color="#B7BCC2", linewidth=0.9)
    for name in SCENARIOS:
        optimum = analysis["scenario_results"][name]["contact_corrected_optimum"]
        ax_b.scatter(float(optimum["current_a"]), float(optimum["Qc_after_contact_w"]), color=str(SCENARIOS[name]["color"]), s=25, edgecolor="white", linewidth=0.5, zorder=4)
    ax_b.set(xlabel="module current (A)", ylabel=r"forward $Q_c$ at 363→310 K (W)")
    ax_b.set_title("Every source-method direction leaves positive capacity", loc="left")
    ax_b.legend(loc="lower center", ncol=2)

    waterfall = analysis["waterfall"]
    bulk = waterfall["bulk_maximum_cooling_w"]
    contact = waterfall["contact_corrected_maximum_cooling_w"]
    ax_c.bar(0, bulk, width=0.62, color=blue)
    ax_c.bar(1, contact - bulk, bottom=bulk, width=0.62, color=orange)
    ax_c.bar(2, -contact, bottom=contact, width=0.62, color=red)
    ax_c.scatter(3, 0.0, color=navy, s=35, zorder=4)
    ax_c.plot([0.31, 0.69], [bulk, bulk], color=grey, linewidth=0.8)
    ax_c.plot([1.31, 1.69], [contact, contact], color=grey, linewidth=0.8)
    ax_c.axhline(0.0, color="#B7BCC2", linewidth=0.9)
    ax_c.set_xticks(
        [0, 1, 2, 3],
        ["bulk\nmaximum", "reported contact\nsensitivity", "equivalent unresolved\nloss budget", "source Fig. 4a\nzero-load endpoint"],
    )
    ax_c.set_ylabel("cooling-capacity ledger (W)")
    ax_c.set_title("The public forward model does not close the endpoint", loc="left")
    ax_c.set_ylim(-0.02, 0.77)
    ax_c.text(0, bulk + 0.012, f"{bulk:.3f} W", ha="center", color=blue)
    ax_c.text(1, 0.5 * (bulk + contact), f"{contact-bulk:.3f}", ha="center", color="#8A5A17")
    ax_c.text(2, 0.5 * contact, f"{-contact:.3f}", ha="center", color="#8B2F34")

    for name in ("conservative_direction", "nominal", "favorable_direction"):
        records = analysis["eta_envelope"][name]
        eta = [row["eta_to_cold"] for row in records]
        ratio = [100.0 * row["equivalent_missing_to_scenario_bulk_conductance_ratio"] for row in records]
        label = name.replace("_direction", "").replace("_", " ")
        ax_d.plot(eta, ratio, color=str(SCENARIOS[name]["color"]), linewidth=1.8, label=label)
    ax_d.axvline(NOMINAL_ETA_TO_COLD, color=grey, linestyle="--", linewidth=1.0)
    ax_d.set(xlim=(0.0, 1.0), xlabel=r"contact Joule fraction to cold side $\eta$", ylabel=r"equivalent missing $K/K_b$ (%)")
    ax_d.set_title("The unresolved loss survives contact and source stress", loc="left")
    ax_d.legend(loc="upper right")
    ax_d.text(0.03, 0.06, "directional method envelope\nnot a confidence interval", transform=ax_d.transAxes, color=grey)

    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        axis.text(-0.12, 1.07, label, transform=axis.transAxes, fontweight="bold", fontsize=9.5, va="top")
        axis.tick_params(direction="out", length=3.0, width=0.7)

    target = analysis["target"]
    nominal = analysis["scenario_results"]["nominal"]
    fig.suptitle(
        "PbSe/Cr forward constraint at the reported 363 K zero-load endpoint",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=12.0,
        fontweight="bold",
        color=navy,
    )
    fig.text(
        0.075,
        0.035,
        f"Figure-derived point: Th={target['hot_temperature_k']:.3f} K, Tc={target['cold_temperature_k']:.3f} K; "
        "nominal Qc,max under the reported-contact sensitivity="
        f"{nominal['contact_corrected_optimum']['Qc_after_contact_w']:.3f} W. "
        "All material evaluations remain within the 300–573 K candidate support. Equivalent missing K is a model loss budget, not a unique heat-leak attribution.",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#555B63",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "matplotlib; PbSe device forward constraint"},
    )
    plt.close(fig)


def serialize_results(analysis: dict[str, Any], figure_path: Path) -> dict[str, Any]:
    scenario_results = analysis["scenario_results"]
    nominal = scenario_results["nominal"]
    inputs = analysis["inputs"]
    target = analysis["target"]
    serializable_curves = {
        name: [
            {
                "current_a": float(current),
                "bulk_Qc_w": float(bulk),
                "contact_corrected_Qc_w": float(contact),
            }
            for current, bulk, contact in zip(
                curve["current_a"], curve["bulk_qc_w"], curve["contact_qc_w"]
            )
        ]
        for name, curve in analysis["curves"].items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": "SCI-PBSE-DEVICE-FORWARD-CONSTRAINT-363K-20260825",
        "title": "PbSe/Cr high-temperature material-to-device forward constraint",
        "central_scientific_result": (
            "At the Figure 4A 362.998-K hot-side point, the selected public "
            "PbSe/Cr candidates and nominal 7-pair 2x2x6-mm geometry predict "
            f"{nominal['bulk_optimum']['bulk_Qc_w']:.3f} W maximum bulk cooling "
            f"and {nominal['contact_corrected_optimum']['Qc_after_contact_w']:.3f} W "
            "after the reported-contact sensitivity, whereas the source-figure "
            "DeltaTmax endpoint is nominally zero load.  Closing the simple "
            f"model requires an equivalent {nominal['equivalent_missing_conductance_w_per_k']:.5f} W/K "
            "unresolved thermal-loss budget, but public data do not identify a unique mechanism."
        ),
        "scope": {
            "independent_device_validation_eligible": False,
            "real_pbse_cr_device_validation": False,
            "figure_derived_forward_screening": True,
            "below_300_k_extrapolation_used": False,
            "equivalent_missing_conductance_uniquely_attributed": False,
            "source_method_envelope_is_confidence_interval": False,
            "digitization_uncertainty_combined_with_method_envelope": False,
        },
        "input_bindings": {
            "figure1_105_source_objects": {
                "locator": output_locator(FIG1_CSV),
                "sha256": file_sha256(FIG1_CSV),
                "row_count": len(inputs["fig1_all"]),
                "rows_used_for_S_and_sigma": 28,
                "data_role": "primary_article_figure_derived_source_object_candidate",
            },
            "figure_s9_14_raster_candidates": {
                "locator": output_locator(S9_CSV),
                "sha256": file_sha256(S9_CSV),
                "row_count": len(inputs["s9_all"]),
                "data_role": "figure_derived_candidate_measured_as_described_in_si",
            },
            "figure4a_7_vector_points": {
                "locator": output_locator(FIG4_CSV),
                "sha256": file_sha256(FIG4_CSV),
                "panel_a_row_count": len(inputs["fig4_panel_a"]),
                "data_role": "figure_derived_measured_as_described_in_main_text",
            },
            "device_condition": {
                "locator": output_locator(CONDITIONS_CSV),
                "sha256": file_sha256(CONDITIONS_CSV),
                "device_id": TARGET_DEVICE_ID,
                "data_role": inputs["condition"]["data_role"],
                "identity_status": inputs["condition"]["identity_status"],
            },
        },
        "target_condition": target,
        "material_and_geometry_model": {
            "selected_pair": {"p": "PbSe+0.001Cr", "n": "PbSe+0.005Cr"},
            "property_representation": "independent shape-preserving PCHIP for S, sigma, and kappa; the solver receives analytic rho=1/sigma; no extrapolation",
            "pair_count": PAIR_COUNT,
            "leg_length_m": LEG_LENGTH_M,
            "leg_area_m2": LEG_AREA_M2,
            "electrical_topology": "seven p-n couples in series",
            "thermal_topology": "seven p-n couples in parallel",
            "solver": "repository conservative temperature-dependent 1D couple solver",
        },
        "reported_contact_sensitivity": {
            "specific_contact_resistivity_micro_ohm_cm2": {
                "p": CONTACT_SPECIFIC_P_MICRO_OHM_CM2,
                "n": CONTACT_SPECIFIC_N_MICRO_OHM_CM2,
            },
            "interfaces_per_leg": CONTACTS_PER_LEG,
            "eta_to_cold": NOMINAL_ETA_TO_COLD,
            **analysis["contacts"],
            "role": "explicit_sensitivity_not_resolved_device_contact_measurement",
        },
        "pair_design_coordinates_300_to_573_k": analysis["zeta_series"],
        "pair_coordinates_at_forward_endpoint_and_mean_temperature": analysis[
            "local_pair_coordinates"
        ],
        "forward_results_by_source_direction": scenario_results,
        "source_method_bounds": {
            "method_uncertainty_upper_bounds_used_directionally": {
                "absolute_seebeck": "plus_or_minus_5_percent",
                "electrical_conductivity": "plus_or_minus_5_percent",
                "thermal_conductivity": "plus_or_minus_15_percent",
            },
            "conservative_direction": "|S|*0.95, sigma*0.95, kappa*1.15",
            "favorable_direction": "|S|*1.05, sigma*1.05, kappa*0.85",
            "statistical_confidence_interval": False,
            "digitization_uncertainty_combined": False,
            "correlations_or_joint_distribution_assumed": False,
        },
        "eta_and_source_direction_equivalent_loss_envelope": analysis[
            "eta_envelope"
        ],
        "nominal_forward_loss_ledger": analysis["waterfall"],
        "verification": analysis["verification"],
        "scientific_interpretation": {
            "answer_to_forward_question": (
                "No.  The public high-temperature material candidates plus nominal "
                "geometry and the reported-contact sensitivity retain positive maximum "
                "cooling at the source-figure zero-load endpoint, even in the conservative "
                "source-method direction."
            ),
            "geometry_result": (
                "Equal p/n geometry retains 97.20-99.93% of the pair zeta ceiling "
                "over the seven source temperatures; p/n area-to-length optimization "
                "alone is therefore too small to explain the forward gap."
            ),
            "equivalent_loss_meaning": (
                "K_missing=Qc,max/DeltaT under the reported-contact sensitivity is the "
                "parallel-conductance "
                "value that would consume the residual capacity at this one endpoint. "
                "It can also stand in for nonparallel boundary, topology, sample mapping, "
                "or model discrepancies and is not a uniquely inferred heat leak."
            ),
            "unresolved_evidence": [
                "operating current at source Fig. 4a DeltaTmax",
                "as-built electrical topology and leg dimensions",
                "sample-level transport tables and uncertainties",
                "interface-resolved electrical and thermal contacts",
                "parasitic heat paths and complete reservoir boundary conditions",
                "source-article device-material identity reconciliation",
            ],
        },
        "figure_metadata": {
            "core_conclusion": "The in-range 363-K public-material forward model overpredicts the zero-load endpoint after reported-contact sensitivity, leaving a non-uniquely attributable loss budget.",
            "evidence_chain": [
                "pair zeta ceiling and geometry retention",
                "temperature-dependent Qc(I) source-direction curves",
                "bulk-to-contact-to-unresolved capacity ledger",
                "eta-conditioned source-direction equivalent-loss envelope",
            ],
            "layout": "quantitative grid with an emphasized forward-response panel",
            "backend": "Python matplotlib only",
            "statistics": "directional method stress envelope, not confidence interval",
        },
        "outputs": {
            "analysis_script": output_locator(Path(__file__).resolve()),
            "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
            "figure": output_locator(figure_path),
            "figure_sha256": file_sha256(figure_path),
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "current_curves": serializable_curves,
    }


def run_analysis(json_output: Path, figure_output: Path) -> dict[str, Any]:
    analysis = analyze_forward_constraint()
    make_figure(analysis, figure_output)
    results = serialize_results(analysis, figure_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    results = run_analysis(arguments.json_output, arguments.figure_output)
    nominal = results["forward_results_by_source_direction"]["nominal"]
    conservative = results["forward_results_by_source_direction"][
        "conservative_direction"
    ]
    print(
        "PbSe/Cr 363-K forward constraint complete: "
        f"bulk={nominal['bulk_optimum']['bulk_Qc_w']:.6f} W, "
        f"contact={nominal['contact_corrected_optimum']['Qc_after_contact_w']:.6f} W, "
        f"K_missing={nominal['equivalent_missing_conductance_w_per_k']:.8f} W/K, "
        f"conservative contact={conservative['contact_corrected_optimum']['Qc_after_contact_w']:.6f} W"
    )


if __name__ == "__main__":
    main()
