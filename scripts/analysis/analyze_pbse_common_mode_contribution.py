#!/usr/bin/env python3
"""Isolate the PbSe/Cr common-mode Thomson contribution at 363 -> 310 K.

This is a controlled, figure-derived material-to-device counterfactual.  The
selected p-0.001Cr/n-0.005Cr Seebeck laws are decomposed as

    alpha(T) = S_p(T) - S_n(T)
    M(T)     = [S_p(T) + S_n(T)] / 2.

The intervention replaces only ``M(T)`` by the constant ``M(T_c)``:

    S_p_flat(T) = M(T_c) + alpha(T)/2
    S_n_flat(T) = M(T_c) - alpha(T)/2.

Thus alpha(T), rho(T), kappa(T), geometry, fixed endpoint temperatures, and
the reported electrical-contact sensitivity remain exactly unchanged.  The
cold-end Seebeck values are also anchored to their original values, so the
cold Peltier term is not directly perturbed.  Kelvin consistency does redistribute
the hot-end Peltier terms as the distributed common-mode Thomson source is removed.
The comparison therefore isolates the complete constitutive effect of the
figure-derived common-mode gradient Gamma(T)=T dM/dT, including its coupling to
asymmetric p/n temperature and resistance fields in the conservative 1D solver.

The source transport objects and the target device point remain public
figure-derived candidates.  The result is a scenario screen, not an as-built
device validation or a unique attribution of the experimental discrepancy.
"""

from __future__ import annotations

import argparse
import copy
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
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import analyze_pbse_device_forward_constraint as forward  # noqa: E402
from scripts.tec_1d_solver import (  # noqa: E402
    PchipTemperatureProperty,
    TemperatureDependentLeg,
    TemperatureDependentNumericalCouple,
    TemperatureDependentOperatingPoint,
    solve_temperature_dependent_couple,
)


SCHEMA_VERSION = "pbse_common_mode_contribution/v1"
DEFAULT_JSON = (
    ROOT
    / "results/scientific_analysis/pbse_common_mode_contribution_results.json"
)
DEFAULT_FIGURE = (
    ROOT / "results/scientific_analysis/pbse_common_mode_contribution.png"
)
ANALYSIS_ID = "SCI-PBSE-COMMON-MODE-THOMSON-363K-20260825"

SIGNED_CURRENT_GRID = np.linspace(
    -forward.CURRENT_MAX_A,
    forward.CURRENT_MAX_A,
    141,
)
PROPERTY_GRID_POINTS = 401
PROFILE_OUTPUT_POINTS = 1001
ALTERNATE_CONSTANT_REFERENCE_V_PER_K = 0.0

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


class FlattenedCommonModeSeebeckProperty(PchipTemperatureProperty):
    """Exact alpha-preserving Seebeck law with constant common mode.

    Subclassing the specified PCHIP property preserves the solver's strict
    closed-domain checks.  Evaluation and differentiation are performed from
    the two original PCHIP laws, so ``S_p_flat-S_n_flat`` equals the original
    differential Seebeck law pointwise rather than only at a resampled grid.
    """

    def __init__(
        self,
        p_source: PchipTemperatureProperty,
        n_source: PchipTemperatureProperty,
        constant_common_mode_v_per_k: float,
        carrier_sign: float,
    ) -> None:
        if carrier_sign not in (-1.0, 1.0):
            raise ValueError("carrier_sign must be +1 for p or -1 for n")
        minimum = max(
            p_source.minimum_temperature_k,
            n_source.minimum_temperature_k,
        )
        maximum = min(
            p_source.maximum_temperature_k,
            n_source.maximum_temperature_k,
        )
        if maximum <= minimum:
            raise ValueError("p/n Seebeck laws have no common domain")
        # The inherited dummy interpolant supplies the specified domain and
        # fail-closed temperature check.  The public methods below replace its
        # values by exact combinations of the original source laws.
        super().__init__((minimum, maximum), (0.0, 0.0))
        object.__setattr__(self, "p_source", p_source)
        object.__setattr__(self, "n_source", n_source)
        object.__setattr__(
            self,
            "constant_common_mode_v_per_k",
            float(constant_common_mode_v_per_k),
        )
        object.__setattr__(self, "carrier_sign", float(carrier_sign))

    def evaluate(self, temperature_k: object) -> FloatArray:
        PchipTemperatureProperty.evaluate(self, temperature_k)
        alpha = self.p_source.evaluate(temperature_k) - self.n_source.evaluate(
            temperature_k
        )
        return np.asarray(
            self.constant_common_mode_v_per_k
            + 0.5 * self.carrier_sign * alpha,
            dtype=float,
        )

    def derivative(self, temperature_k: object) -> FloatArray:
        PchipTemperatureProperty.derivative(self, temperature_k)
        alpha_derivative = self.p_source.derivative(
            temperature_k
        ) - self.n_source.derivative(temperature_k)
        return np.asarray(
            0.5 * self.carrier_sign * alpha_derivative,
            dtype=float,
        )


def build_flattened_common_mode_couple(
    original: TemperatureDependentNumericalCouple,
    constant_common_mode_v_per_k: float,
) -> TemperatureDependentNumericalCouple:
    p_source = original.p_leg.seebeck
    n_source = original.n_leg.seebeck
    if not isinstance(p_source, PchipTemperatureProperty) or not isinstance(
        n_source, PchipTemperatureProperty
    ):
        raise TypeError("the PbSe source Seebeck laws must be PCHIP properties")

    p_leg = TemperatureDependentLeg(
        seebeck=FlattenedCommonModeSeebeckProperty(
            p_source,
            n_source,
            constant_common_mode_v_per_k,
            +1.0,
        ),
        electrical_resistivity=original.p_leg.electrical_resistivity,
        thermal_conductivity=original.p_leg.thermal_conductivity,
        length_m=original.p_leg.length_m,
        area_m2=original.p_leg.area_m2,
    )
    n_leg = TemperatureDependentLeg(
        seebeck=FlattenedCommonModeSeebeckProperty(
            p_source,
            n_source,
            constant_common_mode_v_per_k,
            -1.0,
        ),
        electrical_resistivity=original.n_leg.electrical_resistivity,
        thermal_conductivity=original.n_leg.thermal_conductivity,
        length_m=original.n_leg.length_m,
        area_m2=original.n_leg.area_m2,
    )
    return TemperatureDependentNumericalCouple(
        p_leg=p_leg,
        n_leg=n_leg,
        cold_temperature_k=original.cold_temperature_k,
        hot_temperature_k=original.hot_temperature_k,
    )


def _module_contact_resistance() -> float:
    return float(forward.contact_resistance()["seven_pair_series_ohm"])


def _solve_terminal(
    couple: TemperatureDependentNumericalCouple,
    current_a: float,
    *,
    tight: bool = False,
) -> dict[str, float | None]:
    return forward.solve_module_point(
        couple,
        current_a,
        eta_to_cold=forward.NOMINAL_ETA_TO_COLD,
        module_contact_resistance_ohm=_module_contact_resistance(),
        tight=tight,
    )


def _optimized_capacity(
    couple: TemperatureDependentNumericalCouple,
) -> dict[str, Any]:
    result = forward.optimize_capacity(
        couple,
        eta_to_cold=forward.NOMINAL_ETA_TO_COLD,
        module_contact_resistance_ohm=_module_contact_resistance(),
    )
    return copy.deepcopy(result)


def _optimized_cop(
    couple: TemperatureDependentNumericalCouple,
) -> dict[str, float | None]:
    grid = np.linspace(0.0, forward.CURRENT_MAX_A, 141)
    grid_points = [_solve_terminal(couple, float(current)) for current in grid]
    feasible = [
        float(current)
        for current, point in zip(grid, grid_points)
        if point["COP_after_contact"] is not None
    ]
    if len(feasible) < 3:
        raise RuntimeError("too few positive-cooling points for COP optimization")

    def objective(current: float) -> float:
        point = _solve_terminal(couple, float(current))
        cop = point["COP_after_contact"]
        return 1.0e12 if cop is None else -float(cop)

    optimum = minimize_scalar(
        objective,
        bounds=(min(feasible), max(feasible)),
        method="bounded",
        options={"xatol": 1.0e-9},
    )
    if not optimum.success:
        raise RuntimeError("COP optimization failed")
    return _solve_terminal(couple, float(optimum.x))


def _property_coordinates(
    original: TemperatureDependentNumericalCouple,
    flattened: TemperatureDependentNumericalCouple,
) -> dict[str, Any]:
    cold = original.cold_temperature_k
    hot = original.hot_temperature_k
    temperature = np.linspace(cold, hot, PROPERTY_GRID_POINTS)
    sp = original.p_leg.seebeck.evaluate(temperature)
    sn = original.n_leg.seebeck.evaluate(temperature)
    sp_flat = flattened.p_leg.seebeck.evaluate(temperature)
    sn_flat = flattened.n_leg.seebeck.evaluate(temperature)
    alpha = sp - sn
    alpha_flat = sp_flat - sn_flat
    common = 0.5 * (sp + sn)
    common_flat = 0.5 * (sp_flat + sn_flat)
    common_derivative = 0.5 * (
        original.p_leg.seebeck.derivative(temperature)
        + original.n_leg.seebeck.derivative(temperature)
    )
    gamma = temperature * common_derivative
    gamma_flat = temperature * 0.5 * (
        flattened.p_leg.seebeck.derivative(temperature)
        + flattened.n_leg.seebeck.derivative(temperature)
    )
    rho_errors = [
        np.max(
            np.abs(
                original_leg.electrical_resistivity.evaluate(temperature)
                - flattened_leg.electrical_resistivity.evaluate(temperature)
            )
        )
        for original_leg, flattened_leg in (
            (original.p_leg, flattened.p_leg),
            (original.n_leg, flattened.n_leg),
        )
    ]
    kappa_errors = [
        np.max(
            np.abs(
                original_leg.thermal_conductivity.evaluate(temperature)
                - flattened_leg.thermal_conductivity.evaluate(temperature)
            )
        )
        for original_leg, flattened_leg in (
            (original.p_leg, flattened.p_leg),
            (original.n_leg, flattened.n_leg),
        )
    ]
    return {
        "temperature_k": temperature,
        "original_common_mode_v_per_k": common,
        "flattened_common_mode_v_per_k": common_flat,
        "original_gamma_v_per_k": gamma,
        "flattened_gamma_v_per_k": gamma_flat,
        "original_alpha_v_per_k": alpha,
        "flattened_alpha_v_per_k": alpha_flat,
        "summary": {
            "M_at_cold_v_per_k": float(common[0]),
            "M_at_hot_v_per_k": float(common[-1]),
            "M_change_cold_to_hot_v_per_k": float(common[-1] - common[0]),
            "mean_dM_dT_v_per_k2": float(
                (common[-1] - common[0]) / (hot - cold)
            ),
            "Gamma_range_v_per_k": [float(np.min(gamma)), float(np.max(gamma))],
            "Gamma_temperature_mean_v_per_k": float(
                np.trapezoid(gamma, temperature) / (hot - cold)
            ),
            "alpha_at_cold_v_per_k": float(alpha[0]),
            "alpha_at_hot_v_per_k": float(alpha[-1]),
            "maximum_absolute_alpha_intervention_error_v_per_k": float(
                np.max(np.abs(alpha_flat - alpha))
            ),
            "maximum_flat_common_mode_deviation_v_per_k": float(
                np.max(np.abs(common_flat - common_flat[0]))
            ),
            "maximum_rho_intervention_error_ohm_m": float(max(rho_errors)),
            "maximum_kappa_intervention_error_w_per_m_k": float(
                max(kappa_errors)
            ),
            "cold_anchor_p_error_v_per_k": float(sp_flat[0] - sp[0]),
            "cold_anchor_n_error_v_per_k": float(sn_flat[0] - sn[0]),
        },
    }


def _leg_thomson_ledger(
    solution: Any,
    *,
    carrier_sign: float,
    original: TemperatureDependentNumericalCouple,
    common_mode_active: bool,
) -> dict[str, float]:
    temperature = solution.temperature_k
    coordinate = solution.coordinate_m
    gradient = solution.temperature_gradient_k_per_m
    signed_current = float(solution.signed_current_a)
    current_density = float(solution.current_density_a_per_m2)
    area = forward.LEG_AREA_M2
    pairs = forward.PAIR_COUNT

    p_derivative = original.p_leg.seebeck.derivative(temperature)
    n_derivative = original.n_leg.seebeck.derivative(temperature)
    gamma = temperature * 0.5 * (p_derivative + n_derivative)
    differential_tau = (
        carrier_sign * temperature * 0.5 * (p_derivative - n_derivative)
    )
    active_common_tau = gamma if common_mode_active else np.zeros_like(gamma)

    def thomson_source(tau: FloatArray) -> float:
        # Heat-equation sign convention:
        # (k T')' + rho J^2 - tau J T' = 0.
        return float(
            -pairs
            * area
            * np.trapezoid(tau * current_density * gradient, coordinate)
        )

    joule = float(
        pairs
        * area
        * np.trapezoid(
            solution.electrical_resistivity_ohm_m * current_density**2,
            coordinate,
        )
    )
    total_thomson = thomson_source(solution.thomson_coefficient_v_per_k)
    differential_thomson = thomson_source(differential_tau)
    common_thomson = thomson_source(active_common_tau)
    reference_original_common = thomson_source(gamma)

    cold_peltier = float(
        pairs
        * area
        * solution.seebeck_v_per_k[0]
        * temperature[0]
        * current_density
    )
    cold_conduction = float(
        -pairs
        * area
        * solution.thermal_conductivity_w_per_m_k[0]
        * gradient[0]
    )
    hot_peltier = float(
        pairs
        * area
        * solution.seebeck_v_per_k[-1]
        * temperature[-1]
        * current_density
    )
    hot_conduction = float(
        -pairs
        * area
        * solution.thermal_conductivity_w_per_m_k[-1]
        * gradient[-1]
    )
    return {
        "signed_current_a": signed_current,
        "integrated_joule_generation_w": joule,
        "integrated_total_thomson_source_w": total_thomson,
        "integrated_differential_mode_thomson_source_w": differential_thomson,
        "integrated_active_common_mode_thomson_source_w": common_thomson,
        "reference_original_common_mode_thomson_source_w": reference_original_common,
        "thomson_component_closure_error_w": total_thomson
        - differential_thomson
        - common_thomson,
        "cold_peltier_w": cold_peltier,
        "cold_conduction_w": cold_conduction,
        "cold_heat_w": cold_peltier + cold_conduction,
        "hot_peltier_w": hot_peltier,
        "hot_conduction_w": hot_conduction,
        "hot_heat_w": hot_peltier + hot_conduction,
    }


def _detailed_point(
    couple: TemperatureDependentNumericalCouple,
    current_a: float,
    *,
    original: TemperatureDependentNumericalCouple,
    common_mode_active: bool,
) -> tuple[dict[str, Any], TemperatureDependentOperatingPoint]:
    point = solve_temperature_dependent_couple(
        couple,
        current_a,
        initial_mesh_points=51,
        output_points=PROFILE_OUTPUT_POINTS,
        relative_tolerance=1.0e-9,
        max_nodes=12000,
    )
    contact_resistance = _module_contact_resistance()
    contact_joule = current_a**2 * contact_resistance
    eta = forward.NOMINAL_ETA_TO_COLD
    bulk_qc = forward.PAIR_COUNT * point.Qc_w
    bulk_qh = forward.PAIR_COUNT * point.Qh_w
    bulk_voltage = forward.PAIR_COUNT * point.V_v
    qc = bulk_qc - eta * contact_joule
    qh = bulk_qh + (1.0 - eta) * contact_joule
    voltage = bulk_voltage + current_a * contact_resistance
    input_power = current_a * voltage

    p_ledger = _leg_thomson_ledger(
        point.p_leg,
        carrier_sign=+1.0,
        original=original,
        common_mode_active=common_mode_active,
    )
    n_ledger = _leg_thomson_ledger(
        point.n_leg,
        carrier_sign=-1.0,
        original=original,
        common_mode_active=common_mode_active,
    )
    sum_keys = [
        "integrated_joule_generation_w",
        "integrated_total_thomson_source_w",
        "integrated_differential_mode_thomson_source_w",
        "integrated_active_common_mode_thomson_source_w",
        "reference_original_common_mode_thomson_source_w",
        "thomson_component_closure_error_w",
        "cold_peltier_w",
        "cold_conduction_w",
        "cold_heat_w",
        "hot_peltier_w",
        "hot_conduction_w",
        "hot_heat_w",
    ]
    module_ledger = {key: p_ledger[key] + n_ledger[key] for key in sum_keys}
    profile = {
        "normalized_coordinate": (
            point.p_leg.coordinate_m / couple.p_leg.length_m
        ).tolist(),
        "p_temperature_k": point.p_leg.temperature_k.tolist(),
        "n_temperature_k": point.n_leg.temperature_k.tolist(),
    }
    summary = {
        "current_a": float(current_a),
        "cold_temperature_k": float(couple.cold_temperature_k),
        "hot_temperature_k": float(couple.hot_temperature_k),
        "bulk_Qc_w": float(bulk_qc),
        "bulk_Qh_w": float(bulk_qh),
        "bulk_voltage_v": float(bulk_voltage),
        "contact_joule_w": float(contact_joule),
        "Qc_after_contact_w": float(qc),
        "Qh_after_contact_w": float(qh),
        "terminal_voltage_v": float(voltage),
        "input_power_w": float(input_power),
        "COP_after_contact": (
            float(qc / input_power)
            if qc > 0.0 and input_power > 0.0
            else None
        ),
        "module_energy_residual_w": float(qh - qc - input_power),
        "pair_solver_energy_residual_w": float(point.energy_residual_w),
        "p_midpoint_temperature_k": float(
            point.p_leg.temperature_k[PROFILE_OUTPUT_POINTS // 2]
        ),
        "n_midpoint_temperature_k": float(
            point.n_leg.temperature_k[PROFILE_OUTPUT_POINTS // 2]
        ),
        "p_temperature_range_k": [
            float(np.min(point.p_leg.temperature_k)),
            float(np.max(point.p_leg.temperature_k)),
        ],
        "n_temperature_range_k": [
            float(np.min(point.n_leg.temperature_k)),
            float(np.max(point.n_leg.temperature_k)),
        ],
        "thomson_and_heat_ledger": {
            "sign_convention": (
                "integrated Thomson source = integral[-tau J grad(T)] dV; "
                "negative denotes a cooling term in the conductive heat equation"
            ),
            "p_leg": p_ledger,
            "n_leg": n_ledger,
            "module_net": module_ledger,
        },
        "temperature_profile": profile,
    }
    return summary, point


def _signed_current_sweep(
    original: TemperatureDependentNumericalCouple,
    flattened: TemperatureDependentNumericalCouple,
) -> list[dict[str, float | None]]:
    records = []
    for current in SIGNED_CURRENT_GRID:
        original_point = _solve_terminal(original, float(current))
        flattened_point = _solve_terminal(flattened, float(current))
        records.append(
            {
                "current_a": float(current),
                "original_Qc_after_contact_w": float(
                    original_point["Qc_after_contact_w"]
                ),
                "flattened_Qc_after_contact_w": float(
                    flattened_point["Qc_after_contact_w"]
                ),
                "original_minus_flattened_Qc_w": float(
                    original_point["Qc_after_contact_w"]
                    - flattened_point["Qc_after_contact_w"]
                ),
                "original_terminal_voltage_v": float(
                    original_point["terminal_voltage_v"]
                ),
                "flattened_terminal_voltage_v": float(
                    flattened_point["terminal_voltage_v"]
                ),
                "original_COP_after_contact": original_point[
                    "COP_after_contact"
                ],
                "flattened_COP_after_contact": flattened_point[
                    "COP_after_contact"
                ],
            }
        )
    return records


def analyze_common_mode_contribution() -> dict[str, Any]:
    inputs = forward.load_inputs()
    target_source = inputs["target"]
    hot = float(target_source["hot_side_temperature_k"])
    delta_t = float(target_source["delta_t_max_k"])
    cold = hot - delta_t
    original = forward.build_couple(
        inputs,
        forward.SCENARIOS["nominal"],
        cold,
        hot,
    )
    m_cold = float(
        0.5
        * (
            original.p_leg.seebeck.evaluate([cold])[0]
            + original.n_leg.seebeck.evaluate([cold])[0]
        )
    )
    flattened = build_flattened_common_mode_couple(original, m_cold)
    alternate_flattened = build_flattened_common_mode_couple(
        original,
        ALTERNATE_CONSTANT_REFERENCE_V_PER_K,
    )
    coordinates = _property_coordinates(original, flattened)

    original_capacity = _optimized_capacity(original)
    flattened_capacity = _optimized_capacity(flattened)
    original_cop = _optimized_cop(original)
    flattened_cop = _optimized_cop(flattened)

    original_optimum = original_capacity["contact_corrected_optimum"]
    flattened_optimum = flattened_capacity["contact_corrected_optimum"]
    reference_current = float(original_optimum["current_a"])

    original_reference, original_reference_raw = _detailed_point(
        original,
        reference_current,
        original=original,
        common_mode_active=True,
    )
    flattened_reference, flattened_reference_raw = _detailed_point(
        flattened,
        reference_current,
        original=original,
        common_mode_active=False,
    )
    alternate_reference, alternate_reference_raw = _detailed_point(
        alternate_flattened,
        reference_current,
        original=original,
        common_mode_active=False,
    )
    reverse_original, _ = _detailed_point(
        original,
        -reference_current,
        original=original,
        common_mode_active=True,
    )
    reverse_flattened, _ = _detailed_point(
        flattened,
        -reference_current,
        original=original,
        common_mode_active=False,
    )
    zero_original, zero_original_raw = _detailed_point(
        original,
        0.0,
        original=original,
        common_mode_active=True,
    )
    zero_flattened, zero_flattened_raw = _detailed_point(
        flattened,
        0.0,
        original=original,
        common_mode_active=False,
    )

    original_optimum_tight = _solve_terminal(
        original,
        float(original_optimum["current_a"]),
        tight=True,
    )
    flattened_optimum_tight = _solve_terminal(
        flattened,
        float(flattened_optimum["current_a"]),
        tight=True,
    )

    q_original = float(original_optimum["Qc_after_contact_w"])
    q_flattened = float(flattened_optimum["Qc_after_contact_w"])
    q_effect = q_original - q_flattened
    bulk_q_effect = float(
        original_capacity["bulk_optimum"]["bulk_Qc_w"]
        - flattened_capacity["bulk_optimum"]["bulk_Qc_w"]
    )
    i_effect = float(original_optimum["current_a"] - flattened_optimum["current_a"])
    cop_at_qmax_effect = float(
        original_optimum["COP_after_contact"]
        - flattened_optimum["COP_after_contact"]
    )
    maximum_cop_effect = float(
        original_cop["COP_after_contact"] - flattened_cop["COP_after_contact"]
    )

    fixed_q_effect = float(
        original_reference["Qc_after_contact_w"]
        - flattened_reference["Qc_after_contact_w"]
    )
    reverse_q_effect = float(
        reverse_original["Qc_after_contact_w"]
        - reverse_flattened["Qc_after_contact_w"]
    )
    zero_q_effect = float(
        zero_original["Qc_after_contact_w"]
        - zero_flattened["Qc_after_contact_w"]
    )

    def maximum_profile_difference(
        first: TemperatureDependentOperatingPoint,
        second: TemperatureDependentOperatingPoint,
    ) -> float:
        return float(
            max(
                np.max(np.abs(first.p_leg.temperature_k - second.p_leg.temperature_k)),
                np.max(np.abs(first.n_leg.temperature_k - second.n_leg.temperature_k)),
            )
        )

    original_ledger = original_reference["thomson_and_heat_ledger"]
    flat_ledger = flattened_reference["thomson_and_heat_ledger"]
    cold_partition_response = {
        "p_leg_original_minus_flattened_w": float(
            original_ledger["p_leg"]["cold_heat_w"]
            - flat_ledger["p_leg"]["cold_heat_w"]
        ),
        "n_leg_original_minus_flattened_w": float(
            original_ledger["n_leg"]["cold_heat_w"]
            - flat_ledger["n_leg"]["cold_heat_w"]
        ),
        "module_original_minus_flattened_w": fixed_q_effect,
    }
    peltier_boundary_response = {
        "cold_end_original_minus_flattened_w": {
            "p_leg": float(
                original_ledger["p_leg"]["cold_peltier_w"]
                - flat_ledger["p_leg"]["cold_peltier_w"]
            ),
            "n_leg": float(
                original_ledger["n_leg"]["cold_peltier_w"]
                - flat_ledger["n_leg"]["cold_peltier_w"]
            ),
            "module_net": float(
                original_ledger["module_net"]["cold_peltier_w"]
                - flat_ledger["module_net"]["cold_peltier_w"]
            ),
        },
        "hot_end_original_minus_flattened_w": {
            "p_leg": float(
                original_ledger["p_leg"]["hot_peltier_w"]
                - flat_ledger["p_leg"]["hot_peltier_w"]
            ),
            "n_leg": float(
                original_ledger["n_leg"]["hot_peltier_w"]
                - flat_ledger["n_leg"]["hot_peltier_w"]
            ),
            "module_net": float(
                original_ledger["module_net"]["hot_peltier_w"]
                - flat_ledger["module_net"]["hot_peltier_w"]
            ),
        },
    }
    signed_current_parity_response = {
        "current_magnitude_a": reference_current,
        "positive_current_effect_w": fixed_q_effect,
        "negative_current_effect_w": reverse_q_effect,
        "odd_component_w": 0.5 * (fixed_q_effect - reverse_q_effect),
        "even_component_w": 0.5 * (fixed_q_effect + reverse_q_effect),
    }
    joule_response = {
        "p_leg_original_minus_flattened_w": float(
            original_ledger["p_leg"]["integrated_joule_generation_w"]
            - flat_ledger["p_leg"]["integrated_joule_generation_w"]
        ),
        "n_leg_original_minus_flattened_w": float(
            original_ledger["n_leg"]["integrated_joule_generation_w"]
            - flat_ledger["n_leg"]["integrated_joule_generation_w"]
        ),
        "module_original_minus_flattened_w": float(
            original_ledger["module_net"]["integrated_joule_generation_w"]
            - flat_ledger["module_net"]["integrated_joule_generation_w"]
        ),
    }

    sweep = _signed_current_sweep(original, flattened)
    all_detailed = [
        original_reference,
        flattened_reference,
        alternate_reference,
        reverse_original,
        reverse_flattened,
        zero_original,
        zero_flattened,
    ]
    maximum_temperature = max(
        max(point["p_temperature_range_k"][1], point["n_temperature_range_k"][1])
        for point in all_detailed
    )
    minimum_temperature = min(
        min(point["p_temperature_range_k"][0], point["n_temperature_range_k"][0])
        for point in all_detailed
    )
    common_support = {
        "minimum_temperature_k": max(
            original.p_leg.minimum_valid_temperature_k,
            original.n_leg.minimum_valid_temperature_k,
        ),
        "maximum_temperature_k": min(
            original.p_leg.maximum_valid_temperature_k,
            original.n_leg.maximum_valid_temperature_k,
        ),
    }

    result = {
        "_internal": {
            "inputs": inputs,
            "property_coordinates": coordinates,
        },
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "title": (
            "Common-mode Thomson contribution in the nominal long-leg PbSe/Cr screen"
        ),
        "central_scientific_result": (
            "Flattening only the figure-derived common Seebeck mode M(T) while "
            "preserving alpha(T), rho(T), kappa(T), geometry, contacts, and the "
            "309.894/362.998-K endpoints lowers the reoptimized Qc,max under the "
            "reported-contact sensitivity "
            f"by {1.0e3*q_effect:.3f} mW ({100.0*q_effect/q_original:.3f}% "
            "of the nominal forward residual).  The distributed common-mode Thomson "
            "sources and the associated hot-end Peltier redistribution are each "
            "finite but opposite in the two legs and cancel directly at module level. "
            "Their changed spatial and boundary distribution leaves only a small "
            "asymmetric terminal branch transfer.  This counterfactual difference "
            "cannot explain the "
            "nominal scenario margin."
        ),
        "scope": {
            "independent_device_validation_eligible": False,
            "real_pbse_cr_device_validation": False,
            "figure_derived_source_candidate_scenario_screen": True,
            "counterfactual_flattened_material_was_measured": False,
            "common_mode_contribution_uniquely_explains_device_gap": False,
            "below_300_k_extrapolation_used": False,
            "statistical_confidence_interval": False,
        },
        "target_condition": {
            "device_id": forward.TARGET_DEVICE_ID,
            "figure4a_point_index": int(target_source["point_index"]),
            "hot_temperature_k": hot,
            "cold_temperature_k": cold,
            "delta_t_k": delta_t,
            "observed_heat_load_w": 0.0,
            "endpoint_temperatures_fixed_in_both_models": True,
            "common_property_support_k": common_support,
            "all_detailed_control_temperature_fields_within_source_support": (
                minimum_temperature >= common_support["minimum_temperature_k"]
                and maximum_temperature <= common_support["maximum_temperature_k"]
            ),
            "solved_temperature_field_range_across_detailed_controls_k": [
                float(minimum_temperature),
                float(maximum_temperature),
            ],
        },
        "controlled_intervention": {
            "definition": (
                "alpha(T)=Sp(T)-Sn(T); M(T)=[Sp(T)+Sn(T)]/2; "
                "Sp_flat(T)=M(Tc)+alpha(T)/2; "
                "Sn_flat(T)=M(Tc)-alpha(T)/2"
            ),
            "constant_reference_choice": "M(Tc)",
            "constant_reference_v_per_k": m_cold,
            "cold_p_and_n_Seebeck_values_anchored": True,
            "alpha_T_preserved_pointwise": True,
            "rho_T_preserved_by_shared_property_objects": True,
            "kappa_T_preserved_by_shared_property_objects": True,
            "geometry_preserved": True,
            "electrical_contacts_preserved": True,
            "thermal_boundary_temperatures_preserved": True,
            "property_coordinate_summary": coordinates["summary"],
        },
        "material_and_device_model": {
            "selected_pair": {"p": "PbSe+0.001Cr", "n": "PbSe+0.005Cr"},
            "pair_count": forward.PAIR_COUNT,
            "leg_length_m": forward.LEG_LENGTH_M,
            "leg_area_m2": forward.LEG_AREA_M2,
            "electrical_topology": "seven p-n couples in series",
            "thermal_topology": "seven p-n couples in parallel",
            "electrical_contact_resistance_ohm": _module_contact_resistance(),
            "contact_joule_fraction_to_cold": forward.NOMINAL_ETA_TO_COLD,
            "reservoir_endpoint_model": "shared fixed isothermal p/n ports",
            "solver": "repository conservative temperature-dependent 1D couple solver",
        },
        "optimized_forward_comparison": {
            "original": {
                "bulk_optimum": original_capacity["bulk_optimum"],
                "contact_corrected_optimum": original_optimum,
                "maximum_COP_point": original_cop,
            },
            "flattened_common_mode": {
                "bulk_optimum": flattened_capacity["bulk_optimum"],
                "contact_corrected_optimum": flattened_optimum,
                "maximum_COP_point": flattened_cop,
            },
            "original_minus_flattened": {
                "bulk_Qc_max_w": bulk_q_effect,
                "contact_corrected_Qc_max_w": q_effect,
                "contact_corrected_Qc_max_relative_to_flattened_fraction": q_effect
                / q_flattened,
                "fraction_of_nominal_forward_residual_capacity": q_effect
                / q_original,
                "equivalent_parallel_conductance_w_per_k": q_effect / delta_t,
                "optimized_current_a": i_effect,
                "COP_at_capacity_optimum": cop_at_qmax_effect,
                "maximum_COP": maximum_cop_effect,
            },
        },
        "fixed_current_mechanism": {
            "reference_current_definition": (
                "original Qc maximum under the reported-contact sensitivity"
            ),
            "reference_current_a": reference_current,
            "original": original_reference,
            "flattened_common_mode": flattened_reference,
            "original_minus_flattened": {
                "Qc_after_contact_w": fixed_q_effect,
                "terminal_voltage_v": float(
                    original_reference["terminal_voltage_v"]
                    - flattened_reference["terminal_voltage_v"]
                ),
                "input_power_w": float(
                    original_reference["input_power_w"]
                    - flattened_reference["input_power_w"]
                ),
                "p_midpoint_temperature_k": float(
                    original_reference["p_midpoint_temperature_k"]
                    - flattened_reference["p_midpoint_temperature_k"]
                ),
                "n_midpoint_temperature_k": float(
                    original_reference["n_midpoint_temperature_k"]
                    - flattened_reference["n_midpoint_temperature_k"]
                ),
                "maximum_temperature_profile_difference_k": maximum_profile_difference(
                    original_reference_raw,
                    flattened_reference_raw,
                ),
            },
            "cold_heat_partition_response": cold_partition_response,
            "peltier_boundary_response": peltier_boundary_response,
            "signed_current_parity_response": signed_current_parity_response,
            "integrated_joule_response": joule_response,
            "mechanistic_interpretation": (
                "At shared isothermal ports the distributed common-mode Thomson "
                "integrals are equal and opposite.  Kelvin-consistent flattening "
                "also redistributes the hot-end Peltier terms equally and oppositely, "
                "while the cold-end Peltier terms remain fixed by the M(Tc) anchor. "
                "The changed spatial and boundary heat distribution drives opposite "
                "p/n temperature-field shifts; because rho(T) and kappa(T) differ "
                "by branch, the induced Joule, conductive, and boundary responses "
                "do not cancel perfectly."
            ),
        },
        "reverse_current_control": {
            "current_a": -reference_current,
            "original": reverse_original,
            "flattened_common_mode": reverse_flattened,
            "original_minus_flattened_Qc_w": reverse_q_effect,
            "effect_has_opposite_sign_to_forward": reverse_q_effect
            * fixed_q_effect
            < 0.0,
        },
        "zero_current_control": {
            "current_a": 0.0,
            "original": zero_original,
            "flattened_common_mode": zero_flattened,
            "original_minus_flattened_Qc_w": zero_q_effect,
            "terminal_voltage_difference_v": float(
                zero_original["terminal_voltage_v"]
                - zero_flattened["terminal_voltage_v"]
            ),
            "maximum_temperature_profile_difference_k": maximum_profile_difference(
                zero_original_raw,
                zero_flattened_raw,
            ),
        },
        "constant_reference_invariance_control": {
            "primary_flattened_reference_v_per_k": m_cold,
            "alternate_flattened_reference_v_per_k": ALTERNATE_CONSTANT_REFERENCE_V_PER_K,
            "reference_current_a": reference_current,
            "Qc_difference_w": float(
                flattened_reference["Qc_after_contact_w"]
                - alternate_reference["Qc_after_contact_w"]
            ),
            "terminal_voltage_difference_v": float(
                flattened_reference["terminal_voltage_v"]
                - alternate_reference["terminal_voltage_v"]
            ),
            "maximum_temperature_profile_difference_k": maximum_profile_difference(
                flattened_reference_raw,
                alternate_reference_raw,
            ),
        },
        "signed_current_sweep": sweep,
        "verification": {
            "maximum_absolute_alpha_intervention_error_v_per_k": coordinates[
                "summary"
            ]["maximum_absolute_alpha_intervention_error_v_per_k"],
            "maximum_flat_common_mode_deviation_v_per_k": coordinates["summary"][
                "maximum_flat_common_mode_deviation_v_per_k"
            ],
            "maximum_tight_refinement_Qc_error_w": max(
                abs(
                    float(original_optimum_tight["Qc_after_contact_w"])
                    - q_original
                ),
                abs(
                    float(flattened_optimum_tight["Qc_after_contact_w"])
                    - q_flattened
                ),
            ),
            "maximum_detailed_module_energy_residual_w": max(
                abs(float(point["module_energy_residual_w"]))
                for point in all_detailed
            ),
            "maximum_thomson_component_closure_error_w": max(
                abs(
                    float(
                        point["thomson_and_heat_ledger"][leg][
                            "thomson_component_closure_error_w"
                        ]
                    )
                )
                for point in all_detailed
                for leg in ("p_leg", "n_leg")
            ),
            "original_common_mode_module_cancellation_error_w": abs(
                float(
                    original_ledger["module_net"][
                        "integrated_active_common_mode_thomson_source_w"
                    ]
                )
            ),
            "zero_current_Qc_invariance_error_w": abs(zero_q_effect),
            "zero_current_voltage_invariance_error_v": abs(
                float(
                    zero_original["terminal_voltage_v"]
                    - zero_flattened["terminal_voltage_v"]
                )
            ),
            "constant_reference_Qc_invariance_error_w": abs(
                float(
                    flattened_reference["Qc_after_contact_w"]
                    - alternate_reference["Qc_after_contact_w"]
                )
            ),
            "constant_reference_voltage_invariance_error_v": abs(
                float(
                    flattened_reference["terminal_voltage_v"]
                    - alternate_reference["terminal_voltage_v"]
                )
            ),
            "all_detailed_temperature_fields_inside_source_support": (
                minimum_temperature >= common_support["minimum_temperature_k"]
                and maximum_temperature <= common_support["maximum_temperature_k"]
            ),
        },
        "source_data_for_figure": {
            "property_coordinates": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in coordinates.items()
                if key != "summary"
            },
            "relative_metric_effects_percent": {
                "bulk_Qc_max": 100.0
                * bulk_q_effect
                / float(flattened_capacity["bulk_optimum"]["bulk_Qc_w"]),
                "contact_Qc_max": 100.0 * q_effect / q_flattened,
                "I_at_contact_Qc_max": 100.0
                * i_effect
                / float(flattened_optimum["current_a"]),
                "COP_at_contact_Qc_max": 100.0
                * cop_at_qmax_effect
                / float(flattened_optimum["COP_after_contact"]),
                "maximum_COP": 100.0
                * maximum_cop_effect
                / float(flattened_cop["COP_after_contact"]),
            },
        },
    }
    return result


def make_figure(analysis: dict[str, Any], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.titlesize": 8.3,
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
    blue = "#4C78A8"
    orange = "#E28E2C"
    green = "#4F8C6B"
    red = "#C44E52"
    grey = "#8A9099"
    pale = "#DDE6ED"

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.9))
    fig.subplots_adjust(
        left=0.075,
        right=0.925,
        bottom=0.115,
        top=0.865,
        wspace=0.34,
        hspace=0.42,
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    source = analysis["source_data_for_figure"]["property_coordinates"]
    temperature = np.asarray(source["temperature_k"], dtype=float)
    common = 1.0e6 * np.asarray(source["original_common_mode_v_per_k"], dtype=float)
    common_flat = 1.0e6 * np.asarray(
        source["flattened_common_mode_v_per_k"], dtype=float
    )
    gamma = 1.0e6 * np.asarray(source["original_gamma_v_per_k"], dtype=float)
    ax_a.plot(
        temperature,
        common,
        color=navy,
        linewidth=2.0,
        label=r"original $M(T)$",
    )
    ax_a.plot(
        temperature,
        common_flat,
        color=blue,
        linestyle="--",
        linewidth=1.8,
        label=r"flattened $M(T_c)$",
    )
    ax_a.fill_between(temperature, common_flat, common, color=pale, alpha=0.7)
    ax_a.set(
        xlabel="temperature (K)",
        ylabel=r"common Seebeck mode $M$ ($\mu$V K$^{-1}$)",
    )
    ax_a.set_title(r"The intervention removes only $dM/dT$", loc="left")
    ax_a.legend(loc="upper left")
    ax_a_gamma = ax_a.twinx()
    ax_a_gamma.spines["right"].set_visible(True)
    ax_a_gamma.plot(
        temperature,
        gamma,
        color=orange,
        linewidth=1.4,
        label=r"$\Gamma=T\,dM/dT$",
    )
    ax_a_gamma.set_ylabel(
        r"common-mode Thomson coefficient $\Gamma$ ($\mu$V K$^{-1}$)",
        color=orange,
    )
    ax_a_gamma.tick_params(axis="y", colors=orange)
    ax_a.text(
        0.04,
        0.08,
        r"$\alpha(T)$ unchanged pointwise" + "\n" + r"$S_{p,n}(T_c)$ anchored",
        transform=ax_a.transAxes,
        color="#555B63",
    )

    sweep = analysis["signed_current_sweep"]
    current = np.asarray([row["current_a"] for row in sweep], dtype=float)
    delta_qc_mw = 1.0e3 * np.asarray(
        [row["original_minus_flattened_Qc_w"] for row in sweep], dtype=float
    )
    ax_b.axhline(0.0, color="#B7BCC2", linewidth=0.9)
    ax_b.axvline(0.0, color="#B7BCC2", linewidth=0.9)
    ax_b.fill_between(
        current,
        0.0,
        delta_qc_mw,
        where=delta_qc_mw >= 0.0,
        color=green,
        alpha=0.2,
    )
    ax_b.fill_between(
        current,
        0.0,
        delta_qc_mw,
        where=delta_qc_mw < 0.0,
        color=red,
        alpha=0.15,
    )
    ax_b.plot(current, delta_qc_mw, color=navy, linewidth=2.0)
    fixed = analysis["fixed_current_mechanism"]
    reverse = analysis["reverse_current_control"]
    zero = analysis["zero_current_control"]
    marker_x = np.asarray(
        [
            reverse["current_a"],
            0.0,
            fixed["reference_current_a"],
        ]
    )
    marker_y = 1.0e3 * np.asarray(
        [
            reverse["original_minus_flattened_Qc_w"],
            zero["original_minus_flattened_Qc_w"],
            fixed["original_minus_flattened"]["Qc_after_contact_w"],
        ]
    )
    ax_b.scatter(
        marker_x,
        marker_y,
        color=[red, grey, green],
        edgecolor="white",
        linewidth=0.6,
        s=32,
        zorder=4,
    )
    ax_b.annotate(
        f"forward  {marker_y[2]:+.2f} mW",
        (marker_x[2], marker_y[2]),
        xytext=(-72, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": green, "lw": 0.8},
        color=green,
    )
    ax_b.annotate(
        f"reverse  {marker_y[0]:+.2f} mW",
        (marker_x[0], marker_y[0]),
        xytext=(12, -25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": red, "lw": 0.8},
        color=red,
    )
    ax_b.set(
        xlabel="module current (A)",
        ylabel=r"$Q_c^{original}-Q_c^{flat}$ (mW)",
    )
    ax_b.set_title("The port response is current-coupled and zero at I=0", loc="left")

    metric_effects = analysis["source_data_for_figure"][
        "relative_metric_effects_percent"
    ]
    metric_names = [
        "bulk $Q_{c,max}$",
        "reported-contact $Q_{c,max}$",
        "$I$ at reported-contact $Q_{c,max}$",
        "COP at $Q_{c,max}$",
        "maximum COP",
    ]
    metric_keys = [
        "bulk_Qc_max",
        "contact_Qc_max",
        "I_at_contact_Qc_max",
        "COP_at_contact_Qc_max",
        "maximum_COP",
    ]
    values = np.asarray([float(metric_effects[key]) for key in metric_keys])
    y = np.arange(len(values))
    colors = [green if value >= 0.0 else red for value in values]
    ax_c.axvline(0.0, color="#B7BCC2", linewidth=0.9)
    ax_c.hlines(y, 0.0, values, color=colors, linewidth=2.0)
    ax_c.scatter(values, y, color=colors, s=34, edgecolor="white", linewidth=0.5)
    for index, value in enumerate(values):
        ax_c.text(
            value + 0.018,
            index,
            f"{value:+.3f}%",
            va="center",
            ha="left",
            color=colors[index],
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 0.5,
            },
        )
    ax_c.set_yticks(y, metric_names)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("original relative to flattened common mode (%)")
    ax_c.set_title("All optimized device metrics shift by <0.5%", loc="left")
    ax_c.set_xlim(min(-0.26, 1.3 * float(np.min(values))), max(0.64, 1.3 * float(np.max(values))))

    ledger = fixed["original"]["thomson_and_heat_ledger"]
    partition = fixed["cold_heat_partition_response"]
    common_mw = 1.0e3 * np.asarray(
        [
            ledger["p_leg"]["integrated_active_common_mode_thomson_source_w"],
            ledger["n_leg"]["integrated_active_common_mode_thomson_source_w"],
            ledger["module_net"][
                "integrated_active_common_mode_thomson_source_w"
            ],
        ]
    )
    cold_response_mw = 1.0e3 * np.asarray(
        [
            partition["p_leg_original_minus_flattened_w"],
            partition["n_leg_original_minus_flattened_w"],
            partition["module_original_minus_flattened_w"],
        ]
    )
    x = np.arange(3)
    width = 0.34
    ax_d.axhline(0.0, color="#B7BCC2", linewidth=0.9)
    bars_common = ax_d.bar(
        x - width / 2,
        common_mw,
        width,
        color=orange,
        label="distributed common-mode Thomson source",
    )
    bars_cold = ax_d.bar(
        x + width / 2,
        cold_response_mw,
        width,
        color=blue,
        label=r"cold heat response $Q_c^{original}-Q_c^{flat}$",
    )
    ax_d.set_xticks(x, ["p leg", "n leg", "module net"])
    ax_d.set_ylabel("integrated module-scale heat term (mW)")
    ax_d.set_title("Body sources cancel; full field response does not", loc="left")
    ax_d.legend(loc="lower right")
    for bars in (bars_common, bars_cold):
        for bar in bars:
            value = bar.get_height()
            if abs(value) < 0.005:
                label = "0.00"
            else:
                label = f"{value:+.1f}"
            ax_d.text(
                bar.get_x() + bar.get_width() / 2,
                value + (1.0 if value >= 0.0 else -1.0),
                label,
                ha="center",
                va="bottom" if value >= 0.0 else "top",
                fontsize=6.3,
                color="#555B63",
            )
    ax_d.set_ylim(-31.5, 32.5)

    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        axis.text(
            -0.12,
            1.08,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.5,
            va="top",
        )
        axis.tick_params(direction="out", length=3.0, width=0.7)

    comparison = analysis["optimized_forward_comparison"]
    effect = comparison["original_minus_flattened"]
    target = analysis["target_condition"]
    fig.suptitle(
        "The PbSe candidate common-mode slope is a small nominal-geometry scenario correction",
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
        f"Figure-derived candidate screen at Th={target['hot_temperature_k']:.3f} K and "
        f"Tc={target['cold_temperature_k']:.3f} K: removing dM/dT changes reoptimized "
        "Qc,max under the reported-contact sensitivity by "
        f"{1.0e3*effect['contact_corrected_Qc_max_w']:.3f} mW "
        f"({100.0*effect['fraction_of_nominal_forward_residual_capacity']:.3f}% of the "
        "nominal forward residual). Deterministic counterfactual; no confidence interval "
        "and no as-built device validation.",
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
        metadata={"Software": "matplotlib; PbSe common-mode contribution"},
    )
    plt.close(fig)


def serialize_results(analysis: dict[str, Any], figure_path: Path) -> dict[str, Any]:
    result = copy.deepcopy(analysis)
    internal = result.pop("_internal")
    inputs = internal["inputs"]
    result["input_bindings"] = {
        "figure1_transport_candidates": {
            "locator": output_locator(forward.FIG1_CSV),
            "sha256": file_sha256(forward.FIG1_CSV),
            "row_count": len(inputs["fig1_all"]),
            "rows_used_for_selected_pair_S_and_sigma": 28,
            "data_role": "primary_article_figure_derived_source_object_candidate",
        },
        "figure_s9_thermal_candidates": {
            "locator": output_locator(forward.S9_CSV),
            "sha256": file_sha256(forward.S9_CSV),
            "row_count": len(inputs["s9_all"]),
            "data_role": "figure_derived_candidate_measured_as_described_in_si",
        },
        "figure4a_endpoint": {
            "locator": output_locator(forward.FIG4_CSV),
            "sha256": file_sha256(forward.FIG4_CSV),
            "data_role": "figure_derived_measured_as_described_in_main_text",
        },
        "device_condition": {
            "locator": output_locator(forward.CONDITIONS_CSV),
            "sha256": file_sha256(forward.CONDITIONS_CSV),
            "device_id": forward.TARGET_DEVICE_ID,
            "identity_status": inputs["condition"]["identity_status"],
        },
    }
    result["figure_metadata"] = {
        "core_conclusion": (
            "The figure-derived common-mode Seebeck slope redistributes heat between the "
            "selected PbSe/Cr legs but changes nominal long-leg Qc,max by less "
            "than 0.5%, so it cannot close the nominal scenario margin."
        ),
        "evidence_chain": [
            "exact alpha-preserving M(T) flattening",
            "signed-current port-response control including I=0",
            "reoptimized capacity and COP comparison",
            "leg-resolved common-mode Thomson, Peltier-boundary, and cold-heat ledgers",
        ],
        "layout": "quantitative grid with an emphasized signed-current response panel",
        "backend": "Python matplotlib only",
        "output": "double-column PNG preview at 300 dpi",
        "statistics": (
            "deterministic controlled numerical counterfactual; no sampling, "
            "error bars, p-values, or confidence intervals"
        ),
        "limitation": (
            "figure-derived source candidates and unresolved device identity "
            "prevent as-built device validation or unique discrepancy attribution"
        ),
    }
    result["outputs"] = {
        "analysis_script": output_locator(Path(__file__).resolve()),
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
        "figure": output_locator(figure_path),
        "figure_sha256": file_sha256(figure_path),
    }
    result["software_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    return result


def run_analysis(json_output: Path, figure_output: Path) -> dict[str, Any]:
    analysis = analyze_common_mode_contribution()
    make_figure(analysis, figure_output)
    result = serialize_results(analysis, figure_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--figure-output", type=Path, default=DEFAULT_FIGURE)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    result = run_analysis(arguments.json_output, arguments.figure_output)
    comparison = result["optimized_forward_comparison"]
    original = comparison["original"]["contact_corrected_optimum"]
    flattened = comparison["flattened_common_mode"]["contact_corrected_optimum"]
    effect = comparison["original_minus_flattened"]
    print(
        "PbSe common-mode contribution complete: "
        f"original Qc,max={original['Qc_after_contact_w']:.9f} W at "
        f"{original['current_a']:.9f} A; flat Qc,max="
        f"{flattened['Qc_after_contact_w']:.9f} W at "
        f"{flattened['current_a']:.9f} A; delta="
        f"{1.0e3*effect['contact_corrected_Qc_max_w']:.6f} mW"
    )


if __name__ == "__main__":
    main()
