#!/usr/bin/env python3
"""Quantify an exact, falsifiable topology breaking of the Seebeck null mode.

The constant common-mode theorem requires opposite branch currents to meet at
one shared temperature, or equivalently requires the common shift to be made
globally over every electrically active segment.  A split-pad thermoelectric
couple deliberately violates the reduced p/n-only condition: the p and n
semiconductor endpoints on a nominal device side are held at different
temperatures while their external reference leads and connection path are left
fixed.

For a general collection of split p/n elements, the exact aggregate cold-side
heat increment is

    dQc,Sigma = sum_j C_j*I_j*(Tcp,j - Tcn,j).

Only in the special case C_j=C and I_j=I for N series elements does this reduce
to dQc,Sigma=N*C*I*mean_j(Tcp,j-Tcn,j).  The two cold endpoints are separate,
nonisothermal boundary nodes; dQc,Sigma is their aggregate heat, not heat
extracted from one isothermal cold reservoir.  The corresponding special-case
hot-side and terminal-voltage increments are

    dQh,Sigma = N*C*I*mean_j(Thp,j - Thn,j),
    dV = N*C*[mean_j(Thp,j-Thn,j) - mean_j(Tcp,j-Tcn,j)].

The script proves these identities, checks them with the existing conservative
temperature-dependent single-leg solver, maps their device-scale magnitude,
and generates one final figure.  The PbSe/Cr result file is used
only to anchor N, I, and Qc for a transparent scenario calculation; neither C
nor the split-pad temperature mismatch is claimed to have been measured in
that device.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image
import sympy as sp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tec_1d_solver.temperature_dependent import (  # noqa: E402
    LinearTemperatureProperty,
    TemperatureDependentLeg,
    solve_temperature_dependent_leg,
)


SCHEMA = "topology_breaking_split_pad/v1"
DEFAULT_BENCHMARK = (
    ROOT
    / "results/scientific_analysis/pbse_common_mode_contribution_results.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/scientific_analysis/topology_breaking_split_pad_results.json"
)
DEFAULT_FIGURE_PREFIX = (
    ROOT / "results/scientific_analysis/topology_breaking_split_pad"
)
FIXED_TIMESTAMP = datetime(2026, 8, 26, 0, 0, 0, tzinfo=timezone.utc)
FIXED_ISO_TIMESTAMP = FIXED_TIMESTAMP.isoformat()


class SplitPadError(RuntimeError):
    """Raised when a scientific identity or numerical check fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitPadError(message)


def finite(value: Any, label: str) -> float:
    number = float(value)
    require(math.isfinite(number), f"{label} is non-finite")
    return number


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locator(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def binding(path: Path) -> dict[str, Any]:
    return {
        "locator": locator(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def output_binding(path: Path) -> dict[str, Any]:
    return {
        "output_name": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def expression(value: sp.Expr) -> str:
    return sp.sstr(sp.factor(sp.simplify(value)))


def load_device_anchor(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    require(
        document.get("schema_version") == "pbse_common_mode_contribution/v1",
        "unexpected device-anchor schema",
    )
    scope = document.get("scope", {})
    require(
        scope.get("real_pbse_cr_device_validation") is False,
        "device anchor was relabeled as validation",
    )

    model = document["material_and_device_model"]
    optimum = document["optimized_forward_comparison"]["original"][
        "contact_corrected_optimum"
    ]
    target = document["target_condition"]
    pair_count = int(model["pair_count"])
    current = finite(optimum["current_a"], "anchor current")
    qc = finite(optimum["Qc_after_contact_w"], "anchor Qc")
    cold = finite(target["cold_temperature_k"], "anchor cold temperature")
    hot = finite(target["hot_temperature_k"], "anchor hot temperature")
    require(pair_count > 0 and current > 0.0 and qc > 0.0, "invalid device anchor")
    require(hot > cold > 0.0, "invalid device-anchor temperatures")
    return {
        "data_role": "figure_derived_device_scenario_scale_anchor_only",
        "pair_count": pair_count,
        "current_a": current,
        "reference_qc_w": qc,
        "cold_temperature_k": cold,
        "hot_temperature_k": hot,
        "not_measured_in_anchor": [
            "split-pad p/n endpoint-temperature mismatch",
            "constant relative common-mode displacement C",
            "split-pad topology",
        ],
    }


def build_symbolic_result() -> dict[str, Any]:
    common_shift_j, current_j = sp.symbols("C_j I_j", real=True)
    delta_tc_j, delta_th_j = sp.symbols(
        "delta_Tc_j delta_Th_j", real=True
    )
    element_delta_qc = common_shift_j * current_j * delta_tc_j
    element_delta_qh = common_shift_j * current_j * delta_th_j
    element_delta_pin = (
        common_shift_j * current_j * (delta_th_j - delta_tc_j)
    )
    element_energy_residual = sp.simplify(
        element_delta_qh - element_delta_qc - element_delta_pin
    )

    common_shift, current = sp.symbols("C I", real=True)
    pair_count = sp.symbols("N", positive=True, integer=True)
    mean_delta_tc, mean_delta_th = sp.symbols(
        "mean_delta_Tc mean_delta_Th", real=True
    )
    special_delta_qc = (
        pair_count * common_shift * current * mean_delta_tc
    )
    special_delta_qh = (
        pair_count * common_shift * current * mean_delta_th
    )
    special_delta_voltage = pair_count * common_shift * (
        mean_delta_th - mean_delta_tc
    )
    special_energy_residual = sp.simplify(
        special_delta_qh
        - special_delta_qc
        - current * special_delta_voltage
    )

    isothermal = {
        "delta_Qc_aggregate": expression(
            special_delta_qc.subs(mean_delta_tc, 0)
        ),
        "delta_Qh_aggregate": expression(
            special_delta_qh.subs(mean_delta_th, 0)
        ),
        "delta_V_series": expression(
            special_delta_voltage.subs(
                {mean_delta_tc: 0, mean_delta_th: 0}
            )
        ),
    }
    full_network_completion = {
        "missing_active_path_delta_Qc": expression(-special_delta_qc),
        "missing_active_path_delta_Qh": expression(-special_delta_qh),
        "missing_active_path_delta_V": expression(-special_delta_voltage),
        "complete_network_delta_Qc": expression(
            special_delta_qc - special_delta_qc
        ),
        "complete_network_delta_Qh": expression(
            special_delta_qh - special_delta_qh
        ),
        "complete_network_delta_V": expression(
            special_delta_voltage - special_delta_voltage
        ),
        "scope": (
            "the counterterm represents every omitted electrically active lead, "
            "interconnect, and external connection segment, not one bridge alone"
        ),
    }

    validation_c = np.asarray([50.0e-6, -20.0e-6, 80.0e-6], dtype=float)
    validation_i = np.asarray([0.8, 1.1, -0.4], dtype=float)
    validation_dtc = np.asarray([-4.0, 2.0, 5.0], dtype=float)
    validation_dth = np.asarray([0.0, 1.0, -2.0], dtype=float)
    validation_qc = float(np.sum(validation_c * validation_i * validation_dtc))
    validation_qh = float(np.sum(validation_c * validation_i * validation_dth))
    validation_pin = float(
        np.sum(validation_c * validation_i * (validation_dth - validation_dtc))
    )
    validation_residual = validation_qh - validation_qc - validation_pin

    require(
        expression(element_energy_residual) == "0",
        "elementwise symbolic energy closure failed",
    )
    require(
        expression(special_energy_residual) == "0",
        "same-C series symbolic energy closure failed",
    )
    require(set(isothermal.values()) == {"0"}, "isothermal null failed")
    require(
        full_network_completion["complete_network_delta_Qc"] == "0"
        and full_network_completion["complete_network_delta_Qh"] == "0"
        and full_network_completion["complete_network_delta_V"] == "0",
        "global co-shift completion failed",
    )
    require(
        abs(validation_residual) < 1.0e-18,
        "heterogeneous array energy validation failed",
    )
    return {
        "topology": (
            "declared split-pad/reference-contrast topology: every p/n element "
            "has two separately measured cold-side nodes and, optionally, two "
            "separately measured hot-side nodes; the reference leads and external "
            "connection path define the unshifted Seebeck reference.  Elements "
            "are boundary-resolved, so every shifted-segment endpoint and every "
            "unshifted inter-element connector is included explicitly"
        ),
        "definitions": {
            "delta_Tc_j": "T_cp,j-T_cn,j",
            "delta_Th_j": "T_hp,j-T_hn,j",
            "C_j": (
                "constant Seebeck contrast applied to both semiconductor legs "
                "of element j relative to its calibrated reference path"
            ),
            "I_j": (
                "element current with local conventions I_p,j=+I_j and "
                "I_n,j=-I_j"
            ),
            "delta_Qc_aggregate": (
                "sum of heat increments at the two separately controlled, "
                "generally nonisothermal cold-side nodes of every element; it "
                "is not heat extracted from one isothermal cold reservoir"
            ),
        },
        "general_aggregate_increments": {
            "delta_Qc_aggregate": "sum_j[C_j*I_j*delta_Tc_j]",
            "delta_Qh_aggregate": "sum_j[C_j*I_j*delta_Th_j]",
            "delta_Pin_aggregate": (
                "sum_j[C_j*I_j*(delta_Th_j-delta_Tc_j)]"
            ),
            "elementwise_energy_closure": expression(
                element_energy_residual
            ),
            "aggregate_energy_closure": (
                "delta_Qh_aggregate-delta_Qc_aggregate-"
                "delta_Pin_aggregate=0"
            ),
            "series_voltage_note": (
                "a single terminal delta_V exists for a series array; for a "
                "branched collection use the aggregate incremental electrical "
                "power sum rather than writing I*delta_V with one current"
            ),
        },
        "same_C_same_series_I_special_case": {
            "conditions": (
                "N series elements, C_j=C, I_j=I, "
                "mean_delta_Ts=(1/N)*sum_j(delta_Ts_j)"
            ),
            "delta_Qc_aggregate": expression(special_delta_qc),
            "delta_Qh_aggregate": expression(special_delta_qh),
            "delta_V_terminal": expression(special_delta_voltage),
            "energy_closure": expression(special_energy_residual),
            "cold_split_hot_isothermal": {
                "conditions": "mean_delta_Th=0",
                "delta_Qc_aggregate": "C*I*N*mean_delta_Tc",
                "delta_Qh_aggregate": "0",
                "delta_V_terminal": "-C*N*mean_delta_Tc",
                "power_identity": (
                    "delta_Qh_aggregate-delta_Qc_aggregate="
                    "I*delta_V_terminal"
                ),
            },
        },
        "strict_zero_mode_controls": {
            "shared_isothermal_endpoints": isothermal,
            "global_co_shift_including_all_electrically_active_segments": (
                full_network_completion
            ),
            "zero_current_heat_response_per_element": expression(
                element_delta_qc.subs(current_j, 0)
            ),
        },
        "parity": {
            "exact_reference_contrast": (
                "each term C_j*I_j*delta_Tc_j is odd in C_j, I_j, and "
                "delta_Tc_j separately"
            ),
            "current_odd_projection_removes": (
                "only contributions to the measured reference-state contrast "
                "that are even under current reversal"
            ),
            "current_odd_projection_does_not_remove": (
                "baseline or reference-dependent Peltier, Thomson, contact, and "
                "other feedback terms that are themselves current-odd"
            ),
        },
        "heterogeneous_three_element_energy_validation": {
            "C_j_v_per_k": validation_c.tolist(),
            "I_j_a": validation_i.tolist(),
            "delta_Tc_j_k": validation_dtc.tolist(),
            "delta_Th_j_k": validation_dth.tolist(),
            "delta_Qc_aggregate_w": validation_qc,
            "delta_Qh_aggregate_w": validation_qh,
            "delta_Pin_aggregate_w": validation_pin,
            "energy_residual_w": validation_residual,
        },
        "passed": True,
        "sympy_version": sp.__version__,
    }


def linear_property(
    value_at_300: float, slope_per_k: float
) -> LinearTemperatureProperty:
    return LinearTemperatureProperty(
        300.0, value_at_300, slope_per_k, 250.0, 450.0
    )


def synthetic_legs(common_shift_v_per_k: float) -> tuple[
    TemperatureDependentLeg, TemperatureDependentLeg
]:
    p_leg = TemperatureDependentLeg(
        seebeck=linear_property(
            220.0e-6 + common_shift_v_per_k, 0.5e-6
        ),
        electrical_resistivity=linear_property(1.1e-5, 1.0e-8),
        thermal_conductivity=linear_property(1.4, -2.0e-3),
        length_m=1.2e-3,
        area_m2=1.1e-6,
    )
    n_leg = TemperatureDependentLeg(
        seebeck=linear_property(
            -180.0e-6 + common_shift_v_per_k, -0.3e-6
        ),
        electrical_resistivity=linear_property(2.0e-5, -2.0e-8),
        thermal_conductivity=linear_property(0.9, 1.0e-3),
        length_m=0.8e-3,
        area_m2=0.7e-6,
    )
    return p_leg, n_leg


def solve_split_pair(
    *,
    common_shift_v_per_k: float,
    current_a: float,
    delta_tc_k: float,
    delta_th_k: float,
    cold_mean_k: float = 310.0,
    hot_mean_k: float = 363.0,
) -> dict[str, Any]:
    p_leg, n_leg = synthetic_legs(common_shift_v_per_k)
    p_cold = cold_mean_k + 0.5 * delta_tc_k
    n_cold = cold_mean_k - 0.5 * delta_tc_k
    p_hot = hot_mean_k + 0.5 * delta_th_k
    n_hot = hot_mean_k - 0.5 * delta_th_k
    options = {
        "initial_mesh_points": 31,
        "output_points": 201,
        "relative_tolerance": 1.0e-9,
        "max_nodes": 10000,
    }
    p_solution = solve_temperature_dependent_leg(
        p_leg, +current_a, p_cold, p_hot, **options
    )
    n_solution = solve_temperature_dependent_leg(
        n_leg, -current_a, n_cold, n_hot, **options
    )
    qc = p_solution.cold_heat_rate_w + n_solution.cold_heat_rate_w
    qh = p_solution.hot_heat_rate_w + n_solution.hot_heat_rate_w
    voltage = (
        n_solution.hot_minus_cold_potential_v
        - p_solution.hot_minus_cold_potential_v
    )
    input_power = current_a * voltage
    energy_residual = qh - qc - input_power
    return {
        "Qc_w": qc,
        "Qh_w": qh,
        "V_v": voltage,
        "Pin_w": input_power,
        "energy_residual_w": energy_residual,
        "p_temperature_k": p_solution.temperature_k,
        "n_temperature_k": n_solution.temperature_k,
        "maximum_relative_conservative_residual": max(
            p_solution.maximum_relative_conservative_residual,
            n_solution.maximum_relative_conservative_residual,
        ),
        "maximum_rms_bvp_residual": max(
            p_solution.maximum_rms_bvp_residual,
            n_solution.maximum_rms_bvp_residual,
        ),
    }


def build_numerical_validation() -> dict[str, Any]:
    shifts_uv = [-100.0, 80.0]
    currents_a = [-1.3, 0.0, 1.3]
    mismatch_pairs_k = [
        (0.0, 0.0),
        (6.0, 0.0),
        (-6.0, 0.0),
        (0.0, 4.0),
        (5.0, -3.0),
        (-5.0, 3.0),
    ]
    points: list[dict[str, Any]] = []
    maxima = {
        "absolute_Qc_prediction_error_w": 0.0,
        "absolute_Qh_prediction_error_w": 0.0,
        "absolute_voltage_prediction_error_v": 0.0,
        "absolute_incremental_energy_residual_w": 0.0,
        "temperature_field_change_k": 0.0,
        "absolute_operating_point_energy_residual_w": 0.0,
        "sampled_grid_relative_conservative_residual": 0.0,
        "maximum_rms_bvp_residual": 0.0,
    }
    for shift_uv in shifts_uv:
        common_shift = shift_uv * 1.0e-6
        for current in currents_a:
            for delta_tc, delta_th in mismatch_pairs_k:
                baseline = solve_split_pair(
                    common_shift_v_per_k=0.0,
                    current_a=current,
                    delta_tc_k=delta_tc,
                    delta_th_k=delta_th,
                )
                shifted = solve_split_pair(
                    common_shift_v_per_k=common_shift,
                    current_a=current,
                    delta_tc_k=delta_tc,
                    delta_th_k=delta_th,
                )
                computed = {
                    "delta_Qc_w": shifted["Qc_w"] - baseline["Qc_w"],
                    "delta_Qh_w": shifted["Qh_w"] - baseline["Qh_w"],
                    "delta_V_v": shifted["V_v"] - baseline["V_v"],
                }
                predicted = {
                    "delta_Qc_w": common_shift * current * delta_tc,
                    "delta_Qh_w": common_shift * current * delta_th,
                    "delta_V_v": common_shift * (delta_th - delta_tc),
                }
                incremental_energy_residual = (
                    computed["delta_Qh_w"]
                    - computed["delta_Qc_w"]
                    - current * computed["delta_V_v"]
                )
                temperature_change = max(
                    float(
                        np.max(
                            np.abs(
                                shifted["p_temperature_k"]
                                - baseline["p_temperature_k"]
                            )
                        )
                    ),
                    float(
                        np.max(
                            np.abs(
                                shifted["n_temperature_k"]
                                - baseline["n_temperature_k"]
                            )
                        )
                    ),
                )
                errors = {
                    "delta_Qc_w": computed["delta_Qc_w"]
                    - predicted["delta_Qc_w"],
                    "delta_Qh_w": computed["delta_Qh_w"]
                    - predicted["delta_Qh_w"],
                    "delta_V_v": computed["delta_V_v"]
                    - predicted["delta_V_v"],
                }
                maxima["absolute_Qc_prediction_error_w"] = max(
                    maxima["absolute_Qc_prediction_error_w"],
                    abs(errors["delta_Qc_w"]),
                )
                maxima["absolute_Qh_prediction_error_w"] = max(
                    maxima["absolute_Qh_prediction_error_w"],
                    abs(errors["delta_Qh_w"]),
                )
                maxima["absolute_voltage_prediction_error_v"] = max(
                    maxima["absolute_voltage_prediction_error_v"],
                    abs(errors["delta_V_v"]),
                )
                maxima["absolute_incremental_energy_residual_w"] = max(
                    maxima["absolute_incremental_energy_residual_w"],
                    abs(incremental_energy_residual),
                )
                maxima["temperature_field_change_k"] = max(
                    maxima["temperature_field_change_k"], temperature_change
                )
                maxima["absolute_operating_point_energy_residual_w"] = max(
                    maxima["absolute_operating_point_energy_residual_w"],
                    abs(baseline["energy_residual_w"]),
                    abs(shifted["energy_residual_w"]),
                )
                maxima["sampled_grid_relative_conservative_residual"] = max(
                    maxima["sampled_grid_relative_conservative_residual"],
                    baseline["maximum_relative_conservative_residual"],
                    shifted["maximum_relative_conservative_residual"],
                )
                maxima["maximum_rms_bvp_residual"] = max(
                    maxima["maximum_rms_bvp_residual"],
                    baseline["maximum_rms_bvp_residual"],
                    shifted["maximum_rms_bvp_residual"],
                )
                points.append(
                    {
                        "common_shift_uv_per_k": shift_uv,
                        "current_a": current,
                        "delta_Tc_k": delta_tc,
                        "delta_Th_k": delta_th,
                        "computed": computed,
                        "predicted": predicted,
                        "errors": errors,
                        "incremental_energy_residual_w": incremental_energy_residual,
                        "maximum_temperature_field_change_k": temperature_change,
                    }
                )

    require(
        maxima["absolute_Qc_prediction_error_w"] < 2.0e-12,
        "aggregate cold-side scale law failed",
    )
    require(
        maxima["absolute_Qh_prediction_error_w"] < 2.0e-12,
        "aggregate hot-side scale law failed",
    )
    require(
        maxima["absolute_voltage_prediction_error_v"] < 2.0e-12,
        "voltage scale law failed",
    )
    require(
        maxima["absolute_incremental_energy_residual_w"] < 2.0e-12,
        "incremental energy closure failed",
    )
    require(
        maxima["temperature_field_change_k"] < 1.0e-9,
        "constant shift changed a fixed-endpoint temperature field",
    )
    return {
        "data_role": "synthetic_method_validation_only",
        "solver": (
            "conservative temperature-dependent 1D single-leg BVP; p and n "
            "legs solved at independently fixed endpoints"
        ),
        "validation_point_count": len(points),
        "temperature_dependent_unequal_leg_fixture": {
            "property_support_k": [250.0, 450.0],
            "cold_mean_k": 310.0,
            "hot_mean_k": 363.0,
            "p_geometry_length_area": [1.2e-3, 1.1e-6],
            "n_geometry_length_area": [0.8e-3, 0.7e-6],
        },
        "points": points,
        "maximum_errors": maxima,
        "acceptance": {
            "port_prediction_tolerance_si": 2.0e-12,
            "temperature_invariance_tolerance_k": 1.0e-9,
            "passed": True,
        },
    }


def threshold_delta_t(
    *, pair_count: int, current_a: float, common_shift_v_per_k: float, heat_w: float
) -> float:
    denominator = pair_count * abs(current_a) * abs(common_shift_v_per_k)
    require(denominator > 0.0, "threshold denominator must be positive")
    return heat_w / denominator


def build_device_scale_map(anchor: dict[str, Any]) -> dict[str, Any]:
    pair_count = int(anchor["pair_count"])
    current = float(anchor["current_a"])
    qc_reference = float(anchor["reference_qc_w"])
    c_grid_uv = np.linspace(0.0, 120.0, 121, dtype=float)
    mismatch_grid_k = np.linspace(0.0, 10.0, 101, dtype=float)
    common_shift, mismatch = np.meshgrid(
        c_grid_uv * 1.0e-6, mismatch_grid_k, indexing="xy"
    )
    delta_qc = pair_count * common_shift * current * mismatch
    relative_percent = 100.0 * delta_qc / qc_reference

    crossing_rows: list[dict[str, Any]] = []
    for common_uv in [20.0, 50.0, 80.0, 100.0]:
        common = common_uv * 1.0e-6
        crossing_rows.append(
            {
                "common_shift_uv_per_k": common_uv,
                "delta_T_for_1_mW_operational_scale_k": threshold_delta_t(
                    pair_count=pair_count,
                    current_a=current,
                    common_shift_v_per_k=common,
                    heat_w=1.0e-3,
                ),
                "delta_T_at_1_percent_model_crossing_k": threshold_delta_t(
                    pair_count=pair_count,
                    current_a=current,
                    common_shift_v_per_k=common,
                    heat_w=0.01 * qc_reference,
                ),
            }
        )

    scenario_c_uv = 80.0
    scenario_delta_k = 5.0
    scenario_delta_qc = (
        pair_count * scenario_c_uv * 1.0e-6 * current * scenario_delta_k
    )
    scenario_delta_v = -pair_count * scenario_c_uv * 1.0e-6 * scenario_delta_k
    coherence_sensitivity = []
    for coherence in [1.0, 0.5, 0.25, 0.0]:
        contrast = coherence * scenario_delta_qc
        coherence_sensitivity.append(
            {
                "signed_coherence_factor": coherence,
                "effective_signed_mean_cold_split_k": coherence
                * scenario_delta_k,
                "delta_Qc_mw": 1.0e3 * contrast,
                "percent_of_reference_Qc": 100.0 * contrast / qc_reference,
            }
        )
    return {
        "anchor": anchor,
        "scan_definition": {
            "mathematical_special_case": (
                "C_j=C and I_j=I for all N series elements, so "
                "delta_Qc_aggregate=N*C*I*mean_delta_Tc"
            ),
            "common_shift_range_uv_per_k": [0.0, 120.0],
            "cold_split_range_k": [0.0, 10.0],
            "hot_split_k": 0.0,
            "current_a": current,
            "pair_count": pair_count,
            "reference_qc_w": qc_reference,
            "boundary_quantity": (
                "aggregate heat increment across two separately controlled, "
                "nonisothermal cold-side nodes per element"
            ),
            "not_a_single_reservoir_claim": True,
            "array_mismatch_coordinate": (
                "delta_Tc is the signed mean (1/N)*sum_j(T_cp,j-T_cn,j); "
                "the plotted magnitude therefore includes cancellation among pairs"
            ),
            "interpretation": (
                "counterfactual special-case topology scale map; the isothermal "
                "module Qc is only a magnitude normalization and the anchor device "
                "is not claimed to possess this boundary topology, C, or mismatch"
            ),
        },
        "scale_crossings": crossing_rows,
        "coherence_sensitivity_at_reference_scenario": coherence_sensitivity,
        "reference_scenario": {
            "common_shift_uv_per_k": scenario_c_uv,
            "cold_split_k": scenario_delta_k,
            "delta_Qc_w": scenario_delta_qc,
            "delta_Qc_mw": 1.0e3 * scenario_delta_qc,
            "fraction_of_reference_Qc": scenario_delta_qc / qc_reference,
            "percent_of_reference_Qc": 100.0 * scenario_delta_qc / qc_reference,
            "delta_V_module_v": scenario_delta_v,
            "delta_V_module_mv": 1.0e3 * scenario_delta_v,
            "delta_Qh_w": 0.0,
            "energy_residual_w": -scenario_delta_qc - current * scenario_delta_v,
            "signed_coherence_factor": 1.0,
            "coherence_assumption": (
                "all pairwise p-minus-n cold splits have the same 5 K sign; "
                "the reduction N*C*I*mean_delta_Tc is valid only here; arbitrary "
                "arrays require sum_j(C_j*I_j*delta_Tc_j)"
            ),
            "exceeds_1_mW_operational_threshold": scenario_delta_qc >= 1.0e-3,
            "above_1_percent_model_crossing": (
                scenario_delta_qc >= 0.01 * qc_reference
            ),
            "one_percent_language": (
                "model-scale crossing relative to an isothermal-module "
                "normalization, not an experimental detection threshold"
            ),
        },
        "source_data": {
            "common_shift_uv_per_k": c_grid_uv.tolist(),
            "cold_split_k": mismatch_grid_k.tolist(),
            "delta_Qc_mw": (1.0e3 * delta_qc).tolist(),
            "delta_Qc_percent_reference_normalization": relative_percent.tolist(),
        },
    }


def build_current_reversal(anchor: dict[str, Any]) -> dict[str, Any]:
    pair_count = int(anchor["pair_count"])
    current_reference = float(anchor["current_a"])
    common_shift = 80.0e-6
    current_grid = np.linspace(-3.5, 3.5, 141, dtype=float)
    mismatches = [0.0, 1.0, 5.0]
    traces = []
    for mismatch in mismatches:
        delta_qc = pair_count * common_shift * current_grid * mismatch
        traces.append(
            {
                "cold_split_k": mismatch,
                "current_a": current_grid.tolist(),
                "delta_Qc_mw": (1.0e3 * delta_qc).tolist(),
            }
        )

    forward = pair_count * common_shift * current_reference * 5.0
    reverse = pair_count * common_shift * (-current_reference) * 5.0
    odd = 0.5 * (forward - reverse)
    even = 0.5 * (forward + reverse)
    mismatch_reversed_forward = pair_count * common_shift * current_reference * -5.0
    double_odd = 0.25 * (
        forward
        - reverse
        - mismatch_reversed_forward
        + (-reverse)
    )
    delta_voltage = -pair_count * common_shift * 5.0
    require(abs(odd - forward) < 1.0e-15, "current-odd projection failed")
    require(abs(even) < 1.0e-15, "current-even projection failed")
    require(abs(double_odd - forward) < 1.0e-15, "double-odd projection failed")
    require(
        abs(-forward - current_reference * delta_voltage) < 1.0e-15,
        "current-reversal energy closure failed",
    )
    return {
        "common_shift_uv_per_k": 80.0,
        "special_case_scope": (
            "C_j=C, I_j=I for seven series elements, coherent signed-mean "
            "cold-node split, and isothermal hot endpoint pairs"
        ),
        "traces": traces,
        "reference_magnitude": {
            "current_a": current_reference,
            "cold_split_k": 5.0,
            "forward_delta_Qc_w": forward,
            "reverse_delta_Qc_w": reverse,
            "current_odd_projection_w": odd,
            "current_even_projection_w": even,
            "mismatch_reversed_forward_delta_Qc_w": mismatch_reversed_forward,
            "current_and_mismatch_double_odd_projection_w": double_odd,
            "delta_V_v": delta_voltage,
            "incremental_energy_residual_w": -forward
            - current_reference * delta_voltage,
        },
        "experimental_parity_prediction": (
            "The exact calibrated C-state contrast reverses with current and "
            "with the imposed endpoint mismatch; its current-even projection is zero."
        ),
        "projection_limitation": (
            "Current-odd projection removes only even-in-current terms in the "
            "reference-state contrast.  It does not by itself remove baseline "
            "Peltier or Thomson response, current-odd contact heat, or odd thermal "
            "feedback caused by imperfectly matched reference states."
        ),
    }


def build_experimental_design(
    anchor: dict[str, Any], scale_map: dict[str, Any]
) -> dict[str, Any]:
    crossing_80 = next(
        row
        for row in scale_map["scale_crossings"]
        if row["common_shift_uv_per_k"] == 80.0
    )
    return {
        "device": (
            "A p/n test couple or repeated series array has a shared isothermal "
            "hot block and two separately thermalized cold pads.  The cold p and "
            "n pads are separate electrical terminals connected to a current source "
            "through resolved reference leads; the hot endpoints are series-connected "
            "by an isothermal bridge.  Each cold pad carries a local microthermometer "
            "and differential heat-flow sensor.  In an array, record every pairwise "
            "signed mismatch because cancellations are physical."
        ),
        "practical_material_contrast": (
            "Instead of synthesizing two perfectly matched p/n pairs, compare two "
            "matched sets of terminal reference leads with calibrated Seebeck "
            "difference -C and a common remote isothermal measurement block.  This "
            "is thermodynamically equivalent to moving the semiconductor common "
            "mode by +C relative to fixed leads, but lead resistance, heat leak, "
            "and contact thermal conductance must be independently matched or corrected."
        ),
        "strict_C_switch_requirement": (
            "The exact contrast requires two calibrated reference-lead states "
            "whose Seebeck difference realizes -C while their resistance, thermal "
            "leak, contact state, and endpoint temperatures are matched or corrected. "
            "Reversing delta_T on one unchanged device is not an equivalent C switch."
        ),
        "single_device_modulation_route": {
            "status": (
                "useful experimental approximation; neither mathematically nor "
                "operationally equivalent to the exact calibrated C-state contrast"
            ),
            "protocol": (
                "On one fixed p/n pair, reverse both I and the cold-pad split and "
                "take the double-odd local cold-interface heat projection."
            ),
            "small_split_boundary_term": (
                "for T_cp=T0+delta_T/2 and T_cn=T0-delta_T/2, "
                "d[Q_Peltier,c]/d(delta_T)|0 = I*[M(T0)+Gamma(T0)], where "
                "M=(S_p+S_n)/2 and Gamma=T*dM/dT"
            ),
            "constant_property_limit": (
                "Gamma=0 gives the directly interpretable boundary slope I*M"
            ),
            "why_not_exact_for_total_port_heat": (
                "the same endpoint modulation changes conduction, branch temperature "
                "fields, Thomson transfer, and contact heat; these additional current-odd "
                "terms must be bounded with local interface calorimetry or a calibrated "
                "full thermal model"
            ),
            "invalidation": (
                "do not identify the measured double-odd total-port slope with M "
                "if it depends on split amplitude, current magnitude, hot-block "
                "temperature, or contact configuration beyond propagated uncertainty"
            ),
        },
        "measurement_sequence": [
            "stabilize the hot p/n endpoints to one measured temperature",
            "impose delta_Tc in both signs while recording T_cp and T_cn directly",
            (
                "measure each cold-node and hot-node heat rate separately, then "
                "form Qc,Sigma and Qh,Sigma together with four-terminal voltage "
                "at +I and -I"
            ),
            "repeat with both calibrated reference-lead sets and randomize lead-set order",
            "repeat at delta_Tc=0 as the strict p/n-only null control",
        ],
        "target_observables": {
            "double_odd_cold_heat_contrast": (
                "P_I,delta[Delta Qc_aggregate]="
                "sum_j[C_j*I_j*delta_Tc_j] for the calibrated C-state contrast"
            ),
            "mismatch_odd_voltage_contrast": (
                "for a series array with delta_Th_j=0, "
                "P_delta[Delta V]=-sum_j[C_j*delta_Tc_j]"
            ),
            "hot_heat_contrast": (
                "Delta Qh_aggregate=0 when every delta_Th_j=0"
            ),
            "energy_identity_general": (
                "Delta Qh_aggregate-Delta Qc_aggregate="
                "sum_j[C_j*I_j*(delta_Th_j-delta_Tc_j)]"
            ),
            "energy_identity_same_series_current": (
                "Delta Qh_aggregate-Delta Qc_aggregate=I*Delta V"
            ),
            "normalized_slope": (
                "Delta Qc_aggregate/sum_j(C_j*I_j*delta_Tc_j)=1 "
                "for every nonzero calibrated-contrast point"
            ),
        },
        "operational_detectability": {
            "definition": (
                "mW-scale means |Delta Qc_aggregate| >= 1 mW; the threshold is "
                "independent of any particular calorimeter specification"
            ),
            "at_anchor_current_and_N_with_C_80_uv_per_k": {
                "delta_T_for_1_mW_operational_scale_k": crossing_80[
                    "delta_T_for_1_mW_operational_scale_k"
                ],
            },
            "recommended_combined_standard_uncertainty_w": (
                0.2e-3
            ),
            "reason_for_uncertainty_target": (
                "gives a five-to-one signal-to-standard-uncertainty ratio at "
                "the declared 1 mW operational threshold"
            ),
            "prediction_uncertainty_law": (
                "for the plotted same-C/same-I special case with independent "
                "inputs, u_pred^2=(I*sum(delta_Tj)*u_C)^2"
                "+(C*sum(delta_Tj)*u_I)^2+(C*I)^2*sum_j[u(delta_Tj)^2]; "
                "include measured covariance terms when the same thermometers "
                "or Seebeck calibration are shared across pairs"
            ),
        },
        "model_scale_crossing": {
            "definition": (
                "the 1% line is where the declared special-case model contrast "
                "equals 1% of the isothermal-module Qc normalization; it is not "
                "an experimental threshold or significance criterion"
            ),
            "at_anchor_current_and_N_with_C_80_uv_per_k": {
                "delta_T_at_1_percent_model_crossing_k": crossing_80[
                    "delta_T_at_1_percent_model_crossing_k"
                ]
            },
        },
        "projection_boundary": {
            "current_odd_projection_removes": (
                "only measured contrast components even in current"
            ),
            "does_not_automatically_isolate": [
                "baseline Peltier heat",
                "Thomson redistribution",
                "current-odd contact Peltier heat",
                "current-odd thermal-feedback changes between reference states",
            ],
            "required_interpretation": (
                "attribute the projected result to the exact C law only after "
                "the calibrated lead-state subtraction and all odd confounders "
                "are bounded within propagated uncertainty"
            ),
        },
        "falsification_and_invalidation": [
            (
                "Reject the constant-C split-pad prediction if the fitted "
                "Delta Qc_aggregate versus sum_j(C_j*I_j*delta_Tc_j) slope "
                "differs from unity by more "
                "than three combined standard uncertainties at two or more "
                "independent nonzero operating points."
            ),
            (
                "Reject the assumed zero-mode control if every "
                "delta_Tc_j=delta_Th_j=0 "
                "produces a calibrated reference-state contrast exceeding three combined "
                "standard uncertainties."
            ),
            (
                "Invalidate the reduced thermal/electrical model if the current-"
                "even reference-lead contrast or, in the same-series-current "
                "experiment, Delta Qh_aggregate-Delta Qc_aggregate-I*Delta V exceeds "
                "three propagated standard uncertainties."
            ),
            (
                "Treat curvature with delta_Tc as evidence that C(T), contact "
                "properties, or endpoint temperatures changed; it invalidates "
                "the constant-C experiment but does not falsify the global gauge theorem."
            ),
            (
                "Do not interpret a signal as common-mode topology breaking "
                "unless lead electrical resistance, Joule partition, thermal "
                "conductance, and contact Peltier terms are independently bounded."
            ),
        ],
        "scope": {
            "real_device_observation": False,
            "anchor_device_had_split_pad_topology": False,
            "scanned_C_was_measured_for_anchor": False,
            "scanned_delta_T_was_measured_for_anchor": False,
            "temperature_dependent_C_included": False,
            "explicit_reference_lead_heat_spreading_solved": False,
            "pairwise_mismatch_distribution_measured": False,
            "single_isothermal_cold_reservoir": False,
            "unique_common_mode_breaking_mechanism_claimed": False,
            "single_device_delta_T_modulation_equivalent_to_C_switch": False,
            "result_type": (
                "exact calibrated constant-reference-contrast law for the "
                "declared split-pad topology plus synthetic numerical verification "
                "and a special-case scale map"
            ),
        },
    }


def apply_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "topology-breaking-split-pad-v1",
            "pdf.fonttype": 42,
        }
    )


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.05,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_split_pad_schematic(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(0.0, 10.0)
    axis.axis("off")
    red = "#C84E4E"
    blue = "#3D75A8"
    gray = "#6C727A"

    axis.add_patch(
        patches.FancyBboxPatch(
            (1.0, 8.15),
            8.0,
            0.9,
            boxstyle="round,pad=0.08",
            facecolor="#E8EAED",
            edgecolor=gray,
            linewidth=0.8,
        )
    )
    axis.text(5.0, 8.6, r"isothermal hot bridge/block: $T_{hp}=T_{hn}$", ha="center", va="center")
    axis.add_patch(patches.Rectangle((2.0, 3.2), 1.45, 4.95, facecolor="#F4C6C6", edgecolor=red, linewidth=1.0))
    axis.add_patch(patches.Rectangle((6.55, 2.35), 1.45, 5.8, facecolor="#C8DAEA", edgecolor=blue, linewidth=1.0))
    axis.text(2.72, 5.75, "p", color=red, fontsize=9, fontweight="bold", ha="center")
    axis.text(7.28, 5.35, "n", color=blue, fontsize=9, fontweight="bold", ha="center")
    axis.add_patch(patches.FancyBboxPatch((1.35, 2.35), 2.75, 0.75, boxstyle="round,pad=0.05", facecolor="#FBE7E7", edgecolor=red, linewidth=0.8))
    axis.add_patch(patches.FancyBboxPatch((5.9, 1.5), 2.75, 0.75, boxstyle="round,pad=0.05", facecolor="#E7EFF7", edgecolor=blue, linewidth=0.8))
    axis.text(2.72, 2.72, r"$T_{cp}$", color=red, ha="center", va="center")
    axis.text(7.28, 1.87, r"$T_{cn}$", color=blue, ha="center", va="center")
    axis.plot([0.45, 1.35], [2.72, 2.72], color=gray, linewidth=2.0, solid_capstyle="round")
    axis.plot([8.65, 9.55], [1.87, 1.87], color=gray, linewidth=2.0, solid_capstyle="round")
    axis.text(5.0, 1.25, "separate reference leads to current source", color=gray, ha="center", va="top", fontsize=5.5)
    axis.annotate("", xy=(8.55, 6.3), xytext=(8.55, 4.7), arrowprops={"arrowstyle": "->", "color": "#222222", "lw": 1.0})
    axis.text(8.78, 5.5, r"$I$", va="center")
    axis.annotate("", xy=(5.0, 1.87), xytext=(5.0, 2.72), arrowprops={"arrowstyle": "<->", "color": "#A65A00", "lw": 1.0})
    axis.text(4.75, 2.30, r"$\delta T_c$", color="#A65A00", ha="right", va="center")
    axis.text(5.0, 0.35, r"$\Delta Q_{c,\Sigma}=\sum_j C_j I_j\delta T_{c,j}$", ha="center", va="center", fontsize=8, fontweight="bold")


def render_figure(result: dict[str, Any], prefix: Path) -> list[Path]:
    apply_figure_style()
    fig = plt.figure(figsize=(6.69, 4.85), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.18], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    ax_d = fig.add_subplot(grid[1, 1])

    draw_split_pad_schematic(ax_a)
    add_panel_label(ax_a, "a")

    validation = result["numerical_validation"]["points"]
    predicted_q = []
    computed_q = []
    colors = []
    for point in validation:
        predicted_q.extend(
            [
                1.0e3 * point["predicted"]["delta_Qc_w"],
                1.0e3 * point["predicted"]["delta_Qh_w"],
                1.0e3
                * point["current_a"]
                * point["predicted"]["delta_V_v"],
            ]
        )
        computed_q.extend(
            [
                1.0e3 * point["computed"]["delta_Qc_w"],
                1.0e3 * point["computed"]["delta_Qh_w"],
                1.0e3
                * point["current_a"]
                * point["computed"]["delta_V_v"],
            ]
        )
        colors.extend(["#C84E4E", "#3D75A8", "#555B61"])
    predicted_array = np.asarray(predicted_q)
    computed_array = np.asarray(computed_q)
    limit = 1.08 * max(float(np.max(np.abs(predicted_array))), 0.1)
    ax_b.plot([-limit, limit], [-limit, limit], color="#34383C", linewidth=0.8, zorder=1)
    ax_b.scatter(predicted_array, computed_array, c=colors, s=13, alpha=0.78, edgecolors="white", linewidths=0.25, zorder=2)
    ax_b.set_xlim(-limit, limit)
    ax_b.set_ylim(-limit, limit)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("exact prediction (mW per element)")
    ax_b.set_ylabel("computed port/power increment (mW)")
    ax_b.text(0.04, 0.93, r"cold aggregate $\Delta Q_{c,\Sigma}$", color="#C84E4E", transform=ax_b.transAxes, va="top")
    ax_b.text(0.04, 0.84, r"hot aggregate $\Delta Q_{h,\Sigma}$", color="#3D75A8", transform=ax_b.transAxes, va="top")
    ax_b.text(0.04, 0.75, r"electrical $I\Delta V$", color="#555B61", transform=ax_b.transAxes, va="top")
    maximum_error = result["numerical_validation"]["maximum_errors"]["absolute_Qc_prediction_error_w"]
    ax_b.text(0.96, 0.05, f"max error\n{maximum_error:.1e} W", transform=ax_b.transAxes, ha="right", va="bottom", color="#555B61")
    add_panel_label(ax_b, "b")

    scale = result["device_scale_map"]["source_data"]
    x = np.asarray(scale["cold_split_k"], dtype=float)
    y = np.asarray(scale["common_shift_uv_per_k"], dtype=float)
    z = np.asarray(
        scale["delta_Qc_percent_reference_normalization"], dtype=float
    )
    image = ax_c.pcolormesh(x, y, z.T, shading="auto", cmap="Blues", vmin=0.0, vmax=float(np.max(z)))
    one_mw_percent = 100.0e-3 / result["device_scale_map"]["anchor"]["reference_qc_w"]
    contours = ax_c.contour(x, y, z.T, levels=[one_mw_percent, 1.0], colors=["#A65A00", "#8B1A1A"], linewidths=[1.0, 1.2])
    ax_c.clabel(
        contours,
        fmt={one_mw_percent: "1 mW", 1.0: "1% model crossing"},
        inline=True,
        fontsize=5.5,
    )
    scenario = result["device_scale_map"]["reference_scenario"]
    ax_c.scatter([scenario["cold_split_k"]], [scenario["common_shift_uv_per_k"]], marker="*", s=45, color="#8B1A1A", edgecolor="white", linewidth=0.5, zorder=4)
    ax_c.set_xlabel(
        r"signed-mean split between two cold nodes "
        r"$|\overline{\delta T_c}|$ (K)"
    )
    ax_c.set_ylabel(r"relative common shift $|C|$ ($\mu$V K$^{-1}$)")
    colorbar = fig.colorbar(image, ax=ax_c, fraction=0.048, pad=0.03)
    colorbar.set_label(
        r"$|\Delta Q_{c,\Sigma}|/Q_{c,\mathrm{ref}}$ normalization (%)"
    )
    ax_c.text(
        0.03,
        0.96,
        f"same-C/I series special case: "
        f"N={result['device_scale_map']['anchor']['pair_count']}, "
        f"|I|={result['device_scale_map']['anchor']['current_a']:.2f} A",
        transform=ax_c.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color="#30343A",
    )
    ax_c.text(
        0.97,
        0.96,
        f"star: {scenario['delta_Qc_mw']:.2f} mW\n"
        f"$\\Delta V$={scenario['delta_V_module_mv']:.2f} mV",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color="#6F1616",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    add_panel_label(ax_c, "c")

    reversal = result["current_reversal"]
    trace_colors = {0.0: "#7B8188", 1.0: "#4B83B4", 5.0: "#C84E4E"}
    for trace in reversal["traces"]:
        mismatch = float(trace["cold_split_k"])
        current_axis = np.asarray(trace["current_a"], dtype=float)
        heat_axis = np.asarray(trace["delta_Qc_mw"], dtype=float)
        ax_d.plot(current_axis, heat_axis, color=trace_colors[mismatch], linewidth=1.5 if mismatch == 5.0 else 1.1)
        ax_d.text(current_axis[-1] + 0.08, heat_axis[-1], f"{mismatch:g} K", color=trace_colors[mismatch], va="center", fontsize=6)
    ax_d.axhline(0.0, color="#AEB3B8", linewidth=0.7)
    ax_d.axvline(0.0, color="#AEB3B8", linewidth=0.7)
    ax_d.set_xlim(-3.5, 3.95)
    ax_d.set_xlabel("current I (A)")
    ax_d.set_ylabel(
        r"aggregate split-node $C$ contrast $\Delta Q_{c,\Sigma}$ (mW)"
    )
    ax_d.text(
        0.04,
        0.94,
        r"calibrated $C$ contrast odd in $I$; $\overline{\delta T_c}=0$ null",
        transform=ax_d.transAxes,
        va="top",
        fontsize=5.8,
        color="#30343A",
    )
    add_panel_label(ax_d, "d")

    prefix.parent.mkdir(parents=True, exist_ok=True)
    svg = prefix.with_suffix(".svg")
    pdf = prefix.with_suffix(".pdf")
    tiff = prefix.with_suffix(".tiff")
    png = prefix.with_suffix(".png")
    fig.savefig(
        svg,
        metadata={
            "Date": FIXED_ISO_TIMESTAMP,
            "Creator": None,
        },
    )
    fig.savefig(
        pdf,
        metadata={
            "Creator": "Python matplotlib; split-pad topology study",
            "CreationDate": FIXED_TIMESTAMP,
            "ModDate": FIXED_TIMESTAMP,
        },
    )
    fig.savefig(tiff, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(
        png,
        dpi=300,
        metadata={
            "Software": "Python matplotlib; split-pad topology study",
            "Creation Time": FIXED_ISO_TIMESTAMP,
        },
    )
    plt.close(fig)

    with Image.open(tiff) as raster:
        require(raster.width >= 3000 and raster.height >= 2000, "TIFF resolution too small")
    for path in (svg, pdf, tiff, png):
        require(path.exists() and path.stat().st_size > 1000, f"missing figure {path.name}")
    return [svg, pdf, tiff, png]


def build_result(benchmark_path: Path) -> dict[str, Any]:
    anchor = load_device_anchor(benchmark_path)
    symbolic = build_symbolic_result()
    numerical = build_numerical_validation()
    scale_map = build_device_scale_map(anchor)
    reversal = build_current_reversal(anchor)
    experimental = build_experimental_design(anchor, scale_map)
    scope = {
        "declared_exact_topology": (
            "split-pad/reference-contrast network with separately measured "
            "nonisothermal p and n boundary nodes and no unaccounted "
            "electrically active connector"
        ),
        "general_sum_requires_boundary_resolved_elements": True,
        "unique_topology_breaking_mechanism_claimed": False,
        "aggregate_cold_heat_definition": (
            "Delta Qc,Sigma is the sum over two nonisothermal cold-side nodes "
            "per element; it is not heat from one isothermal cold reservoir"
        ),
        "single_isothermal_cold_reservoir_interpretation": False,
        "exact_C_switch_requires_calibrated_reference_lead_contrast": True,
        "single_device_delta_T_modulation_equivalent_to_C_switch": False,
        "current_odd_projection_isolates_all_odd_physics": False,
        "real_device_observation": False,
        "anchor_device_validation": False,
        "anchor_Qc_role": (
            "magnitude normalization for a counterfactual special-case map only"
        ),
        "scanned_C_measured_for_anchor": False,
        "scanned_split_measured_for_anchor": False,
        "finite_reference_lead_heat_spreading_solved": False,
        "temperature_dependent_C_included": False,
    }
    return {
        "schema_version": SCHEMA,
        "analysis_id": "split_pad_common_mode_topology_breaking",
        "version_metadata": {
            "status": "REFERENCE",
            "version_id": "split-pad-2026-08-26",
            "version_timestamp": FIXED_ISO_TIMESTAMP,
            "version_note": "reference calculation dated 2026-08-26",
        },
        "title": "Nonisothermal split pads activate a constant Seebeck common mode",
        "central_scientific_result": (
            "In the declared split-pad/reference-contrast topology, nonisothermal "
            "opposite-current endpoints activate the exact aggregate boundary law "
            "Delta Q_s,Sigma=sum_j[C_j*I_j*(T_ps,j-T_ns,j)].  Shared isothermal "
            "endpoints or a global co-shift of every electrically active segment "
            "restore the exact null. This calculation evaluates one explicit "
            "breaking topology."
        ),
        "scope": scope,
        "input_bindings": {"device_scale_anchor": binding(benchmark_path)},
        "symbolic_theory": symbolic,
        "numerical_validation": numerical,
        "device_scale_map": scale_map,
        "current_reversal": reversal,
        "experimental_design": experimental,
        "figure_metadata": {
            "core_conclusion": (
                "The declared split-node/reference-contrast topology converts "
                "the constant relative common mode into the exact aggregate law "
                "sum_j(C_j*I_j*delta_T_j), while isothermal and globally shifted "
                "controls remain null."
            ),
            "layout": "schematic and device-scale map",
            "backend": "Python/matplotlib only",
            "target": "double-column scientific figure",
            "final_size_in": [6.69, 4.85],
            "panel_map": {
                "a": (
                    "two nonisothermal cold nodes and exact aggregate heat law"
                ),
                "b": "temperature-dependent solver versus exact prediction",
                "c": (
                    "same-C/same-I scale map with 1 mW line and 1% model crossing"
                ),
                "d": (
                    "exact calibrated C-contrast parity and split-node zero control"
                ),
            },
            "evidence_hierarchy": {
                "emphasis": (
                    "special-case device-scale normalization map under the exact law"
                ),
                "mechanism": "split-pad schematic",
                "validation": "independent temperature-dependent BVP solutions",
                "controls": "current parity and delta_Tc=0 trace",
            },
            "statistics": (
                "deterministic exact identities and solver tolerances; no "
                "probabilistic interval or experimental replicates"
            ),
            "limitation": (
                "The device anchor supplies normalization only; C, split nodes, "
                "and the topology are counterfactual until measured.  Current "
                "odd projection alone does not remove odd Peltier/Thomson/contact feedback."
            ),
        },
        "reproduction_record": {
            "fixed_metadata_timestamp": FIXED_ISO_TIMESTAMP,
            "deterministic_svg_hash_salt": (
                "topology-breaking-split-pad-v1"
            ),
            "byte_identity_scope": [
                "JSON",
                "SVG",
                "PDF",
                "PNG",
                "TIFF",
            ],
            "verification_test": (
                "tests.test_topology_breaking_split_pad."
                "TestTopologyBreakingSplitPad."
                "test_complete_rebuild_is_byte_identical"
            ),
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sympy": sp.__version__,
            "matplotlib": mpl.__version__,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-prefix", type=Path, default=DEFAULT_FIGURE_PREFIX)
    args = parser.parse_args()

    try:
        result = build_result(args.benchmark)
        figure_paths = render_figure(result, args.figure_prefix)
        script_path = Path(__file__).resolve()
        result["outputs"] = {
            "analysis_script": binding(script_path),
            "figures": [output_binding(path) for path in figure_paths],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(json_bytes(result))
    except (OSError, ValueError, KeyError, TypeError, SplitPadError) as exc:
        print(f"split-pad topology analysis rejected: {exc}", file=sys.stderr)
        return 1

    scenario = result["device_scale_map"]["reference_scenario"]
    print(
        "split-pad topology analysis passed; "
        f"reference contrast={scenario['delta_Qc_mw']:.3f} mW "
        f"({scenario['percent_of_reference_Qc']:.3f}% Qc); "
        f"validation points={result['numerical_validation']['validation_point_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
