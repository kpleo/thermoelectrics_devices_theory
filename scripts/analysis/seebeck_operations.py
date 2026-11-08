#!/usr/bin/env python3
"""Fixed-reference Seebeck operations and cold-port response checks.

The nonlinear boundary-value solutions check common and differential leg
perturbations at fixed current and fixed material-end temperatures. The global
constant shift is checked separately by analytic reference-lead integration;
that check is not a finite-element simulation of a complete instrument.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.integrate import simpson

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import geometry_matching as study
from scripts.analysis.analyze_bts_bst_endpoint_zero_common_mode import CommonModePerturbedProperty
from scripts.analysis.analyze_common_mode_transfer_kernel import first_order_port_response
from scripts.tec_1d_solver import PchipTemperatureProperty

OUTPUT = ROOT / "results/transport_validation/operations"


def mode(name: str, cold: float, hot: float, amplitude: float = 10e-6):
    """Return m or d in V/K, with its temperature derivative."""
    if name == "constant":
        return (lambda t: np.full_like(np.asarray(t, dtype=float), amplitude),
                lambda t: np.zeros_like(np.asarray(t, dtype=float)))
    if name == "affine":
        return (lambda t: amplitude * (1 + (np.asarray(t) - cold) / (hot - cold)),
                lambda t: np.full_like(np.asarray(t, dtype=float), amplitude / (hot - cold)))
    item = study.make_mode(name, cold, hot, amplitude)
    return item.value, item.derivative


def perturb(couple, value, derivative, epsilon: float, operation: str):
    weights = {"common": (1., 1.), "differential": (.5, -.5)}[operation]
    legs = []
    for leg, weight in zip((couple.p_leg, couple.n_leg), weights):
        seebeck = CommonModePerturbedProperty(leg.seebeck, epsilon=epsilon * weight,
                                             mode=value, mode_derivative=derivative)
        legs.append(replace(leg, seebeck=seebeck))
    return replace(couple, p_leg=legs[0], n_leg=legs[1])


def operation_check(couple, current: float, operation: str, shape: str,
                    epsilons=(.02, .01, .005)) -> dict:
    value, derivative = mode(shape, couple.cold_temperature_k, couple.hot_temperature_k)
    baseline = study.tight_solve(couple, current)
    response = first_order_port_response(couple, baseline, mode_value=value,
                                         gamma_value=lambda t: np.asarray(t)*derivative(t), port="cold")
    # Integrate the measures before multiplying by signed branch currents.
    # This explicitly checks the measure sum in the differential formula.
    moments = []
    for name, solution in (("p_leg", baseline.p_leg), ("n_leg", baseline.n_leg)):
        t = solution.temperature_k
        moments.append(float(simpson(response[name]["collection_function"].value * t *
                                     derivative(t) * solution.temperature_gradient_k_per_m,
                                     x=solution.coordinate_m)))
    cold = couple.cold_temperature_k
    if operation == "common":
        endpoint = 0.
        distributed = current * (moments[0] - moments[1])
    elif operation == "differential":
        endpoint = current * cold * float(value(np.asarray([cold]))[0])
        distributed = .5 * current * (moments[0] + moments[1])
    else:
        raise ValueError(operation)
    target = endpoint + distributed
    comparisons = []
    for epsilon in epsilons:
        plus = study.tight_solve(perturb(couple, value, derivative, epsilon, operation), current)
        minus = study.tight_solve(perturb(couple, value, derivative, -epsilon, operation), current)
        observed = (plus.Qc_w - minus.Qc_w) / (2*epsilon)
        comparisons.append({"epsilon": epsilon, "nonlinear_central_derivative_w": observed,
                            "absolute_error_w": abs(observed-target),
                            "max_relative_energy_residual": max(baseline.relative_energy_residual,
                                plus.relative_energy_residual, minus.relative_energy_residual)})
    grid = np.linspace(couple.cold_temperature_k, couple.hot_temperature_k, 101)
    shifted = perturb(couple, value, derivative, 1., operation)
    dp = shifted.p_leg.seebeck.evaluate(grid)-couple.p_leg.seebeck.evaluate(grid)
    dn = shifted.n_leg.seebeck.evaluate(grid)-couple.n_leg.seebeck.evaluate(grid)
    expected_alpha = np.zeros_like(grid) if operation == "common" else value(grid)
    expected_common = value(grid) if operation == "common" else np.zeros_like(grid)
    return {"operation": operation, "shape": shape, "current_a": current,
            "cold_temperature_k": cold, "hot_temperature_k": couple.hot_temperature_k,
            "direct_endpoint_derivative_w": endpoint, "distributed_derivative_w": distributed,
            "predicted_total_derivative_w": target, "gamma_measure_moments_v": moments,
            "alpha_identity_max_error_v_per_k": float(np.max(np.abs(dp-dn-expected_alpha))),
            "common_identity_max_error_v_per_k": float(np.max(np.abs((dp+dn)/2-expected_common))),
            "checks": comparisons}


def global_reference_check(shift: float = 25e-6) -> dict:
    """Integrate -C dT through a sample and both reference leads."""
    temperatures = (298., 310., 363., 298.)
    labels = ("reference_to_cold_lead", "nonisothermal_sample", "hot_to_reference_lead")
    coordinate = np.linspace(0., 1., 1001)
    records = []
    for index, (a, b, label) in enumerate(zip(temperatures[:-1], temperatures[1:], labels)):
        # A nonuniform temperature field ensures the test is path-based.
        t = a+(b-a)*coordinate+2*np.sin(np.pi*coordinate)
        gradient = (b-a)+2*np.pi*np.cos(np.pi*coordinate)
        s = (5e-6+0*t) if index != 1 else 200e-6+.3e-6*(t-310.)
        before = float(simpson(-s*gradient, x=coordinate))
        after = float(simpson(-(s+shift)*gradient, x=coordinate))
        expected = -shift*(b-a)
        records.append({"segment": label, "start_temperature_k": a, "end_temperature_k": b,
                        "voltage_before_v": before, "voltage_after_v": after,
                        "voltage_change_v": after-before, "analytic_voltage_change_v": expected,
                        "absolute_error_v": abs(after-before-expected)})
    closed_change = sum(row["voltage_change_v"] for row in records)
    return {"method": "analytic reference-circuit relation with numerical line integration; not network FEM",
            "shift_v_per_k": shift, "measurement_reference_temperature_k": temperatures[0],
            "segments": records, "closed_measurement_voltage_change_v": closed_change,
            "bare_nonisothermal_sample_voltage_change_v": records[1]["voltage_change_v"],
            "node_heat_shift_sum_w": float(shift*310.*sum((1., -1.))),
            "interpretation": "The bare nonisothermal voltage changes by -C DeltaT. Including both shifted leads to one reference temperature cancels that change."}


def zero_current_counterexample() -> dict:
    couple, _ = study.load_material_model("constant")
    knots = np.asarray([250., 300., 330., 400., 600.])
    conductivity = 1.5*(1+.4*(knots-300.)/30.)
    couple = replace(couple, p_leg=replace(couple.p_leg,
                     thermal_conductivity=PchipTemperatureProperty(knots, conductivity)))
    baseline = study.tight_solve(couple, 0.)
    value, derivative = mode("affine", 300., 330.)
    response = first_order_port_response(couple, baseline, mode_value=value,
                                         gamma_value=lambda t: t*derivative(t), port="cold")
    mismatch = study.measure_mismatch(response)
    controls = [operation_check(couple, 0., "common", name, epsilons=(.02,))
                for name in ("constant", "affine", "parabola")]
    return {"current_a": 0., "p_kappa_description": "1.5*(1+0.4*(T-300)/30) W/(m K)",
            "n_kappa_w_per_m_k": 1., **mismatch,
            "maximum_temperature_profile_difference_k": float(np.max(np.abs(
                baseline.p_leg.temperature_k-baseline.n_leg.temperature_k))),
            "common_operation_checks": controls,
            "interpretation": "Unequal measures have zero common-mode heat response at I=0; nonzero current is required for the necessity statement."}


def constant_measure_matching() -> dict:
    base, _ = study.load_material_model("constant", "unmatched")
    couple = study.rescale_geometry(base, np.sqrt(3))
    records = []
    for current in (1., 3.):
        baseline = study.tight_solve(couple, current)
        value, derivative = mode("affine", 300., 330.)
        response = first_order_port_response(couple, baseline, mode_value=value,
                                             gamma_value=lambda t: t*derivative(t), port="cold")
        psis = [response[x]["collection_function"].value for x in ("p_leg", "n_leg")]
        u = baseline.p_leg.coordinate_m/couple.p_leg.length_m
        records.append({"current_a": current, **study.measure_mismatch(response),
                        "max_temperature_profile_difference_k": float(np.max(np.abs(
                            baseline.p_leg.temperature_k-baseline.n_leg.temperature_k))),
                        "max_adjoint_difference": float(np.max(np.abs(psis[0]-psis[1]))),
                        "max_adjoint_error_against_1_minus_u": float(max(np.max(np.abs(p-(1-u))) for p in psis)),
                        "internal_temperature_maximum": bool(np.max(baseline.p_leg.temperature_k)>330.+1e-8),
                        "checks": [operation_check(couple, current, "common", name, epsilons=(.01, .005, .002))
                                   for name in ("affine", "parabola", "cold_skew", "hot_skew")]})
    return {"area_ratio_n_over_p": float(np.sqrt(3)), "alpha_v_per_k": 400e-6,
            "seebeck_values_v_per_k": [220e-6, -180e-6], "records": records,
            "analytic_reason": "Constant S, rho, k and matched R/K give equal g, T_p(u)=T_n(u), and psi_p=psi_n=1-u; the complete oriented measures agree. This proves first-order cancellation for every continuous Gamma, including nonmonotonic baselines.",
            "scope": "First-order response at fixed nonzero current and fixed shared endpoints; finite-amplitude cancellation is not inferred."}


def run(output: Path = OUTPUT) -> dict:
    base, _ = study.load_material_model("constant")
    checks = [operation_check(base, 1., operation, shape)
              for operation in ("common", "differential")
              for shape in ("constant", "affine", "parabola")]
    bts, _ = study.load_material_model("bts_bst")
    bts_checks = [operation_check(bts, 3., "differential", shape, epsilons=(.01, .005))
                  for shape in ("constant", "affine")]
    data = {"schema_version": 1, "date": "2026-10-10",
            "units": "Seebeck perturbations in V/K; responses in W per pair per unit epsilon",
            "common_formula": "deltaQc=I integral Gamma_m d(mu_p-mu_n)",
            "differential_formula": "deltaQc=I Tc d(Tc)+(I/2) integral T d'(T) d(mu_p+mu_n)",
            "fixed_conditions": "Total current and shared material-end temperatures fixed; reference leads unchanged for leg operations.",
            "constant_property_operations": checks, "bts_bst_differential_operations": bts_checks,
            "global_constant_shift": global_reference_check(),
            "zero_current_counterexample": zero_current_counterexample(),
            "constant_property_measure_matching": constant_measure_matching()}
    matching = [check for row in data["constant_property_measure_matching"]["records"] for check in row["checks"]]
    data["validation"] = {
        "maximum_leg_operation_central_difference_error_w": max(x["absolute_error_w"] for r in checks+bts_checks for x in r["checks"]),
        "maximum_matched_case_finest_step_error_w": max(r["checks"][-1]["absolute_error_w"] for r in matching),
        "maximum_relative_energy_residual": max(x["max_relative_energy_residual"] for r in checks+bts_checks+matching for x in r["checks"]),
        "global_reference_voltage_change_v": data["global_constant_shift"]["closed_measurement_voltage_change_v"],
        "zero_current_measure_difference_normalized": data["zero_current_counterexample"]["measure_l1_normalized"],
        "note": "Near cancellation, absolute errors are used. Finite centered steps show O(epsilon^2) corrections and do not establish exact finite-amplitude cancellation."}
    if data["validation"]["maximum_leg_operation_central_difference_error_w"] > 1e-9:
        raise AssertionError("leg-operation derivative check failed")
    if data["validation"]["maximum_matched_case_finest_step_error_w"] > 1e-9:
        raise AssertionError("matched-case derivative check failed")
    data["runtime"] = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__}
    data["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), ROOT/"scripts/analysis/geometry_matching.py",
                  ROOT/"scripts/analysis/analyze_common_mode_transfer_kernel.py",
                  ROOT/"scripts/analysis/analyze_bts_bst_endpoint_zero_common_mode.py",
                  ROOT/"scripts/tec_1d_solver/temperature_dependent.py",
                  ROOT/"data/raw/nwae448_bts_bst/nwae448_digitized_transport.csv")}
    output.mkdir(parents=True, exist_ok=True)
    (output/"operation_checks.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = run(args.output)
    print(json.dumps({"output": str(args.output),
                      "operation_cases": len(data["constant_property_operations"])+len(data["bts_bst_differential_operations"]),
                      "maximum_error_w": max(x["absolute_error_w"] for r in
                          data["constant_property_operations"]+data["bts_bst_differential_operations"] for x in r["checks"])}, indent=2))
