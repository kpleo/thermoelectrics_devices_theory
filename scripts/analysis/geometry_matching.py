#!/usr/bin/env python3
"""Fixed-area thermoelectric geometry and common-mode response study.

All response values are first derivatives with respect to the dimensionless
amplitude epsilon of S_p,n -> S_p,n + epsilon*m(T).  Model systems distinguish
Seebeck-magnitude matching from cold-port transfer matching at fixed alpha.
Published-property systems are representative calculations, not fitted devices.
No contact resistances are included in these fixed-leg-end one-dimensional
calculations.  Geometry changes retain total semiconductor area and leg length.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import analyze_bts_bst_endpoint_zero_common_mode as bts
from scripts.analysis import analyze_common_mode_transfer_kernel as kernel
from scripts.analysis import analyze_pbse_device_forward_constraint as pbse
from scripts.tec_1d_solver import (
    ConstantTemperatureProperty,
    TemperatureDependentLeg,
    TemperatureDependentNumericalCouple,
    solve_temperature_dependent_couple,
)

OUTPUT = ROOT / "results/transport_validation/geometry"
MODE_NAMES = ("linear", "parabola", "cold_skew", "hot_skew")
DATE = "2026-10-10"


@dataclass(frozen=True)
class CommonMode:
    name: str
    value: Callable
    derivative: Callable
    gamma: Callable
    amplitude_v_per_k: float


def make_mode(name: str, cold: float, hot: float, amplitude: float = 10e-6) -> CommonMode:
    """Four smooth shapes with equal max |m| on the endpoint interval.

    The polynomial continues analytically if a leg has an interior temperature
    maximum above Th; its actual maximum on that field is recorded separately.
    """
    if hot <= cold or amplitude <= 0 or name not in MODE_NAMES:
        raise ValueError("invalid common-mode definition")
    span = hot - cold

    def value(t):
        u = (np.asarray(t, dtype=float) - cold) / span
        if name == "linear":
            return amplitude * u
        if name == "parabola":
            return amplitude * 4 * u * (1 - u)
        if name == "cold_skew":
            return amplitude * 27 / 4 * u * (1 - u) ** 2
        return amplitude * 27 / 4 * u**2 * (1 - u)

    def derivative(t):
        u = (np.asarray(t, dtype=float) - cold) / span
        if name == "linear":
            return np.ones_like(u) * amplitude / span
        if name == "parabola":
            return amplitude / span * (4 - 8 * u)
        if name == "cold_skew":
            return amplitude / span * 27 / 4 * (1 - 4 * u + 3 * u**2)
        return amplitude / span * 27 / 4 * (2 * u - 3 * u**2)

    return CommonMode(name, value, derivative, lambda t: np.asarray(t) * derivative(t), amplitude)


def _source_record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@lru_cache(maxsize=None)
def load_material_model(system: str, seebeck_case: str = "matched") -> tuple[TemperatureDependentNumericalCouple, dict]:
    """Return an immutable equal-area reference couple and source metadata."""
    if system == "constant":
        if seebeck_case not in {"matched", "unmatched"}:
            raise ValueError("seebeck_case must be matched or unmatched")
        sp, sn = (200e-6, -200e-6) if seebeck_case == "matched" else (220e-6, -180e-6)

        def leg(s, rho, kappa):
            return TemperatureDependentLeg(
                seebeck=ConstantTemperatureProperty(s, 250, 600),
                electrical_resistivity=ConstantTemperatureProperty(rho, 250, 600),
                thermal_conductivity=ConstantTemperatureProperty(kappa, 250, 600),
                length_m=1e-3, area_m2=1e-6,
            )
        couple = TemperatureDependentNumericalCouple(leg(sp, 1e-5, 1.5), leg(sn, 2e-5, 1.0), 300, 330)
        return couple, {"system": system, "current_a": 1.0, "data_role": "controlled_constant_property_model", "sources": [], "analytic_matching_ratio_n_over_p": float(np.sqrt(3)), "pair_count_for_scaling": 1}
    if system == "bts_bst":
        transport, validation = bts.validate_sources_and_load_transport()
        couple = bts.build_couple(bts.build_legs(transport), 323, 373)
        return couple, {"system": system, "current_a": 3.0, "data_role": "published_figure_derived_properties", "doi": bts.DOI, "sources": [_source_record(bts.TRANSPORT_CSV), _source_record(bts.SOURCE_RECORD_PATH)], "property_domain_k": validation["curve_completeness"]["temperature_domain_k"], "pair_count_for_scaling": 7}
    if system == "pbse":
        inputs = pbse.load_inputs()
        hot = float(inputs["target"]["hot_side_temperature_k"])
        cold = hot - float(inputs["target"]["delta_t_max_k"])
        prior = ROOT / "results/scientific_analysis/pbse_common_mode_contribution_results.json"
        current = json.loads(prior.read_text())["fixed_current_mechanism"]["reference_current_a"]
        couple = pbse.build_couple(inputs, pbse.SCENARIOS["nominal"], cold, hot)
        return couple, {"system": system, "current_a": current, "data_role": "published_figure_derived_candidate_properties", "sources": [_source_record(p) for p in [pbse.FIG1_CSV, pbse.S9_CSV, pbse.FIG4_CSV, pbse.CONDITIONS_CSV, prior]], "pair_count_for_scaling": 7, "source_limitation": "Published material and module labels do not uniquely bind these curves to an as-built module."}
    raise ValueError(f"unknown system {system}")


def rescale_geometry(couple: TemperatureDependentNumericalCouple, area_ratio_n_over_p: float) -> TemperatureDependentNumericalCouple:
    """Vary only area ratio while retaining total area and each leg length."""
    ratio = float(area_ratio_n_over_p)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("area ratio must be positive")
    total = couple.p_leg.area_m2 + couple.n_leg.area_m2
    return replace(couple, p_leg=replace(couple.p_leg, area_m2=total / (1 + ratio)), n_leg=replace(couple.n_leg, area_m2=total * ratio / (1 + ratio)))


def perturbed_couple(couple, mode: CommonMode, epsilon: float):
    return bts.build_couple((couple.p_leg, couple.n_leg), couple.cold_temperature_k, couple.hot_temperature_k, epsilon=epsilon, mode=mode.value, mode_derivative=mode.derivative)


def tight_solve(couple, current: float, output_points: int = 1001):
    return solve_temperature_dependent_couple(couple, current, initial_mesh_points=61, output_points=output_points, relative_tolerance=1e-9, max_nodes=16000)


def measure_mismatch(response: dict, bins: int = 256) -> dict:
    """Binned oriented mu(B)=integral 1_B(T) psi dT, valid at turning points.

    Segmentwise linear interpolation integrates psi exactly within each
    temperature bin.  The L1 distance is a resolution-dependent diagnostic,
    not a proof of equality of continuous measures.
    """
    temps = [response[k]["temperature_k"] for k in ("p_leg", "n_leg")]
    edges = np.linspace(min(map(np.min, temps)), max(map(np.max, temps)), bins + 1)
    masses = []
    for key in ("p_leg", "n_leg"):
        t = response[key]["temperature_k"]
        psi = response[key]["collection_function"].value
        mass = np.zeros(bins)
        for ta, tb, pa, pb in zip(t[:-1], t[1:], psi[:-1], psi[1:]):
            if abs(tb - ta) < 1e-14:
                continue
            lower, upper = sorted((ta, tb))
            left = max(0, int(np.searchsorted(edges, lower, side="right") - 1))
            right = min(bins - 1, int(np.searchsorted(edges, upper, side="left")))
            for j in range(left, right + 1):
                a, b = max(lower, edges[j]), min(upper, edges[j + 1])
                if b <= a:
                    continue
                psi_a = pa + (pb - pa) * (a - ta) / (tb - ta)
                psi_b = pa + (pb - pa) * (b - ta) / (tb - ta)
                mass[j] += np.sign(tb - ta) * 0.5 * (psi_a + psi_b) * (b - a)
        masses.append(mass)
    l1 = float(np.sum(np.abs(masses[0] - masses[1])))
    scale = float(0.5 * np.sum(np.abs(masses[0]) + np.abs(masses[1])))
    return {"measure_l1_k": l1, "measure_l1_normalized": l1 / scale if scale else 0.0, "measure_bins": bins}


def evaluate_geometry(system: str, ratio: float, seebeck_case: str = "matched", *, profile_rows: list | None = None, output_points: int = 1001, measure_bins: int = 256):
    base, metadata = load_material_model(system, seebeck_case)
    couple = rescale_geometry(base, ratio)
    current = metadata["current_a"]
    point = tight_solve(couple, current, output_points=output_points)
    temperatures = np.concatenate([point.p_leg.temperature_k, point.n_leg.temperature_k])
    record = {"system": system, "seebeck_case": seebeck_case if system == "constant" else "published", "area_ratio_n_over_p": float(ratio), "p_area_m2": couple.p_leg.area_m2, "n_area_m2": couple.n_leg.area_m2, "p_length_m": couple.p_leg.length_m, "n_length_m": couple.n_leg.length_m, "current_a": current, "cold_temperature_k": couple.cold_temperature_k, "hot_temperature_k": couple.hot_temperature_k, "qc_per_pair_w": point.Qc_w, "qh_per_pair_w": point.Qh_w, "voltage_per_pair_v": point.V_v, "input_power_per_pair_w": point.Pin_w, "cop": point.COP, "relative_energy_residual": point.relative_energy_residual, "temperature_minimum_k": float(np.min(temperatures)), "temperature_maximum_k": float(np.max(temperatures)), "p_monotone": bool(np.all(point.p_leg.temperature_gradient_k_per_m > 0)), "n_monotone": bool(np.all(point.n_leg.temperature_gradient_k_per_m > 0)), "status": "success"}
    mode_records = []
    for index, name in enumerate(MODE_NAMES):
        mode = make_mode(name, couple.cold_temperature_k, couple.hot_temperature_k)
        response = kernel.first_order_port_response(couple, point, mode_value=mode.value, gamma_value=mode.gamma, port="cold")
        if index == 0:
            record.update(measure_mismatch(response, bins=measure_bins))
            if profile_rows is not None:
                for branch in ("p_leg", "n_leg"):
                    for u, t, psi in zip(response[branch]["collection_function"].normalized_coordinate, response[branch]["temperature_k"], response[branch]["collection_function"].value):
                        profile_rows.append({"system": system, "seebeck_case": record["seebeck_case"], "area_ratio_n_over_p": ratio, "branch": branch, "normalized_coordinate": float(u), "temperature_k": float(t), "collection_function": float(psi)})
        p = response["p_leg"]["distributed_kernel_derivative_w"]
        n = response["n_leg"]["distributed_kernel_derivative_w"]
        total = response["pair_total_port_derivative_w"]
        mode_records.append({**record, "mode": name, "mode_peak_on_endpoint_interval_v_per_k": mode.amplitude_v_per_k, "mode_max_abs_on_field_v_per_k": float(np.max(np.abs(mode.value(temperatures)))), "p_distributed_derivative_w": p, "n_distributed_derivative_w": n, "direct_endpoint_pair_derivative_w": response["pair_direct_endpoint_derivative_w"], "delta_qc_per_pair_w": total, "delta_qc_scaled_module_w": total * metadata["pair_count_for_scaling"], "surviving_branch_fraction": abs(total) / max(abs(p) + abs(n), 1e-30)})
    return record, mode_records


def central_difference(system, ratio, seebeck_case, mode_name, epsilon):
    base, metadata = load_material_model(system, seebeck_case)
    couple = rescale_geometry(base, ratio)
    mode = make_mode(mode_name, couple.cold_temperature_k, couple.hot_temperature_k)
    baseline = tight_solve(couple, metadata["current_a"])
    response = kernel.first_order_port_response(couple, baseline, mode_value=mode.value, gamma_value=mode.gamma, port="cold")
    plus = tight_solve(perturbed_couple(couple, mode, epsilon), metadata["current_a"])
    minus = tight_solve(perturbed_couple(couple, mode, -epsilon), metadata["current_a"])
    value = (plus.Qc_w - minus.Qc_w) / (2 * epsilon)
    target = response["pair_total_port_derivative_w"]
    return {"system": system, "seebeck_case": seebeck_case if system == "constant" else "published", "area_ratio_n_over_p": ratio, "mode": mode_name, "epsilon": epsilon, "adjoint_derivative_w": target, "central_difference_derivative_w": value, "absolute_error_w": abs(value - target), "error_over_sum_abs_branch_derivatives": abs(value - target) / max(abs(response["p_leg"]["distributed_kernel_derivative_w"]) + abs(response["n_leg"]["distributed_kernel_derivative_w"]), 1e-30), "maximum_relative_energy_residual": max(p.relative_energy_residual for p in [baseline, plus, minus]), "status": "success"}


def _write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_study(output: Path = OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    rows, geometry_rows, failures, profiles, fd_rows, optima, resolutions = [], [], [], [], [], [], []
    cache = {}

    def evaluate(system, ratio, case="matched"):
        key = (system, float(ratio), case)
        if key not in cache:
            cache[key] = evaluate_geometry(system, ratio, case)
        return cache[key]

    grids = {"constant": np.unique(np.r_[np.geomspace(.5, 3.2, 21), 1, np.sqrt(3)]), "bts_bst": np.unique(np.r_[np.geomspace(.4, 2.5, 21), 1]), "pbse": np.unique(np.r_[np.geomspace(.6, 1.8, 13), 1])}
    for system, grid in grids.items():
        print(f"Scanning {system}: {len(grid)} area ratios", flush=True)
        for ratio in grid:
            try:
                g, m = evaluate(system, ratio)
                geometry_rows.append(g)
                rows.extend(m)
            except Exception as exc:
                failures.append({"stage": "scan", "system": system, "area_ratio_n_over_p": float(ratio), "error_type": type(exc).__name__, "message": str(exc)})
        _write_csv(output / "geometry_scan.csv", geometry_rows)
        _write_csv(output / "common_mode_response_scan.csv", rows)

    # Full two-by-two design: same alpha, two Seebeck-magnitude arrangements,
    # and matched/unmatched transfer. Both geometries use dissimilar rho/kappa.
    factorial = []
    for case in ("matched", "unmatched"):
        for ratio in (1., float(np.sqrt(3))):
            _, records = evaluate("constant", ratio, case)
            factorial.extend([{**r, "transfer_case": "matched" if ratio != 1 else "unmatched"} for r in records])
    _write_csv(output / "constant_two_by_two.csv", factorial)

    for system in grids:
        valid = sorted([g for g in geometry_rows if g["system"] == system], key=lambda g: g["area_ratio_n_over_p"])
        for mode in MODE_NAMES:
            index = MODE_NAMES.index(mode)
            samples = [next(r for r in rows if r["system"] == system and r["area_ratio_n_over_p"] == g["area_ratio_n_over_p"] and r["mode"] == mode) for g in valid]
            roots = []
            for a, b in zip(samples[:-1], samples[1:]):
                if a["delta_qc_per_pair_w"] * b["delta_qc_per_pair_w"] < 0:
                    try:
                        root = brentq(lambda r: evaluate(system, r)[1][index]["delta_qc_per_pair_w"], a["area_ratio_n_over_p"], b["area_ratio_n_over_p"], xtol=1e-7)
                        if not roots or abs(root - roots[-1]) > 1e-5:
                            roots.append(float(root))
                    except Exception as exc:
                        failures.append({"stage": "root", "system": system, "mode": mode, "message": str(exc)})
            best = min(samples, key=lambda r: abs(r["delta_qc_per_pair_w"]))
            for root in roots or [best["area_ratio_n_over_p"]]:
                g, records = evaluate(system, root)
                optima.append({**g, "objective": mode, "kind": "bracketed_root" if roots else "smallest_magnitude_on_grid", **{f"{r['mode']}_delta_qc_w": r["delta_qc_per_pair_w"] for r in records}})
        best = min(valid, key=lambda g: g["measure_l1_normalized"])
        index = valid.index(best)
        lo = valid[max(0, index - 1)]["area_ratio_n_over_p"]
        hi = valid[min(len(valid) - 1, index + 1)]["area_ratio_n_over_p"]
        minimum = minimize_scalar(lambda log_r: evaluate(system, np.exp(log_r))[0]["measure_l1_normalized"], bounds=(np.log(lo), np.log(hi)), method="bounded", options={"xatol": 1e-5})
        ratio = float(np.exp(minimum.x))
        g, records = evaluate(system, ratio)
        optima.append({**g, "objective": "binned_measure_l1", "kind": "local_minimum_within_scan_bracket", **{f"{r['mode']}_delta_qc_w": r["delta_qc_per_pair_w"] for r in records}})
        targeted_roots = [r["area_ratio_n_over_p"] for r in optima if r["system"] == system and r["objective"] == "parabola" and r["kind"] == "bracketed_root"]
        target = float(np.sqrt(3)) if system == "constant" else min(targeted_roots, key=lambda r: abs(np.log(r))) if targeted_roots else ratio
        for chosen in dict.fromkeys((1., ratio, target)):
            evaluate_geometry(system, chosen, profile_rows=profiles)
            for name in MODE_NAMES:
                for epsilon in (.03, .01, .003):
                    try:
                        fd_rows.append(central_difference(system, chosen, "matched", name, epsilon))
                    except Exception as exc:
                        failures.append({"stage": "central_difference", "system": system, "area_ratio_n_over_p": chosen, "mode": name, "epsilon": epsilon, "message": str(exc)})
        for output_points, bins in ((501, 256), (1001, 128), (1001, 256), (1001, 512), (2001, 256)):
            resolved, modes = evaluate_geometry(system, ratio, output_points=output_points, measure_bins=bins)
            resolutions.append({"system": system, "area_ratio_n_over_p": ratio, "output_points": output_points, "measure_bins": bins, "measure_l1_k": resolved["measure_l1_k"], "measure_l1_normalized": resolved["measure_l1_normalized"], **{f"{m['mode']}_delta_qc_w": m["delta_qc_per_pair_w"] for m in modes}})
        print(f"Completed {system}; minimum measure mismatch {g['measure_l1_normalized']:.4g} at An/Ap={ratio:.5g}", flush=True)
        _write_csv(output / "mode_specific_roots_and_measure_minima.csv", optima)
        _write_csv(output / "central_difference_checks.csv", fd_rows)

    _write_csv(output / "collection_profiles.csv", profiles)
    _write_csv(output / "measure_and_quadrature_resolution.csv", resolutions)
    material_rows = []
    for system in grids:
        base, _ = load_material_model(system)
        for branch in ("p_leg", "n_leg"):
            leg = getattr(base, branch)
            for temperature in np.linspace(leg.minimum_valid_temperature_k, leg.maximum_valid_temperature_k, 101):
                material_rows.append({"system": system, "branch": branch, "temperature_k": float(temperature), "seebeck_v_per_k": float(leg.seebeck.evaluate([temperature])[0]), "resistivity_ohm_m": float(leg.electrical_resistivity.evaluate([temperature])[0]), "conductivity_w_per_m_k": float(leg.thermal_conductivity.evaluate([temperature])[0])})
    _write_csv(output / "sampled_material_properties.csv", material_rows)
    summary = {"schema_version": "geometry_matching/v1", "study_date": DATE, "solver": "conservative one-dimensional temperature-dependent BVP with independent adjoint and nonlinear central differences", "sources": {s: load_material_model(s)[1] for s in grids}, "fixed_constraints": ["total semiconductor cross-sectional area", "each leg length", "current", "cold/hot leg-end temperatures", "material properties", "fixed reference leads"], "perturbations": {"definition": "S_p,n(T) -> S_p,n(T) + epsilon*m(T)", "amplitude_v_per_k": 10e-6, "normalization": "max |m| over [Tc,Th]", "modes": list(MODE_NAMES), "endpoint_zero_modes": list(MODE_NAMES[1:]), "note": "A finite set of modes is not a proof of cancellation for every continuous Gamma(T)."}, "interpretation": ["Constant-property matched Seebeck pair has zero baseline M but generally nonzero response to a common perturbation.", "Transfer cancellation is not equivalent to maximum cooling capacity or COP.", "Measure distance is binned and resolution dependent; roots are mode specific unless complete measures match.", "Published-property calculations have ideal fixed leg ends and no contact layers; they do not reproduce an experimental module."], "counts": {"geometries": len(geometry_rows), "response_rows": len(rows), "factorial_rows": len(factorial), "central_difference_checks": len(fd_rows), "failures": len(failures)}, "max_fd_absolute_error_w": max(r["absolute_error_w"] for r in fd_rows), "max_relative_energy_residual": max(r["maximum_relative_energy_residual"] for r in fd_rows), "optima": optima, "failures": failures}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["counts"]), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run_study(args.output)
