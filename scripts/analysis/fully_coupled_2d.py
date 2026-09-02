#!/usr/bin/env python3
"""Fully coupled temperature-dependent two-dimensional thermoelectric check.

This calculation includes fully temperature-dependent electrical and thermal
transport in the two-dimensional model. At every nonlinear residual evaluation it solves

    div[sigma(T,x,y) grad(Psi)] = 0

at prescribed total branch current, reconstructs the divergence-free current
field and Joule heating, and then evaluates

    -div[kappa(T,x,y) grad(T)] - rho(T,x,y)|J|^2
      + div[J G_tau(T)] = 0,

where ``G_tau'(T)=tau(T)``.  The finite-volume Thomson term is a conservative
face divergence.  A temperature perturbation therefore changes ``sigma``, the
current distribution, Joule heating, ``kappa``, the conduction operator, and
the Thomson term in one nonlinear residual.

The electrical variables are eliminated exactly at each temperature state.
The central-difference Jacobian of this reduced residual is consequently the
complete electrothermal Schur-complement Jacobian, including the ``sigma_T``
current-redistribution term.  The cold-port adjoint built from that Jacobian is
checked against independent nonlinear +/- common-mode solves.  This is a
numerical physics check on a synthetic heterogeneous geometry; it is not a
PbSe parameterization or experimental validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.validate_independent_2d_common_mode import (  # noqa: E402
    BranchConfig,
    CommonModeBasis,
    Grid2D,
    ThermalAssembly,
    _pair_configs,
    assemble_thermal,
    assemble_thomson_divergence,
    material_fields,
    solve_electrical,
)


FloatArray = NDArray[np.float64]

SCHEMA_VERSION = "fully_coupled_2d/v1"
ANALYSIS_ID = "FULLY-COUPLED-2D-20260826"
DEFAULT_JSON = ROOT / "results/scientific_analysis/fully_coupled_2d.json"


class Coupled2DError(RuntimeError):
    """Raised when a physical or numerical acceptance criterion fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Coupled2DError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locator(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


@dataclass(frozen=True)
class TemperatureLaw2D:
    reference_temperature_k: float = 325.0
    rho_log_slope_per_k: float = 3.2e-3
    k_log_slope_per_k: float = -2.4e-3

    def rho(self, reference: FloatArray, temperature_k: FloatArray) -> FloatArray:
        return np.asarray(
            reference
            * np.exp(
                self.rho_log_slope_per_k
                * (temperature_k - self.reference_temperature_k)
            ),
            dtype=float,
        )

    def kappa(self, reference: FloatArray, temperature_k: FloatArray) -> FloatArray:
        return np.asarray(
            reference
            * np.exp(
                self.k_log_slope_per_k
                * (temperature_k - self.reference_temperature_k)
            ),
            dtype=float,
        )


@dataclass(frozen=True)
class CoupledState:
    temperature_k: FloatArray
    epsilon: float
    residual_max_w: float
    residual_relative: float
    nonlinear_function_evaluations: int
    converged: bool


@dataclass(frozen=True)
class EvaluatedState:
    residual_w: FloatArray
    electrical: Any
    thermal: ThermalAssembly
    rho_ohm_m: FloatArray
    k_w_per_mk: FloatArray
    thomson_divergence_w: FloatArray


@dataclass(frozen=True)
class FullyCoupledBranch:
    config: BranchConfig
    grid: Grid2D
    rho_reference_ohm_m: FloatArray
    k_reference_w_per_mk: FloatArray
    law: TemperatureLaw2D
    cold_temperature_k: float
    hot_temperature_k: float
    base_basis: CommonModeBasis
    common_basis: CommonModeBasis

    @property
    def residual_scale_w(self) -> float:
        area = self.config.width_m * self.config.depth_m
        conduction = (
            self.config.k0_w_per_mk
            * area
            * (self.hot_temperature_k - self.cold_temperature_k)
            / self.config.length_m
        )
        return max(conduction / self.grid.size, 1.0e-8)


def build_model(
    config: BranchConfig,
    nx: int,
    ny: int,
    *,
    cold_temperature_k: float = 300.0,
    hot_temperature_k: float = 350.0,
) -> FullyCoupledBranch:
    grid = Grid2D(nx, ny, config.length_m, config.width_m, config.depth_m)
    rho_ref, k_ref = material_fields(grid, config)
    base_basis = CommonModeBasis(
        name=f"{config.name}_base",
        kind="linear_gamma",
        anchor_k=300.0,
        amplitude=config.seebeck_base_slope_v_per_k2,
    )
    common_basis = CommonModeBasis(
        name="shared_linear_common_mode",
        kind="linear_gamma",
        anchor_k=300.0,
        amplitude=1.0e-6,
    )
    return FullyCoupledBranch(
        config=config,
        grid=grid,
        rho_reference_ohm_m=rho_ref,
        k_reference_w_per_mk=k_ref,
        law=TemperatureLaw2D(),
        cold_temperature_k=float(cold_temperature_k),
        hot_temperature_k=float(hot_temperature_k),
        base_basis=base_basis,
        common_basis=common_basis,
    )


def _thomson_divergence(
    model: FullyCoupledBranch,
    temperature_k: FloatArray,
    electrical: Any,
    thermal: ThermalAssembly,
    epsilon: float,
) -> FloatArray:
    matrix, boundary_weights = assemble_thomson_divergence(
        model.grid, electrical, thermal
    )
    primitive = model.base_basis.thomson_primitive(temperature_k)
    perturbation = model.common_basis.thomson_primitive(temperature_k)
    result = np.asarray(matrix @ (primitive + epsilon * perturbation), dtype=float)
    for iy in range(model.grid.ny):
        cold_index = model.grid.index(0, iy)
        hot_index = model.grid.index(model.grid.nx - 1, iy)
        cold = model.cold_temperature_k
        hot = model.hot_temperature_k
        result[cold_index] += boundary_weights[cold_index] * float(
            model.base_basis.thomson_primitive(cold)
            + epsilon * model.common_basis.thomson_primitive(cold)
        )
        result[hot_index] += boundary_weights[hot_index] * float(
            model.base_basis.thomson_primitive(hot)
            + epsilon * model.common_basis.thomson_primitive(hot)
        )
    return result


def evaluate_state(
    model: FullyCoupledBranch,
    temperature_k: FloatArray,
    epsilon: float,
    *,
    frozen_electrical: Any | None = None,
    frozen_k_w_per_mk: FloatArray | None = None,
) -> EvaluatedState:
    temperature = np.asarray(temperature_k, dtype=float)
    require(temperature.shape == (model.grid.size,), "temperature shape mismatch")
    require(bool(np.all(np.isfinite(temperature))), "temperature is non-finite")
    shaped_temperature = temperature.reshape(model.grid.nx, model.grid.ny)
    rho = model.law.rho(model.rho_reference_ohm_m, shaped_temperature)
    kappa = model.law.kappa(model.k_reference_w_per_mk, shaped_temperature)
    if frozen_k_w_per_mk is not None:
        kappa = np.asarray(frozen_k_w_per_mk, dtype=float)
    require(float(np.min(rho)) > 0.0, "rho lost positivity")
    require(float(np.min(kappa)) > 0.0, "kappa lost positivity")
    electrical = (
        frozen_electrical
        if frozen_electrical is not None
        else solve_electrical(model.grid, rho, model.config.signed_current_a)
    )
    thermal = assemble_thermal(
        model.grid,
        kappa,
        model.config,
        model.cold_temperature_k,
        model.hot_temperature_k,
    )
    thomson = _thomson_divergence(
        model, temperature, electrical, thermal, epsilon
    )
    residual = np.asarray(
        thermal.matrix @ temperature
        - thermal.boundary_rhs_w
        - electrical.joule_power_by_cell_w
        + thomson,
        dtype=float,
    )
    return EvaluatedState(
        residual_w=residual,
        electrical=electrical,
        thermal=thermal,
        rho_ohm_m=rho,
        k_w_per_mk=kappa,
        thomson_divergence_w=thomson,
    )


def initial_temperature(model: FullyCoupledBranch) -> FloatArray:
    fraction = model.grid.x_centres / model.config.length_m
    one_d = model.cold_temperature_k + fraction * (
        model.hot_temperature_k - model.cold_temperature_k
    )
    return np.repeat(one_d[:, None], model.grid.ny, axis=1).reshape(-1)


def solve_coupled(
    model: FullyCoupledBranch,
    epsilon: float,
    initial: FloatArray | None = None,
) -> CoupledState:
    guess = initial_temperature(model) if initial is None else np.asarray(initial, dtype=float)
    scale = model.residual_scale_w

    def scaled_residual(value: FloatArray) -> FloatArray:
        return evaluate_state(model, value, epsilon).residual_w / scale

    solution = root(
        scaled_residual,
        guess,
        method="krylov",
        options={"fatol": 2.0e-10, "maxiter": 120, "disp": False},
    )
    temperature = np.asarray(solution.x, dtype=float)
    evaluated = evaluate_state(model, temperature, epsilon)
    residual_max = float(np.max(np.abs(evaluated.residual_w)))
    relative = residual_max / scale
    require(bool(solution.success), f"coupled nonlinear solve failed: {solution.message}")
    require(relative < 5.0e-9, f"coupled residual too large: {relative:.3e}")
    require(float(np.min(temperature)) > 0.0, "non-physical coupled temperature")
    return CoupledState(
        temperature_k=temperature,
        epsilon=float(epsilon),
        residual_max_w=residual_max,
        residual_relative=relative,
        nonlinear_function_evaluations=int(solution.nfev),
        converged=True,
    )


def _seebeck(
    model: FullyCoupledBranch,
    temperature_k: FloatArray,
    epsilon: float,
) -> FloatArray:
    return np.asarray(
        model.config.seebeck_base_v_per_k
        + model.base_basis.mode(temperature_k)
        + epsilon * model.common_basis.mode(temperature_k),
        dtype=float,
    )


def _seebeck_primitive(
    model: FullyCoupledBranch,
    temperature_k: FloatArray,
    epsilon: float,
) -> FloatArray:
    return np.asarray(
        model.config.seebeck_base_v_per_k * temperature_k
        + model.base_basis.seebeck_primitive_increment(temperature_k)
        + epsilon * model.common_basis.seebeck_primitive_increment(temperature_k),
        dtype=float,
    )


def metrics(
    model: FullyCoupledBranch,
    temperature_k: FloatArray,
    epsilon: float,
) -> dict[str, float]:
    evaluated = evaluate_state(model, temperature_k, epsilon)
    grid = model.grid
    temperature = np.asarray(temperature_k, dtype=float).reshape(grid.nx, grid.ny)
    electrical = evaluated.electrical
    thermal = evaluated.thermal
    cold = np.full(grid.ny, model.cold_temperature_k, dtype=float)
    hot = np.full(grid.ny, model.hot_temperature_k, dtype=float)
    ix = electrical.ix_faces_a
    qc_peltier = float(np.sum(_seebeck(model, cold, epsilon) * cold * ix[0, :]))
    qh_peltier = float(np.sum(_seebeck(model, hot, epsilon) * hot * ix[-1, :]))
    qc_conduction = float(
        np.sum(thermal.cold_conductance_w_per_k * (cold - temperature[0, :]))
    )
    qh_conduction = float(
        np.sum(thermal.hot_conductance_w_per_k * (temperature[-1, :] - hot))
    )
    q_side_bottom = float(
        np.sum(
            thermal.bottom_conductance_w_per_k
            * (temperature[:, 0] - thermal.bottom_ambient_k)
        )
    )
    q_side_top = float(
        np.sum(
            thermal.top_conductance_w_per_k
            * (temperature[:, -1] - thermal.top_ambient_k)
        )
    )
    seebeck_power = float(
        np.sum(
            ix[-1, :] * _seebeck_primitive(model, hot, epsilon)
            - ix[0, :] * _seebeck_primitive(model, cold, epsilon)
        )
    )
    electrical_power = electrical.joule_power_total_w + seebeck_power
    qc = qc_peltier + qc_conduction
    qh = qh_peltier + qh_conduction
    qside = q_side_bottom + q_side_top
    energy_residual = qh - qc + qside - electrical_power
    return {
        "Qc_w": qc,
        "Qh_w": qh,
        "Qside_w": qside,
        "P_electrical_w": electrical_power,
        "Joule_power_w": float(electrical.joule_power_total_w),
        "Seebeck_power_w": seebeck_power,
        "energy_residual_w": float(energy_residual),
        "energy_residual_relative": float(
            abs(energy_residual)
            / max(abs(qc) + abs(qh) + abs(qside) + abs(electrical_power), 1.0e-30)
        ),
        "electrical_resistance_ohm": float(electrical.effective_resistance_ohm),
        "current_divergence_max_a": float(electrical.divergence_max_a),
        "terminal_current_mismatch_a": float(electrical.terminal_current_mismatch_a),
        "rho_minimum_ohm_m": float(np.min(evaluated.rho_ohm_m)),
        "rho_maximum_ohm_m": float(np.max(evaluated.rho_ohm_m)),
        "k_minimum_w_per_mk": float(np.min(evaluated.k_w_per_mk)),
        "k_maximum_w_per_mk": float(np.max(evaluated.k_w_per_mk)),
        "temperature_minimum_k": float(np.min(temperature)),
        "temperature_maximum_k": float(np.max(temperature)),
    }


def central_jacobian(
    function: Callable[[FloatArray], FloatArray],
    state: FloatArray,
    *,
    relative_step: float = 2.0e-5,
) -> FloatArray:
    state = np.asarray(state, dtype=float)
    baseline = np.asarray(function(state), dtype=float)
    jacobian = np.zeros((baseline.size, state.size), dtype=float)
    for column in range(state.size):
        step = relative_step * max(abs(float(state[column])), 100.0)
        plus = state.copy()
        minus = state.copy()
        plus[column] += step
        minus[column] -= step
        jacobian[:, column] = (
            np.asarray(function(plus), dtype=float)
            - np.asarray(function(minus), dtype=float)
        ) / (2.0 * step)
    return jacobian


def central_vector_derivative(
    function: Callable[[float], FloatArray],
    *,
    step: float,
) -> FloatArray:
    return (
        np.asarray(function(step), dtype=float)
        - np.asarray(function(-step), dtype=float)
    ) / (2.0 * step)


def analyze_branch(
    model: FullyCoupledBranch,
    *,
    epsilon_step: float = 2.0e-3,
) -> dict[str, Any]:
    baseline = solve_coupled(model, 0.0)
    base_evaluation = evaluate_state(model, baseline.temperature_k, 0.0)
    base_metrics = metrics(model, baseline.temperature_k, 0.0)

    full_jacobian = central_jacobian(
        lambda temperature: evaluate_state(model, temperature, 0.0).residual_w,
        baseline.temperature_k,
    )
    frozen_electrical_jacobian = central_jacobian(
        lambda temperature: evaluate_state(
            model,
            temperature,
            0.0,
            frozen_electrical=base_evaluation.electrical,
        ).residual_w,
        baseline.temperature_k,
    )
    frozen_k_jacobian = central_jacobian(
        lambda temperature: evaluate_state(
            model,
            temperature,
            0.0,
            frozen_k_w_per_mk=base_evaluation.k_w_per_mk,
        ).residual_w,
        baseline.temperature_k,
    )
    residual_epsilon = central_vector_derivative(
        lambda epsilon: evaluate_state(
            model, baseline.temperature_k, epsilon
        ).residual_w,
        step=epsilon_step,
    )

    objective_gradient = central_jacobian(
        lambda temperature: np.asarray(
            [metrics(model, temperature, 0.0)["Qc_w"]], dtype=float
        ),
        baseline.temperature_k,
    ).reshape(-1)
    objective_direct = float(
        central_vector_derivative(
            lambda epsilon: np.asarray(
                [metrics(model, baseline.temperature_k, epsilon)["Qc_w"]],
                dtype=float,
            ),
            step=epsilon_step,
        )[0]
    )
    tangent = np.linalg.solve(full_jacobian, -residual_epsilon)
    adjoint = np.linalg.solve(full_jacobian.T, objective_gradient)
    adjoint_derivative = float(objective_direct - adjoint @ residual_epsilon)
    tangent_derivative = float(objective_direct + objective_gradient @ tangent)

    minus = solve_coupled(model, -epsilon_step, baseline.temperature_k)
    plus = solve_coupled(model, epsilon_step, baseline.temperature_k)
    minus_metrics = metrics(model, minus.temperature_k, -epsilon_step)
    plus_metrics = metrics(model, plus.temperature_k, epsilon_step)
    nonlinear_derivative = float(
        (plus_metrics["Qc_w"] - minus_metrics["Qc_w"])
        / (2.0 * epsilon_step)
    )

    minus_electrical = evaluate_state(
        model, minus.temperature_k, -epsilon_step
    ).electrical
    plus_electrical = evaluate_state(
        model, plus.temperature_k, epsilon_step
    ).electrical
    current_derivative_x = (
        plus_electrical.ix_faces_a - minus_electrical.ix_faces_a
    ) / (2.0 * epsilon_step)
    current_derivative_y = (
        plus_electrical.iy_faces_a - minus_electrical.iy_faces_a
    ) / (2.0 * epsilon_step)
    joule_derivative = float(
        (plus_electrical.joule_power_total_w - minus_electrical.joule_power_total_w)
        / (2.0 * epsilon_step)
    )
    current_derivative_norm = float(
        math.sqrt(
            np.sum(current_derivative_x**2) + np.sum(current_derivative_y**2)
        )
    )
    baseline_current_norm = float(
        math.sqrt(
            np.sum(base_evaluation.electrical.ix_faces_a**2)
            + np.sum(base_evaluation.electrical.iy_faces_a**2)
        )
    )

    sv = np.linalg.svd(full_jacobian, compute_uv=False)
    relative_error = abs(adjoint_derivative - nonlinear_derivative) / max(
        abs(adjoint_derivative), 1.0e-30
    )
    tangent_adjoint_error = abs(adjoint_derivative - tangent_derivative) / max(
        abs(adjoint_derivative), 1.0e-30
    )
    electrothermal_jacobian_fraction = float(
        np.linalg.norm(full_jacobian - frozen_electrical_jacobian)
        / np.linalg.norm(full_jacobian)
    )
    kappa_jacobian_fraction = float(
        np.linalg.norm(full_jacobian - frozen_k_jacobian)
        / np.linalg.norm(full_jacobian)
    )

    return {
        "branch": model.config.name,
        "signed_current_a": float(model.config.signed_current_a),
        "grid": {
            "nx": model.grid.nx,
            "ny": model.grid.ny,
            "cells": model.grid.size,
        },
        "constitutive_temperature_dependence": {
            "rho_log_slope_per_k": model.law.rho_log_slope_per_k,
            "k_log_slope_per_k": model.law.k_log_slope_per_k,
            "rho_fractional_change_per_50_k": float(
                math.exp(50.0 * model.law.rho_log_slope_per_k) - 1.0
            ),
            "k_fractional_change_per_50_k": float(
                math.exp(50.0 * model.law.k_log_slope_per_k) - 1.0
            ),
        },
        "baseline": {
            "nonlinear_function_evaluations": baseline.nonlinear_function_evaluations,
            "residual_max_w": baseline.residual_max_w,
            "residual_relative": baseline.residual_relative,
            **base_metrics,
        },
        "adjoint": {
            "direct_Qc_derivative_w": objective_direct,
            "field_Qc_derivative_w": float(-adjoint @ residual_epsilon),
            "total_Qc_derivative_w": adjoint_derivative,
            "tangent_Qc_derivative_w": tangent_derivative,
            "nonlinear_central_Qc_derivative_w": nonlinear_derivative,
            "adjoint_to_nonlinear_relative_error": float(relative_error),
            "adjoint_to_tangent_relative_error": float(tangent_adjoint_error),
            "epsilon_step": epsilon_step,
        },
        "complete_jacobian_diagnostics": {
            "smallest_singular_value_w_per_k": float(np.min(sv)),
            "largest_singular_value_w_per_k": float(np.max(sv)),
            "condition_number": float(np.max(sv) / np.min(sv)),
            "sigma_T_current_redistribution_jacobian_fraction": (
                electrothermal_jacobian_fraction
            ),
            "kappa_T_conduction_jacobian_fraction": kappa_jacobian_fraction,
            "current_field_derivative_l2_a_per_epsilon": current_derivative_norm,
            "relative_current_field_derivative_per_epsilon": float(
                current_derivative_norm / max(baseline_current_norm, 1.0e-30)
            ),
            "joule_power_derivative_w_per_epsilon": joule_derivative,
        },
        "perturbed_states": {
            "minus": {
                "residual_relative": minus.residual_relative,
                "nonlinear_function_evaluations": minus.nonlinear_function_evaluations,
                **minus_metrics,
            },
            "plus": {
                "residual_relative": plus.residual_relative,
                "nonlinear_function_evaluations": plus.nonlinear_function_evaluations,
                **plus_metrics,
            },
        },
    }


def analyze_grid(nx: int, ny: int) -> dict[str, Any]:
    p_config, n_config = _pair_configs(1.10, "property_contrast", 1.0)
    models = [build_model(p_config, nx, ny), build_model(n_config, nx, ny)]
    branches = [analyze_branch(model) for model in models]
    pair_adjoint = float(
        sum(record["adjoint"]["total_Qc_derivative_w"] for record in branches)
    )
    pair_nonlinear = float(
        sum(
            record["adjoint"]["nonlinear_central_Qc_derivative_w"]
            for record in branches
        )
    )
    pair_relative = abs(pair_adjoint - pair_nonlinear) / max(
        abs(pair_adjoint), 1.0e-30
    )
    return {
        "grid": {"nx": nx, "ny": ny, "cells_per_branch": nx * ny},
        "branches": branches,
        "pair_adjoint_Qc_derivative_w": pair_adjoint,
        "pair_nonlinear_central_Qc_derivative_w": pair_nonlinear,
        "pair_adjoint_to_nonlinear_relative_error": float(pair_relative),
        "maximum_energy_residual_relative": float(
            max(
                record[state]["energy_residual_relative"]
                for record in branches
                for state in ("baseline",)
            )
        ),
        "maximum_all_state_energy_residual_relative": float(
            max(
                [
                    record["baseline"]["energy_residual_relative"]
                    for record in branches
                ]
                + [
                    record["perturbed_states"][sign]["energy_residual_relative"]
                    for record in branches
                    for sign in ("minus", "plus")
                ]
            )
        ),
        "minimum_sigma_T_jacobian_fraction": float(
            min(
                record["complete_jacobian_diagnostics"][
                    "sigma_T_current_redistribution_jacobian_fraction"
                ]
                for record in branches
            )
        ),
        "minimum_relative_current_field_derivative": float(
            min(
                record["complete_jacobian_diagnostics"][
                    "relative_current_field_derivative_per_epsilon"
                ]
                for record in branches
            )
        ),
    }


def build_results() -> dict[str, Any]:
    grids = [(8, 5), (12, 7), (16, 9)]
    records = []
    for nx, ny in grids:
        print(f"running fully coupled grid {nx}x{ny}", flush=True)
        records.append(analyze_grid(nx, ny))
    p_config, _ = _pair_configs(1.10, "property_contrast", 1.0)
    negative_model = build_model(p_config, 8, 5)
    negative_model = replace(
        negative_model,
        law=replace(negative_model.law, rho_log_slope_per_k=0.0),
    )
    print("running sigma_T=0 current-redistribution negative control", flush=True)
    negative_branch = analyze_branch(negative_model)
    negative_sigma_fraction = negative_branch["complete_jacobian_diagnostics"][
        "sigma_T_current_redistribution_jacobian_fraction"
    ]
    negative_current_response = negative_branch["complete_jacobian_diagnostics"][
        "relative_current_field_derivative_per_epsilon"
    ]
    medium = records[-2]["pair_adjoint_Qc_derivative_w"]
    fine = records[-1]["pair_adjoint_Qc_derivative_w"]
    grid_change = abs(fine - medium) / max(abs(fine), 1.0e-30)
    max_adjoint_error = max(
        record["pair_adjoint_to_nonlinear_relative_error"] for record in records
    )
    max_energy = max(
        record["maximum_all_state_energy_residual_relative"] for record in records
    )
    minimum_sigma_fraction = min(
        record["minimum_sigma_T_jacobian_fraction"] for record in records
    )
    minimum_current_response = min(
        record["minimum_relative_current_field_derivative"] for record in records
    )
    minimum_singular = min(
        branch["complete_jacobian_diagnostics"]["smallest_singular_value_w_per_k"]
        for record in records
        for branch in record["branches"]
    )
    checks = {
        "all_nonlinear_states_converged": True,
        "energy_closure_pass": bool(max_energy < 2.0e-8),
        "adjoint_vs_independent_nonlinear_pass": bool(max_adjoint_error < 2.0e-4),
        "medium_to_fine_grid_change_pass": bool(grid_change < 1.5e-2),
        "complete_reduced_jacobian_nonsingular": bool(minimum_singular > 1.0e-9),
        "sigma_T_current_redistribution_resolved": bool(
            minimum_sigma_fraction > 1.0e-5 and minimum_current_response > 1.0e-6
        ),
        "sigma_T_zero_negative_control_pass": bool(
            negative_sigma_fraction < 1.0e-10
            and negative_current_response < 1.0e-10
        ),
    }
    require(all(checks.values()), f"fully coupled 2-D checks failed: {checks}")
    inputs = [
        Path(__file__),
        ROOT / "scripts/analysis/validate_independent_2d_common_mode.py",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "scientific_question": (
            "Does the common-mode cold-port adjoint remain valid when sigma(T) "
            "redistributes the two-dimensional current field and kappa(T) changes "
            "the conduction operator?"
        ),
        "model": {
            "electrical_equation": "div[sigma(T,x,y) grad(Psi)]=0 at fixed total current",
            "thermal_equation": (
                "-div[kappa(T,x,y)grad(T)]-rho(T,x,y)|J|^2+div[J G_tau(T)]=0"
            ),
            "jacobian": (
                "Central derivative of the reduced residual after an exact electrical "
                "solve at every temperature state; this is the full electrothermal "
                "Schur complement and includes dJ/dT from sigma_T."
            ),
            "geometry_and_data_role": (
                "Synthetic heterogeneous two-dimensional branches; not fitted to "
                "PbSe, BTS/BST, or an as-built device."
            ),
        },
        "grid_records": records,
        "sigma_T_zero_negative_control": {
            "grid": negative_branch["grid"],
            "rho_log_slope_per_k": 0.0,
            "sigma_T_current_redistribution_jacobian_fraction": float(
                negative_sigma_fraction
            ),
            "relative_current_field_derivative_per_epsilon": float(
                negative_current_response
            ),
            "interpretation": (
                "With rho_T set exactly to zero, the fixed-current electrical "
                "field is independent of temperature and both current-redistribution "
                "diagnostics collapse to numerical zero."
            ),
        },
        "summary": {
            "fine_grid_pair_adjoint_Qc_derivative_w": float(fine),
            "medium_to_fine_pair_adjoint_relative_change": float(grid_change),
            "maximum_pair_adjoint_to_nonlinear_relative_error": float(
                max_adjoint_error
            ),
            "maximum_energy_residual_relative": float(max_energy),
            "minimum_sigma_T_jacobian_fraction": float(minimum_sigma_fraction),
            "minimum_relative_current_field_derivative_per_epsilon": float(
                minimum_current_response
            ),
            "minimum_reduced_jacobian_singular_value_w_per_k": float(
                minimum_singular
            ),
        },
        "validation_checks": checks,
        "scope": {
            "supported_result": (
                "A conservative heterogeneous 2-D example with temperature-dependent "
                "sigma and kappa, including sigma_T-driven current redistribution in "
                "the complete electrothermal Jacobian, reproduces the common-mode "
                "cold-port adjoint response."
            ),
            "limitation": (
                "The synthetic calculation validates a particular PbSe/BTS device, "
                "contact stack, radiation model, or three-dimensional package."
            ),
        },
        "input_bindings": {
            locator(path): sha256_file(path) for path in inputs if path.is_file()
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


def write_output(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_results()
    write_output(result, args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
