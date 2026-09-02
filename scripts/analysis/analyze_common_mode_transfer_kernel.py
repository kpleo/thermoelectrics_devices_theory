#!/usr/bin/env python3
"""Derive and verify the first-order common-mode-to-cold-port transfer kernel.

The analysis starts from the production one-dimensional thermoelectric equation

    [K(T) T']' + r(T) I_i**2 - tau(T) I_i T' = 0,

where ``K=kappa*A``, ``r=rho/A``, and the signed branch currents are
``I_p=+I`` and ``I_n=-I``.  A shared Seebeck perturbation
``S_i -> S_i + epsilon*m(T)`` changes the Thomson coefficient by
``Gamma(T)=T*m'(T)``.  Linearization about an unperturbed branch gives

    L_i y_i = I_i Gamma(T_i) T_i',       y_i(0)=y_i(L_i)=0.

The cold-port adjoint collection function satisfies

    K_i psi_i'' + I_i tau_i psi_i' + r_i,T I_i**2 psi_i = 0,
    psi_i(0)=1, psi_i(L_i)=0.

Green's identity then gives the exact first variation

    dH_c,i/depsilon = I_i T_c,i m(T_c,i)
                       + integral psi_i I_i Gamma(T_i) T_i' dx.

For shared isothermal p/n ports the direct terms cancel.  If both temperature
fields are monotone, the pair response becomes

    dQc/depsilon = I integral Gamma(T) [psi_p(x_p(T))-psi_n(x_n(T))] dT.

Thus the necessary-and-sufficient condition for cancellation for every
admissible ``Gamma`` is equality of the two *oriented* temperature-space
collection measures ``mu_i(B)=integral 1_B(T_i) psi_i dT_i``.  The signed
branch currents then form the total measure ``I(mu_p-mu_n)``.  Pointwise
equality of the two collection functions in temperature coordinates is the
monotone special case.  Identical raw legs are sufficient but not necessary.

The script verifies this result against the temperature-dependent solver, an
independent constant-property closed form, the finite thermal-contact boundary
network, a deliberately non-isothermal topology, and the figure-derived PbSe/Cr
candidate.  The PbSe calculation remains a deterministic scenario screen; it
is not promoted to an as-built device validation.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Editable text is mandatory for the primary SVG artifact.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["svg.hashsalt"] = "common-mode-transfer-kernel-v1"

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid, simpson, solve_ivp
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import analyze_pbse_common_mode_contribution as common  # noqa: E402
from scripts.analysis import analyze_pbse_device_forward_constraint as forward  # noqa: E402
from scripts.tec_1d_solver import (  # noqa: E402
    BoundaryNetworkSolverOptions,
    ConstantTemperatureProperty,
    FixedCurrentBoundaryNetwork,
    PchipTemperatureProperty,
    ReservoirThermalContacts,
    SeriesElectricalContacts,
    SYNTHETIC_DATA_ROLE,
    TemperatureDependentLeg,
    TemperatureDependentNumericalCouple,
    TemperatureDependentOperatingPoint,
    TwoReservoirParasitic,
    solve_fixed_current_boundary_network,
    solve_temperature_dependent_couple,
    solve_temperature_dependent_leg,
)


FloatArray = NDArray[np.float64]

SCHEMA_VERSION = "common_mode_transfer_kernel/v1"
ANALYSIS_ID = "SCI-COMMON-MODE-TRANSFER-KERNEL-20260826"
DEFAULT_JSON = (
    ROOT / "results/scientific_analysis/common_mode_transfer_kernel_results.json"
)
DEFAULT_FIGURE_STEM = (
    ROOT / "results/scientific_analysis/common_mode_transfer_kernel"
)
PBSE_COMMON_RESULTS = (
    ROOT / "results/scientific_analysis/pbse_common_mode_contribution_results.json"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_locator(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _constant_property(value: float, minimum: float = 250.0, maximum: float = 500.0):
    return ConstantTemperatureProperty(value, minimum, maximum)


class AnalyticCommonModePerturbedProperty(PchipTemperatureProperty):
    """Add ``epsilon*[offset+b(T-anchor)]`` to an existing Seebeck law."""

    def __init__(
        self,
        source: Any,
        *,
        epsilon: float,
        offset_v_per_k: float,
        slope_v_per_k2: float,
        anchor_temperature_k: float,
    ) -> None:
        super().__init__(
            (source.minimum_temperature_k, source.maximum_temperature_k),
            (0.0, 0.0),
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "epsilon", float(epsilon))
        object.__setattr__(self, "offset_v_per_k", float(offset_v_per_k))
        object.__setattr__(self, "slope_v_per_k2", float(slope_v_per_k2))
        object.__setattr__(self, "anchor_temperature_k", float(anchor_temperature_k))

    def evaluate(self, temperature_k: object) -> FloatArray:
        PchipTemperatureProperty.evaluate(self, temperature_k)
        temperature = np.asarray(temperature_k, dtype=float)
        mode = self.offset_v_per_k + self.slope_v_per_k2 * (
            temperature - self.anchor_temperature_k
        )
        return np.asarray(
            self.source.evaluate(temperature)
            + self.epsilon * mode,
            dtype=float,
        )

    def derivative(self, temperature_k: object) -> FloatArray:
        PchipTemperatureProperty.derivative(self, temperature_k)
        temperature = np.asarray(temperature_k, dtype=float)
        return np.asarray(
            self.source.derivative(temperature)
            + self.epsilon * np.full_like(
                temperature, self.slope_v_per_k2, dtype=float
            ),
            dtype=float,
        )


class EmpiricalCommonModePerturbedProperty(PchipTemperatureProperty):
    """Add an exact fraction of the empirical p/n common mode to one branch."""

    def __init__(
        self,
        base: Any,
        p_source: Any,
        n_source: Any,
        *,
        epsilon: float,
        anchor_temperature_k: float,
    ) -> None:
        minimum = max(
            base.minimum_temperature_k,
            p_source.minimum_temperature_k,
            n_source.minimum_temperature_k,
        )
        maximum = min(
            base.maximum_temperature_k,
            p_source.maximum_temperature_k,
            n_source.maximum_temperature_k,
        )
        super().__init__((minimum, maximum), (0.0, 0.0))
        anchor = 0.5 * float(
            p_source.evaluate([anchor_temperature_k])[0]
            + n_source.evaluate([anchor_temperature_k])[0]
        )
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "p_source", p_source)
        object.__setattr__(self, "n_source", n_source)
        object.__setattr__(self, "epsilon", float(epsilon))
        object.__setattr__(self, "anchor_common_mode_v_per_k", anchor)

    def evaluate(self, temperature_k: object) -> FloatArray:
        PchipTemperatureProperty.evaluate(self, temperature_k)
        common_mode = 0.5 * (
            self.p_source.evaluate(temperature_k)
            + self.n_source.evaluate(temperature_k)
        )
        return np.asarray(
            self.base.evaluate(temperature_k)
            + self.epsilon
            * (common_mode - self.anchor_common_mode_v_per_k),
            dtype=float,
        )

    def derivative(self, temperature_k: object) -> FloatArray:
        PchipTemperatureProperty.derivative(self, temperature_k)
        common_derivative = 0.5 * (
            self.p_source.derivative(temperature_k)
            + self.n_source.derivative(temperature_k)
        )
        return np.asarray(
            self.base.derivative(temperature_k)
            + self.epsilon * common_derivative,
            dtype=float,
        )


def _replace_leg_seebeck(
    leg: TemperatureDependentLeg,
    seebeck: Any,
) -> TemperatureDependentLeg:
    return TemperatureDependentLeg(
        seebeck=seebeck,
        electrical_resistivity=leg.electrical_resistivity,
        thermal_conductivity=leg.thermal_conductivity,
        length_m=leg.length_m,
        area_m2=leg.area_m2,
    )


def build_analytic_perturbed_couple(
    base: TemperatureDependentNumericalCouple,
    *,
    epsilon: float,
    offset_v_per_k: float,
    slope_v_per_k2: float,
    anchor_temperature_k: float,
) -> TemperatureDependentNumericalCouple:
    def shifted(leg: TemperatureDependentLeg) -> TemperatureDependentLeg:
        return _replace_leg_seebeck(
            leg,
            AnalyticCommonModePerturbedProperty(
                leg.seebeck,
                epsilon=epsilon,
                offset_v_per_k=offset_v_per_k,
                slope_v_per_k2=slope_v_per_k2,
                anchor_temperature_k=anchor_temperature_k,
            ),
        )

    return TemperatureDependentNumericalCouple(
        p_leg=shifted(base.p_leg),
        n_leg=shifted(base.n_leg),
        cold_temperature_k=base.cold_temperature_k,
        hot_temperature_k=base.hot_temperature_k,
    )


def build_empirical_perturbed_couple(
    flattened: TemperatureDependentNumericalCouple,
    original: TemperatureDependentNumericalCouple,
    *,
    epsilon: float,
) -> TemperatureDependentNumericalCouple:
    anchor = flattened.cold_temperature_k

    def shifted(leg: TemperatureDependentLeg) -> TemperatureDependentLeg:
        return _replace_leg_seebeck(
            leg,
            EmpiricalCommonModePerturbedProperty(
                leg.seebeck,
                original.p_leg.seebeck,
                original.n_leg.seebeck,
                epsilon=epsilon,
                anchor_temperature_k=anchor,
            ),
        )

    return TemperatureDependentNumericalCouple(
        p_leg=shifted(flattened.p_leg),
        n_leg=shifted(flattened.n_leg),
        cold_temperature_k=flattened.cold_temperature_k,
        hot_temperature_k=flattened.hot_temperature_k,
    )


def _tight_couple_solve(
    couple: TemperatureDependentNumericalCouple,
    current_a: float,
    *,
    output_points: int = 1001,
) -> TemperatureDependentOperatingPoint:
    return solve_temperature_dependent_couple(
        couple,
        current_a,
        initial_mesh_points=51,
        output_points=output_points,
        relative_tolerance=1.0e-9,
        max_nodes=16000,
    )


@dataclass(frozen=True)
class CollectionFunction:
    normalized_coordinate: FloatArray
    value: FloatArray
    initial_slope_per_unit_coordinate: float
    endpoint_error: float
    integration_success: bool
    integration_message: str
    function_evaluations: int


def solve_collection_function(
    leg: TemperatureDependentLeg,
    solution: Any,
    *,
    port: str,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> CollectionFunction:
    """Solve the exact formal adjoint along a nonlinear-solver baseline field.

    In the normalized coordinate ``u=x/L`` the adjoint equation is

        psi_uu + c1 psi_u + c0 psi = 0,
        c1 = I_i tau L/(kappa A),
        c0 = rho_T I_i**2 L**2/(kappa A**2).

    The cold collection function has endpoint values ``(1,0)``.  The hot
    collection function has values ``(0,-1)`` because the reported hot heat
    flux contains ``-K*T'(L)``.
    """

    if port not in {"cold", "hot"}:
        raise ValueError("port must be 'cold' or 'hot'")
    coordinate = np.asarray(solution.coordinate_m, dtype=float)
    normalized = coordinate / leg.length_m
    temperature = np.asarray(solution.temperature_k, dtype=float)
    tau = temperature * leg.seebeck.derivative(temperature)
    drho_dtemperature = leg.electrical_resistivity.derivative(temperature)
    conductivity = leg.thermal_conductivity.evaluate(temperature)
    signed_current = float(solution.signed_current_a)
    c1 = (
        signed_current
        * tau
        * leg.length_m
        / (conductivity * leg.area_m2)
    )
    c0 = (
        drho_dtemperature
        * signed_current**2
        * leg.length_m**2
        / (conductivity * leg.area_m2**2)
    )
    c1_law = PchipInterpolator(normalized, c1, extrapolate=False)
    c0_law = PchipInterpolator(normalized, c0, extrapolate=False)

    def ode(unit_coordinate: float, state: FloatArray) -> FloatArray:
        first = float(c1_law(unit_coordinate))
        zeroth = float(c0_law(unit_coordinate))
        return np.asarray(
            [state[1], -first * state[1] - zeroth * state[0]],
            dtype=float,
        )

    common_options = {
        "method": "DOP853",
        "rtol": rtol,
        "atol": atol,
        "dense_output": True,
        "max_step": 0.01,
    }
    if port == "cold":
        first_initial = (1.0, 0.0)
        second_initial = (0.0, 1.0)
        target_at_hot = 0.0
    else:
        first_initial = (0.0, 0.0)
        second_initial = (0.0, 1.0)
        target_at_hot = -1.0
    first_solution = solve_ivp(
        ode,
        (0.0, 1.0),
        first_initial,
        **common_options,
    )
    second_solution = solve_ivp(
        ode,
        (0.0, 1.0),
        second_initial,
        **common_options,
    )
    if not first_solution.success or not second_solution.success:
        raise RuntimeError(
            "adjoint initial-value integration failed: "
            f"{first_solution.message}; {second_solution.message}"
        )
    denominator = float(second_solution.y[0, -1])
    if abs(denominator) < 1.0e-14:
        raise RuntimeError("adjoint shooting denominator is numerically singular")
    slope = (
        target_at_hot - float(first_solution.y[0, -1])
    ) / denominator
    values = np.asarray(
        first_solution.sol(normalized)[0]
        + slope * second_solution.sol(normalized)[0],
        dtype=float,
    )
    target_at_cold = 1.0 if port == "cold" else 0.0
    endpoint_error = max(
        abs(float(values[0]) - target_at_cold),
        abs(float(values[-1]) - target_at_hot),
    )
    return CollectionFunction(
        normalized_coordinate=normalized,
        value=values,
        initial_slope_per_unit_coordinate=float(
            first_initial[1] + slope * second_initial[1]
        ),
        endpoint_error=float(endpoint_error),
        integration_success=True,
        integration_message=(
            f"{first_solution.message}; {second_solution.message}"
        ),
        function_evaluations=int(first_solution.nfev + second_solution.nfev),
    )


def first_order_port_response(
    couple: TemperatureDependentNumericalCouple,
    point: TemperatureDependentOperatingPoint,
    *,
    mode_value: Callable[[FloatArray], FloatArray],
    gamma_value: Callable[[FloatArray], FloatArray],
    port: str,
) -> dict[str, Any]:
    """Evaluate the adjoint kernel for one fixed-endpoint p/n couple."""

    records: dict[str, Any] = {}
    for name, leg, solution in (
        ("p_leg", couple.p_leg, point.p_leg),
        ("n_leg", couple.n_leg, point.n_leg),
    ):
        collection = solve_collection_function(leg, solution, port=port)
        temperature = np.asarray(solution.temperature_k, dtype=float)
        coordinate = np.asarray(solution.coordinate_m, dtype=float)
        gradient = np.asarray(solution.temperature_gradient_k_per_m, dtype=float)
        signed_current = float(solution.signed_current_a)
        endpoint_temperature = (
            float(temperature[0]) if port == "cold" else float(temperature[-1])
        )
        direct = signed_current * endpoint_temperature * float(
            np.asarray(mode_value(np.asarray([endpoint_temperature])))[0]
        )
        gamma = np.asarray(gamma_value(temperature), dtype=float)
        density_per_m = (
            collection.value * signed_current * gamma * gradient
        )
        # Simpson integration is exact for the cubic constant-property benchmark
        # and materially tightens the independent adjoint/closed-form comparison.
        distributed = float(simpson(density_per_m, x=coordinate))
        density_per_unit_coordinate = density_per_m * leg.length_m
        records[name] = {
            "signed_current_a": signed_current,
            "direct_endpoint_peltier_derivative_w": direct,
            "distributed_kernel_derivative_w": distributed,
            "total_port_derivative_w": direct + distributed,
            "collection_function": collection,
            "temperature_k": temperature,
            "gamma_v_per_k": gamma,
            "density_w_per_unit_coordinate": density_per_unit_coordinate,
        }
    pair_total = sum(
        float(records[name]["total_port_derivative_w"])
        for name in ("p_leg", "n_leg")
    )
    pair_direct = sum(
        float(records[name]["direct_endpoint_peltier_derivative_w"])
        for name in ("p_leg", "n_leg")
    )
    pair_distributed = sum(
        float(records[name]["distributed_kernel_derivative_w"])
        for name in ("p_leg", "n_leg")
    )
    return {
        "port": port,
        "p_leg": records["p_leg"],
        "n_leg": records["n_leg"],
        "pair_direct_endpoint_derivative_w": pair_direct,
        "pair_distributed_kernel_derivative_w": pair_distributed,
        "pair_total_port_derivative_w": pair_total,
    }


def _make_constant_leg(
    *,
    seebeck_v_per_k: float,
    resistivity_ohm_m: float,
    conductivity_w_per_m_k: float,
    length_m: float,
    area_m2: float,
) -> TemperatureDependentLeg:
    return TemperatureDependentLeg(
        seebeck=_constant_property(seebeck_v_per_k),
        electrical_resistivity=_constant_property(resistivity_ohm_m),
        thermal_conductivity=_constant_property(conductivity_w_per_m_k),
        length_m=length_m,
        area_m2=area_m2,
    )


def _analytic_fixture(
    *,
    symmetric: bool = False,
    matched_r_over_k: bool = False,
) -> TemperatureDependentNumericalCouple:
    p_length = 1.0e-3
    p_area = 1.0e-6
    p_rho = 1.0e-5
    p_kappa = 1.5
    if symmetric:
        n_length = p_length
        n_area = p_area
        n_rho = p_rho
        n_kappa = p_kappa
    elif matched_r_over_k:
        n_length = 1.5e-3
        n_area = 1.2e-6
        n_kappa = 1.0
        target = p_rho * p_length**2 / (p_kappa * p_area**2)
        n_rho = target * n_kappa * n_area**2 / n_length**2
    else:
        n_length = 1.0e-3
        n_area = 1.0e-6
        n_rho = 2.0e-5
        n_kappa = 1.0
    return TemperatureDependentNumericalCouple(
        p_leg=_make_constant_leg(
            seebeck_v_per_k=220.0e-6,
            resistivity_ohm_m=p_rho,
            conductivity_w_per_m_k=p_kappa,
            length_m=p_length,
            area_m2=p_area,
        ),
        n_leg=_make_constant_leg(
            seebeck_v_per_k=-180.0e-6,
            resistivity_ohm_m=n_rho,
            conductivity_w_per_m_k=n_kappa,
            length_m=n_length,
            area_m2=n_area,
        ),
        cold_temperature_k=300.0,
        hot_temperature_k=330.0,
    )


def _analytic_linear_mode_derivative(
    couple: TemperatureDependentNumericalCouple,
    current_a: float,
    slope_v_per_k2: float,
) -> dict[str, float]:
    def curvature(leg: TemperatureDependentLeg) -> float:
        rho = float(leg.electrical_resistivity.evaluate([300.0])[0])
        kappa = float(leg.thermal_conductivity.evaluate([300.0])[0])
        return (
            rho
            * current_a**2
            * leg.length_m**2
            / (2.0 * kappa * leg.area_m2**2)
        )

    g_p = curvature(couple.p_leg)
    g_n = curvature(couple.n_leg)
    g_mean = 0.5 * (g_p + g_n)
    delta_g = g_p - g_n
    cold = couple.cold_temperature_k
    delta_temperature = couple.hot_temperature_k - cold
    derivative = (
        current_a
        * slope_v_per_k2
        * delta_g
        * (cold / 6.0 + delta_temperature / 12.0 + g_mean / 30.0)
    )
    alpha = float(
        couple.p_leg.seebeck.evaluate([cold])[0]
        - couple.n_leg.seebeck.evaluate([cold])[0]
    )
    mu = slope_v_per_k2 * cold / alpha
    delta_g_dimensionless = delta_g / cold
    g_mean_dimensionless = g_mean / cold
    theta = delta_temperature / cold
    bracket = 1.0 / 6.0 + theta / 12.0 + g_mean_dimensionless / 30.0
    breaking_number = mu * delta_g_dimensionless * bracket
    return {
        "g_p_k": g_p,
        "g_n_k": g_n,
        "g_mean_k": g_mean,
        "delta_g_k": delta_g,
        "derivative_w": derivative,
        "alpha_v_per_k": alpha,
        "mu_common_slope_to_alpha": mu,
        "delta_G": delta_g_dimensionless,
        "G_mean": g_mean_dimensionless,
        "theta": theta,
        "dimensionless_bracket": bracket,
        "breaking_number_signed": breaking_number,
        "derivative_over_cold_peltier_scale": (
            derivative / (alpha * current_a * cold)
        ),
    }


def analyze_analytic_benchmark() -> dict[str, Any]:
    current = 0.5
    slope = 1.0e-6
    couple = _analytic_fixture()
    baseline = _tight_couple_solve(couple, current)
    mode_value = lambda temperature: slope * (temperature - 300.0)
    gamma_value = lambda temperature: slope * temperature
    kernel = first_order_port_response(
        couple,
        baseline,
        mode_value=mode_value,
        gamma_value=gamma_value,
        port="cold",
    )
    closed_form = _analytic_linear_mode_derivative(couple, current, slope)
    steps = [0.1, 0.03, 0.01, 0.003, 0.001]
    convergence = []
    for step in steps:
        plus = _tight_couple_solve(
            build_analytic_perturbed_couple(
                couple,
                epsilon=step,
                offset_v_per_k=0.0,
                slope_v_per_k2=slope,
                anchor_temperature_k=300.0,
            ),
            current,
            output_points=401,
        )
        minus = _tight_couple_solve(
            build_analytic_perturbed_couple(
                couple,
                epsilon=-step,
                offset_v_per_k=0.0,
                slope_v_per_k2=slope,
                anchor_temperature_k=300.0,
            ),
            current,
            output_points=401,
        )
        numerical = (plus.Qc_w - minus.Qc_w) / (2.0 * step)
        convergence.append(
            {
                "central_step": step,
                "numerical_derivative_w": numerical,
                "relative_error_to_closed_form": abs(
                    numerical - closed_form["derivative_w"]
                )
                / abs(closed_form["derivative_w"]),
            }
        )

    def cancellation_control(kind: str) -> dict[str, float]:
        trial = _analytic_fixture(
            symmetric=kind == "identical",
            matched_r_over_k=kind == "matched_r_over_k",
        )
        base = _tight_couple_solve(trial, current)
        response = first_order_port_response(
            trial,
            base,
            mode_value=mode_value,
            gamma_value=gamma_value,
            port="cold",
        )
        formula = _analytic_linear_mode_derivative(trial, current, slope)
        return {
            "kernel_derivative_w": float(
                response["pair_total_port_derivative_w"]
            ),
            "closed_form_derivative_w": formula["derivative_w"],
            "delta_g_k": formula["delta_g_k"],
            "maximum_collection_function_difference": float(
                np.max(
                    np.abs(
                        response["p_leg"]["collection_function"].value
                        - response["n_leg"]["collection_function"].value
                    )
                )
            ),
        }

    return {
        "fixture": {
            "cold_temperature_k": 300.0,
            "hot_temperature_k": 330.0,
            "current_a": current,
            "common_mode_slope_b_v_per_k2": slope,
            "p": {"rho_ohm_m": 1.0e-5, "kappa_w_per_m_k": 1.5},
            "n": {"rho_ohm_m": 2.0e-5, "kappa_w_per_m_k": 1.0},
            "shared_length_m": 1.0e-3,
            "shared_area_m2": 1.0e-6,
        },
        "closed_form": closed_form,
        "adjoint_kernel_derivative_w": float(
            kernel["pair_total_port_derivative_w"]
        ),
        "adjoint_relative_error_to_closed_form": abs(
            float(kernel["pair_total_port_derivative_w"])
            - closed_form["derivative_w"]
        )
        / abs(closed_form["derivative_w"]),
        "central_difference_convergence": convergence,
        "exact_identical_branch_control": cancellation_control("identical"),
        "nonidentical_but_matched_R_over_K_control": cancellation_control(
            "matched_r_over_k"
        ),
        "interpretation": (
            "For constant rho and kappa, the collection function is exactly 1-u. "
            "The leading symmetry breaker is delta_g=(I^2/2) delta(R/K), so "
            "the response is cubic in current at low Joule heating.  Different "
            "raw geometries and transport coefficients still cancel when R/K is matched."
        ),
    }


def _central_temperature_jacobian(
    couple: TemperatureDependentNumericalCouple,
    current_a: float,
    cold_temperature_k: float,
    hot_temperature_k: float,
    step_k: float,
) -> FloatArray:
    columns = []
    for cold_delta, hot_delta in ((step_k, 0.0), (0.0, step_k)):
        plus_couple = TemperatureDependentNumericalCouple(
            p_leg=couple.p_leg,
            n_leg=couple.n_leg,
            cold_temperature_k=cold_temperature_k + cold_delta,
            hot_temperature_k=hot_temperature_k + hot_delta,
        )
        minus_couple = TemperatureDependentNumericalCouple(
            p_leg=couple.p_leg,
            n_leg=couple.n_leg,
            cold_temperature_k=cold_temperature_k - cold_delta,
            hot_temperature_k=hot_temperature_k - hot_delta,
        )
        plus = _tight_couple_solve(plus_couple, current_a, output_points=401)
        minus = _tight_couple_solve(minus_couple, current_a, output_points=401)
        columns.append(
            np.asarray(
                [plus.Qc_w - minus.Qc_w, plus.Qh_w - minus.Qh_w],
                dtype=float,
            )
            / (2.0 * step_k)
        )
    return np.column_stack(columns)


def analyze_boundary_contact_dressing() -> dict[str, Any]:
    """Validate the Schur-complement dressing by finite thermal contacts."""

    current = 0.5
    slope = 1.0e-6
    reservoir_cold = 300.0
    reservoir_hot = 330.0
    base_couple = _analytic_fixture()
    electrical = SeriesElectricalContacts(
        resistance_ohm=0.01,
        joule_fraction_to_cold_node=0.4,
    )
    thermal = ReservoirThermalContacts(
        cold_resistance_k_per_w=0.8,
        hot_resistance_k_per_w=0.5,
    )
    parasitic = TwoReservoirParasitic(thermal_conductance_w_per_k=2.0e-4)
    options = BoundaryNetworkSolverOptions(
        nonlinear_tolerance=1.0e-11,
        max_function_evaluations=120,
        temperature_residual_tolerance_k=1.0e-7,
        node_energy_residual_tolerance_w=1.0e-9,
        global_energy_residual_fraction_tolerance=1.0e-9,
        bulk_initial_mesh_points=41,
        bulk_output_points=401,
        bulk_relative_tolerance=1.0e-9,
        bulk_max_nodes=16000,
    )

    def network_for(couple: TemperatureDependentNumericalCouple) -> FixedCurrentBoundaryNetwork:
        return FixedCurrentBoundaryNetwork(
            p_leg=couple.p_leg,
            n_leg=couple.n_leg,
            cold_reservoir_temperature_k=reservoir_cold,
            hot_reservoir_temperature_k=reservoir_hot,
            electrical_contacts=electrical,
            thermal_contacts=thermal,
            parasitic=parasitic,
            energy_scale_w=0.1,
            data_role=SYNTHETIC_DATA_ROLE,
        )

    baseline_report = solve_fixed_current_boundary_network(
        network_for(base_couple), current, options=options
    )
    baseline = baseline_report.require_point()
    fixed_endpoint_couple = TemperatureDependentNumericalCouple(
        p_leg=base_couple.p_leg,
        n_leg=base_couple.n_leg,
        cold_temperature_k=baseline.cold_leg_temperature_k,
        hot_temperature_k=baseline.hot_leg_temperature_k,
    )
    fixed_point = _tight_couple_solve(fixed_endpoint_couple, current)
    mode_value = lambda temperature: slope * (temperature - reservoir_cold)
    gamma_value = lambda temperature: slope * temperature
    cold_response = first_order_port_response(
        fixed_endpoint_couple,
        fixed_point,
        mode_value=mode_value,
        gamma_value=gamma_value,
        port="cold",
    )
    hot_response = first_order_port_response(
        fixed_endpoint_couple,
        fixed_point,
        mode_value=mode_value,
        gamma_value=gamma_value,
        port="hot",
    )
    q_epsilon = np.asarray(
        [
            cold_response["pair_total_port_derivative_w"],
            hot_response["pair_total_port_derivative_w"],
        ],
        dtype=float,
    )
    port_jacobian = _central_temperature_jacobian(
        fixed_endpoint_couple,
        current,
        baseline.cold_leg_temperature_k,
        baseline.hot_leg_temperature_k,
        1.0e-3,
    )
    contact_jacobian = np.asarray(
        [
            [
                1.0
                + thermal.cold_resistance_k_per_w * port_jacobian[0, 0],
                thermal.cold_resistance_k_per_w * port_jacobian[0, 1],
            ],
            [
                -thermal.hot_resistance_k_per_w * port_jacobian[1, 0],
                1.0
                - thermal.hot_resistance_k_per_w * port_jacobian[1, 1],
            ],
        ],
        dtype=float,
    )
    forcing = np.asarray(
        [
            thermal.cold_resistance_k_per_w * q_epsilon[0],
            -thermal.hot_resistance_k_per_w * q_epsilon[1],
        ],
        dtype=float,
    )
    endpoint_derivative = -np.linalg.solve(contact_jacobian, forcing)
    dressed_derivative = float(
        q_epsilon[0] + port_jacobian[0] @ endpoint_derivative
    )

    central_step = 0.01
    plus_couple = build_analytic_perturbed_couple(
        base_couple,
        epsilon=central_step,
        offset_v_per_k=0.0,
        slope_v_per_k2=slope,
        anchor_temperature_k=reservoir_cold,
    )
    minus_couple = build_analytic_perturbed_couple(
        base_couple,
        epsilon=-central_step,
        offset_v_per_k=0.0,
        slope_v_per_k2=slope,
        anchor_temperature_k=reservoir_cold,
    )
    plus = solve_fixed_current_boundary_network(
        network_for(plus_couple), current, options=options
    ).require_point()
    minus = solve_fixed_current_boundary_network(
        network_for(minus_couple), current, options=options
    ).require_point()
    numerical = (plus.qc_net_w - minus.qc_net_w) / (2.0 * central_step)
    return {
        "network": {
            "cold_thermal_contact_k_per_w": thermal.cold_resistance_k_per_w,
            "hot_thermal_contact_k_per_w": thermal.hot_resistance_k_per_w,
            "electrical_contact_resistance_ohm": electrical.resistance_ohm,
            "joule_fraction_to_cold": electrical.joule_fraction_to_cold_node,
            "parasitic_conductance_w_per_k": parasitic.thermal_conductance_w_per_k,
            "current_a": current,
        },
        "baseline_leg_endpoint_temperatures_k": [
            baseline.cold_leg_temperature_k,
            baseline.hot_leg_temperature_k,
        ],
        "fixed_endpoint_port_derivatives_w": q_epsilon.tolist(),
        "bulk_port_temperature_jacobian_w_per_k": port_jacobian.tolist(),
        "contact_residual_jacobian": contact_jacobian.tolist(),
        "endpoint_temperature_derivative_k": endpoint_derivative.tolist(),
        "schur_complement_dressed_Qc_derivative_w": dressed_derivative,
        "full_boundary_network_central_derivative_w": numerical,
        "relative_validation_error": abs(numerical - dressed_derivative)
        / max(abs(dressed_derivative), 1.0e-30),
        "thermal_contact_dressing_factor": dressed_derivative / q_epsilon[0],
        "fixed_current_zero_direct_derivative_terms": {
            "series_electrical_contact_joule": True,
            "joule_partition_eta": True,
            "two_reservoir_parasitic_at_fixed_reservoir_temperatures": True,
        },
        "interpretation": (
            "Thermal contacts enter through the implicit endpoint-temperature "
            "Schur complement.  At fixed current, aggregate electrical-contact "
            "Joule heat, its partition, and a reservoir-to-reservoir parasitic have "
            "no direct common-mode derivative; they still change the operating point."
        ),
    }


def analyze_nonisothermal_topology_breaking() -> dict[str, Any]:
    """Show that unequal corresponding cold endpoints expose even a constant mode."""

    current = 0.5
    constant_mode = 50.0e-6
    p_leg = _analytic_fixture(symmetric=True).p_leg
    n_leg = _analytic_fixture(symmetric=True).n_leg
    p_cold, n_cold = 300.0, 305.0
    p_hot, n_hot = 330.0, 335.0

    def shifted_leg(leg: TemperatureDependentLeg, epsilon: float) -> TemperatureDependentLeg:
        return _replace_leg_seebeck(
            leg,
            AnalyticCommonModePerturbedProperty(
                leg.seebeck,
                epsilon=epsilon,
                offset_v_per_k=constant_mode,
                slope_v_per_k2=0.0,
                anchor_temperature_k=300.0,
            ),
        )

    step = 0.01

    def cold_heat(epsilon: float) -> float:
        p = solve_temperature_dependent_leg(
            shifted_leg(p_leg, epsilon),
            +current,
            p_cold,
            p_hot,
            initial_mesh_points=41,
            output_points=401,
            relative_tolerance=1.0e-9,
            max_nodes=12000,
        )
        n = solve_temperature_dependent_leg(
            shifted_leg(n_leg, epsilon),
            -current,
            n_cold,
            n_hot,
            initial_mesh_points=41,
            output_points=401,
            relative_tolerance=1.0e-9,
            max_nodes=12000,
        )
        return float(p.cold_heat_rate_w + n.cold_heat_rate_w)

    numerical = (cold_heat(step) - cold_heat(-step)) / (2.0 * step)
    exact = current * constant_mode * (p_cold - n_cold)
    return {
        "constant_common_mode_v_per_k": constant_mode,
        "current_a": current,
        "p_endpoint_temperatures_k": [p_cold, p_hot],
        "n_endpoint_temperatures_k": [n_cold, n_hot],
        "distributed_gamma_v_per_k": 0.0,
        "exact_direct_cold_peltier_derivative_w": exact,
        "production_solver_central_derivative_w": numerical,
        "absolute_validation_error_w": abs(numerical - exact),
        "scientific_condition": (
            "For a constant shared mode, dQc/dε=I*C*(Tc,p-Tc,n). "
            "The isothermal corresponding-endpoint condition is therefore part "
            "of the physical invariance theorem, not a numerical convenience."
        ),
    }


def _empirical_mode_functions(
    original: TemperatureDependentNumericalCouple,
    anchor_temperature_k: float,
) -> tuple[Callable[[FloatArray], FloatArray], Callable[[FloatArray], FloatArray]]:
    anchor = 0.5 * float(
        original.p_leg.seebeck.evaluate([anchor_temperature_k])[0]
        + original.n_leg.seebeck.evaluate([anchor_temperature_k])[0]
    )

    def mode(temperature: FloatArray) -> FloatArray:
        return np.asarray(
            0.5
            * (
                original.p_leg.seebeck.evaluate(temperature)
                + original.n_leg.seebeck.evaluate(temperature)
            )
            - anchor,
            dtype=float,
        )

    def gamma(temperature: FloatArray) -> FloatArray:
        return np.asarray(
            0.5
            * temperature
            * (
                original.p_leg.seebeck.derivative(temperature)
                + original.n_leg.seebeck.derivative(temperature)
            ),
            dtype=float,
        )

    return mode, gamma


def _effective_r_over_k(leg: TemperatureDependentLeg, solution: Any) -> dict[str, float]:
    coordinate = np.asarray(solution.coordinate_m, dtype=float)
    temperature = np.asarray(solution.temperature_k, dtype=float)
    resistance = float(
        np.trapezoid(
            leg.electrical_resistivity.evaluate(temperature), coordinate
        )
        / leg.area_m2
    )
    thermal_resistance = float(
        np.trapezoid(
            1.0
            / (leg.thermal_conductivity.evaluate(temperature) * leg.area_m2),
            coordinate,
        )
    )
    conductance = 1.0 / thermal_resistance
    return {
        "electrical_resistance_ohm": resistance,
        "thermal_conductance_w_per_k": conductance,
        "R_over_K_k_per_a2": resistance / conductance,
    }


def _first_positive_threshold(
    records: list[dict[str, Any]], target_fraction: float
) -> float | None:
    filtered = []
    for record in records:
        current = float(record["current_a"])
        baseline = float(record["flattened_Qc_after_contact_w"])
        if current <= 0.0 or baseline <= 0.0:
            continue
        fraction = abs(float(record["original_minus_flattened_Qc_w"])) / baseline
        filtered.append((current, fraction))
    for (i0, f0), (i1, f1) in zip(filtered[:-1], filtered[1:]):
        if f0 < target_fraction <= f1:
            weight = (target_fraction - f0) / (f1 - f0)
            return i0 + weight * (i1 - i0)
    return None


def analyze_pbse_kernel() -> dict[str, Any]:
    if not PBSE_COMMON_RESULTS.exists():
        raise FileNotFoundError(
            "the evaluated PbSe common-mode result is required before kernel analysis"
        )
    prior = json.loads(PBSE_COMMON_RESULTS.read_text(encoding="utf-8"))
    inputs = forward.load_inputs()
    target = inputs["target"]
    hot = float(target["hot_side_temperature_k"])
    cold = hot - float(target["delta_t_max_k"])
    original = forward.build_couple(
        inputs,
        forward.SCENARIOS["nominal"],
        cold,
        hot,
    )
    common_at_cold = 0.5 * float(
        original.p_leg.seebeck.evaluate([cold])[0]
        + original.n_leg.seebeck.evaluate([cold])[0]
    )
    flattened = common.build_flattened_common_mode_couple(
        original, common_at_cold
    )
    current = float(prior["fixed_current_mechanism"]["reference_current_a"])
    baseline = _tight_couple_solve(flattened, current, output_points=2001)
    mode, gamma = _empirical_mode_functions(original, cold)
    kernel = first_order_port_response(
        flattened,
        baseline,
        mode_value=mode,
        gamma_value=gamma,
        port="cold",
    )
    central_step = 0.03
    plus = _tight_couple_solve(
        build_empirical_perturbed_couple(
            flattened, original, epsilon=central_step
        ),
        current,
        output_points=701,
    )
    minus = _tight_couple_solve(
        build_empirical_perturbed_couple(
            flattened, original, epsilon=-central_step
        ),
        current,
        output_points=701,
    )
    original_point = _tight_couple_solve(original, current, output_points=1001)
    pair_kernel = float(kernel["pair_total_port_derivative_w"])
    module_kernel = forward.PAIR_COUNT * pair_kernel
    module_central = forward.PAIR_COUNT * (
        plus.Qc_w - minus.Qc_w
    ) / (2.0 * central_step)
    module_full = forward.PAIR_COUNT * (original_point.Qc_w - baseline.Qc_w)
    contact_resistance = float(
        forward.contact_resistance()["seven_pair_series_ohm"]
    )
    baseline_contact_qc = (
        forward.PAIR_COUNT * baseline.Qc_w
        - forward.NOMINAL_ETA_TO_COLD * current**2 * contact_resistance
    )
    p_derivative = forward.PAIR_COUNT * float(
        kernel["p_leg"]["total_port_derivative_w"]
    )
    n_derivative = forward.PAIR_COUNT * float(
        kernel["n_leg"]["total_port_derivative_w"]
    )
    absolute_branch_norm = abs(p_derivative) + abs(n_derivative)
    survival = abs(module_kernel) / absolute_branch_norm

    effective = {
        name: _effective_r_over_k(leg, solution)
        for name, leg, solution in (
            ("p_leg", flattened.p_leg, baseline.p_leg),
            ("n_leg", flattened.n_leg, baseline.n_leg),
        )
    }
    z_p = effective["p_leg"]["R_over_K_k_per_a2"]
    z_n = effective["n_leg"]["R_over_K_k_per_a2"]
    impedance_asymmetry = (z_p - z_n) / (z_p + z_n)
    current_squared_half = 0.5 * current**2
    g_p = current_squared_half * z_p
    g_n = current_squared_half * z_n
    alpha_cold = float(
        original.p_leg.seebeck.evaluate([cold])[0]
        - original.n_leg.seebeck.evaluate([cold])[0]
    )
    temperature_grid = np.linspace(cold, hot, 401)
    mean_dmdt = float(
        np.trapezoid(
            0.5
            * (
                original.p_leg.seebeck.derivative(temperature_grid)
                + original.n_leg.seebeck.derivative(temperature_grid)
            ),
            temperature_grid,
        )
        / (hot - cold)
    )
    constant_scale_prediction = (
        current
        * mean_dmdt
        * (g_p - g_n)
        * (
            cold / 6.0
            + (hot - cold) / 12.0
            + 0.5 * (g_p + g_n) / 30.0
        )
        * forward.PAIR_COUNT
    )

    normalized = np.asarray(
        kernel["p_leg"]["collection_function"].normalized_coordinate,
        dtype=float,
    )
    p_density = forward.PAIR_COUNT * np.asarray(
        kernel["p_leg"]["density_w_per_unit_coordinate"], dtype=float
    )
    n_density = forward.PAIR_COUNT * np.asarray(
        kernel["n_leg"]["density_w_per_unit_coordinate"], dtype=float
    )
    cumulative = np.concatenate(
        ([0.0], cumulative_trapezoid(p_density + n_density, normalized))
    )
    linear_fraction = abs(module_kernel) / baseline_contact_qc
    full_fraction = abs(module_full) / baseline_contact_qc
    gamma_multipliers = {
        "one_percent_local_linear": 0.01 / linear_fraction,
        "ten_percent_local_linear": 0.10 / linear_fraction,
        "one_percent_using_full_unit_profile_ratio": 0.01 / full_fraction,
        "ten_percent_using_full_unit_profile_ratio": 0.10 / full_fraction,
    }
    cubic_current_multipliers = {
        "one_percent_scale_law": (0.01 / linear_fraction) ** (1.0 / 3.0),
        "ten_percent_scale_law": (0.10 / linear_fraction) ** (1.0 / 3.0),
    }
    branch_norm_fraction = absolute_branch_norm / baseline_contact_qc
    asymmetry_multipliers = {
        "one_percent_constant_property_local": 0.01 / linear_fraction,
        "ten_percent_constant_property_local": 0.10 / linear_fraction,
        "required_absolute_epsilon_R_over_K_for_one_percent": abs(
            impedance_asymmetry
        )
        * 0.01
        / linear_fraction,
        "required_absolute_epsilon_R_over_K_for_ten_percent": abs(
            impedance_asymmetry
        )
        * 0.10
        / linear_fraction,
        "definition": "epsilon_RK=(R_p/K_p-R_n/K_n)/(R_p/K_p+R_n/K_n)",
        "scope": (
            "local constant-property scaling at fixed mean R/K and Q scale; "
            "not a finite extrapolation of the variable-property PbSe model"
        ),
    }
    transfer_survival_requirements = {
        "current_absolute_branch_norm_fraction_of_Qc": branch_norm_fraction,
        "current_survival_fraction": survival,
        "required_survival_for_one_percent_at_fixed_branch_norm": (
            0.01 / branch_norm_fraction
        ),
        "required_survival_for_ten_percent_at_fixed_branch_norm": (
            0.10 / branch_norm_fraction
        ),
        "ten_percent_possible_at_fixed_branch_norm": (
            0.10 / branch_norm_fraction <= 1.0
        ),
    }
    current_sweep = prior["signed_current_sweep"]
    one_percent_current = _first_positive_threshold(current_sweep, 0.01)
    ten_percent_current = _first_positive_threshold(current_sweep, 0.10)
    positive_records = [
        row
        for row in current_sweep
        if float(row["current_a"]) > 0.0
        and float(row["flattened_Qc_after_contact_w"]) > 0.0
    ]
    maximum_record = max(
        positive_records,
        key=lambda row: abs(float(row["original_minus_flattened_Qc_w"]))
        / float(row["flattened_Qc_after_contact_w"]),
    )
    max_fraction = abs(
        float(maximum_record["original_minus_flattened_Qc_w"])
    ) / float(maximum_record["flattened_Qc_after_contact_w"])

    def serializable_leg(name: str) -> dict[str, Any]:
        leg_record = kernel[name]
        collection = leg_record["collection_function"]
        return {
            "signed_current_a": leg_record["signed_current_a"],
            "direct_endpoint_peltier_derivative_w_per_pair": leg_record[
                "direct_endpoint_peltier_derivative_w"
            ],
            "distributed_kernel_derivative_w_per_pair": leg_record[
                "distributed_kernel_derivative_w"
            ],
            "module_total_port_derivative_w": forward.PAIR_COUNT
            * leg_record["total_port_derivative_w"],
            "collection_initial_slope_per_unit_coordinate": (
                collection.initial_slope_per_unit_coordinate
            ),
            "collection_endpoint_error": collection.endpoint_error,
            "collection_integration_function_evaluations": (
                collection.function_evaluations
            ),
        }

    return {
        "status": "figure_derived_candidate_scenario_screen_not_device_validation",
        "target": {
            "cold_temperature_k": cold,
            "hot_temperature_k": hot,
            "current_a": current,
            "pair_count": forward.PAIR_COUNT,
        },
        "first_order_kernel": {
            "p_leg": serializable_leg("p_leg"),
            "n_leg": serializable_leg("n_leg"),
            "module_derivative_w": module_kernel,
            "module_central_difference_derivative_w": module_central,
            "relative_adjoint_to_central_error": abs(
                module_kernel - module_central
            )
            / abs(module_kernel),
            "absolute_branch_transfer_norm_w": absolute_branch_norm,
            "cancellation_survival_fraction": survival,
            "baseline_contact_corrected_Qc_w": baseline_contact_qc,
            "response_fraction_of_baseline_Qc": linear_fraction,
        },
        "finite_unit_profile": {
            "original_minus_flattened_Qc_w": module_full,
            "response_fraction_of_baseline_Qc": full_fraction,
            "nonlinear_departure_from_first_order_fraction": (
                module_full / module_kernel - 1.0
            ),
        },
        "effective_transport_coordinates": {
            **effective,
            "signed_R_over_K_asymmetry": impedance_asymmetry,
            "joule_curvature_g_p_k": g_p,
            "joule_curvature_g_n_k": g_n,
            "joule_curvature_difference_k": g_p - g_n,
            "constant_property_scale_prediction_w": constant_scale_prediction,
            "constant_property_prediction_over_exact_kernel": (
                constant_scale_prediction / module_kernel
            ),
            "scope": (
                "R and K are path-effective coordinates evaluated on the flattened "
                "baseline fields; the closed form is an order-of-magnitude reduction, "
                "not an exact replacement for the variable-property kernel."
            ),
        },
        "thresholds": {
            "gamma_profile_multipliers_at_reference_current": gamma_multipliers,
            "R_over_K_asymmetry_scale": asymmetry_multipliers,
            "transfer_cancellation_scale": transfer_survival_requirements,
            "cubic_current_scale_multipliers_at_fixed_Q_scale": (
                cubic_current_multipliers
            ),
            "observed_full_solver_current_for_one_percent_a": one_percent_current,
            "observed_full_solver_current_for_ten_percent_a": ten_percent_current,
            "validated_positive_current_range_a": [0.0, forward.CURRENT_MAX_A],
            "maximum_observed_fraction_in_validated_current_range": max_fraction,
            "current_at_maximum_observed_fraction_a": float(
                maximum_record["current_a"]
            ),
            "ten_percent_conclusion": (
                "not attained within the evaluated 0-3.5 A current range; "
                "values beyond this range are not extrapolated"
            ),
        },
        "source_data_for_figure": {
            "normalized_coordinate": normalized.tolist(),
            "p_collection_function": kernel["p_leg"][
                "collection_function"
            ].value.tolist(),
            "n_collection_function": kernel["n_leg"][
                "collection_function"
            ].value.tolist(),
            "p_temperature_k": kernel["p_leg"]["temperature_k"].tolist(),
            "n_temperature_k": kernel["n_leg"]["temperature_k"].tolist(),
            "p_density_w_per_unit_coordinate": p_density.tolist(),
            "n_density_w_per_unit_coordinate": n_density.tolist(),
            "net_cumulative_derivative_w": cumulative.tolist(),
            "signed_current_sweep": current_sweep,
        },
        "scope": {
            "as_built_device_validation": False,
            "statistical_confidence_interval": False,
            "raw_sample_matched_transport": False,
            "first_order_kernel_mathematically_and_numerically_verified": True,
            "finite_profile_response_is_nonlinear": True,
        },
    }


def _serialize_collection_free(result: dict[str, Any]) -> dict[str, Any]:
    """Return a deepcopy safe for strict JSON serialization."""

    return copy.deepcopy(result)


def analyze_transfer_kernel() -> dict[str, Any]:
    analytic = analyze_analytic_benchmark()
    boundary = analyze_boundary_contact_dressing()
    topology = analyze_nonisothermal_topology_breaking()
    pbse = analyze_pbse_kernel()
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "title": "Common-mode thermopower reaches a cold port through branch-transfer mismatch",
        "central_scientific_result": (
            "A temperature-dependent common Seebeck mode is not controlled by its "
            "integrated Thomson heat alone.  Its exact first-order cold-port response "
            "is the overlap of Gamma(T) with the difference between the p- and n-leg "
            "adjoint heat-collection measures.  Shared isothermal endpoints remove the "
            "direct Peltier term; equal transfer measures are necessary and sufficient "
            "for cancellation for every Gamma.  Near the constant-property symmetric "
            "limit the breaking scales as I^3 Gamma delta(R/K)."
        ),
        "theory": {
            "baseline_equation": (
                "[K_i(T_i) T_i']' + r_i(T_i) I_i^2 - tau_i(T_i) I_i T_i' = 0; "
                "K_i=kappa_i A_i, r_i=rho_i/A_i, I_p=+I, I_n=-I"
            ),
            "linearized_equation": (
                "L_i y_i = I_i Gamma(T_i) T_i'; y_i(0)=y_i(L_i)=0"
            ),
            "formal_linear_operator": (
                "L_i y=[K_i y'+K_i,T T_i' y]'+r_i,T I_i^2 y"
                "-I_i[tau_i y'+tau_i,T T_i' y]"
            ),
            "cold_adjoint": (
                "K_i psi_i''+I_i tau_i psi_i'+r_i,T I_i^2 psi_i=0; "
                "psi_i(0)=1, psi_i(L_i)=0"
            ),
            "branch_port_derivative": (
                "dH_c,i/dε=I_i T_c,i m(T_c,i)+"
                "integral psi_i I_i Gamma(T_i) T_i' dx"
            ),
            "monotone_shared_port_kernel": (
                "dQc/dε=I integral_Tc^Th Gamma(T)[psi_p(x_p(T))-"
                "psi_n(x_n(T))] dT"
            ),
            "general_necessary_and_sufficient_zero_condition": (
                "The direct endpoint term must vanish and the two oriented "
                "pushforward measures mu_i(B)=integral 1_B(T_i) psi_i dT_i "
                "must be identical; the opposite branch currents then give the "
                "total signed measure I(mu_p-mu_n). "
                "For one specified Gamma, zero response requires only orthogonality; "
                "for every Gamma, equality of the measures is necessary and sufficient."
            ),
            "constant_property_closed_form": (
                "For m=b(T-Tc), g_i=rho_i I^2 L_i^2/(2 kappa_i A_i^2): "
                "dQc/dε=I b (g_p-g_n)[Tc/6+DeltaT/12+(g_p+g_n)/60]."
            ),
            "dimensionless_breaking_number": (
                "dQc/(alpha I Tc)=mu DeltaG[1/6+theta/12+Gbar/30], "
                "mu=bTc/alpha, DeltaG=(g_p-g_n)/Tc, "
                "Gbar=(g_p+g_n)/(2Tc), theta=DeltaT/Tc"
            ),
            "engineering_interpretation": {
                "geometry_rho_kappa": (
                    "enter first through R_i/K_i=rho_i L_i^2/(kappa_i A_i^2) "
                    "in the constant-property limit and through the full adjoint "
                    "coefficients for temperature-dependent transport"
                ),
                "thermal_contacts": (
                    "dress the fixed-endpoint kernel through an implicit "
                    "endpoint-temperature Schur complement"
                ),
                "electrical_contacts_fixed_current": (
                    "have no direct first-order term, although they alter the chosen "
                    "operating point and optimized-current problem"
                ),
                "nonisothermal_corresponding_endpoints": (
                    "expose the direct term I[m(Tc,p)Tc,p-m(Tc,n)Tc,n], even for "
                    "a constant common shift"
                ),
                "A_S_equal_magnitude_heuristic": (
                    "does not enter the kernel or the near-symmetry breaking number"
                ),
            },
        },
        "analytic_and_symmetry_validation": analytic,
        "finite_contact_boundary_validation": boundary,
        "nonisothermal_topology_validation": topology,
        "pbse_candidate_application": pbse,
        "interpretation": {
            "scientific_advance": (
                "The result converts a qualitative statement about unequal branch "
                "transfer into a predictive, falsifiable port-sensitivity kernel and "
                "an analytic near-symmetry breaking number."
            ),
            "what_is_now_closed": [
                "exact first-order map from Gamma(T) to fixed-current Qc",
                "necessary-and-sufficient cancellation condition for arbitrary Gamma",
                "explicit rho/kappa/geometry scale law",
                "finite thermal-contact dressing",
                "non-isothermal topology-breaking control",
                "temperature-dependent-solver and independent closed-form verification",
            ],
            "remaining_limit": (
                "The PbSe/Cr magnitude remains figure-derived and lacks raw, "
                "sample-matched uncertainty and device validation.  The kernel "
                "does not substitute for that applied evidence."
            ),
            "summary": (
                "The exact kernel, finite-contact reduction, and independent "
                "two-dimensional calculation define the supported physical scope."
            ),
        },
        "scope": {
            "new_experiment": False,
            "device_validation": False,
            "first_order_theorem": True,
            "constant_property_closed_form": True,
            "finite_amplitude_pbse_effect_requires_full_nonlinear_solver": True,
            "ten_percent_pbse_effect_demonstrated": False,
        },
    }


def make_figure(result: dict[str, Any], output_stem: Path) -> list[Path]:
    """Create a compact four-panel visual argument using Python only.

    Figure design
    -------------
    Core conclusion: the common mode reaches Qc only through unequal branch
    collection, and the PbSe scenario crosses 1% only away from its
    capacity optimum while 10% is absent in the validated current range.
    Layout: quantitative grid emphasizing the PbSe collection kernel.
    Output: 7.15 x 5.25 in, editable SVG plus PDF/PNG.
    Panel a: exact analytic/solver/adjoint validation.
    Panel b: PbSe collection functions and branch mismatch.
    Panel c: local kernel density and cumulative terminal response.
    Panel d: full-solver current threshold plus 1%/10% decision lines.
    Limitations: figure-derived PbSe inputs; deterministic curves have no
    sampling error bars and are explicitly not an as-built validation.
    """

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "svg.hashsalt": "common-mode-transfer-kernel-v1",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.3,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    navy = "#17324D"
    blue = "#4C78A8"
    orange = "#E28E2C"
    red = "#C44E52"
    green = "#4F8C6B"
    grey = "#8A9099"
    pale = "#DDE6ED"

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25))
    fig.subplots_adjust(
        left=0.09,
        right=0.97,
        bottom=0.12,
        top=0.88,
        wspace=0.34,
        hspace=0.45,
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    analytic = result["analytic_and_symmetry_validation"]
    convergence = analytic["central_difference_convergence"]
    step = np.asarray([row["central_step"] for row in convergence])
    error = np.asarray(
        [row["relative_error_to_closed_form"] for row in convergence]
    )
    ax_a.loglog(step, error, "o-", color=blue, linewidth=1.6, markersize=4)
    ax_a.axhline(1.0e-6, color=grey, linestyle="--", linewidth=0.8)
    ax_a.set(
        xlabel=r"central perturbation step $\epsilon$",
        ylabel="relative derivative error",
    )
    ax_a.set_title("Closed form, adjoint, and temperature-dependent solver agree", loc="left")
    ax_a.text(
        0.04,
        0.08,
        r"$dQ_c/d\epsilon\propto I^3\,\Gamma\,\Delta(R/K)$"
        + "\n"
        + "matched R/K control: zero",
        transform=ax_a.transAxes,
        color=navy,
    )

    pbse = result["pbse_candidate_application"]
    source = pbse["source_data_for_figure"]
    u = np.asarray(source["normalized_coordinate"], dtype=float)
    psi_p = np.asarray(source["p_collection_function"], dtype=float)
    psi_n = np.asarray(source["n_collection_function"], dtype=float)
    ax_b.plot(u, psi_p, color=red, linewidth=1.8, label=r"p leg $\psi_p$")
    ax_b.plot(u, psi_n, color=blue, linewidth=1.8, label=r"n leg $\psi_n$")
    ax_b.fill_between(u, psi_p, psi_n, color=pale, alpha=0.8)
    ax_b.set(
        xlabel="normalized cold-to-hot coordinate",
        ylabel="cold-port collection function",
        ylim=(-0.03, 1.03),
    )
    ax_b.set_title("PbSe branch collection is measurably unequal", loc="left")
    ax_b.legend(loc="upper right")
    survival = 100.0 * pbse["first_order_kernel"][
        "cancellation_survival_fraction"
    ]
    ax_b.text(
        0.05,
        0.08,
        f"{survival:.1f}% of |p|+|n| transfer survives cancellation",
        transform=ax_b.transAxes,
        color=navy,
    )

    p_density = 1.0e3 * np.asarray(
        source["p_density_w_per_unit_coordinate"], dtype=float
    )
    n_density = 1.0e3 * np.asarray(
        source["n_density_w_per_unit_coordinate"], dtype=float
    )
    cumulative = 1.0e3 * np.asarray(
        source["net_cumulative_derivative_w"], dtype=float
    )
    ax_c.axhline(0.0, color=grey, linewidth=0.8)
    ax_c.plot(u, p_density, color=red, linewidth=1.2, label="p local")
    ax_c.plot(u, n_density, color=blue, linewidth=1.2, label="n local")
    ax_c.plot(
        u,
        p_density + n_density,
        color=navy,
        linewidth=1.8,
        label="net local",
    )
    ax_c.set(
        xlabel="normalized cold-to-hot coordinate",
        ylabel=r"kernel density (mW per $u$)",
    )
    ax_c.set_title("Large opposing transfers leave a small port residue", loc="left")
    ax_c.legend(loc="upper right", ncol=2)
    twin = ax_c.twinx()
    twin.spines["right"].set_visible(True)
    twin.plot(u, cumulative, color=orange, linewidth=1.4, linestyle="--")
    twin.set_ylabel("cumulative net derivative (mW)", color=orange)
    twin.tick_params(axis="y", colors=orange)

    sweep = source["signed_current_sweep"]
    current = []
    fractions = []
    for row in sweep:
        i = float(row["current_a"])
        base = float(row["flattened_Qc_after_contact_w"])
        if i >= 0.0 and base > 0.0:
            current.append(i)
            fractions.append(
                100.0
                * abs(float(row["original_minus_flattened_Qc_w"]))
                / base
            )
    ax_d.plot(current, fractions, color=navy, linewidth=2.0)
    ax_d.axhline(1.0, color=green, linestyle="--", linewidth=1.0, label="1%")
    ax_d.axhline(10.0, color=red, linestyle="--", linewidth=1.0, label="10%")
    reference_current = pbse["target"]["current_a"]
    reference_fraction = 100.0 * pbse["finite_unit_profile"][
        "response_fraction_of_baseline_Qc"
    ]
    ax_d.scatter(
        [reference_current],
        [reference_fraction],
        color=orange,
        edgecolor="white",
        linewidth=0.6,
        s=30,
        zorder=4,
    )
    one_percent = pbse["thresholds"][
        "observed_full_solver_current_for_one_percent_a"
    ]
    if one_percent is not None:
        ax_d.annotate(
            f"1% at {one_percent:.2f} A",
            (one_percent, 1.0),
            xytext=(-55, 20),
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "color": green, "lw": 0.8},
            color=green,
        )
    ax_d.set(
        xlabel="module current (A)",
        ylabel=r"$|\Delta Q_c|/Q_c^{flat}$ (%)",
        ylim=(0.0, 10.6),
    )
    ax_d.set_title("1% is off-optimum; 10% is not observed", loc="left")
    ax_d.legend(loc="upper left")
    ax_d.text(
        0.04,
        0.42,
        f"capacity optimum: {reference_fraction:.2f}%",
        transform=ax_d.transAxes,
        color=orange,
    )

    for label, axis in zip("abcd", axes.ravel()):
        axis.text(
            -0.14,
            1.08,
            label,
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9.0,
            va="top",
        )
        axis.tick_params(direction="out", length=3.0, width=0.7)

    fig.suptitle(
        "A common Seebeck slope becomes device-visible only through branch-transfer mismatch",
        x=0.09,
        y=0.965,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
        color=navy,
    )
    fig.text(
        0.09,
        0.025,
        "PbSe/Cr curves are figure-derived candidates. Lines are deterministic model "
        "responses, not confidence intervals or as-built device validation.",
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#555B63",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_stem.with_suffix(".svg"),
        output_stem.with_suffix(".pdf"),
        output_stem.with_suffix(".png"),
    ]
    fig.savefig(
        outputs[0],
        bbox_inches="tight",
        facecolor="white",
        metadata={"Date": None, "Creator": None},
    )
    fig.savefig(
        outputs[1],
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "matplotlib; common-mode transfer-kernel analysis",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        outputs[2],
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "matplotlib; common-mode transfer-kernel analysis"},
    )
    plt.close(fig)
    return outputs


def serialize_results(
    result: dict[str, Any],
    figure_paths: list[Path],
) -> dict[str, Any]:
    serialized = _serialize_collection_free(result)
    serialized["input_bindings"] = {
        "validated_pbse_common_mode_result": {
            "locator": _output_locator(PBSE_COMMON_RESULTS),
            "sha256": _file_sha256(PBSE_COMMON_RESULTS),
            "data_role": "figure_derived_candidate_scenario_screen",
        },
        "production_temperature_solver": {
            "locator": "scripts/tec_1d_solver/temperature_dependent.py",
            "sha256": _file_sha256(
                ROOT / "scripts/tec_1d_solver/temperature_dependent.py"
            ),
        },
        "production_boundary_network": {
            "locator": "scripts/tec_1d_solver/boundary_network.py",
            "sha256": _file_sha256(
                ROOT / "scripts/tec_1d_solver/boundary_network.py"
            ),
        },
    }
    serialized["figure_metadata"] = {
        "core_conclusion": (
            "A variable common Seebeck mode reaches Qc only through unequal p/n "
            "heat-collection functions; the evaluated PbSe scenario exceeds 1% only "
            "away from its capacity optimum and never reaches 10% over 0-3.5 A."
        ),
        "evidence_chain": [
            "independent analytic, adjoint, and nonlinear-solver convergence",
            "PbSe p/n collection functions",
            "leg-resolved kernel density and cumulative residual",
            "full nonlinear signed-current threshold screen",
        ],
        "layout": "quantitative grid with an emphasized collection-kernel panel",
        "backend": "Python matplotlib only",
        "export": "7.15 x 5.25 in editable SVG, PDF, and 400-dpi PNG",
        "statistics": (
            "deterministic theory and numerical verification; no sampling error bars, "
            "p-values, or confidence intervals"
        ),
        "limitation": (
            "PbSe transport remains figure-derived and sample identity is unresolved"
        ),
    }
    serialized["outputs"] = {
        "analysis_script": {
            "locator": _output_locator(Path(__file__).resolve()),
            "sha256": _file_sha256(Path(__file__).resolve()),
        },
        "figures": [
            {"locator": _output_locator(path), "sha256": _file_sha256(path)}
            for path in figure_paths
        ],
    }
    serialized["software_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    return serialized


def run_analysis(json_output: Path, figure_stem: Path) -> dict[str, Any]:
    result = analyze_transfer_kernel()
    figure_paths = make_figure(result, figure_stem)
    serialized = serialize_results(result, figure_paths)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(serialized, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return serialized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE_STEM)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    result = run_analysis(arguments.json_output, arguments.figure_stem)
    analytic = result["analytic_and_symmetry_validation"]
    pbse = result["pbse_candidate_application"]
    print(
        "Common-mode transfer-kernel analysis complete: "
        f"analytic relative error={analytic['adjoint_relative_error_to_closed_form']:.3e}; "
        f"PbSe first-order derivative="
        f"{1.0e3*pbse['first_order_kernel']['module_derivative_w']:.6f} mW; "
        f"finite unit-profile response="
        f"{1.0e3*pbse['finite_unit_profile']['original_minus_flattened_Qc_w']:.6f} mW"
    )


if __name__ == "__main__":
    main()
