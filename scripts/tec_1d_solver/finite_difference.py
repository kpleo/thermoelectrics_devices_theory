"""Finite-difference constant-property p/n couple for KT2 verification.

This is the first numerical 1D baseline.  It solves each leg's local temperature
boundary-value problem on an independent mesh while retaining opposite local current
directions.  It intentionally stops at constant properties; the later nonlinear
temperature-dependent solver must be verified against this implementation and the
separate closed-form reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from .constant_properties import ConstantPropertyCouple, OperatingPoint


def _positive_finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ConstantPropertyLeg:
    """One uniform thermoelectric leg in SI units."""

    seebeck_v_per_k: float
    electrical_resistivity_ohm_m: float
    thermal_conductivity_w_per_m_k: float
    length_m: float
    area_m2: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "seebeck_v_per_k", _finite("seebeck_v_per_k", self.seebeck_v_per_k)
        )
        for name in (
            "electrical_resistivity_ohm_m",
            "thermal_conductivity_w_per_m_k",
            "length_m",
            "area_m2",
        ):
            object.__setattr__(self, name, _positive_finite(name, getattr(self, name)))

    @property
    def electrical_resistance_ohm(self) -> float:
        return self.electrical_resistivity_ohm_m * self.length_m / self.area_m2

    @property
    def thermal_conductance_w_per_k(self) -> float:
        return self.thermal_conductivity_w_per_m_k * self.area_m2 / self.length_m


@dataclass(frozen=True)
class ConstantPropertyNumericalCouple:
    """p/n legs sharing common isothermal cold and hot planes."""

    p_leg: ConstantPropertyLeg
    n_leg: ConstantPropertyLeg
    cold_temperature_k: float
    hot_temperature_k: float

    def __post_init__(self) -> None:
        if not isinstance(self.p_leg, ConstantPropertyLeg) or not isinstance(
            self.n_leg, ConstantPropertyLeg
        ):
            raise TypeError("p_leg and n_leg must be ConstantPropertyLeg objects")
        object.__setattr__(
            self,
            "cold_temperature_k",
            _positive_finite("cold_temperature_k", self.cold_temperature_k),
        )
        object.__setattr__(
            self,
            "hot_temperature_k",
            _positive_finite("hot_temperature_k", self.hot_temperature_k),
        )
        if self.hot_temperature_k < self.cold_temperature_k:
            raise ValueError("hot_temperature_k must be >= cold_temperature_k")

    def analytic_reference(self) -> ConstantPropertyCouple:
        return ConstantPropertyCouple(
            seebeck_p_v_per_k=self.p_leg.seebeck_v_per_k,
            seebeck_n_v_per_k=self.n_leg.seebeck_v_per_k,
            electrical_resistance_ohm=(
                self.p_leg.electrical_resistance_ohm
                + self.n_leg.electrical_resistance_ohm
            ),
            thermal_conductance_w_per_k=(
                self.p_leg.thermal_conductance_w_per_k
                + self.n_leg.thermal_conductance_w_per_k
            ),
            cold_temperature_k=self.cold_temperature_k,
            hot_temperature_k=self.hot_temperature_k,
        )


@dataclass(frozen=True)
class LegFieldSolution:
    """Mesh and endpoint flux results for one leg."""

    coordinate_m: NDArray[np.float64]
    temperature_k: NDArray[np.float64]
    signed_current_a: float
    current_density_a_per_m2: float
    cold_heat_rate_w: float
    hot_heat_rate_w: float
    hot_minus_cold_potential_v: float
    local_energy_residual_w: float


@dataclass(frozen=True)
class NumericalOperatingPoint:
    """Numerical terminal results and the two resolved leg fields."""

    current_a: float
    Qc_w: float
    Qh_w: float
    V_v: float
    Pin_w: float
    COP: float | None
    energy_residual_w: float
    p_leg: LegFieldSolution
    n_leg: LegFieldSolution

    @property
    def relative_energy_residual(self) -> float:
        scale = max(abs(self.Qc_w), abs(self.Qh_w), abs(self.Pin_w))
        return 0.0 if scale == 0.0 else abs(self.energy_residual_w) / scale

    def terminal_point(self) -> OperatingPoint:
        return OperatingPoint(
            current_a=self.current_a,
            Qc_w=self.Qc_w,
            Qh_w=self.Qh_w,
            V_v=self.V_v,
            Pin_w=self.Pin_w,
            COP=self.COP,
            energy_residual_w=self.energy_residual_w,
        )


def _solve_leg(
    leg: ConstantPropertyLeg,
    signed_current_a: float,
    cold_temperature_k: float,
    hot_temperature_k: float,
    n_cells: int,
) -> LegFieldSolution:
    if isinstance(n_cells, bool) or not isinstance(n_cells, int):
        raise TypeError("n_cells must be an integer")
    if n_cells < 2:
        raise ValueError("n_cells must be >= 2 for second-order endpoint fluxes")
    signed_current = _finite("signed_current_a", signed_current_a)
    coordinate = np.linspace(0.0, leg.length_m, n_cells + 1, dtype=float)
    dx = leg.length_m / n_cells
    current_density = signed_current / leg.area_m2

    temperature = np.empty(n_cells + 1, dtype=float)
    temperature[0] = cold_temperature_k
    temperature[-1] = hot_temperature_k
    interior_count = n_cells - 1
    matrix = np.diag(np.full(interior_count, -2.0))
    if interior_count > 1:
        matrix += np.diag(np.ones(interior_count - 1), 1)
        matrix += np.diag(np.ones(interior_count - 1), -1)
    source = (
        -leg.electrical_resistivity_ohm_m
        * current_density
        * current_density
        / leg.thermal_conductivity_w_per_m_k
    )
    rhs = np.full(interior_count, source * dx * dx)
    rhs[0] -= cold_temperature_k
    rhs[-1] -= hot_temperature_k
    temperature[1:-1] = np.linalg.solve(matrix, rhs)

    derivative_cold = (-3.0 * temperature[0] + 4.0 * temperature[1] - temperature[2]) / (2.0 * dx)
    derivative_hot = (3.0 * temperature[-1] - 4.0 * temperature[-2] + temperature[-3]) / (2.0 * dx)
    heat_flux_cold = (
        leg.seebeck_v_per_k * cold_temperature_k * current_density
        - leg.thermal_conductivity_w_per_m_k * derivative_cold
    )
    heat_flux_hot = (
        leg.seebeck_v_per_k * hot_temperature_k * current_density
        - leg.thermal_conductivity_w_per_m_k * derivative_hot
    )
    cold_heat = leg.area_m2 * heat_flux_cold
    hot_heat = leg.area_m2 * heat_flux_hot

    temperature_gradient = np.gradient(temperature, coordinate, edge_order=2)
    potential_gradient = (
        -leg.electrical_resistivity_ohm_m * current_density
        - leg.seebeck_v_per_k * temperature_gradient
    )
    potential_difference = float(np.trapezoid(potential_gradient, coordinate))
    local_electrical_power = signed_current * (-potential_difference)
    local_reservoir_difference = hot_heat - cold_heat
    local_energy_residual = local_reservoir_difference - local_electrical_power

    return LegFieldSolution(
        coordinate_m=coordinate,
        temperature_k=temperature,
        signed_current_a=signed_current,
        current_density_a_per_m2=current_density,
        cold_heat_rate_w=cold_heat,
        hot_heat_rate_w=hot_heat,
        hot_minus_cold_potential_v=potential_difference,
        local_energy_residual_w=local_energy_residual,
    )

def solve_constant_couple(
    couple: ConstantPropertyNumericalCouple,
    current_a: float,
    *,
    n_cells: int = 40,
) -> NumericalOperatingPoint:
    """Solve both legs and assemble terminal quantities.

    Both meshes point cold-to-hot, hence local conventional currents are ``+I`` in
    the p leg and ``-I`` in the n leg.
    """

    if not isinstance(couple, ConstantPropertyNumericalCouple):
        raise TypeError("couple must be a ConstantPropertyNumericalCouple")
    current = _finite("current_a", current_a)
    p_solution = _solve_leg(
        couple.p_leg,
        +current,
        couple.cold_temperature_k,
        couple.hot_temperature_k,
        n_cells,
    )
    n_solution = _solve_leg(
        couple.n_leg,
        -current,
        couple.cold_temperature_k,
        couple.hot_temperature_k,
        n_cells,
    )
    q_cold = p_solution.cold_heat_rate_w + n_solution.cold_heat_rate_w
    q_hot = p_solution.hot_heat_rate_w + n_solution.hot_heat_rate_w
    voltage = (
        n_solution.hot_minus_cold_potential_v
        - p_solution.hot_minus_cold_potential_v
    )
    input_power = current * voltage
    cop = q_cold / input_power if q_cold > 0.0 and input_power > 0.0 else None
    residual = q_hot - q_cold - input_power
    return NumericalOperatingPoint(
        current_a=current,
        Qc_w=q_cold,
        Qh_w=q_hot,
        V_v=voltage,
        Pin_w=input_power,
        COP=cop,
        energy_residual_w=residual,
        p_leg=p_solution,
        n_leg=n_solution,
    )
