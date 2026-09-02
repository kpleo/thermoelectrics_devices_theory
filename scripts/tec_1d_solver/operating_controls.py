"""Explicit operating-control wrappers for the temperature-dependent bulk solver.

The conservative leg solver is current controlled and uses fixed endpoint
temperatures.  This module adds scalar outer solves for three additional ideal-bulk
operating modes without changing the underlying thermoelectric equations:

* fixed terminal voltage at fixed endpoint temperatures;
* fixed positive electrical input power at fixed endpoint temperatures; and
* fixed cold-side heat load (including the adiabatic ``Qc=0`` limit) at a fixed hot
  endpoint temperature and fixed current.

Every nonlinear control solve requires an explicit finite bracket.  This is
intentional: voltage, power, or heat-load equations can have multiple branches, and
silently searching for a root would make the selected operating point ambiguous.
Contacts and parasitic heat paths belong to the separate boundary-network layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from scipy.optimize import root_scalar

from .temperature_dependent import (
    TemperatureDependentNumericalCouple,
    TemperatureDependentOperatingPoint,
    solve_temperature_dependent_couple,
)


class ControlSolveError(RuntimeError):
    """Raised when an explicitly bracketed operating-control solve fails."""


@dataclass(frozen=True)
class ControlledOperatingPoint:
    """One converged ideal-bulk operating point and its scalar-control diagnostic."""

    control_mode: str
    target_value_si: float
    achieved_value_si: float
    control_residual_si: float
    current_a: float
    cold_temperature_k: float
    hot_temperature_k: float
    bracket: tuple[float, float]
    scalar_iterations: int
    bulk_solver_evaluations: int
    operating_point: TemperatureDependentOperatingPoint

    @property
    def relative_control_residual(self) -> float:
        scale = max(abs(self.target_value_si), abs(self.achieved_value_si), 1.0e-15)
        return abs(self.control_residual_si) / scale


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive(name: str, value: object) -> float:
    converted = _finite(name, value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return converted


def _nonnegative(name: str, value: object) -> float:
    converted = _finite(name, value)
    if converted < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return converted


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _bracket(name: str, values: object) -> tuple[float, float]:
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise TypeError(f"{name} must contain exactly two real endpoints")
    lower = _finite(f"{name}[0]", values[0])
    upper = _finite(f"{name}[1]", values[1])
    if not lower < upper:
        raise ValueError(f"{name} endpoints must be strictly increasing")
    return lower, upper


def _solver_options(
    *,
    initial_mesh_points: int,
    output_points: int,
    relative_tolerance: float,
    max_nodes: int,
) -> dict[str, int | float]:
    n_initial = _integer(
        "initial_mesh_points", initial_mesh_points, minimum=3
    )
    n_output = _integer("output_points", output_points, minimum=3)
    n_max = _integer("max_nodes", max_nodes, minimum=3)
    if n_max < n_initial:
        raise ValueError("max_nodes must be >= initial_mesh_points")
    return {
        "initial_mesh_points": n_initial,
        "output_points": n_output,
        "relative_tolerance": _positive("relative_tolerance", relative_tolerance),
        "max_nodes": n_max,
    }


def _root_options(
    *, xtol: float, rtol: float, max_iterations: int
) -> tuple[float, float, int]:
    absolute = _positive("root_absolute_tolerance", xtol)
    relative = _positive("root_relative_tolerance", rtol)
    minimum_rtol = 4.0 * float.fromhex("0x1.0000000000000p-52")
    if relative < minimum_rtol:
        raise ValueError(
            "root_relative_tolerance is below the scipy bracketing minimum"
        )
    iterations = _integer("root_max_iterations", max_iterations, minimum=1)
    return absolute, relative, iterations


def _solve_bracketed(
    residual: Callable[[float], float],
    bracket: tuple[float, float],
    *,
    xtol: float,
    rtol: float,
    max_iterations: int,
    label: str,
) -> tuple[float, int]:
    def checked_residual(value: float) -> float:
        return _finite(f"{label} residual", residual(value))

    lower_residual = checked_residual(bracket[0])
    upper_residual = checked_residual(bracket[1])
    if lower_residual == 0.0:
        return bracket[0], 0
    if upper_residual == 0.0:
        return bracket[1], 0
    if math.copysign(1.0, lower_residual) == math.copysign(1.0, upper_residual):
        raise ControlSolveError(
            f"{label} is not bracketed: endpoint residuals are "
            f"{lower_residual:.12g} and {upper_residual:.12g}"
        )
    result = root_scalar(
        checked_residual,
        bracket=bracket,
        method="brentq",
        xtol=xtol,
        rtol=rtol,
        maxiter=max_iterations,
    )
    if not result.converged or result.root is None:
        raise ControlSolveError(
            f"{label} solve did not converge after {result.iterations} iterations"
        )
    return float(result.root), int(result.iterations)


def _current_control_solve(
    couple: TemperatureDependentNumericalCouple,
    *,
    target: float,
    current_bracket_a: tuple[float, float] | list[float],
    observable: Callable[[TemperatureDependentOperatingPoint], float],
    control_mode: str,
    initial_mesh_points: int,
    output_points: int,
    relative_tolerance: float,
    max_nodes: int,
    root_absolute_tolerance: float,
    root_relative_tolerance: float,
    root_max_iterations: int,
) -> ControlledOperatingPoint:
    if not isinstance(couple, TemperatureDependentNumericalCouple):
        raise TypeError("couple must be a TemperatureDependentNumericalCouple")
    bracket = _bracket("current_bracket_a", current_bracket_a)
    solver_options = _solver_options(
        initial_mesh_points=initial_mesh_points,
        output_points=output_points,
        relative_tolerance=relative_tolerance,
        max_nodes=max_nodes,
    )
    xtol, rtol, maxiter = _root_options(
        xtol=root_absolute_tolerance,
        rtol=root_relative_tolerance,
        max_iterations=root_max_iterations,
    )
    cache: dict[float, TemperatureDependentOperatingPoint] = {}

    def evaluate(current: float) -> TemperatureDependentOperatingPoint:
        key = float(current)
        if key not in cache:
            cache[key] = solve_temperature_dependent_couple(
                couple, key, **solver_options
            )
        return cache[key]

    def residual(current: float) -> float:
        return float(observable(evaluate(current))) - target

    current, iterations = _solve_bracketed(
        residual,
        bracket,
        xtol=xtol,
        rtol=rtol,
        max_iterations=maxiter,
        label=control_mode,
    )
    point = evaluate(current)
    achieved = float(observable(point))
    return ControlledOperatingPoint(
        control_mode=control_mode,
        target_value_si=target,
        achieved_value_si=achieved,
        control_residual_si=achieved - target,
        current_a=current,
        cold_temperature_k=couple.cold_temperature_k,
        hot_temperature_k=couple.hot_temperature_k,
        bracket=bracket,
        scalar_iterations=iterations,
        bulk_solver_evaluations=len(cache),
        operating_point=point,
    )


def solve_bulk_at_voltage(
    couple: TemperatureDependentNumericalCouple,
    target_voltage_v: float,
    *,
    current_bracket_a: tuple[float, float] | list[float],
    initial_mesh_points: int = 41,
    output_points: int = 401,
    relative_tolerance: float = 1.0e-7,
    max_nodes: int = 10000,
    root_absolute_tolerance: float = 1.0e-10,
    root_relative_tolerance: float = 1.0e-10,
    root_max_iterations: int = 100,
) -> ControlledOperatingPoint:
    """Solve the ideal bulk couple at a fixed signed terminal voltage."""

    target = _finite("target_voltage_v", target_voltage_v)
    return _current_control_solve(
        couple,
        target=target,
        current_bracket_a=current_bracket_a,
        observable=lambda point: point.V_v,
        control_mode="fixed_voltage",
        initial_mesh_points=initial_mesh_points,
        output_points=output_points,
        relative_tolerance=relative_tolerance,
        max_nodes=max_nodes,
        root_absolute_tolerance=root_absolute_tolerance,
        root_relative_tolerance=root_relative_tolerance,
        root_max_iterations=root_max_iterations,
    )


def solve_bulk_at_input_power(
    couple: TemperatureDependentNumericalCouple,
    target_input_power_w: float,
    *,
    current_bracket_a: tuple[float, float] | list[float],
    initial_mesh_points: int = 41,
    output_points: int = 401,
    relative_tolerance: float = 1.0e-7,
    max_nodes: int = 10000,
    root_absolute_tolerance: float = 1.0e-10,
    root_relative_tolerance: float = 1.0e-10,
    root_max_iterations: int = 100,
) -> ControlledOperatingPoint:
    """Solve at fixed positive electrical input power on an explicit branch."""

    target = _positive("target_input_power_w", target_input_power_w)
    return _current_control_solve(
        couple,
        target=target,
        current_bracket_a=current_bracket_a,
        observable=lambda point: point.Pin_w,
        control_mode="fixed_input_power",
        initial_mesh_points=initial_mesh_points,
        output_points=output_points,
        relative_tolerance=relative_tolerance,
        max_nodes=max_nodes,
        root_absolute_tolerance=root_absolute_tolerance,
        root_relative_tolerance=root_relative_tolerance,
        root_max_iterations=root_max_iterations,
    )


def solve_bulk_at_cold_heat_load(
    couple: TemperatureDependentNumericalCouple,
    current_a: float,
    target_cooling_w: float,
    *,
    cold_temperature_bracket_k: tuple[float, float] | list[float],
    initial_mesh_points: int = 41,
    output_points: int = 401,
    relative_tolerance: float = 1.0e-7,
    max_nodes: int = 10000,
    root_absolute_tolerance: float = 1.0e-9,
    root_relative_tolerance: float = 1.0e-10,
    root_max_iterations: int = 100,
) -> ControlledOperatingPoint:
    """Solve cold endpoint temperature for a fixed nonnegative cold heat load.

    ``target_cooling_w=0`` is the ideal adiabatic-cold-end condition used to find a
    zero-load temperature span.  The hot endpoint remains fixed at the input couple's
    hot temperature; the input couple's cold temperature is only a template value.
    """

    if not isinstance(couple, TemperatureDependentNumericalCouple):
        raise TypeError("couple must be a TemperatureDependentNumericalCouple")
    current = _finite("current_a", current_a)
    target = _nonnegative("target_cooling_w", target_cooling_w)
    bracket = _bracket(
        "cold_temperature_bracket_k", cold_temperature_bracket_k
    )
    if bracket[0] <= 0.0:
        raise ValueError("cold_temperature_bracket_k must be above 0 K")
    if bracket[1] > couple.hot_temperature_k:
        raise ValueError(
            "cold_temperature_bracket_k must not exceed the fixed hot temperature"
        )
    common_minimum_temperature = max(
        couple.p_leg.minimum_valid_temperature_k,
        couple.n_leg.minimum_valid_temperature_k,
    )
    common_maximum_temperature = min(
        couple.p_leg.maximum_valid_temperature_k,
        couple.n_leg.maximum_valid_temperature_k,
        couple.hot_temperature_k,
    )
    if (
        bracket[0] < common_minimum_temperature
        or bracket[1] > common_maximum_temperature
    ):
        raise ValueError(
            "cold_temperature_bracket_k must lie inside the common leg-property "
            "domain and at or below the fixed hot temperature"
        )
    solver_options = _solver_options(
        initial_mesh_points=initial_mesh_points,
        output_points=output_points,
        relative_tolerance=relative_tolerance,
        max_nodes=max_nodes,
    )
    xtol, rtol, maxiter = _root_options(
        xtol=root_absolute_tolerance,
        rtol=root_relative_tolerance,
        max_iterations=root_max_iterations,
    )
    cache: dict[float, TemperatureDependentOperatingPoint] = {}

    def evaluate(cold_temperature: float) -> TemperatureDependentOperatingPoint:
        key = float(cold_temperature)
        if key not in cache:
            trial = TemperatureDependentNumericalCouple(
                p_leg=couple.p_leg,
                n_leg=couple.n_leg,
                cold_temperature_k=key,
                hot_temperature_k=couple.hot_temperature_k,
            )
            cache[key] = solve_temperature_dependent_couple(
                trial, current, **solver_options
            )
        return cache[key]

    def residual(cold_temperature: float) -> float:
        return evaluate(cold_temperature).Qc_w - target

    cold_temperature, iterations = _solve_bracketed(
        residual,
        bracket,
        xtol=xtol,
        rtol=rtol,
        max_iterations=maxiter,
        label="fixed_cold_heat_load",
    )
    point = evaluate(cold_temperature)
    return ControlledOperatingPoint(
        control_mode=(
            "adiabatic_cold_end" if target == 0.0 else "fixed_cold_heat_load"
        ),
        target_value_si=target,
        achieved_value_si=point.Qc_w,
        control_residual_si=point.Qc_w - target,
        current_a=current,
        cold_temperature_k=cold_temperature,
        hot_temperature_k=couple.hot_temperature_k,
        bracket=bracket,
        scalar_iterations=iterations,
        bulk_solver_evaluations=len(cache),
        operating_point=point,
    )
