#!/usr/bin/env python3
"""Independent exact benchmarks for a temperature-dependent 1D TE solver.

This module deliberately does not import the main nonlinear solver. It provides
two closed-form references with the same cold-to-hot coordinate and signed-current
convention:

* a nonzero-current manufactured solution with a linear temperature field,
  temperature-dependent Seebeck coefficient and resistivity, and nonzero
  Thomson coefficient; and
* a zero-current conduction solution with linearly varying thermal
  conductivity and Seebeck coefficient.

The references are intended for solver verification, not material modelling.
In particular, the manufactured resistivity is chosen to make the prescribed
temperature field exact at one specified current density.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]

EXPANDED_BALANCE_ABS_TOLERANCE_W_PER_M3 = 1.0e-6
ENERGY_ABS_TOLERANCE_W = 1.0e-14
ENDPOINT_TEMPERATURE_ABS_TOLERANCE_K = 1.0e-12


def _array(value: float | Array) -> Array:
    return np.asarray(value, dtype=float)


def _scalar_or_array(original: float | Array, value: Array) -> float | Array:
    return float(value) if np.ndim(original) == 0 else value


@dataclass(frozen=True)
class ManufacturedLinearTemperatureCase:
    """One-leg manufactured solution with a nonzero Thomson term.

    The exact fields are

    ``T(x) = Tc + g*x``, ``S(T) = S0 + b*(T-Tc)``, and
    ``rho(T) = b*T*g/J``.

    With constant ``kappa`` these definitions enforce
    ``rho*J**2 = (T*dS/dT)*J*T'``.  Therefore the Joule and Thomson
    terms cancel in the expanded heat equation while both remain nonzero.
    """

    cold_temperature_k: float = 280.0
    hot_temperature_k: float = 340.0
    length_m: float = 1.2e-3
    area_m2: float = 0.8e-6
    signed_current_a: float = 0.48
    seebeck_at_cold_v_per_k: float = 150.0e-6
    dseebeck_dtemperature_v_per_k2: float = 0.25e-6
    thermal_conductivity_w_per_m_k: float = 1.6

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("all manufactured-case parameters must be finite")
        if self.hot_temperature_k <= self.cold_temperature_k:
            raise ValueError("hot_temperature_k must exceed cold_temperature_k")
        if self.length_m <= 0.0 or self.area_m2 <= 0.0:
            raise ValueError("length_m and area_m2 must be positive")
        if self.signed_current_a == 0.0:
            raise ValueError("signed_current_a must be nonzero")
        if self.thermal_conductivity_w_per_m_k <= 0.0:
            raise ValueError("thermal_conductivity_w_per_m_k must be positive")
        endpoint_rho = self.resistivity_ohm_m(
            np.array([self.cold_temperature_k, self.hot_temperature_k])
        )
        if np.any(np.asarray(endpoint_rho) <= 0.0):
            raise ValueError("manufactured resistivity must stay positive")

    @property
    def temperature_gradient_k_per_m(self) -> float:
        return (
            self.hot_temperature_k - self.cold_temperature_k
        ) / self.length_m

    @property
    def current_density_a_per_m2(self) -> float:
        return self.signed_current_a / self.area_m2

    def temperature_k(self, coordinate_m: float | Array) -> float | Array:
        x = _array(coordinate_m)
        value = self.cold_temperature_k + self.temperature_gradient_k_per_m * x
        return _scalar_or_array(coordinate_m, value)

    def seebeck_v_per_k(self, temperature_k: float | Array) -> float | Array:
        temperature = _array(temperature_k)
        value = self.seebeck_at_cold_v_per_k + (
            self.dseebeck_dtemperature_v_per_k2
            * (temperature - self.cold_temperature_k)
        )
        return _scalar_or_array(temperature_k, value)

    def resistivity_ohm_m(self, temperature_k: float | Array) -> float | Array:
        temperature = _array(temperature_k)
        value = (
            self.dseebeck_dtemperature_v_per_k2
            * temperature
            * self.temperature_gradient_k_per_m
            / self.current_density_a_per_m2
        )
        return _scalar_or_array(temperature_k, value)

    def thomson_coefficient_v_per_k(
        self, temperature_k: float | Array
    ) -> float | Array:
        temperature = _array(temperature_k)
        value = temperature * self.dseebeck_dtemperature_v_per_k2
        return _scalar_or_array(temperature_k, value)

    def heat_flux_w_per_m2(self, coordinate_m: float | Array) -> float | Array:
        temperature = _array(self.temperature_k(coordinate_m))
        seebeck = _array(self.seebeck_v_per_k(temperature))
        value = (
            seebeck * temperature * self.current_density_a_per_m2
            - self.thermal_conductivity_w_per_m_k
            * self.temperature_gradient_k_per_m
        )
        return _scalar_or_array(coordinate_m, value)

    def potential_v(self, coordinate_m: float | Array) -> float | Array:
        """Potential with the arbitrary reference ``phi(0)=0``."""

        temperature = _array(self.temperature_k(coordinate_m))
        seebeck = _array(self.seebeck_v_per_k(temperature))
        cold_st = self.seebeck_at_cold_v_per_k * self.cold_temperature_k
        value = cold_st - seebeck * temperature
        return _scalar_or_array(coordinate_m, value)

    def terminal_reference(self) -> dict[str, float]:
        q_cold = float(self.heat_flux_w_per_m2(0.0))
        q_hot = float(self.heat_flux_w_per_m2(self.length_m))
        delta_phi = float(self.potential_v(self.length_m))
        cold_heat = self.area_m2 * q_cold
        hot_heat = self.area_m2 * q_hot
        electrical_power = self.signed_current_a * (-delta_phi)
        return {
            "cold_heat_flux_w_per_m2": q_cold,
            "hot_heat_flux_w_per_m2": q_hot,
            "cold_heat_rate_w": cold_heat,
            "hot_heat_rate_w": hot_heat,
            "hot_minus_cold_potential_v": delta_phi,
            "electrical_power_absorbed_w": electrical_power,
            "local_energy_residual_w": hot_heat - cold_heat - electrical_power,
        }

    def expanded_balance_terms_w_per_m3(
        self, coordinate_m: float | Array
    ) -> dict[str, float | Array]:
        """Terms in ``(k*T')' + rho*J^2 - tau*J*T' = 0``."""

        temperature = self.temperature_k(coordinate_m)
        rho = _array(self.resistivity_ohm_m(temperature))
        tau = _array(self.thomson_coefficient_v_per_k(temperature))
        joule = rho * self.current_density_a_per_m2**2
        thomson_subtracted = (
            -tau
            * self.current_density_a_per_m2
            * self.temperature_gradient_k_per_m
        )
        zero = np.zeros_like(joule)
        total = zero + joule + thomson_subtracted
        return {
            "conduction_divergence": _scalar_or_array(coordinate_m, zero),
            "joule": _scalar_or_array(coordinate_m, joule),
            "negative_thomson": _scalar_or_array(
                coordinate_m, thomson_subtracted
            ),
            "sum": _scalar_or_array(coordinate_m, total),
        }


@dataclass(frozen=True)
class ZeroCurrentLinearKappaCase:
    """One-leg exact zero-current reference with linear ``kappa(T)``."""

    cold_temperature_k: float = 250.0
    hot_temperature_k: float = 330.0
    length_m: float = 1.0e-3
    area_m2: float = 1.1e-6
    kappa_at_cold_w_per_m_k: float = 1.2
    dkappa_dtemperature_w_per_m_k2: float = 0.003
    seebeck_at_cold_v_per_k: float = 200.0e-6
    dseebeck_dtemperature_v_per_k2: float = -0.30e-6

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("all zero-current case parameters must be finite")
        if self.hot_temperature_k <= self.cold_temperature_k:
            raise ValueError("hot_temperature_k must exceed cold_temperature_k")
        if self.length_m <= 0.0 or self.area_m2 <= 0.0:
            raise ValueError("length_m and area_m2 must be positive")
        kappa_hot = self.kappa_w_per_m_k(self.hot_temperature_k)
        if self.kappa_at_cold_w_per_m_k <= 0.0 or kappa_hot <= 0.0:
            raise ValueError("linear thermal conductivity must stay positive")

    @property
    def delta_temperature_k(self) -> float:
        return self.hot_temperature_k - self.cold_temperature_k

    def kappa_w_per_m_k(self, temperature_k: float | Array) -> float | Array:
        temperature = _array(temperature_k)
        value = self.kappa_at_cold_w_per_m_k + (
            self.dkappa_dtemperature_w_per_m_k2
            * (temperature - self.cold_temperature_k)
        )
        return _scalar_or_array(temperature_k, value)

    @property
    def integrated_kappa_w_per_m(self) -> float:
        delta = self.delta_temperature_k
        return (
            self.kappa_at_cold_w_per_m_k * delta
            + 0.5 * self.dkappa_dtemperature_w_per_m_k2 * delta**2
        )

    @property
    def heat_flux_w_per_m2(self) -> float:
        return -self.integrated_kappa_w_per_m / self.length_m

    def temperature_k(self, coordinate_m: float | Array) -> float | Array:
        """Invert ``integral(kappa dT) = (x/L)*integral_c^h(kappa dT)``."""

        x = _array(coordinate_m)
        fraction = x / self.length_m
        b = self.dkappa_dtemperature_w_per_m_k2
        if b == 0.0:
            rise = self.delta_temperature_k * fraction
        else:
            h = self.integrated_kappa_w_per_m
            discriminant = self.kappa_at_cold_w_per_m_k**2 + 2.0 * b * h * fraction
            rise = (
                2.0
                * h
                * fraction
                / (self.kappa_at_cold_w_per_m_k + np.sqrt(discriminant))
            )
        value = self.cold_temperature_k + rise
        return _scalar_or_array(coordinate_m, value)

    @property
    def hot_minus_cold_potential_v(self) -> float:
        delta = self.delta_temperature_k
        integral_s = (
            self.seebeck_at_cold_v_per_k * delta
            + 0.5 * self.dseebeck_dtemperature_v_per_k2 * delta**2
        )
        return -integral_s

    def terminal_reference(self) -> dict[str, float]:
        heat_rate = self.area_m2 * self.heat_flux_w_per_m2
        return {
            "cold_heat_flux_w_per_m2": self.heat_flux_w_per_m2,
            "hot_heat_flux_w_per_m2": self.heat_flux_w_per_m2,
            "cold_heat_rate_w": heat_rate,
            "hot_heat_rate_w": heat_rate,
            "hot_minus_cold_potential_v": self.hot_minus_cold_potential_v,
            "electrical_power_absorbed_w": 0.0,
            "local_energy_residual_w": 0.0,
        }


def benchmark_report() -> dict[str, Any]:
    manufactured = ManufacturedLinearTemperatureCase()
    zero_current = ZeroCurrentLinearKappaCase()
    x_manufactured = np.linspace(0.0, manufactured.length_m, 101)
    balance = manufactured.expanded_balance_terms_w_per_m3(x_manufactured)
    x_zero = np.linspace(0.0, zero_current.length_m, 101)
    t_zero = np.asarray(zero_current.temperature_k(x_zero))
    k_zero = np.asarray(zero_current.kappa_w_per_m_k(t_zero))
    dtdx_zero = np.gradient(t_zero, x_zero, edge_order=2)
    reconstructed_q = -k_zero * dtdx_zero
    rise_zero = t_zero - zero_current.cold_temperature_k
    integrated_to_x = (
        zero_current.kappa_at_cold_w_per_m_k * rise_zero
        + 0.5
        * zero_current.dkappa_dtemperature_w_per_m_k2
        * rise_zero**2
    )
    exact_integral_to_x = (
        zero_current.integrated_kappa_w_per_m * x_zero / zero_current.length_m
    )

    manufactured_terminal = manufactured.terminal_reference()
    zero_terminal = zero_current.terminal_reference()
    checks = {
        "manufactured_resistivity_positive": bool(
            np.all(
                np.asarray(
                    manufactured.resistivity_ohm_m(
                        manufactured.temperature_k(x_manufactured)
                    )
                )
                > 0.0
            )
        ),
        "manufactured_joule_and_thomson_nonzero": bool(
            np.max(np.abs(np.asarray(balance["joule"]))) > 0.0
            and np.max(np.abs(np.asarray(balance["negative_thomson"]))) > 0.0
        ),
        "manufactured_expanded_balance_exact": bool(
            np.max(np.abs(np.asarray(balance["sum"])))
            <= EXPANDED_BALANCE_ABS_TOLERANCE_W_PER_M3
        ),
        "manufactured_terminal_energy_exact": bool(
            abs(manufactured_terminal["local_energy_residual_w"])
            <= ENERGY_ABS_TOLERANCE_W
        ),
        "zero_current_endpoint_temperatures_exact": bool(
            abs(t_zero[0] - zero_current.cold_temperature_k)
            <= ENDPOINT_TEMPERATURE_ABS_TOLERANCE_K
            and abs(t_zero[-1] - zero_current.hot_temperature_k)
            <= ENDPOINT_TEMPERATURE_ABS_TOLERANCE_K
        ),
        "zero_current_conductivity_integral_exact": bool(
            np.max(np.abs(integrated_to_x - exact_integral_to_x)) <= 1.0e-10
        ),
    }
    return {
        "schema_version": 1,
        "scope": (
            "synthetic exact verification cases only; PbSe/Cr material data and "
            "device-specific validation are outside this benchmark"
        ),
        "self_consistency_tolerances": {
            "expanded_balance_absolute_w_per_m3": (
                EXPANDED_BALANCE_ABS_TOLERANCE_W_PER_M3
            ),
            "energy_absolute_w": ENERGY_ABS_TOLERANCE_W,
            "endpoint_temperature_absolute_k": (
                ENDPOINT_TEMPERATURE_ABS_TOLERANCE_K
            ),
            "zero_current_conductivity_integral_absolute_w_per_m": 1.0e-10,
        },
        "manufactured_linear_temperature_case": {
            "parameters": asdict(manufactured),
            "derived": {
                "temperature_gradient_k_per_m": manufactured.temperature_gradient_k_per_m,
                "current_density_a_per_m2": manufactured.current_density_a_per_m2,
                "rho_cold_ohm_m": manufactured.resistivity_ohm_m(
                    manufactured.cold_temperature_k
                ),
                "rho_hot_ohm_m": manufactured.resistivity_ohm_m(
                    manufactured.hot_temperature_k
                ),
                "max_expanded_balance_abs_w_per_m3": float(
                    np.max(np.abs(np.asarray(balance["sum"])))
                ),
            },
            "terminal_reference": manufactured_terminal,
        },
        "zero_current_linear_kappa_case": {
            "parameters": asdict(zero_current),
            "derived": {
                "integrated_kappa_w_per_m": zero_current.integrated_kappa_w_per_m,
                "midpoint_temperature_k": float(
                    zero_current.temperature_k(0.5 * zero_current.length_m)
                ),
                "finite_difference_flux_spread_w_per_m2_diagnostic_only": float(
                    np.ptp(reconstructed_q)
                ),
            },
            "terminal_reference": zero_terminal,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/solver_verification/KT2_temperature_dependent_verification.json"
        ),
        help="JSON output path; stdout is also printed",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = benchmark_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
