"""Analytic constant-property thermoelectric-couple reference.

Sign and unit convention
------------------------
All inputs and outputs use SI units.  ``S_p`` and ``S_n`` are in V/K, total
couple resistance ``R`` is in ohm, total parallel thermal conductance ``K`` is
in W/K, temperatures are in K, and current is in A.  The temperature
difference is ``delta_T = T_h - T_c >= 0`` and
``alpha = S_p - S_n``.  Positive current is defined by the supplied scalar
coordinate.  It cools the cold junction when ``alpha * I > 0`` is large enough
to overcome Joule heating and conductive backflow.

For uniform properties, no contacts or parasitic heat leaks, and equal
partition of bulk Joule heat between the two ends, the model is

    Qc  = alpha * I * Tc - 0.5 * I**2 * R - K * delta_T
    Qh  = alpha * I * Th + 0.5 * I**2 * R - K * delta_T
    V   = I * R + alpha * delta_T
    Pin = I * V

``Qc`` is heat extracted from the cold reservoir and ``Qh`` is heat rejected
to the hot reservoir.  Thus ``Qh - Qc = Pin``.  A negative ``Qc`` means that
the device heats, rather than cools, the nominal cold reservoir.  Cooling COP
is returned only when both ``Qc`` and ``Pin`` are positive; otherwise it is
``None`` because the operating point is not a powered refrigerator.

This module is an analytic verification target, not the temperature-dependent
nonlinear solver.  In particular, its half-Joule split must not be copied into
nonuniform, segmented, or contact-resistance models without a derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


def _finite_float(name: str, value: object) -> float:
    """Return *value* as a finite float, with an informative validation error."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class ConstantPropertyCouple:
    """Parameters of a uniform p/n thermoelectric couple in SI units.

    ``electrical_resistance_ohm`` is the total series resistance of both legs.
    ``thermal_conductance_w_per_k`` is their total parallel conductance.
    Individual Seebeck coefficients are retained so common-mode shifts can be
    tested explicitly, although only their difference enters this ideal limit.
    """

    seebeck_p_v_per_k: float
    seebeck_n_v_per_k: float
    electrical_resistance_ohm: float
    thermal_conductance_w_per_k: float
    cold_temperature_k: float
    hot_temperature_k: float

    def __post_init__(self) -> None:
        for name in (
            "seebeck_p_v_per_k",
            "seebeck_n_v_per_k",
            "electrical_resistance_ohm",
            "thermal_conductance_w_per_k",
            "cold_temperature_k",
            "hot_temperature_k",
        ):
            object.__setattr__(self, name, _finite_float(name, getattr(self, name)))

        if self.electrical_resistance_ohm <= 0.0:
            raise ValueError("electrical_resistance_ohm must be > 0")
        if self.thermal_conductance_w_per_k < 0.0:
            raise ValueError("thermal_conductance_w_per_k must be >= 0")
        if self.cold_temperature_k <= 0.0 or self.hot_temperature_k <= 0.0:
            raise ValueError("absolute temperatures must be > 0 K")
        if self.hot_temperature_k < self.cold_temperature_k:
            raise ValueError("hot_temperature_k must be >= cold_temperature_k")

    @property
    def delta_seebeck_v_per_k(self) -> float:
        """Couple Seebeck difference ``S_p - S_n`` in V/K."""

        return self.seebeck_p_v_per_k - self.seebeck_n_v_per_k

    @property
    def common_mode_seebeck_v_per_k(self) -> float:
        """Common mode ``(S_p + S_n) / 2`` in V/K."""

        return 0.5 * (self.seebeck_p_v_per_k + self.seebeck_n_v_per_k)

    @property
    def delta_temperature_k(self) -> float:
        """Applied temperature difference ``T_h - T_c`` in K."""

        return self.hot_temperature_k - self.cold_temperature_k

    def evaluate(self, current_a: float) -> "OperatingPoint":
        """Evaluate all terminal quantities at ``current_a``."""

        return evaluate(self, current_a)


@dataclass(frozen=True)
class OperatingPoint:
    """Terminal observables for one current, all in SI units."""

    current_a: float
    Qc_w: float
    Qh_w: float
    V_v: float
    Pin_w: float
    COP: Optional[float]
    energy_residual_w: float

    @property
    def relative_energy_residual(self) -> float:
        """Absolute energy residual scaled by the largest terminal power."""

        scale = max(abs(self.Qc_w), abs(self.Qh_w), abs(self.Pin_w))
        if scale == 0.0:
            return 0.0
        return abs(self.energy_residual_w) / scale


def evaluate(couple: ConstantPropertyCouple, current_a: float) -> OperatingPoint:
    """Return ``Qc``, ``Qh``, ``V``, ``Pin``, COP, and energy residual.

    The energy residual is signed and defined as ``Qh - Qc - Pin`` in W.
    It should differ from zero only by floating-point roundoff.
    """

    if not isinstance(couple, ConstantPropertyCouple):
        raise TypeError("couple must be a ConstantPropertyCouple")
    current = _finite_float("current_a", current_a)
    alpha = couple.delta_seebeck_v_per_k
    resistance = couple.electrical_resistance_ohm
    conduction = couple.thermal_conductance_w_per_k * couple.delta_temperature_k
    joule_half = 0.5 * current * current * resistance

    q_cold = alpha * current * couple.cold_temperature_k - joule_half - conduction
    q_hot = alpha * current * couple.hot_temperature_k + joule_half - conduction
    voltage = current * resistance + alpha * couple.delta_temperature_k
    input_power = current * voltage
    cooling_cop = q_cold / input_power if q_cold > 0.0 and input_power > 0.0 else None
    residual = q_hot - q_cold - input_power

    return OperatingPoint(
        current_a=current,
        Qc_w=q_cold,
        Qh_w=q_hot,
        V_v=voltage,
        Pin_w=input_power,
        COP=cooling_cop,
        energy_residual_w=residual,
    )


def Qc(couple: ConstantPropertyCouple, current_a: float) -> float:
    """Cold-side cooling power in W."""

    return evaluate(couple, current_a).Qc_w


def Qh(couple: ConstantPropertyCouple, current_a: float) -> float:
    """Hot-side rejected heat in W."""

    return evaluate(couple, current_a).Qh_w


def V(couple: ConstantPropertyCouple, current_a: float) -> float:
    """Terminal voltage in V."""

    return evaluate(couple, current_a).V_v


def Pin(couple: ConstantPropertyCouple, current_a: float) -> float:
    """Electrical input power ``I * V`` in W."""

    return evaluate(couple, current_a).Pin_w


def COP(couple: ConstantPropertyCouple, current_a: float) -> Optional[float]:
    """Cooling coefficient of performance, or ``None`` outside cooling mode."""

    return evaluate(couple, current_a).COP


def energy_residual(couple: ConstantPropertyCouple, current_a: float) -> float:
    """Signed terminal energy residual ``Qh - Qc - Pin`` in W."""

    return evaluate(couple, current_a).energy_residual_w


def I_opt_for_max_Qc(couple: ConstantPropertyCouple) -> float:
    """Unconstrained current maximizing ``Qc`` in A.

    Differentiating the concave quadratic in current gives
    ``I_opt = (S_p - S_n) * T_c / R``.  The sign therefore follows the chosen
    current convention and the sign of the Seebeck difference.  The result is
    the maximum of the analytic curve even when that maximum remains negative
    because conductive backflow is too large for refrigeration.
    """

    if not isinstance(couple, ConstantPropertyCouple):
        raise TypeError("couple must be a ConstantPropertyCouple")
    return (
        couple.delta_seebeck_v_per_k
        * couple.cold_temperature_k
        / couple.electrical_resistance_ohm
    )
