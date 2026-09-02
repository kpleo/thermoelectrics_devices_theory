"""Conservative 1D thermoelectric solver with temperature-dependent properties.

The two leg coordinates both point from the nominal cold-side plane to the
nominal hot-side plane.  Those names identify physical terminals, not a
required ordering of their solved temperatures: active operation may produce
``T_cold-side > T_hot-side``.  The local signed currents are therefore ``+I``
in the p leg and ``-I`` in the n leg.  For each leg the solver advances the
*total* heat flux ``q`` rather than adding separate Peltier or Thomson sources::

    T'   = (S(T) * T * J - q) / kappa(T)
    q'   = rho(T) * J**2 + S(T) * J * T'
    phi' = -rho(T) * J - S(T) * T'

These equations are exactly equivalent to

    (kappa T')' + rho J**2 - T (dS/dT) J T' = 0,

but the two forms must not be used simultaneously.  Fixed endpoint
temperatures close the boundary-value problem; contacts, parasitic heat leaks,
radiation, and nonuniform area are deliberately outside this module.

Internally, each leg uses the canonical constant-Seebeck gauge
``S0 = S(T_min)`` with ``Sbar = S - S0``, ``qbar = q - S0*T*J``, and
``phibar = phi + S0*(T - T_cold)``.  This leaves the physical equations and
reported fields unchanged while making a constant common-mode Seebeck shift
numerically invisible to the boundary-value solve.

All inputs and outputs use SI units. Property laws carry an explicit closed
temperature validity interval and refuse extrapolation. Synthetic or
room-temperature laws therefore cannot enter an unsupported domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_bvp
from scipy.interpolate import PchipInterpolator

from .constant_properties import OperatingPoint


FloatArray = NDArray[np.float64]


class PropertyDomainError(ValueError):
    """Raised when a property law is evaluated outside its declared domain."""


class SolverConvergenceError(RuntimeError):
    """Raised when the nonlinear boundary-value solve does not converge."""


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


def _positive_finite(name: str, value: object) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _validate_temperature_domain(
    minimum_temperature_k: object,
    maximum_temperature_k: object,
) -> tuple[float, float]:
    minimum = _positive_finite("minimum_temperature_k", minimum_temperature_k)
    maximum = _positive_finite("maximum_temperature_k", maximum_temperature_k)
    if maximum <= minimum:
        raise ValueError("maximum_temperature_k must be > minimum_temperature_k")
    return minimum, maximum


def _checked_temperature_array(
    temperature_k: ArrayLike,
    *,
    minimum_temperature_k: float,
    maximum_temperature_k: float,
) -> FloatArray:
    temperature = np.asarray(temperature_k, dtype=float)
    if not np.all(np.isfinite(temperature)):
        raise PropertyDomainError("temperature must contain only finite values")
    if np.any(temperature <= 0.0):
        raise PropertyDomainError("absolute temperature must be > 0 K")
    # Nonlinear BVP iterates can land one or two representable floats beyond an
    # exactly imposed closed-domain endpoint.  Admit only a small machine-scale
    # halo, then project it back to the specified endpoint before evaluating a
    # property law.  This is numerical roundoff handling, not extrapolation.
    epsilon = np.finfo(float).eps
    lower_roundoff = 16.0 * epsilon * max(abs(minimum_temperature_k), 1.0)
    upper_roundoff = 16.0 * epsilon * max(abs(maximum_temperature_k), 1.0)
    if np.any(temperature < minimum_temperature_k - lower_roundoff) or np.any(
        temperature > maximum_temperature_k + upper_roundoff
    ):
        observed_min = float(np.min(temperature))
        observed_max = float(np.max(temperature))
        raise PropertyDomainError(
            "temperature outside declared property domain "
            f"[{minimum_temperature_k:g}, {maximum_temperature_k:g}] K; "
            f"observed [{observed_min:g}, {observed_max:g}] K"
        )
    return np.asarray(
        np.clip(temperature, minimum_temperature_k, maximum_temperature_k),
        dtype=float,
    )


@dataclass(frozen=True)
class ConstantTemperatureProperty:
    """Constant SI-valued property on an explicit temperature interval."""

    value_si: float
    minimum_temperature_k: float
    maximum_temperature_k: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_si", _finite("value_si", self.value_si))
        minimum, maximum = _validate_temperature_domain(
            self.minimum_temperature_k, self.maximum_temperature_k
        )
        object.__setattr__(self, "minimum_temperature_k", minimum)
        object.__setattr__(self, "maximum_temperature_k", maximum)

    def evaluate(self, temperature_k: ArrayLike) -> FloatArray:
        temperature = _checked_temperature_array(
            temperature_k,
            minimum_temperature_k=self.minimum_temperature_k,
            maximum_temperature_k=self.maximum_temperature_k,
        )
        return np.full_like(temperature, self.value_si, dtype=float)

    def derivative(self, temperature_k: ArrayLike) -> FloatArray:
        temperature = _checked_temperature_array(
            temperature_k,
            minimum_temperature_k=self.minimum_temperature_k,
            maximum_temperature_k=self.maximum_temperature_k,
        )
        return np.zeros_like(temperature, dtype=float)


@dataclass(frozen=True)
class LinearTemperatureProperty:
    """Linear SI-valued property on an explicit temperature interval.

    ``value_si(T) = value_at_reference_si + slope_si_per_k *
    (T - reference_temperature_k)``.
    """

    reference_temperature_k: float
    value_at_reference_si: float
    slope_si_per_k: float
    minimum_temperature_k: float
    maximum_temperature_k: float

    def __post_init__(self) -> None:
        reference = _positive_finite(
            "reference_temperature_k", self.reference_temperature_k
        )
        value = _finite("value_at_reference_si", self.value_at_reference_si)
        slope = _finite("slope_si_per_k", self.slope_si_per_k)
        minimum, maximum = _validate_temperature_domain(
            self.minimum_temperature_k, self.maximum_temperature_k
        )
        if not minimum <= reference <= maximum:
            raise ValueError(
                "reference_temperature_k must lie inside the declared domain"
            )
        object.__setattr__(self, "reference_temperature_k", reference)
        object.__setattr__(self, "value_at_reference_si", value)
        object.__setattr__(self, "slope_si_per_k", slope)
        object.__setattr__(self, "minimum_temperature_k", minimum)
        object.__setattr__(self, "maximum_temperature_k", maximum)

    def evaluate(self, temperature_k: ArrayLike) -> FloatArray:
        temperature = _checked_temperature_array(
            temperature_k,
            minimum_temperature_k=self.minimum_temperature_k,
            maximum_temperature_k=self.maximum_temperature_k,
        )
        return np.asarray(
            self.value_at_reference_si
            + self.slope_si_per_k * (temperature - self.reference_temperature_k),
            dtype=float,
        )

    def derivative(self, temperature_k: ArrayLike) -> FloatArray:
        temperature = _checked_temperature_array(
            temperature_k,
            minimum_temperature_k=self.minimum_temperature_k,
            maximum_temperature_k=self.maximum_temperature_k,
        )
        return np.full_like(temperature, self.slope_si_per_k, dtype=float)


@dataclass(frozen=True)
class PchipTemperatureProperty:
    """Shape-preserving tabulated property on its closed knot interval.

    SciPy's piecewise-cubic Hermite interpolant is used with extrapolation
    disabled.  Temperatures must be strictly increasing, positive, finite, and
    paired one-to-one with finite SI-valued samples.  The first and last knots
    are the declared validity limits; evaluation at those limits is allowed,
    while every value outside them is rejected before interpolation.

    The analytic derivative of the piecewise polynomial is returned by
    :meth:`derivative`, so a tabulated Seebeck law supplies an internally
    consistent ``tau = T dS/dT`` to the conservative solver.
    """

    temperature_knots_k: ArrayLike
    values_si: ArrayLike
    _interpolator: PchipInterpolator = field(
        init=False, repr=False, compare=False
    )
    _derivative_interpolator: object = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        try:
            raw_temperature = np.asarray(self.temperature_knots_k)
            raw_values = np.asarray(self.values_si)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "temperature_knots_k and values_si must be one-dimensional "
                "real numeric sequences"
            ) from exc
        if np.iscomplexobj(raw_temperature) or np.iscomplexobj(raw_values):
            raise TypeError("temperature_knots_k and values_si must be real-valued")
        try:
            temperature = np.asarray(raw_temperature, dtype=float)
            values = np.asarray(raw_values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "temperature_knots_k and values_si must be one-dimensional "
                "real numeric sequences"
            ) from exc
        if temperature.ndim != 1 or values.ndim != 1:
            raise ValueError("temperature_knots_k and values_si must be one-dimensional")
        if temperature.size < 2:
            raise ValueError("at least two temperature/value knots are required")
        if values.size != temperature.size:
            raise ValueError(
                "temperature_knots_k and values_si must have the same length"
            )
        if not np.all(np.isfinite(temperature)):
            raise ValueError("temperature_knots_k must contain only finite values")
        if np.any(temperature <= 0.0):
            raise ValueError("temperature_knots_k must all be > 0 K")
        if np.any(np.diff(temperature) <= 0.0):
            raise ValueError(
                "temperature_knots_k must be strictly increasing; duplicate "
                "knots are not allowed"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("values_si must contain only finite values")

        # Store immutable Python tuples so construction from lists and ndarrays
        # has the same reproducible representation and equality semantics.
        temperature_tuple = tuple(float(item) for item in temperature)
        values_tuple = tuple(float(item) for item in values)
        interpolator = PchipInterpolator(
            np.asarray(temperature_tuple, dtype=float),
            np.asarray(values_tuple, dtype=float),
            extrapolate=False,
        )
        object.__setattr__(self, "temperature_knots_k", temperature_tuple)
        object.__setattr__(self, "values_si", values_tuple)
        object.__setattr__(self, "_interpolator", interpolator)
        object.__setattr__(self, "_derivative_interpolator", interpolator.derivative())

    @property
    def minimum_temperature_k(self) -> float:
        return float(self.temperature_knots_k[0])

    @property
    def maximum_temperature_k(self) -> float:
        return float(self.temperature_knots_k[-1])

    def evaluate(self, temperature_k: ArrayLike) -> FloatArray:
        temperature = _checked_temperature_array(
            temperature_k,
            minimum_temperature_k=self.minimum_temperature_k,
            maximum_temperature_k=self.maximum_temperature_k,
        )
        return np.asarray(self._interpolator(temperature), dtype=float)

    def value(self, temperature_k: ArrayLike) -> FloatArray:
        """Alias for :meth:`evaluate` for property-law client code."""

        return self.evaluate(temperature_k)

    def derivative(self, temperature_k: ArrayLike) -> FloatArray:
        temperature = _checked_temperature_array(
            temperature_k,
            minimum_temperature_k=self.minimum_temperature_k,
            maximum_temperature_k=self.maximum_temperature_k,
        )
        return np.asarray(self._derivative_interpolator(temperature), dtype=float)


TemperatureProperty: TypeAlias = (
    ConstantTemperatureProperty
    | LinearTemperatureProperty
    | PchipTemperatureProperty
)
_TEMPERATURE_PROPERTY_TYPES = (
    ConstantTemperatureProperty,
    LinearTemperatureProperty,
    PchipTemperatureProperty,
)


def _minimum_on_closed_interval(
    property_law: TemperatureProperty,
    minimum_temperature_k: float,
    maximum_temperature_k: float,
) -> float:
    """Return the exact candidate minimum for a supported property law.

    Constant and linear laws need only the common-domain endpoints.  For a
    PCHIP law, all contained knots and every real stationary point of the
    piecewise cubic are also checked.  This keeps positive transport
    coefficients fail-closed over the *whole* common domain rather than at a
    few arbitrary samples.
    """

    candidates = [minimum_temperature_k, maximum_temperature_k]
    if isinstance(property_law, PchipTemperatureProperty):
        candidates.extend(
            knot
            for knot in property_law.temperature_knots_k
            if minimum_temperature_k <= knot <= maximum_temperature_k
        )
        stationary_points = np.asarray(
            property_law._derivative_interpolator.roots(extrapolate=False),
            dtype=float,
        )
        candidates.extend(
            float(point)
            for point in stationary_points
            if math.isfinite(float(point))
            and minimum_temperature_k <= point <= maximum_temperature_k
        )
    candidate_array = np.unique(np.asarray(candidates, dtype=float))
    values = property_law.evaluate(candidate_array)
    if np.any(~np.isfinite(values)):
        raise ValueError("property law returned non-finite values on its domain")
    return float(np.min(values))


@dataclass(frozen=True)
class TemperatureDependentLeg:
    """One uniform-area leg with temperature-dependent bulk properties.

    The values returned by ``seebeck``, ``electrical_resistivity``, and
    ``thermal_conductivity`` are interpreted as V/K, ohm m, and W/(m K),
    respectively.
    """

    seebeck: TemperatureProperty
    electrical_resistivity: TemperatureProperty
    thermal_conductivity: TemperatureProperty
    length_m: float
    area_m2: float

    def __post_init__(self) -> None:
        for name in ("seebeck", "electrical_resistivity", "thermal_conductivity"):
            if not isinstance(getattr(self, name), _TEMPERATURE_PROPERTY_TYPES):
                raise TypeError(
                    f"{name} must be a ConstantTemperatureProperty, "
                    "LinearTemperatureProperty, or PchipTemperatureProperty"
                )
        object.__setattr__(self, "length_m", _positive_finite("length_m", self.length_m))
        object.__setattr__(self, "area_m2", _positive_finite("area_m2", self.area_m2))

        minimum = self.minimum_valid_temperature_k
        maximum = self.maximum_valid_temperature_k
        if maximum <= minimum:
            raise ValueError("the three property-law temperature domains do not overlap")
        minimum_resistivity = _minimum_on_closed_interval(
            self.electrical_resistivity, minimum, maximum
        )
        minimum_conductivity = _minimum_on_closed_interval(
            self.thermal_conductivity, minimum, maximum
        )
        if minimum_resistivity <= 0.0:
            raise ValueError(
                "electrical_resistivity must remain > 0 throughout the common domain"
            )
        if minimum_conductivity <= 0.0:
            raise ValueError(
                "thermal_conductivity must remain > 0 throughout the common domain"
            )

    @property
    def minimum_valid_temperature_k(self) -> float:
        return max(
            self.seebeck.minimum_temperature_k,
            self.electrical_resistivity.minimum_temperature_k,
            self.thermal_conductivity.minimum_temperature_k,
        )

    @property
    def maximum_valid_temperature_k(self) -> float:
        return min(
            self.seebeck.maximum_temperature_k,
            self.electrical_resistivity.maximum_temperature_k,
            self.thermal_conductivity.maximum_temperature_k,
        )


@dataclass(frozen=True)
class TemperatureDependentNumericalCouple:
    """p/n legs sharing fixed isothermal nominal cold- and hot-side planes.

    The terminal names encode topology and sign conventions.  Their numerical
    temperatures may reverse under active or reverse-current operation.
    """

    p_leg: TemperatureDependentLeg
    n_leg: TemperatureDependentLeg
    cold_temperature_k: float
    hot_temperature_k: float

    def __post_init__(self) -> None:
        if not isinstance(self.p_leg, TemperatureDependentLeg) or not isinstance(
            self.n_leg, TemperatureDependentLeg
        ):
            raise TypeError("p_leg and n_leg must be TemperatureDependentLeg objects")
        cold = _positive_finite("cold_temperature_k", self.cold_temperature_k)
        hot = _positive_finite("hot_temperature_k", self.hot_temperature_k)
        object.__setattr__(self, "cold_temperature_k", cold)
        object.__setattr__(self, "hot_temperature_k", hot)
        endpoint_minimum = min(cold, hot)
        endpoint_maximum = max(cold, hot)
        for leg_name, leg in (("p_leg", self.p_leg), ("n_leg", self.n_leg)):
            if (
                endpoint_minimum < leg.minimum_valid_temperature_k
                or endpoint_maximum > leg.maximum_valid_temperature_k
            ):
                raise PropertyDomainError(
                    f"{leg_name} endpoint temperatures [{cold:g}, {hot:g}] K are "
                    "outside its common property domain "
                    f"[{leg.minimum_valid_temperature_k:g}, "
                    f"{leg.maximum_valid_temperature_k:g}] K"
                )


@dataclass(frozen=True)
class TemperatureDependentLegSolution:
    """Resolved fields, endpoint rates, and diagnostics for one leg."""

    coordinate_m: FloatArray
    temperature_k: FloatArray
    heat_flux_w_per_m2: FloatArray
    potential_v: FloatArray
    temperature_gradient_k_per_m: FloatArray
    electric_field_v_per_m: FloatArray
    seebeck_v_per_k: FloatArray
    electrical_resistivity_ohm_m: FloatArray
    thermal_conductivity_w_per_m_k: FloatArray
    thomson_coefficient_v_per_k: FloatArray
    signed_current_a: float
    current_density_a_per_m2: float
    cold_heat_rate_w: float
    hot_heat_rate_w: float
    hot_minus_cold_potential_v: float
    local_energy_residual_w: float
    relative_local_energy_residual: float
    maximum_rms_bvp_residual: float
    maximum_relative_conservative_residual: float
    adaptive_mesh_nodes: int
    nonlinear_iterations: int


@dataclass(frozen=True)
class TemperatureDependentOperatingPoint:
    """Terminal observables and both temperature-dependent leg solutions."""

    current_a: float
    Qc_w: float
    Qh_w: float
    V_v: float
    Pin_w: float
    COP: float | None
    energy_residual_w: float
    p_leg: TemperatureDependentLegSolution
    n_leg: TemperatureDependentLegSolution

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


def _leg_properties(
    leg: TemperatureDependentLeg, temperature_k: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    try:
        seebeck = leg.seebeck.evaluate(temperature_k)
        dseebeck_dtemperature = leg.seebeck.derivative(temperature_k)
        resistivity = leg.electrical_resistivity.evaluate(temperature_k)
        conductivity = leg.thermal_conductivity.evaluate(temperature_k)
    except PropertyDomainError:
        raise
    if np.any(~np.isfinite(seebeck)) or np.any(~np.isfinite(dseebeck_dtemperature)):
        raise ValueError("Seebeck law returned non-finite values")
    if np.any(~np.isfinite(resistivity)) or np.any(resistivity <= 0.0):
        raise ValueError("electrical_resistivity must evaluate to finite values > 0")
    if np.any(~np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
        raise ValueError("thermal_conductivity must evaluate to finite values > 0")
    return seebeck, resistivity, conductivity, dseebeck_dtemperature


def _validate_integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def solve_temperature_dependent_leg(
    leg: TemperatureDependentLeg,
    signed_current_a: float,
    cold_temperature_k: float,
    hot_temperature_k: float,
    *,
    initial_mesh_points: int = 41,
    output_points: int = 401,
    relative_tolerance: float = 1.0e-7,
    max_nodes: int = 10000,
) -> TemperatureDependentLegSolution:
    """Solve one fixed-temperature leg on a nominal cold-to-hot coordinate.

    ``signed_current_a`` is the conventional current along the local coordinate.
    The endpoint names are topological and do not require ``hot >= cold``.
    A converged solution that exits any property validity interval is rejected;
    there is no property extrapolation.
    """

    if not isinstance(leg, TemperatureDependentLeg):
        raise TypeError("leg must be a TemperatureDependentLeg")
    signed_current = _finite("signed_current_a", signed_current_a)
    cold = _positive_finite("cold_temperature_k", cold_temperature_k)
    hot = _positive_finite("hot_temperature_k", hot_temperature_k)
    endpoint_minimum = min(cold, hot)
    endpoint_maximum = max(cold, hot)
    if (
        endpoint_minimum < leg.minimum_valid_temperature_k
        or endpoint_maximum > leg.maximum_valid_temperature_k
    ):
        raise PropertyDomainError(
            f"endpoint temperatures [{cold:g}, {hot:g}] K are outside common "
            f"property domain [{leg.minimum_valid_temperature_k:g}, "
            f"{leg.maximum_valid_temperature_k:g}] K"
        )
    n_initial = _validate_integer("initial_mesh_points", initial_mesh_points, minimum=3)
    n_output = _validate_integer("output_points", output_points, minimum=3)
    n_max = _validate_integer("max_nodes", max_nodes, minimum=n_initial)
    tolerance = _positive_finite("relative_tolerance", relative_tolerance)
    current_density = signed_current / leg.area_m2
    gauge_seebeck = float(
        leg.seebeck.evaluate(
            np.asarray([leg.minimum_valid_temperature_k], dtype=float)
        )[0]
    )

    # Solve on the dimensionless coordinate u=x/L.  On the raw metre-scale
    # coordinate, adaptive refinement near an interior temperature turning
    # point can drive interval widths to O(1e-12 m).  At T=O(300 K), one
    # floating-point ulp divided by such an interval is O(1e-2--1e-1 K/m), so
    # solve_bvp's derivative residual rises instead of converging.  Mapping the
    # fixed domain to [0, 1] removes that scale-induced roundoff floor while all
    # public coordinates and gradients remain in SI units.
    unit_coordinate_initial = np.linspace(0.0, 1.0, n_initial, dtype=float)
    coordinate_initial_m = leg.length_m * unit_coordinate_initial
    temperature_initial = np.linspace(cold, hot, n_initial, dtype=float)
    seebeck, resistivity, conductivity, _ = _leg_properties(
        leg, temperature_initial
    )
    reduced_seebeck = seebeck - gauge_seebeck
    temperature_gradient_initial = np.full(
        n_initial, (hot - cold) / leg.length_m, dtype=float
    )
    reduced_heat_flux_initial = (
        reduced_seebeck * temperature_initial * current_density
        - conductivity * temperature_gradient_initial
    )
    # qbar can be O(10^4--10^6 W/m2) while T and phibar are O(10^2) and
    # O(10^-2).  Keeping that raw scale in the collocation state creates a
    # roundoff floor in solve_bvp's residual estimator (most visible for tiny
    # nonzero current).  Scale only qbar; all public fields remain physical SI.
    heat_flux_scale = max(
        float(np.max(np.abs(reduced_heat_flux_initial))),
        float(
            np.max(
                np.abs(reduced_seebeck * temperature_initial * current_density)
            )
        ),
        float(np.max(np.abs(conductivity * temperature_gradient_initial))),
        float(np.max(np.abs(resistivity * current_density * current_density)))
        * leg.length_m,
        1.0,
    )
    potential_gradient_initial = (
        -resistivity * current_density
        - reduced_seebeck * temperature_gradient_initial
    )
    potential_initial = np.zeros(n_initial, dtype=float)
    if n_initial > 1:
        increments = (
            0.5
            * (potential_gradient_initial[1:] + potential_gradient_initial[:-1])
            * np.diff(coordinate_initial_m)
        )
        potential_initial[1:] = np.cumsum(increments)
    state_initial = np.vstack(
        (
            temperature_initial,
            reduced_heat_flux_initial / heat_flux_scale,
            potential_initial,
        )
    )

    def differential_equations(
        _unit_coordinate: FloatArray, state: FloatArray
    ) -> FloatArray:
        temperature = state[0]
        reduced_heat_flux = state[1] * heat_flux_scale
        local_seebeck, local_resistivity, local_conductivity, _ = _leg_properties(
            leg, temperature
        )
        local_reduced_seebeck = local_seebeck - gauge_seebeck
        temperature_gradient = (
            local_reduced_seebeck * temperature * current_density
            - reduced_heat_flux
        ) / local_conductivity
        reduced_heat_flux_gradient = (
            local_resistivity * current_density * current_density
            + local_reduced_seebeck * current_density * temperature_gradient
        )
        reduced_potential_gradient = (
            -local_resistivity * current_density
            - local_reduced_seebeck * temperature_gradient
        )
        return leg.length_m * np.vstack(
            (
                temperature_gradient,
                reduced_heat_flux_gradient / heat_flux_scale,
                reduced_potential_gradient,
            )
        )

    def differential_equation_jacobian(
        _unit_coordinate: FloatArray, state: FloatArray
    ) -> FloatArray:
        """Analytic state Jacobian for the conservative first-order system.

        Supplying this is also part of the property-domain firewall.  SciPy's
        numerical Jacobian otherwise perturbs an endpoint temperature by about
        ``sqrt(eps) * (1 + |T|)``, which asks a closed-domain property law to
        extrapolate even when the physical iterate lies exactly on its bound.
        """

        temperature = state[0]
        reduced_heat_flux = state[1] * heat_flux_scale
        local_seebeck, local_resistivity, local_conductivity, dseebeck = (
            _leg_properties(leg, temperature)
        )
        local_reduced_seebeck = local_seebeck - gauge_seebeck
        dresistivity = leg.electrical_resistivity.derivative(temperature)
        dconductivity = leg.thermal_conductivity.derivative(temperature)
        if np.any(~np.isfinite(dresistivity)) or np.any(
            ~np.isfinite(dconductivity)
        ):
            raise ValueError("transport-property derivative returned non-finite values")

        temperature_gradient = (
            local_reduced_seebeck * temperature * current_density
            - reduced_heat_flux
        ) / local_conductivity
        gradient_derivative_temperature = (
            current_density * (local_reduced_seebeck + temperature * dseebeck)
            - temperature_gradient * dconductivity
        ) / local_conductivity
        gradient_derivative_physical_heat_flux = -1.0 / local_conductivity
        gradient_derivative_scaled_heat_flux = (
            heat_flux_scale * gradient_derivative_physical_heat_flux
        )

        jacobian = np.zeros((3, 3, temperature.size), dtype=float)
        jacobian[0, 0] = gradient_derivative_temperature
        jacobian[0, 1] = gradient_derivative_scaled_heat_flux
        jacobian[1, 0] = (
            dresistivity * current_density * current_density
            + current_density
            * (
                dseebeck * temperature_gradient
                + local_reduced_seebeck * gradient_derivative_temperature
            )
        ) / heat_flux_scale
        jacobian[1, 1] = (
            local_reduced_seebeck
            * current_density
            * gradient_derivative_physical_heat_flux
        )
        jacobian[2, 0] = (
            -dresistivity * current_density
            - dseebeck * temperature_gradient
            - local_reduced_seebeck * gradient_derivative_temperature
        )
        jacobian[2, 1] = (
            -local_reduced_seebeck * gradient_derivative_scaled_heat_flux
        )
        return leg.length_m * jacobian

    def boundary_conditions(state_cold: FloatArray, state_hot: FloatArray) -> FloatArray:
        return np.asarray(
            [state_cold[0] - cold, state_hot[0] - hot, state_cold[2]],
            dtype=float,
        )

    def boundary_condition_jacobian(
        _state_cold: FloatArray, _state_hot: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        cold_jacobian = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        hot_jacobian = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=float,
        )
        return cold_jacobian, hot_jacobian

    try:
        result = solve_bvp(
            differential_equations,
            boundary_conditions,
            unit_coordinate_initial,
            state_initial,
            tol=tolerance,
            max_nodes=n_max,
            verbose=0,
            bc_tol=tolerance,
            fun_jac=differential_equation_jacobian,
            bc_jac=boundary_condition_jacobian,
        )
    except PropertyDomainError as exc:
        raise PropertyDomainError(
            "nonlinear leg solve requested property values outside the declared "
            f"temperature domain: {exc}"
        ) from exc
    if not result.success:
        raise SolverConvergenceError(
            f"temperature-dependent leg BVP failed (status {result.status}): "
            f"{result.message}"
        )

    unit_coordinate = np.linspace(0.0, 1.0, n_output, dtype=float)
    coordinate = leg.length_m * unit_coordinate
    state = np.asarray(result.sol(unit_coordinate), dtype=float)
    temperature = np.asarray(state[0], dtype=float).copy()
    reduced_heat_flux = state[1] * heat_flux_scale
    reduced_potential = np.asarray(state[2], dtype=float).copy()
    # Restore the exact semantics of the imposed Dirichlet conditions after
    # spline evaluation, which can otherwise return an adjacent float.
    temperature[0] = cold
    temperature[-1] = hot
    reduced_potential -= reduced_potential[0]
    heat_flux = (
        reduced_heat_flux + gauge_seebeck * temperature * current_density
    )
    potential = reduced_potential - gauge_seebeck * (temperature - cold)
    potential[0] = 0.0
    local_seebeck, local_resistivity, local_conductivity, dseebeck = _leg_properties(
        leg, temperature
    )
    local_reduced_seebeck = local_seebeck - gauge_seebeck
    temperature_gradient = (
        local_reduced_seebeck * temperature * current_density
        - reduced_heat_flux
    ) / local_conductivity
    electric_field = (
        local_resistivity * current_density
        + local_seebeck * temperature_gradient
    )
    thomson = temperature * dseebeck

    cold_heat = leg.area_m2 * float(heat_flux[0])
    hot_heat = leg.area_m2 * float(heat_flux[-1])
    potential_difference = float(potential[-1] - potential[0])
    local_electrical_power = signed_current * (-potential_difference)
    local_energy_residual = hot_heat - cold_heat - local_electrical_power
    local_scale = max(abs(cold_heat), abs(hot_heat), abs(local_electrical_power))
    relative_local_residual = (
        0.0 if local_scale == 0.0 else abs(local_energy_residual) / local_scale
    )

    # Independent sampled-grid diagnostic for q' = J E.  The BVP residual is
    # also retained because the sampled finite-difference diagnostic naturally
    # becomes less accurate on coarse output grids.
    numerical_heat_flux_gradient = np.gradient(
        heat_flux, coordinate, edge_order=2
    )
    conservative_residual = numerical_heat_flux_gradient - current_density * electric_field
    conservative_scale = np.maximum(
        np.maximum(np.abs(numerical_heat_flux_gradient), np.abs(current_density * electric_field)),
        1.0,
    )
    conservative_relative = np.abs(conservative_residual) / conservative_scale
    interior = conservative_relative[1:-1]
    maximum_relative_conservative_residual = float(
        np.max(interior if interior.size else conservative_relative)
    )
    maximum_rms_bvp_residual = (
        float(np.max(result.rms_residuals)) if result.rms_residuals.size else 0.0
    )

    return TemperatureDependentLegSolution(
        coordinate_m=coordinate,
        temperature_k=temperature,
        heat_flux_w_per_m2=heat_flux,
        potential_v=potential,
        temperature_gradient_k_per_m=temperature_gradient,
        electric_field_v_per_m=electric_field,
        seebeck_v_per_k=local_seebeck,
        electrical_resistivity_ohm_m=local_resistivity,
        thermal_conductivity_w_per_m_k=local_conductivity,
        thomson_coefficient_v_per_k=thomson,
        signed_current_a=signed_current,
        current_density_a_per_m2=current_density,
        cold_heat_rate_w=cold_heat,
        hot_heat_rate_w=hot_heat,
        hot_minus_cold_potential_v=potential_difference,
        local_energy_residual_w=local_energy_residual,
        relative_local_energy_residual=relative_local_residual,
        maximum_rms_bvp_residual=maximum_rms_bvp_residual,
        maximum_relative_conservative_residual=maximum_relative_conservative_residual,
        adaptive_mesh_nodes=int(result.x.size),
        nonlinear_iterations=int(result.niter),
    )


def solve_temperature_dependent_couple(
    couple: TemperatureDependentNumericalCouple,
    current_a: float,
    *,
    initial_mesh_points: int = 41,
    output_points: int = 401,
    relative_tolerance: float = 1.0e-7,
    max_nodes: int = 10000,
) -> TemperatureDependentOperatingPoint:
    """Solve both legs and assemble ``Qc``, ``Qh``, ``V``, and ``I V``.

    The local current convention is enforced internally: ``Jp=+I/Ap`` and
    ``Jn=-I/An``.  The reported signed energy residual is
    ``Qh - Qc - I*V``.
    """

    if not isinstance(couple, TemperatureDependentNumericalCouple):
        raise TypeError("couple must be a TemperatureDependentNumericalCouple")
    current = _finite("current_a", current_a)
    common_options = {
        "initial_mesh_points": initial_mesh_points,
        "output_points": output_points,
        "relative_tolerance": relative_tolerance,
        "max_nodes": max_nodes,
    }
    p_solution = solve_temperature_dependent_leg(
        couple.p_leg,
        +current,
        couple.cold_temperature_k,
        couple.hot_temperature_k,
        **common_options,
    )
    n_solution = solve_temperature_dependent_leg(
        couple.n_leg,
        -current,
        couple.cold_temperature_k,
        couple.hot_temperature_k,
        **common_options,
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
    return TemperatureDependentOperatingPoint(
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
