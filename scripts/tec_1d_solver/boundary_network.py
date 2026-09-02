"""Fixed-current reservoir/contact/bulk/parasitic thermoelectric network.

This module is the boundary-network layer around the conservative
temperature-dependent bulk solver.  The bulk-leg endpoint temperatures are
unknowns whenever a thermal contact resistance is nonzero.  The network solves
the cold- and hot-side contact-node balances together with the bulk problem and
then applies a passive two-reservoir heat leak.  Side names encode topology;
active operation is allowed to reverse the numerical ordering of the two leg
endpoint temperatures.

The sign convention is:

* ``Qc > 0`` is heat entering the device from the cold reservoir;
* ``Qh > 0`` is heat leaving the device into the hot reservoir;
* ``Pin = I * V`` and ``Qh - Qc - Pin = 0`` for this two-terminal model.

An aggregate series electrical-contact model is intentionally conditional on
an explicit Joule-heat partition ``eta``.  There is no 0.5 default.  A caller
with resolved contact-layer properties should model that layer explicitly
instead of using this reduction.

This module treats synthetic inputs as method checks only. It does not convert
them into PbSe/Cr material evidence and does not implement fixed-voltage or
fixed-power control.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from .temperature_dependent import (
    PropertyDomainError,
    SolverConvergenceError,
    TemperatureDependentLeg,
    TemperatureDependentNumericalCouple,
    TemperatureDependentOperatingPoint,
    solve_temperature_dependent_couple,
)


SYNTHETIC_DATA_ROLE = "synthetic_method_validation_only"
# A BVP can leave either the specified low- or high-temperature support.
# Do not mislabel every such failure as low-temperature extrapolation; the
# detailed exception remains in diagnostics for the batch layer to refine.
REJECTION_PROPERTY_DOMAIN = "NETWORK_PROPERTY_DOMAIN_EXCURSION"
REJECTION_NUMERICAL_INVALID = "LT_NUMERICAL_INVALID"


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


def _nonnegative_finite(name: str, value: object) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _positive_finite(name: str, value: object) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _integer_at_least(name: str, value: object, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class SeriesElectricalContacts:
    """Aggregate series electrical contacts with explicit Joule partition.

    ``joule_fraction_to_cold_node`` is the fraction of the total contact Joule
    power delivered to the cold inner contact node.  The hot-node fraction is
    exactly ``1 - eta``.  The field has no default by design.
    """

    resistance_ohm: float
    joule_fraction_to_cold_node: float

    def __post_init__(self) -> None:
        resistance = _nonnegative_finite("resistance_ohm", self.resistance_ohm)
        eta = _finite(
            "joule_fraction_to_cold_node", self.joule_fraction_to_cold_node
        )
        if not 0.0 <= eta <= 1.0:
            raise ValueError("joule_fraction_to_cold_node must lie in [0, 1]")
        object.__setattr__(self, "resistance_ohm", resistance)
        object.__setattr__(self, "joule_fraction_to_cold_node", eta)


@dataclass(frozen=True)
class ReservoirThermalContacts:
    """Cold and hot reservoir-to-leg thermal resistances in K/W.

    Zero is an ideal isothermal contact.  It is handled as an exact temperature
    constraint, never by division through a very small resistance.
    """

    cold_resistance_k_per_w: float
    hot_resistance_k_per_w: float

    def __post_init__(self) -> None:
        cold = _nonnegative_finite(
            "cold_resistance_k_per_w", self.cold_resistance_k_per_w
        )
        hot = _nonnegative_finite(
            "hot_resistance_k_per_w", self.hot_resistance_k_per_w
        )
        object.__setattr__(self, "cold_resistance_k_per_w", cold)
        object.__setattr__(self, "hot_resistance_k_per_w", hot)


@dataclass(frozen=True)
class TwoReservoirParasitic:
    """Linear passive heat leak directly from hot to cold reservoir."""

    thermal_conductance_w_per_k: float

    def __post_init__(self) -> None:
        conductance = _nonnegative_finite(
            "thermal_conductance_w_per_k", self.thermal_conductance_w_per_k
        )
        object.__setattr__(self, "thermal_conductance_w_per_k", conductance)


@dataclass(frozen=True)
class FixedCurrentBoundaryNetwork:
    """Physical inputs for the current-controlled two-reservoir network."""

    p_leg: TemperatureDependentLeg
    n_leg: TemperatureDependentLeg
    cold_reservoir_temperature_k: float
    hot_reservoir_temperature_k: float
    electrical_contacts: SeriesElectricalContacts
    thermal_contacts: ReservoirThermalContacts
    parasitic: TwoReservoirParasitic
    energy_scale_w: float
    data_role: str

    def __post_init__(self) -> None:
        if not isinstance(self.p_leg, TemperatureDependentLeg):
            raise TypeError("p_leg must be a TemperatureDependentLeg")
        if not isinstance(self.n_leg, TemperatureDependentLeg):
            raise TypeError("n_leg must be a TemperatureDependentLeg")
        if not isinstance(self.electrical_contacts, SeriesElectricalContacts):
            raise TypeError("electrical_contacts must be SeriesElectricalContacts")
        if not isinstance(self.thermal_contacts, ReservoirThermalContacts):
            raise TypeError("thermal_contacts must be ReservoirThermalContacts")
        if not isinstance(self.parasitic, TwoReservoirParasitic):
            raise TypeError("parasitic must be TwoReservoirParasitic")

        cold = _positive_finite(
            "cold_reservoir_temperature_k", self.cold_reservoir_temperature_k
        )
        hot = _positive_finite(
            "hot_reservoir_temperature_k", self.hot_reservoir_temperature_k
        )
        if hot < cold:
            raise ValueError(
                "hot_reservoir_temperature_k must be >= "
                "cold_reservoir_temperature_k"
            )
        energy_scale = _positive_finite("energy_scale_w", self.energy_scale_w)
        if self.data_role != SYNTHETIC_DATA_ROLE:
            raise ValueError(
                f"data_role must be {SYNTHETIC_DATA_ROLE!r} at the current criterion"
            )

        domain_minimum = self.minimum_valid_leg_temperature_k
        domain_maximum = self.maximum_valid_leg_temperature_k
        if domain_maximum <= domain_minimum:
            raise PropertyDomainError(
                "the p/n bulk-leg property temperature domains do not overlap"
            )
        if cold < domain_minimum or hot > domain_maximum:
            raise PropertyDomainError(
                f"reservoir temperatures [{cold:g}, {hot:g}] K are outside the "
                "shared bulk-leg property domain "
                f"[{domain_minimum:g}, {domain_maximum:g}] K"
            )

        object.__setattr__(self, "cold_reservoir_temperature_k", cold)
        object.__setattr__(self, "hot_reservoir_temperature_k", hot)
        object.__setattr__(self, "energy_scale_w", energy_scale)

    @property
    def minimum_valid_leg_temperature_k(self) -> float:
        return max(
            self.p_leg.minimum_valid_temperature_k,
            self.n_leg.minimum_valid_temperature_k,
        )

    @property
    def maximum_valid_leg_temperature_k(self) -> float:
        return min(
            self.p_leg.maximum_valid_temperature_k,
            self.n_leg.maximum_valid_temperature_k,
        )


@dataclass(frozen=True)
class BoundaryNetworkSolverOptions:
    """Nonlinear, bulk-solver, and numerical validation settings."""

    nonlinear_tolerance: float = 1.0e-10
    max_function_evaluations: int = 100
    temperature_residual_tolerance_k: float = 1.0e-7
    node_energy_residual_tolerance_w: float = 1.0e-8
    global_energy_residual_fraction_tolerance: float = 1.0e-8
    bulk_initial_mesh_points: int = 31
    bulk_output_points: int = 201
    bulk_relative_tolerance: float = 1.0e-8
    bulk_max_nodes: int = 10000

    def __post_init__(self) -> None:
        nonlinear = _positive_finite(
            "nonlinear_tolerance", self.nonlinear_tolerance
        )
        # scipy rejects tolerances below machine precision by disabling every
        # termination condition; keep the reported result deterministic.
        if nonlinear < 1.0e-14:
            raise ValueError("nonlinear_tolerance must be >= 1e-14")
        object.__setattr__(self, "nonlinear_tolerance", nonlinear)
        object.__setattr__(
            self,
            "max_function_evaluations",
            _integer_at_least(
                "max_function_evaluations", self.max_function_evaluations, 1
            ),
        )
        for name in (
            "temperature_residual_tolerance_k",
            "node_energy_residual_tolerance_w",
            "global_energy_residual_fraction_tolerance",
            "bulk_relative_tolerance",
        ):
            object.__setattr__(self, name, _positive_finite(name, getattr(self, name)))
        initial = _integer_at_least(
            "bulk_initial_mesh_points", self.bulk_initial_mesh_points, 3
        )
        output = _integer_at_least("bulk_output_points", self.bulk_output_points, 3)
        maximum = _integer_at_least("bulk_max_nodes", self.bulk_max_nodes, initial)
        object.__setattr__(self, "bulk_initial_mesh_points", initial)
        object.__setattr__(self, "bulk_output_points", output)
        object.__setattr__(self, "bulk_max_nodes", maximum)


@dataclass(frozen=True)
class BoundaryNetworkConvergenceDiagnostics:
    """Nonlinear status plus independent contact-node residuals."""

    method: str
    converged: bool
    status: int | None
    message: str
    function_evaluations: int
    bulk_solve_calls: int
    cold_temperature_law_residual_k: float | None
    hot_temperature_law_residual_k: float | None
    cold_node_energy_residual_w: float | None
    hot_node_energy_residual_w: float | None
    maximum_absolute_temperature_residual_k: float | None
    maximum_absolute_node_energy_residual_w: float | None


@dataclass(frozen=True)
class BoundaryNetworkOperatingPoint:
    """Accepted fixed-current operating point and complete energy ledger."""

    data_role: str
    current_a: float
    cold_reservoir_temperature_k: float
    hot_reservoir_temperature_k: float
    cold_leg_temperature_k: float
    hot_leg_temperature_k: float
    voltage_bulk_v: float
    voltage_contacts_v: float
    voltage_terminal_v: float
    input_power_w: float
    contact_joule_power_w: float
    contact_joule_to_cold_node_w: float
    contact_joule_to_hot_node_w: float
    contact_joule_partition_residual_w: float
    qc_bulk_w: float
    qh_bulk_w: float
    qc_te_w: float
    qh_te_w: float
    q_parasitic_w: float
    qc_net_w: float
    qh_net_w: float
    q_ambient_out_w: float
    cop: float | None
    energy_residual_w: float
    energy_residual_fraction: float
    bulk_point: TemperatureDependentOperatingPoint

    # Protocol-name aliases keep serialization adapters and future operating
    # controls unambiguous without duplicating stored state.
    @property
    def evidence_role(self) -> str:
        return self.data_role

    @property
    def tc_reservoir_k(self) -> float:
        return self.cold_reservoir_temperature_k

    @property
    def th_reservoir_k(self) -> float:
        return self.hot_reservoir_temperature_k

    @property
    def tc_leg_k(self) -> float:
        return self.cold_leg_temperature_k

    @property
    def th_leg_k(self) -> float:
        return self.hot_leg_temperature_k

    @property
    def voltage_leg_v(self) -> float:
        return self.voltage_bulk_v

    @property
    def pin_w(self) -> float:
        return self.input_power_w

    @property
    def Qc_w(self) -> float:
        return self.qc_net_w

    @property
    def Qh_w(self) -> float:
        return self.qh_net_w

    @property
    def V_v(self) -> float:
        return self.voltage_terminal_v

    @property
    def Pin_w(self) -> float:
        return self.input_power_w

    @property
    def COP(self) -> float | None:
        return self.cop

    @property
    def relative_energy_residual(self) -> float:
        return self.energy_residual_fraction


@dataclass(frozen=True)
class BoundaryNetworkSolveReport:
    """Accepted point or a fail-closed rejection with traceable diagnostics."""

    accepted: bool
    rejection_codes: tuple[str, ...]
    data_role: str
    point: BoundaryNetworkOperatingPoint | None
    diagnostics: BoundaryNetworkConvergenceDiagnostics

    def __post_init__(self) -> None:
        if self.accepted:
            if self.rejection_codes or self.point is None:
                raise ValueError(
                    "accepted report must have a point and no rejection codes"
                )
        elif not self.rejection_codes or self.point is not None:
            raise ValueError(
                "rejected report must have rejection codes and no operating point"
            )

    def require_point(self) -> BoundaryNetworkOperatingPoint:
        """Return the accepted point or raise with the preserved reason codes."""

        if self.point is None:
            codes = ", ".join(self.rejection_codes)
            raise RuntimeError(f"boundary-network solve was rejected: {codes}")
        return self.point


@dataclass(frozen=True)
class _NodeEvaluation:
    bulk_point: TemperatureDependentOperatingPoint
    cold_temperature_law_residual_k: float
    hot_temperature_law_residual_k: float
    cold_node_energy_residual_w: float
    hot_node_energy_residual_w: float


def _empty_diagnostics(
    *,
    method: str,
    message: str,
    bulk_solve_calls: int,
) -> BoundaryNetworkConvergenceDiagnostics:
    return BoundaryNetworkConvergenceDiagnostics(
        method=method,
        converged=False,
        status=None,
        message=message,
        function_evaluations=bulk_solve_calls,
        bulk_solve_calls=bulk_solve_calls,
        cold_temperature_law_residual_k=None,
        hot_temperature_law_residual_k=None,
        cold_node_energy_residual_w=None,
        hot_node_energy_residual_w=None,
        maximum_absolute_temperature_residual_k=None,
        maximum_absolute_node_energy_residual_w=None,
    )


def solve_fixed_current_boundary_network(
    network: FixedCurrentBoundaryNetwork,
    current_a: float,
    *,
    options: BoundaryNetworkSolverOptions | None = None,
) -> BoundaryNetworkSolveReport:
    """Solve the two thermal contact nodes at a prescribed signed current.

    A property-domain excursion or nonlinear/numerical failure is returned as a
    rejected report.  Invalid static inputs are rejected earlier by their
    dataclass constructors.  No failed sample is silently dropped.
    """

    if not isinstance(network, FixedCurrentBoundaryNetwork):
        raise TypeError("network must be a FixedCurrentBoundaryNetwork")
    current = _finite("current_a", current_a)
    if options is None:
        controls = BoundaryNetworkSolverOptions()
    elif isinstance(options, BoundaryNetworkSolverOptions):
        controls = options
    else:
        raise TypeError("options must be a BoundaryNetworkSolverOptions or None")

    electrical = network.electrical_contacts
    thermal = network.thermal_contacts
    contact_joule_power = current * current * electrical.resistance_ohm
    heat_to_cold = (
        electrical.joule_fraction_to_cold_node * contact_joule_power
    )
    heat_to_hot = (
        (1.0 - electrical.joule_fraction_to_cold_node) * contact_joule_power
    )

    bulk_options = {
        "initial_mesh_points": controls.bulk_initial_mesh_points,
        "output_points": controls.bulk_output_points,
        "relative_tolerance": controls.bulk_relative_tolerance,
        "max_nodes": controls.bulk_max_nodes,
    }
    bulk_solve_calls = 0

    def evaluate_nodes(cold_leg_k: float, hot_leg_k: float) -> _NodeEvaluation:
        nonlocal bulk_solve_calls
        bulk_solve_calls += 1
        couple = TemperatureDependentNumericalCouple(
            p_leg=network.p_leg,
            n_leg=network.n_leg,
            cold_temperature_k=cold_leg_k,
            hot_temperature_k=hot_leg_k,
        )
        bulk = solve_temperature_dependent_couple(
            couple,
            current,
            **bulk_options,
        )
        qc_te = bulk.Qc_w - heat_to_cold
        qh_te = bulk.Qh_w + heat_to_hot
        cold_temperature_residual = (
            cold_leg_k
            - network.cold_reservoir_temperature_k
            + thermal.cold_resistance_k_per_w * qc_te
        )
        hot_temperature_residual = (
            hot_leg_k
            - network.hot_reservoir_temperature_k
            - thermal.hot_resistance_k_per_w * qh_te
        )

        # At an ideal (Rth=0) contact, reservoir heat is the reaction flux from
        # node balance; a Fourier quotient is undefined and is not fabricated.
        if thermal.cold_resistance_k_per_w == 0.0:
            cold_node_residual = 0.0
        else:
            cold_node_residual = (
                (network.cold_reservoir_temperature_k - cold_leg_k)
                / thermal.cold_resistance_k_per_w
                + heat_to_cold
                - bulk.Qc_w
            )
        if thermal.hot_resistance_k_per_w == 0.0:
            hot_node_residual = 0.0
        else:
            hot_node_residual = (
                (hot_leg_k - network.hot_reservoir_temperature_k)
                / thermal.hot_resistance_k_per_w
                - bulk.Qh_w
                - heat_to_hot
            )
        return _NodeEvaluation(
            bulk_point=bulk,
            cold_temperature_law_residual_k=cold_temperature_residual,
            hot_temperature_law_residual_k=hot_temperature_residual,
            cold_node_energy_residual_w=cold_node_residual,
            hot_node_energy_residual_w=hot_node_residual,
        )

    domain_minimum = network.minimum_valid_leg_temperature_k
    domain_maximum = network.maximum_valid_leg_temperature_k
    domain_span = domain_maximum - domain_minimum

    cold_is_ideal = thermal.cold_resistance_k_per_w == 0.0
    hot_is_ideal = thermal.hot_resistance_k_per_w == 0.0
    if cold_is_ideal and not hot_is_ideal:
        # Eliminate the ideal-contact temperature exactly.  Keeping it as a
        # least-squares variable would let relaxed tolerances accept a finite
        # temperature jump across Rth=0.
        method = "bounded_least_squares_hot_temperature_cold_exact"
        fixed_cold = network.cold_reservoir_temperature_k
        free_span = domain_span

        def decode_temperatures(unit_variables: np.ndarray) -> tuple[float, float]:
            hot_leg = domain_minimum + float(unit_variables[0]) * free_span
            return fixed_cold, hot_leg

        initial_variables = np.asarray(
            [
                0.0
                if free_span == 0.0
                else (
                    network.hot_reservoir_temperature_k - domain_minimum
                )
                / free_span
            ],
            dtype=float,
        )
    elif hot_is_ideal and not cold_is_ideal:
        method = "bounded_least_squares_cold_temperature_hot_exact"
        fixed_hot = network.hot_reservoir_temperature_k
        free_span = domain_span

        def decode_temperatures(unit_variables: np.ndarray) -> tuple[float, float]:
            cold_leg = domain_minimum + float(unit_variables[0]) * free_span
            return cold_leg, fixed_hot

        initial_variables = np.asarray(
            [
                0.0
                if free_span == 0.0
                else (
                    network.cold_reservoir_temperature_k - domain_minimum
                )
                / free_span
            ],
            dtype=float,
        )
    else:
        method = "bounded_least_squares_independent_terminal_temperatures"

        def decode_temperatures(unit_variables: np.ndarray) -> tuple[float, float]:
            # Terminal names identify topology rather than numerical temperature
            # order.  Bound each endpoint independently to the shared property
            # domain so reverse-current operation can produce Tc_leg > Th_leg.
            cold_leg = domain_minimum + float(unit_variables[0]) * domain_span
            hot_leg = domain_minimum + float(unit_variables[1]) * domain_span
            return cold_leg, hot_leg

        initial_variables = np.asarray(
            [
                (
                    network.cold_reservoir_temperature_k - domain_minimum
                )
                / domain_span,
                (
                    network.hot_reservoir_temperature_k - domain_minimum
                )
                / domain_span,
            ],
            dtype=float,
        )
    initial_variables = np.clip(initial_variables, 0.0, 1.0)

    try:
        if cold_is_ideal and hot_is_ideal:
            method = "exact_zero_thermal_resistance_clamp"
            cold_leg_temperature = network.cold_reservoir_temperature_k
            hot_leg_temperature = network.hot_reservoir_temperature_k
            final_evaluation = evaluate_nodes(
                cold_leg_temperature, hot_leg_temperature
            )
            scipy_success = True
            scipy_status = 0
            scipy_message = "both ideal thermal contacts imposed exactly"
            scipy_nfev = 1
        else:

            def normalized_residual(unit_variables: np.ndarray) -> np.ndarray:
                cold_leg, hot_leg = decode_temperatures(unit_variables)
                evaluated = evaluate_nodes(cold_leg, hot_leg)
                residuals: list[float] = []
                if not cold_is_ideal:
                    residuals.append(
                        evaluated.cold_node_energy_residual_w
                        / network.energy_scale_w
                    )
                if not hot_is_ideal:
                    residuals.append(
                        evaluated.hot_node_energy_residual_w
                        / network.energy_scale_w
                    )
                return np.asarray(residuals, dtype=float)

            scipy_result = least_squares(
                normalized_residual,
                initial_variables,
                bounds=(
                    np.zeros(initial_variables.size, dtype=float),
                    np.ones(initial_variables.size, dtype=float),
                ),
                xtol=controls.nonlinear_tolerance,
                ftol=controls.nonlinear_tolerance,
                gtol=controls.nonlinear_tolerance,
                max_nfev=controls.max_function_evaluations,
                x_scale="jac",
                verbose=0,
            )
            cold_leg_temperature, hot_leg_temperature = decode_temperatures(
                np.asarray(scipy_result.x, dtype=float)
            )
            final_evaluation = evaluate_nodes(
                cold_leg_temperature, hot_leg_temperature
            )
            scipy_success = bool(scipy_result.success)
            scipy_status = int(scipy_result.status)
            scipy_message = str(scipy_result.message)
            scipy_nfev = int(scipy_result.nfev)
    except PropertyDomainError as exc:
        diagnostics = _empty_diagnostics(
            method=method,
            message=f"property-domain rejection: {exc}",
            bulk_solve_calls=bulk_solve_calls,
        )
        return BoundaryNetworkSolveReport(
            accepted=False,
            rejection_codes=(REJECTION_PROPERTY_DOMAIN,),
            data_role=network.data_role,
            point=None,
            diagnostics=diagnostics,
        )
    except SolverConvergenceError as exc:
        diagnostics = _empty_diagnostics(
            method=method,
            message=f"bulk-solver rejection: {exc}",
            bulk_solve_calls=bulk_solve_calls,
        )
        return BoundaryNetworkSolveReport(
            accepted=False,
            rejection_codes=(REJECTION_NUMERICAL_INVALID,),
            data_role=network.data_role,
            point=None,
            diagnostics=diagnostics,
        )

    temperature_residuals = (
        abs(final_evaluation.cold_temperature_law_residual_k),
        abs(final_evaluation.hot_temperature_law_residual_k),
    )
    node_residuals = (
        abs(final_evaluation.cold_node_energy_residual_w),
        abs(final_evaluation.hot_node_energy_residual_w),
    )
    maximum_temperature_residual = max(temperature_residuals)
    maximum_node_residual = max(node_residuals)
    contact_constraints_pass = (
        maximum_temperature_residual
        <= controls.temperature_residual_tolerance_k
        and maximum_node_residual <= controls.node_energy_residual_tolerance_w
    )
    nonlinear_converged = scipy_success and contact_constraints_pass
    diagnostics = BoundaryNetworkConvergenceDiagnostics(
        method=method,
        converged=nonlinear_converged,
        status=scipy_status,
        message=scipy_message,
        function_evaluations=scipy_nfev,
        bulk_solve_calls=bulk_solve_calls,
        cold_temperature_law_residual_k=(
            final_evaluation.cold_temperature_law_residual_k
        ),
        hot_temperature_law_residual_k=(
            final_evaluation.hot_temperature_law_residual_k
        ),
        cold_node_energy_residual_w=final_evaluation.cold_node_energy_residual_w,
        hot_node_energy_residual_w=final_evaluation.hot_node_energy_residual_w,
        maximum_absolute_temperature_residual_k=maximum_temperature_residual,
        maximum_absolute_node_energy_residual_w=maximum_node_residual,
    )
    if not nonlinear_converged:
        return BoundaryNetworkSolveReport(
            accepted=False,
            rejection_codes=(REJECTION_NUMERICAL_INVALID,),
            data_role=network.data_role,
            point=None,
            diagnostics=diagnostics,
        )

    bulk = final_evaluation.bulk_point
    voltage_contacts = current * electrical.resistance_ohm
    voltage_terminal = bulk.V_v + voltage_contacts
    input_power = current * voltage_terminal
    contact_partition_residual = (
        heat_to_cold + heat_to_hot - contact_joule_power
    )
    qc_te = bulk.Qc_w - heat_to_cold
    qh_te = bulk.Qh_w + heat_to_hot
    reservoir_delta_temperature = (
        network.hot_reservoir_temperature_k
        - network.cold_reservoir_temperature_k
    )
    q_parasitic = (
        network.parasitic.thermal_conductance_w_per_k
        * reservoir_delta_temperature
    )
    qc_net = qc_te - q_parasitic
    qh_net = qh_te - q_parasitic
    energy_residual = qh_net - qc_net - input_power
    energy_denominator = max(
        abs(qh_net),
        abs(qc_net),
        abs(input_power),
        network.energy_scale_w,
    )
    energy_fraction = abs(energy_residual) / energy_denominator
    cooling_cop = (
        qc_net / input_power
        if qc_net > 0.0 and input_power > 0.0
        else None
    )
    partition_tolerance = 1.0e-10 * max(1.0, abs(contact_joule_power))
    ledger_pass = (
        energy_fraction
        <= controls.global_energy_residual_fraction_tolerance
        and abs(contact_partition_residual) <= partition_tolerance
    )
    if not ledger_pass:
        failed_diagnostics = BoundaryNetworkConvergenceDiagnostics(
            method=diagnostics.method,
            converged=False,
            status=diagnostics.status,
            message=(
                f"{diagnostics.message}; global/contact energy ledger failed "
                f"(fraction={energy_fraction:.6g}, contact partition residual="
                f"{contact_partition_residual:.6g} W)"
            ),
            function_evaluations=diagnostics.function_evaluations,
            bulk_solve_calls=diagnostics.bulk_solve_calls,
            cold_temperature_law_residual_k=(
                diagnostics.cold_temperature_law_residual_k
            ),
            hot_temperature_law_residual_k=(
                diagnostics.hot_temperature_law_residual_k
            ),
            cold_node_energy_residual_w=diagnostics.cold_node_energy_residual_w,
            hot_node_energy_residual_w=diagnostics.hot_node_energy_residual_w,
            maximum_absolute_temperature_residual_k=(
                diagnostics.maximum_absolute_temperature_residual_k
            ),
            maximum_absolute_node_energy_residual_w=(
                diagnostics.maximum_absolute_node_energy_residual_w
            ),
        )
        return BoundaryNetworkSolveReport(
            accepted=False,
            rejection_codes=(REJECTION_NUMERICAL_INVALID,),
            data_role=network.data_role,
            point=None,
            diagnostics=failed_diagnostics,
        )

    point = BoundaryNetworkOperatingPoint(
        data_role=network.data_role,
        current_a=current,
        cold_reservoir_temperature_k=network.cold_reservoir_temperature_k,
        hot_reservoir_temperature_k=network.hot_reservoir_temperature_k,
        cold_leg_temperature_k=cold_leg_temperature,
        hot_leg_temperature_k=hot_leg_temperature,
        voltage_bulk_v=bulk.V_v,
        voltage_contacts_v=voltage_contacts,
        voltage_terminal_v=voltage_terminal,
        input_power_w=input_power,
        contact_joule_power_w=contact_joule_power,
        contact_joule_to_cold_node_w=heat_to_cold,
        contact_joule_to_hot_node_w=heat_to_hot,
        contact_joule_partition_residual_w=contact_partition_residual,
        qc_bulk_w=bulk.Qc_w,
        qh_bulk_w=bulk.Qh_w,
        qc_te_w=qc_te,
        qh_te_w=qh_te,
        q_parasitic_w=q_parasitic,
        qc_net_w=qc_net,
        qh_net_w=qh_net,
        q_ambient_out_w=0.0,
        cop=cooling_cop,
        energy_residual_w=energy_residual,
        energy_residual_fraction=energy_fraction,
        bulk_point=bulk,
    )
    return BoundaryNetworkSolveReport(
        accepted=True,
        rejection_codes=(),
        data_role=network.data_role,
        point=point,
        diagnostics=diagnostics,
    )


__all__ = [
    "SYNTHETIC_DATA_ROLE",
    "REJECTION_PROPERTY_DOMAIN",
    "REJECTION_NUMERICAL_INVALID",
    "SeriesElectricalContacts",
    "ReservoirThermalContacts",
    "TwoReservoirParasitic",
    "FixedCurrentBoundaryNetwork",
    "BoundaryNetworkSolverOptions",
    "BoundaryNetworkConvergenceDiagnostics",
    "BoundaryNetworkOperatingPoint",
    "BoundaryNetworkSolveReport",
    "solve_fixed_current_boundary_network",
]
