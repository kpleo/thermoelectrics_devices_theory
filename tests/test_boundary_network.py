"""Energy-ledger and rejection tests for the fixed-current boundary network."""

from __future__ import annotations

import math
import unittest

import numpy as np

from scripts.tec_1d_solver.boundary_network import (
    REJECTION_NUMERICAL_INVALID,
    REJECTION_PROPERTY_DOMAIN,
    SYNTHETIC_DATA_ROLE,
    BoundaryNetworkSolverOptions,
    FixedCurrentBoundaryNetwork,
    ReservoirThermalContacts,
    SeriesElectricalContacts,
    TwoReservoirParasitic,
    solve_fixed_current_boundary_network,
)
from scripts.tec_1d_solver.temperature_dependent import (
    ConstantTemperatureProperty,
    PropertyDomainError,
    TemperatureDependentLeg,
    TemperatureDependentNumericalCouple,
    solve_temperature_dependent_couple,
)


def synthetic_network(
    *,
    domain: tuple[float, float] = (250.0, 360.0),
    cold_reservoir_k: float = 290.0,
    hot_reservoir_k: float = 320.0,
    electrical_resistance_ohm: float = 0.0,
    eta_to_cold: float = 0.3,
    cold_thermal_resistance_k_per_w: float = 0.0,
    hot_thermal_resistance_k_per_w: float = 0.0,
    parasitic_conductance_w_per_k: float = 0.0,
    energy_scale_w: float = 1.0e-3,
    data_role: str = SYNTHETIC_DATA_ROLE,
) -> FixedCurrentBoundaryNetwork:
    def constant(value: float) -> ConstantTemperatureProperty:
        return ConstantTemperatureProperty(value, *domain)

    p_leg = TemperatureDependentLeg(
        seebeck=constant(220.0e-6),
        electrical_resistivity=constant(1.3e-5),
        thermal_conductivity=constant(1.7),
        length_m=1.5e-3,
        area_m2=1.2e-6,
    )
    n_leg = TemperatureDependentLeg(
        seebeck=constant(-135.0e-6),
        electrical_resistivity=constant(2.2e-5),
        thermal_conductivity=constant(0.9),
        length_m=0.8e-3,
        area_m2=0.7e-6,
    )
    return FixedCurrentBoundaryNetwork(
        p_leg=p_leg,
        n_leg=n_leg,
        cold_reservoir_temperature_k=cold_reservoir_k,
        hot_reservoir_temperature_k=hot_reservoir_k,
        electrical_contacts=SeriesElectricalContacts(
            resistance_ohm=electrical_resistance_ohm,
            joule_fraction_to_cold_node=eta_to_cold,
        ),
        thermal_contacts=ReservoirThermalContacts(
            cold_resistance_k_per_w=cold_thermal_resistance_k_per_w,
            hot_resistance_k_per_w=hot_thermal_resistance_k_per_w,
        ),
        parasitic=TwoReservoirParasitic(
            thermal_conductance_w_per_k=parasitic_conductance_w_per_k
        ),
        energy_scale_w=energy_scale_w,
        data_role=data_role,
    )


class BoundaryNetworkTests(unittest.TestCase):
    def assert_energy_closed(self, point: object) -> None:
        self.assertAlmostEqual(
            point.qh_net_w - point.qc_net_w,  # type: ignore[attr-defined]
            point.input_power_w,  # type: ignore[attr-defined]
            delta=2.0e-11,
        )
        self.assertLess(
            point.energy_residual_fraction, 1.0e-9  # type: ignore[attr-defined]
        )

    def test_zero_contact_and_heat_leak_limit_returns_bulk_solver(self) -> None:
        network = synthetic_network(eta_to_cold=0.173)
        direct_couple = TemperatureDependentNumericalCouple(
            p_leg=network.p_leg,
            n_leg=network.n_leg,
            cold_temperature_k=network.cold_reservoir_temperature_k,
            hot_temperature_k=network.hot_reservoir_temperature_k,
        )
        for current in (-0.8, 0.0, 1.0):
            with self.subTest(current=current):
                direct = solve_temperature_dependent_couple(direct_couple, current)
                report = solve_fixed_current_boundary_network(network, current)
                self.assertTrue(report.accepted)
                self.assertEqual(report.rejection_codes, ())
                point = report.require_point()
                self.assertEqual(
                    report.diagnostics.method,
                    "exact_zero_thermal_resistance_clamp",
                )
                self.assertEqual(
                    point.cold_leg_temperature_k,
                    network.cold_reservoir_temperature_k,
                )
                self.assertEqual(
                    point.hot_leg_temperature_k,
                    network.hot_reservoir_temperature_k,
                )
                for network_field, bulk_field in (
                    ("qc_net_w", "Qc_w"),
                    ("qh_net_w", "Qh_w"),
                    ("voltage_terminal_v", "V_v"),
                    ("input_power_w", "Pin_w"),
                ):
                    self.assertAlmostEqual(
                        getattr(point, network_field),
                        getattr(direct, bulk_field),
                        delta=2.0e-11,
                    )
                self.assertEqual(point.q_parasitic_w, 0.0)
                self.assertEqual(point.voltage_contacts_v, 0.0)
                self.assertEqual(point.contact_joule_power_w, 0.0)
                self.assertEqual(point.evidence_role, SYNTHETIC_DATA_ROLE)
                self.assertEqual(point.tc_reservoir_k, point.cold_reservoir_temperature_k)
                self.assertEqual(point.th_reservoir_k, point.hot_reservoir_temperature_k)
                self.assertEqual(point.tc_leg_k, point.cold_leg_temperature_k)
                self.assertEqual(point.th_leg_k, point.hot_leg_temperature_k)
                self.assertEqual(point.voltage_leg_v, point.voltage_bulk_v)
                self.assertEqual(point.pin_w, point.input_power_w)
                self.assertEqual(point.Qc_w, point.qc_net_w)
                self.assertEqual(point.Qh_w, point.qh_net_w)
                self.assertEqual(point.V_v, point.voltage_terminal_v)
                self.assertEqual(point.Pin_w, point.input_power_w)
                self.assertEqual(point.COP, point.cop)
                self.assert_energy_closed(point)

    def test_eta_is_required_and_endpoint_partitions_close_for_both_signs(self) -> None:
        with self.assertRaises(TypeError):
            SeriesElectricalContacts(resistance_ohm=0.02)  # type: ignore[call-arg]

        resistance = 0.02
        for current in (-0.8, 1.0):
            for eta in (0.0, 1.0):
                with self.subTest(current=current, eta=eta):
                    report = solve_fixed_current_boundary_network(
                        synthetic_network(
                            electrical_resistance_ohm=resistance,
                            eta_to_cold=eta,
                        ),
                        current,
                    )
                    point = report.require_point()
                    expected_joule = current * current * resistance
                    self.assertAlmostEqual(
                        point.voltage_contacts_v, current * resistance, places=15
                    )
                    self.assertAlmostEqual(
                        point.contact_joule_power_w, expected_joule, places=15
                    )
                    self.assertAlmostEqual(
                        point.contact_joule_to_cold_node_w,
                        eta * expected_joule,
                        places=15,
                    )
                    self.assertAlmostEqual(
                        point.contact_joule_to_hot_node_w,
                        (1.0 - eta) * expected_joule,
                        places=15,
                    )
                    self.assertAlmostEqual(
                        point.qc_te_w,
                        point.qc_bulk_w - eta * expected_joule,
                        places=15,
                    )
                    self.assertAlmostEqual(
                        point.qh_te_w,
                        point.qh_bulk_w + (1.0 - eta) * expected_joule,
                        places=15,
                    )
                    self.assertLess(
                        abs(point.contact_joule_partition_residual_w), 1.0e-15
                    )
                    self.assert_energy_closed(point)

    def test_finite_thermal_contacts_are_solved_as_unknown_nodes(self) -> None:
        network = synthetic_network(
            electrical_resistance_ohm=0.01,
            eta_to_cold=0.2,
            cold_thermal_resistance_k_per_w=5.0,
            hot_thermal_resistance_k_per_w=7.0,
        )
        for current in (-0.8, 0.0, 1.0):
            with self.subTest(current=current):
                report = solve_fixed_current_boundary_network(network, current)
                self.assertTrue(report.accepted, report.diagnostics.message)
                self.assertTrue(report.diagnostics.converged)
                point = report.require_point()
                self.assertLess(
                    report.diagnostics.maximum_absolute_temperature_residual_k,
                    1.0e-8,
                )
                self.assertLess(
                    report.diagnostics.maximum_absolute_node_energy_residual_w,
                    1.0e-10,
                )
                self.assertAlmostEqual(
                    network.cold_reservoir_temperature_k
                    - point.cold_leg_temperature_k,
                    network.thermal_contacts.cold_resistance_k_per_w
                    * point.qc_te_w,
                    delta=1.0e-10,
                )
                self.assertAlmostEqual(
                    point.hot_leg_temperature_k
                    - network.hot_reservoir_temperature_k,
                    network.thermal_contacts.hot_resistance_k_per_w
                    * point.qh_te_w,
                    delta=1.0e-10,
                )
                self.assert_energy_closed(point)

        zero_current = solve_fixed_current_boundary_network(network, 0.0).require_point()
        # Passive conduction warms the nominal cold leg face and cools the hot
        # leg face relative to their reservoirs.
        self.assertGreater(
            zero_current.cold_leg_temperature_k,
            network.cold_reservoir_temperature_k,
        )
        self.assertLess(
            zero_current.hot_leg_temperature_k,
            network.hot_reservoir_temperature_k,
        )

    def test_finite_contacts_match_independent_constant_property_linear_system(self) -> None:
        alpha = 220.0e-6 - (-135.0e-6)
        bulk_resistance = (
            1.3e-5 * 1.5e-3 / 1.2e-6
            + 2.2e-5 * 0.8e-3 / 0.7e-6
        )
        bulk_conductance = (
            1.7 * 1.2e-6 / 1.5e-3
            + 0.9 * 0.7e-6 / 0.8e-3
        )
        contact_resistance = 0.02
        cold_rth = 5.0
        hot_rth = 7.0
        for current in (-0.8, 0.0, 1.0):
            for eta in (0.0, 1.0):
                with self.subTest(current=current, eta=eta):
                    network = synthetic_network(
                        electrical_resistance_ohm=contact_resistance,
                        eta_to_cold=eta,
                        cold_thermal_resistance_k_per_w=cold_rth,
                        hot_thermal_resistance_k_per_w=hot_rth,
                    )
                    point = solve_fixed_current_boundary_network(
                        network, current
                    ).require_point()
                    joule_bulk_half = 0.5 * current**2 * bulk_resistance
                    joule_contact = current**2 * contact_resistance
                    heat_cold = eta * joule_contact
                    heat_hot = (1.0 - eta) * joule_contact
                    coefficient_matrix = np.asarray(
                        [
                            [
                                1.0 + cold_rth * (alpha * current + bulk_conductance),
                                -cold_rth * bulk_conductance,
                            ],
                            [
                                -hot_rth * bulk_conductance,
                                1.0
                                - hot_rth * (alpha * current - bulk_conductance),
                            ],
                        ],
                        dtype=float,
                    )
                    right_hand_side = np.asarray(
                        [
                            network.cold_reservoir_temperature_k
                            + cold_rth * (joule_bulk_half + heat_cold),
                            network.hot_reservoir_temperature_k
                            + hot_rth * (joule_bulk_half + heat_hot),
                        ],
                        dtype=float,
                    )
                    exact_cold, exact_hot = np.linalg.solve(
                        coefficient_matrix, right_hand_side
                    )
                    self.assertAlmostEqual(
                        point.cold_leg_temperature_k,
                        float(exact_cold),
                        delta=2.0e-9,
                    )
                    self.assertAlmostEqual(
                        point.hot_leg_temperature_k,
                        float(exact_hot),
                        delta=2.0e-9,
                    )
                    exact_qc_bulk = (
                        alpha * exact_cold * current
                        - joule_bulk_half
                        - bulk_conductance * (exact_hot - exact_cold)
                    )
                    exact_qh_bulk = (
                        alpha * exact_hot * current
                        + joule_bulk_half
                        - bulk_conductance * (exact_hot - exact_cold)
                    )
                    self.assertAlmostEqual(
                        point.qc_bulk_w, exact_qc_bulk, delta=2.0e-11
                    )
                    self.assertAlmostEqual(
                        point.qh_bulk_w, exact_qh_bulk, delta=2.0e-11
                    )
                    self.assert_energy_closed(point)

    def test_reverse_current_allows_terminal_temperature_inversion(self) -> None:
        alpha = 220.0e-6 - (-135.0e-6)
        bulk_resistance = (
            1.3e-5 * 1.5e-3 / 1.2e-6
            + 2.2e-5 * 0.8e-3 / 0.7e-6
        )
        bulk_conductance = (
            1.7 * 1.2e-6 / 1.5e-3
            + 0.9 * 0.7e-6 / 0.8e-3
        )
        current = -1.0
        contact_resistance = 0.2 * bulk_resistance
        eta = 0.37
        finite_thermal_resistance = 0.1 / bulk_conductance

        for cold_rth, hot_rth in (
            (finite_thermal_resistance, finite_thermal_resistance),
            (0.0, finite_thermal_resistance),
            (finite_thermal_resistance, 0.0),
        ):
            with self.subTest(cold_rth=cold_rth, hot_rth=hot_rth):
                network = synthetic_network(
                    cold_reservoir_k=300.0,
                    hot_reservoir_k=300.0,
                    electrical_resistance_ohm=contact_resistance,
                    eta_to_cold=eta,
                    cold_thermal_resistance_k_per_w=cold_rth,
                    hot_thermal_resistance_k_per_w=hot_rth,
                )
                report = solve_fixed_current_boundary_network(network, current)
                self.assertTrue(report.accepted, report.diagnostics.message)
                point = report.require_point()

                joule_bulk_half = 0.5 * current**2 * bulk_resistance
                joule_contact = current**2 * contact_resistance
                coefficient_matrix = np.asarray(
                    [
                        [
                            1.0 + cold_rth * (alpha * current + bulk_conductance),
                            -cold_rth * bulk_conductance,
                        ],
                        [
                            -hot_rth * bulk_conductance,
                            1.0
                            - hot_rth * (alpha * current - bulk_conductance),
                        ],
                    ],
                    dtype=float,
                )
                right_hand_side = np.asarray(
                    [
                        300.0
                        + cold_rth
                        * (joule_bulk_half + eta * joule_contact),
                        300.0
                        + hot_rth
                        * (joule_bulk_half + (1.0 - eta) * joule_contact),
                    ],
                    dtype=float,
                )
                exact_cold, exact_hot = np.linalg.solve(
                    coefficient_matrix, right_hand_side
                )
                self.assertGreater(exact_cold, exact_hot)
                self.assertAlmostEqual(
                    point.cold_leg_temperature_k, exact_cold, delta=2.0e-9
                )
                self.assertAlmostEqual(
                    point.hot_leg_temperature_k, exact_hot, delta=2.0e-9
                )
                self.assert_energy_closed(point)

    def test_one_ideal_thermal_contact_is_an_exact_constraint(self) -> None:
        for cold_resistance, hot_resistance in ((0.0, 7.0), (5.0, 0.0)):
            with self.subTest(
                cold_resistance=cold_resistance,
                hot_resistance=hot_resistance,
            ):
                network = synthetic_network(
                    cold_thermal_resistance_k_per_w=cold_resistance,
                    hot_thermal_resistance_k_per_w=hot_resistance,
                )
                report = solve_fixed_current_boundary_network(network, 1.0)
                point = report.require_point()
                if cold_resistance == 0.0:
                    self.assertAlmostEqual(
                        point.cold_leg_temperature_k,
                        network.cold_reservoir_temperature_k,
                        delta=1.0e-10,
                    )
                if hot_resistance == 0.0:
                    self.assertAlmostEqual(
                        point.hot_leg_temperature_k,
                        network.hot_reservoir_temperature_k,
                        delta=1.0e-10,
                    )
                self.assertTrue(report.diagnostics.converged)
                self.assert_energy_closed(point)

        # Exact elimination must not degrade when a caller deliberately uses
        # loose nonlinear and acceptance tolerances for a screening run.
        loose_options = BoundaryNetworkSolverOptions(
            nonlinear_tolerance=1.0e-1,
            temperature_residual_tolerance_k=1.0,
            node_energy_residual_tolerance_w=1.0,
        )
        for cold_resistance, hot_resistance in ((0.0, 7.0), (5.0, 0.0)):
            with self.subTest(
                loose_cold_resistance=cold_resistance,
                loose_hot_resistance=hot_resistance,
            ):
                loose_point = solve_fixed_current_boundary_network(
                    synthetic_network(
                        cold_thermal_resistance_k_per_w=cold_resistance,
                        hot_thermal_resistance_k_per_w=hot_resistance,
                    ),
                    1.0,
                    options=loose_options,
                ).require_point()
                if cold_resistance == 0.0:
                    self.assertEqual(loose_point.cold_leg_temperature_k, 290.0)
                if hot_resistance == 0.0:
                    self.assertEqual(loose_point.hot_leg_temperature_k, 320.0)

    def test_passive_two_terminal_parasitic_lowers_both_ports_equally(self) -> None:
        baseline = solve_fixed_current_boundary_network(
            synthetic_network(parasitic_conductance_w_per_k=0.0), 1.0
        ).require_point()
        conductance = 2.0e-4
        leaked = solve_fixed_current_boundary_network(
            synthetic_network(parasitic_conductance_w_per_k=conductance), 1.0
        ).require_point()
        expected_leak = conductance * (
            leaked.hot_reservoir_temperature_k - leaked.cold_reservoir_temperature_k
        )
        self.assertGreater(expected_leak, 0.0)
        self.assertAlmostEqual(leaked.q_parasitic_w, expected_leak, places=15)
        self.assertAlmostEqual(
            baseline.qc_net_w - leaked.qc_net_w, expected_leak, places=14
        )
        self.assertAlmostEqual(
            baseline.qh_net_w - leaked.qh_net_w, expected_leak, places=14
        )
        self.assertAlmostEqual(
            baseline.qh_net_w - baseline.qc_net_w,
            leaked.qh_net_w - leaked.qc_net_w,
            places=14,
        )
        self.assertEqual(baseline.voltage_terminal_v, leaked.voltage_terminal_v)
        self.assert_energy_closed(leaked)

    def test_invalid_physics_and_unsupported_role_are_rejected_at_input(self) -> None:
        for kwargs in (
            {"resistance_ohm": -1.0, "joule_fraction_to_cold_node": 0.2},
            {"resistance_ohm": 0.0, "joule_fraction_to_cold_node": -0.01},
            {"resistance_ohm": 0.0, "joule_fraction_to_cold_node": 1.01},
            {"resistance_ohm": math.inf, "joule_fraction_to_cold_node": 0.2},
        ):
            with self.subTest(electrical=kwargs), self.assertRaises(ValueError):
                SeriesElectricalContacts(**kwargs)
        with self.assertRaises(ValueError):
            ReservoirThermalContacts(-1.0, 0.0)
        with self.assertRaises(ValueError):
            ReservoirThermalContacts(0.0, -1.0)
        with self.assertRaises(ValueError):
            TwoReservoirParasitic(-1.0e-4)
        with self.assertRaises(ValueError):
            synthetic_network(cold_reservoir_k=321.0, hot_reservoir_k=320.0)
        with self.assertRaises(ValueError):
            synthetic_network(energy_scale_w=0.0)
        with self.assertRaises(ValueError):
            synthetic_network(data_role="measured")
        with self.assertRaises(PropertyDomainError):
            synthetic_network(cold_reservoir_k=240.0, hot_reservoir_k=320.0)
        with self.assertRaises(ValueError):
            BoundaryNetworkSolverOptions(nonlinear_tolerance=1.0e-16)
        with self.assertRaises(ValueError):
            BoundaryNetworkSolverOptions(node_energy_residual_tolerance_w=0.0)
        with self.assertRaises(TypeError):
            solve_fixed_current_boundary_network(synthetic_network(), True)
        with self.assertRaises(ValueError):
            solve_fixed_current_boundary_network(synthetic_network(), math.inf)

    def test_solve_domain_excursion_and_nonconvergence_are_preserved(self) -> None:
        # The specified property support is closed: exact endpoint operation
        # and small signed currents must survive BVP roundoff handling.
        for current in (-0.01, 0.0, 1.0e-8, 0.01):
            with self.subTest(closed_domain_current=current):
                closed_domain = solve_fixed_current_boundary_network(
                    synthetic_network(domain=(290.0, 320.0)), current
                )
                self.assertTrue(
                    closed_domain.accepted, closed_domain.diagnostics.message
                )

        domain_report = solve_fixed_current_boundary_network(
            synthetic_network(domain=(290.0, 320.0)), 10.0
        )
        self.assertFalse(domain_report.accepted)
        self.assertEqual(
            domain_report.rejection_codes, (REJECTION_PROPERTY_DOMAIN,)
        )
        self.assertIsNone(domain_report.point)
        self.assertIn("property-domain rejection", domain_report.diagnostics.message)
        with self.assertRaises(RuntimeError):
            domain_report.require_point()

        nonconverged = solve_fixed_current_boundary_network(
            synthetic_network(
                electrical_resistance_ohm=0.02,
                cold_thermal_resistance_k_per_w=5.0,
                hot_thermal_resistance_k_per_w=7.0,
            ),
            1.0,
            options=BoundaryNetworkSolverOptions(max_function_evaluations=1),
        )
        self.assertFalse(nonconverged.accepted)
        self.assertEqual(
            nonconverged.rejection_codes, (REJECTION_NUMERICAL_INVALID,)
        )
        self.assertFalse(nonconverged.diagnostics.converged)
        self.assertIsNone(nonconverged.point)


if __name__ == "__main__":
    unittest.main()
