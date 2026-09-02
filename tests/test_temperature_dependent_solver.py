"""Verification tests for the conservative temperature-dependent 1D solver."""

from __future__ import annotations

import math
import unittest

import numpy as np

from scripts.analysis.temperature_dependent_exact_benchmarks import (
    ManufacturedLinearTemperatureCase,
    ZeroCurrentLinearKappaCase,
)
from scripts.tec_1d_solver.boundary_network import (
    SYNTHETIC_DATA_ROLE,
    FixedCurrentBoundaryNetwork,
    ReservoirThermalContacts,
    SeriesElectricalContacts,
    TwoReservoirParasitic,
    solve_fixed_current_boundary_network,
)
from scripts.tec_1d_solver import (
    ConstantPropertyLeg,
    ConstantPropertyNumericalCouple,
    ConstantTemperatureProperty,
    LinearTemperatureProperty,
    PchipTemperatureProperty,
    PropertyDomainError,
    TemperatureDependentLeg,
    TemperatureDependentNumericalCouple,
    solve_temperature_dependent_couple,
    solve_temperature_dependent_leg,
)


DOMAIN = (250.0, 400.0)


def constant_property(value: float) -> ConstantTemperatureProperty:
    return ConstantTemperatureProperty(value, *DOMAIN)


class _PositiveLogPchipForRegression(PchipTemperatureProperty):
    """Test-only positive log-PCHIP matching the sensitivity transport law."""

    def evaluate(self, temperature_k: object) -> np.ndarray:
        log_value = PchipTemperatureProperty.evaluate(self, temperature_k)
        return np.asarray(np.exp(log_value), dtype=float)

    def derivative(self, temperature_k: object) -> np.ndarray:
        log_value = PchipTemperatureProperty.evaluate(self, temperature_k)
        dlog_dtemperature = PchipTemperatureProperty.derivative(self, temperature_k)
        return np.asarray(np.exp(log_value) * dlog_dtemperature, dtype=float)


def constant_limit_couples() -> tuple[
    TemperatureDependentNumericalCouple, ConstantPropertyNumericalCouple
]:
    p_parameters = (220.0e-6, 1.3e-5, 1.7, 1.5e-3, 1.2e-6)
    n_parameters = (-135.0e-6, 2.2e-5, 0.9, 0.8e-3, 0.7e-6)
    variable = TemperatureDependentNumericalCouple(
        p_leg=TemperatureDependentLeg(
            constant_property(p_parameters[0]),
            constant_property(p_parameters[1]),
            constant_property(p_parameters[2]),
            p_parameters[3],
            p_parameters[4],
        ),
        n_leg=TemperatureDependentLeg(
            constant_property(n_parameters[0]),
            constant_property(n_parameters[1]),
            constant_property(n_parameters[2]),
            n_parameters[3],
            n_parameters[4],
        ),
        cold_temperature_k=280.0,
        hot_temperature_k=335.0,
    )
    constant = ConstantPropertyNumericalCouple(
        p_leg=ConstantPropertyLeg(*p_parameters),
        n_leg=ConstantPropertyLeg(*n_parameters),
        cold_temperature_k=280.0,
        hot_temperature_k=335.0,
    )
    return variable, constant


def shifted_linear_couple(
    *, common_value_shift: float = 0.0, common_slope_shift: float = 0.0
) -> TemperatureDependentNumericalCouple:
    domain = (270.0, 360.0)
    reference = 300.0

    def linear(value: float, slope: float) -> LinearTemperatureProperty:
        return LinearTemperatureProperty(reference, value, slope, *domain)

    return TemperatureDependentNumericalCouple(
        p_leg=TemperatureDependentLeg(
            linear(220.0e-6 + common_value_shift, 0.5e-6 + common_slope_shift),
            linear(1.1e-5, 1.0e-8),
            linear(1.4, -2.0e-3),
            length_m=1.2e-3,
            area_m2=1.1e-6,
        ),
        n_leg=TemperatureDependentLeg(
            linear(-180.0e-6 + common_value_shift, -0.3e-6 + common_slope_shift),
            linear(2.0e-5, -2.0e-8),
            linear(0.9, 1.0e-3),
            length_m=0.8e-3,
            area_m2=0.7e-6,
        ),
        cold_temperature_k=295.0,
        hot_temperature_k=325.0,
    )


class TemperatureDependentSolverTests(unittest.TestCase):
    def test_constant_property_limit_matches_independent_reference(self) -> None:
        variable, constant = constant_limit_couples()
        analytic_model = constant.analytic_reference()
        for current in (-0.4, 0.0, 0.9):
            with self.subTest(current=current):
                point = solve_temperature_dependent_couple(
                    variable,
                    current,
                    initial_mesh_points=31,
                    output_points=201,
                    relative_tolerance=1.0e-9,
                )
                analytic = analytic_model.evaluate(current)
                for field in ("Qc_w", "Qh_w", "V_v", "Pin_w"):
                    self.assertAlmostEqual(
                        getattr(point, field), getattr(analytic, field), delta=2.0e-10
                    )
                self.assertEqual(point.p_leg.signed_current_a, current)
                self.assertEqual(point.n_leg.signed_current_a, -current)
                self.assertLess(point.relative_energy_residual, 1.0e-9)
                self.assertLess(point.p_leg.relative_local_energy_residual, 1.0e-9)
                self.assertLess(point.n_leg.relative_local_energy_residual, 1.0e-9)

    def test_reversed_endpoint_temperatures_obey_spatial_reversal_symmetry(self) -> None:
        leg = shifted_linear_couple().p_leg
        cold_side_temperature = 325.0
        hot_side_temperature = 295.0
        signed_current = 0.7
        direct = solve_temperature_dependent_leg(
            leg,
            signed_current,
            cold_side_temperature,
            hot_side_temperature,
            initial_mesh_points=31,
            output_points=201,
            relative_tolerance=1.0e-9,
        )
        spatially_reversed = solve_temperature_dependent_leg(
            leg,
            -signed_current,
            hot_side_temperature,
            cold_side_temperature,
            initial_mesh_points=31,
            output_points=201,
            relative_tolerance=1.0e-9,
        )

        np.testing.assert_allclose(
            direct.temperature_k,
            spatially_reversed.temperature_k[::-1],
            rtol=0.0,
            atol=2.0e-10,
        )
        np.testing.assert_allclose(
            direct.heat_flux_w_per_m2,
            -spatially_reversed.heat_flux_w_per_m2[::-1],
            rtol=0.0,
            atol=5.0e-8,
        )
        self.assertAlmostEqual(
            direct.hot_minus_cold_potential_v,
            -spatially_reversed.hot_minus_cold_potential_v,
            delta=2.0e-13,
        )
        self.assertAlmostEqual(
            direct.cold_heat_rate_w,
            -spatially_reversed.hot_heat_rate_w,
            delta=2.0e-13,
        )
        self.assertAlmostEqual(
            direct.hot_heat_rate_w,
            -spatially_reversed.cold_heat_rate_w,
            delta=2.0e-13,
        )
        self.assertLess(direct.relative_local_energy_residual, 1.0e-9)

    def test_unit_coordinate_prevents_turning_point_mesh_collapse(self) -> None:
        # Exact n-leg state from the failed S1 pilot case
        # oat_n_thermal_conductivity_target_ratio_low / variable common mode.
        # On an x-in-metres BVP this case exhausted 30,000 nodes at tol=5e-8:
        # the mesh collapsed to ~6.6e-13 m around an interior T maximum and the
        # temperature-equation residual rose to O(1e-2).  The unit-coordinate
        # solve must retain the specified tolerance, not mask it by relaxing
        # solver controls.
        leg = TemperatureDependentLeg(
            seebeck=PchipTemperatureProperty(
                (250.0, 360.0), (-75.25e-6, -206.7e-6)
            ),
            electrical_resistivity=_PositiveLogPchipForRegression(
                (250.0, 360.0),
                (
                    math.log(2.5271363809934766e-5),
                    math.log(1.8628396871975623e-5),
                ),
            ),
            thermal_conductivity=_PositiveLogPchipForRegression(
                (250.0, 360.0), (math.log(0.45), math.log(2.067657038994663))
            ),
            length_m=0.8e-3,
            area_m2=0.7e-6,
        )
        common = {
            "initial_mesh_points": 31,
            "output_points": 201,
        }
        solution = solve_temperature_dependent_leg(
            leg,
            1.0,
            276.04832450747267,
            296.7480057771923,
            relative_tolerance=5.0e-8,
            max_nodes=500,
            **common,
        )
        reference = solve_temperature_dependent_leg(
            leg,
            1.0,
            276.04832450747267,
            296.7480057771923,
            relative_tolerance=1.0e-9,
            max_nodes=2000,
            **common,
        )

        self.assertLess(solution.adaptive_mesh_nodes, 250)
        self.assertLess(reference.adaptive_mesh_nodes, 1000)
        self.assertLessEqual(solution.maximum_rms_bvp_residual, 5.0e-8)
        self.assertLessEqual(reference.maximum_rms_bvp_residual, 1.0e-9)
        self.assertGreater(
            float(np.max(solution.temperature_k))
            - max(solution.temperature_k[0], solution.temperature_k[-1]),
            0.04,
        )
        np.testing.assert_allclose(
            solution.temperature_k,
            reference.temperature_k,
            rtol=0.0,
            atol=2.0e-8,
        )
        np.testing.assert_allclose(
            solution.heat_flux_w_per_m2,
            reference.heat_flux_w_per_m2,
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            solution.potential_v, reference.potential_v, rtol=0.0, atol=1.0e-12
        )
        for field in (
            "cold_heat_rate_w",
            "hot_heat_rate_w",
            "hot_minus_cold_potential_v",
        ):
            self.assertAlmostEqual(
                getattr(solution, field), getattr(reference, field), delta=1.0e-10
            )
        self.assertLess(solution.relative_local_energy_residual, 1.0e-12)
        self.assertLess(reference.relative_local_energy_residual, 1.0e-12)

    def test_zero_current_variable_conductivity_has_exact_integral_solution(self) -> None:
        case = ZeroCurrentLinearKappaCase()
        domain = (case.cold_temperature_k - 10.0, case.hot_temperature_k + 10.0)
        leg = TemperatureDependentLeg(
            seebeck=LinearTemperatureProperty(
                case.cold_temperature_k,
                case.seebeck_at_cold_v_per_k,
                case.dseebeck_dtemperature_v_per_k2,
                *domain,
            ),
            electrical_resistivity=ConstantTemperatureProperty(1.5e-5, *domain),
            thermal_conductivity=LinearTemperatureProperty(
                case.cold_temperature_k,
                case.kappa_at_cold_w_per_m_k,
                case.dkappa_dtemperature_w_per_m_k2,
                *domain,
            ),
            length_m=case.length_m,
            area_m2=case.area_m2,
        )
        solution = solve_temperature_dependent_leg(
            leg,
            0.0,
            case.cold_temperature_k,
            case.hot_temperature_k,
            initial_mesh_points=21,
            output_points=301,
            relative_tolerance=1.0e-9,
        )
        exact_temperature = np.asarray(case.temperature_k(solution.coordinate_m))
        exact_terminal = case.terminal_reference()

        np.testing.assert_allclose(
            solution.temperature_k, exact_temperature, rtol=0.0, atol=2.0e-8
        )
        np.testing.assert_allclose(
            solution.heat_flux_w_per_m2,
            case.heat_flux_w_per_m2,
            rtol=2.0e-10,
            atol=1.0e-7,
        )
        self.assertAlmostEqual(
            solution.cold_heat_rate_w,
            exact_terminal["cold_heat_rate_w"],
            delta=2.0e-12,
        )
        self.assertAlmostEqual(
            solution.hot_heat_rate_w,
            exact_terminal["hot_heat_rate_w"],
            delta=2.0e-12,
        )
        self.assertAlmostEqual(
            solution.hot_minus_cold_potential_v,
            exact_terminal["hot_minus_cold_potential_v"],
            delta=2.0e-11,
        )
        self.assertLess(solution.relative_local_energy_residual, 1.0e-10)

    def test_nonzero_thomson_manufactured_solution_is_exact(self) -> None:
        case = ManufacturedLinearTemperatureCase()
        domain = (case.cold_temperature_k - 10.0, case.hot_temperature_k + 10.0)
        resistivity_slope = (
            case.dseebeck_dtemperature_v_per_k2
            * case.temperature_gradient_k_per_m
            / case.current_density_a_per_m2
        )
        leg = TemperatureDependentLeg(
            seebeck=LinearTemperatureProperty(
                case.cold_temperature_k,
                case.seebeck_at_cold_v_per_k,
                case.dseebeck_dtemperature_v_per_k2,
                *domain,
            ),
            electrical_resistivity=LinearTemperatureProperty(
                case.cold_temperature_k,
                resistivity_slope * case.cold_temperature_k,
                resistivity_slope,
                *domain,
            ),
            thermal_conductivity=ConstantTemperatureProperty(
                case.thermal_conductivity_w_per_m_k, *domain
            ),
            length_m=case.length_m,
            area_m2=case.area_m2,
        )
        solution = solve_temperature_dependent_leg(
            leg,
            case.signed_current_a,
            case.cold_temperature_k,
            case.hot_temperature_k,
            initial_mesh_points=9,
            output_points=101,
            relative_tolerance=1.0e-10,
        )
        exact_temperature = np.asarray(case.temperature_k(solution.coordinate_m))
        exact_heat_flux = np.asarray(case.heat_flux_w_per_m2(solution.coordinate_m))
        exact_potential = np.asarray(case.potential_v(solution.coordinate_m))
        exact_terminal = case.terminal_reference()

        np.testing.assert_allclose(
            solution.temperature_k, exact_temperature, rtol=0.0, atol=2.0e-11
        )
        np.testing.assert_allclose(
            solution.heat_flux_w_per_m2, exact_heat_flux, rtol=0.0, atol=1.0e-8
        )
        np.testing.assert_allclose(
            solution.potential_v, exact_potential, rtol=0.0, atol=2.0e-14
        )
        joule_source = (
            solution.electrical_resistivity_ohm_m
            * case.current_density_a_per_m2**2
        )
        thomson_source = (
            solution.thomson_coefficient_v_per_k
            * case.current_density_a_per_m2
            * solution.temperature_gradient_k_per_m
        )
        np.testing.assert_allclose(joule_source, thomson_source, rtol=2.0e-14)
        self.assertAlmostEqual(
            solution.cold_heat_rate_w,
            exact_terminal["cold_heat_rate_w"],
            delta=1.0e-12,
        )
        self.assertAlmostEqual(
            solution.hot_heat_rate_w,
            exact_terminal["hot_heat_rate_w"],
            delta=1.0e-12,
        )
        self.assertAlmostEqual(
            solution.hot_minus_cold_potential_v,
            exact_terminal["hot_minus_cold_potential_v"],
            delta=1.0e-12,
        )
        self.assertGreater(
            float(np.min(np.abs(solution.thomson_coefficient_v_per_k))), 0.0
        )
        self.assertLess(solution.maximum_rms_bvp_residual, 1.0e-10)
        self.assertLess(solution.relative_local_energy_residual, 1.0e-12)

    def test_constant_common_seebeck_shift_is_invariant_for_ideal_couple(self) -> None:
        baseline = solve_temperature_dependent_couple(
            shifted_linear_couple(), 0.7, relative_tolerance=1.0e-9
        )
        shifted = solve_temperature_dependent_couple(
            shifted_linear_couple(common_value_shift=80.0e-6),
            0.7,
            relative_tolerance=1.0e-9,
        )
        for field in ("Qc_w", "Qh_w", "V_v", "Pin_w"):
            with self.subTest(field=field):
                self.assertAlmostEqual(
                    getattr(baseline, field), getattr(shifted, field), delta=2.0e-11
                )
        np.testing.assert_allclose(
            baseline.p_leg.temperature_k,
            shifted.p_leg.temperature_k,
            rtol=0.0,
            atol=2.0e-9,
        )
        np.testing.assert_allclose(
            baseline.n_leg.temperature_k,
            shifted.n_leg.temperature_k,
            rtol=0.0,
            atol=2.0e-9,
        )

    def test_canonical_gauge_field_transforms(self) -> None:
        """Freeze physical q/phi reconstruction for both legs and orientations."""

        solve_options = {
            "initial_mesh_points": 31,
            "output_points": 121,
            "relative_tolerance": 1.0e-9,
        }
        shifts = (-80.0e-6, -40.0e-6, 40.0e-6, 80.0e-6)
        for cold, hot in ((295.0, 325.0), (325.0, 295.0)):
            for terminal_current in (-1.0, 0.0, 1.0):
                baseline_couple = shifted_linear_couple()
                for leg_name, baseline_leg, signed_current in (
                    ("p", baseline_couple.p_leg, terminal_current),
                    ("n", baseline_couple.n_leg, -terminal_current),
                ):
                    baseline = solve_temperature_dependent_leg(
                        baseline_leg,
                        signed_current,
                        cold,
                        hot,
                        **solve_options,
                    )
                    for shift in shifts:
                        shifted_leg = getattr(
                            shifted_linear_couple(common_value_shift=shift),
                            f"{leg_name}_leg",
                        )
                        shifted = solve_temperature_dependent_leg(
                            shifted_leg,
                            signed_current,
                            cold,
                            hot,
                            **solve_options,
                        )
                        with self.subTest(
                            leg=leg_name,
                            current=terminal_current,
                            cold=cold,
                            hot=hot,
                            shift=shift,
                        ):
                            np.testing.assert_array_equal(
                                shifted.coordinate_m, baseline.coordinate_m
                            )
                            np.testing.assert_allclose(
                                shifted.temperature_k,
                                baseline.temperature_k,
                                rtol=0.0,
                                atol=1.0e-12,
                            )
                            np.testing.assert_allclose(
                                shifted.temperature_gradient_k_per_m,
                                baseline.temperature_gradient_k_per_m,
                                rtol=0.0,
                                atol=1.0e-8,
                            )
                            np.testing.assert_allclose(
                                shifted.heat_flux_w_per_m2,
                                baseline.heat_flux_w_per_m2
                                + shift
                                * baseline.temperature_k
                                * baseline.current_density_a_per_m2,
                                rtol=0.0,
                                atol=1.0e-8,
                            )
                            np.testing.assert_allclose(
                                shifted.potential_v,
                                baseline.potential_v
                                - shift * (baseline.temperature_k - cold),
                                rtol=0.0,
                                atol=1.0e-15,
                            )
                            np.testing.assert_allclose(
                                shifted.thomson_coefficient_v_per_k,
                                baseline.thomson_coefficient_v_per_k,
                                rtol=0.0,
                                atol=1.0e-18,
                            )
                            self.assertAlmostEqual(
                                shifted.cold_heat_rate_w,
                                baseline.cold_heat_rate_w
                                + shift * cold * signed_current,
                                delta=1.0e-12,
                            )
                            self.assertAlmostEqual(
                                shifted.hot_heat_rate_w,
                                baseline.hot_heat_rate_w
                                + shift * hot * signed_current,
                                delta=1.0e-12,
                            )
                            self.assertAlmostEqual(
                                shifted.hot_minus_cold_potential_v,
                                baseline.hot_minus_cold_potential_v
                                - shift * (hot - cold),
                                delta=1.0e-15,
                            )
                            self.assertLess(
                                shifted.relative_local_energy_residual, 1.0e-10
                            )

    def test_common_gauge_with_contacts_and_endpoint_inversion(
        self,
    ) -> None:
        """Freeze ideal-couple invariance through the finite-contact network."""

        def network(common_shift: float) -> FixedCurrentBoundaryNetwork:
            couple = shifted_linear_couple(common_value_shift=common_shift)
            return FixedCurrentBoundaryNetwork(
                p_leg=couple.p_leg,
                n_leg=couple.n_leg,
                cold_reservoir_temperature_k=300.0,
                hot_reservoir_temperature_k=300.0,
                electrical_contacts=SeriesElectricalContacts(
                    resistance_ohm=0.01,
                    joule_fraction_to_cold_node=0.37,
                ),
                thermal_contacts=ReservoirThermalContacts(
                    cold_resistance_k_per_w=20.0,
                    hot_resistance_k_per_w=20.0,
                ),
                parasitic=TwoReservoirParasitic(
                    thermal_conductance_w_per_k=1.0e-4
                ),
                energy_scale_w=0.01,
                data_role=SYNTHETIC_DATA_ROLE,
            )

        for current in (-1.0, 0.0, 1.0):
            baseline_report = solve_fixed_current_boundary_network(
                network(0.0), current
            )
            self.assertTrue(baseline_report.accepted, baseline_report.diagnostics)
            baseline = baseline_report.require_point()
            if current == -1.0:
                self.assertGreater(
                    baseline.cold_leg_temperature_k,
                    baseline.hot_leg_temperature_k,
                )
            for shift in (-80.0e-6, -40.0e-6, 40.0e-6, 80.0e-6):
                shifted_report = solve_fixed_current_boundary_network(
                    network(shift), current
                )
                with self.subTest(current=current, shift=shift):
                    self.assertTrue(
                        shifted_report.accepted, shifted_report.diagnostics
                    )
                    shifted = shifted_report.require_point()
                    for field in (
                        "cold_leg_temperature_k",
                        "hot_leg_temperature_k",
                        "qc_net_w",
                        "qh_net_w",
                        "voltage_terminal_v",
                        "input_power_w",
                    ):
                        self.assertAlmostEqual(
                            getattr(shifted, field),
                            getattr(baseline, field),
                            delta=1.0e-11,
                        )
                    self.assertLess(shifted.energy_residual_fraction, 1.0e-9)

    def test_temperature_dependent_common_shift_breaks_general_invariance(self) -> None:
        baseline_couple = shifted_linear_couple()
        shifted_couple = shifted_linear_couple(common_slope_shift=1.0e-6)
        sample_temperature = np.asarray([295.0, 310.0, 325.0])
        baseline_difference = (
            baseline_couple.p_leg.seebeck.evaluate(sample_temperature)
            - baseline_couple.n_leg.seebeck.evaluate(sample_temperature)
        )
        shifted_difference = (
            shifted_couple.p_leg.seebeck.evaluate(sample_temperature)
            - shifted_couple.n_leg.seebeck.evaluate(sample_temperature)
        )
        np.testing.assert_allclose(baseline_difference, shifted_difference, atol=1.0e-18)

        baseline = solve_temperature_dependent_couple(baseline_couple, 0.7)
        shifted = solve_temperature_dependent_couple(shifted_couple, 0.7)
        self.assertGreater(abs(shifted.Qc_w - baseline.Qc_w), 1.0e-5)
        self.assertGreater(abs(shifted.V_v - baseline.V_v), 1.0e-6)
        self.assertLess(baseline.relative_energy_residual, 1.0e-9)
        self.assertLess(shifted.relative_energy_residual, 1.0e-9)

    def test_property_domains_and_nonphysical_transport_are_fail_closed(self) -> None:
        law = LinearTemperatureProperty(300.0, 1.0, 0.01, 280.0, 320.0)
        with self.assertRaises(PropertyDomainError):
            law.evaluate(np.asarray([300.0, 321.0]))
        with self.assertRaises(ValueError):
            LinearTemperatureProperty(330.0, 1.0, 0.0, 280.0, 320.0)
        with self.assertRaises(ValueError):
            TemperatureDependentLeg(
                seebeck=ConstantTemperatureProperty(200.0e-6, 280.0, 320.0),
                electrical_resistivity=LinearTemperatureProperty(
                    300.0, 1.0e-6, -1.0e-6, 280.0, 320.0
                ),
                thermal_conductivity=ConstantTemperatureProperty(1.0, 280.0, 320.0),
                length_m=1.0e-3,
                area_m2=1.0e-6,
            )

    def test_closed_property_domain_endpoints_survive_bvp_jacobian_probes(self) -> None:
        domain = (290.0, 320.0)

        def constant(value: float) -> ConstantTemperatureProperty:
            return ConstantTemperatureProperty(value, *domain)

        couple = TemperatureDependentNumericalCouple(
            p_leg=TemperatureDependentLeg(
                constant(220.0e-6),
                constant(1.3e-5),
                constant(1.7),
                length_m=1.5e-3,
                area_m2=1.2e-6,
            ),
            n_leg=TemperatureDependentLeg(
                constant(-135.0e-6),
                constant(2.2e-5),
                constant(0.9),
                length_m=0.8e-3,
                area_m2=0.7e-6,
            ),
            cold_temperature_k=domain[0],
            hot_temperature_k=domain[1],
        )

        # Without the analytic BVP Jacobian, SciPy's finite-difference probe
        # asks for T_max + roughly 4.8e-6 K and falsely trips the strict domain.
        # Heat-flux state scaling is also needed for the tiny-current residual.
        for current in (0.0, 1.0e-8, 1.0):
            with self.subTest(current=current):
                point = solve_temperature_dependent_couple(couple, current)
                self.assertEqual(point.p_leg.temperature_k[0], domain[0])
                self.assertEqual(point.p_leg.temperature_k[-1], domain[1])
                self.assertEqual(point.n_leg.temperature_k[0], domain[0])
                self.assertEqual(point.n_leg.temperature_k[-1], domain[1])
                self.assertLess(point.relative_energy_residual, 1.0e-10)

        narrow_leg = TemperatureDependentLeg(
            seebeck=ConstantTemperatureProperty(200.0e-6, 299.0, 311.0),
            electrical_resistivity=ConstantTemperatureProperty(1.0e-5, 299.0, 311.0),
            thermal_conductivity=ConstantTemperatureProperty(1.0, 299.0, 311.0),
            length_m=1.0e-3,
            area_m2=1.0e-6,
        )
        with self.assertRaises(PropertyDomainError):
            solve_temperature_dependent_leg(narrow_leg, 5.0, 300.0, 310.0)

    def test_invalid_solver_controls_are_rejected(self) -> None:
        variable, _ = constant_limit_couples()
        with self.assertRaises(ValueError):
            solve_temperature_dependent_couple(variable, 0.5, initial_mesh_points=2)
        with self.assertRaises(TypeError):
            solve_temperature_dependent_couple(  # type: ignore[arg-type]
                variable, 0.5, output_points=20.5
            )
        with self.assertRaises(ValueError):
            solve_temperature_dependent_couple(variable, math.inf)


if __name__ == "__main__":
    unittest.main()
