"""Tests for fail-closed, differentiable tabulated property laws."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.tec_1d_solver import (
    ConstantTemperatureProperty,
    LinearTemperatureProperty,
    PchipTemperatureProperty,
    PropertyDomainError,
    TemperatureDependentLeg,
    solve_temperature_dependent_leg,
)


KNOTS = np.asarray([270.0, 287.0, 305.0, 331.0, 350.0])


class PchipTemperaturePropertyTests(unittest.TestCase):
    def test_linear_data_have_exact_values_and_analytic_derivative(self) -> None:
        reference_temperature = 300.0
        reference_value = 210.0e-6
        slope = 0.65e-6
        knot_values = reference_value + slope * (KNOTS - reference_temperature)
        law = PchipTemperatureProperty(KNOTS, knot_values)

        sample = np.linspace(KNOTS[0], KNOTS[-1], 121)
        expected = reference_value + slope * (sample - reference_temperature)
        np.testing.assert_allclose(law.evaluate(sample), expected, rtol=0.0, atol=2e-19)
        np.testing.assert_allclose(law.value(sample), expected, rtol=0.0, atol=2e-19)
        np.testing.assert_allclose(
            law.derivative(sample), np.full_like(sample, slope), rtol=2e-14, atol=2e-20
        )
        self.assertEqual(np.asarray(law.evaluate(300.0)).shape, ())
        self.assertAlmostEqual(float(law.evaluate(300.0)), reference_value, delta=1e-20)
        self.assertEqual(law.minimum_temperature_k, float(KNOTS[0]))
        self.assertEqual(law.maximum_temperature_k, float(KNOTS[-1]))

    def test_nonlinear_monotone_positive_data_remain_shape_preserving(self) -> None:
        values = np.asarray([0.42, 0.47, 0.71, 0.74, 1.05])
        law = PchipTemperatureProperty(KNOTS, values)
        sample = np.linspace(KNOTS[0], KNOTS[-1], 2001)
        interpolated = law.evaluate(sample)

        self.assertGreater(float(np.min(interpolated)), 0.0)
        self.assertGreaterEqual(float(np.min(np.diff(interpolated))), -2e-15)
        self.assertGreaterEqual(float(np.min(law.derivative(sample))), -2e-15)
        self.assertGreaterEqual(float(np.min(interpolated)), float(values[0]))
        self.assertLessEqual(float(np.max(interpolated)), float(values[-1]))
        np.testing.assert_array_equal(law.evaluate(KNOTS), values)

    def test_closed_domain_allows_boundaries_and_refuses_every_extrapolation(self) -> None:
        law = PchipTemperatureProperty(KNOTS, np.linspace(1.0, 2.0, KNOTS.size))
        np.testing.assert_allclose(law.evaluate([KNOTS[0], KNOTS[-1]]), [1.0, 2.0])
        self.assertTrue(np.all(np.isfinite(law.derivative([KNOTS[0], KNOTS[-1]]))))

        # A nonlinear solver may propose a representable value one ULP beyond
        # an imposed closed endpoint.  That roundoff is projected to the
        # endpoint, while the physical-scale excursions below remain errors.
        just_below = np.nextafter(KNOTS[0], -np.inf)
        just_above = np.nextafter(KNOTS[-1], np.inf)
        np.testing.assert_allclose(
            law.evaluate([just_below, just_above]), [1.0, 2.0], rtol=0.0, atol=0.0
        )

        for invalid_temperature in (
            KNOTS[0] - 1e-9,
            KNOTS[-1] + 1e-9,
            0.0,
            np.nan,
            np.inf,
        ):
            with self.subTest(invalid_temperature=invalid_temperature):
                with self.assertRaises(PropertyDomainError):
                    law.evaluate(invalid_temperature)
                with self.assertRaises(PropertyDomainError):
                    law.derivative(invalid_temperature)

    def test_malformed_tables_are_rejected_deterministically(self) -> None:
        invalid_cases = (
            ([300.0], [1.0]),
            ([280.0, 300.0], [1.0]),
            ([280.0, 280.0, 310.0], [1.0, 1.1, 1.2]),
            ([300.0, 280.0], [1.0, 1.2]),
            ([0.0, 300.0], [1.0, 1.2]),
            ([280.0, np.nan], [1.0, 1.2]),
            ([280.0, 300.0], [1.0, np.inf]),
            ([[280.0, 300.0]], [[1.0, 1.2]]),
        )
        for temperature, values in invalid_cases:
            with self.subTest(temperature=temperature, values=values):
                with self.assertRaises((TypeError, ValueError)):
                    PchipTemperatureProperty(temperature, values)
        with self.assertRaises(TypeError):
            PchipTemperatureProperty([280.0, 300.0], [1.0 + 1.0j, 2.0])

        from_list = PchipTemperatureProperty([280.0, 300.0], [1.0, 2.0])
        from_array = PchipTemperatureProperty(
            np.asarray([280.0, 300.0]), np.asarray([1.0, 2.0])
        )
        self.assertEqual(from_list, from_array)
        self.assertIsInstance(from_list.temperature_knots_k, tuple)
        self.assertIsInstance(from_list.values_si, tuple)

    def test_leg_checks_transport_positivity_over_entire_common_domain(self) -> None:
        positive = PchipTemperatureProperty(KNOTS, [1.0, 0.8, 0.7, 0.9, 1.1])
        signed_seebeck = PchipTemperatureProperty(
            KNOTS, [210e-6, 205e-6, 215e-6, 230e-6, 225e-6]
        )
        invalid_interior = PchipTemperatureProperty(
            KNOTS, [1.0, 0.5, -0.1, 0.7, 1.0]
        )

        # Signed Seebeck is permitted, but rho and kappa are fail-closed even
        # when their two domain endpoints are positive.
        for resistivity, conductivity in (
            (invalid_interior, positive),
            (positive, invalid_interior),
        ):
            with self.subTest(
                invalid="rho" if resistivity is invalid_interior else "kappa"
            ):
                with self.assertRaisesRegex(ValueError, "must remain > 0"):
                    TemperatureDependentLeg(
                        signed_seebeck,
                        resistivity,
                        conductivity,
                        length_m=1e-3,
                        area_m2=1e-6,
                    )

        valid_leg = TemperatureDependentLeg(
            signed_seebeck,
            PchipTemperatureProperty(KNOTS, np.asarray(positive.values_si) * 1e-5),
            positive,
            length_m=1e-3,
            area_m2=1e-6,
        )
        self.assertEqual(valid_leg.minimum_valid_temperature_k, KNOTS[0])
        self.assertEqual(valid_leg.maximum_valid_temperature_k, KNOTS[-1])

    def test_tabulated_constant_limit_matches_production_solver(self) -> None:
        tabulated_leg = TemperatureDependentLeg(
            PchipTemperatureProperty(KNOTS, np.full(KNOTS.size, 205e-6)),
            PchipTemperatureProperty(KNOTS, np.full(KNOTS.size, 1.6e-5)),
            PchipTemperatureProperty(KNOTS, np.full(KNOTS.size, 1.25)),
            length_m=1.1e-3,
            area_m2=0.9e-6,
        )
        constant_leg = TemperatureDependentLeg(
            ConstantTemperatureProperty(205e-6, KNOTS[0], KNOTS[-1]),
            ConstantTemperatureProperty(1.6e-5, KNOTS[0], KNOTS[-1]),
            ConstantTemperatureProperty(1.25, KNOTS[0], KNOTS[-1]),
            length_m=1.1e-3,
            area_m2=0.9e-6,
        )
        self._assert_solver_limits_match(tabulated_leg, constant_leg)

    def test_tabulated_linear_limit_matches_production_solver(self) -> None:
        reference = 305.0
        parameters = (
            (205e-6, 0.45e-6),
            (1.6e-5, 1.1e-8),
            (1.25, -2.2e-3),
        )

        def table(value: float, slope: float) -> PchipTemperatureProperty:
            return PchipTemperatureProperty(KNOTS, value + slope * (KNOTS - reference))

        def linear(value: float, slope: float) -> LinearTemperatureProperty:
            return LinearTemperatureProperty(
                reference, value, slope, KNOTS[0], KNOTS[-1]
            )

        tabulated_leg = TemperatureDependentLeg(
            *(table(value, slope) for value, slope in parameters),
            length_m=1.1e-3,
            area_m2=0.9e-6,
        )
        linear_leg = TemperatureDependentLeg(
            *(linear(value, slope) for value, slope in parameters),
            length_m=1.1e-3,
            area_m2=0.9e-6,
        )
        self._assert_solver_limits_match(tabulated_leg, linear_leg)

    def test_nonlinear_common_mode_preserves_delta_s_by_construction(self) -> None:
        common_mode = np.asarray([0.0, 18e-6, -7e-6, 31e-6, 22e-6])
        baseline_p = PchipTemperatureProperty(KNOTS, np.full(KNOTS.size, 220e-6))
        baseline_n = PchipTemperatureProperty(KNOTS, np.full(KNOTS.size, -160e-6))
        shifted_p = PchipTemperatureProperty(KNOTS, common_mode + 220e-6)
        shifted_n = PchipTemperatureProperty(KNOTS, common_mode - 160e-6)
        sample = np.linspace(KNOTS[0], KNOTS[-1], 1001)

        baseline_delta = baseline_p.evaluate(sample) - baseline_n.evaluate(sample)
        shifted_delta = shifted_p.evaluate(sample) - shifted_n.evaluate(sample)
        np.testing.assert_allclose(shifted_delta, baseline_delta, rtol=0.0, atol=3e-19)
        np.testing.assert_allclose(
            shifted_p.derivative(sample),
            shifted_n.derivative(sample),
            rtol=0.0,
            atol=2e-20,
        )

    def _assert_solver_limits_match(
        self,
        tabulated_leg: TemperatureDependentLeg,
        reference_leg: TemperatureDependentLeg,
    ) -> None:
        options = {
            "initial_mesh_points": 31,
            "output_points": 151,
            "relative_tolerance": 1e-9,
        }
        tabulated = solve_temperature_dependent_leg(
            tabulated_leg, 0.28, 290.0, 325.0, **options
        )
        reference = solve_temperature_dependent_leg(
            reference_leg, 0.28, 290.0, 325.0, **options
        )
        for field in (
            "temperature_k",
            "heat_flux_w_per_m2",
            "potential_v",
            "thomson_coefficient_v_per_k",
        ):
            with self.subTest(field=field):
                np.testing.assert_allclose(
                    getattr(tabulated, field),
                    getattr(reference, field),
                    rtol=2e-11,
                    atol=2e-12,
                )
        for field in (
            "cold_heat_rate_w",
            "hot_heat_rate_w",
            "hot_minus_cold_potential_v",
        ):
            with self.subTest(field=field):
                self.assertAlmostEqual(
                    getattr(tabulated, field), getattr(reference, field), delta=2e-12
                )
        self.assertLess(tabulated.relative_local_energy_residual, 1e-9)


if __name__ == "__main__":
    unittest.main()
