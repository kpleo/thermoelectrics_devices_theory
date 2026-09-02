"""Verification tests for the first numerical 1D finite-difference baseline."""

from __future__ import annotations

import unittest

from scripts.tec_1d_solver import (
    ConstantPropertyLeg,
    ConstantPropertyNumericalCouple,
    solve_constant_couple,
)


def numerical_couple() -> ConstantPropertyNumericalCouple:
    return ConstantPropertyNumericalCouple(
        p_leg=ConstantPropertyLeg(
            seebeck_v_per_k=220.0e-6,
            electrical_resistivity_ohm_m=1.0e-5,
            thermal_conductivity_w_per_m_k=1.5,
            length_m=1.0e-3,
            area_m2=1.0e-6,
        ),
        n_leg=ConstantPropertyLeg(
            seebeck_v_per_k=-180.0e-6,
            electrical_resistivity_ohm_m=2.0e-5,
            thermal_conductivity_w_per_m_k=1.0,
            length_m=1.0e-3,
            area_m2=1.0e-6,
        ),
        cold_temperature_k=300.0,
        hot_temperature_k=310.0,
    )


class ConstantFiniteDifferenceTests(unittest.TestCase):
    def assert_matches_analytic(self, current: float, n_cells: int) -> None:
        couple = numerical_couple()
        numerical = solve_constant_couple(couple, current, n_cells=n_cells)
        analytic = couple.analytic_reference().evaluate(current)
        for field in ("Qc_w", "Qh_w", "V_v", "Pin_w"):
            with self.subTest(field=field, current=current, n_cells=n_cells):
                self.assertAlmostEqual(
                    getattr(numerical, field), getattr(analytic, field), places=11
                )
        self.assertLess(numerical.relative_energy_residual, 1.0e-11)

    def test_terminal_quantities_match_independent_closed_form(self) -> None:
        for current in (-1.0, 0.0, 0.7, 2.0):
            self.assert_matches_analytic(current, n_cells=20)

    def test_quadratic_solution_is_mesh_invariant_at_endpoints(self) -> None:
        couple = numerical_couple()
        points = [solve_constant_couple(couple, 1.1, n_cells=n) for n in (2, 4, 8, 32)]
        reference = points[-1]
        for point in points[:-1]:
            self.assertAlmostEqual(point.Qc_w, reference.Qc_w, places=10)
            self.assertAlmostEqual(point.Qh_w, reference.Qh_w, places=10)
            self.assertAlmostEqual(point.V_v, reference.V_v, places=10)

    def test_unequal_leg_lengths_and_areas_map_to_series_R_parallel_K(self) -> None:
        couple = ConstantPropertyNumericalCouple(
            p_leg=ConstantPropertyLeg(245e-6, 1.3e-5, 1.7, 1.5e-3, 1.2e-6),
            n_leg=ConstantPropertyLeg(-135e-6, 2.2e-5, 0.9, 0.8e-3, 0.7e-6),
            cold_temperature_k=280.0,
            hot_temperature_k=335.0,
        )
        analytic_model = couple.analytic_reference()
        for current in (-0.4, 0.9):
            numerical = solve_constant_couple(couple, current, n_cells=24)
            analytic = analytic_model.evaluate(current)
            for field in ("Qc_w", "Qh_w", "V_v", "Pin_w"):
                with self.subTest(field=field, current=current):
                    self.assertAlmostEqual(
                        getattr(numerical, field), getattr(analytic, field), places=10
                    )
            self.assertLess(numerical.relative_energy_residual, 1.0e-10)

    def test_local_current_directions_are_opposite(self) -> None:
        point = solve_constant_couple(numerical_couple(), 0.8, n_cells=10)
        self.assertEqual(point.p_leg.signed_current_a, 0.8)
        self.assertEqual(point.n_leg.signed_current_a, -0.8)
        self.assertGreater(point.p_leg.current_density_a_per_m2, 0.0)
        self.assertLess(point.n_leg.current_density_a_per_m2, 0.0)

    def test_each_leg_and_global_energy_balance_close(self) -> None:
        point = solve_constant_couple(numerical_couple(), 1.3, n_cells=12)
        self.assertAlmostEqual(point.p_leg.local_energy_residual_w, 0.0, places=11)
        self.assertAlmostEqual(point.n_leg.local_energy_residual_w, 0.0, places=11)
        self.assertAlmostEqual(point.energy_residual_w, 0.0, places=11)

    def test_invalid_mesh_and_leg_inputs_are_rejected(self) -> None:
        couple = numerical_couple()
        with self.assertRaises(ValueError):
            solve_constant_couple(couple, 1.0, n_cells=1)
        with self.assertRaises(TypeError):
            solve_constant_couple(couple, 1.0, n_cells=2.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ConstantPropertyLeg(200e-6, 1e-5, 1.5, 1e-3, 0.0)


if __name__ == "__main__":
    unittest.main()
