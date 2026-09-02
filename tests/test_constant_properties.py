"""Regression tests for the constant-property analytic reference.

The tests use :mod:`unittest` so they run in a bare Python installation, while
remaining directly discoverable by pytest when that optional runner is present.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

from scripts.tec_1d_solver import (
    COP,
    I_opt_for_max_Qc,
    Pin,
    Qc,
    Qh,
    V,
    ConstantPropertyCouple,
    energy_residual,
)


def reference_couple() -> ConstantPropertyCouple:
    return ConstantPropertyCouple(
        seebeck_p_v_per_k=220.0e-6,
        seebeck_n_v_per_k=-180.0e-6,
        electrical_resistance_ohm=0.1,
        thermal_conductance_w_per_k=0.003,
        cold_temperature_k=300.0,
        hot_temperature_k=310.0,
    )


class ConstantPropertyReferenceTests(unittest.TestCase):
    def test_named_api_matches_hand_calculation(self) -> None:
        couple = reference_couple()
        current = 1.0
        alpha = 400.0e-6
        delta_t = 10.0

        self.assertAlmostEqual(Qc(couple, current), alpha * 300.0 - 0.05 - 0.03)
        self.assertAlmostEqual(Qh(couple, current), alpha * 310.0 + 0.05 - 0.03)
        self.assertAlmostEqual(V(couple, current), 0.1 + alpha * delta_t)
        self.assertAlmostEqual(Pin(couple, current), current * V(couple, current))
        self.assertAlmostEqual(COP(couple, current), Qc(couple, current) / Pin(couple, current))

    def test_fixed_delta_seebeck_is_invariant_to_common_mode_shift(self) -> None:
        baseline = reference_couple()
        shift = 80.0e-6
        shifted = ConstantPropertyCouple(
            seebeck_p_v_per_k=baseline.seebeck_p_v_per_k + shift,
            seebeck_n_v_per_k=baseline.seebeck_n_v_per_k + shift,
            electrical_resistance_ohm=baseline.electrical_resistance_ohm,
            thermal_conductance_w_per_k=baseline.thermal_conductance_w_per_k,
            cold_temperature_k=baseline.cold_temperature_k,
            hot_temperature_k=baseline.hot_temperature_k,
        )

        self.assertAlmostEqual(
            baseline.delta_seebeck_v_per_k, shifted.delta_seebeck_v_per_k
        )
        self.assertNotEqual(
            baseline.common_mode_seebeck_v_per_k,
            shifted.common_mode_seebeck_v_per_k,
        )
        self.assertEqual(baseline.evaluate(0.9), shifted.evaluate(0.9))
        self.assertEqual(
            I_opt_for_max_Qc(baseline), I_opt_for_max_Qc(shifted)
        )

    def test_terminal_energy_balance_closes_for_both_current_directions(self) -> None:
        couple = reference_couple()
        for current in (-2.0, -0.4, 0.0, 0.8, 2.0):
            with self.subTest(current=current):
                point = couple.evaluate(current)
                self.assertAlmostEqual(point.Qh_w - point.Qc_w, point.Pin_w, places=14)
                self.assertAlmostEqual(energy_residual(couple, current), 0.0, places=14)
                self.assertLess(point.relative_energy_residual, 1.0e-14)

    def test_zero_current_is_open_circuit_with_only_conductive_backflow(self) -> None:
        couple = reference_couple()
        point = couple.evaluate(0.0)
        conductive_backflow = -couple.thermal_conductance_w_per_k * couple.delta_temperature_k

        self.assertAlmostEqual(point.Qc_w, conductive_backflow)
        self.assertAlmostEqual(point.Qh_w, conductive_backflow)
        self.assertAlmostEqual(
            point.V_v,
            couple.delta_seebeck_v_per_k * couple.delta_temperature_k,
        )
        self.assertEqual(point.Pin_w, 0.0)
        self.assertIsNone(point.COP)

    def test_current_reversal_flips_peltier_but_not_joule_or_conduction(self) -> None:
        couple = reference_couple()
        current = 0.7
        forward = couple.evaluate(current)
        reverse = couple.evaluate(-current)
        expected_peltier_difference = (
            2.0
            * couple.delta_seebeck_v_per_k
            * current
            * couple.cold_temperature_k
        )
        expected_non_peltier_sum = (
            -current * current * couple.electrical_resistance_ohm
            - 2.0
            * couple.thermal_conductance_w_per_k
            * couple.delta_temperature_k
        )

        self.assertAlmostEqual(forward.Qc_w - reverse.Qc_w, expected_peltier_difference)
        self.assertAlmostEqual(forward.Qc_w + reverse.Qc_w, expected_non_peltier_sum)
        self.assertIsNone(reverse.COP)

    def test_analytic_current_is_global_maximum_of_concave_qc_curve(self) -> None:
        couple = reference_couple()
        optimum = I_opt_for_max_Qc(couple)
        self.assertAlmostEqual(
            optimum,
            couple.delta_seebeck_v_per_k
            * couple.cold_temperature_k
            / couple.electrical_resistance_ohm,
        )
        q_optimum = Qc(couple, optimum)
        self.assertGreater(q_optimum, Qc(couple, optimum - 0.25))
        self.assertGreater(q_optimum, Qc(couple, optimum + 0.25))

    def test_invalid_inputs_are_rejected(self) -> None:
        valid = dict(
            seebeck_p_v_per_k=200.0e-6,
            seebeck_n_v_per_k=-200.0e-6,
            electrical_resistance_ohm=0.1,
            thermal_conductance_w_per_k=0.003,
            cold_temperature_k=300.0,
            hot_temperature_k=310.0,
        )
        for field, value in (
            ("electrical_resistance_ohm", 0.0),
            ("thermal_conductance_w_per_k", -0.1),
            ("cold_temperature_k", 0.0),
            ("seebeck_p_v_per_k", float("nan")),
        ):
            with self.subTest(field=field):
                parameters = dict(valid)
                parameters[field] = value
                with self.assertRaises(ValueError):
                    ConstantPropertyCouple(**parameters)

        reversed_temperatures = dict(valid)
        reversed_temperatures["hot_temperature_k"] = 299.0
        with self.assertRaises(ValueError):
            ConstantPropertyCouple(**reversed_temperatures)
        with self.assertRaises(ValueError):
            reference_couple().evaluate(float("inf"))

    def test_cli_example_emits_machine_readable_energy_closed_result(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "scripts.tec_1d_solver"],
            check=True,
            capture_output=True,
            text=True,
        )
        output = json.loads(completed.stdout)
        point = output["operating_point"]

        self.assertEqual(output["units"]["heat_and_power"], "W")
        self.assertAlmostEqual(
            point["current_a"], output["I_opt_for_max_Qc_a"]
        )
        self.assertGreater(point["Qc_w"], 0.0)
        self.assertGreater(point["COP"], 0.0)
        self.assertAlmostEqual(point["energy_residual_w"], 0.0, places=14)


if __name__ == "__main__":
    unittest.main()
