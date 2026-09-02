"""Self-tests for the independent temperature-dependent exact references."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.analysis.temperature_dependent_exact_benchmarks import (
    ManufacturedLinearTemperatureCase,
    ZeroCurrentLinearKappaCase,
    benchmark_report,
)


class ManufacturedThomsonReferenceTests(unittest.TestCase):
    def test_joule_and_thomson_terms_are_nonzero_and_cancel(self) -> None:
        case = ManufacturedLinearTemperatureCase()
        x = np.linspace(0.0, case.length_m, 17)
        terms = case.expanded_balance_terms_w_per_m3(x)

        joule = np.asarray(terms["joule"])
        negative_thomson = np.asarray(terms["negative_thomson"])
        residual = np.asarray(terms["sum"])
        self.assertTrue(np.all(joule > 0.0))
        np.testing.assert_allclose(
            joule, -negative_thomson, rtol=5.0e-16, atol=1.0e-9
        )
        self.assertLess(float(np.max(np.abs(residual))), 1.0e-6)

    def test_exact_terminal_energy_ledger_uses_signed_current(self) -> None:
        case = ManufacturedLinearTemperatureCase()
        terminal = case.terminal_reference()
        delta_q = terminal["hot_heat_rate_w"] - terminal["cold_heat_rate_w"]
        expected_power = case.signed_current_a * (
            -terminal["hot_minus_cold_potential_v"]
        )

        self.assertAlmostEqual(delta_q, expected_power, places=15)
        self.assertAlmostEqual(
            terminal["local_energy_residual_w"], 0.0, places=15
        )
        self.assertGreater(
            case.resistivity_ohm_m(case.cold_temperature_k), 0.0
        )
        self.assertGreater(
            case.resistivity_ohm_m(case.hot_temperature_k), 0.0
        )


class ZeroCurrentVariableKappaReferenceTests(unittest.TestCase):
    def test_implicit_conductivity_integral_defines_exact_profile(self) -> None:
        case = ZeroCurrentLinearKappaCase()
        x = np.linspace(0.0, case.length_m, 31)
        temperature = np.asarray(case.temperature_k(x))
        rise = temperature - case.cold_temperature_k
        integrated_to_x = (
            case.kappa_at_cold_w_per_m_k * rise
            + 0.5 * case.dkappa_dtemperature_w_per_m_k2 * rise**2
        )
        expected = case.integrated_kappa_w_per_m * x / case.length_m

        np.testing.assert_allclose(integrated_to_x, expected, rtol=2.0e-15, atol=2.0e-14)
        self.assertAlmostEqual(temperature[0], case.cold_temperature_k)
        self.assertAlmostEqual(temperature[-1], case.hot_temperature_k)

    def test_endpoint_flux_and_open_circuit_voltage_have_correct_sign(self) -> None:
        case = ZeroCurrentLinearKappaCase()
        terminal = case.terminal_reference()
        expected_phi = -(
            case.seebeck_at_cold_v_per_k * case.delta_temperature_k
            + 0.5
            * case.dseebeck_dtemperature_v_per_k2
            * case.delta_temperature_k**2
        )

        self.assertLess(terminal["cold_heat_flux_w_per_m2"], 0.0)
        self.assertEqual(
            terminal["cold_heat_flux_w_per_m2"],
            terminal["hot_heat_flux_w_per_m2"],
        )
        self.assertAlmostEqual(
            terminal["hot_minus_cold_potential_v"], expected_phi, places=15
        )
        self.assertEqual(terminal["electrical_power_absorbed_w"], 0.0)

    def test_machine_readable_reference_report_is_self_consistent(self) -> None:
        report = benchmark_report()
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["checks"].values()))
        self.assertIn("PbSe/Cr material data", report["scope"])
        self.assertIn("outside this benchmark", report["scope"])


if __name__ == "__main__":
    unittest.main()
