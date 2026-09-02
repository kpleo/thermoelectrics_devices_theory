"""Tests for the independent conservative 2-D common-mode validation."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from scripts.analysis import validate_independent_2d_common_mode as analysis


class IndependentTwoDimensionalValidationTests(unittest.TestCase):
    def test_common_mode_basis_calculus(self) -> None:
        temperatures = np.array([307.0, 326.0, 348.0])
        step = 1.0e-3
        for basis in analysis.common_mode_bases():
            numerical_mode_derivative = (
                basis.mode(temperatures + step) - basis.mode(temperatures - step)
            ) / (2.0 * step)
            np.testing.assert_allclose(
                temperatures * numerical_mode_derivative,
                basis.gamma(temperatures),
                rtol=2.0e-8,
                atol=2.0e-12,
            )
            numerical_g_derivative = (
                basis.thomson_primitive(temperatures + step)
                - basis.thomson_primitive(temperatures - step)
            ) / (2.0 * step)
            np.testing.assert_allclose(
                numerical_g_derivative,
                basis.gamma(temperatures),
                rtol=2.0e-8,
                atol=2.0e-12,
            )

    def test_electrical_network_conserves_current_and_joule_power(self) -> None:
        p_config, _ = analysis._pair_configs(1.1)
        model = analysis.build_branch_model(p_config, 16, 10)
        electrical = model.electrical
        self.assertLess(electrical.divergence_max_a, 2.0e-12)
        self.assertLess(electrical.terminal_current_mismatch_a, 2.0e-12)
        expected = p_config.signed_current_a * electrical.electrochemical_drop_v
        self.assertAlmostEqual(electrical.joule_power_total_w, expected, places=12)

    def test_adjoint_matches_independent_difference_for_all_gamma_bases(self) -> None:
        for basis in analysis.common_mode_bases():
            case = analysis.run_pair_kernel_case(
                16,
                10,
                common_basis=basis,
                finite_difference_step=1.0e-3,
            )
            self.assertLess(case["relative_difference"], 2.0e-6)
            self.assertLess(case["maximum_base_energy_residual_w"], 2.0e-10)

    def test_constant_shift_null_and_split_pad_interlock(self) -> None:
        shared = analysis.run_constant_shift_case(
            16,
            10,
            cold_p_k=300.0,
            cold_n_k=300.0,
            hot_p_k=350.0,
            hot_n_k=350.0,
        )
        self.assertLess(
            max(abs(shared["increment"][key]) for key in ("Qc_w", "Qh_w", "P_electrical_w")),
            2.0e-12,
        )

        split = analysis.run_constant_shift_case(
            16,
            10,
            cold_p_k=302.0,
            cold_n_k=298.0,
            hot_p_k=350.0,
            hot_n_k=350.0,
        )
        self.assertAlmostEqual(
            split["increment"]["Qc_w"], split["prediction"]["delta_Qc_w"], places=12
        )
        self.assertAlmostEqual(
            split["increment"]["V_power_conjugate_v"],
            split["prediction"]["delta_V_v"],
            places=12,
        )
        self.assertLess(abs(split["errors"]["incremental_energy_interlock_w"]), 2.0e-12)

    def test_nonmonotone_temperature_does_not_break_adjoint(self) -> None:
        clean, _ = analysis.run_nonmonotone_oriented_measure_case()
        self.assertLess(clean["relative_difference"], 2.0e-6)
        for branch in clean["temperature_topology"].values():
            self.assertTrue(branch["peak_is_interior"])
            self.assertTrue(branch["has_positive_and_negative_axial_gradient"])
            self.assertGreaterEqual(branch["axial_gradient_sign_changes"], 1)

    def test_result_has_complete_matrix(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "results/scientific_analysis/independent_2d_common_mode_validation_results.json"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], analysis.SCHEMA)
        self.assertTrue(document["all_checks_passed"])
        matrix = document["cross_family_matrix"]
        self.assertEqual(matrix["design"]["total_cases"], 27)
        self.assertEqual(len(matrix["cases"]), 27)
        self.assertEqual(len(matrix["design"]["mismatch_families"]), 3)
        self.assertEqual(len(matrix["design"]["gamma_bases"]), 3)
        self.assertEqual(len(matrix["design"]["sidewall_coupling_multipliers"]), 3)


if __name__ == "__main__":
    unittest.main()
