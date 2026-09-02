"""Physics, topology, and reproducibility tests for the transfer-kernel result."""

from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import numpy as np

from scripts.analysis import analyze_common_mode_transfer_kernel as module


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT / "results/scientific_analysis/common_mode_transfer_kernel_results.json"
)
FIGURE_STEM = ROOT / "results/scientific_analysis/common_mode_transfer_kernel"


class CommonModeTransferKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_schema_and_scope_are_explicit(self) -> None:
        self.assertEqual(
            self.result["schema_version"], "common_mode_transfer_kernel/v1"
        )
        boundary = self.result["scope"]
        self.assertTrue(boundary["first_order_theorem"])
        self.assertTrue(boundary["constant_property_closed_form"])
        self.assertFalse(boundary["device_validation"])
        self.assertFalse(boundary["ten_percent_pbse_effect_demonstrated"])

    def test_adjoint_matches_independent_closed_form(self) -> None:
        benchmark = self.result["analytic_and_symmetry_validation"]
        self.assertLess(benchmark["adjoint_relative_error_to_closed_form"], 1.0e-10)
        self.assertLess(
            benchmark["central_difference_convergence"][-1][
                "relative_error_to_closed_form"
            ],
            2.0e-8,
        )
        self.assertTrue(
            math.isclose(
                benchmark["closed_form"]["derivative_w"],
                -4.37962962962963e-5,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
        )

    def test_dimensionless_breaking_number_closes_exactly(self) -> None:
        closed = self.result["analytic_and_symmetry_validation"]["closed_form"]
        self.assertTrue(
            math.isclose(
                closed["breaking_number_signed"],
                closed["derivative_over_cold_peltier_scale"],
                rel_tol=2.0e-15,
                abs_tol=1.0e-18,
            )
        )
        self.assertNotIn(
            "A_S", self.result["theory"]["dimensionless_breaking_number"]
        )

    def test_transfer_matching_not_raw_identity_is_the_zero_condition(self) -> None:
        benchmark = self.result["analytic_and_symmetry_validation"]
        identical = benchmark["exact_identical_branch_control"]
        matched = benchmark["nonidentical_but_matched_R_over_K_control"]
        self.assertEqual(identical["kernel_derivative_w"], 0.0)
        self.assertLess(abs(matched["kernel_derivative_w"]), 1.0e-15)
        self.assertLess(abs(matched["delta_g_k"]), 1.0e-12)
        self.assertLess(
            matched["maximum_collection_function_difference"], 1.0e-12
        )

    def test_finite_thermal_contacts_follow_schur_complement(self) -> None:
        contact = self.result["finite_contact_boundary_validation"]
        self.assertLess(contact["relative_validation_error"], 1.0e-5)
        self.assertGreater(
            abs(contact["thermal_contact_dressing_factor"] - 1.0), 1.0e-4
        )
        direct = contact["fixed_current_zero_direct_derivative_terms"]
        self.assertTrue(all(direct.values()))
        jacobian = np.asarray(contact["contact_residual_jacobian"], dtype=float)
        self.assertGreater(abs(float(np.linalg.det(jacobian))), 1.0e-3)

    def test_nonisothermal_port_exposes_constant_common_mode(self) -> None:
        topology = self.result["nonisothermal_topology_validation"]
        self.assertEqual(topology["distributed_gamma_v_per_k"], 0.0)
        self.assertLess(topology["absolute_validation_error_w"], 1.0e-12)
        self.assertTrue(
            math.isclose(
                topology["exact_direct_cold_peltier_derivative_w"],
                -1.25e-4,
                rel_tol=1.0e-14,
                abs_tol=1.0e-16,
            )
        )

    def test_pbse_adjoint_matches_production_central_difference(self) -> None:
        pbse = self.result["pbse_candidate_application"]
        kernel = pbse["first_order_kernel"]
        self.assertLess(kernel["relative_adjoint_to_central_error"], 2.0e-5)
        self.assertGreater(kernel["p_leg"]["module_total_port_derivative_w"], 0.0)
        self.assertLess(kernel["n_leg"]["module_total_port_derivative_w"], 0.0)
        self.assertAlmostEqual(kernel["module_derivative_w"], 0.003481945, places=8)
        self.assertAlmostEqual(
            kernel["p_leg"]["direct_endpoint_peltier_derivative_w_per_pair"],
            0.0,
            places=15,
        )
        self.assertAlmostEqual(
            kernel["n_leg"]["direct_endpoint_peltier_derivative_w_per_pair"],
            0.0,
            places=15,
        )

    def test_pbse_full_profile_is_small_and_nonlinear(self) -> None:
        pbse = self.result["pbse_candidate_application"]
        finite = pbse["finite_unit_profile"]
        self.assertAlmostEqual(
            finite["original_minus_flattened_Qc_w"], 0.003090779, places=8
        )
        self.assertGreater(finite["response_fraction_of_baseline_Qc"], 0.004)
        self.assertLess(finite["response_fraction_of_baseline_Qc"], 0.006)
        self.assertLess(finite["nonlinear_departure_from_first_order_fraction"], -0.1)
        self.assertGreater(
            finite["nonlinear_departure_from_first_order_fraction"], -0.13
        )

    def test_pbse_thresholds_separate_one_and_ten_percent(self) -> None:
        thresholds = self.result["pbse_candidate_application"]["thresholds"]
        one_percent = thresholds["observed_full_solver_current_for_one_percent_a"]
        self.assertIsNotNone(one_percent)
        self.assertGreater(one_percent, 3.20)
        self.assertLess(one_percent, 3.27)
        self.assertIsNone(
            thresholds["observed_full_solver_current_for_ten_percent_a"]
        )
        self.assertLess(
            thresholds["maximum_observed_fraction_in_validated_current_range"],
            0.02,
        )
        self.assertGreater(
            thresholds["gamma_profile_multipliers_at_reference_current"][
                "ten_percent_local_linear"
            ],
            15.0,
        )
        asymmetry = thresholds["R_over_K_asymmetry_scale"]
        self.assertLess(
            asymmetry["required_absolute_epsilon_R_over_K_for_one_percent"],
            1.0,
        )
        self.assertGreater(
            asymmetry["required_absolute_epsilon_R_over_K_for_ten_percent"],
            1.0,
        )
        self.assertFalse(
            thresholds["transfer_cancellation_scale"][
                "ten_percent_possible_at_fixed_branch_norm"
            ]
        )

    def test_source_data_integrates_to_reported_pbse_kernel(self) -> None:
        pbse = self.result["pbse_candidate_application"]
        source = pbse["source_data_for_figure"]
        cumulative = np.asarray(source["net_cumulative_derivative_w"], dtype=float)
        self.assertTrue(
            math.isclose(
                float(cumulative[-1]),
                pbse["first_order_kernel"]["module_derivative_w"],
                rel_tol=5.0e-5,
                abs_tol=1.0e-9,
            )
        )
        self.assertEqual(
            len(source["normalized_coordinate"]),
            len(source["p_collection_function"]),
        )

    def test_figure_outputs_exist_and_svg_text_is_editable(self) -> None:
        for suffix in (".svg", ".pdf", ".png"):
            path = FIGURE_STEM.with_suffix(suffix)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
        svg = FIGURE_STEM.with_suffix(".svg").read_text(encoding="utf-8")
        self.assertIn("<text", svg)
        self.assertNotIn("fonttype", svg.lower())
        recorded_hashes = {
            row["locator"]: row["sha256"]
            for row in self.result["outputs"]["figures"]
        }
        for suffix in (".svg", ".pdf", ".png"):
            path = FIGURE_STEM.with_suffix(suffix)
            locator = str(path.relative_to(ROOT))
            self.assertEqual(recorded_hashes[locator], module._file_sha256(path))

    def test_script_can_build_the_analytic_fixture_without_saved_json(self) -> None:
        fixture = module._analytic_fixture(symmetric=True)
        point = module._tight_couple_solve(fixture, 0.5, output_points=201)
        response = module.first_order_port_response(
            fixture,
            point,
            mode_value=lambda temperature: 1.0e-6 * (temperature - 300.0),
            gamma_value=lambda temperature: 1.0e-6 * temperature,
            port="cold",
        )
        self.assertLess(abs(response["pair_total_port_derivative_w"]), 1.0e-15)


if __name__ == "__main__":
    unittest.main()
