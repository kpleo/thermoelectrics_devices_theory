"""Scientific and reproducibility tests for the PbSe/Cr forward constraint."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.analysis import analyze_pbse_device_forward_constraint as module
from scripts.numerical_assertions import assert_nested_close, assert_raster_equivalent


ROOT = Path(__file__).resolve().parents[1]
RESULT_JSON = (
    ROOT
    / "results/scientific_analysis/pbse_device_forward_constraint_results.json"
)
FIGURE = ROOT / "results/scientific_analysis/pbse_device_forward_constraint.png"
SCRIPT = ROOT / "scripts/analysis/analyze_pbse_device_forward_constraint.py"


class PbSeDeviceForwardConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))

    def test_target_is_the_measured_high_temperature_endpoint_without_extrapolation(self) -> None:
        target = self.result["target_condition"]
        self.assertEqual(target["device_id"], "SCI-DTMAX-7P-L6")
        self.assertEqual(target["figure4a_point_index"], 7)
        self.assertAlmostEqual(target["hot_temperature_k"], 362.9980470654064)
        self.assertAlmostEqual(target["delta_t_max_k"], 53.10391221888338)
        self.assertAlmostEqual(target["cold_temperature_k"], 309.894134846523)
        self.assertAlmostEqual(
            target["hot_temperature_k"] - target["cold_temperature_k"],
            target["delta_t_max_k"],
        )
        self.assertTrue(target["all_endpoints_inside_candidate_support"])
        self.assertFalse(target["below_300_k_extrapolation_used"])
        for bounds in target["support_by_scenario"].values():
            self.assertLessEqual(
                bounds["minimum_common_temperature_k"],
                target["cold_temperature_k"],
            )
            self.assertGreaterEqual(
                bounds["maximum_common_temperature_k"],
                target["hot_temperature_k"],
            )

    def test_input_roles_bound_the_interpretation(self) -> None:
        bindings = self.result["input_bindings"]
        self.assertEqual(bindings["figure1_105_source_objects"]["row_count"], 105)
        self.assertEqual(
            bindings["figure_s9_14_raster_candidates"]["row_count"], 14
        )
        self.assertEqual(bindings["figure4a_7_vector_points"]["panel_a_row_count"], 7)
        self.assertEqual(
            bindings["figure1_105_source_objects"]["data_role"],
            "primary_article_figure_derived_source_object_candidate",
        )
        self.assertEqual(
            bindings["figure_s9_14_raster_candidates"]["data_role"],
            "figure_derived_candidate_measured_as_described_in_si",
        )
        self.assertEqual(
            bindings["figure4a_7_vector_points"]["data_role"],
            "figure_derived_measured_as_described_in_main_text",
        )
        boundary = self.result["scope"]
        self.assertTrue(boundary["figure_derived_forward_screening"])
        self.assertFalse(boundary["independent_device_validation_eligible"])
        self.assertFalse(boundary["real_pbse_cr_device_validation"])
        self.assertFalse(boundary["below_300_k_extrapolation_used"])

    def test_reported_specific_contact_conversion_and_nominal_loss_ledger(self) -> None:
        contact = self.result["reported_contact_sensitivity"]
        self.assertAlmostEqual(contact["p_per_interface_ohm"], 0.00065)
        self.assertAlmostEqual(contact["n_per_interface_ohm"], 0.000075)
        self.assertAlmostEqual(contact["per_pair_ohm"], 0.00145)
        self.assertAlmostEqual(contact["seven_pair_series_ohm"], 0.01015)
        self.assertEqual(contact["interfaces_per_leg"], 2)
        self.assertEqual(contact["eta_to_cold"], 0.5)
        self.assertEqual(
            contact["role"],
            "explicit_sensitivity_not_resolved_device_contact_measurement",
        )

        nominal = self.result["forward_results_by_source_direction"]["nominal"]
        bulk = nominal["bulk_optimum"]
        corrected = nominal["contact_corrected_optimum"]
        self.assertAlmostEqual(bulk["bulk_Qc_w"], 0.7050432946413303, places=8)
        self.assertAlmostEqual(bulk["current_a"], 2.8662609234537975, places=7)
        self.assertAlmostEqual(
            corrected["Qc_after_contact_w"], 0.6641530436722118, places=8
        )
        self.assertAlmostEqual(corrected["current_a"], 2.8106854181890806, places=7)
        self.assertAlmostEqual(
            nominal["equivalent_missing_conductance_w_per_k"],
            0.012506668829496213,
            places=10,
        )
        self.assertAlmostEqual(
            nominal["equivalent_missing_to_bulk_conductance_ratio"],
            0.8656274667208326,
            places=8,
        )
        ledger = self.result["nominal_forward_loss_ledger"]
        self.assertAlmostEqual(
            ledger["reported_contact_sensitivity_effect_on_reoptimized_maximum_w"],
            -0.04089025096911847,
            places=8,
        )
        self.assertAlmostEqual(
            ledger["model_equivalent_unresolved_loss_to_reach_zero_load_w"],
            -corrected["Qc_after_contact_w"],
        )
        self.assertFalse(ledger["unique_physical_attribution_allowed"])
        self.assertFalse(
            self.result["scope"][
                "equivalent_missing_conductance_uniquely_attributed"
            ]
        )

    def test_directional_source_method_stress_does_not_close_the_gap(self) -> None:
        scenarios = self.result["forward_results_by_source_direction"]
        conservative = scenarios["conservative_direction"]
        nominal = scenarios["nominal"]
        favorable = scenarios["favorable_direction"]
        q_values = [
            conservative["contact_corrected_optimum"]["Qc_after_contact_w"],
            nominal["contact_corrected_optimum"]["Qc_after_contact_w"],
            favorable["contact_corrected_optimum"]["Qc_after_contact_w"],
        ]
        self.assertGreater(q_values[0], 0.35)
        self.assertLess(q_values[0], q_values[1])
        self.assertLess(q_values[1], q_values[2])
        self.assertAlmostEqual(q_values[0], 0.36089204043035655, places=8)
        self.assertAlmostEqual(q_values[2], 0.9766226121932771, places=8)

        bounds = self.result["source_method_bounds"]
        self.assertEqual(
            bounds["conservative_direction"],
            "|S|*0.95, sigma*0.95, kappa*1.15",
        )
        self.assertFalse(bounds["statistical_confidence_interval"])
        self.assertFalse(bounds["digitization_uncertainty_combined"])
        self.assertFalse(bounds["correlations_or_joint_distribution_assumed"])

        envelope = self.result[
            "eta_and_source_direction_equivalent_loss_envelope"
        ]
        for records in envelope.values():
            self.assertEqual(len(records), 21)
            self.assertEqual(records[0]["eta_to_cold"], 0.0)
            self.assertEqual(records[-1]["eta_to_cold"], 1.0)
            self.assertGreater(
                records[0]["Qc_after_contact_w"],
                records[-1]["Qc_after_contact_w"],
            )

    def test_pair_coordinate_shows_geometry_is_only_a_small_correction(self) -> None:
        design = self.result["pair_design_coordinates_300_to_573_k"]
        points = design["points"]
        self.assertEqual(len(points), 7)
        self.assertEqual(
            [point["nominal_temperature_k"] for point in points],
            [300.0, 323.0, 373.0, 423.0, 473.0, 523.0, 573.0],
        )
        self.assertAlmostEqual(points[0]["zeta_geometry_ceiling"], 0.6522574123404413)
        self.assertAlmostEqual(points[0]["zeta_equal_geometry"], 0.6517817268406634)
        self.assertAlmostEqual(points[0]["optimal_gp_over_gn"], 1.0555566939978152)
        summary = design["summary"]
        self.assertGreaterEqual(summary["equal_geometry_retention_range"][0], 0.9719)
        self.assertLessEqual(summary["equal_geometry_retention_range"][1], 1.0)
        self.assertLess(summary["maximum_ceiling_gain_from_geometry_fraction"], 0.029)
        self.assertAlmostEqual(
            summary["optimal_gp_over_gn_range"][1], 1.4108725043970154
        )

    def test_solver_resistivity_is_reciprocal_of_the_sigma_pchip(self) -> None:
        inputs = module.load_inputs()
        target = inputs["target"]
        hot = float(target["hot_side_temperature_k"])
        cold = hot - float(target["delta_t_max_k"])
        couple = module.build_couple(inputs, module.SCENARIOS["nominal"], cold, hot)
        temperature = 0.5 * (cold + hot)
        sigma_t, sigma_values = module._fig1_arrays(inputs["p_sigma"])
        sigma_law = module.PchipTemperatureProperty(sigma_t, sigma_values)
        expected_rho = 1.0 / float(sigma_law.evaluate([temperature])[0])
        actual_rho = float(
            couple.p_leg.electrical_resistivity.evaluate([temperature])[0]
        )
        self.assertAlmostEqual(actual_rho, expected_rho, places=16)
        step = 1.0e-3
        finite_difference = (
            float(
                couple.p_leg.electrical_resistivity.evaluate(
                    [temperature + step]
                )[0]
            )
            - float(
                couple.p_leg.electrical_resistivity.evaluate(
                    [temperature - step]
                )[0]
            )
        ) / (2.0 * step)
        analytic = float(
            couple.p_leg.electrical_resistivity.derivative([temperature])[0]
        )
        np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-8, atol=0.0)

    def test_solver_refinement_energy_and_curve_checks_are_tight(self) -> None:
        verification = self.result["verification"]
        self.assertTrue(verification["all_current_curve_points_accepted"])
        self.assertEqual(verification["current_curve_point_count_per_scenario"], 141)
        self.assertEqual(verification["current_curve_range_a"], [0.0, 3.5])
        self.assertLess(
            verification["maximum_contact_Qc_change_under_tight_solver_refinement_w"],
            1.0e-8,
        )
        self.assertLess(
            verification["maximum_absolute_tight_module_energy_residual_w"],
            2.0e-14,
        )
        self.assertLess(
            verification[
                "maximum_eta_half_curve_interpolation_vs_direct_Qc_error_w"
            ],
            1.0e-4,
        )
        for scenario in self.result["forward_results_by_source_direction"].values():
            point = scenario["contact_corrected_optimum"]
            self.assertAlmostEqual(
                point["Qh_after_contact_w"] - point["Qc_after_contact_w"],
                point["input_power_w"],
                places=12,
            )

    def test_committed_artifacts_and_rebuild_are_deterministic(self) -> None:
        outputs = self.result["outputs"]
        self.assertEqual(outputs["analysis_script_sha256"], module.file_sha256(SCRIPT))
        self.assertEqual(outputs["figure_sha256"], module.file_sha256(FIGURE))
        with Image.open(FIGURE) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertGreaterEqual(image.width, 3000)
            self.assertGreaterEqual(image.height, 1900)

        with tempfile.TemporaryDirectory() as temporary:
            rebuilt_json = Path(temporary) / "result.json"
            rebuilt_figure = Path(temporary) / "figure.png"
            rebuilt = module.run_analysis(rebuilt_json, rebuilt_figure)
            cross_platform = os.environ.get("CROSS_PLATFORM_REBUILD") == "1"
            if cross_platform:
                assert_nested_close(
                    self,
                    rebuilt["forward_results_by_source_direction"],
                    self.result["forward_results_by_source_direction"],
                )
                assert_nested_close(
                    self,
                    rebuilt["pair_design_coordinates_300_to_573_k"],
                    self.result["pair_design_coordinates_300_to_573_k"],
                )
                assert_raster_equivalent(self, rebuilt_figure, FIGURE)
            else:
                self.assertEqual(
                    rebuilt["forward_results_by_source_direction"],
                    self.result["forward_results_by_source_direction"],
                )
                self.assertEqual(
                    rebuilt["pair_design_coordinates_300_to_573_k"],
                    self.result["pair_design_coordinates_300_to_573_k"],
                )
                self.assertEqual(
                    module.file_sha256(rebuilt_figure), module.file_sha256(FIGURE)
                )
            rebuilt_document = json.loads(rebuilt_json.read_text(encoding="utf-8"))
            scrubbed_committed = copy.deepcopy(self.result)
            scrubbed_rebuilt = copy.deepcopy(rebuilt_document)
            scrubbed_committed["outputs"]["figure"] = "SCRUBBED"
            scrubbed_rebuilt["outputs"]["figure"] = "SCRUBBED"
            if cross_platform:
                scrubbed_committed["outputs"]["figure_sha256"] = "SCRUBBED"
                scrubbed_rebuilt["outputs"]["figure_sha256"] = "SCRUBBED"
                assert_nested_close(self, scrubbed_rebuilt, scrubbed_committed)
            else:
                self.assertEqual(scrubbed_rebuilt, scrubbed_committed)


if __name__ == "__main__":
    unittest.main()
