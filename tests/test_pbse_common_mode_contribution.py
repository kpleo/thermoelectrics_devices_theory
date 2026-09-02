"""Physics and reproducibility tests for the PbSe common-mode experiment."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.analysis import analyze_pbse_common_mode_contribution as module
from scripts.numerical_assertions import assert_nested_close, assert_raster_equivalent


ROOT = Path(__file__).resolve().parents[1]
RESULT_JSON = (
    ROOT
    / "results/scientific_analysis/pbse_common_mode_contribution_results.json"
)
FIGURE = ROOT / "results/scientific_analysis/pbse_common_mode_contribution.png"
SCRIPT = ROOT / "scripts/analysis/analyze_pbse_common_mode_contribution.py"


class PbSeCommonModeContributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))

    def test_target_and_scope_remain_candidate_level(self) -> None:
        target = self.result["target_condition"]
        self.assertEqual(target["device_id"], "SCI-DTMAX-7P-L6")
        self.assertEqual(target["figure4a_point_index"], 7)
        self.assertAlmostEqual(target["hot_temperature_k"], 362.9980470654064)
        self.assertAlmostEqual(target["cold_temperature_k"], 309.894134846523)
        self.assertAlmostEqual(target["delta_t_k"], 53.10391221888338)
        self.assertTrue(target["endpoint_temperatures_fixed_in_both_models"])
        self.assertTrue(
            target["all_detailed_control_temperature_fields_within_source_support"]
        )
        self.assertGreaterEqual(
            target["solved_temperature_field_range_across_detailed_controls_k"][0],
            target["common_property_support_k"]["minimum_temperature_k"],
        )
        self.assertLessEqual(
            target["solved_temperature_field_range_across_detailed_controls_k"][1],
            target["common_property_support_k"]["maximum_temperature_k"],
        )

        boundary = self.result["scope"]
        self.assertTrue(boundary["figure_derived_source_candidate_scenario_screen"])
        self.assertFalse(boundary["independent_device_validation_eligible"])
        self.assertFalse(boundary["real_pbse_cr_device_validation"])
        self.assertFalse(boundary["counterfactual_flattened_material_was_measured"])
        self.assertFalse(
            boundary["common_mode_contribution_uniquely_explains_device_gap"]
        )
        self.assertFalse(boundary["below_300_k_extrapolation_used"])
        self.assertFalse(boundary["statistical_confidence_interval"])

    def test_intervention_preserves_alpha_transport_geometry_and_cold_anchor(self) -> None:
        intervention = self.result["controlled_intervention"]
        self.assertEqual(intervention["constant_reference_choice"], "M(Tc)")
        self.assertAlmostEqual(
            intervention["constant_reference_v_per_k"],
            2.597909474386578e-06,
            places=18,
        )
        self.assertTrue(intervention["alpha_T_preserved_pointwise"])
        self.assertTrue(intervention["cold_p_and_n_Seebeck_values_anchored"])
        self.assertTrue(intervention["rho_T_preserved_by_shared_property_objects"])
        self.assertTrue(intervention["kappa_T_preserved_by_shared_property_objects"])
        self.assertTrue(intervention["geometry_preserved"])
        self.assertTrue(intervention["electrical_contacts_preserved"])
        self.assertTrue(intervention["thermal_boundary_temperatures_preserved"])

        summary = intervention["property_coordinate_summary"]
        self.assertEqual(
            summary["maximum_absolute_alpha_intervention_error_v_per_k"], 0.0
        )
        self.assertEqual(summary["maximum_flat_common_mode_deviation_v_per_k"], 0.0)
        self.assertEqual(summary["maximum_rho_intervention_error_ohm_m"], 0.0)
        self.assertEqual(
            summary["maximum_kappa_intervention_error_w_per_m_k"], 0.0
        )
        self.assertEqual(summary["cold_anchor_p_error_v_per_k"], 0.0)
        self.assertEqual(summary["cold_anchor_n_error_v_per_k"], 0.0)
        self.assertAlmostEqual(
            summary["M_change_cold_to_hot_v_per_k"],
            3.788626654863063e-06,
            places=18,
        )
        self.assertAlmostEqual(
            summary["Gamma_temperature_mean_v_per_k"],
            2.4020419248628548e-05,
            places=16,
        )

    def test_real_common_mode_changes_capacity_by_only_three_milliwatts(self) -> None:
        comparison = self.result["optimized_forward_comparison"]
        original = comparison["original"]["contact_corrected_optimum"]
        flattened = comparison["flattened_common_mode"][
            "contact_corrected_optimum"
        ]
        effect = comparison["original_minus_flattened"]

        self.assertAlmostEqual(original["Qc_after_contact_w"], 0.6641530436722118)
        self.assertAlmostEqual(original["current_a"], 2.8106854181890806)
        self.assertAlmostEqual(flattened["Qc_after_contact_w"], 0.6610861653663038)
        self.assertAlmostEqual(flattened["current_a"], 2.801083194176617)
        self.assertAlmostEqual(
            effect["contact_corrected_Qc_max_w"],
            0.0030668783059080162,
            places=12,
        )
        self.assertAlmostEqual(
            effect["fraction_of_nominal_forward_residual_capacity"],
            0.004617728300921034,
            places=12,
        )
        self.assertLess(
            effect["fraction_of_nominal_forward_residual_capacity"], 0.005
        )
        self.assertAlmostEqual(
            effect["equivalent_parallel_conductance_w_per_k"],
            5.775239860421161e-05,
            places=14,
        )
        self.assertGreater(effect["maximum_COP"], 0.0)
        self.assertLess(effect["COP_at_capacity_optimum"], 0.0)

    def test_leg_common_mode_sources_cancel_but_cold_heat_does_not(self) -> None:
        mechanism = self.result["fixed_current_mechanism"]
        original = mechanism["original"]
        flattened = mechanism["flattened_common_mode"]
        ledger = original["thomson_and_heat_ledger"]
        p_common = ledger["p_leg"][
            "integrated_active_common_mode_thomson_source_w"
        ]
        n_common = ledger["n_leg"][
            "integrated_active_common_mode_thomson_source_w"
        ]
        net_common = ledger["module_net"][
            "integrated_active_common_mode_thomson_source_w"
        ]
        self.assertAlmostEqual(p_common, -0.025096745402564884, places=10)
        self.assertAlmostEqual(n_common, 0.025096749767350178, places=10)
        self.assertLess(abs(net_common), 1.0e-8)
        self.assertEqual(
            flattened["thomson_and_heat_ledger"]["module_net"][
                "integrated_active_common_mode_thomson_source_w"
            ],
            0.0,
        )

        partition = mechanism["cold_heat_partition_response"]
        self.assertGreater(partition["p_leg_original_minus_flattened_w"], 0.020)
        self.assertLess(partition["n_leg_original_minus_flattened_w"], -0.017)
        self.assertAlmostEqual(
            partition["module_original_minus_flattened_w"],
            0.0030907786299619877,
            places=12,
        )

        peltier = mechanism["peltier_boundary_response"]
        cold_peltier = peltier["cold_end_original_minus_flattened_w"]
        self.assertEqual(cold_peltier["p_leg"], 0.0)
        self.assertEqual(cold_peltier["n_leg"], 0.0)
        self.assertEqual(cold_peltier["module_net"], 0.0)
        hot_peltier = peltier["hot_end_original_minus_flattened_w"]
        self.assertAlmostEqual(hot_peltier["p_leg"], 0.02705804303511705)
        self.assertAlmostEqual(hot_peltier["n_leg"], -0.027058043130137155)
        self.assertLess(abs(hot_peltier["module_net"]), 1.0e-9)

        parity = mechanism["signed_current_parity_response"]
        self.assertAlmostEqual(parity["odd_component_w"], 0.0036697347358145004)
        self.assertAlmostEqual(parity["even_component_w"], -0.0005789561058525127)
        self.assertNotAlmostEqual(
            parity["positive_current_effect_w"],
            -parity["negative_current_effect_w"],
        )
        self.assertGreater(
            mechanism["original_minus_flattened"][
                "maximum_temperature_profile_difference_k"
            ],
            0.6,
        )
        self.assertLess(
            mechanism["original_minus_flattened"][
                "maximum_temperature_profile_difference_k"
            ],
            0.7,
        )

    def test_reverse_zero_and_constant_reference_controls_close(self) -> None:
        forward_effect = self.result["fixed_current_mechanism"][
            "original_minus_flattened"
        ]["Qc_after_contact_w"]
        reverse = self.result["reverse_current_control"]
        self.assertAlmostEqual(reverse["current_a"], -2.8106854181890806)
        self.assertTrue(reverse["effect_has_opposite_sign_to_forward"])
        self.assertLess(reverse["original_minus_flattened_Qc_w"], 0.0)
        self.assertLess(
            reverse["original_minus_flattened_Qc_w"] * forward_effect,
            0.0,
        )

        zero = self.result["zero_current_control"]
        self.assertEqual(zero["original_minus_flattened_Qc_w"], 0.0)
        self.assertEqual(zero["maximum_temperature_profile_difference_k"], 0.0)
        self.assertLess(abs(zero["terminal_voltage_difference_v"]), 5.0e-14)

        reference = self.result["constant_reference_invariance_control"]
        self.assertLess(abs(reference["Qc_difference_w"]), 1.0e-14)
        self.assertLess(abs(reference["terminal_voltage_difference_v"]), 1.0e-14)
        self.assertEqual(reference["maximum_temperature_profile_difference_k"], 0.0)

    def test_current_sweep_and_numerical_closure(self) -> None:
        sweep = self.result["signed_current_sweep"]
        self.assertEqual(len(sweep), 141)
        self.assertEqual(sweep[0]["current_a"], -3.5)
        self.assertEqual(sweep[70]["current_a"], 0.0)
        self.assertEqual(sweep[-1]["current_a"], 3.5)
        self.assertEqual(sweep[70]["original_minus_flattened_Qc_w"], 0.0)
        self.assertLess(sweep[0]["original_minus_flattened_Qc_w"], 0.0)
        self.assertGreater(sweep[-1]["original_minus_flattened_Qc_w"], 0.0)

        verification = self.result["verification"]
        self.assertTrue(
            verification["all_detailed_temperature_fields_inside_source_support"]
        )
        self.assertLess(verification["maximum_tight_refinement_Qc_error_w"], 2.0e-9)
        self.assertLess(
            verification["maximum_detailed_module_energy_residual_w"], 2.0e-14
        )
        self.assertLess(
            verification["maximum_thomson_component_closure_error_w"], 1.0e-14
        )
        self.assertLess(
            verification["original_common_mode_module_cancellation_error_w"],
            1.0e-8,
        )
        self.assertLess(
            verification["constant_reference_Qc_invariance_error_w"], 1.0e-14
        )

    def test_input_roles_and_figure_metadata_are_explicit(self) -> None:
        bindings = self.result["input_bindings"]
        self.assertEqual(bindings["figure1_transport_candidates"]["row_count"], 105)
        self.assertEqual(bindings["figure_s9_thermal_candidates"]["row_count"], 14)
        self.assertEqual(
            bindings["figure1_transport_candidates"]["data_role"],
            "primary_article_figure_derived_source_object_candidate",
        )
        self.assertEqual(
            bindings["figure_s9_thermal_candidates"]["data_role"],
            "figure_derived_candidate_measured_as_described_in_si",
        )
        metadata = self.result["figure_metadata"]
        self.assertEqual(metadata["backend"], "Python matplotlib only")
        self.assertIn("no sampling", metadata["statistics"])
        self.assertIn("device validation", metadata["limitation"])

    def test_artifacts_and_rebuild_are_deterministic(self) -> None:
        outputs = self.result["outputs"]
        self.assertEqual(outputs["analysis_script_sha256"], module.file_sha256(SCRIPT))
        self.assertEqual(outputs["figure_sha256"], module.file_sha256(FIGURE))
        with Image.open(FIGURE) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertGreaterEqual(image.width, 3400)
            self.assertGreaterEqual(image.height, 1900)

        with tempfile.TemporaryDirectory() as temporary:
            rebuilt_json = Path(temporary) / "result.json"
            rebuilt_figure = Path(temporary) / "figure.png"
            rebuilt = module.run_analysis(rebuilt_json, rebuilt_figure)
            cross_platform = os.environ.get("CROSS_PLATFORM_REBUILD") == "1"
            if cross_platform:
                assert_nested_close(
                    self,
                    rebuilt["optimized_forward_comparison"],
                    self.result["optimized_forward_comparison"],
                )
                assert_nested_close(
                    self,
                    rebuilt["fixed_current_mechanism"],
                    self.result["fixed_current_mechanism"],
                )
                assert_raster_equivalent(self, rebuilt_figure, FIGURE)
            else:
                self.assertEqual(
                    rebuilt["optimized_forward_comparison"],
                    self.result["optimized_forward_comparison"],
                )
                self.assertEqual(
                    rebuilt["fixed_current_mechanism"],
                    self.result["fixed_current_mechanism"],
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
