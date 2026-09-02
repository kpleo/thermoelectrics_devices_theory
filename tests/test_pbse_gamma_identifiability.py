"""Physics, evidence-boundary, and artifact tests for PbSe Gamma identifiability."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts.analysis import analyze_pbse_gamma_identifiability as module
from scripts.numerical_assertions import assert_nested_close, assert_raster_equivalent


ROOT = Path(__file__).resolve().parents[1]
RESULT_JSON = (
    ROOT / "results/scientific_analysis/pbse_gamma_identifiability_results.json"
)
FIGURE_PREFIX = ROOT / "results/scientific_analysis/pbse_gamma_identifiability"
SCRIPT = ROOT / "scripts/analysis/analyze_pbse_gamma_identifiability.py"


class PbSeGammaIdentifiabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))

    def test_inputs_are_hash_bound_and_claims_remain_candidate_level(self) -> None:
        bindings = self.result["input_bindings"]
        self.assertEqual(bindings["hash_algorithm"], "sha256")
        expected = {
            module.output_locator(path): digest
            for path, digest in module.EXPECTED_INPUT_SHA256.items()
        }
        self.assertEqual(bindings["bound_input_hashes"], expected)
        self.assertEqual(
            bindings["supporting_information"]["sha256"],
            module.EXPECTED_SI_SHA256,
        )
        self.assertEqual(
            bindings["roles"]["figure1"],
            "primary_article_figure_derived_source_object_candidate",
        )
        self.assertFalse(bindings["roles"]["independent_device_validation_eligible"])

        boundary = self.result["scope"]
        self.assertFalse(boundary["statistical_confidence_interval"])
        self.assertFalse(boundary["probability_distribution_assumed"])
        self.assertFalse(boundary["p_n_covariance_assumed"])
        self.assertFalse(boundary["real_pbse_cr_device_validation"])
        self.assertFalse(boundary["public_data_identify_nominal_effect_sign"])
        self.assertFalse(
            boundary["public_data_identify_nominal_effect_order_of_magnitude"]
        )

    def test_nominal_three_milliwatt_result_is_reproduced(self) -> None:
        target = self.result["target_condition"]
        self.assertAlmostEqual(target["hot_temperature_k"], 362.9980470654064)
        self.assertAlmostEqual(target["cold_temperature_k"], 309.894134846523)
        self.assertFalse(target["below_300_k_extrapolation_used"])
        nominal = self.result["nominal_result"]["reoptimized"]
        self.assertAlmostEqual(
            nominal["original_minus_flattened_Qc_max_w"],
            module.EXPECTED_NOMINAL_EFFECT_W,
            places=12,
        )
        self.assertAlmostEqual(
            target["nominal_optimum_current_a"], 2.8106854181890806
        )
        self.assertLess(
            self.result["verification"]["nominal_effect_reproduction_error_w"],
            5.0e-13,
        )

    def test_smooth_alpha_preserving_cases_reverse_the_sign(self) -> None:
        cases = self.result["seebeck_identifiability"][
            "decisive_alpha_preserving_cases"
        ]
        negative = cases["negative"]
        positive = cases["positive"]
        negative_effect = negative["reoptimized"][
            "original_minus_flattened_Qc_max_w"
        ]
        positive_effect = positive["reoptimized"][
            "original_minus_flattened_Qc_max_w"
        ]
        self.assertAlmostEqual(1.0e3 * negative_effect, -0.11995119268914944)
        self.assertAlmostEqual(1.0e3 * positive_effect, 5.487772403725)
        self.assertLess(negative_effect, 0.0)
        self.assertGreater(positive_effect, 0.0)
        for case in (negative, positive):
            summary = case["property_summary"]
            self.assertLess(
                summary["maximum_alpha_change_from_nominal_v_per_k"], 1.0e-15
            )
            self.assertLessEqual(
                summary[
                    "maximum_relative_p_Seebeck_deviation_on_common_support"
                ],
                0.05 + 2.0e-12,
            )
            self.assertLessEqual(
                summary[
                    "maximum_relative_n_Seebeck_deviation_on_common_support"
                ],
                0.05 + 2.0e-12,
            )
            self.assertTrue(summary["p_Seebeck_monotone_increasing"])
            self.assertTrue(summary["n_Seebeck_monotone_decreasing"])

        sweep = self.result["alpha_preserving_common_drift_sweep"]
        self.assertAlmostEqual(
            sweep["zero_crossing_normalized_amplitude"],
            -0.9661999952082848,
            places=6,
        )
        self.assertLess(
            abs(
                sweep["zero_crossing_record"]["reoptimized"][
                    "original_minus_flattened_Qc_max_w"
                ]
            ),
            2.0e-9,
        )
        self.assertLess(
            self.result["verification"][
                "maximum_constant_common_offset_effect_change_w"
            ],
            1.0e-12,
        )

    def test_interpolation_and_digitization_are_not_the_decisive_uncertainty(self) -> None:
        records = {
            record["scenario_id"]: record
            for record in self.result["interpolation_and_digitization"]["records"]
        }
        effects_mw = {
            key: 1.0e3
            * value["reoptimized"]["original_minus_flattened_Qc_max_w"]
            for key, value in records.items()
        }
        self.assertAlmostEqual(effects_mw["piecewise_linear"], 3.1009643169314094)
        self.assertAlmostEqual(
            effects_mw["bound_checked_smooth_quadratic"], 2.804576250905577
        )
        self.assertAlmostEqual(
            effects_mw["independent_vector_extractor_B"], 3.0529005903658923
        )
        self.assertTrue(all(value > 0.0 for value in effects_mw.values()))
        digitization = self.result["interpolation_and_digitization"][
            "digitization_only"
        ]
        self.assertFalse(digitization["measurement_uncertainty_combined"])
        for carrier in ("p", "n"):
            self.assertLess(
                digitization["route_differences"][carrier][
                    "maximum_relative_Seebeck_difference"
                ],
                1.0e-4,
            )

    def test_complete_monotone_corner_screen_contains_both_signs(self) -> None:
        screen = self.result["monotone_pointwise_corner_screen"]
        self.assertEqual(screen["raw_binary_corner_count_before_monotonicity"], 256)
        self.assertEqual(screen["valid_p_corner_count"], 9)
        self.assertEqual(screen["valid_n_corner_count"], 5)
        self.assertEqual(screen["evaluated_pair_corner_count"], 45)
        self.assertTrue(screen["complete_monotone_binary_corner_enumeration"])
        self.assertFalse(screen["continuous_interior_global_extrema_certified"])
        lower, upper = screen["reoptimized_effect_envelope_w"]
        self.assertAlmostEqual(1.0e3 * lower, -28.59354729388597)
        self.assertAlmostEqual(1.0e3 * upper, 10.411141432757454)
        self.assertLess(lower, 0.0)
        self.assertGreater(upper, 0.0)
        for case in (screen["minimum_case"], screen["maximum_case"]):
            summary = case["property_summary"]
            self.assertTrue(summary["p_Seebeck_monotone_increasing"])
            self.assertTrue(summary["n_Seebeck_monotone_decreasing"])

    def test_sigma_kappa_are_a_separate_branch_transfer_layer(self) -> None:
        transfer = self.result["branch_transfer_condition_sensitivity"]
        self.assertEqual(transfer["corner_count"], 16)
        self.assertFalse(transfer["seebeck_perturbed"])
        self.assertFalse(transfer["pooled_with_seebeck_identifiability"])
        self.assertFalse(transfer["statistical_interval"])
        lower, upper = transfer["reoptimized_effect_envelope_w"]
        self.assertAlmostEqual(1.0e3 * lower, -0.9578916785211167)
        self.assertAlmostEqual(1.0e3 * upper, 6.6479386381194105)
        self.assertLess(lower, 0.0)
        self.assertGreater(upper, 0.0)
        limits = self.result["measurement_limits"]
        self.assertFalse(limits["sigma_kappa_and_Seebeck_pooled"])
        self.assertEqual(limits["electrical_conductivity_relative_method_ceiling"], 0.05)
        self.assertEqual(limits["thermal_conductivity_relative_method_ceiling"], 0.15)

    def test_local_sensitivity_identifies_the_raw_data_priority(self) -> None:
        sensitivity = self.result["single_knot_local_sensitivity"]
        self.assertTrue(sensitivity["top_four_all_at_323_or_373_k"])
        records = sensitivity["records_ranked_by_absolute_effect"]
        top_four_keys = {
            (record["carrier"], round(record["temperature_k"]))
            for record in records[:4]
        }
        self.assertEqual(
            top_four_keys,
            {("p", 323), ("n", 323), ("p", 373), ("n", 373)},
        )
        inactive = [
            record
            for record in records
            if record["knot_index_one_based"] >= 5
        ]
        self.assertEqual(len(inactive), 6)
        self.assertTrue(
            all(
                record["effect_change_mw_per_one_percent_relative_knot_change"]
                == 0.0
                for record in inactive
            )
        )
        priorities = self.result["raw_data_priorities"]
        self.assertEqual(priorities[0]["rank"], 1)
        self.assertIn("323 and 373 K", priorities[0]["request"])
        self.assertIn("covariance", priorities[0]["request"])

    def test_numerical_closure_and_diagnostic_exports(self) -> None:
        self.assertLess(
            self.result["verification"]["maximum_optimized_module_energy_residual_w"],
            2.0e-14,
        )
        outputs = self.result["outputs"]
        self.assertEqual(outputs["analysis_script_sha256"], module.file_sha256(SCRIPT))
        self.assertEqual(set(outputs["figures"]), {"png", "svg", "pdf", "tiff"})
        for suffix, entry in outputs["figures"].items():
            path = FIGURE_PREFIX.with_suffix(f".{suffix}")
            self.assertTrue(path.is_file())
            self.assertEqual(entry["sha256"], module.file_sha256(path))
        with Image.open(FIGURE_PREFIX.with_suffix(".png")) as image:
            self.assertGreaterEqual(image.width, 4000)
            self.assertGreaterEqual(image.height, 3000)
            self.assertEqual(image.mode, "RGBA")
        svg = FIGURE_PREFIX.with_suffix(".svg").read_text(encoding="utf-8")
        self.assertIn("Public Seebeck curves do not resolve", svg)
        self.assertIn("<text", svg)

    def test_rebuild_is_numerically_and_visually_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rebuilt_json = directory / "result.json"
            rebuilt_prefix = directory / "figure"
            rebuilt = module.run_analysis(rebuilt_json, rebuilt_prefix)
            cross_platform = os.environ.get("CROSS_PLATFORM_REBUILD") == "1"
            for key in (
                "nominal_result",
                "seebeck_identifiability",
                "alpha_preserving_common_drift_sweep",
                "monotone_pointwise_corner_screen",
                "branch_transfer_condition_sensitivity",
                "single_knot_local_sensitivity",
            ):
                if cross_platform:
                    assert_nested_close(self, rebuilt[key], self.result[key])
                else:
                    self.assertEqual(rebuilt[key], self.result[key])
            if cross_platform:
                assert_raster_equivalent(
                    self,
                    rebuilt_prefix.with_suffix(".png"),
                    FIGURE_PREFIX.with_suffix(".png"),
                )
            else:
                self.assertEqual(
                    module.file_sha256(rebuilt_prefix.with_suffix(".png")),
                    module.file_sha256(FIGURE_PREFIX.with_suffix(".png")),
                )


if __name__ == "__main__":
    unittest.main()
