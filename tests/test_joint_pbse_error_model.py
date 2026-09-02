"""Scientific and reproducibility tests for joint_pbse_error_model.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest

from PIL import Image

from scripts.analysis import joint_pbse_error_model as module


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/scientific_analysis/joint_pbse_error_model_results.json"
SOURCE = ROOT / "results/scientific_analysis/joint_pbse_error_model_source_data.csv"
SCRIPT = ROOT / "scripts/analysis/joint_pbse_error_model.py"
FIGURE_PREFIX = ROOT / "results/scientific_analysis/joint_pbse_error_model_figure"


class JointPbSeErrorModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.scenarios = {
            scenario["scenario_id"]: scenario
            for scenario in cls.result["joint_correlation_scenarios"]
        }

    def test_scope_is_deterministic_not_statistical(self) -> None:
        boundary = self.result["scope"]
        self.assertFalse(boundary["probability_distribution_assumed"])
        self.assertFalse(boundary["confidence_interval_reported"])
        self.assertFalse(boundary["correlation_matrices_empirically_estimated"])
        self.assertTrue(
            boundary["correlation_entries_are_deterministic_scenario_parameters"]
        )
        self.assertFalse(boundary["global_nonlinear_ellipsoid_minima_certified"])
        self.assertTrue(
            boundary[
                "negative_case_proves_sign_nonidentifiability_for_its_stated_geometry"
            ]
        )
        self.assertFalse(boundary["positive_tested_minimum_proves_global_identifiability"])
        self.assertFalse(boundary["real_pbse_cr_device_validation"])

    def test_error_coordinates_are_physically_separated(self) -> None:
        model = self.result["physical_error_model"]
        drift = model["shared_additive_drift"]
        self.assertTrue(drift["same_additive_deltaS_applied_to_p_and_n"])
        self.assertTrue(drift["independent_of_sigma_and_kappa_in_all_scenarios"])
        self.assertFalse(drift["public_record_proves_p_n_same_calibration_state"])
        self.assertAlmostEqual(drift["scale_v_per_k"], 9.78787990643773e-06)
        transport = model["transport_scale_errors"]
        self.assertEqual(transport["sigma_axis_scale_relative"], 0.05)
        self.assertEqual(transport["kappa_axis_scale_relative"], 0.15)
        self.assertTrue(model["reported_ranges_are_not_confidence_intervals"])
        for scenario in self.scenarios.values():
            self.assertEqual(
                scenario["named_correlations"][
                    "shared_drift_maximum_absolute_cross_correlation"
                ],
                0.0,
            )

    def test_nominal_response_and_shared_drift_threshold(self) -> None:
        nominal = self.result["nominal_result"]["reoptimized_response"]
        self.assertAlmostEqual(
            1.0e3 * nominal["original_minus_flattened_Qc_max_w"],
            3.0668783059080162,
        )
        drift = self.result["shared_instrument_drift_axis"]
        self.assertAlmostEqual(
            drift["zero_crossing_fraction_of_full_negative_axis"],
            0.9661999952082848,
            places=6,
        )
        self.assertAlmostEqual(
            1.0e6 * drift["zero_crossing_endpoint_additive_drift_v_per_k"],
            -9.457049518699402,
            places=5,
        )
        self.assertAlmostEqual(
            1.0e3 * drift["sweep"][-1]["reoptimized_response_w"],
            -0.11995119268914944,
        )
        self.assertTrue(drift["full_negative_axis_is_sign_reversing_case"])

    def test_opposite_relative_signs_do_not_cancel_common_additive_error(self) -> None:
        structure = self.result["single_datum_sign_structure"]
        self.assertFalse(structure["same_temperature_branch_cancellation"])
        self.assertTrue(structure["cross_temperature_partial_cancellation"])
        by_temperature = {
            row["nominal_temperature_label_k"]: row
            for row in structure["temperature_sums"]
        }
        for row in by_temperature.values():
            self.assertTrue(row["relative_sensitivity_signs_are_opposite"])
            self.assertTrue(row["common_additive_contributions_have_same_sign"])
        self.assertAlmostEqual(
            by_temperature[323][
                "p_plus_n_contribution_mw_per_positive_1_uV_per_K"
            ],
            -0.8839365235280385,
            places=6,
        )
        self.assertAlmostEqual(
            by_temperature[373][
                "p_plus_n_contribution_mw_per_positive_1_uV_per_K"
            ],
            0.7519543704212215,
            places=6,
        )

    def test_all_correlation_shapes_are_psd_and_explicit(self) -> None:
        self.assertEqual(len(self.scenarios), 4)
        for scenario in self.scenarios.values():
            self.assertGreaterEqual(
                scenario["minimum_correlation_matrix_eigenvalue"], -1.0e-12
            )
            matrix = scenario["correlation_shape_matrix"]
            self.assertEqual(len(matrix), 9)
            self.assertTrue(all(len(row) == 9 for row in matrix))
            self.assertTrue(all(abs(matrix[i][i] - 1.0) < 1.0e-12 for i in range(9)))
        moderate = self.scenarios["moderate_band_transport_coupling"][
            "named_correlations"
        ]
        self.assertAlmostEqual(moderate["p_n_Seebeck_323K"], 0.60)
        self.assertAlmostEqual(moderate["p_Seebeck_323K_sigma_p"], -0.3464101615)
        self.assertAlmostEqual(moderate["sigma_p_kappa_p"], 0.4898979486)

    def test_joint_nonlinear_intervals_depend_on_correlation_geometry(self) -> None:
        expected_mw = {
            "independent_residual_axes": {
                0.0: (-8.41816, 10.19703),
                0.5: (-6.49964, 10.00819),
                1.0: (-0.11995, 7.45427),
            },
            "same_run_scale_dominated": {
                0.0: (0.33367, 7.70207),
                0.5: (0.33961, 7.62900),
                1.0: (-0.11995, 7.40233),
            },
            "moderate_band_transport_coupling": {
                0.0: (-0.19817, 8.20403),
                0.5: (-0.05624, 8.06502),
                1.0: (-0.11995, 7.64930),
            },
            "strong_band_locked_coupling": {
                0.0: (0.64773, 6.97753),
                0.5: (0.59913, 7.23535),
                1.0: (-0.11995, 7.83039),
            },
        }
        for scenario_id, expected_allocations in expected_mw.items():
            scenario = self.scenarios[scenario_id]
            allocations = {
                row["shared_additive_drift_fraction"]: row
                for row in scenario["allocations"]
            }
            for fraction, (expected_minimum, expected_maximum) in expected_allocations.items():
                row = allocations[fraction]
                self.assertAlmostEqual(
                    1.0e3 * row["minimum_evaluated_nonlinear_case_w"],
                    expected_minimum,
                    places=4,
                )
                self.assertAlmostEqual(
                    1.0e3 * row["maximum_evaluated_nonlinear_case_w"],
                    expected_maximum,
                    places=4,
                )
                self.assertEqual(
                    row["evaluated_nonlinear_case_interval_contains_zero"],
                    expected_minimum <= 0.0 <= expected_maximum,
                )
                self.assertFalse(row["positive_sign_globally_certified"])

    def test_negative_joint_case_simultaneously_moves_S_sigma_and_kappa(self) -> None:
        moderate = self.scenarios["moderate_band_transport_coupling"]
        allocation = next(
            row for row in moderate["allocations"]
            if row["shared_additive_drift_fraction"] == 0.5
        )
        case = allocation["case_library"]["minimum_case"]
        coordinate = case["normalized_coordinates"]
        self.assertTrue(
            any(abs(coordinate[name]) > 1.0e-4 for name in module.PARAMETER_NAMES[1:5])
        )
        self.assertTrue(
            any(abs(coordinate[name]) > 1.0e-4 for name in module.PARAMETER_NAMES[5:7])
        )
        self.assertTrue(
            any(abs(coordinate[name]) > 1.0e-4 for name in module.PARAMETER_NAMES[7:9])
        )
        self.assertLess(
            case["reoptimized_response"]["original_minus_flattened_Qc_max_w"],
            0.0,
        )
        self.assertTrue(case["property_admissibility"]["admissible"])

    def test_positive_library_minimum_is_not_overclaimed(self) -> None:
        for scenario_id in (
            "same_run_scale_dominated",
            "strong_band_locked_coupling",
        ):
            scenario = self.scenarios[scenario_id]
            for fraction in (0.0, 0.5):
                allocation = next(
                    row for row in scenario["allocations"]
                    if row["shared_additive_drift_fraction"] == fraction
                )
                self.assertTrue(allocation["positive_sign_survives_tested_library"])
                self.assertFalse(allocation["positive_sign_globally_certified"])
                self.assertFalse(
                    allocation["case_library"]["global_nonlinear_minimum_certified"]
                )

    def test_conservation_continuous_ceiling_and_ellipsoid_closure(self) -> None:
        verification = self.result["verification"]
        self.assertTrue(verification["all_cases_admissible"])
        self.assertTrue(verification["all_scenario_shapes_positive_semidefinite"])
        self.assertTrue(verification["shared_drift_cross_correlations_exactly_zero"])
        self.assertLessEqual(
            verification["maximum_case_ellipsoid_norm_squared"],
            1.0 + 2.0e-10,
        )
        self.assertLessEqual(
            verification["maximum_case_relative_Seebeck_deviation"],
            0.05 + 3.0e-12,
        )
        self.assertLess(
            verification["maximum_optimized_module_energy_residual_w"], 2.0e-14
        )
        self.assertLess(verification["maximum_gradient_step_difference_w"], 4.0e-8)
        for scenario in self.scenarios.values():
            for allocation in scenario["allocations"]:
                for case in allocation["case_library"]["records"]:
                    self.assertTrue(case["property_admissibility"]["admissible"])
                    self.assertLessEqual(
                        case["ellipsoid_norm_squared"], 1.0 + 2.0e-10
                    )

    def test_artifacts_and_source_data_are_hash_bound(self) -> None:
        outputs = self.result["outputs"]
        self.assertEqual(
            outputs["analysis_script"]["sha256"], module.file_sha256(SCRIPT)
        )
        self.assertEqual(
            outputs["source_data"]["sha256"], module.file_sha256(SOURCE)
        )
        with SOURCE.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 23)
        self.assertEqual({row["panel"] for row in rows}, {"a", "b", "c"})
        for suffix in ("png", "svg", "pdf", "tiff"):
            path = FIGURE_PREFIX.with_suffix(f".{suffix}")
            self.assertTrue(path.is_file())
            self.assertEqual(
                outputs["figures"][suffix]["sha256"], module.file_sha256(path)
            )
        with Image.open(FIGURE_PREFIX.with_suffix(".png")) as image:
            self.assertGreaterEqual(image.width, 2000)
            self.assertGreaterEqual(image.height, 3500)
        svg = FIGURE_PREFIX.with_suffix(".svg").read_text(encoding="utf-8")
        self.assertIn("zero at 0.966", svg)
        self.assertIn("moderate band", svg)
        self.assertIn("<text", svg)

    def test_two_point_solver_reproduction(self) -> None:
        context = module.build_context()
        evaluator = module.ResponseEvaluator(context)
        origin = evaluator.evaluate([0.0] * 9)
        full_negative_drift = evaluator.evaluate([-1.0] + [0.0] * 8)
        self.assertAlmostEqual(
            1.0e3
            * origin["reoptimized_response"][
                "original_minus_flattened_Qc_max_w"
            ],
            3.0668783059080162,
        )
        self.assertAlmostEqual(
            1.0e3
            * full_negative_drift["reoptimized_response"][
                "original_minus_flattened_Qc_max_w"
            ],
            -0.11995119268914944,
        )


if __name__ == "__main__":
    unittest.main()
