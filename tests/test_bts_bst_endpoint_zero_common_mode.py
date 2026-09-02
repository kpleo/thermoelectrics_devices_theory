from __future__ import annotations

import csv
import json
from pathlib import Path
import unittest

import numpy as np

from scripts.analysis import analyze_bts_bst_endpoint_zero_common_mode as analysis


class TestBtsBstEndpointZeroCommonMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result_path = analysis.DEFAULT_JSON
        if not cls.result_path.is_file():
            raise AssertionError(
                "run analyze_bts_bst_endpoint_zero_common_mode.py before tests"
            )
        cls.result = json.loads(cls.result_path.read_text(encoding="utf-8"))

    def test_source_hashes_identity_and_license(self) -> None:
        transport, validation = analysis.validate_sources_and_load_transport()
        self.assertEqual(validation["article_identity"]["doi"], analysis.DOI)
        self.assertEqual(validation["article_identity"]["pmcid"], analysis.PMCID)
        self.assertEqual(
            validation["article_identity"]["corresponding_author_of_interest"],
            "Li-Dong Zhao",
        )
        self.assertEqual(validation["article_identity"]["license"], "CC BY 4.0")
        self.assertEqual(len(validation["source_hashes"]), 6)
        self.assertEqual(len(transport["temperature_k"]), 6)

    def test_transport_is_strictly_branch_complete(self) -> None:
        transport = self.result["transport"]
        expected = {
            "temperature_k",
            "p_seebeck_v_per_k",
            "n_seebeck_v_per_k",
            "p_sigma_s_per_m",
            "n_sigma_s_per_m",
            "p_kappa_w_per_m_k",
            "n_kappa_w_per_m_k",
        }
        self.assertEqual(set(transport), expected)
        for values in transport.values():
            self.assertEqual(len(values), 6)
        self.assertTrue(np.all(np.asarray(transport["p_seebeck_v_per_k"]) > 0.0))
        self.assertTrue(np.all(np.asarray(transport["n_seebeck_v_per_k"]) < 0.0))
        for key in ("p_sigma_s_per_m", "n_sigma_s_per_m", "p_kappa_w_per_m_k", "n_kappa_w_per_m_k"):
            self.assertTrue(np.all(np.asarray(transport[key]) > 0.0))

    def test_pixel_calibration_is_reproducible(self) -> None:
        with analysis.TRANSPORT_CSV.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        for row in rows:
            for prefix, value_column, suffix in (
                ("seebeck", "seebeck_uv_per_k", "uv_per_k"),
                ("sigma", "sigma_1e3_s_per_cm", "1e3_s_per_cm"),
                ("kappa", "kappa_w_per_m_k", "w_per_m_k"),
            ):
                value = analysis._linear_pixel_value(
                    float(row[f"{prefix}_pixel_y"]),
                    float(row[f"{prefix}_axis_y_top_px"]),
                    float(row[f"{prefix}_axis_y_bottom_px"]),
                    float(row[f"{prefix}_axis_top_{suffix}"]),
                    float(row[f"{prefix}_axis_bottom_{suffix}"]),
                )
                self.assertAlmostEqual(value, float(row[value_column]), places=6)

    def test_endpoint_zero_invariants_are_exact(self) -> None:
        invariants = self.result["exact_invariants"]
        for key, value in invariants.items():
            self.assertLess(abs(value), 1.0e-14, key)

    def test_primary_adjoint_matches_independent_nonlinear_bvp(self) -> None:
        primary = self.result["primary_nominal_result"]
        self.assertAlmostEqual(primary["current_a"], 3.0, places=15)
        self.assertAlmostEqual(
            1.0e3 * primary["adjoint_response_module_w"],
            1.3144084350294556,
            places=9,
        )
        self.assertLess(primary["adjoint_fd_relative_error"], 3.0e-7)
        self.assertLess(primary["maximum_relative_energy_residual"], 1.0e-12)
        self.assertGreater(primary["p_branch_adjoint_w"], 0.0)
        self.assertLess(primary["n_branch_adjoint_w"], 0.0)
        self.assertLess(
            abs(primary["p_branch_adjoint_w"]),
            31.0 * abs(primary["adjoint_response_per_pair_w"]),
        )

    def test_current_scan_never_exits_probe_bound(self) -> None:
        scan = self.result["current_scan"]
        self.assertEqual([row["current_a"] for row in scan], list(np.linspace(0.0, 3.0, 7)))
        for row in scan:
            self.assertGreaterEqual(row["temperature_minimum_k"], 323.0 - 1.0e-8)
            self.assertLessEqual(row["temperature_maximum_k"], 373.0 + 1.0e-8)
            self.assertGreaterEqual(row["normalized_mode_minimum_on_field"], -1.0e-10)
            self.assertLessEqual(row["normalized_mode_maximum_on_field"], 1.0 + 1.0e-10)
        self.assertTrue(
            self.result["validation_checks"][
                "current_scan_probe_globally_bounded_pass"
            ]
        )

    def test_zero_response_checks_are_zero(self) -> None:
        controls = self.result["zero_response_checks"]
        self.assertEqual(controls["zero_current_endpoint_zero_response_w"], 0.0)
        self.assertEqual(
            controls["identical_transport_antisymmetric_seebeck_response_w"],
            0.0,
        )
        self.assertEqual(controls["constant_common_offset_qc_change_w"], 0.0)
        self.assertLess(
            controls["constant_common_offset_energy_residual_fraction"], 1.0e-12
        )

    def test_method_bound_envelope_crosses_zero(self) -> None:
        envelope = self.result["method_bound_corner_envelope"]
        self.assertEqual(envelope["corner_count"], 64)
        self.assertTrue(envelope["interval_crosses_zero"])
        low = 1.0e3 * envelope["minimum"]["response_module_w"]
        high = 1.0e3 * envelope["maximum"]["response_module_w"]
        self.assertAlmostEqual(low, -2.9073455916088267, places=8)
        self.assertAlmostEqual(high, 5.728291936160587, places=8)
        self.assertLess(envelope["maximum_adjoint_fd_relative_error"], 2.0e-5)
        self.assertIn("not_confidence_interval", envelope["statistical_status"])

    def test_contact_nuisance_is_dressed_not_fitted(self) -> None:
        nuisance = self.result["thermal_contact_and_parasitic_nuisance"]
        self.assertIn("not_fitted", nuisance["status"])
        self.assertEqual(
            [row["contact_ratio_chi_equal_cold_hot"] for row in nuisance["records"]],
            [0.0, 0.05, 0.10, 0.25],
        )
        for row in nuisance["records"]:
            self.assertLess(row["relative_adjoint_fd_error"], 3.0e-6)
            self.assertLess(row["maximum_relative_energy_residual"], 1.0e-12)
        self.assertGreater(
            nuisance["records"][0]["contact_dressed_adjoint_response_module_w"],
            nuisance["records"][-1]["contact_dressed_adjoint_response_module_w"],
        )

    def test_published_device_resistance_is_a_no_fit_scale_check(self) -> None:
        records = self.result["device_resistance_and_geometry_scale"]["records"]
        self.assertEqual(len(records), 2)
        for row in records:
            self.assertGreater(row["published_minus_bulk_ohm"], 0.0)
            self.assertLess(row["residual_fraction_of_published"], 0.12)
        self.assertAlmostEqual(
            records[0]["geometry_scaled_bulk_resistance_ohm"],
            0.1239087882421234,
            places=12,
        )
        self.assertAlmostEqual(
            records[1]["geometry_scaled_bulk_resistance_ohm"],
            0.14613036906179444,
            places=12,
        )

    def test_public_data_do_not_identify_response_sign(self) -> None:
        checks = self.result["validation_checks"]
        self.assertTrue(all(checks.values()))
        interpretation = self.result["interpretation"]
        self.assertFalse(interpretation["sign_identified_from_public_data"])
        self.assertIn(
            "sign or magnitude",
            interpretation["limitation"],
        )
        self.assertIn("BTS/BST material family", interpretation["supported_result"])

    def test_artifacts_are_present_and_svg_text_is_editable(self) -> None:
        figure = self.result["outputs"]["figure"]
        for key in ("png", "svg", "pdf", "tiff"):
            path = analysis.ROOT / figure[key]
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 1000)
        svg = (analysis.ROOT / figure["svg"]).read_text(encoding="utf-8")
        self.assertIn("<text", svg)
        self.assertNotIn("fonttype", svg)


if __name__ == "__main__":
    unittest.main()
