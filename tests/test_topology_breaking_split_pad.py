from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analysis/topology_breaking_split_pad.py"
RESULT = ROOT / "results/scientific_analysis/topology_breaking_split_pad_results.json"
FIGURE_PREFIX = ROOT / "results/scientific_analysis/topology_breaking_split_pad"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TestTopologyBreakingSplitPad(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stdout + completed.stderr)
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_schema_and_scope(self) -> None:
        self.assertEqual(
            self.result["schema_version"], "topology_breaking_split_pad/v1"
        )
        self.assertEqual(self.result["version_metadata"]["status"], "REFERENCE")
        boundary = self.result["scope"]
        self.assertFalse(boundary["real_device_observation"])
        self.assertFalse(boundary["anchor_device_validation"])
        self.assertFalse(boundary["unique_topology_breaking_mechanism_claimed"])
        self.assertFalse(boundary["single_isothermal_cold_reservoir_interpretation"])
        self.assertFalse(boundary["single_device_delta_T_modulation_equivalent_to_C_switch"])
        self.assertFalse(boundary["current_odd_projection_isolates_all_odd_physics"])
        self.assertTrue(
            boundary["exact_C_switch_requires_calibrated_reference_lead_contrast"]
        )
        self.assertEqual(
            self.result["device_scale_map"]["anchor"]["data_role"],
            "figure_derived_device_scenario_scale_anchor_only",
        )

    def test_general_aggregate_law_energy_and_strict_nulls(self) -> None:
        symbolic = self.result["symbolic_theory"]
        self.assertTrue(symbolic["passed"])
        increments = symbolic["general_aggregate_increments"]
        self.assertEqual(
            increments["delta_Qc_aggregate"],
            "sum_j[C_j*I_j*delta_Tc_j]",
        )
        self.assertEqual(increments["elementwise_energy_closure"], "0")
        self.assertIn(
            "delta_Pin_aggregate=0", increments["aggregate_energy_closure"]
        )
        special = symbolic["same_C_same_series_I_special_case"]
        self.assertEqual(special["energy_closure"], "0")
        self.assertEqual(
            special["cold_split_hot_isothermal"]["delta_Qc_aggregate"],
            "C*I*N*mean_delta_Tc",
        )
        isothermal = symbolic["strict_zero_mode_controls"][
            "shared_isothermal_endpoints"
        ]
        self.assertEqual(set(isothermal.values()), {"0"})
        global_control = symbolic["strict_zero_mode_controls"][
            "global_co_shift_including_all_electrically_active_segments"
        ]
        self.assertEqual(global_control["complete_network_delta_Qc"], "0")
        self.assertEqual(global_control["complete_network_delta_Qh"], "0")
        self.assertEqual(global_control["complete_network_delta_V"], "0")
        heterogeneous = symbolic["heterogeneous_three_element_energy_validation"]
        self.assertLess(abs(heterogeneous["energy_residual_w"]), 1.0e-18)
        self.assertEqual(len(heterogeneous["C_j_v_per_k"]), 3)

    def test_temperature_dependent_solver_recovers_exact_scale_law(self) -> None:
        validation = self.result["numerical_validation"]
        self.assertEqual(validation["validation_point_count"], 36)
        self.assertTrue(validation["acceptance"]["passed"])
        errors = validation["maximum_errors"]
        self.assertLess(errors["absolute_Qc_prediction_error_w"], 2.0e-12)
        self.assertLess(errors["absolute_Qh_prediction_error_w"], 2.0e-12)
        self.assertLess(errors["absolute_voltage_prediction_error_v"], 2.0e-12)
        self.assertLess(errors["absolute_incremental_energy_residual_w"], 2.0e-12)
        self.assertLess(errors["temperature_field_change_k"], 1.0e-9)

    def test_special_case_reaches_mw_and_one_percent_model_crossing(self) -> None:
        scenario = self.result["device_scale_map"]["reference_scenario"]
        anchor = self.result["device_scale_map"]["anchor"]
        expected = (
            anchor["pair_count"]
            * scenario["common_shift_uv_per_k"]
            * 1.0e-6
            * anchor["current_a"]
            * scenario["cold_split_k"]
        )
        self.assertAlmostEqual(scenario["delta_Qc_w"], expected, places=15)
        self.assertTrue(scenario["exceeds_1_mW_operational_threshold"])
        self.assertTrue(scenario["above_1_percent_model_crossing"])
        self.assertIn("not an experimental", scenario["one_percent_language"])
        self.assertGreater(scenario["delta_Qc_mw"], 7.0)
        self.assertGreater(scenario["percent_of_reference_Qc"], 1.0)
        self.assertLess(abs(scenario["energy_residual_w"]), 1.0e-15)

    def test_exact_contrast_parity_and_projection_boundary(self) -> None:
        reference = self.result["current_reversal"]["reference_magnitude"]
        self.assertAlmostEqual(
            reference["reverse_delta_Qc_w"],
            -reference["forward_delta_Qc_w"],
            places=15,
        )
        self.assertAlmostEqual(
            reference["current_odd_projection_w"],
            reference["forward_delta_Qc_w"],
            places=15,
        )
        self.assertLess(abs(reference["current_even_projection_w"]), 1.0e-15)
        self.assertAlmostEqual(
            reference["current_and_mismatch_double_odd_projection_w"],
            reference["forward_delta_Qc_w"],
            places=15,
        )
        self.assertLess(abs(reference["incremental_energy_residual_w"]), 1.0e-15)
        limitation = self.result["current_reversal"]["projection_limitation"]
        self.assertIn("does not by itself remove baseline", limitation)
        projection = self.result["experimental_design"]["projection_boundary"]
        self.assertIn("baseline Peltier heat", projection["does_not_automatically_isolate"])

    def test_declared_scale_crossings_are_exact(self) -> None:
        scale = self.result["device_scale_map"]
        anchor = scale["anchor"]
        row = next(
            item
            for item in scale["scale_crossings"]
            if item["common_shift_uv_per_k"] == 80.0
        )
        prefactor = anchor["pair_count"] * anchor["current_a"] * 80.0e-6
        self.assertAlmostEqual(
            row["delta_T_for_1_mW_operational_scale_k"] * prefactor,
            1.0e-3,
            places=15,
        )
        self.assertAlmostEqual(
            row["delta_T_at_1_percent_model_crossing_k"] * prefactor,
            0.01 * anchor["reference_qc_w"],
            places=15,
        )

    def test_single_device_delta_t_modulation_is_not_equated_to_C_switch(self) -> None:
        design = self.result["experimental_design"]
        self.assertIn("not an equivalent C switch", design["strict_C_switch_requirement"])
        route = design["single_device_modulation_route"]
        self.assertIn("neither", route["status"])
        self.assertIn("M(T0)+Gamma(T0)", route["small_split_boundary_term"])
        self.assertIn("additional current-odd", route["why_not_exact_for_total_port_heat"])

    def test_figure_bundle_and_hashes(self) -> None:
        artifact_by_name = {
            item["output_name"]: item
            for item in self.result["outputs"]["figures"]
        }
        for suffix in (".svg", ".pdf", ".tiff", ".png"):
            path = FIGURE_PREFIX.with_suffix(suffix)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(artifact_by_name[path.name]["sha256"], sha256_file(path))
        with Image.open(FIGURE_PREFIX.with_suffix(".tiff")) as image:
            self.assertGreaterEqual(image.width, 3000)
            self.assertGreaterEqual(image.height, 2000)
        svg_text = FIGURE_PREFIX.with_suffix(".svg").read_text(encoding="utf-8")
        self.assertIn("<text", svg_text)
        self.assertIn("2026-08-26T00:00:00+00:00", svg_text)

    def test_complete_rebuild_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory(prefix="split-pad-repro-") as directory:
            temporary = Path(directory)
            rebuild_hashes = []
            for run_id in ("run_a", "run_b"):
                run_dir = temporary / run_id
                run_dir.mkdir()
                output = run_dir / "topology_breaking_split_pad_results.json"
                figure_prefix = run_dir / "topology_breaking_split_pad"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--output",
                        str(output),
                        "--figure-prefix",
                        str(figure_prefix),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stdout + completed.stderr,
                )
                paths = [output] + [
                    figure_prefix.with_suffix(suffix)
                    for suffix in (".svg", ".pdf", ".png", ".tiff")
                ]
                rebuild_hashes.append(
                    {path.suffix: sha256_file(path) for path in paths}
                )
            self.assertEqual(rebuild_hashes[0], rebuild_hashes[1])


if __name__ == "__main__":
    unittest.main()
