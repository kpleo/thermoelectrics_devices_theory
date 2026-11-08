"""Independent invariants for fixed-area common-mode geometry calculations."""
import math
import unittest

import numpy as np

from scripts.analysis import geometry_matching as study


class GeometryStudyTests(unittest.TestCase):
    def test_area_and_differential_seebeck_are_preserved(self):
        for case in ("matched", "unmatched"):
            base, _ = study.load_material_model("constant", case)
            model = study.rescale_geometry(base, math.sqrt(3))
            self.assertAlmostEqual(model.p_leg.area_m2 + model.n_leg.area_m2, 2e-6)
            self.assertAlmostEqual(float(model.p_leg.seebeck.evaluate([315])[0] - model.n_leg.seebeck.evaluate([315])[0]), 400e-6)
            rk_p = 1e-5 * model.p_leg.length_m**2 / (1.5 * model.p_leg.area_m2**2)
            rk_n = 2e-5 * model.n_leg.length_m**2 / model.n_leg.area_m2**2
            self.assertAlmostEqual(rk_p, rk_n, places=11)

    def test_equal_peak_and_endpoint_zero_shapes(self):
        t = np.linspace(300, 330, 30001)
        for name in study.MODE_NAMES:
            mode = study.make_mode(name, 300, 330)
            self.assertAlmostEqual(np.max(np.abs(mode.value(t))), 10e-6, places=12)
            if name != "linear":
                np.testing.assert_allclose(mode.value(np.array([300, 330])), 0, atol=1e-18)
            h = 1e-3
            tt = np.array([305., 315., 325.])
            np.testing.assert_allclose((mode.value(tt+h)-mode.value(tt-h))/(2*h), mode.derivative(tt), rtol=2e-7, atol=1e-14)

    def test_transfer_matched_pair_cancels_all_tested_modes(self):
        geometry, rows = study.evaluate_geometry("constant", math.sqrt(3), "unmatched")
        self.assertLess(geometry["measure_l1_normalized"], 1e-9)
        self.assertTrue(all(abs(r["delta_qc_per_pair_w"]) < 1e-12 for r in rows))

    def test_seebeck_matching_does_not_cancel_common_response(self):
        _, matched = study.evaluate_geometry("constant", 1, "matched")
        _, unmatched = study.evaluate_geometry("constant", 1, "unmatched")
        for a, b in zip(matched, unmatched):
            self.assertAlmostEqual(a["delta_qc_per_pair_w"], b["delta_qc_per_pair_w"], places=12)
        # Independent constant-property linear-m exact expression.
        gp, gn, i, slope = 1e-5*1e-6/(2*1.5*1e-12), 2e-5*1e-6/(2*1e-12), 1., 10e-6/30
        expected = i*slope*(gp-gn)*(300/6+30/12+(.5*(gp+gn))/30)
        self.assertAlmostEqual(matched[0]["delta_qc_per_pair_w"], expected, places=12)
        self.assertGreater(abs(expected), 1e-6)

    def test_central_difference_checks_the_adjoint(self):
        check = study.central_difference("constant", 1, "matched", "parabola", .01)
        self.assertLess(check["absolute_error_w"], 1e-10)
        self.assertLess(check["maximum_relative_energy_residual"], 1e-9)


if __name__ == "__main__":
    unittest.main()
