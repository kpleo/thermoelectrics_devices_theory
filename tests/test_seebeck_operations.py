"""Physical response identities distinct from implementation-level checks."""
import unittest

from scripts.analysis import seebeck_operations as operations


class OperationTests(unittest.TestCase):
    def test_constant_differential_has_only_endpoint_response(self):
        base, _ = operations.study.load_material_model("constant")
        result = operations.operation_check(base, 1., "differential", "constant", (.01,))
        self.assertAlmostEqual(result["predicted_total_derivative_w"], .003, places=14)
        self.assertEqual(result["distributed_derivative_w"], 0.)
        self.assertLess(result["checks"][0]["absolute_error_w"], 1e-11)

    def test_affine_differential_matches_polynomial_integral(self):
        base, _ = operations.study.load_material_model("constant")
        result = operations.operation_check(base, 1., "differential", "affine", (.01,))
        gp = 1e-5*1e-6/(2*1.5*1e-12)
        gn = 2e-5*1e-6/(2*1.*1e-12)
        def moment(g):
            return 300*30/2+30**2/6+g*300/6+g*30/12+g**2/60
        expected = .003+.5*(10e-6/30)*(moment(gp)+moment(gn))
        self.assertAlmostEqual(result["predicted_total_derivative_w"], expected, places=12)
        self.assertLess(result["checks"][0]["absolute_error_w"], 1e-9)

    def test_common_constant_is_null_but_common_affine_need_not_be(self):
        base, _ = operations.study.load_material_model("constant")
        constant = operations.operation_check(base, 1., "common", "constant", (.01,))
        affine = operations.operation_check(base, 1., "common", "affine", (.01,))
        self.assertEqual(constant["predicted_total_derivative_w"], 0.)
        self.assertLess(constant["checks"][0]["absolute_error_w"], 1e-11)
        self.assertGreater(abs(affine["predicted_total_derivative_w"]), 1e-6)
        for result in (constant, affine):
            self.assertLess(result["alpha_identity_max_error_v_per_k"], 1e-18)

    def test_global_shift_requires_common_measurement_reference(self):
        result = operations.global_reference_check()
        self.assertAlmostEqual(result["bare_nonisothermal_sample_voltage_change_v"], -25e-6*53, places=14)
        self.assertLess(abs(result["closed_measurement_voltage_change_v"]), 1e-14)
        self.assertEqual(result["node_heat_shift_sum_w"], 0.)

    def test_zero_current_cannot_establish_measure_equality(self):
        result = operations.zero_current_counterexample()
        self.assertGreater(result["measure_l1_normalized"], .01)
        for record in result["common_operation_checks"]:
            self.assertEqual(record["predicted_total_derivative_w"], 0.)
            self.assertLess(record["checks"][0]["absolute_error_w"], 1e-11)

    def test_matching_survives_temperature_turning_point(self):
        result = operations.constant_measure_matching()
        self.assertTrue(result["records"][-1]["internal_temperature_maximum"])
        for record in result["records"]:
            self.assertLess(record["measure_l1_normalized"], 1e-9)
            for check in record["checks"]:
                self.assertLess(abs(check["predicted_total_derivative_w"]), 1e-12)
                self.assertLess(check["checks"][-1]["absolute_error_w"], 1e-9)


if __name__ == "__main__":
    unittest.main()
