"""Small-mesh checks of finite-depth electrical lead constraints."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.analysis.electrode_spreading import depth_metrics, spreading_problem


class PackagedSpreadingTests(unittest.TestCase):
    def test_opposite_half_depth_patches_have_correct_electrical_constraints(self):
        problem,groups,metadata=spreading_problem(refinement=1)
        depth=metadata["geometry"]["depth_m"]
        y=problem.mesh.p[1]
        p=problem.electrodes["p"].nodes
        n=problem.electrodes["n"].nodes

        self.assertGreater(len(p),0)
        self.assertGreater(len(n),0)
        self.assertLess(len(p),len(groups["p_terminal"]))
        self.assertLess(len(n),len(groups["n_terminal"]))
        np.testing.assert_array_equal(p,groups["p_terminal"][y[groups["p_terminal"]]<=depth/2+1e-13])
        np.testing.assert_array_equal(n,groups["n_terminal"][y[groups["n_terminal"]]>=depth/2-1e-13])
        self.assertAlmostEqual(float(np.min(y[p])),0.)
        self.assertAlmostEqual(float(np.max(y[p])),depth/2)
        self.assertAlmostEqual(float(np.min(y[n])),depth/2)
        self.assertAlmostEqual(float(np.max(y[n])),depth)
        self.assertEqual(len(np.intersect1d(p,n)),0)

        # Two physical terminals have one free terminal voltage and one ground.
        self.assertEqual(problem.ground,"n")
        self.assertEqual(set(problem.terminal_columns),{"p"})
        p_column=problem.terminal_columns["p"]
        self.assertEqual(problem.P[p].nnz,len(p))
        np.testing.assert_array_equal(problem.P[p,p_column].toarray().ravel(),np.ones(len(p)))
        self.assertEqual(problem.P[n].nnz,0)
        self.assertEqual(problem.current_rhs[p_column],3.)
        self.assertEqual(problem.electrodes["n"].inward_current_a,-3.)

        # Removed parts of the old electrodes remain free conductor nodes.
        for old,patch in ((groups["p_terminal"],p),(groups["n_terminal"],n)):
            released=np.setdiff1d(old,patch)
            self.assertGreater(len(released),0)
            self.assertEqual(problem.P[released].nnz,len(released))
            self.assertEqual(problem.P[released,p_column].nnz,0)

    def test_coarse_solution_is_finite_and_ceramic_has_no_electric_current(self):
        problem,_,_=spreading_problem(refinement=1)
        result=problem.solve()
        metrics,current_density=depth_metrics(problem,result)

        for field in (result.temperature_k,result.potential_v,current_density,
                      result.thermal_reaction_w,result.electrical_reaction_a):
            self.assertTrue(np.all(np.isfinite(field)))
        self.assertTrue(np.isfinite(result.summary["inward_heat_w"]["cold"]))
        self.assertLess(abs(result.summary["energy_residual_w"]),1e-9)
        self.assertLess(abs(result.summary["current_balance_a"]),1e-9)
        self.assertGreater(metrics["thermal_only_node_count"],0)
        self.assertTrue(metrics["thermal_only_nodes_have_no_electrical_unknowns"])
        self.assertEqual(metrics["ceramic_max_abs_current_a_per_m2"],0.)
        self.assertGreater(metrics["conducting_y_current_rms_fraction"],.1)
        for electrode in problem.electrodes.values():
            self.assertEqual(float(np.ptp(result.potential_v[electrode.nodes])),0.)


if __name__=="__main__":
    unittest.main()
