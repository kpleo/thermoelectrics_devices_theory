"""Geometry and electrical topology checks for the representative 3D pair."""
import unittest

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from scripts.analysis.packaged_device import Geometry, make_mesh, problem


def cell_graph_components(mesh, cells):
    nodes = np.unique(mesh.t[:, cells])
    lookup = np.full(mesh.nvertices, -1, dtype=int)
    lookup[nodes] = np.arange(len(nodes))
    blocks = mesh.t[:, cells]
    rows, cols = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            rows.extend(lookup[blocks[i]])
            cols.extend(lookup[blocks[j]])
            rows.extend(lookup[blocks[j]])
            cols.extend(lookup[blocks[i]])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
    return connected_components(graph, directed=False, return_labels=False)


class PackagedDeviceGeometryTests(unittest.TestCase):
    def test_layer_volumes_match_independent_analytic_geometry(self):
        g = Geometry()
        for ratio, refinement in ((1., 1), (1.0424, 1), (.75, 2)):
            mesh, ids, _, meta = make_mesh(g, ratio, refinement)
            vertices = mesh.p[:, mesh.t].transpose(2, 1, 0)
            tetra_volume = np.abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1])) / 6
            observed = np.bincount(ids, weights=tetra_volume, minlength=5)
            total_width = g.total_leg_area_m2 / g.depth_m + g.gap_m
            total_plate_area = total_width * g.depth_m
            expected = np.array([
                g.total_leg_area_m2 / (1+ratio) * g.leg_length_m,
                g.total_leg_area_m2 * ratio / (1+ratio) * g.leg_length_m,
                (g.total_leg_area_m2 + total_plate_area) * g.copper_thickness_m,
                2 * total_plate_area * g.ceramic_thickness_m,
                2 * g.total_leg_area_m2 * g.contact_thickness_m,
            ])
            np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-22)
            self.assertAlmostEqual(meta['width_x_m'], total_width)
            self.assertEqual(set(ids.tolist()), {0, 1, 2, 3, 4})

    def test_empty_gap_and_solid_layer_classification(self):
        g = Geometry()
        ratio = 1.0424
        mesh, ids, _, meta = make_mesh(g, ratio, 1)
        centroids = mesh.p[:, mesh.t].mean(axis=1)
        x, z = centroids[0], centroids[2]
        wp = g.total_leg_area_m2 / (1+ratio) / g.depth_m
        layers = meta['layer_z_m']
        gap = (x > wp) & (x < wp + g.gap_m)
        void_height = (z > layers[1]) & (z < layers[5])
        self.assertFalse(np.any(gap & void_height))
        self.assertTrue(np.all(ids[gap & (z > layers[5]) & (z < layers[6])] == 2))
        self.assertTrue(np.all(ids[(z < layers[1]) | (z > layers[6])] == 3))

    def test_electrical_connectivity_only_through_hot_bridge(self):
        g = Geometry()
        mesh, ids, _, meta = make_mesh(g, 1, 1)
        z = mesh.p[2, mesh.t].mean(axis=0)
        conducting = ids != 3
        self.assertEqual(cell_graph_components(mesh, np.flatnonzero(conducting)), 1)
        hot_bridge = (ids == 2) & (z > meta['layer_z_m'][5])
        self.assertEqual(cell_graph_components(mesh, np.flatnonzero(conducting & ~hot_bridge)), 2)

    def test_boundary_groups_are_on_correct_surfaces(self):
        g = Geometry()
        mesh, _, groups, meta = make_mesh(g, 1.0424, 1)
        zlevels = {'cold': 0, 'hot': meta['layer_z_m'][-1],
                   'p_cold_leg': meta['layer_z_m'][3], 'n_cold_leg': meta['layer_z_m'][3],
                   'p_hot_leg': meta['layer_z_m'][4], 'n_hot_leg': meta['layer_z_m'][4]}
        boundary_nodes = mesh.boundary_nodes()
        for name, z in zlevels.items():
            np.testing.assert_allclose(mesh.p[2, groups[name]], z, atol=1e-13)
        for name in ('cold', 'hot', 'p_terminal', 'n_terminal'):
            self.assertTrue(np.all(np.isin(groups[name], boundary_nodes)))
        self.assertEqual(len(np.intersect1d(groups['cold'], groups['hot'])), 0)
        self.assertEqual(len(np.intersect1d(groups['p_terminal'], groups['n_terminal'])), 0)
        for name, x in [('p_terminal', 0), ('n_terminal', meta['width_x_m'])]:
            np.testing.assert_allclose(mesh.p[0, groups[name]], x, atol=1e-13)
            z = mesh.p[2, groups[name]]
            self.assertTrue(np.all(z >= meta['layer_z_m'][1]-1e-13))
            self.assertTrue(np.all(z <= meta['layer_z_m'][2]+1e-13))

    def test_ceramic_has_no_independent_electrical_unknown(self):
        instance, groups, _ = problem(refinement=1)
        conducting_nodes = np.unique(instance.mesh.t[:, instance.ids != 3])
        ceramic_only_nodes = np.setdiff1d(np.unique(instance.mesh.t[:, instance.ids == 3]), conducting_nodes)
        self.assertGreater(len(ceramic_only_nodes), 0)
        self.assertEqual(instance.P[ceramic_only_nodes].nnz, 0)
        self.assertTrue(np.all(np.isin(groups['p_terminal'], conducting_nodes)))
        self.assertTrue(np.all(np.isin(groups['n_terminal'], conducting_nodes)))
        self.assertEqual(instance.P[groups['n_terminal']].nnz, 0)
        electrode_map = instance.P[groups['p_terminal']].toarray()
        np.testing.assert_array_equal(electrode_map, np.repeat(electrode_map[:1], len(electrode_map), axis=0))
        self.assertEqual(np.sum(electrode_map[0]), 1)
        self.assertEqual(len(instance.terminal_columns), 1)
        self.assertAlmostEqual(np.sum(instance.current_rhs), 3)

    def test_effective_contact_parameters_have_correct_areal_units(self):
        g = Geometry()
        instance, _, _ = problem(geometry=g)
        contact = instance.materials[4]
        rho = float(contact.resistivity.evaluate(np.array([350.]))[0])
        kappa = float(contact.kappa.evaluate(np.array([350.]))[0])
        self.assertAlmostEqual(rho*g.contact_thickness_m, g.contact_electrical_resistance_ohm_m2)
        self.assertAlmostEqual(g.contact_thickness_m/kappa, g.contact_thermal_resistance_m2_k_w)


if __name__ == '__main__':
    unittest.main()
