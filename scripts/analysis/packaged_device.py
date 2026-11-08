"""Representative 3D thermoelectric pair with electrodes and thermal plates.

Published axial BTS/BST properties are used as scalar constitutive laws.  The
layer thicknesses, gap and contact laws below are explicit model assumptions,
not a reconstruction of a measured module.  The external ceramic faces, not
the semiconductor ends, are held at prescribed reservoir temperatures.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys

import numpy as np
from skfem import MeshTet

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.thermoelectric_fem import (
    CoupledFEM, Electrode, Material, common_mode, constant, load_bts_bst,
    discrete_adjoint_cold_response,
)


@dataclass(frozen=True)
class Geometry:
    total_leg_area_m2: float = 2 * (1.95e-3)**2
    depth_m: float = 1.95e-3
    leg_length_m: float = 4.8e-3
    gap_m: float = 1.0e-3
    copper_thickness_m: float = 0.20e-3
    contact_thickness_m: float = 0.02e-3
    ceramic_thickness_m: float = 0.50e-3
    ceramic_kappa_w_mk: float = 25.0
    copper_kappa_w_mk: float = 400.0
    copper_resistivity_ohm_m: float = 1.72e-8
    copper_seebeck_v_k: float = 1.5e-6
    contact_electrical_resistance_ohm_m2: float = 3.5e-10
    contact_thermal_resistance_m2_k_w: float = 1.0e-6


def segments(ends, counts):
    return np.concatenate([np.linspace(a, b, count + 1)[:-1]
                           for a, b, count in zip(ends[:-1], ends[1:], counts)] + [[ends[-1]]])


def make_mesh(geometry: Geometry, area_ratio: float, refinement: int):
    if area_ratio <= 0 or refinement < 1:
        raise ValueError("positive area ratio and refinement are required")
    g = geometry
    wp = g.total_leg_area_m2 / (1 + area_ratio) / g.depth_m
    wn = wp * area_ratio
    xends = [0, wp, wp + g.gap_m, wp + g.gap_m + wn]
    zends = np.cumsum([0, g.ceramic_thickness_m, g.copper_thickness_m,
                      g.contact_thickness_m, g.leg_length_m,
                      g.contact_thickness_m, g.copper_thickness_m,
                      g.ceramic_thickness_m])
    mesh = MeshTet.init_tensor(segments(xends, [2*refinement]*3),
                              np.linspace(0, g.depth_m, 2*refinement+1),
                              segments(zends, [refinement]*3+[8*refinement]+[refinement]*3))

    def classify(current_mesh):
        x, _, z = current_mesh.p[:, current_mesh.t].mean(axis=1)
        in_p, in_n = x < wp, x > wp + g.gap_m
        in_legs = in_p | in_n
        ids = np.full(current_mesh.nelements, -1, dtype=int)
        ids[(z < zends[1]) | (z > zends[6])] = 3  # electrically insulating plates
        ids[((z > zends[1]) & (z < zends[2]) & in_legs)
            | ((z > zends[5]) & (z < zends[6]))] = 2  # separate cold pads; hot bridge
        ids[in_legs & (((z > zends[2]) & (z < zends[3]))
                       | ((z > zends[4]) & (z < zends[5])))] = 4
        ids[(z > zends[3]) & (z < zends[4]) & in_p] = 0
        ids[(z > zends[3]) & (z < zends[4]) & in_n] = 1
        return ids

    mesh = mesh.remove_elements(np.flatnonzero(classify(mesh) < 0)).remove_unused_nodes()
    ids = classify(mesh)
    if np.any(ids < 0):
        raise RuntimeError("air cells were not removed")
    x, _, z = mesh.p
    close = lambda a, b: np.isclose(a, b, rtol=0, atol=1e-12)
    # Side faces of the cold copper pads are equipotential electrical contacts.
    terminal_z = (z >= zends[1]-1e-12) & (z <= zends[2]+1e-12)
    groups = {
        "cold": np.flatnonzero(close(z, 0)),
        "hot": np.flatnonzero(close(z, zends[-1])),
        "p_terminal": np.flatnonzero(close(x, 0) & terminal_z),
        "n_terminal": np.flatnonzero(close(x, xends[-1]) & terminal_z),
        "p_cold_leg": np.flatnonzero(close(z, zends[3]) & (x <= wp+1e-12)),
        "n_cold_leg": np.flatnonzero(close(z, zends[3]) & (x >= wp+g.gap_m-1e-12)),
        "p_hot_leg": np.flatnonzero(close(z, zends[4]) & (x <= wp+1e-12)),
        "n_hot_leg": np.flatnonzero(close(z, zends[4]) & (x >= wp+g.gap_m-1e-12)),
    }
    if any(len(nodes) == 0 for nodes in groups.values()):
        raise RuntimeError("a boundary group is empty")
    volumes = np.bincount(ids, weights=np.abs(np.linalg.det(
        np.moveaxis(mesh.p[:, mesh.t[1:]] - mesh.p[:, mesh.t[0]][:, None, :], 2, 0))) / 6,
        minlength=5)
    expected = np.array([
        g.total_leg_area_m2/(1+area_ratio)*g.leg_length_m,
        g.total_leg_area_m2*area_ratio/(1+area_ratio)*g.leg_length_m,
        (g.total_leg_area_m2+xends[-1]*g.depth_m)*g.copper_thickness_m,
        2*xends[-1]*g.depth_m*g.ceramic_thickness_m,
        2*g.total_leg_area_m2*g.contact_thickness_m,
    ])
    if not np.allclose(volumes, expected, rtol=1e-10, atol=1e-22):
        raise RuntimeError(f"material volumes differ: {volumes} vs {expected}")
    metadata = {"geometry": asdict(g), "area_ratio_n_over_p": area_ratio,
                "refinement": refinement, "layer_z_m": list(zends), "width_x_m": xends[-1],
                "material_volumes_m3": list(volumes), "material_names": ["p", "n", "copper", "ceramic", "contact"]}
    return mesh, ids, groups, metadata


def problem(*, area_ratio=1.0, refinement=1, epsilon=0.0, mode="endpoint_zero",
            current_a=3.0, cold_k=323.0, hot_k=373.0, geometry=None):
    g = geometry or Geometry()
    mesh, ids, groups, metadata = make_mesh(g, area_ratio, refinement)
    p, n, _ = load_bts_bst()
    materials = [common_mode(mat, epsilon, mode, cold_k, hot_k) for mat in (p, n)]
    materials.extend([
        Material("nominal copper", constant(g.copper_kappa_w_mk), constant(g.copper_seebeck_v_k),
                 constant(g.copper_resistivity_ohm_m)),
        Material("nominal thermal plate", constant(g.ceramic_kappa_w_mk)),
        Material("nominal effective contact layer", constant(g.contact_thickness_m/g.contact_thermal_resistance_m2_k_w),
                 constant(0.0), constant(g.contact_electrical_resistance_ohm_m2/g.contact_thickness_m)),
    ])
    instance = CoupledFEM(mesh, materials, ids,
                          {"cold": (groups["cold"], cold_k), "hot": (groups["hot"], hot_k)},
                          {"p": Electrode(groups["p_terminal"], current_a),
                           "n": Electrode(groups["n_terminal"], -current_a)}, "n",
                          cold_k+(hot_k-cold_k)*mesh.p[2]/metadata["layer_z_m"][-1])
    return instance, groups, metadata


def run_case(*, output: Path, area_ratio=1.0, refinement=1, epsilons=(0.02, 0.01),
             mode="endpoint_zero", current_a=3.0, geometry=None):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite a completed case: {output}")
    base, groups, metadata = problem(area_ratio=area_ratio, refinement=refinement,
                                      current_a=current_a, geometry=geometry)
    result = base.solve()
    payload = {"scope": "representative_3d_device_not_experimental_module_reconstruction",
               "material_source_doi": "10.1093/nsr/nwae448",
               "property_direction": "published_curves_assigned_to_leg_axis_and_used_as_scalar_laws",
               "external_reservoir_temperatures_k": [323.0, 373.0],
               "boundary_conditions": "fixed external plate temperatures; insulated remaining thermal surfaces; equipotential cold copper side electrodes with fixed total current",
               "cold_heat_definition": "inward total heat through external cold ceramic face",
               "geometry_and_discretization": metadata,
               "current_a": current_a,
               "mode": "reservoir_anchored_parabola" if mode == "endpoint_zero" else mode,
               "internal_mode_name": mode, "mode_amplitude_v_k": 10e-6,
               "perturbation_scope": "semiconductor S only; reference electrodes and contacts fixed",
               "endpoint_interpretation": "perturbation is not constrained to vanish at semiconductor faces; full response includes endpoint and state-feedback effects",
               "baseline": result.summary, "centered_responses": [],
               "leg_face_temperature_ranges_k": {
                   name: [float(np.min(result.temperature_k[nodes])), float(np.max(result.temperature_k[nodes]))]
                   for name, nodes in groups.items() if "leg" in name},
               "status": "computed_pending_convergence_comparison"}
    # A centered derivative is a response of the whole composite device.  Do
    # not silently substitute the ideal fixed-leg-end collection formula.
    for eps in epsilons:
        pair = []
        problems = []
        for sign in (1, -1):
            changed, _, _ = problem(area_ratio=area_ratio, refinement=refinement,
                                     epsilon=sign*eps, mode=mode, current_a=current_a, geometry=geometry)
            pair.append(changed.solve(initial=result))
            problems.append(changed)
        derivative = (pair[0].summary["inward_heat_w"]["cold"]-pair[1].summary["inward_heat_w"]["cold"])/(2*eps)
        adjoint = discrete_adjoint_cold_response(base, result, problems[0], problems[1], eps)
        payload["centered_responses"].append({"epsilon": eps, "cold_heat_derivative_w": derivative,
                                               "adjoint": adjoint,
                                               "adjoint_difference_w": derivative-adjoint["derivative_w"],
                                               "plus": pair[0].summary, "minus": pair[1].summary})
    all_summaries = [result.summary]+[item[side] for item in payload["centered_responses"] for side in ("plus", "minus")]
    payload["checks"] = {
        "max_abs_energy_residual_w": max(abs(item["energy_residual_w"]) for item in all_summaries),
        "max_abs_current_balance_a": max(abs(item["current_balance_a"]) for item in all_summaries),
        "max_electrode_spread_v": max(max(item["electrode_potential_spread_v"].values()) for item in all_summaries),
    }
    if payload["checks"]["max_abs_energy_residual_w"] > 1e-7 or payload["checks"]["max_abs_current_balance_a"] > 1e-7:
        raise RuntimeError("conservation checks failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2)+"\n")
    np.savez_compressed(output.with_suffix(".npz"), coordinates_m=base.mesh.p, cells=base.mesh.t,
                        material_ids=base.ids, temperature_k=result.temperature_k, potential_v=result.potential_v)
    print(json.dumps({"output": str(output), "nodes": result.summary["nodes"],
                      "cold_heat_w": result.summary["inward_heat_w"]["cold"],
                      "responses_w": [i["cold_heat_derivative_w"] for i in payload["centered_responses"]],
                      "checks": payload["checks"]}))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--area-ratio", type=float, default=1.0)
    parser.add_argument("--refinement", type=int, default=1)
    parser.add_argument("--mode", choices=["endpoint_zero", "linear", "constant"], default="endpoint_zero")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.02, 0.01])
    parser.add_argument("--current", type=float, default=3.0)
    args = parser.parse_args()
    run_case(output=args.output, area_ratio=args.area_ratio, refinement=args.refinement,
             mode=args.mode, epsilons=args.epsilons, current_a=args.current)


if __name__ == "__main__":
    main()
