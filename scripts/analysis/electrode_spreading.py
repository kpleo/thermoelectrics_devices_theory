#!/usr/bin/env python3
"""Three-dimensional current spreading from finite-depth side electrodes.

The p-side copper electrode occupies the front half of the device depth and
the n-side copper electrode the back half.  This breaks the depth invariance
of the reference packaged geometry without changing its material layout.
Remaining external electrical surfaces are insulating.  External ceramic
reservoir temperatures, current and all constitutive laws are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts.analysis.packaged_device import problem as full_depth_problem
from scripts.analysis.thermoelectric_fem import CoupledFEM,Electrode,discrete_adjoint_cold_response

OUTPUT=ROOT/"results/transport_validation/fem_spreading"


def spreading_problem(*,area_ratio=1.,refinement=1,epsilon=0.,current_a=3.):
    base,groups,metadata=full_depth_problem(area_ratio=area_ratio,refinement=refinement,
                                           epsilon=epsilon,mode="endpoint_zero",current_a=current_a)
    depth=metadata["geometry"]["depth_m"]
    y=base.mesh.p[1]
    if not np.any(np.isclose(y,depth/2,atol=1e-13,rtol=0)):
        raise ValueError("mesh must resolve the electrode boundary at half depth")
    pnodes=groups["p_terminal"][y[groups["p_terminal"]]<=depth/2+1e-13]
    nnodes=groups["n_terminal"][y[groups["n_terminal"]]>=depth/2-1e-13]
    if len(pnodes)==len(groups["p_terminal"]) or len(nnodes)==len(groups["n_terminal"]):
        raise ValueError("localized electrode must be smaller than the full-depth side")
    localized=CoupledFEM(base.mesh,base.materials,base.ids,base.thermal_boundaries,
                         {"p":Electrode(pnodes,current_a),"n":Electrode(nnodes,-current_a)},
                         "n",base.initial_temperature)
    metadata["electrode_localization"]={"p_y_interval_m":[0.,depth/2],"n_y_interval_m":[depth/2,depth],
                                        "p_nodes":len(pnodes),"n_nodes":len(nnodes),
                                        "full_depth_p_nodes":len(groups["p_terminal"]),
                                        "full_depth_n_nodes":len(groups["n_terminal"])}
    return localized,groups,metadata


def depth_metrics(problem,result):
    xz=np.round(problem.mesh.p[[0,2]].T,12)
    _,groups=np.unique(xz,axis=0,return_inverse=True)
    ranges=[]
    for group in np.unique(groups):
        nodes=np.flatnonzero(groups==group)
        values=result.temperature_k[nodes]
        ranges.append((float(np.ptp(values)),nodes[int(np.argmin(values))],nodes[int(np.argmax(values))]))
    delta,cold_node,hot_node=max(ranges,key=lambda entry:entry[0])
    parameters=problem._parameters(result.temperature_k,result.potential_v)
    j=parameters["j"]
    weight=problem.basis.dx
    names=[material.name for material in problem.materials]
    per_material={}
    for index,name in enumerate(names):
        mask=problem.ids==index
        volume=float(np.sum(weight[mask]))
        square=float(np.sum(np.sum(j[:,mask]**2,axis=0)*weight[mask]))
        y_square=float(np.sum(j[1,mask]**2*weight[mask]))
        per_material[name]={"current_rms_a_per_m2":np.sqrt(square/volume),
                            "y_current_rms_a_per_m2":np.sqrt(y_square/volume),
                            "y_current_rms_fraction":np.sqrt(y_square/square) if square else 0.,
                            "volume_m3":volume}
    conducting=np.asarray([problem.materials[index].conducting for index in problem.ids])
    all_square=float(np.sum(np.sum(j[:,conducting]**2,axis=0)*weight[conducting]))
    all_y_square=float(np.sum(j[1,conducting]**2*weight[conducting]))
    conductor_nodes=np.unique(problem.mesh.t[:,conducting])
    ceramic_nodes=np.unique(problem.mesh.t[:,~conducting])
    only_thermal=np.setdiff1d(ceramic_nodes,conductor_nodes)
    return {"max_same_xz_depth_temperature_difference_k":delta,
            "depth_temperature_extreme_coordinates_m":{"minimum_T":list(problem.mesh.p[:,cold_node]),
                                                        "maximum_T":list(problem.mesh.p[:,hot_node])},
            "conducting_y_current_rms_fraction":np.sqrt(all_y_square/all_square),
            "per_material":per_material,
            "thermal_only_node_count":len(only_thermal),
            "thermal_only_nodes_have_no_electrical_unknowns":problem.P[only_thermal].nnz==0,
            "ceramic_max_abs_current_a_per_m2":float(np.max(np.abs(j[:,~conducting]))),
            "internal_temperature_minimum_k":float(np.min(result.temperature_k)),
            "internal_temperature_maximum_k":float(np.max(result.temperature_k))},j


def run_case(area_ratio,refinement,output):
    started=time.perf_counter()
    base,groups,metadata=spreading_problem(area_ratio=area_ratio,refinement=refinement)
    result=base.solve()
    metrics,j=depth_metrics(base,result)
    eps=.01
    shifted=[]
    states=[]
    for sign in (1,-1):
        changed,_,_=spreading_problem(area_ratio=area_ratio,refinement=refinement,epsilon=sign*eps)
        shifted.append(changed)
        states.append(changed.solve(initial=result))
    derivative=(states[0].summary["inward_heat_w"]["cold"]-states[1].summary["inward_heat_w"]["cold"])/(2*eps)
    adjoint=discrete_adjoint_cold_response(base,result,*shifted,eps)
    summaries=[result.summary]+[state.summary for state in states]
    payload={"scope":"representative true-three-dimensional localized-lead case, not experimental reconstruction",
             "material_source_doi":"10.1093/nsr/nwae448",
             "properties":"published scalar BTS/BST curves; copper/contact/ceramic laws and dimensions are explicit assumptions",
             "geometry_and_discretization":metadata,"baseline":result.summary,
             "mode":"reservoir-anchored parabola, S perturbation only in p/n semiconductor",
             "mode_amplitude_v_per_k":1e-5,"epsilon":eps,
             "cold_heat_derivative_w":derivative,"discrete_adjoint":adjoint,
             "adjoint_difference_w":derivative-adjoint["derivative_w"],
             "plus":states[0].summary,"minus":states[1].summary,
             "depth_nonuniformity":metrics,
             "leg_face_temperature_ranges_k":{name:[float(np.min(result.temperature_k[nodes])),float(np.max(result.temperature_k[nodes]))]
                                               for name,nodes in groups.items() if "leg" in name},
             "checks":{"maximum_absolute_energy_residual_w":max(abs(item["energy_residual_w"]) for item in summaries),
                       "maximum_absolute_current_balance_a":max(abs(item["current_balance_a"]) for item in summaries),
                       "maximum_electrode_potential_spread_v":max(max(item["electrode_potential_spread_v"].values()) for item in summaries)},
             "convergence_interpretation":"finite electrode edges cause local field concentration; compare absolute response differences across meshes without forcing a relative cancellation tolerance",
             "elapsed_seconds":time.perf_counter()-started}
    if metrics["max_same_xz_depth_temperature_difference_k"]<1e-8 or metrics["conducting_y_current_rms_fraction"]<1e-5:
        raise RuntimeError("localized electrodes did not generate a resolved three-dimensional field")
    if not metrics["thermal_only_nodes_have_no_electrical_unknowns"] or metrics["ceramic_max_abs_current_a_per_m2"]!=0:
        raise RuntimeError("thermal-only ceramic acquired an electrical current")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    j_cell=np.sum(j*base.basis.dx,axis=-1)/np.sum(base.basis.dx,axis=-1)
    np.savez_compressed(output.with_suffix(".npz"),coordinates_m=base.mesh.p,cells=base.mesh.t,material_ids=base.ids,
                        temperature_k=result.temperature_k,potential_v=result.potential_v,current_density_a_per_m2=j_cell,
                        p_electrode_nodes=base.electrodes["p"].nodes,n_electrode_nodes=base.electrodes["n"].nodes)
    print(json.dumps({"output":str(output),"nodes":int(base.basis.N),"ratio":area_ratio,"refinement":refinement,
                      "delta_T_y_k":metrics["max_same_xz_depth_temperature_difference_k"],
                      "jy_rms_fraction":metrics["conducting_y_current_rms_fraction"],
                      "response_w":derivative,"adjoint_difference_w":payload["adjoint_difference_w"],
                      "elapsed_seconds":payload["elapsed_seconds"]}),flush=True)
    return payload


def summarize(output_directory):
    rows=[]
    for path in sorted(output_directory.glob("ratio*_refine*.json")):
        item=json.loads(path.read_text())
        meta=item["geometry_and_discretization"]
        rows.append({"file":path.name,"ratio":meta["area_ratio_n_over_p"],"refinement":meta["refinement"],
                     "cold_heat_w":item["baseline"]["inward_heat_w"]["cold"],
                     "response_w":item["cold_heat_derivative_w"],"adjoint_difference_w":item["adjoint_difference_w"],
                     "raw_max_depth_temperature_difference_k":item["depth_nonuniformity"]["max_same_xz_depth_temperature_difference_k"],
                     "conducting_y_current_rms_fraction":item["depth_nonuniformity"]["conducting_y_current_rms_fraction"]})
    controls=[]
    for refinement in (1,2,4):
        suffix="ratio1_refine1_adjoint" if refinement==1 else f"ratio1_refine{refinement}"
        full_path=ROOT/f"results/transport_validation/fem_packaged/{suffix}.npz"
        local_path=output_directory/f"ratio1.00000000_refine{refinement}.npz"
        if not full_path.exists() or not local_path.exists():
            continue
        full=np.load(full_path)
        local=np.load(local_path)
        if not np.array_equal(full["coordinates_m"],local["coordinates_m"]):
            raise ValueError("depth-invariance control uses a different mesh")
        base,_,_=full_depth_problem(refinement=refinement)
        metrics,_=depth_metrics(base,SimpleNamespace(temperature_k=full["temperature_k"],potential_v=full["potential_v"]))
        _,group=np.unique(np.round(base.mesh.p[[0,2]].T,12),axis=0,return_inverse=True)
        def span(field):
            return max(float(np.ptp(field[group==g])) for g in np.unique(group))
        controls.append({"refinement":refinement,"full_depth_source":str(full_path.relative_to(ROOT)),
                         "full_depth_y_current_rms_fraction":metrics["conducting_y_current_rms_fraction"],
                         "full_depth_raw_max_depth_temperature_difference_k":metrics["max_same_xz_depth_temperature_difference_k"],
                         "localized_minus_full_depth_temperature_span_k":span(local["temperature_k"]-full["temperature_k"]),
                         "full_depth_potential_span_v":span(full["potential_v"]),
                         "localized_potential_span_v":span(local["potential_v"])})
    differences=[]
    for ratio in sorted({row["ratio"] for row in rows}):
        selected=sorted((row for row in rows if row["ratio"]==ratio),key=lambda row:row["refinement"])
        for coarse,fine in zip(selected[:-1],selected[1:]):
            differences.append({"ratio":ratio,"coarse_refinement":coarse["refinement"],"fine_refinement":fine["refinement"],
                                "absolute_response_difference_w":abs(fine["response_w"]-coarse["response_w"])})
    summary={"cases":rows,"successive_mesh_differences":differences,"full_depth_controls":controls,
             "interpretation":["Finite-depth electrodes create genuine three-dimensional current spreading.",
                               "Uncorrected maximum cross-depth temperature variation also contains tetrahedral discretization error; compare with the full-depth control on the same mesh.",
                               "The localized-minus-full-depth temperature field isolates the response to electrode localization on each mesh.",
                               "Current spreading is strongest in copper. The semiconductor legs remain much closer to axial transport.",
                               "Global heat-response convergence is assessed through absolute mesh differences; local electrode-edge peaks are not treated as converged physical maxima."]}
    (output_directory/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinements",type=int,nargs="+",default=[1,2,4])
    parser.add_argument("--ratios",type=float,nargs="+",default=[1.,1.04240001])
    parser.add_argument("--output-directory",type=Path,default=OUTPUT)
    parser.add_argument("--summarize-only",action="store_true")
    args=parser.parse_args()
    for refinement in ([] if args.summarize_only else args.refinements):
        for ratio in args.ratios:
            name=f"ratio{ratio:.8f}_refine{refinement}.json"
            output=args.output_directory/name
            if output.exists():
                saved=json.loads(output.read_text())
                print(json.dumps({"existing_result":str(output),"response_w":saved["cold_heat_derivative_w"]}),flush=True)
                continue
            run_case(ratio,refinement,output)
    summarize(args.output_directory)


if __name__=="__main__":
    main()
