#!/usr/bin/env python3
"""Independent finite elements for publication-derived PbSe/Cr input curves.

Only the original material property laws are shared with the one-dimensional
study.  The governing equations, electric fields and total heat reactions are
solved by the independent tetrahedral finite-element code.  These scalar
source-table-driven calculations are not a reconstruction of an as-built
module; the publication does not uniquely bind all curves to that module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.geometry_matching import load_material_model
from scripts.analysis.thermoelectric_fem import Material, PropertyLaw, solve_pair

OUTPUT = ROOT/"results/transport_validation/fem_pbse"
GEOMETRY = ROOT/"results/transport_validation/geometry"
MESHES = {"coarse":(1,12),"medium":(2,24),"fine":(3,48)}


def read_csv(path):
    with path.open(newline="",encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_pbse_inputs():
    """Wrap original laws and their derivatives, without invoking its solver."""
    couple,source=load_material_model("pbse")
    materials=[]
    domains={}
    for branch,leg in (("p",couple.p_leg),("n",couple.n_leg)):
        def wrap(law):
            return PropertyLaw(law.evaluate,law.derivative,
                               (law.minimum_temperature_k,law.maximum_temperature_k))
        materials.append(Material(f"publication-derived {branch}-PbSe/Cr",
                                  wrap(leg.thermal_conductivity),wrap(leg.seebeck),
                                  wrap(leg.electrical_resistivity)))
        domains[branch]={name:list(law.domain) for name,law in
                         (("S",materials[-1].seebeck),("rho",materials[-1].resistivity),("kappa",materials[-1].kappa))}
    root_table=GEOMETRY/"mode_specific_roots_and_measure_minima.csv"
    roots=[row for row in read_csv(root_table)
           if row["system"]=="pbse" and row["objective"]=="parabola" and row["kind"]=="bracketed_root"]
    if len(roots)!=1:
        raise ValueError("expected one PbSe endpoint-zero root from the geometry study")
    root=roots[0]
    if couple.p_leg.length_m!=couple.n_leg.length_m:
        raise ValueError("this driver requires equal leg lengths")
    common={"current_a":source["current_a"],"length_m":couple.p_leg.length_m,
            "cold_k":couple.cold_temperature_k,"hot_k":couple.hot_temperature_k}
    for field,name in (("current_a","current_a"),("cold_temperature_k","cold_k"),("hot_temperature_k","hot_k")):
        if not np.isclose(float(root[field]),common[name],rtol=0,atol=1e-10):
            raise ValueError("geometry-root operating point differs from the material model")
    return materials,couple.p_leg.area_m2+couple.n_leg.area_m2,common,domains,source,root


def references(root):
    original=[row for row in read_csv(GEOMETRY/"geometry_scan.csv")
              if row["system"]=="pbse" and abs(float(row["area_ratio_n_over_p"])-1)<1e-12]
    if len(original)!=1:
        raise ValueError("equal-area one-dimensional reference is missing or duplicated")
    responses=[row for row in read_csv(GEOMETRY/"common_mode_response_scan.csv")
               if row["system"]=="pbse" and abs(float(row["area_ratio_n_over_p"])-1)<1e-12]
    return {"original":{"cold_heat_w":float(original[0]["qc_per_pair_w"]),
                        "linear":float(next(row["delta_qc_per_pair_w"] for row in responses if row["mode"]=="linear")),
                        "endpoint_zero":float(next(row["delta_qc_per_pair_w"] for row in responses if row["mode"]=="parabola"))},
            "adjusted":{"cold_heat_w":float(root["qc_per_pair_w"]),"linear":float(root["linear_delta_qc_w"]),
                        "endpoint_zero":float(root["parabola_delta_qc_w"])}}


def check_field_domain(result,domains):
    records={}
    for branch,leg in result["legs"].items():
        lo,hi=leg["temperature_range_k"]
        available_lo=max(domain[0] for domain in domains[branch].values())
        available_hi=min(domain[1] for domain in domains[branch].values())
        records[branch]={"minimum_k":lo,"maximum_k":hi,"property_minimum_k":available_lo,
                         "property_maximum_k":available_hi,"no_extrapolation":lo>=available_lo-1e-9 and hi<=available_hi+1e-9}
    if not all(row["no_extrapolation"] for row in records.values()):
        raise ValueError("finite-element temperature leaves a material property domain")
    return records


def run_level(level):
    started=time.perf_counter()
    (p,n),total_area,common,domains,source,root=load_pbse_inputs()
    ref=references(root)
    nt,nz=MESHES[level]
    cases=[]
    for name,ratio in (("original",1.),("adjusted",float(root["area_ratio_n_over_p"]))):
        kwargs={**common,"p_area_m2":total_area/(1+ratio),"n_area_m2":total_area*ratio/(1+ratio),
                "transverse_cells":nt,"axial_cells":nz}
        base=solve_pair(p,n,**kwargs)
        field_checks=[check_field_domain(base,domains)]
        modes={}
        for mode in ("linear","endpoint_zero"):
            epsilons=(.03,.01,.003) if level=="medium" and mode=="endpoint_zero" else (.01,)
            values=[]
            for eps in epsilons:
                plus,minus=(solve_pair(p,n,**kwargs,epsilon=e,mode=mode) for e in (eps,-eps))
                field_checks.extend([check_field_domain(plus,domains),check_field_domain(minus,domains)])
                derivative=(plus["cold_heat_w"]-minus["cold_heat_w"])/(2*eps)
                values.append({"epsilon":eps,"derivative_w":derivative,
                               "one_dimensional_reference_w":ref[name][mode],
                               "absolute_error_to_1d_w":abs(derivative-ref[name][mode]),
                               "plus":plus,"minus":minus})
            modes[mode]=values
        all_results=[base]+[entry[sign] for values in modes.values() for entry in values for sign in ("plus","minus")]
        cases.append({"geometry":name,"area_ratio_n_over_p":ratio,"inputs":kwargs,
                      "baseline":base,"one_dimensional_baseline_cold_heat_w":ref[name]["cold_heat_w"],
                      "baseline_cold_heat_absolute_error_w":abs(base["cold_heat_w"]-ref[name]["cold_heat_w"]),
                      "modes":modes,"field_domain_checks":field_checks,
                      "maximum_energy_residual_w":max(abs(result["energy_residual_w"]) for result in all_results),
                      "maximum_current_balance_a":max(abs(leg["current_balance_a"]) for result in all_results for leg in result["legs"].values()),
                      "maximum_temperature_k":max(leg["temperature_range_k"][1] for result in all_results for leg in result["legs"].values())})
    return {"method":"independent nonlinear tetrahedral P1 finite elements, complete coupled T/phi weak form",
            "material_role":"source-CSV-driven scalar property model, not as-built experimental module reconstruction",
            "source":source,"property_domains_k":domains,"mesh_level":level,
            "root_table_sha256":hashlib.sha256((GEOMETRY/"mode_specific_roots_and_measure_minima.csv").read_bytes()).hexdigest(),
            "perturbation":"S_p,n -> S_p,n + epsilon*m(T), mode amplitude 10 microV/K on [Tc,Th]",
            "mode_extension":"polynomial evaluated at actual internal temperature, including any maximum above Th",
            "fixed_constraints":["total semiconductor area","each leg length","current","cold/hot leg-end temperature","material curves"],
            "cases":cases,"elapsed_seconds":time.perf_counter()-started}


def summarize(reports):
    rows=[]
    for report in reports:
        for case in report["cases"]:
            for mode,values in case["modes"].items():
                entry=next(v for v in values if v["epsilon"]==.01)
                rows.append({"mesh":report["mesh_level"],"geometry":case["geometry"],"mode":mode,
                             "ratio_n_over_p":case["area_ratio_n_over_p"],"cold_heat_w":case["baseline"]["cold_heat_w"],
                             "cold_heat_absolute_error_w":case["baseline_cold_heat_absolute_error_w"],
                             "derivative_w":entry["derivative_w"],"reference_1d_w":entry["one_dimensional_reference_w"],
                             "derivative_absolute_error_w":entry["absolute_error_to_1d_w"],
                             "maximum_temperature_k":case["maximum_temperature_k"]})
    return {"comparison":rows,"all_property_domains_respected":all(
                values["no_extrapolation"] for report in reports for case in report["cases"]
                for check in case["field_domain_checks"] for values in check.values()),
            "maximum_energy_residual_w":max(case["maximum_energy_residual_w"] for report in reports for case in report["cases"]),
            "maximum_current_balance_a":max(case["maximum_current_balance_a"] for report in reports for case in report["cases"])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh",choices=(*MESHES,"all"),default="all")
    parser.add_argument("--output-directory",type=Path,default=OUTPUT)
    args=parser.parse_args()
    args.output_directory.mkdir(parents=True,exist_ok=True)
    reports=[]
    for level in (MESHES if args.mesh=="all" else (args.mesh,)):
        report=run_level(level)
        output=args.output_directory/f"pbse_{level}.json"
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        reports.append(report)
        print(json.dumps({"level":level,"elapsed_seconds":report["elapsed_seconds"],"output":str(output)}),flush=True)
    (args.output_directory/"summary.json").write_text(json.dumps(summarize(reports),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
