"""Collect independent finite-element accuracy and composite-device responses."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/transport_validation"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converged-results", type=Path, required=True)
    args = parser.parse_args()
    args.converged_results = args.converged_results.resolve()
    original = read_csv(RESULTS/"geometry/common_mode_response_scan.csv")
    roots = read_csv(RESULTS/"geometry/mode_specific_roots_and_measure_minima.csv")
    bts_original = {r["mode"]:float(r["delta_qc_per_pair_w"]) for r in original
                    if r["system"]=="bts_bst" and abs(float(r["area_ratio_n_over_p"])-1)<1e-12}
    bts_root = next(r for r in roots if r["system"]=="bts_bst" and r["objective"]=="parabola" and r["kind"]=="bracketed_root")
    fixed_rows=[]
    for level, path in (("coarse", RESULTS/"fem_fixed_ends/fixed_end_coarse.json"),
                        ("medium", RESULTS/"fem_fixed_ends/fixed_end_medium.json"),
                        ("fine", args.converged_results/"fixed_end_fine.json")):
        report=json.loads(path.read_text())
        for system,key in (("constant","synthetic"),("bts_bst","bts_bst")):
            for case in report[key]:
                ratio=case["area_ratio_n_over_p"]
                for mode,value in case["modes"].items():
                    mode_key="parabola" if mode=="endpoint_zero" else mode
                    if system=="constant":
                        reference=value["analytic_derivative_w"]
                        geometry=f"S_{case['seebeck_matched']}_transfer_{case['transfer_matched']}"
                    else:
                        reference=bts_original[mode_key] if ratio==1 else float(bts_root[f"{mode_key}_delta_qc_w"])
                        geometry="original" if ratio==1 else "adjusted"
                    branch_values=[(value["plus"]["legs"][b]["inward_heat_w"]["cold"]-
                                    value["minus"]["legs"][b]["inward_heat_w"]["cold"])/(2*value["epsilon"]) for b in ("p","n")]
                    error=abs(value["derivative_w"]-reference)
                    fixed_rows.append({"system":system,"mesh":level,"geometry":geometry,"mode":mode,
                                       "ratio_n_over_p":ratio,"response_w":value["derivative_w"],
                                       "reference_w":reference,"absolute_error_w":error,
                                       "error_over_sum_abs_branch_response":error/max(sum(abs(v) for v in branch_values),1e-30),
                                       "cold_heat_w":case["baseline"]["cold_heat_w"]})
    pbse=json.loads((RESULTS/"fem_pbse/summary.json").read_text())
    packaged=[]
    energy=[]
    for label in ("equal","one_mode_1d"):
        for mode,refinements in (("parabola",(1,2,4)),("linear",(4,))):
            for refinement in refinements:
                path=args.converged_results/"packaged"/f"{label}_{mode}_r{refinement}.json"
                report=json.loads(path.read_text())
                fine=next(v for v in report["centered_responses"] if v["epsilon"]==.01)
                step=report["centered_responses"][0]
                baseline=report["baseline"]
                reference=bts_original[mode] if label=="equal" else float(bts_root[f"{mode}_delta_qc_w"])
                packaged.append({"geometry":label,"mode":report["mode"],"refinement":refinement,
                                  "nodes":baseline["nodes"],"tetrahedra":baseline["tetrahedra"],
                                  "ratio_n_over_p":report["geometry_and_discretization"]["area_ratio_n_over_p"],
                                  "cold_heat_w":baseline["inward_heat_w"]["cold"],
                                  "input_power_w":baseline["electrical_power_w"],
                                  "cop":baseline["inward_heat_w"]["cold"]/baseline["electrical_power_w"],
                                  "response_w":fine["cold_heat_derivative_w"],
                                  "adjoint_response_w":fine["adjoint"]["derivative_w"],
                                  "adjoint_absolute_difference_w":abs(fine["adjoint_difference_w"]),
                                  "epsilon_change_w":abs(fine["cold_heat_derivative_w"]-step["cold_heat_derivative_w"]),
                                  "ideal_fixed_leg_end_reference_w":reference,
                                  "source":str(path.relative_to(ROOT))})
                energy.append(report["checks"]["max_abs_energy_residual_w"])
    changes=[]
    for label in ("equal","one_mode_1d"):
        rows=[r for r in packaged if r["geometry"]==label and r["mode"]=="reservoir_anchored_parabola"]
        coarse,medium,fine=sorted(rows,key=lambda r:r["refinement"])
        changes.append({"geometry":label,"medium_to_fine_response_difference_w":abs(fine["response_w"]-medium["response_w"]),
                        "medium_to_fine_response_relative_change":abs(fine["response_w"]-medium["response_w"])/abs(fine["response_w"]),
                        "medium_to_fine_cold_heat_relative_change":abs(fine["cold_heat_w"]-medium["cold_heat_w"])/abs(fine["cold_heat_w"]),
                        "successive_response_correction_ratio":abs((medium["response_w"]-coarse["response_w"])/(fine["response_w"]-medium["response_w"]))})
    out=RESULTS/"fem_comparison"
    out.mkdir(exist_ok=True)
    write_csv(out/"fixed_end_comparison.csv",fixed_rows)
    write_csv(out/"packaged_comparison.csv",packaged)
    summary={"status":"converged_results",
             "fixed_end_finest": [r for r in fixed_rows if r["mesh"]=="fine"],
             "pbse_finest": [r for r in pbse["comparison"] if r["mesh"]=="fine"],
             "packaged_finest": [r for r in packaged if r["refinement"]==4],
             "packaged_mesh_comparison":changes,
             "packaged_max_energy_residual_w":max(energy),
             "packaged_max_adjoint_error_w":max(r["adjoint_absolute_difference_w"] for r in packaged),
             "packaged_max_epsilon_change_w":max(r["epsilon_change_w"] for r in packaged),
             "interpretation":["Fixed-end FEM compares like-for-like material laws and boundary conditions with analytic/1D references.",
                               "Composite-device cold heat is measured at the external ceramic face, not a semiconductor face.",
                               "Parabolic perturbations vanish at reservoir temperatures, not necessarily at actual semiconductor ends.",
                               "Mesh differences measure numerical convergence, not experimental uncertainty.",
                               "Packaging coefficients and scalar transverse transport are declared model assumptions."]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"fixed_end_rows":len(fixed_rows),"packaged_rows":len(packaged),
                      "packaged_adjoint_max_error_w":summary["packaged_max_adjoint_error_w"],
                      "packaged_mesh_comparison":changes},indent=2))


if __name__=="__main__":
    main()
