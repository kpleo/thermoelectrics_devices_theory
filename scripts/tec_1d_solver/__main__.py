"""Run the constant-property analytic reference from the command line."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from .constant_properties import ConstantPropertyCouple, I_opt_for_max_Qc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the ideal constant-property thermoelectric-couple "
            "reference. Seebeck CLI inputs are in microvolt/K; all output "
            "quantities are SI."
        )
    )
    parser.add_argument("--seebeck-p-uv-per-k", type=float, default=200.0)
    parser.add_argument("--seebeck-n-uv-per-k", type=float, default=-200.0)
    parser.add_argument("--resistance-ohm", type=float, default=0.1)
    parser.add_argument("--thermal-conductance-w-per-k", type=float, default=0.003)
    parser.add_argument("--cold-temperature-k", type=float, default=300.0)
    parser.add_argument("--hot-temperature-k", type=float, default=310.0)
    parser.add_argument(
        "--current-a",
        type=float,
        default=None,
        help="Current in A; defaults to the analytic current maximizing Qc.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    couple = ConstantPropertyCouple(
        seebeck_p_v_per_k=args.seebeck_p_uv_per_k * 1.0e-6,
        seebeck_n_v_per_k=args.seebeck_n_uv_per_k * 1.0e-6,
        electrical_resistance_ohm=args.resistance_ohm,
        thermal_conductance_w_per_k=args.thermal_conductance_w_per_k,
        cold_temperature_k=args.cold_temperature_k,
        hot_temperature_k=args.hot_temperature_k,
    )
    optimum_current = I_opt_for_max_Qc(couple)
    current = optimum_current if args.current_a is None else args.current_a
    point = couple.evaluate(current)

    output = {
        "model": "constant_property_ideal_thermoelectric_couple",
        "units": {
            "current": "A",
            "heat_and_power": "W",
            "voltage": "V",
            "temperature": "K",
            "seebeck": "V/K",
            "energy_residual": "W",
        },
        "parameters": {
            "seebeck_p_v_per_k": couple.seebeck_p_v_per_k,
            "seebeck_n_v_per_k": couple.seebeck_n_v_per_k,
            "delta_seebeck_v_per_k": couple.delta_seebeck_v_per_k,
            "electrical_resistance_ohm": couple.electrical_resistance_ohm,
            "thermal_conductance_w_per_k": couple.thermal_conductance_w_per_k,
            "cold_temperature_k": couple.cold_temperature_k,
            "hot_temperature_k": couple.hot_temperature_k,
        },
        "operating_point": {
            "current_a": point.current_a,
            "Qc_w": point.Qc_w,
            "Qh_w": point.Qh_w,
            "V_v": point.V_v,
            "Pin_w": point.Pin_w,
            "COP": point.COP,
            "energy_residual_w": point.energy_residual_w,
            "relative_energy_residual": point.relative_energy_residual,
        },
        "I_opt_for_max_Qc_a": optimum_current,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
