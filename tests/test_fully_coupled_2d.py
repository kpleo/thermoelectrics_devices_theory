from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis import fully_coupled_2d as coupled


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/scientific_analysis/fully_coupled_2d.json"


def test_full_coupling_result_passes_numerical_checks() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == coupled.SCHEMA_VERSION
    assert all(payload["validation_checks"].values())
    assert len(payload["grid_records"]) == 3
    assert payload["summary"]["maximum_pair_adjoint_to_nonlinear_relative_error"] < 1.0e-7
    assert payload["summary"]["medium_to_fine_pair_adjoint_relative_change"] < 5.0e-3
    assert payload["summary"]["maximum_energy_residual_relative"] < 1.0e-11
    assert payload["summary"]["minimum_sigma_T_jacobian_fraction"] > 1.0e-5
    assert (
        payload["summary"]["minimum_relative_current_field_derivative_per_epsilon"]
        > 1.0e-4
    )
    assert (
        payload["sigma_T_zero_negative_control"][
            "sigma_T_current_redistribution_jacobian_fraction"
        ]
        == 0.0
    )
    assert (
        payload["sigma_T_zero_negative_control"][
            "relative_current_field_derivative_per_epsilon"
        ]
        == 0.0
    )


def test_small_grid_independent_rerun_resolves_sigma_t_current_response() -> None:
    result = coupled.analyze_grid(6, 4)
    assert result["pair_adjoint_to_nonlinear_relative_error"] < 1.0e-7
    assert result["maximum_all_state_energy_residual_relative"] < 1.0e-11
    assert result["minimum_sigma_T_jacobian_fraction"] > 1.0e-4
    assert result["minimum_relative_current_field_derivative"] > 1.0e-4
