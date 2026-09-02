from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.analysis import adjoint_nondegeneracy as validation


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/scientific_analysis/adjoint_nondegeneracy.json"


def test_result_covers_all_reported_one_dimensional_states() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == validation.SCHEMA_VERSION
    assert payload["coverage"]["couple_baselines"] == 1051
    assert payload["coverage"]["leg_baselines"] == 2102
    assert payload["summary"]["maximum_poincare_eta"] < 1.0
    assert payload["summary"]["minimum_absolute_shooting_determinant"] > 0.3
    assert all(payload["validation_checks"].values())
    assert {
        "signed_current_sweep",
        "monotone_pointwise_corners",
        "six_property_method_corners",
        "thermal_contact_endpoint_states",
    }.issubset(payload["coverage"]["scenario_families"])


def test_primary_pbse_reference_state_reproduces_stored_bound() -> None:
    cases = [
        case
        for case in validation.build_pbse_cases()
        if case.scenario_family == "primary_reference_state"
        and case.variant == "original"
    ]
    assert len(cases) == 1
    case = cases[0]
    point = validation._solve(case.couple, case.current_a)
    record = validation._diagnose_leg(
        case.couple.p_leg, point.p_leg, case=case, leg_name="p"
    )
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    stored = next(
        row
        for row in payload["records"]
        if row["material_family"] == "PbSe/Cr"
        and row["scenario_family"] == "primary_reference_state"
        and row["variant"] == "original"
        and row["leg"] == "p"
    )
    assert math.isclose(
        record["poincare_eta"],
        stored["poincare_eta"],
        rel_tol=2.0e-10,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        record["homogeneous_shooting_determinant"],
        stored["homogeneous_shooting_determinant"],
        rel_tol=2.0e-10,
        abs_tol=1.0e-12,
    )
