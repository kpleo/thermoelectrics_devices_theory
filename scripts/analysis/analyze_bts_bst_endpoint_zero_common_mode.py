#!/usr/bin/env python3
"""Cross-material endpoint-zero common-mode test for a BTS/BST module.

The source is Liu et al., National Science Review 12, nwae448 (2025), DOI
10.1093/nsr/nwae448, with Li-Dong Zhao as a corresponding author.  The
analysis uses only figure-derived n-BTS+0.2%Cu and p-BST transport curves and
the explicitly reported seven-pair geometry.  It does not extrapolate below
300 K and does not fit unknown device boundary conditions.

The common Seebeck perturbation is

    delta M(T) = 4 M_pk (T-Tc)(Th-T)/(Th-Tc)^2.

It is added to both legs.  Therefore alpha=S_p-S_n is unchanged pointwise,
delta M(Tc)=delta M(Th)=0, and the open-circuit voltage integral is unchanged.
The cold-port response is consequently a distributed Thomson branch-transfer
test rather than an endpoint Peltier or voltage test.

The adjoint first variation is compared against an independent nonlinear
central difference of the conservative temperature-dependent BVP.  The
analysis also evaluates zero-response limits, global energy closure,
dimensionless thermal-contact sensitivity, and deterministic method-bound
property corners.  The property corners use the reported measurement-error
ceilings as a deterministic range, not as a confidence interval or as
independent probability distributions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Iterable
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["svg.hashsalt"] = "bts-bst-endpoint-zero-common-mode-v1"


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.analyze_common_mode_transfer_kernel import (  # noqa: E402
    first_order_port_response,
)
from scripts.tec_1d_solver import (  # noqa: E402
    BoundaryNetworkSolverOptions,
    FixedCurrentBoundaryNetwork,
    PchipTemperatureProperty,
    ReservoirThermalContacts,
    SeriesElectricalContacts,
    SYNTHETIC_DATA_ROLE,
    TemperatureDependentLeg,
    TemperatureDependentNumericalCouple,
    TemperatureDependentOperatingPoint,
    TwoReservoirParasitic,
    solve_fixed_current_boundary_network,
    solve_temperature_dependent_couple,
)


FloatArray = NDArray[np.float64]

SCHEMA_VERSION = "bts_bst_endpoint_zero_common_mode/v1"
ANALYSIS_ID = "SCI-BTS-BST-ENDPOINT-ZERO-COMMON-MODE-20260826"
DOI = "10.1093/nsr/nwae448"
PMCID = "PMC11737397"

RAW_DIR = ROOT / "data/raw/nwae448_bts_bst"
SOURCE_RECORD_PATH = RAW_DIR / "nwae448_source_records.json"
TRANSPORT_CSV = RAW_DIR / "nwae448_digitized_transport.csv"
DEFAULT_JSON = (
    ROOT
    / "results/scientific_analysis/"
    "bts_bst_endpoint_zero_common_mode_results.json"
)
DEFAULT_SCAN_CSV = (
    ROOT
    / "results/scientific_analysis/"
    "bts_bst_endpoint_zero_common_mode_current_scan.csv"
)
DEFAULT_FIGURE_STEM = (
    ROOT
    / "results/scientific_analysis/"
    "bts_bst_endpoint_zero_common_mode"
)

EXPECTED_SOURCE_HASHES = {
    RAW_DIR / "PMC11737397_fulltext.xml": (
        "ae0eb05433da7842a7951c4df38bb9493e58f795d32e4da9a6cccabf8fb22cc8"
    ),
    RAW_DIR / "PMC11737397_SupplementaryFiles.zip": (
        "248b98e28d56f3d8dea8388f34ae906f26fe38291073010a24ea4f38a0092eb3"
    ),
    RAW_DIR / "nwae448_supplement.pdf": (
        "147ec2c7f52d2328e552290aed28e0c9cd01ad6de5b7fc3708adfb2d76755d66"
    ),
    RAW_DIR / "europepmc_bundle/nwae448fig2.jpg": (
        "02ca88b9dc185a240c32bf64cae0db27204cd96242c36abf7ea22def6f091f9e"
    ),
    RAW_DIR / "europepmc_bundle/nwae448fig5.jpg": (
        "ae130541f8fc8210203235357de6cbf76ddd46527547f28b905a0804fe8d6107"
    ),
    RAW_DIR / "europepmc_bundle/nwae448fig6.jpg": (
        "78e9e7416c3e1dfa3d82727981224d01d3cc56e1533850256e7fd56ecfefc215"
    ),
}

TEMPERATURE_KNOTS_K = np.asarray([300.0, 323.0, 373.0, 423.0, 473.0, 523.0])
LEG_LENGTH_M = 4.8e-3
LEG_WIDTH_M = 1.95e-3
LEG_AREA_M2 = LEG_WIDTH_M**2
PAIR_COUNT = 7

PRIMARY_COLD_TEMPERATURE_K = 323.0
PRIMARY_HOT_TEMPERATURE_K = 373.0
PRIMARY_CURRENT_A = 3.0
MODE_PEAK_V_PER_K = 10.0e-6
CENTRAL_EPSILON = 3.0e-3

SEEBECK_METHOD_BOUND = 0.05
SIGMA_METHOD_BOUND = 0.05
KAPPA_METHOD_BOUND = 0.08


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_locator(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _linear_pixel_value(
    pixel_y: float,
    top_pixel_y: float,
    bottom_pixel_y: float,
    top_value: float,
    bottom_value: float,
) -> float:
    return top_value + (pixel_y - top_pixel_y) * (
        bottom_value - top_value
    ) / (bottom_pixel_y - top_pixel_y)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_sources_and_load_transport() -> tuple[dict[str, FloatArray], dict[str, Any]]:
    """Fail closed on source identity, hashes, and pixel-value arithmetic."""

    for path, expected in EXPECTED_SOURCE_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(f"required source object is absent: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(f"source hash mismatch for {path}: {observed} != {expected}")

    source_record = json.loads(SOURCE_RECORD_PATH.read_text(encoding="utf-8"))
    article = source_record["article"]
    if article["doi"] != DOI or article["pmcid"] != PMCID:
        raise ValueError("source record DOI/PMCID identity mismatch")
    if article["corresponding_author_of_interest"] != "Li-Dong Zhao":
        raise ValueError("source record does not identify Li-Dong Zhao exactly")
    if article["license"] != "CC BY 4.0":
        raise ValueError("unexpected article license")

    xml_root = ET.parse(RAW_DIR / "PMC11737397_fulltext.xml").getroot()
    doi_nodes = xml_root.findall(".//article-id[@pub-id-type='doi']")
    pmc_nodes = xml_root.findall(".//article-id[@pub-id-type='pmcid']")
    if [node.text for node in doi_nodes] != [DOI]:
        raise ValueError("full-text XML DOI mismatch")
    if [node.text for node in pmc_nodes] != [PMCID]:
        raise ValueError("full-text XML PMCID mismatch")
    authors = [
        (
            node.findtext("./name/given-names"),
            node.findtext("./name/surname"),
        )
        for node in xml_root.findall(".//contrib-group[@content-type='author']/contrib")
    ]
    if ("Li-Dong", "Zhao") not in authors:
        raise ValueError("Li-Dong Zhao is absent from the article author list")
    license_links = [
        node.attrib.get("{http://www.w3.org/1999/xlink}href", "")
        for node in xml_root.findall(".//permissions//ext-link")
    ]
    if "https://creativecommons.org/licenses/by/4.0/" not in license_links:
        raise ValueError("CC BY 4.0 license link is absent from full-text XML")

    rows = _read_rows(TRANSPORT_CSV)
    if len(rows) != 12:
        raise ValueError("expected exactly 12 transport rows (six per branch)")
    by_branch = {branch: [row for row in rows if row["branch"] == branch] for branch in ("p", "n")}
    if any(len(group) != 6 for group in by_branch.values()):
        raise ValueError("transport table is not branch complete")

    for row in rows:
        for prefix, value_column, tolerance in (
            ("seebeck", "seebeck_uv_per_k", 1.0e-6),
            ("sigma", "sigma_1e3_s_per_cm", 1.0e-6),
            ("kappa", "kappa_w_per_m_k", 1.0e-6),
        ):
            recomputed = _linear_pixel_value(
                float(row[f"{prefix}_pixel_y"]),
                float(row[f"{prefix}_axis_y_top_px"]),
                float(row[f"{prefix}_axis_y_bottom_px"]),
                float(row[
                    f"{prefix}_axis_top_"
                    + (
                        "uv_per_k"
                        if prefix == "seebeck"
                        else "1e3_s_per_cm"
                        if prefix == "sigma"
                        else "w_per_m_k"
                    )
                ]),
                float(row[
                    f"{prefix}_axis_bottom_"
                    + (
                        "uv_per_k"
                        if prefix == "seebeck"
                        else "1e3_s_per_cm"
                        if prefix == "sigma"
                        else "w_per_m_k"
                    )
                ]),
            )
            if abs(recomputed - float(row[value_column])) > tolerance:
                raise ValueError(
                    f"pixel calibration mismatch for {row['material_id']} "
                    f"at {row['temperature_k']} K, {prefix}"
                )

    def column(branch: str, key: str) -> FloatArray:
        ordered = sorted(by_branch[branch], key=lambda row: float(row["temperature_k"]))
        return np.asarray([float(row[key]) for row in ordered], dtype=float)

    p_temperature = column("p", "temperature_k")
    n_temperature = column("n", "temperature_k")
    if not np.array_equal(p_temperature, TEMPERATURE_KNOTS_K) or not np.array_equal(
        n_temperature, TEMPERATURE_KNOTS_K
    ):
        raise ValueError("p/n temperature knots are not the frozen six-point grid")

    transport = {
        "temperature_k": TEMPERATURE_KNOTS_K.copy(),
        "p_seebeck_v_per_k": column("p", "seebeck_uv_per_k") * 1.0e-6,
        "n_seebeck_v_per_k": column("n", "seebeck_uv_per_k") * 1.0e-6,
        # 10^3 S cm^-1 = 10^5 S m^-1.
        "p_sigma_s_per_m": column("p", "sigma_1e3_s_per_cm") * 1.0e5,
        "n_sigma_s_per_m": column("n", "sigma_1e3_s_per_cm") * 1.0e5,
        "p_kappa_w_per_m_k": column("p", "kappa_w_per_m_k"),
        "n_kappa_w_per_m_k": column("n", "kappa_w_per_m_k"),
    }
    if np.any(transport["p_seebeck_v_per_k"] <= 0.0) or np.any(
        transport["n_seebeck_v_per_k"] >= 0.0
    ):
        raise ValueError("branch Seebeck signs are inconsistent with p/n identity")

    p_pf = (
        transport["p_seebeck_v_per_k"] ** 2
        * transport["p_sigma_s_per_m"]
    )
    n_pf = (
        transport["n_seebeck_v_per_k"] ** 2
        * transport["n_sigma_s_per_m"]
    )
    p_zt = p_pf * TEMPERATURE_KNOTS_K / transport["p_kappa_w_per_m_k"]
    n_zt = n_pf * TEMPERATURE_KNOTS_K / transport["n_kappa_w_per_m_k"]
    if not (0.9 < p_zt[0] < 1.1 and 1.2 < n_zt[0] < 1.35):
        raise ValueError("digitized S/sigma/kappa fail the independent 300 K ZT cross-check")

    validation = {
        "source_record_locator": relative_locator(SOURCE_RECORD_PATH),
        "transport_locator": relative_locator(TRANSPORT_CSV),
        "source_hashes": {
            relative_locator(path): expected
            for path, expected in EXPECTED_SOURCE_HASHES.items()
        },
        "article_identity": article,
        "source_roles": {
            "n_branch": "article Fig. 2(a,c) plus Fig. 5(a), figure-derived",
            "p_branch": "supplement Fig. S9(a,b,d), PDF page 17, figure-derived",
        },
        "curve_completeness": {
            "p": ["S", "sigma", "kappa_tot"],
            "n": ["S", "sigma", "kappa_tot"],
            "temperature_domain_k": [300.0, 523.0],
            "knots_per_property": 6,
        },
        "independent_curve_cross_checks": {
            "p_power_factor_300_uW_per_cm_k2": float(1.0e4 * p_pf[0]),
            "n_power_factor_300_uW_per_cm_k2": float(1.0e4 * n_pf[0]),
            "p_zt_300": float(p_zt[0]),
            "n_zt_300": float(n_zt[0]),
        },
        "measurement_method_bounds": {
            "seebeck_relative": SEEBECK_METHOD_BOUND,
            "electrical_conductivity_relative": SIGMA_METHOD_BOUND,
            "thermal_conductivity_relative": KAPPA_METHOD_BOUND,
            "interpretation": (
                "source-reported measurement-error ceilings used only as a "
                "deterministic method-bound range; not a confidence interval"
            ),
        },
        "explicit_absences": source_record["explicit_absences"],
    }
    return transport, validation


class CommonModePerturbedProperty(PchipTemperatureProperty):
    """Add epsilon*m(T) to an existing Seebeck property."""

    def __init__(
        self,
        source: Any,
        *,
        epsilon: float,
        mode: Callable[[FloatArray], FloatArray],
        mode_derivative: Callable[[FloatArray], FloatArray],
    ) -> None:
        super().__init__(
            (source.minimum_temperature_k, source.maximum_temperature_k),
            (0.0, 0.0),
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "epsilon", float(epsilon))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "mode_derivative", mode_derivative)

    def evaluate(self, temperature_k: object) -> FloatArray:
        checked = PchipTemperatureProperty.evaluate(self, temperature_k)
        temperature = np.asarray(temperature_k, dtype=float)
        value = self.source.evaluate(temperature) + self.epsilon * self.mode(temperature)
        return np.asarray(value, dtype=float).reshape(np.shape(checked))

    def derivative(self, temperature_k: object) -> FloatArray:
        checked = PchipTemperatureProperty.derivative(self, temperature_k)
        temperature = np.asarray(temperature_k, dtype=float)
        value = self.source.derivative(temperature) + self.epsilon * self.mode_derivative(
            temperature
        )
        return np.asarray(value, dtype=float).reshape(np.shape(checked))


def endpoint_zero_mode(
    cold_temperature_k: float,
    hot_temperature_k: float,
    peak_v_per_k: float = MODE_PEAK_V_PER_K,
) -> tuple[
    Callable[[FloatArray], FloatArray],
    Callable[[FloatArray], FloatArray],
    Callable[[FloatArray], FloatArray],
]:
    span = float(hot_temperature_k - cold_temperature_k)
    if span <= 0.0:
        raise ValueError("endpoint-zero mode requires Th > Tc")

    def mode(temperature: FloatArray) -> FloatArray:
        value = np.asarray(temperature, dtype=float)
        return 4.0 * peak_v_per_k * (value - cold_temperature_k) * (
            hot_temperature_k - value
        ) / span**2

    def derivative(temperature: FloatArray) -> FloatArray:
        value = np.asarray(temperature, dtype=float)
        return 4.0 * peak_v_per_k * (
            cold_temperature_k + hot_temperature_k - 2.0 * value
        ) / span**2

    def gamma(temperature: FloatArray) -> FloatArray:
        value = np.asarray(temperature, dtype=float)
        return value * derivative(value)

    return mode, derivative, gamma


def make_leg(
    temperature_k: FloatArray,
    seebeck_v_per_k: FloatArray,
    sigma_s_per_m: FloatArray,
    kappa_w_per_m_k: FloatArray,
    *,
    area_multiplier: float = 1.0,
) -> TemperatureDependentLeg:
    return TemperatureDependentLeg(
        seebeck=PchipTemperatureProperty(temperature_k, seebeck_v_per_k),
        electrical_resistivity=PchipTemperatureProperty(
            temperature_k, 1.0 / sigma_s_per_m
        ),
        thermal_conductivity=PchipTemperatureProperty(
            temperature_k, kappa_w_per_m_k
        ),
        length_m=LEG_LENGTH_M,
        area_m2=LEG_AREA_M2 * area_multiplier,
    )


def build_legs(
    transport: dict[str, FloatArray],
    *,
    gains: tuple[float, float, float, float, float, float] | None = None,
    area_multiplier: float = 1.0,
) -> tuple[TemperatureDependentLeg, TemperatureDependentLeg]:
    if gains is None:
        p_s_gain = n_s_gain = p_sigma_gain = n_sigma_gain = p_k_gain = n_k_gain = 1.0
    else:
        p_s_gain, n_s_gain, p_sigma_gain, n_sigma_gain, p_k_gain, n_k_gain = gains
    p_leg = make_leg(
        transport["temperature_k"],
        p_s_gain * transport["p_seebeck_v_per_k"],
        p_sigma_gain * transport["p_sigma_s_per_m"],
        p_k_gain * transport["p_kappa_w_per_m_k"],
        area_multiplier=area_multiplier,
    )
    n_leg = make_leg(
        transport["temperature_k"],
        n_s_gain * transport["n_seebeck_v_per_k"],
        n_sigma_gain * transport["n_sigma_s_per_m"],
        n_k_gain * transport["n_kappa_w_per_m_k"],
        area_multiplier=area_multiplier,
    )
    return p_leg, n_leg


def replace_leg_seebeck(
    leg: TemperatureDependentLeg,
    epsilon: float,
    mode: Callable[[FloatArray], FloatArray],
    mode_derivative: Callable[[FloatArray], FloatArray],
) -> TemperatureDependentLeg:
    return TemperatureDependentLeg(
        seebeck=CommonModePerturbedProperty(
            leg.seebeck,
            epsilon=epsilon,
            mode=mode,
            mode_derivative=mode_derivative,
        ),
        electrical_resistivity=leg.electrical_resistivity,
        thermal_conductivity=leg.thermal_conductivity,
        length_m=leg.length_m,
        area_m2=leg.area_m2,
    )


def build_couple(
    legs: tuple[TemperatureDependentLeg, TemperatureDependentLeg],
    cold_temperature_k: float,
    hot_temperature_k: float,
    *,
    epsilon: float = 0.0,
    mode: Callable[[FloatArray], FloatArray] | None = None,
    mode_derivative: Callable[[FloatArray], FloatArray] | None = None,
) -> TemperatureDependentNumericalCouple:
    p_leg, n_leg = legs
    if epsilon != 0.0:
        if mode is None or mode_derivative is None:
            raise ValueError("perturbed couple requires mode and derivative")
        p_leg = replace_leg_seebeck(p_leg, epsilon, mode, mode_derivative)
        n_leg = replace_leg_seebeck(n_leg, epsilon, mode, mode_derivative)
    return TemperatureDependentNumericalCouple(
        p_leg=p_leg,
        n_leg=n_leg,
        cold_temperature_k=cold_temperature_k,
        hot_temperature_k=hot_temperature_k,
    )


def tight_solve(
    couple: TemperatureDependentNumericalCouple,
    current_a: float,
    *,
    output_points: int = 601,
) -> TemperatureDependentOperatingPoint:
    return solve_temperature_dependent_couple(
        couple,
        current_a,
        initial_mesh_points=51,
        output_points=output_points,
        relative_tolerance=1.0e-9,
        max_nodes=16000,
    )


def fixed_endpoint_response(
    legs: tuple[TemperatureDependentLeg, TemperatureDependentLeg],
    current_a: float,
    cold_temperature_k: float,
    hot_temperature_k: float,
) -> dict[str, Any]:
    mode, mode_derivative, gamma = endpoint_zero_mode(
        cold_temperature_k, hot_temperature_k
    )
    base_couple = build_couple(legs, cold_temperature_k, hot_temperature_k)
    baseline = tight_solve(base_couple, current_a)
    adjoint = first_order_port_response(
        base_couple,
        baseline,
        mode_value=mode,
        gamma_value=gamma,
        port="cold",
    )
    plus = tight_solve(
        build_couple(
            legs,
            cold_temperature_k,
            hot_temperature_k,
            epsilon=CENTRAL_EPSILON,
            mode=mode,
            mode_derivative=mode_derivative,
        ),
        current_a,
        output_points=401,
    )
    minus = tight_solve(
        build_couple(
            legs,
            cold_temperature_k,
            hot_temperature_k,
            epsilon=-CENTRAL_EPSILON,
            mode=mode,
            mode_derivative=mode_derivative,
        ),
        current_a,
        output_points=401,
    )
    finite_difference = (plus.Qc_w - minus.Qc_w) / (2.0 * CENTRAL_EPSILON)
    adjoint_value = float(adjoint["pair_total_port_derivative_w"])
    p_branch = float(adjoint["p_leg"]["distributed_kernel_derivative_w"])
    n_branch = float(adjoint["n_leg"]["distributed_kernel_derivative_w"])
    all_temperature = np.concatenate(
        [baseline.p_leg.temperature_k, baseline.n_leg.temperature_k]
    )
    normalized_mode = mode(all_temperature) / MODE_PEAK_V_PER_K
    absolute_error = abs(finite_difference - adjoint_value)
    relative_error = absolute_error / max(abs(adjoint_value), 1.0e-14)
    return {
        "current_a": float(current_a),
        "baseline_qc_per_pair_w": float(baseline.Qc_w),
        "baseline_qc_module_w": float(PAIR_COUNT * baseline.Qc_w),
        "baseline_voltage_per_pair_v": float(baseline.V_v),
        "adjoint_response_per_pair_w": adjoint_value,
        "adjoint_response_module_w": float(PAIR_COUNT * adjoint_value),
        "finite_difference_response_per_pair_w": float(finite_difference),
        "finite_difference_response_module_w": float(PAIR_COUNT * finite_difference),
        "adjoint_fd_absolute_error_w_per_pair": float(absolute_error),
        "adjoint_fd_relative_error": float(relative_error),
        "maximum_relative_energy_residual": float(
            max(
                baseline.relative_energy_residual,
                plus.relative_energy_residual,
                minus.relative_energy_residual,
            )
        ),
        "p_branch_adjoint_w": p_branch,
        "n_branch_adjoint_w": n_branch,
        "absolute_branch_cancellation_survival_fraction": float(
            abs(adjoint_value) / max(abs(p_branch) + abs(n_branch), 1.0e-30)
        ),
        "temperature_minimum_k": float(np.min(all_temperature)),
        "temperature_maximum_k": float(np.max(all_temperature)),
        "normalized_mode_minimum_on_field": float(np.min(normalized_mode)),
        "normalized_mode_maximum_on_field": float(np.max(normalized_mode)),
    }


def analyze_invariants(
    legs: tuple[TemperatureDependentLeg, TemperatureDependentLeg],
) -> dict[str, float]:
    cold = PRIMARY_COLD_TEMPERATURE_K
    hot = PRIMARY_HOT_TEMPERATURE_K
    mode, derivative, _ = endpoint_zero_mode(cold, hot)
    grid = np.linspace(cold, hot, 2001)
    p_leg, n_leg = legs
    alpha = p_leg.seebeck.evaluate(grid) - n_leg.seebeck.evaluate(grid)
    perturbed_alpha = (
        p_leg.seebeck.evaluate(grid)
        + mode(grid)
        - n_leg.seebeck.evaluate(grid)
        - mode(grid)
    )
    voc_change = float(np.trapezoid(perturbed_alpha - alpha, grid))
    return {
        "mode_at_cold_v_per_k": float(mode(np.asarray([cold]))[0]),
        "mode_at_hot_v_per_k": float(mode(np.asarray([hot]))[0]),
        "maximum_pointwise_alpha_change_v_per_k": float(
            np.max(np.abs(perturbed_alpha - alpha))
        ),
        "open_circuit_voltage_change_v": voc_change,
        "mode_derivative_integral_v_per_k": float(
            np.trapezoid(derivative(grid), grid)
        ),
    }


def analyze_current_scan(
    legs: tuple[TemperatureDependentLeg, TemperatureDependentLeg],
) -> list[dict[str, Any]]:
    # Stop at the selected 3 A primary point.  Over this interval both
    # branch fields remain inside [Tc, Th], so the parabolic probe is globally
    # bounded by 0 <= deltaM/Mpk <= 1 on every realized field.  Higher-current
    # Joule hot spots can exceed Th, where the polynomial changes sign and no
    # longer has Mpk as its field-wide amplitude; those states are intentionally
    # excluded rather than used to amplify the response.
    return [
        fixed_endpoint_response(
            legs,
            current,
            PRIMARY_COLD_TEMPERATURE_K,
            PRIMARY_HOT_TEMPERATURE_K,
        )
        for current in np.linspace(0.0, PRIMARY_CURRENT_A, 7)
    ]


def analyze_zero_response_checks(
    transport: dict[str, FloatArray],
    legs: tuple[TemperatureDependentLeg, TemperatureDependentLeg],
) -> dict[str, Any]:
    cold = PRIMARY_COLD_TEMPERATURE_K
    hot = PRIMARY_HOT_TEMPERATURE_K
    current = PRIMARY_CURRENT_A
    mode, mode_derivative, gamma = endpoint_zero_mode(cold, hot)

    p_leg, _ = legs
    symmetric_n = TemperatureDependentLeg(
        seebeck=PchipTemperatureProperty(
            transport["temperature_k"],
            -transport["p_seebeck_v_per_k"],
        ),
        electrical_resistivity=p_leg.electrical_resistivity,
        thermal_conductivity=p_leg.thermal_conductivity,
        length_m=p_leg.length_m,
        area_m2=p_leg.area_m2,
    )
    symmetric_couple = build_couple((p_leg, symmetric_n), cold, hot)
    symmetric_point = tight_solve(symmetric_couple, current)
    symmetric = first_order_port_response(
        symmetric_couple,
        symmetric_point,
        mode_value=mode,
        gamma_value=gamma,
        port="cold",
    )

    constant_mode = lambda temperature: np.full_like(
        np.asarray(temperature, dtype=float), MODE_PEAK_V_PER_K
    )
    zero_derivative = lambda temperature: np.zeros_like(
        np.asarray(temperature, dtype=float)
    )
    base_couple = build_couple(legs, cold, hot)
    base = tight_solve(base_couple, current)
    constant_plus = tight_solve(
        build_couple(
            legs,
            cold,
            hot,
            epsilon=1.0,
            mode=constant_mode,
            mode_derivative=zero_derivative,
        ),
        current,
    )
    return {
        "zero_current_endpoint_zero_response_w": float(
            fixed_endpoint_response(legs, 0.0, cold, hot)[
                "adjoint_response_per_pair_w"
            ]
        ),
        "identical_transport_antisymmetric_seebeck_response_w": float(
            symmetric["pair_total_port_derivative_w"]
        ),
        "constant_common_offset_qc_change_w": float(constant_plus.Qc_w - base.Qc_w),
        "constant_common_offset_energy_residual_fraction": float(
            constant_plus.relative_energy_residual
        ),
    }


def analyze_method_bound_corners(
    transport: dict[str, FloatArray],
) -> dict[str, Any]:
    """Complete 2^6 branch-level gain corner screen at the primary point."""

    records: list[dict[str, Any]] = []
    for signs in itertools.product((-1, 1), repeat=6):
        gains = (
            1.0 + signs[0] * SEEBECK_METHOD_BOUND,
            1.0 + signs[1] * SEEBECK_METHOD_BOUND,
            1.0 + signs[2] * SIGMA_METHOD_BOUND,
            1.0 + signs[3] * SIGMA_METHOD_BOUND,
            1.0 + signs[4] * KAPPA_METHOD_BOUND,
            1.0 + signs[5] * KAPPA_METHOD_BOUND,
        )
        legs = build_legs(transport, gains=gains)
        response = fixed_endpoint_response(
            legs,
            PRIMARY_CURRENT_A,
            PRIMARY_COLD_TEMPERATURE_K,
            PRIMARY_HOT_TEMPERATURE_K,
        )
        records.append(
            {
                "signs_pS_nS_pSigma_nSigma_pKappa_nKappa": list(signs),
                "gains_pS_nS_pSigma_nSigma_pKappa_nKappa": list(gains),
                "response_module_w": response["adjoint_response_module_w"],
                "baseline_qc_module_w": response["baseline_qc_module_w"],
                "response_over_qc": (
                    response["adjoint_response_module_w"]
                    / response["baseline_qc_module_w"]
                ),
                "adjoint_fd_relative_error": response["adjoint_fd_relative_error"],
                "maximum_relative_energy_residual": response[
                    "maximum_relative_energy_residual"
                ],
            }
        )
    minimum = min(records, key=lambda record: record["response_module_w"])
    maximum = max(records, key=lambda record: record["response_module_w"])
    return {
        "coordinate_definition": (
            "six independent branch-level gain signs at the source-reported "
            "S/sigma 5% and kappa 8% error ceilings"
        ),
        "statistical_status": "deterministic_method_bound_range_not_confidence_interval",
        "corner_count": len(records),
        "minimum": minimum,
        "maximum": maximum,
        "interval_crosses_zero": bool(
            minimum["response_module_w"] <= 0.0 <= maximum["response_module_w"]
        ),
        "maximum_adjoint_fd_relative_error": float(
            max(record["adjoint_fd_relative_error"] for record in records)
        ),
        "maximum_relative_energy_residual": float(
            max(record["maximum_relative_energy_residual"] for record in records)
        ),
        "records": records,
    }


def _endpoint_jacobian(
    legs: tuple[TemperatureDependentLeg, TemperatureDependentLeg],
    current_a: float,
    cold_temperature_k: float,
    hot_temperature_k: float,
    step_k: float = 1.0e-3,
) -> FloatArray:
    columns = []
    for cold_delta, hot_delta in ((step_k, 0.0), (0.0, step_k)):
        plus = tight_solve(
            build_couple(
                legs,
                cold_temperature_k + cold_delta,
                hot_temperature_k + hot_delta,
            ),
            current_a,
            output_points=401,
        )
        minus = tight_solve(
            build_couple(
                legs,
                cold_temperature_k - cold_delta,
                hot_temperature_k - hot_delta,
            ),
            current_a,
            output_points=401,
        )
        columns.append(
            np.asarray(
                [plus.Qc_w - minus.Qc_w, plus.Qh_w - minus.Qh_w], dtype=float
            )
            / (2.0 * step_k)
        )
    return np.column_stack(columns)


def analyze_contact_nuisance(
    transport: dict[str, FloatArray],
) -> dict[str, Any]:
    """Dimensionless contact stress, explicitly not a fit to the device."""

    super_legs = build_legs(transport, area_multiplier=float(PAIR_COUNT))
    module_current = PAIR_COUNT * PRIMARY_CURRENT_A
    midpoint = 0.5 * (
        PRIMARY_COLD_TEMPERATURE_K + PRIMARY_HOT_TEMPERATURE_K
    )
    p_leg, n_leg = super_legs
    module_thermal_conductance = (
        p_leg.area_m2
        / p_leg.length_m
        * float(
            p_leg.thermal_conductivity.evaluate([midpoint])[0]
            + n_leg.thermal_conductivity.evaluate([midpoint])[0]
        )
    )
    controls = BoundaryNetworkSolverOptions(
        nonlinear_tolerance=1.0e-11,
        max_function_evaluations=150,
        temperature_residual_tolerance_k=1.0e-7,
        node_energy_residual_tolerance_w=1.0e-8,
        global_energy_residual_fraction_tolerance=1.0e-9,
        bulk_initial_mesh_points=41,
        bulk_output_points=401,
        bulk_relative_tolerance=1.0e-9,
        bulk_max_nodes=16000,
    )

    def network(
        legs: tuple[TemperatureDependentLeg, TemperatureDependentLeg],
        thermal_resistance_k_per_w: float,
        parasitic_ratio: float = 0.0,
    ) -> FixedCurrentBoundaryNetwork:
        return FixedCurrentBoundaryNetwork(
            p_leg=legs[0],
            n_leg=legs[1],
            cold_reservoir_temperature_k=PRIMARY_COLD_TEMPERATURE_K,
            hot_reservoir_temperature_k=PRIMARY_HOT_TEMPERATURE_K,
            electrical_contacts=SeriesElectricalContacts(
                resistance_ohm=0.0,
                joule_fraction_to_cold_node=0.5,
            ),
            thermal_contacts=ReservoirThermalContacts(
                cold_resistance_k_per_w=thermal_resistance_k_per_w,
                hot_resistance_k_per_w=thermal_resistance_k_per_w,
            ),
            parasitic=TwoReservoirParasitic(
                thermal_conductance_w_per_k=(
                    parasitic_ratio * module_thermal_conductance
                )
            ),
            energy_scale_w=1.0,
            data_role=SYNTHETIC_DATA_ROLE,
        )

    records = []
    for contact_ratio in (0.0, 0.05, 0.10, 0.25):
        resistance = contact_ratio / module_thermal_conductance
        baseline = solve_fixed_current_boundary_network(
            network(super_legs, resistance), module_current, options=controls
        ).require_point()
        cold = baseline.cold_leg_temperature_k
        hot = baseline.hot_leg_temperature_k
        mode, mode_derivative, gamma = endpoint_zero_mode(cold, hot)
        bulk_couple = build_couple(super_legs, cold, hot)
        cold_response = first_order_port_response(
            bulk_couple,
            baseline.bulk_point,
            mode_value=mode,
            gamma_value=gamma,
            port="cold",
        )
        hot_response = first_order_port_response(
            bulk_couple,
            baseline.bulk_point,
            mode_value=mode,
            gamma_value=gamma,
            port="hot",
        )
        q_epsilon = np.asarray(
            [
                cold_response["pair_total_port_derivative_w"],
                hot_response["pair_total_port_derivative_w"],
            ],
            dtype=float,
        )
        port_jacobian = _endpoint_jacobian(
            super_legs, module_current, cold, hot
        )
        contact_matrix = np.asarray(
            [
                [
                    1.0 + resistance * port_jacobian[0, 0],
                    resistance * port_jacobian[0, 1],
                ],
                [
                    -resistance * port_jacobian[1, 0],
                    1.0 - resistance * port_jacobian[1, 1],
                ],
            ],
            dtype=float,
        )
        forcing = np.asarray(
            [resistance * q_epsilon[0], -resistance * q_epsilon[1]], dtype=float
        )
        endpoint_derivative = -np.linalg.solve(contact_matrix, forcing)
        dressed = float(
            q_epsilon[0] + port_jacobian[0] @ endpoint_derivative
        )

        plus_legs = tuple(
            replace_leg_seebeck(leg, CENTRAL_EPSILON, mode, mode_derivative)
            for leg in super_legs
        )
        minus_legs = tuple(
            replace_leg_seebeck(leg, -CENTRAL_EPSILON, mode, mode_derivative)
            for leg in super_legs
        )
        plus = solve_fixed_current_boundary_network(
            network(plus_legs, resistance), module_current, options=controls
        ).require_point()
        minus = solve_fixed_current_boundary_network(
            network(minus_legs, resistance), module_current, options=controls
        ).require_point()
        finite_difference = (plus.Qc_w - minus.Qc_w) / (2.0 * CENTRAL_EPSILON)
        records.append(
            {
                "contact_ratio_chi_equal_cold_hot": contact_ratio,
                "thermal_resistance_each_k_per_w": resistance,
                "cold_leg_temperature_k": cold,
                "hot_leg_temperature_k": hot,
                "baseline_qc_module_w": baseline.Qc_w,
                "fixed_endpoint_adjoint_response_module_w": float(q_epsilon[0]),
                "contact_dressed_adjoint_response_module_w": dressed,
                "full_network_finite_difference_response_module_w": float(
                    finite_difference
                ),
                "relative_adjoint_fd_error": float(
                    abs(finite_difference - dressed) / max(abs(dressed), 1.0e-14)
                ),
                "cold_endpoint_derivative_k": float(endpoint_derivative[0]),
                "hot_endpoint_derivative_k": float(endpoint_derivative[1]),
                "maximum_relative_energy_residual": float(
                    max(plus.relative_energy_residual, minus.relative_energy_residual)
                ),
            }
        )

    return {
        "status": "dimensionless_stress_coordinates_not_fitted_device_truths",
        "module_midpoint_bulk_thermal_conductance_w_per_k": float(
            module_thermal_conductance
        ),
        "contact_ratio_definition": "chi=Rth_each*Kbulk_module_midpoint",
        "parasitic_ratio_definition": "eta=Gpar/Kbulk_module_midpoint",
        "parasitic_ratios_checked": [0.0, 0.1, 0.2],
        "parasitic_first_variation_statement": (
            "A Seebeck-independent reservoir-to-reservoir heat leak is "
            "epsilon-independent at fixed reservoir temperatures, so it changes "
            "baseline Qc but its endpoint-zero first variation is exactly zero."
        ),
        "records": records,
        "maximum_relative_adjoint_fd_error": float(
            max(record["relative_adjoint_fd_error"] for record in records)
        ),
        "maximum_relative_energy_residual": float(
            max(record["maximum_relative_energy_residual"] for record in records)
        ),
    }


def analyze_device_resistance_scale(
    legs: tuple[TemperatureDependentLeg, TemperatureDependentLeg],
) -> dict[str, Any]:
    p_leg, n_leg = legs
    records = []
    for temperature, reported in ((303.0, 0.131), (343.0, 0.164)):
        rho_p = float(p_leg.electrical_resistivity.evaluate([temperature])[0])
        rho_n = float(n_leg.electrical_resistivity.evaluate([temperature])[0])
        bulk = PAIR_COUNT * LEG_LENGTH_M / LEG_AREA_M2 * (rho_p + rho_n)
        residual = reported - bulk
        records.append(
            {
                "temperature_k": temperature,
                "published_device_resistance_ohm": reported,
                "geometry_scaled_bulk_resistance_ohm": bulk,
                "published_minus_bulk_ohm": residual,
                "residual_fraction_of_published": residual / reported,
            }
        )
    n_contact_resistivity_ohm_m2 = 3.5e-10
    return {
        "records": records,
        "geometry": {
            "pair_count": PAIR_COUNT,
            "leg_length_m": LEG_LENGTH_M,
            "leg_width_m": LEG_WIDTH_M,
            "leg_area_m2": LEG_AREA_M2,
            "device_dimensions_m": [0.010, 0.010, 0.006],
            "contact_layer": "Ni electroplated on both leg surfaces",
            "solder": "Sn-Ag-Cu on low- and high-temperature sides",
        },
        "displayed_n_contact_scale": {
            "published_approximate_contact_resistivity_ohm_m2": (
                n_contact_resistivity_ohm_m2
            ),
            "single_interface_resistance_at_reported_leg_area_ohm": (
                n_contact_resistivity_ohm_m2 / LEG_AREA_M2
            ),
            "scope_warning": (
                "Fig. 6(c) is the displayed Cu-to-n-BTS+0.2%Cu stack only; "
                "it is not promoted to a full p/n module contact inventory."
            ),
        },
        "interpretation": (
            "The no-fit bulk prediction accounts for most of Rdevice at both "
            "reported temperatures.  The positive residual is retained as an "
            "unresolved aggregate of contacts, electrodes, wiring, temperature "
            "nonuniformity, and digitization rather than fitted away."
        ),
    }


def serialize_transport(transport: dict[str, FloatArray]) -> dict[str, Any]:
    return {key: np.asarray(value, dtype=float).tolist() for key, value in transport.items()}


def analyze() -> dict[str, Any]:
    transport, validation = validate_sources_and_load_transport()
    nominal_legs = build_legs(transport)
    invariants = analyze_invariants(nominal_legs)
    current_scan = analyze_current_scan(nominal_legs)
    primary = min(
        current_scan, key=lambda record: abs(record["current_a"] - PRIMARY_CURRENT_A)
    )
    zero_response_checks = analyze_zero_response_checks(transport, nominal_legs)
    method_corners = analyze_method_bound_corners(transport)
    contacts = analyze_contact_nuisance(transport)
    resistance = analyze_device_resistance_scale(nominal_legs)

    maximum_adjoint_fd_error = max(
        record["adjoint_fd_relative_error"]
        for record in current_scan
        if abs(record["adjoint_response_per_pair_w"]) > 1.0e-10
    )
    maximum_energy_residual = max(
        [record["maximum_relative_energy_residual"] for record in current_scan]
        + [
            method_corners["maximum_relative_energy_residual"],
            contacts["maximum_relative_energy_residual"],
        ]
    )
    current_scan_mode_bounded = all(
        record["normalized_mode_minimum_on_field"] >= -1.0e-10
        and record["normalized_mode_maximum_on_field"] <= 1.0 + 1.0e-10
        and record["temperature_minimum_k"] >= PRIMARY_COLD_TEMPERATURE_K - 1.0e-8
        and record["temperature_maximum_k"] <= PRIMARY_HOT_TEMPERATURE_K + 1.0e-8
        for record in current_scan
    )

    sign_identified = bool(not method_corners["interval_crosses_zero"])
    if sign_identified:
        limitation = (
            "The deterministic property range preserves the response sign, "
            "although its width still reflects the reported branch-level "
            "measurement limits."
        )
    else:
        limitation = (
            "The deterministic property range crosses zero, so the published "
            "BTS/BST data do not identify the physical sign or magnitude of a "
            "material-specific common-mode Thomson response."
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "generated_utc": "2026-08-26T00:00:00Z",
        "source_validation": validation,
        "transport": serialize_transport(transport),
        "model": {
            "equation": "[kappa A T']' + rho I_i^2/A - tau I_i T' = 0",
            "signed_branch_currents": {"p": "+I", "n": "-I"},
            "common_mode": (
                "deltaM=4*Mpk*(T-Tc)*(Th-T)/(Th-Tc)^2 added to both legs"
            ),
            "mode_peak_v_per_k_on_endpoint_interval": MODE_PEAK_V_PER_K,
            "primary_fixed_endpoint_window_k": [
                PRIMARY_COLD_TEMPERATURE_K,
                PRIMARY_HOT_TEMPERATURE_K,
            ],
            "primary_current_a": PRIMARY_CURRENT_A,
            "central_epsilon": CENTRAL_EPSILON,
            "current_scan_range_a": [0.0, PRIMARY_CURRENT_A],
            "probe_field_bound": "0 <= deltaM/Mpk <= 1 on every scanned field",
            "scope": (
                "material-supported fixed-endpoint framework test; not a "
                "reconstruction of the below-300-K DeltaTmax state"
            ),
        },
        "exact_invariants": invariants,
        "primary_nominal_result": primary,
        "current_scan": current_scan,
        "zero_response_checks": zero_response_checks,
        "method_bound_corner_envelope": method_corners,
        "thermal_contact_and_parasitic_nuisance": contacts,
        "device_resistance_and_geometry_scale": resistance,
        "validation_checks": {
            "source_hashes_and_identity_pass": True,
            "branch_complete_S_sigma_kappa_pass": True,
            "endpoint_and_alpha_invariants_pass": bool(
                max(
                    abs(invariants["mode_at_cold_v_per_k"]),
                    abs(invariants["mode_at_hot_v_per_k"]),
                    abs(invariants["maximum_pointwise_alpha_change_v_per_k"]),
                    abs(invariants["open_circuit_voltage_change_v"]),
                )
                < 1.0e-14
            ),
            "adjoint_vs_independent_nonlinear_pass": bool(
                maximum_adjoint_fd_error < 2.0e-5
                and method_corners["maximum_adjoint_fd_relative_error"] < 2.0e-5
                and contacts["maximum_relative_adjoint_fd_error"] < 2.0e-5
            ),
            "zero_response_checks_pass": bool(
                max(
                    abs(value)
                    for key, value in zero_response_checks.items()
                    if key.endswith("_w")
                )
                < 1.0e-10
            ),
            "energy_closure_pass": bool(maximum_energy_residual < 1.0e-10),
            "current_scan_probe_globally_bounded_pass": bool(
                current_scan_mode_bounded
            ),
            "no_sub_300_k_extrapolation": True,
            "unknown_device_boundaries_not_fitted": True,
        },
        "validation_summary": {
            "maximum_adjoint_fd_relative_error_current_scan": float(
                maximum_adjoint_fd_error
            ),
            "maximum_relative_energy_residual": float(maximum_energy_residual),
            "current_scan_probe_globally_bounded": bool(
                current_scan_mode_bounded
            ),
            "nominal_response_module_mw_per_10uV_per_K_peak": float(
                1.0e3 * primary["adjoint_response_module_w"]
            ),
            "method_bound_interval_module_mw": [
                float(
                    1.0e3
                    * method_corners["minimum"]["response_module_w"]
                ),
                float(
                    1.0e3
                    * method_corners["maximum"]["response_module_w"]
                ),
            ],
            "method_bound_interval_crosses_zero": method_corners[
                "interval_crosses_zero"
            ],
        },
        "interpretation": {
            "supported_result": (
                "The endpoint-zero adjoint law transfers to the independent "
                "BTS/BST material family and agrees with independent nonlinear "
                "BVP calculations."
            ),
            "limitation": limitation,
            "sign_identified_from_public_data": sign_identified,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    if not all(
        result["validation_checks"][key]
        for key in (
            "source_hashes_and_identity_pass",
            "branch_complete_S_sigma_kappa_pass",
            "endpoint_and_alpha_invariants_pass",
            "adjoint_vs_independent_nonlinear_pass",
            "zero_response_checks_pass",
            "energy_closure_pass",
            "current_scan_probe_globally_bounded_pass",
            "no_sub_300_k_extrapolation",
            "unknown_device_boundaries_not_fitted",
        )
    ):
        raise RuntimeError("one or more cross-material validation checks failed")
    return result


def write_scan_csv(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "current_a",
        "baseline_qc_module_w",
        "adjoint_response_module_w",
        "finite_difference_response_module_w",
        "adjoint_fd_relative_error",
        "maximum_relative_energy_residual",
        "temperature_minimum_k",
        "temperature_maximum_k",
        "normalized_mode_minimum_on_field",
        "normalized_mode_maximum_on_field",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in result["current_scan"]:
            writer.writerow({key: record[key] for key in fields})


def make_figure(result: dict[str, Any], stem: Path) -> dict[str, Any]:
    """Diagnostic figure for reproducibility, response, identifiability, and scale."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    transport = result["transport"]
    temperature = np.asarray(transport["temperature_k"], dtype=float)
    p_s = 1.0e6 * np.asarray(transport["p_seebeck_v_per_k"], dtype=float)
    n_s = 1.0e6 * np.asarray(transport["n_seebeck_v_per_k"], dtype=float)
    scan = result["current_scan"]
    currents = np.asarray([record["current_a"] for record in scan])
    adjoint_mw = 1.0e3 * np.asarray(
        [record["adjoint_response_module_w"] for record in scan]
    )
    fd_mw = 1.0e3 * np.asarray(
        [record["finite_difference_response_module_w"] for record in scan]
    )

    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(temperature, p_s, "o-", color="#D55E00", label="p-BST")
    ax.plot(temperature, n_s, "s-", color="#0072B2", label="n-BTS+0.2%Cu")
    ax.axvspan(
        PRIMARY_COLD_TEMPERATURE_K,
        PRIMARY_HOT_TEMPERATURE_K,
        color="#E6AB02",
        alpha=0.14,
        label="323-373 K test",
    )
    ax.set(xlabel="temperature (K)", ylabel=r"Seebeck $S$ ($\mu$V K$^{-1}$)")
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.text(
        -0.12, 1.03, "a", transform=ax.transAxes, va="top", fontweight="bold",
        clip_on=False,
    )

    ax = axes[0, 1]
    monotone = currents <= PRIMARY_CURRENT_A
    ax.plot(
        currents[monotone], adjoint_mw[monotone], "-", lw=1.8,
        color="#0072B2", label="adjoint",
    )
    ax.plot(
        currents[monotone],
        fd_mw[monotone],
        "o",
        ms=3.4,
        mfc="white",
        mec="#D55E00",
        label="nonlinear central difference",
    )
    ax.axvline(PRIMARY_CURRENT_A, color="0.55", lw=0.8, ls="--")
    ax.set(
        xlabel="module current (A)",
        ylabel=r"$\delta Q_c$ (mW; $M_{pk}=10$ $\mu$V K$^{-1}$)",
    )
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(
        -0.12, 1.03, "b", transform=ax.transAxes, va="top", fontweight="bold",
        clip_on=False,
    )

    ax = axes[1, 0]
    envelope = result["method_bound_corner_envelope"]
    low = 1.0e3 * envelope["minimum"]["response_module_w"]
    high = 1.0e3 * envelope["maximum"]["response_module_w"]
    nominal = result["validation_summary"][
        "nominal_response_module_mw_per_10uV_per_K_peak"
    ]
    ax.hlines(1.0, low, high, color="#CC79A7", lw=5, alpha=0.65)
    ax.plot(nominal, 1.0, "D", color="black", ms=5)
    contacts = result["thermal_contact_and_parasitic_nuisance"]["records"]
    contact_values = [
        1.0e3 * record["contact_dressed_adjoint_response_module_w"]
        for record in contacts
    ]
    ax.plot(
        contact_values,
        np.full(len(contact_values), 0.64),
        "o-",
        color="#009E73",
        ms=3.5,
    )
    ax.axvline(0.0, color="0.25", lw=0.9)
    ax.set_yticks([0.64, 1.0], ["contact stress", "method-bound\ncorners"])
    ax.set_xlabel(r"$\delta Q_c$ at 3 A (mW; $M_{pk}=10$ $\mu$V K$^{-1}$)")
    ax.set_ylim(0.40, 1.28)
    ax.text(
        -0.12, 1.03, "c", transform=ax.transAxes, va="top", fontweight="bold",
        clip_on=False,
    )
    ax.text(
        0.03, 0.88, "64 method-bound corners: sign crosses zero",
        transform=ax.transAxes,
        fontsize=7,
        color="#8E3B72",
    )
    ax.annotate(
        "nominal", (nominal, 1.0), xytext=(5, 6), textcoords="offset points",
        fontsize=7,
    )
    ax.annotate(
        r"$\chi=0$-$0.25$", (max(contact_values), 0.64),
        xytext=(5, -11), textcoords="offset points", fontsize=7,
        color="#007A58",
    )

    ax = axes[1, 1]
    resistance = result["device_resistance_and_geometry_scale"]["records"]
    labels = [f"{record['temperature_k']:.0f} K" for record in resistance]
    x = np.arange(len(labels), dtype=float)
    bulk = np.asarray(
        [record["geometry_scaled_bulk_resistance_ohm"] for record in resistance]
    )
    published = np.asarray(
        [record["published_device_resistance_ohm"] for record in resistance]
    )
    width = 0.34
    ax.bar(x - width / 2, bulk, width, color="#56B4E9", label="no-fit bulk")
    ax.bar(
        x + width / 2,
        published,
        width,
        facecolor="white",
        edgecolor="black",
        label=r"published $R_{device}$",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"seven-pair resistance ($\Omega$)")
    ax.set_ylim(0.0, 0.19)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(
        -0.12, 1.03, "d", transform=ax.transAxes, va="top", fontweight="bold",
        clip_on=False,
    )
    for axis in axes.flat:
        axis.tick_params(labelsize=8, direction="out", length=3)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    pdf = stem.with_suffix(".pdf")
    tiff = stem.with_suffix(".tiff")
    fixed_time = datetime(2026, 8, 26, tzinfo=timezone.utc)
    figure.savefig(
        png,
        dpi=300,
        metadata={"Software": "matplotlib", "Creation Time": "2026-08-26T00:00:00Z"},
    )
    figure.savefig(
        svg,
        metadata={
            "Date": "2026-08-26",
            "Creator": None,
            "Title": "BTS/BST endpoint-zero common-mode validation",
        },
    )
    figure.savefig(
        pdf,
        metadata={
            "Title": "BTS/BST endpoint-zero common-mode validation",
            "Author": "reproducible analysis",
            "Creator": "matplotlib",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        },
    )
    figure.savefig(tiff, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)
    return {
        "png": relative_locator(png),
        "svg": relative_locator(svg),
        "pdf": relative_locator(pdf),
        "tiff": relative_locator(tiff),
        "layout": "cross_material_transfer_with_property_bounds",
        "conclusion": (
            "The adjoint transfer agrees with independent nonlinear calculations, "
            "but the property range obtained from the published measurement limits "
            "does not identify its physical sign or magnitude."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-scan-csv", type=Path, default=DEFAULT_SCAN_CSV)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze()
    write_scan_csv(result, args.output_scan_csv)
    result["outputs"] = {
        "results_json": relative_locator(args.output_json),
        "current_scan_csv": relative_locator(args.output_scan_csv),
        "figure": make_figure(result, args.figure_stem),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = result["validation_summary"]
    print(
        "BTS/BST endpoint-zero analysis complete: "
        f"nominal={summary['nominal_response_module_mw_per_10uV_per_K_peak']:.6f} mW; "
        f"method-bound interval={summary['method_bound_interval_module_mw']}; "
        "sign_identified="
        f"{result['interpretation']['sign_identified_from_public_data']}"
    )


if __name__ == "__main__":
    main()
