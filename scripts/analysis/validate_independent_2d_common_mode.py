#!/usr/bin/env python3
"""Independent conservative 2-D validation of Seebeck common-mode physics.

This script intentionally does not import the repository one-dimensional TEC
solver.  Each thermoelectric branch is represented by a rectangular 2-D
finite-volume domain with unit-cell-scale transverse heterogeneity.  The
electrical problem is solved first,

    div[sigma(x,y) grad(Psi)] = 0,       J = -sigma grad(Psi),

where ``Psi = phi + integral S(T)dT`` and the total branch current is fixed by
scaling the electrochemical-potential solution.  The resulting divergence-free
two-dimensional current field drives an independently assembled thermal model,

    div(k grad T) + rho |J|^2 - tau(T) J.grad(T) = 0.

Electrical edge dissipation is distributed conservatively to thermal control
volumes.  The Thomson term is also conservative: for the shared perturbation
``m(T)=b(T-T0)``, ``tau=epsilon*b*T`` and
``tau J.grad(T)=div[J*(epsilon*b*T**2/2)]``.  This construction closes terminal
heat, side loss, and electrical power to the finite-volume solver tolerance.

The model is deliberately limited.  It resolves 2-D current crowding, spatially
heterogeneous temperature-independent rho and k, branch-specific linear base
Seebeck laws (and hence base Thomson transport), asymmetric Robin side losses,
Joule heat, and temperature-dependent common Seebeck perturbations.  It does
not include temperature-dependent rho/k, contact mechanics, radiation, full
3-D spreading, or experimental calibration, and must not be described as a
COMSOL or experimental validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import scipy
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import erf


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "independent_2d_common_mode_validation/v1"
ANALYSIS_ID = "SCI-INDEPENDENT-2D-COMMON-MODE-20260826"
FIXED_TIMESTAMP = datetime(2026, 8, 26, tzinfo=timezone.utc)
DEFAULT_OUTPUT = (
    ROOT
    / "results/scientific_analysis/independent_2d_common_mode_validation_results.json"
)
DEFAULT_FIGURE_PREFIX = (
    ROOT / "results/scientific_analysis/independent_2d_common_mode_validation"
)

FloatArray = NDArray[np.float64]


class ValidationError(RuntimeError):
    """Raised when a physical or numerical acceptance criterion fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locator(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def binding(path: Path) -> dict[str, Any]:
    return {
        "locator": locator(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _finite(value: float, label: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"{label} is non-finite")
    return result


@dataclass(frozen=True)
class Grid2D:
    nx: int
    ny: int
    length_m: float
    width_m: float
    depth_m: float

    def __post_init__(self) -> None:
        require(self.nx >= 4 and self.ny >= 4, "2-D grid is too small")
        require(
            self.length_m > 0.0 and self.width_m > 0.0 and self.depth_m > 0.0,
            "grid dimensions must be positive",
        )

    @property
    def dx(self) -> float:
        return self.length_m / self.nx

    @property
    def dy(self) -> float:
        return self.width_m / self.ny

    @property
    def volume(self) -> float:
        return self.dx * self.dy * self.depth_m

    @property
    def size(self) -> int:
        return self.nx * self.ny

    @property
    def x_centres(self) -> FloatArray:
        return (np.arange(self.nx, dtype=float) + 0.5) * self.dx

    @property
    def y_centres(self) -> FloatArray:
        return (np.arange(self.ny, dtype=float) + 0.5) * self.dy

    def index(self, ix: int, iy: int) -> int:
        return ix * self.ny + iy


@dataclass(frozen=True)
class BranchConfig:
    name: str
    signed_current_a: float
    seebeck_base_v_per_k: float
    seebeck_base_slope_v_per_k2: float = 0.0
    length_m: float = 1.2e-3
    width_m: float = 0.80e-3
    depth_m: float = 0.80e-3
    rho0_ohm_m: float = 1.4e-5
    k0_w_per_mk: float = 1.25
    pattern_phase: float = 0.0
    side_h_bottom_w_per_m2k: float = 110.0
    side_h_top_w_per_m2k: float = 520.0
    side_ambient_bottom_k: float = 316.0
    side_ambient_top_k: float = 294.0


@dataclass(frozen=True)
class CommonModeBasis:
    """A common perturbation ``m(T)`` and its ``Gamma=T*dm/dT`` basis."""

    name: str
    kind: str
    anchor_k: float = 300.0
    amplitude: float = 1.0e-6
    centre_k: float = 330.0
    width_k: float = 12.0

    def gamma(self, temperature_k: FloatArray | float) -> FloatArray:
        temperature = np.asarray(temperature_k, dtype=float)
        if self.kind == "constant_gamma":
            return np.full_like(temperature, self.amplitude, dtype=float)
        if self.kind == "linear_gamma":
            return self.amplitude * temperature
        if self.kind == "localized_gamma":
            z = (temperature - self.centre_k) / self.width_k
            return self.amplitude * temperature * np.exp(-0.5 * z**2)
        raise ValidationError(f"unknown common-mode basis: {self.kind}")

    def mode(self, temperature_k: FloatArray | float) -> FloatArray:
        """Return m(T), anchored to zero at ``anchor_k``."""

        temperature = np.asarray(temperature_k, dtype=float)
        if self.kind == "constant_gamma":
            require(bool(np.all(temperature > 0.0)), "constant-Gamma basis requires T>0")
            return self.amplitude * np.log(temperature / self.anchor_k)
        if self.kind == "linear_gamma":
            return self.amplitude * (temperature - self.anchor_k)
        if self.kind == "localized_gamma":
            factor = self.amplitude * self.width_k * math.sqrt(math.pi / 2.0)
            z = (temperature - self.centre_k) / (math.sqrt(2.0) * self.width_k)
            za = (self.anchor_k - self.centre_k) / (
                math.sqrt(2.0) * self.width_k
            )
            return factor * (erf(z) - erf(za))
        raise ValidationError(f"unknown common-mode basis: {self.kind}")

    def thomson_primitive(self, temperature_k: FloatArray | float) -> FloatArray:
        """Return G(T)-G(anchor), where G'(T)=Gamma(T)."""

        temperature = np.asarray(temperature_k, dtype=float)
        if self.kind == "constant_gamma":
            return self.amplitude * (temperature - self.anchor_k)
        if self.kind == "linear_gamma":
            return 0.5 * self.amplitude * (
                temperature**2 - self.anchor_k**2
            )
        if self.kind == "localized_gamma":
            width = self.width_k
            centre = self.centre_k

            def raw(value: FloatArray | float) -> FloatArray:
                value_array = np.asarray(value, dtype=float)
                z = (value_array - centre) / width
                return self.amplitude * (
                    -(width**2) * np.exp(-0.5 * z**2)
                    + centre
                    * width
                    * math.sqrt(math.pi / 2.0)
                    * erf(z / math.sqrt(2.0))
                )

            return raw(temperature) - raw(self.anchor_k)
        raise ValidationError(f"unknown common-mode basis: {self.kind}")

    def seebeck_primitive_increment(
        self, temperature_k: FloatArray | float
    ) -> FloatArray:
        """Return an antiderivative of m(T): T*m(T)-G(T)."""

        temperature = np.asarray(temperature_k, dtype=float)
        return temperature * self.mode(temperature) - self.thomson_primitive(
            temperature
        )


def common_mode_bases() -> tuple[CommonModeBasis, ...]:
    return (
        CommonModeBasis(
            name="constant_Gamma",
            kind="constant_gamma",
            amplitude=3.0e-4,
        ),
        CommonModeBasis(
            name="linear_Gamma",
            kind="linear_gamma",
            amplitude=1.0e-6,
        ),
        CommonModeBasis(
            name="localized_Gamma",
            kind="localized_gamma",
            amplitude=1.4e-6,
            centre_k=330.0,
            width_k=12.0,
        ),
    )


@dataclass(frozen=True)
class ElectricalState:
    psi_unit: FloatArray
    ix_faces_a: FloatArray
    iy_faces_a: FloatArray
    joule_power_by_cell_w: FloatArray
    joule_power_total_w: float
    electrochemical_drop_v: float
    effective_resistance_ohm: float
    divergence_max_a: float
    terminal_current_mismatch_a: float


@dataclass(frozen=True)
class ThermalAssembly:
    matrix: csr_matrix
    boundary_rhs_w: FloatArray
    cold_conductance_w_per_k: FloatArray
    hot_conductance_w_per_k: FloatArray
    bottom_conductance_w_per_k: FloatArray
    top_conductance_w_per_k: FloatArray
    cold_temperature_k: FloatArray
    hot_temperature_k: FloatArray
    bottom_ambient_k: FloatArray
    top_ambient_k: FloatArray


@dataclass(frozen=True)
class ThermalState:
    temperature_k: FloatArray
    epsilon: float
    iterations: int
    relative_residual: float
    converged: bool


@dataclass(frozen=True)
class BranchModel:
    config: BranchConfig
    grid: Grid2D
    rho_ohm_m: FloatArray
    k_w_per_mk: FloatArray
    electrical: ElectricalState
    thermal: ThermalAssembly
    thomson_matrix: csr_matrix
    base_thomson_boundary_w: FloatArray
    thomson_boundary_w: FloatArray
    base_seebeck_basis: CommonModeBasis
    common_basis: CommonModeBasis


def _harmonic(a: float, b: float) -> float:
    require(a > 0.0 and b > 0.0, "harmonic inputs must be positive")
    return 2.0 * a * b / (a + b)


def material_fields(grid: Grid2D, config: BranchConfig) -> tuple[FloatArray, FloatArray]:
    """Create smooth positive heterogeneity with no separable 1-D reduction."""

    xx, yy = np.meshgrid(
        grid.x_centres / grid.length_m,
        grid.y_centres / grid.width_m,
        indexing="ij",
    )
    phase = config.pattern_phase
    rho_log_pattern = (
        0.33 * np.sin(2.0 * np.pi * xx + phase) * np.cos(np.pi * yy)
        + 0.17 * (yy - 0.5)
        + 0.11 * np.sin(3.0 * np.pi * xx - 2.0 * np.pi * yy + 0.4 + phase)
    )
    k_pattern = (
        1.0
        + 0.27 * np.cos(2.0 * np.pi * xx + 0.3 + phase) * np.sin(np.pi * yy)
        + 0.16 * (yy - 0.5)
        + 0.09 * np.sin(np.pi * xx + 2.0 * np.pi * yy - phase)
    )
    rho = config.rho0_ohm_m * np.exp(rho_log_pattern)
    k = config.k0_w_per_mk * k_pattern
    require(float(np.min(rho)) > 0.0, "rho field lost positivity")
    require(float(np.min(k)) > 0.0, "k field lost positivity")
    return rho.astype(float), k.astype(float)


def solve_electrical(
    grid: Grid2D,
    rho_ohm_m: FloatArray,
    signed_current_a: float,
) -> ElectricalState:
    """Solve an independent heterogeneous resistor-network finite volume model."""

    require(rho_ohm_m.shape == (grid.nx, grid.ny), "rho shape mismatch")
    sigma = 1.0 / rho_ohm_m
    matrix = lil_matrix((grid.size, grid.size), dtype=float)
    rhs = np.zeros(grid.size, dtype=float)

    gx = np.zeros((grid.nx + 1, grid.ny), dtype=float)
    gy = np.zeros((grid.nx, grid.ny + 1), dtype=float)
    face_area_x = grid.dy * grid.depth_m
    face_area_y = grid.dx * grid.depth_m

    for iy in range(grid.ny):
        gx[0, iy] = sigma[0, iy] * face_area_x / (0.5 * grid.dx)
        gx[-1, iy] = sigma[-1, iy] * face_area_x / (0.5 * grid.dx)
    for ix in range(1, grid.nx):
        for iy in range(grid.ny):
            gx[ix, iy] = (
                _harmonic(sigma[ix - 1, iy], sigma[ix, iy])
                * face_area_x
                / grid.dx
            )
    for ix in range(grid.nx):
        for iy in range(1, grid.ny):
            gy[ix, iy] = (
                _harmonic(sigma[ix, iy - 1], sigma[ix, iy])
                * face_area_y
                / grid.dy
            )

    for ix in range(grid.nx):
        for iy in range(grid.ny):
            row = grid.index(ix, iy)
            if ix == 0:
                matrix[row, row] += gx[0, iy]
                rhs[row] += gx[0, iy]  # Psi_left = 1 V
            else:
                neighbour = grid.index(ix - 1, iy)
                conductance = gx[ix, iy]
                matrix[row, row] += conductance
                matrix[row, neighbour] -= conductance
            if ix == grid.nx - 1:
                matrix[row, row] += gx[-1, iy]  # Psi_right = 0 V
            else:
                neighbour = grid.index(ix + 1, iy)
                conductance = gx[ix + 1, iy]
                matrix[row, row] += conductance
                matrix[row, neighbour] -= conductance
            if iy > 0:
                neighbour = grid.index(ix, iy - 1)
                conductance = gy[ix, iy]
                matrix[row, row] += conductance
                matrix[row, neighbour] -= conductance
            if iy < grid.ny - 1:
                neighbour = grid.index(ix, iy + 1)
                conductance = gy[ix, iy + 1]
                matrix[row, row] += conductance
                matrix[row, neighbour] -= conductance

    matrix_csr = matrix.tocsr()
    psi = np.asarray(spsolve(matrix_csr, rhs), dtype=float).reshape(grid.nx, grid.ny)

    ix_unit = np.zeros_like(gx)
    iy_unit = np.zeros_like(gy)
    ix_unit[0, :] = gx[0, :] * (1.0 - psi[0, :])
    ix_unit[-1, :] = gx[-1, :] * psi[-1, :]
    for ix in range(1, grid.nx):
        ix_unit[ix, :] = gx[ix, :] * (psi[ix - 1, :] - psi[ix, :])
    for iy in range(1, grid.ny):
        iy_unit[:, iy] = gy[:, iy] * (psi[:, iy - 1] - psi[:, iy])

    unit_left = float(np.sum(ix_unit[0, :]))
    unit_right = float(np.sum(ix_unit[-1, :]))
    unit_current = 0.5 * (unit_left + unit_right)
    require(unit_current > 0.0, "electrical unit conductance is non-positive")
    scale = signed_current_a / unit_current
    ix_faces = ix_unit * scale
    iy_faces = iy_unit * scale

    divergence = (
        ix_faces[1:, :] - ix_faces[:-1, :]
        + iy_faces[:, 1:] - iy_faces[:, :-1]
    )
    terminal_mismatch = abs(float(np.sum(ix_faces[0, :]) - np.sum(ix_faces[-1, :])))

    joule = np.zeros((grid.nx, grid.ny), dtype=float)
    # Boundary half-cell resistor powers belong wholly to their adjacent cell.
    joule[0, :] += gx[0, :] * (scale * (1.0 - psi[0, :])) ** 2
    joule[-1, :] += gx[-1, :] * (scale * psi[-1, :]) ** 2
    # Interior edge powers are split equally between adjacent control volumes.
    for ix in range(1, grid.nx):
        power = gx[ix, :] * (scale * (psi[ix - 1, :] - psi[ix, :])) ** 2
        joule[ix - 1, :] += 0.5 * power
        joule[ix, :] += 0.5 * power
    for iy in range(1, grid.ny):
        power = gy[:, iy] * (scale * (psi[:, iy - 1] - psi[:, iy])) ** 2
        joule[:, iy - 1] += 0.5 * power
        joule[:, iy] += 0.5 * power

    delta_psi = scale
    network_power = signed_current_a * delta_psi
    joule_total = float(np.sum(joule))
    require(
        abs(joule_total - network_power) <= 2.0e-11 * max(joule_total, 1.0e-30),
        "electrical network did not conserve Joule power",
    )
    require(
        float(np.max(np.abs(divergence))) <= 5.0e-11 * max(abs(signed_current_a), 1.0),
        "electrical finite volumes did not conserve current",
    )

    return ElectricalState(
        psi_unit=psi.reshape(-1),
        ix_faces_a=ix_faces,
        iy_faces_a=iy_faces,
        joule_power_by_cell_w=joule.reshape(-1),
        joule_power_total_w=joule_total,
        electrochemical_drop_v=delta_psi,
        effective_resistance_ohm=abs(delta_psi / signed_current_a),
        divergence_max_a=float(np.max(np.abs(divergence))),
        terminal_current_mismatch_a=terminal_mismatch,
    )


def _as_boundary(values: float | Iterable[float], count: int, label: str) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(count, float(array), dtype=float)
    require(array.shape == (count,), f"{label} boundary shape mismatch")
    require(bool(np.all(np.isfinite(array))), f"{label} boundary is non-finite")
    return array


def assemble_thermal(
    grid: Grid2D,
    k_w_per_mk: FloatArray,
    config: BranchConfig,
    cold_temperature_k: float | Iterable[float],
    hot_temperature_k: float | Iterable[float],
) -> ThermalAssembly:
    """Assemble ``-div(k grad T)`` with Dirichlet ends and Robin sides."""

    cold = _as_boundary(cold_temperature_k, grid.ny, "cold")
    hot = _as_boundary(hot_temperature_k, grid.ny, "hot")
    bottom_ambient = np.full(grid.nx, config.side_ambient_bottom_k, dtype=float)
    top_ambient = np.full(grid.nx, config.side_ambient_top_k, dtype=float)

    matrix = lil_matrix((grid.size, grid.size), dtype=float)
    rhs = np.zeros(grid.size, dtype=float)
    cold_g = np.zeros(grid.ny, dtype=float)
    hot_g = np.zeros(grid.ny, dtype=float)
    bottom_g = np.zeros(grid.nx, dtype=float)
    top_g = np.zeros(grid.nx, dtype=float)
    area_x = grid.dy * grid.depth_m
    area_y = grid.dx * grid.depth_m

    for iy in range(grid.ny):
        cold_g[iy] = k_w_per_mk[0, iy] * area_x / (0.5 * grid.dx)
        hot_g[iy] = k_w_per_mk[-1, iy] * area_x / (0.5 * grid.dx)
    for ix in range(grid.nx):
        kb = k_w_per_mk[ix, 0]
        kt = k_w_per_mk[ix, -1]
        bottom_g[ix] = area_y / (
            0.5 * grid.dy / kb + 1.0 / config.side_h_bottom_w_per_m2k
        )
        top_g[ix] = area_y / (
            0.5 * grid.dy / kt + 1.0 / config.side_h_top_w_per_m2k
        )

    for ix in range(grid.nx):
        for iy in range(grid.ny):
            row = grid.index(ix, iy)
            if ix == 0:
                matrix[row, row] += cold_g[iy]
                rhs[row] += cold_g[iy] * cold[iy]
            else:
                conductance = (
                    _harmonic(k_w_per_mk[ix - 1, iy], k_w_per_mk[ix, iy])
                    * area_x
                    / grid.dx
                )
                matrix[row, row] += conductance
                matrix[row, grid.index(ix - 1, iy)] -= conductance
            if ix == grid.nx - 1:
                matrix[row, row] += hot_g[iy]
                rhs[row] += hot_g[iy] * hot[iy]
            else:
                conductance = (
                    _harmonic(k_w_per_mk[ix, iy], k_w_per_mk[ix + 1, iy])
                    * area_x
                    / grid.dx
                )
                matrix[row, row] += conductance
                matrix[row, grid.index(ix + 1, iy)] -= conductance
            if iy == 0:
                matrix[row, row] += bottom_g[ix]
                rhs[row] += bottom_g[ix] * bottom_ambient[ix]
            else:
                conductance = (
                    _harmonic(k_w_per_mk[ix, iy - 1], k_w_per_mk[ix, iy])
                    * area_y
                    / grid.dy
                )
                matrix[row, row] += conductance
                matrix[row, grid.index(ix, iy - 1)] -= conductance
            if iy == grid.ny - 1:
                matrix[row, row] += top_g[ix]
                rhs[row] += top_g[ix] * top_ambient[ix]
            else:
                conductance = (
                    _harmonic(k_w_per_mk[ix, iy], k_w_per_mk[ix, iy + 1])
                    * area_y
                    / grid.dy
                )
                matrix[row, row] += conductance
                matrix[row, grid.index(ix, iy + 1)] -= conductance

    return ThermalAssembly(
        matrix=matrix.tocsr(),
        boundary_rhs_w=rhs,
        cold_conductance_w_per_k=cold_g,
        hot_conductance_w_per_k=hot_g,
        bottom_conductance_w_per_k=bottom_g,
        top_conductance_w_per_k=top_g,
        cold_temperature_k=cold,
        hot_temperature_k=hot,
        bottom_ambient_k=bottom_ambient,
        top_ambient_k=top_ambient,
    )


def assemble_thomson_divergence(
    grid: Grid2D,
    electrical: ElectricalState,
    thermal: ThermalAssembly,
) -> tuple[csr_matrix, FloatArray]:
    """Return ``C`` and boundary current weights for conservative div[J G]."""

    matrix = lil_matrix((grid.size, grid.size), dtype=float)
    boundary = np.zeros(grid.size, dtype=float)
    ix_faces = electrical.ix_faces_a
    iy_faces = electrical.iy_faces_a

    for ix in range(1, grid.nx):
        for iy in range(grid.ny):
            left = grid.index(ix - 1, iy)
            right = grid.index(ix, iy)
            current = ix_faces[ix, iy]
            for column in (left, right):
                matrix[left, column] += 0.5 * current
                matrix[right, column] -= 0.5 * current
    for ix in range(grid.nx):
        for iy in range(1, grid.ny):
            bottom = grid.index(ix, iy - 1)
            top = grid.index(ix, iy)
            current = iy_faces[ix, iy]
            for column in (bottom, top):
                matrix[bottom, column] += 0.5 * current
                matrix[top, column] -= 0.5 * current

    # Store outward boundary current weights.  The caller multiplies these by
    # the selected basis G(T), so this assembly is shared by all Gamma bases.
    for iy in range(grid.ny):
        boundary[grid.index(0, iy)] = -ix_faces[0, iy]
        boundary[grid.index(grid.nx - 1, iy)] = ix_faces[-1, iy]
    return matrix.tocsr(), boundary


def build_branch_model(
    config: BranchConfig,
    nx: int,
    ny: int,
    cold_temperature_k: float | Iterable[float] = 300.0,
    hot_temperature_k: float | Iterable[float] = 350.0,
    common_basis: CommonModeBasis | None = None,
) -> BranchModel:
    if common_basis is None:
        common_basis = common_mode_bases()[1]
    grid = Grid2D(nx, ny, config.length_m, config.width_m, config.depth_m)
    rho, k = material_fields(grid, config)
    electrical = solve_electrical(grid, rho, config.signed_current_a)
    thermal = assemble_thermal(grid, k, config, cold_temperature_k, hot_temperature_k)
    thomson_matrix, boundary_current_weights = assemble_thomson_divergence(
        grid, electrical, thermal
    )
    base_seebeck_basis = CommonModeBasis(
        name=f"{config.name}_base_linear_Gamma",
        kind="linear_gamma",
        anchor_k=300.0,
        amplitude=config.seebeck_base_slope_v_per_k2,
    )
    base_thomson_boundary = np.zeros(grid.size, dtype=float)
    thomson_boundary = np.zeros(grid.size, dtype=float)
    for iy in range(grid.ny):
        base_thomson_boundary[grid.index(0, iy)] = (
            boundary_current_weights[grid.index(0, iy)]
            * float(
                base_seebeck_basis.thomson_primitive(
                    thermal.cold_temperature_k[iy]
                )
            )
        )
        base_thomson_boundary[grid.index(grid.nx - 1, iy)] = (
            boundary_current_weights[grid.index(grid.nx - 1, iy)]
            * float(
                base_seebeck_basis.thomson_primitive(
                    thermal.hot_temperature_k[iy]
                )
            )
        )
        thomson_boundary[grid.index(0, iy)] = (
            boundary_current_weights[grid.index(0, iy)]
            * float(common_basis.thomson_primitive(thermal.cold_temperature_k[iy]))
        )
        thomson_boundary[grid.index(grid.nx - 1, iy)] = (
            boundary_current_weights[grid.index(grid.nx - 1, iy)]
            * float(common_basis.thomson_primitive(thermal.hot_temperature_k[iy]))
        )
    return BranchModel(
        config=config,
        grid=grid,
        rho_ohm_m=rho,
        k_w_per_mk=k,
        electrical=electrical,
        thermal=thermal,
        thomson_matrix=thomson_matrix,
        base_thomson_boundary_w=base_thomson_boundary,
        thomson_boundary_w=thomson_boundary,
        base_seebeck_basis=base_seebeck_basis,
        common_basis=common_basis,
    )


def thomson_divergence(model: BranchModel, temperature_k: FloatArray) -> FloatArray:
    g_cell = model.common_basis.thomson_primitive(temperature_k)
    return np.asarray(
        model.thomson_matrix @ g_cell + model.thomson_boundary_w,
        dtype=float,
    )


def base_thomson_divergence(
    model: BranchModel, temperature_k: FloatArray
) -> FloatArray:
    g_cell = model.base_seebeck_basis.thomson_primitive(temperature_k)
    return np.asarray(
        model.thomson_matrix @ g_cell + model.base_thomson_boundary_w,
        dtype=float,
    )


def solve_thermal(
    model: BranchModel,
    epsilon: float,
    initial_temperature_k: FloatArray | None = None,
    relative_tolerance: float = 2.0e-11,
    maximum_iterations: int = 30,
) -> ThermalState:
    rhs = model.thermal.boundary_rhs_w + model.electrical.joule_power_by_cell_w
    if initial_temperature_k is None:
        temperature = np.asarray(spsolve(model.thermal.matrix, rhs), dtype=float)
    else:
        temperature = np.asarray(initial_temperature_k, dtype=float).copy()
    require(temperature.shape == (model.grid.size,), "thermal initial state shape mismatch")

    scale = max(float(np.max(np.abs(rhs))), 1.0e-30)
    converged = False
    relative = math.inf
    iteration = 0
    for iteration in range(1, maximum_iterations + 1):
        base_advective = base_thomson_divergence(model, temperature)
        perturbation_advective = thomson_divergence(model, temperature)
        residual = (
            model.thermal.matrix @ temperature
            - rhs
            + base_advective
            + epsilon * perturbation_advective
        )
        relative = float(np.max(np.abs(residual))) / scale
        if relative <= relative_tolerance:
            converged = True
            break
        gamma_diagonal = diags(
            model.base_seebeck_basis.gamma(temperature)
            + epsilon * model.common_basis.gamma(temperature),
            offsets=0,
            shape=(model.grid.size, model.grid.size),
            format="csr",
        )
        jacobian = model.thermal.matrix + model.thomson_matrix @ gamma_diagonal
        correction = np.asarray(spsolve(jacobian, -residual), dtype=float)

        damping = 1.0
        accepted = False
        while damping >= 1.0 / 64.0:
            trial = temperature + damping * correction
            trial_residual = (
                model.thermal.matrix @ trial
                - rhs
                + base_thomson_divergence(model, trial)
                + epsilon * thomson_divergence(model, trial)
            )
            if float(np.max(np.abs(trial_residual))) < float(np.max(np.abs(residual))):
                temperature = trial
                accepted = True
                break
            damping *= 0.5
        require(accepted, "thermal Newton line search failed")

    require(converged, f"thermal Newton solve did not converge: residual={relative:.3e}")
    require(bool(np.all(np.isfinite(temperature))), "thermal state is non-finite")
    require(float(np.min(temperature)) > 0.0, "thermal state is non-physical")
    return ThermalState(temperature, epsilon, iteration, relative, converged)


def branch_metrics(
    model: BranchModel,
    state: ThermalState,
    constant_shift_v_per_k: float = 0.0,
) -> dict[str, float]:
    grid = model.grid
    temperature = state.temperature_k.reshape(grid.nx, grid.ny)
    cold = model.thermal.cold_temperature_k
    hot = model.thermal.hot_temperature_k
    epsilon = state.epsilon
    basis = model.common_basis
    base_basis = model.base_seebeck_basis
    base = model.config.seebeck_base_v_per_k + constant_shift_v_per_k

    def seebeck(boundary_temperature: FloatArray) -> FloatArray:
        return (
            base
            + base_basis.mode(boundary_temperature)
            + epsilon * basis.mode(boundary_temperature)
        )

    def primitive(boundary_temperature: FloatArray) -> FloatArray:
        return (
            base * boundary_temperature
            + base_basis.seebeck_primitive_increment(boundary_temperature)
            + epsilon
            * basis.seebeck_primitive_increment(boundary_temperature)
        )

    ix = model.electrical.ix_faces_a
    qc_peltier = float(np.sum(seebeck(cold) * cold * ix[0, :]))
    qh_peltier = float(np.sum(seebeck(hot) * hot * ix[-1, :]))
    qc_conduction = float(
        np.sum(model.thermal.cold_conductance_w_per_k * (cold - temperature[0, :]))
    )
    qh_conduction = float(
        np.sum(model.thermal.hot_conductance_w_per_k * (temperature[-1, :] - hot))
    )
    q_side_bottom = float(
        np.sum(
            model.thermal.bottom_conductance_w_per_k
            * (temperature[:, 0] - model.thermal.bottom_ambient_k)
        )
    )
    q_side_top = float(
        np.sum(
            model.thermal.top_conductance_w_per_k
            * (temperature[:, -1] - model.thermal.top_ambient_k)
        )
    )
    qc = qc_peltier + qc_conduction
    qh = qh_peltier + qh_conduction
    qside = q_side_bottom + q_side_top
    seebeck_power = float(
        np.sum(ix[-1, :] * primitive(hot) - ix[0, :] * primitive(cold))
    )
    electrical_power = model.electrical.joule_power_total_w + seebeck_power
    energy_residual = qh - qc + qside - electrical_power
    return {
        "Qc_w": _finite(qc, "Qc"),
        "Qh_w": _finite(qh, "Qh"),
        "Qside_w": _finite(qside, "Qside"),
        "P_electrical_w": _finite(electrical_power, "electrical power"),
        "energy_residual_w": _finite(energy_residual, "energy residual"),
        "Qc_peltier_w": qc_peltier,
        "Qc_conduction_w": qc_conduction,
        "Qh_peltier_w": qh_peltier,
        "Qh_conduction_w": qh_conduction,
        "Joule_power_w": model.electrical.joule_power_total_w,
        "Seebeck_power_w": seebeck_power,
    }


def solve_adjoint_collection(
    model: BranchModel, base_state: ThermalState
) -> FloatArray:
    rhs = np.zeros(model.grid.size, dtype=float)
    for iy, conductance in enumerate(model.thermal.cold_conductance_w_per_k):
        rhs[model.grid.index(0, iy)] = conductance
    base_gamma = diags(
        model.base_seebeck_basis.gamma(base_state.temperature_k),
        offsets=0,
        shape=(model.grid.size, model.grid.size),
        format="csr",
    )
    tangent = model.thermal.matrix + model.thomson_matrix @ base_gamma
    collection = np.asarray(spsolve(tangent.T, rhs), dtype=float)
    require(float(np.min(collection)) >= -2.0e-12, "adjoint collection became negative")
    require(float(np.max(collection)) <= 1.0 + 2.0e-12, "adjoint collection exceeded one")
    return collection


def adjoint_qc_derivative(model: BranchModel, base_state: ThermalState) -> dict[str, Any]:
    collection = solve_adjoint_collection(model, base_state)
    r_epsilon = thomson_divergence(model, base_state.temperature_k)
    cold = model.thermal.cold_temperature_k
    direct = float(
        np.sum(
            model.common_basis.mode(cold)
            * cold
            * model.electrical.ix_faces_a[0, :]
        )
    )
    field = float(collection @ r_epsilon)
    return {
        "direct_w_per_epsilon": direct,
        "field_w_per_epsilon": field,
        "total_w_per_epsilon": direct + field,
        "collection": collection,
        "r_epsilon_w": r_epsilon,
    }


def _pair_configs(
    current_a: float,
    family: str = "property_contrast",
    side_multiplier: float = 1.0,
) -> tuple[BranchConfig, BranchConfig]:
    require(side_multiplier > 0.0, "side multiplier must be positive")
    if family == "near_matched":
        p = BranchConfig(
            name="p",
            signed_current_a=current_a,
            seebeck_base_v_per_k=215.0e-6,
            seebeck_base_slope_v_per_k2=0.45e-6,
            rho0_ohm_m=1.30e-5,
            k0_w_per_mk=1.18,
            pattern_phase=0.20,
            side_h_bottom_w_per_m2k=100.0,
            side_h_top_w_per_m2k=480.0,
            side_ambient_bottom_k=316.0,
            side_ambient_top_k=294.0,
        )
        n = BranchConfig(
            name="n",
            signed_current_a=-current_a,
            seebeck_base_v_per_k=-185.0e-6,
            seebeck_base_slope_v_per_k2=-0.32e-6,
            rho0_ohm_m=1.45e-5,
            k0_w_per_mk=1.08,
            pattern_phase=0.62,
            side_h_bottom_w_per_m2k=115.0,
            side_h_top_w_per_m2k=450.0,
            side_ambient_bottom_k=315.0,
            side_ambient_top_k=295.0,
        )
    elif family == "property_contrast":
        p = BranchConfig(
            name="p",
            signed_current_a=current_a,
            seebeck_base_v_per_k=215.0e-6,
            seebeck_base_slope_v_per_k2=0.55e-6,
            rho0_ohm_m=1.18e-5,
            k0_w_per_mk=1.42,
            pattern_phase=0.15,
            side_h_bottom_w_per_m2k=95.0,
            side_h_top_w_per_m2k=570.0,
            side_ambient_bottom_k=317.0,
            side_ambient_top_k=293.0,
        )
        n = BranchConfig(
            name="n",
            signed_current_a=-current_a,
            seebeck_base_v_per_k=-185.0e-6,
            seebeck_base_slope_v_per_k2=-0.38e-6,
            rho0_ohm_m=1.88e-5,
            k0_w_per_mk=0.92,
            pattern_phase=1.05,
            side_h_bottom_w_per_m2k=145.0,
            side_h_top_w_per_m2k=410.0,
            side_ambient_bottom_k=314.0,
            side_ambient_top_k=296.0,
        )
    elif family == "geometry_property_contrast":
        p = BranchConfig(
            name="p",
            signed_current_a=current_a,
            seebeck_base_v_per_k=215.0e-6,
            seebeck_base_slope_v_per_k2=0.65e-6,
            length_m=1.48e-3,
            width_m=0.62e-3,
            depth_m=0.72e-3,
            rho0_ohm_m=1.02e-5,
            k0_w_per_mk=1.62,
            pattern_phase=0.35,
            side_h_bottom_w_per_m2k=75.0,
            side_h_top_w_per_m2k=650.0,
            side_ambient_bottom_k=318.0,
            side_ambient_top_k=292.0,
        )
        n = BranchConfig(
            name="n",
            signed_current_a=-current_a,
            seebeck_base_v_per_k=-185.0e-6,
            seebeck_base_slope_v_per_k2=-0.46e-6,
            length_m=0.84e-3,
            width_m=0.98e-3,
            depth_m=0.66e-3,
            rho0_ohm_m=2.25e-5,
            k0_w_per_mk=0.71,
            pattern_phase=1.35,
            side_h_bottom_w_per_m2k=190.0,
            side_h_top_w_per_m2k=340.0,
            side_ambient_bottom_k=312.0,
            side_ambient_top_k=298.0,
        )
    else:
        raise ValidationError(f"unknown mismatch family: {family}")

    def scaled(config: BranchConfig) -> BranchConfig:
        return replace(
            config,
            side_h_bottom_w_per_m2k=(
                config.side_h_bottom_w_per_m2k * side_multiplier
            ),
            side_h_top_w_per_m2k=config.side_h_top_w_per_m2k * side_multiplier,
        )

    return scaled(p), scaled(n)


def run_pair_kernel_case(
    nx: int,
    ny: int,
    current_a: float = 1.10,
    finite_difference_step: float = 1.0e-3,
    family: str = "property_contrast",
    side_multiplier: float = 1.0,
    common_basis: CommonModeBasis | None = None,
    cold_temperature_k: float = 300.0,
    hot_temperature_k: float = 350.0,
) -> dict[str, Any]:
    if common_basis is None:
        common_basis = common_mode_bases()[1]
    p_config, n_config = _pair_configs(current_a, family, side_multiplier)
    models = [
        build_branch_model(
            p_config,
            nx,
            ny,
            cold_temperature_k,
            hot_temperature_k,
            common_basis,
        ),
        build_branch_model(
            n_config,
            nx,
            ny,
            cold_temperature_k,
            hot_temperature_k,
            common_basis,
        ),
    ]
    base_states = [solve_thermal(model, 0.0) for model in models]
    adjoints = [
        adjoint_qc_derivative(model, state)
        for model, state in zip(models, base_states)
    ]
    pair_adjoint = float(sum(value["total_w_per_epsilon"] for value in adjoints))

    pair_qc: dict[str, float] = {}
    solve_records: dict[str, Any] = {}
    perturbed_energy_residuals: list[float] = []
    for label, epsilon in (("minus", -finite_difference_step), ("plus", finite_difference_step)):
        values = []
        records = []
        for model, base_state in zip(models, base_states):
            state = solve_thermal(model, epsilon, base_state.temperature_k)
            metrics = branch_metrics(model, state)
            values.append(metrics["Qc_w"])
            perturbed_energy_residuals.append(abs(metrics["energy_residual_w"]))
            records.append(
                {
                    "branch": model.config.name,
                    "iterations": state.iterations,
                    "relative_residual": state.relative_residual,
                    "energy_residual_w": metrics["energy_residual_w"],
                }
            )
        pair_qc[label] = float(sum(values))
        solve_records[label] = records
    finite_difference = (
        pair_qc["plus"] - pair_qc["minus"]
    ) / (2.0 * finite_difference_step)
    absolute_error = abs(finite_difference - pair_adjoint)
    relative_error = absolute_error / max(abs(pair_adjoint), 1.0e-30)

    base_metrics = [
        branch_metrics(model, state) for model, state in zip(models, base_states)
    ]
    max_energy = max(abs(value["energy_residual_w"]) for value in base_metrics)

    return {
        "mismatch_family": family,
        "side_multiplier": side_multiplier,
        "common_mode_basis": {
            "name": common_basis.name,
            "kind": common_basis.kind,
            "anchor_k": common_basis.anchor_k,
            "amplitude": common_basis.amplitude,
            "centre_k": common_basis.centre_k,
            "width_k": common_basis.width_k,
        },
        "endpoint_temperatures_k": {
            "cold": cold_temperature_k,
            "hot": hot_temperature_k,
        },
        "current_a": current_a,
        "grid": {"nx": nx, "ny": ny, "cells_per_branch": nx * ny},
        "finite_difference_step": finite_difference_step,
        "adjoint_pair_dQc_w_per_epsilon": pair_adjoint,
        "finite_difference_pair_dQc_w_per_epsilon": finite_difference,
        "absolute_difference_w_per_epsilon": absolute_error,
        "relative_difference": relative_error,
        "branch_adjoint": {
            model.config.name: {
                "direct_w_per_epsilon": float(adjoint["direct_w_per_epsilon"]),
                "field_w_per_epsilon": float(adjoint["field_w_per_epsilon"]),
                "total_w_per_epsilon": float(adjoint["total_w_per_epsilon"]),
            }
            for model, adjoint in zip(models, adjoints)
        },
        "pair_qc_w": pair_qc,
        "nonlinear_solve_records": solve_records,
        "maximum_base_energy_residual_w": max_energy,
        "maximum_energy_residual_w": max(
            [max_energy, *perturbed_energy_residuals]
        ),
        "models": models,
        "base_states": base_states,
        "collections": [value["collection"] for value in adjoints],
    }


def _pair_totals(metrics: list[dict[str, float]], current_a: float) -> dict[str, float]:
    totals = {
        key: float(sum(branch[key] for branch in metrics))
        for key in ("Qc_w", "Qh_w", "Qside_w", "P_electrical_w")
    }
    totals["V_power_conjugate_v"] = totals["P_electrical_w"] / current_a
    totals["energy_residual_w"] = (
        totals["Qh_w"]
        - totals["Qc_w"]
        + totals["Qside_w"]
        - totals["P_electrical_w"]
    )
    return totals


def run_constant_shift_case(
    nx: int,
    ny: int,
    *,
    cold_p_k: float,
    cold_n_k: float,
    hot_p_k: float,
    hot_n_k: float,
    current_a: float = 1.10,
    shift_v_per_k: float = 80.0e-6,
) -> dict[str, Any]:
    p_config, n_config = _pair_configs(current_a)
    models = [
        build_branch_model(p_config, nx, ny, cold_p_k, hot_p_k),
        build_branch_model(n_config, nx, ny, cold_n_k, hot_n_k),
    ]
    states = [solve_thermal(model, 0.0) for model in models]
    base_metrics = [branch_metrics(model, state, 0.0) for model, state in zip(models, states)]
    shifted_metrics = [
        branch_metrics(model, state, shift_v_per_k) for model, state in zip(models, states)
    ]
    base_total = _pair_totals(base_metrics, current_a)
    shifted_total = _pair_totals(shifted_metrics, current_a)
    increment = {key: shifted_total[key] - base_total[key] for key in base_total}

    predicted_qc = shift_v_per_k * current_a * (cold_p_k - cold_n_k)
    predicted_qh = shift_v_per_k * current_a * (hot_p_k - hot_n_k)
    predicted_v = shift_v_per_k * (
        (hot_p_k - hot_n_k) - (cold_p_k - cold_n_k)
    )
    predicted_power = current_a * predicted_v
    predicted = {
        "delta_Qc_w": predicted_qc,
        "delta_Qh_w": predicted_qh,
        "delta_Qside_w": 0.0,
        "delta_V_v": predicted_v,
        "delta_P_w": predicted_power,
    }
    errors = {
        "Qc_w": increment["Qc_w"] - predicted_qc,
        "Qh_w": increment["Qh_w"] - predicted_qh,
        "Qside_w": increment["Qside_w"],
        "V_v": increment["V_power_conjugate_v"] - predicted_v,
        "P_w": increment["P_electrical_w"] - predicted_power,
        "incremental_energy_interlock_w": (
            increment["Qh_w"]
            - increment["Qc_w"]
            + increment["Qside_w"]
            - current_a * increment["V_power_conjugate_v"]
        ),
    }
    return {
        "temperatures_k": {
            "cold_p": cold_p_k,
            "cold_n": cold_n_k,
            "hot_p": hot_p_k,
            "hot_n": hot_n_k,
        },
        "current_a": current_a,
        "constant_shift_v_per_k": shift_v_per_k,
        "base_pair": base_total,
        "shifted_pair": shifted_total,
        "increment": increment,
        "prediction": predicted,
        "errors": errors,
        "maximum_branch_energy_residual_w": max(
            abs(value["energy_residual_w"])
            for value in base_metrics + shifted_metrics
        ),
    }


def non_1d_diagnostics(model: BranchModel, state: ThermalState) -> dict[str, float]:
    grid = model.grid
    temperature = state.temperature_k.reshape(grid.nx, grid.ny)
    ix_density = 0.5 * (
        model.electrical.ix_faces_a[:-1, :] + model.electrical.ix_faces_a[1:, :]
    ) / (grid.dy * grid.depth_m)
    iy_density = 0.5 * (
        model.electrical.iy_faces_a[:, :-1] + model.electrical.iy_faces_a[:, 1:]
    ) / (grid.dx * grid.depth_m)
    lateral_span = float(np.max(np.ptp(temperature, axis=1)))
    transverse_ratio = float(np.max(np.abs(iy_density)) / np.max(np.abs(ix_density)))
    return {
        "maximum_lateral_temperature_span_k": lateral_span,
        "maximum_transverse_to_axial_current_density_ratio": transverse_ratio,
        "rho_max_to_min_ratio": float(np.max(model.rho_ohm_m) / np.min(model.rho_ohm_m)),
        "k_max_to_min_ratio": float(np.max(model.k_w_per_mk) / np.min(model.k_w_per_mk)),
        "electrical_divergence_max_a_per_cell": model.electrical.divergence_max_a,
        "terminal_current_mismatch_a": model.electrical.terminal_current_mismatch_a,
    }


def _clean_kernel_result(case: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if key not in {"models", "base_states", "collections"}}


def _config_summary(config: BranchConfig) -> dict[str, float]:
    return {
        "length_m": config.length_m,
        "width_m": config.width_m,
        "depth_m": config.depth_m,
        "seebeck_base_slope_v_per_k2": config.seebeck_base_slope_v_per_k2,
        "rho0_ohm_m": config.rho0_ohm_m,
        "k0_w_per_mk": config.k0_w_per_mk,
        "side_h_bottom_w_per_m2k": config.side_h_bottom_w_per_m2k,
        "side_h_top_w_per_m2k": config.side_h_top_w_per_m2k,
    }


def mismatch_family_definitions() -> dict[str, Any]:
    definitions: dict[str, Any] = {}
    for family in ("near_matched", "property_contrast", "geometry_property_contrast"):
        p_config, n_config = _pair_configs(1.10, family, 1.0)
        definitions[family] = {
            "p": _config_summary(p_config),
            "n": _config_summary(n_config),
        }
    return definitions


def run_cross_family_matrix() -> list[dict[str, Any]]:
    """Stress three mismatch families, three Gamma bases, and three side couplings."""

    entries: list[dict[str, Any]] = []
    for family in ("near_matched", "property_contrast", "geometry_property_contrast"):
        for basis in common_mode_bases():
            for side_multiplier in (0.25, 1.0, 4.0):
                case = run_pair_kernel_case(
                    24,
                    16,
                    family=family,
                    side_multiplier=side_multiplier,
                    common_basis=basis,
                    finite_difference_step=1.0e-3,
                )
                entries.append(_clean_kernel_result(case))
    return entries


def run_nonmonotone_oriented_measure_case() -> tuple[dict[str, Any], dict[str, Any]]:
    """Force an internal Joule-heating maximum so no single-valued x(T) exists."""

    case = run_pair_kernel_case(
        32,
        20,
        current_a=3.0,
        finite_difference_step=7.5e-4,
        family="geometry_property_contrast",
        side_multiplier=0.25,
        common_basis=common_mode_bases()[1],
        cold_temperature_k=300.0,
        hot_temperature_k=300.0,
    )
    branches: dict[str, Any] = {}
    for model, state in zip(case["models"], case["base_states"]):
        temperature = state.temperature_k.reshape(model.grid.nx, model.grid.ny)
        profile = np.mean(temperature, axis=1)
        increments = np.diff(profile)
        signs = np.sign(increments[np.abs(increments) > 1.0e-8])
        sign_changes = int(np.count_nonzero(signs[1:] * signs[:-1] < 0.0))
        peak_index = int(np.argmax(profile))
        branches[model.config.name] = {
            "minimum_temperature_k": float(np.min(temperature)),
            "maximum_temperature_k": float(np.max(temperature)),
            "mean_profile_peak_temperature_k": float(profile[peak_index]),
            "mean_profile_peak_cell_index": peak_index,
            "mean_profile_left_cell_temperature_k": float(profile[0]),
            "mean_profile_right_cell_temperature_k": float(profile[-1]),
            "axial_gradient_sign_changes": sign_changes,
            "peak_is_interior": 0 < peak_index < model.grid.nx - 1,
            "has_positive_and_negative_axial_gradient": bool(
                np.any(increments > 0.0) and np.any(increments < 0.0)
            ),
        }
    clean = _clean_kernel_result(case)
    clean["temperature_topology"] = branches
    return clean, case


def build_results() -> tuple[dict[str, Any], dict[str, Any]]:
    grid_shapes = [(16, 10), (32, 20), (64, 40)]
    grid_convergence_by_basis: dict[str, list[dict[str, Any]]] = {}
    raw_grid_cases: dict[str, list[dict[str, Any]]] = {}
    for basis in common_mode_bases():
        cases = [
            run_pair_kernel_case(nx, ny, common_basis=basis)
            for nx, ny in grid_shapes
        ]
        raw_grid_cases[basis.name] = cases
        grid_convergence_by_basis[basis.name] = [
            _clean_kernel_result(case) for case in cases
        ]
    kernel_cases = raw_grid_cases["linear_Gamma"]
    finest = kernel_cases[-1]
    p_model = finest["models"][0]
    p_state = finest["base_states"][0]

    gauge_null = run_constant_shift_case(
        64,
        40,
        cold_p_k=300.0,
        cold_n_k=300.0,
        hot_p_k=350.0,
        hot_n_k=350.0,
    )
    split_pad = run_constant_shift_case(
        64,
        40,
        cold_p_k=302.0,
        cold_n_k=298.0,
        hot_p_k=350.0,
        hot_n_k=350.0,
    )
    cross_family_matrix = run_cross_family_matrix()
    nonmonotone, nonmonotone_raw = run_nonmonotone_oriented_measure_case()

    grid_changes: dict[str, float] = {}
    for basis_name, cases in raw_grid_cases.items():
        finest_adjoint = cases[-1]["adjoint_pair_dQc_w_per_epsilon"]
        medium_adjoint = cases[-2]["adjoint_pair_dQc_w_per_epsilon"]
        grid_changes[basis_name] = abs(finest_adjoint - medium_adjoint) / max(
            abs(finest_adjoint), 1.0e-30
        )
    grid_change = grid_changes["linear_Gamma"]
    non_1d = non_1d_diagnostics(p_model, p_state)
    null_scale = max(
        abs(gauge_null["base_pair"]["Qc_w"]),
        abs(gauge_null["base_pair"]["Qh_w"]),
        1.0e-30,
    )
    null_relative = max(
        abs(gauge_null["increment"]["Qc_w"]),
        abs(gauge_null["increment"]["Qh_w"]),
        abs(gauge_null["increment"]["P_electrical_w"]),
    ) / null_scale
    topology_error = max(abs(value) for value in split_pad["errors"].values())
    matrix_worst_relative = max(
        entry["relative_difference"] for entry in cross_family_matrix
    )
    matrix_worst_energy = max(
        entry["maximum_energy_residual_w"] for entry in cross_family_matrix
    )
    nonmonotone_topology = nonmonotone["temperature_topology"]

    checks = {
        "genuine_2d_temperature": non_1d["maximum_lateral_temperature_span_k"] > 0.25,
        "genuine_2d_current": non_1d["maximum_transverse_to_axial_current_density_ratio"] > 0.01,
        "current_conservation": non_1d["electrical_divergence_max_a_per_cell"] < 2.0e-11,
        "base_energy_conservation": max(
            case["maximum_energy_residual_w"]
            for cases in raw_grid_cases.values()
            for case in cases
        ) < 2.0e-10,
        "adjoint_finite_difference": max(
            case["relative_difference"]
            for cases in raw_grid_cases.values()
            for case in cases
        ) < 2.0e-6,
        "grid_convergence_all_gamma_bases": max(grid_changes.values()) < 0.015,
        "cross_family_matrix_adjoint_fd": matrix_worst_relative < 2.0e-6,
        "cross_family_matrix_energy": matrix_worst_energy < 2.0e-10,
        "nonmonotone_oriented_measure": (
            nonmonotone["relative_difference"] < 2.0e-6
            and all(
                branch["peak_is_interior"]
                and branch["has_positive_and_negative_axial_gradient"]
                and branch["axial_gradient_sign_changes"] >= 1
                for branch in nonmonotone_topology.values()
            )
        ),
        "shared_isothermal_null": null_relative < 2.0e-12,
        "split_pad_direct_law": topology_error < 2.0e-11,
        "split_pad_energy_voltage_interlock": abs(
            split_pad["errors"]["incremental_energy_interlock_w"]
        ) < 2.0e-11,
    }
    require(all(checks.values()), f"one or more 2-D validation checks failed: {checks}")

    figure_payload = {
        "p_model": p_model,
        "p_state": p_state,
        "p_collection": finest["collections"][0],
        "kernel_cases": kernel_cases,
        "grid_cases_by_basis": raw_grid_cases,
        "cross_family_matrix": cross_family_matrix,
        "nonmonotone": nonmonotone,
        "gauge_null": gauge_null,
        "split_pad": split_pad,
    }
    results = {
        "schema_version": SCHEMA,
        "analysis_id": ANALYSIS_ID,
        "timestamp_utc": FIXED_TIMESTAMP.isoformat(),
        "scientific_question": (
            "Do the common-mode null, first-order Gamma transfer kernel, and split-pad "
            "topology law survive a conservative heterogeneous two-dimensional current/heat model?"
        ),
        "answer": (
            "Yes within the stated 2-D thermal-advection/Joule model: the shared-isothermal "
            "constant shift remains null, the discrete 2-D adjoint predicts the finite-difference "
            "Gamma response with mesh convergence, and the split-pad heat/voltage law closes "
            "together with side heat and electrical power."
        ),
        "model_definition": {
            "geometry": "independent rectangular 2-D finite volumes, one domain per p/n branch",
            "electrical": (
                "heterogeneous temperature-independent sigma; electrochemical potential solve; "
                "insulated sidewalls; total signed current fixed by linear scaling"
            ),
            "thermal": (
                "heterogeneous temperature-independent k; exact resistor-edge Joule power; "
                "Dirichlet end temperatures; asymmetric Robin side losses; branch-specific "
                "linear base Seebeck/Thomson transport; conservative perturbation Thomson flux"
            ),
            "common_mode": (
                "S_i -> S_i + C + epsilon*m(T), with constant, linear, and localized "
                "Gamma(T)=T*dm/dT bases shared by p/n"
            ),
            "independence": (
                "does not import or call scripts/tec_1d_solver or any existing 1-D production solve"
            ),
            "not_included": [
                "temperature-dependent electrical resistivity or thermal conductivity",
                "fully coupled 3-D current/heat spreading",
                "finite pad/contact thermal conductance; end faces are Dirichlet",
                "contact mechanics or contact-property calibration",
                "radiation",
                "experimental or COMSOL validation",
            ],
        },
        "conservative_identities": {
            "thermal_equation": "div(k grad T)+rho|J|^2-tau J.grad(T)=0",
            "thomson_flux": (
                "tau=epsilon*Gamma(T); choose G'(T)=Gamma(T), then "
                "tau J.grad(T)=div[J*epsilon*G(T)] because div(J)=0"
            ),
            "branch_energy": "Qh-Qc+Qside=P_electrical",
            "two_dimensional_first_variation": (
                "dQc/depsilon = integral_cold[Tc*m(Tc)*Jx]dy + "
                "integral_domain[psi*Gamma(T)*J.grad(T)]dA; the discrete adjoint uses "
                "the transpose of the full base-Thomson tangent operator"
            ),
            "oriented_measure_note": (
                "the domain integral is evaluated directly and therefore does not require a "
                "single-valued inverse x(T); the equal-end-temperature Joule case supplies an "
                "interior maximum and opposite axial temperature-gradient orientations"
            ),
            "pair_constant_shift": {
                "delta_Qc": "C*I*(Tcp-Tcn) for isothermal branch pads",
                "delta_Qh": "C*I*(Thp-Thn) for isothermal branch pads",
                "delta_V": "C*((Thp-Thn)-(Tcp-Tcn))",
                "interlock": "delta_Qh-delta_Qc+delta_Qside=I*delta_V",
            },
        },
        "grid_convergence": [_clean_kernel_result(case) for case in kernel_cases],
        "grid_convergence_by_gamma_basis": grid_convergence_by_basis,
        "medium_to_fine_adjoint_relative_change": grid_change,
        "medium_to_fine_adjoint_relative_change_by_gamma_basis": grid_changes,
        "cross_family_matrix": {
            "design": {
                "mismatch_families": [
                    "near_matched",
                    "property_contrast",
                    "geometry_property_contrast",
                ],
                "mismatch_family_definitions_at_side_multiplier_1": (
                    mismatch_family_definitions()
                ),
                "gamma_bases": [basis.name for basis in common_mode_bases()],
                "gamma_basis_definitions": {
                    "constant_Gamma": "Gamma(T)=gamma0; m(T)=gamma0*ln(T/T0)",
                    "linear_Gamma": "Gamma(T)=b*T; m(T)=b*(T-T0)",
                    "localized_Gamma": (
                        "Gamma(T)=b*T*exp[-(T-Tstar)^2/(2*w^2)]; m(T) is its "
                        "anchored Gaussian integral"
                    ),
                },
                "sidewall_coupling_multipliers": [0.25, 1.0, 4.0],
                "total_cases": len(cross_family_matrix),
            },
            "worst_adjoint_finite_difference_relative_error": matrix_worst_relative,
            "worst_energy_residual_w": matrix_worst_energy,
            "cases": cross_family_matrix,
        },
        "nonmonotone_oriented_measure_case": nonmonotone,
        "non_1d_diagnostics_finest_p_leg": non_1d,
        "shared_isothermal_constant_shift_null": gauge_null,
        "split_pad_topology_breaking": split_pad,
        "validation_checks": checks,
        "all_checks_passed": True,
        "scope": {
            "supports": [
                "the exact constant-shift null is not an artifact of a 1-D temperature profile",
                "the first-order common-mode transfer mechanism extends to a conservative 2-D adjoint",
                "the split-pad direct term and heat-voltage-energy interlock survive 2-D heterogeneity and side loss",
            ],
            "does_not_support": [
                "as-built PbSe/Cr device validation",
                "quantitative prediction for any specific experimental device",
                "equivalence to commercial multiphysics software",
                "neglect of all 3-D/contact effects",
            ],
            "parameter_coverage": (
                "Finite thermal-contact-strength sweeps are evaluated in the one-dimensional model. "
                "This independent two-dimensional calculation spans three sidewall couplings "
                "while retaining Dirichlet pad temperatures."
            ),
            "supported_interpretation": (
                "The calculation provides numerical evidence for the analytic kernel "
                "and topology result."
            ),
        },
        "figure_metadata": {
            "core_conclusion": (
                "A conservative heterogeneous 2-D model preserves the exact null and split-pad law "
                "while independently converging to the adjoint Gamma response."
            ),
            "evidence_chain": {
                "a": "non-1-D temperature and current field",
                "b": "2-D adjoint collection field",
                "c": "three-Gamma-basis adjoint/finite-difference grid convergence",
                "d": "27-case matrix, null, topology, and energy closure errors",
            },
            "layout": "quantitative grid with an emphasized field map",
            "backend": "Python/matplotlib only",
            "export": "183-mm double-column SVG/PDF plus 600-dpi TIFF and 300-dpi PNG preview",
            "limitations": [
                "deterministic numerical evidence has no replicate statistics",
                "model limitations accompany the supported interpretation",
                "log-scale closure errors must not be interpreted as experimental accuracy",
            ],
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": mpl.__version__,
        },
    }
    return results, figure_payload


def make_figure(payload: dict[str, Any], prefix: Path) -> list[Path]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    mpl.rcParams["svg.hashsalt"] = "independent-2d-common-mode-v1"

    model: BranchModel = payload["p_model"]
    state: ThermalState = payload["p_state"]
    collection = payload["p_collection"].reshape(model.grid.nx, model.grid.ny)
    temperature = state.temperature_k.reshape(model.grid.nx, model.grid.ny)
    x_mm = model.grid.x_centres * 1.0e3
    y_mm = model.grid.y_centres * 1.0e3
    jx = 0.5 * (
        model.electrical.ix_faces_a[:-1, :] + model.electrical.ix_faces_a[1:, :]
    ) / (model.grid.dy * model.grid.depth_m)
    jy = 0.5 * (
        model.electrical.iy_faces_a[:, :-1] + model.electrical.iy_faces_a[:, 1:]
    ) / (model.grid.dx * model.grid.depth_m)
    jnorm = np.hypot(jx, jy)

    fig = plt.figure(figsize=(7.20, 4.75), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1.0), height_ratios=(1.0, 0.88))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[0, 1])
    ax_d = fig.add_subplot(grid[1, 1])

    map_a = ax_a.pcolormesh(x_mm, y_mm, temperature.T, shading="nearest", cmap="magma")
    skip_x = max(1, model.grid.nx // 16)
    skip_y = max(1, model.grid.ny // 10)
    ax_a.quiver(
        x_mm[::skip_x],
        y_mm[::skip_y],
        (jx / jnorm)[::skip_x, ::skip_y].T,
        (jy / jnorm)[::skip_x, ::skip_y].T,
        color="white",
        alpha=0.78,
        width=0.0040,
        scale=19,
        headwidth=3.0,
    )
    color_a = fig.colorbar(map_a, ax=ax_a, pad=0.02, fraction=0.046)
    color_a.set_label("Temperature (K)")
    ax_a.set(xlabel="Axial position (mm)", ylabel="Transverse position (mm)")
    ax_a.set_title("Heterogeneous 2-D thermal/current field", loc="left")

    map_b = ax_b.pcolormesh(x_mm, y_mm, collection.T, shading="nearest", cmap="cividis", vmin=0, vmax=1)
    color_b = fig.colorbar(map_b, ax=ax_b, pad=0.02, fraction=0.046)
    color_b.set_label(r"Adjoint collection $\psi$")
    ax_b.set(xlabel="Axial position (mm)", ylabel="Transverse position (mm)")
    ax_b.set_title("Cold-port collection field", loc="left")

    kernel_cases = payload["kernel_cases"]
    basis_styles = {
        "constant_Gamma": ("#245A73", "o", r"constant $\Gamma$"),
        "linear_Gamma": ("#D17A45", "s", r"linear $\Gamma$"),
        "localized_Gamma": ("#477D63", "^", r"localized $\Gamma$"),
    }
    for basis_name, cases in payload["grid_cases_by_basis"].items():
        color, marker, label = basis_styles[basis_name]
        cells = np.array(
            [case["grid"]["cells_per_branch"] for case in cases], dtype=float
        )
        adjoint_mw = 1.0e3 * np.array(
            [case["adjoint_pair_dQc_w_per_epsilon"] for case in cases]
        )
        fd_mw = 1.0e3 * np.array(
            [case["finite_difference_pair_dQc_w_per_epsilon"] for case in cases]
        )
        ax_c.plot(
            cells,
            adjoint_mw,
            marker=marker,
            color=color,
            lw=1.25,
            ms=4.0,
            label=label,
        )
        ax_c.plot(
            cells,
            fd_mw,
            linestyle="none",
            marker=marker,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            ms=2.7,
        )
    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks(cells)
    ax_c.set_xticklabels([f"{int(value)}" for value in cells])
    ax_c.set_xlabel("Cells per branch")
    ax_c.set_ylabel(r"$dQ_c/d\epsilon$ (mW)")
    ax_c.set_title("Independent kernel convergence", loc="left")
    ax_c.grid(axis="both", color="#D7DCE0", lw=0.5, alpha=0.65)
    ax_c.legend(
        title="lines: adjoint | open: FD",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.73),
        ncol=3,
        columnspacing=0.9,
        handlelength=1.7,
        handletextpad=0.4,
        title_fontsize=5.8,
    )

    gauge = payload["gauge_null"]
    split = payload["split_pad"]
    matrix_cases = payload["cross_family_matrix"]
    error_values = np.array(
        [
            max(
                abs(gauge["increment"]["Qc_w"]),
                abs(gauge["increment"]["Qh_w"]),
                abs(gauge["increment"]["P_electrical_w"]),
            ),
            max(abs(value) for value in split["errors"].values()),
            max(case["maximum_energy_residual_w"] for case in matrix_cases),
            max(case["absolute_difference_w_per_epsilon"] for case in matrix_cases),
        ],
        dtype=float,
    )
    error_values = np.maximum(error_values, 1.0e-18)
    labels = ["shared-pad\nnull", "split-pad\nlaw", "27-case\nenergy", "27-case\nadjoint–FD"]
    colors = ["#6B7883", "#477D63", "#477D63", "#245A73"]
    positions = np.arange(len(labels))
    ax_d.bar(positions, error_values, color=colors, width=0.66)
    ax_d.set_yscale("log")
    ax_d.set_xticks(positions, labels)
    ax_d.set_ylabel("Absolute closure error (W)")
    ax_d.set_title("Null and topology checks close", loc="left")
    ax_d.grid(axis="y", color="#D7DCE0", lw=0.5, alpha=0.65)
    signal = abs(split["prediction"]["delta_Qc_w"])
    ax_d.text(
        0.02,
        0.96,
        (
            f"split signal = {1e3*signal:.3f} mW\n"
            f"nonmonotone kernel rel. err. = "
            f"{payload['nonmonotone']['relative_difference']:.1e}"
        ),
        transform=ax_d.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="#334047",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        axis.text(
            -0.14,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
        )

    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = [prefix.with_suffix(suffix) for suffix in (".svg", ".pdf", ".tiff", ".png")]
    fig.savefig(
        paths[0],
        metadata={"Date": FIXED_TIMESTAMP.isoformat(), "Creator": None},
    )
    fig.savefig(
        paths[1],
        metadata={"CreationDate": FIXED_TIMESTAMP, "ModDate": FIXED_TIMESTAMP},
    )
    fig.savefig(
        paths[2],
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(paths[3], dpi=300, metadata={"Date": FIXED_TIMESTAMP.isoformat()})
    plt.close(fig)

    with Image.open(paths[2]) as image:
        require(image.info.get("dpi", (0, 0))[0] >= 599.0, "TIFF is not 600 dpi")
    return paths


def run(output: Path = DEFAULT_OUTPUT, figure_prefix: Path = DEFAULT_FIGURE_PREFIX) -> dict[str, Any]:
    results, figure_payload = build_results()
    figure_paths = make_figure(figure_payload, figure_prefix)
    results["provenance"] = {
        "script": binding(Path(__file__)),
        "figures": [binding(path) for path in figure_paths],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-prefix", type=Path, default=DEFAULT_FIGURE_PREFIX)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    results = run(arguments.output, arguments.figure_prefix)
    finest = results["grid_convergence"][-1]
    print(
        json.dumps(
            {
                "output": locator(arguments.output),
                "all_checks_passed": results["all_checks_passed"],
                "finest_adjoint_w_per_epsilon": finest["adjoint_pair_dQc_w_per_epsilon"],
                "finest_fd_w_per_epsilon": finest["finite_difference_pair_dQc_w_per_epsilon"],
                "finest_relative_difference": finest["relative_difference"],
                "medium_to_fine_relative_change": results[
                    "medium_to_fine_adjoint_relative_change"
                ],
                "split_pad_delta_Qc_w": results["split_pad_topology_breaking"][
                    "increment"
                ]["Qc_w"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
