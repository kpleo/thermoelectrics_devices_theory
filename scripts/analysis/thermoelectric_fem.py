#!/usr/bin/env python3
"""Independent tetrahedral finite elements for coupled thermoelectric transport.

The unknowns are temperature T and electrochemical potential phi.  This module
assembles the conservation laws directly, without calling the 1D solver:

    j = -sigma(T) [grad(phi) + S(T) grad(T)],
    q = -kappa(T) grad(T) + S(T) T j,
    div(j) = 0,                 div(q) + j.grad(phi) = 0.

The thermal Dirichlet reaction is inward *total* heat.  Peltier heat is already
in q and must not be added again.  An electrical terminal surface is
equipotential; its inward current is imposed as a single constraint, rather
than a prescribed current density. Conducting electrode bodies retain finite
resistance. Material IDs are per tetrahedron; thermal-only regions have no
electrical unknowns except nodes shared with a conducting region.

The fixed-end examples are genuine 3D P1 finite-element discretizations of a
laterally uniform physical problem.  They test dimensional reduction, not
contact spreading or complete encapsulated-device performance.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import bmat, coo_matrix
from scipy.sparse.linalg import spsolve
from skfem import Basis, BilinearForm, ElementTetP1, LinearForm, MeshTet, asm
from skfem.helpers import dot, grad

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class PropertyLaw:
    value: Callable[[np.ndarray], np.ndarray]
    slope: Callable[[np.ndarray], np.ndarray]
    domain: tuple[float, float] = (1.0, 2000.0)

    def evaluate(self, temperature: np.ndarray) -> np.ndarray:
        t = np.asarray(temperature, dtype=float)
        if np.min(t) < self.domain[0] - 1e-9 or np.max(t) > self.domain[1] + 1e-9:
            raise ValueError(f"temperature outside property domain {self.domain}")
        return np.broadcast_to(self.value(t), t.shape)

    def derivative(self, temperature: np.ndarray) -> np.ndarray:
        self.evaluate(temperature)
        return np.broadcast_to(self.slope(np.asarray(temperature, dtype=float)), np.shape(temperature))


def constant(value: float) -> PropertyLaw:
    return PropertyLaw(lambda t: np.full_like(t, value, dtype=float), lambda t: np.zeros_like(t, dtype=float))


def tabulated(temperature: np.ndarray, value: np.ndarray) -> PropertyLaw:
    curve = PchipInterpolator(temperature, value, extrapolate=False)
    return PropertyLaw(curve, curve.derivative(), (float(temperature[0]), float(temperature[-1])))


@dataclass(frozen=True)
class Material:
    name: str
    kappa: PropertyLaw
    seebeck: PropertyLaw | None = None
    resistivity: PropertyLaw | None = None

    @property
    def conducting(self) -> bool:
        return self.resistivity is not None


@dataclass(frozen=True)
class Electrode:
    nodes: np.ndarray
    inward_current_a: float = 0.0


@dataclass
class FEMResult:
    temperature_k: np.ndarray
    potential_v: np.ndarray
    thermal_reaction_w: np.ndarray
    electrical_reaction_a: np.ndarray
    summary: dict[str, Any]


@LinearForm
def _electrical_residual(v, w):
    return -dot(w.j, grad(v))


@LinearForm
def _thermal_residual(v, w):
    return w.k * dot(w.T.grad, grad(v)) - w.S * w.T * dot(w.j, grad(v)) + dot(w.j, w.P.grad) * v


@BilinearForm
def _ee(u, v, w):
    return w.sigma * dot(grad(u), grad(v))


@BilinearForm
def _et(u, v, w):
    return dot(w.a * u + w.sigma * w.S * grad(u), grad(v))


@BilinearForm
def _te(u, v, w):
    return w.S * w.T * w.sigma * dot(grad(u), grad(v)) + dot(w.j - w.sigma * w.P.grad, grad(u)) * v


@BilinearForm
def _tt(u, v, w):
    dj = -w.a * u - w.sigma * w.S * grad(u)
    return (
        w.k * dot(grad(u), grad(v))
        + w.dk * u * dot(w.T.grad, grad(v))
        - (w.dS * w.T + w.S) * u * dot(w.j, grad(v))
        - w.S * w.T * dot(dj, grad(v))
        + dot(dj, w.P.grad) * v
    )


class CoupledFEM:
    """Conforming multi-material mesh with ideal interfaces and electrodes.

    Parameters
    ----------
    thermal_boundaries:
        ``{name: (node_indices, fixed_temperature_K)}``; nodes must be disjoint.
    electrodes:
        ``{name: Electrode(nodes, inward_current_A)}``.  The ground electrode's
        current is a reaction, and its potential is zero.  All other currents
        are prescribed; its declared ground current is used to check balance.
    initial_temperature:
        Nodal starting field.  Use physical reservoir temperatures and keep
        iterates within the tabulated-material range.
    """

    def __init__(self, mesh: MeshTet, materials: list[Material], material_ids: np.ndarray,
                 thermal_boundaries: dict[str, tuple[np.ndarray, float]],
                 electrodes: dict[str, Electrode], ground: str,
                 initial_temperature: np.ndarray, *, quadrature_order: int = 4):
        self.mesh = mesh
        self.basis = Basis(mesh, ElementTetP1(), intorder=quadrature_order)
        self.materials = materials
        self.ids = np.asarray(material_ids, dtype=int)
        if self.ids.shape != (mesh.nelements,) or np.any(self.ids < 0) or np.any(self.ids >= len(materials)):
            raise ValueError("one valid material ID is required per tetrahedron")
        self.thermal_boundaries = thermal_boundaries
        self.electrodes = electrodes
        self.ground = ground
        self.initial_temperature = np.array(initial_temperature, dtype=float)
        self.fixed = np.concatenate([np.asarray(nodes, dtype=int) for nodes, _ in thermal_boundaries.values()])
        if len(np.unique(self.fixed)) != len(self.fixed):
            raise ValueError("thermal Dirichlet groups must be disjoint")
        self.free_t = np.setdiff1d(np.arange(self.basis.N), self.fixed)
        conductive_cells = np.flatnonzero([materials[i].conducting for i in self.ids])
        active = np.unique(mesh.t[:, conductive_cells])
        electrode_nodes = np.concatenate([np.asarray(e.nodes, dtype=int) for e in electrodes.values()])
        if len(np.unique(electrode_nodes)) != len(electrode_nodes):
            raise ValueError("electrode groups must be disjoint")
        if not np.all(np.isin(electrode_nodes, active)):
            raise ValueError("electrodes must belong to conducting elements")
        if ground not in electrodes:
            raise ValueError("ground electrode is missing")
        if abs(sum(e.inward_current_a for e in electrodes.values())) > 1e-10:
            raise ValueError("declared electrode currents do not sum to zero")
        interior = np.setdiff1d(active, electrode_nodes)
        rows = list(interior)
        cols = list(range(len(interior)))
        self.terminal_columns: dict[str, int] = {}
        n_e = len(interior)
        for name, electrode in electrodes.items():
            if name != ground:
                self.terminal_columns[name] = n_e
                rows.extend(electrode.nodes)
                cols.extend([n_e] * len(electrode.nodes))
                n_e += 1
        self.P = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.basis.N, n_e)).tocsr()
        self.current_rhs = np.zeros(n_e)
        for name, column in self.terminal_columns.items():
            self.current_rhs[column] = electrodes[name].inward_current_a
        self.current_scale = max(1.0, max(abs(e.inward_current_a) for e in electrodes.values()))
        # 1 W is only a numerical residual scale, not a physical normalization.
        self.heat_scale = 1.0

    def _parameters(self, temperature: np.ndarray, potential: np.ndarray) -> dict[str, Any]:
        t = self.basis.interpolate(temperature)
        p = self.basis.interpolate(potential)
        shape = t.shape
        fields = {name: np.zeros(shape) for name in ("k", "dk", "S", "dS", "sigma", "dsigma")}
        for index, material in enumerate(self.materials):
            mask = self.ids == index
            if not np.any(mask):
                continue
            local_t = np.asarray(t)[mask]
            fields["k"][mask] = material.kappa.evaluate(local_t)
            fields["dk"][mask] = material.kappa.derivative(local_t)
            if material.conducting:
                if material.seebeck is None:
                    raise ValueError("conducting materials require a Seebeck law")
                rho = material.resistivity.evaluate(local_t)
                if np.any(rho <= 0):
                    raise ValueError("resistivity must be positive")
                fields["sigma"][mask] = 1.0 / rho
                fields["dsigma"][mask] = -material.resistivity.derivative(local_t) / rho**2
                fields["S"][mask] = material.seebeck.evaluate(local_t)
                fields["dS"][mask] = material.seebeck.derivative(local_t)
        if np.any(fields["k"] <= 0):
            raise ValueError("thermal conductivity must be positive")
        force = p.grad + fields["S"] * t.grad
        j = -fields["sigma"] * force
        a = fields["dsigma"] * force + fields["sigma"] * fields["dS"] * t.grad
        return {**fields, "T": t, "P": p, "j": j, "a": a}

    def residual(self, temperature: np.ndarray, e: np.ndarray, *, jacobian: bool = False):
        parameters = self._parameters(temperature, self.P @ e)
        re = asm(_electrical_residual, self.basis, **parameters)
        rt = asm(_thermal_residual, self.basis, **parameters)
        reduced_e = self.P.T @ re - self.current_rhs
        residual = np.r_[reduced_e / self.current_scale, rt[self.free_t] / self.heat_scale]
        if not jacobian:
            return residual, re, rt, parameters
        ee = self.P.T @ asm(_ee, self.basis, **parameters) @ self.P
        et = (self.P.T @ asm(_et, self.basis, **parameters))[:, self.free_t]
        te = (asm(_te, self.basis, **parameters) @ self.P)[self.free_t]
        tt = asm(_tt, self.basis, **parameters)[self.free_t][:, self.free_t]
        jac = bmat([[ee / self.current_scale, et / self.current_scale],
                    [te / self.heat_scale, tt / self.heat_scale]], format="csc")
        return residual, jac, re, rt, parameters

    def solve(self, *, tolerance: float = 2e-11, max_iterations: int = 40,
              initial: FEMResult | None = None) -> FEMResult:
        temperature = self.initial_temperature.copy() if initial is None else initial.temperature_k.copy()
        for nodes, value in self.thermal_boundaries.values():
            temperature[nodes] = value
        e = np.zeros(self.P.shape[1])
        if initial is not None:
            multiplicity = np.asarray(self.P.sum(axis=0)).ravel()
            e = np.asarray(self.P.T @ initial.potential_v).ravel() / multiplicity
        history = []
        for iteration in range(max_iterations + 1):
            residual, jac, re, rt, parameters = self.residual(temperature, e, jacobian=True)
            norm = float(np.max(np.abs(residual)))
            history.append(norm)
            if norm < tolerance:
                break
            if iteration == max_iterations:
                raise RuntimeError(f"Newton did not converge: residual={norm:g}")
            step = spsolve(jac, -residual)
            if not np.all(np.isfinite(step)):
                raise RuntimeError("singular or nonfinite Newton step")
            ne = len(e)
            accepted = False
            for reduction in range(14):
                fraction = 0.5**reduction
                trial_t = temperature.copy()
                trial_t[self.free_t] += fraction * step[ne:]
                trial_e = e + fraction * step[:ne]
                try:
                    trial_r = self.residual(trial_t, trial_e)[0]
                    trial_norm = float(np.max(np.abs(trial_r)))
                except ValueError:
                    continue
                if trial_norm < norm or trial_norm < tolerance:
                    temperature, e = trial_t, trial_e
                    accepted = True
                    break
            if not accepted:
                raise RuntimeError(f"Newton line search failed at {norm:g}")
        potential = np.asarray(self.P @ e)
        heat = {name: float(np.sum(rt[nodes])) for name, (nodes, _) in self.thermal_boundaries.items()}
        current = {name: float(np.sum(re[electrode.nodes])) for name, electrode in self.electrodes.items()}
        voltage = {name: float(np.mean(potential[electrode.nodes])) for name, electrode in self.electrodes.items()}
        electrical_power = float(sum(current[name] * voltage[name] for name in current))
        volume_power = float(-np.sum(dot(parameters["j"], parameters["P"].grad) * self.basis.dx))
        global_energy = float(sum(heat.values()) + electrical_power)
        temperature_range = [float(np.min(temperature)), float(np.max(temperature))]
        summary = {
            "nodes": int(self.basis.N), "tetrahedra": int(self.mesh.nelements),
            "electrical_unknowns": int(self.P.shape[1]), "thermal_unknowns": int(len(self.free_t)),
            "iterations": iteration, "residual_history": history,
            "inward_heat_w": heat, "inward_current_a": current,
            "electrode_potential_v": voltage,
            "electrical_power_w": electrical_power, "volume_electrical_power_w": volume_power,
            "energy_residual_w": global_energy,
            "electrical_power_identity_residual_w": electrical_power - volume_power,
            "current_balance_a": float(sum(current.values())), "temperature_range_k": temperature_range,
            "electrode_potential_spread_v": {
                name: float(np.ptp(potential[el.nodes])) for name, el in self.electrodes.items()
            },
        }
        return FEMResult(temperature, potential, rt, re, summary)


def discrete_adjoint_cold_response(problem: CoupledFEM, result: FEMResult,
                                   plus_problem: CoupledFEM, minus_problem: CoupledFEM,
                                   epsilon: float, *, boundary: str = "cold") -> dict[str, Any]:
    """Derivative of total reservoir heat using the complete discrete adjoint.

    Plus and minus problems differ only in constitutive laws by +/-epsilon.
    Both residual derivatives are evaluated at the *unperturbed* state.  The
    finite-element Jacobian supplies the exact state derivatives.  The linear
    solve uses J.T lambda = dQc/dstate and dQc/depsilon = direct-lambda.dR/deps.
    This is a general multi-material reservoir sensitivity, not an application
    of the fixed-leg one-dimensional cancellation criterion to encapsulation.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    for shifted in (plus_problem, minus_problem):
        if not np.array_equal(shifted.mesh.p, problem.mesh.p) or not np.array_equal(shifted.mesh.t, problem.mesh.t):
            raise ValueError("adjoint comparison requires the same mesh")
        if shifted.P.shape != problem.P.shape or (shifted.P != problem.P).nnz:
            raise ValueError("electrical constraints must be identical")
        if not np.array_equal(shifted.free_t, problem.free_t) or not np.array_equal(shifted.current_rhs, problem.current_rhs):
            raise ValueError("state constraints must be identical")
        for name, (nodes, value) in problem.thermal_boundaries.items():
            other_nodes, other_value = shifted.thermal_boundaries[name]
            if not np.array_equal(nodes, other_nodes) or value != other_value:
                raise ValueError("thermal boundary conditions must be identical")
    multiplicity = np.asarray(problem.P.sum(axis=0)).ravel()
    e = np.asarray(problem.P.T @ result.potential_v).ravel()/multiplicity
    residual, jac, _, _, parameters = problem.residual(result.temperature_k, e, jacobian=True)
    if np.max(np.abs(residual)) > 1e-8:
        raise ValueError("the base state is not converged for this problem")
    nodes = problem.thermal_boundaries[boundary][0]
    full_te = asm(_te, problem.basis, **parameters)
    full_tt = asm(_tt, problem.basis, **parameters)
    g_e = np.asarray((full_te @ problem.P)[nodes].sum(axis=0)).ravel()
    g_t = np.asarray(full_tt[nodes][:, problem.free_t].sum(axis=0)).ravel()
    gradient = np.r_[g_e,g_t]
    adjoint = spsolve(jac.T.tocsc(), gradient)
    plus_r,_,plus_rt,_ = plus_problem.residual(result.temperature_k, e)
    minus_r,_,minus_rt,_ = minus_problem.residual(result.temperature_k, e)
    partial_residual = (plus_r-minus_r)/(2*epsilon)
    direct = float((np.sum(plus_rt[nodes])-np.sum(minus_rt[nodes]))/(2*epsilon))
    feedback = float(-adjoint@partial_residual)
    return {"derivative_w":direct+feedback,"fixed_state_direct_w":direct,
            "coupled_state_feedback_w":feedback,"epsilon":epsilon,"boundary":boundary,
            "adjoint_equation_residual":float(np.max(np.abs(jac.T@adjoint-gradient))),
            "base_residual":float(np.max(np.abs(residual))),
            "method":"full coupled discrete adjoint of the reservoir heat reaction"}


def rectangular_leg(material: Material, signed_current_a: float, *, length_m: float,
                    area_m2: float, cold_k: float, hot_k: float,
                    transverse_cells: int = 2, axial_cells: int = 24,
                    quadrature_order: int = 4) -> CoupledFEM:
    width = np.sqrt(area_m2)
    mesh = MeshTet.init_tensor(np.linspace(0, width, transverse_cells + 1),
                              np.linspace(0, width, transverse_cells + 1),
                              np.linspace(0, length_m, axial_cells + 1))
    cold = np.flatnonzero(np.isclose(mesh.p[2], 0, atol=1e-13))
    hot = np.flatnonzero(np.isclose(mesh.p[2], length_m, atol=1e-13))
    return CoupledFEM(mesh, [material], np.zeros(mesh.nelements, dtype=int),
                      {"cold": (cold, cold_k), "hot": (hot, hot_k)},
                      {"cold": Electrode(cold, signed_current_a),
                       "hot": Electrode(hot, -signed_current_a)}, "cold",
                      cold_k + (hot_k-cold_k) * mesh.p[2] / length_m,
                      quadrature_order=quadrature_order)


def common_mode(material: Material, epsilon: float, mode: str, cold_k: float,
                hot_k: float, amplitude_v_per_k: float = 10e-6) -> Material:
    source = material.seebeck
    span = hot_k - cold_k
    if mode == "linear":
        m = lambda t: amplitude_v_per_k * (t-cold_k) / span
        dm = lambda t: np.full_like(t, amplitude_v_per_k / span)
    elif mode == "endpoint_zero":
        m = lambda t: 4 * amplitude_v_per_k * (t-cold_k) * (hot_k-t) / span**2
        dm = lambda t: 4 * amplitude_v_per_k * (cold_k+hot_k-2*t) / span**2
    elif mode == "constant":
        m = lambda t: np.full_like(t, amplitude_v_per_k)
        dm = lambda t: np.zeros_like(t)
    else:
        raise ValueError(f"unknown mode {mode}")
    law = PropertyLaw(lambda t: source.evaluate(t) + epsilon*m(t),
                      lambda t: source.derivative(t) + epsilon*dm(t), source.domain)
    return replace(material, seebeck=law)


def solve_pair(p: Material, n: Material, *, current_a: float, length_m: float,
               p_area_m2: float, n_area_m2: float, cold_k: float, hot_k: float,
               transverse_cells: int = 2, axial_cells: int = 24,
               epsilon: float = 0.0, mode: str = "endpoint_zero",
               amplitude_v_per_k: float = 10e-6) -> dict[str, Any]:
    legs = {}
    for branch, material, signed_i, area in (("p", p, current_a, p_area_m2), ("n", n, -current_a, n_area_m2)):
        shifted = common_mode(material, epsilon, mode, cold_k, hot_k, amplitude_v_per_k)
        problem = rectangular_leg(shifted, signed_i, length_m=length_m, area_m2=area,
                                  cold_k=cold_k, hot_k=hot_k,
                                  transverse_cells=transverse_cells, axial_cells=axial_cells)
        legs[branch] = problem.solve().summary
    qc = sum(leg["inward_heat_w"]["cold"] for leg in legs.values())
    qh = -sum(leg["inward_heat_w"]["hot"] for leg in legs.values())
    power = sum(leg["electrical_power_w"] for leg in legs.values())
    return {"cold_heat_w": qc, "hot_heat_w": qh, "electrical_power_w": power,
            "voltage_v": power / current_a if current_a else None,
            "energy_residual_w": qh-qc-power, "legs": legs}


def constant_pair(*, seebeck_matched: bool, transfer_matched: bool):
    p = Material("synthetic p", constant(1.5), constant(200e-6 if seebeck_matched else 220e-6), constant(1e-5))
    n = Material("synthetic n", constant(1.0), constant(-200e-6 if seebeck_matched else -180e-6), constant(2e-5))
    ratio = np.sqrt(3.0) if transfer_matched else 1.0
    return p, n, 2e-6 / (1+ratio), 2e-6 * ratio / (1+ratio)


def load_bts_bst():
    """Read the frozen publication-derived curves independently of any solver."""
    path = ROOT/"data/raw/nwae448_bts_bst/nwae448_digitized_transport.csv"
    expected = "aa4fb3e56849690c734d684f32b8a6441ce6fce73ebfc8711316c3d55047604b"
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError("BTS/BST input CSV hash differs from the verified source table")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12:
        raise ValueError("expected twelve source-table rows")
    materials = []
    for branch in ("p", "n"):
        selected = sorted((row for row in rows if row["branch"]==branch), key=lambda row:float(row["temperature_k"]))
        def values(name):
            return np.asarray([float(row[name]) for row in selected])
        t = values("temperature_k")
        if not np.array_equal(t,np.asarray([300.,323.,373.,423.,473.,523.])):
            raise ValueError("unexpected temperature grid")
        materials.append(Material(f"published {branch}-BTS/BST",tabulated(t,values("kappa_w_per_m_k")),
                                  tabulated(t,values("seebeck_uv_per_k")*1e-6),
                                  tabulated(t,1/(values("sigma_1e3_s_per_cm")*1e5))))
    source = {"doi":"10.1093/nsr/nwae448","transport_csv":str(path.relative_to(ROOT)),
              "transport_sha256":observed,"data_role":"publication-figure-derived scalar transport curves",
              "interpolation":"PCHIP of S, rho=1/sigma knots, and kappa; no extrapolation",
              "temperature_domain_k":[300.,523.],"raw_source_validation":"upstream source and pixel-table validation completed locally"}
    return materials[0], materials[1], source


def analytic_constant_leg(material: Material, current: float, length: float,
                          area: float, cold: float, hot: float) -> dict[str, float]:
    t = np.asarray([cold])
    s, rho, k = (float(law.evaluate(t)[0]) for law in (material.seebeck, material.resistivity, material.kappa))
    r, g = rho*length/area, k*area/length
    return {"cold_heat_w": s*current*cold-g*(hot-cold)-0.5*r*current**2,
            "hot_heat_w": s*current*hot-g*(hot-cold)+0.5*r*current**2,
            "voltage_hot_minus_cold_v": -r*current-s*(hot-cold)}


def analytic_constant_pair_derivative(p: Material, n: Material, *, current_a: float,
                                     length_m: float, p_area_m2: float, n_area_m2: float,
                                     cold_k: float, hot_k: float, mode: str,
                                     amplitude_v_per_k: float = 10e-6) -> float:
    """Independent first variation from the exact quadratic base profiles.

    delta Qc = I integral_0^1 (1-u) gamma[Tp(u)] Tp'(u) du
               - I integral_0^1 (1-u) gamma[Tn(u)] Tn'(u) du.
    Endpoint contributions cancel for the pair.  Gaussian quadrature integrates
    these polynomial fixtures exactly, including nonmonotone base profiles.
    """
    u, weights = np.polynomial.legendre.leggauss(24)
    u, weights = (u+1)/2, weights/2
    span = hot_k-cold_k
    value = 0.0
    for material, signed_i, area in ((p, current_a, p_area_m2), (n, -current_a, n_area_m2)):
        rho = float(material.resistivity.evaluate(np.asarray([cold_k]))[0])
        k = float(material.kappa.evaluate(np.asarray([cold_k]))[0])
        h = rho*current_a**2*length_m**2/(2*k*area**2)
        t = cold_k+span*u+h*u*(1-u)
        dt = span+h*(1-2*u)
        if mode == "linear":
            ds = np.full_like(t, amplitude_v_per_k/span)
        elif mode == "endpoint_zero":
            ds = 4*amplitude_v_per_k*(cold_k+hot_k-2*t)/span**2
        elif mode == "constant":
            ds = np.zeros_like(t)
        else:
            raise ValueError(mode)
        value += signed_i * float(np.sum(weights*(1-u)*t*ds*dt))
    return value


def run_fixed_end_examples(*, axial_cells: int, transverse_cells: int, bts_ratio: float):
    reports = []
    for sm in (True, False):
        for tm in (False, True):
            p, n, ap, an = constant_pair(seebeck_matched=sm, transfer_matched=tm)
            kwargs = dict(current_a=1.0, length_m=1.0e-3, p_area_m2=ap, n_area_m2=an,
                          cold_k=300.0, hot_k=330.0, axial_cells=axial_cells,
                          transverse_cells=transverse_cells)
            baseline = solve_pair(p, n, **kwargs)
            modes = {}
            for mode in ("linear", "endpoint_zero"):
                eps = 0.01
                plus, minus = (solve_pair(p, n, **kwargs, epsilon=e, mode=mode) for e in (eps, -eps))
                derivative = (plus["cold_heat_w"]-minus["cold_heat_w"])/(2*eps)
                oracle = analytic_constant_pair_derivative(p, n, **{k:v for k,v in kwargs.items() if k not in ("axial_cells", "transverse_cells")}, mode=mode)
                modes[mode] = {"epsilon": eps, "derivative_w": derivative, "analytic_derivative_w": oracle,
                               "absolute_derivative_error_w": abs(derivative-oracle),
                               "plus": plus, "minus": minus}
            reports.append({"seebeck_matched": sm, "transfer_matched": tm, "inputs": kwargs,
                            "area_ratio_n_over_p": an/ap, "baseline": baseline, "modes": modes})
    p, n, source = load_bts_bst()
    bts = []
    total_area = 2*(1.95e-3)**2
    for ratio in (1.0, bts_ratio):
        kwargs = dict(current_a=3.0, length_m=4.8e-3, p_area_m2=total_area/(1+ratio),
                      n_area_m2=total_area*ratio/(1+ratio), cold_k=323.0, hot_k=373.0,
                      axial_cells=axial_cells, transverse_cells=transverse_cells)
        baseline = solve_pair(p, n, **kwargs)
        modes = {}
        for mode in ("linear", "endpoint_zero"):
            eps = 0.01
            plus, minus = (solve_pair(p, n, **kwargs, epsilon=e, mode=mode) for e in (eps, -eps))
            modes[mode] = {"epsilon": eps, "derivative_w": (plus["cold_heat_w"]-minus["cold_heat_w"])/(2*eps),
                           "plus": plus, "minus": minus}
        bts.append({"area_ratio_n_over_p": ratio, "inputs": kwargs, "baseline": baseline, "modes": modes})
    return {"method": "independent nonlinear conservative 3D tetrahedral P1 finite elements",
            "boundary_scope": "fixed isothermal leg ends, adiabatic sides, ideal equipotential electrodes, fixed total current",
            "source_data": source, "synthetic": reports, "bts_bst": bts}


def run_perturbation_stability(*, axial_cells: int, transverse_cells: int, bts_ratio: float):
    """Central-difference stability; epsilon multiplies a 10 microV/K mode."""
    p,n,source = load_bts_bst()
    total_area = 2*(1.95e-3)**2
    cases = []
    for ratio in (1.0,bts_ratio):
        kwargs = dict(current_a=3.,length_m=.0048,p_area_m2=total_area/(1+ratio),
                      n_area_m2=total_area*ratio/(1+ratio),cold_k=323.,hot_k=373.,
                      axial_cells=axial_cells,transverse_cells=transverse_cells)
        for mode in ("linear","endpoint_zero"):
            values=[]
            for eps in (.1,.03,.01,.003):
                plus,minus=(solve_pair(p,n,**kwargs,epsilon=e,mode=mode) for e in (eps,-eps))
                values.append({"epsilon":eps,"derivative_w":(plus["cold_heat_w"]-minus["cold_heat_w"])/(2*eps),
                               "plus_cold_heat_w":plus["cold_heat_w"],"minus_cold_heat_w":minus["cold_heat_w"],
                               "maximum_energy_residual_w":max(abs(plus["energy_residual_w"]),abs(minus["energy_residual_w"]))})
            cases.append({"area_ratio_n_over_p":ratio,"mode":mode,"inputs":kwargs,"values":values,
                          "last_two_derivative_change_w":abs(values[-1]["derivative_w"]-values[-2]["derivative_w"])})
    return {"method":"central perturbation of complete nonlinear finite-element solves",
            "mode_amplitude_v_per_k":1e-5,"cases":cases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results/transport_validation/fem_fixed_ends/fixed_end_examples.json")
    parser.add_argument("--axial-cells", type=int, default=24)
    parser.add_argument("--transverse-cells", type=int, default=2)
    parser.add_argument("--bts-ratio", type=float, default=1.04240001, help="A_n/A_p, fixed total leg area")
    parser.add_argument("--stability-only", action="store_true")
    args = parser.parse_args()
    runner = run_perturbation_stability if args.stability_only else run_fixed_end_examples
    report = runner(axial_cells=args.axial_cells, transverse_cells=args.transverse_cells, bts_ratio=args.bts_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),"stability_only":args.stability_only}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
