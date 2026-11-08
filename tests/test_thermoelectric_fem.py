"""Independent analytic checks of the coupled tetrahedral thermoelectric FEM."""

from __future__ import annotations

import unittest
import numpy as np
from skfem import MeshTet

from scripts.analysis.thermoelectric_fem import (
    CoupledFEM, Electrode, Material, PropertyLaw, analytic_constant_leg,
    analytic_constant_pair_derivative, common_mode, constant, constant_pair,
    discrete_adjoint_cold_response, load_bts_bst, rectangular_leg, solve_pair,
)


class ThermoelectricFEMTests(unittest.TestCase):
    def test_constant_property_heat_voltage_and_both_current_directions(self):
        material = Material("test", constant(1.2), constant(220e-6), constant(1.4e-5))
        for current in (-0.7, 0.0, 0.7):
            result = rectangular_leg(material, current, length_m=.0012, area_m2=1e-6,
                                     cold_k=280., hot_k=310., transverse_cells=1, axial_cells=12).solve()
            exact = analytic_constant_leg(material, current, .0012, 1e-6, 280., 310.)
            self.assertAlmostEqual(result.summary["inward_heat_w"]["cold"], exact["cold_heat_w"], places=10)
            self.assertAlmostEqual(-result.summary["inward_heat_w"]["hot"], exact["hot_heat_w"], places=10)
            self.assertAlmostEqual(result.summary["electrode_potential_v"]["hot"], exact["voltage_hot_minus_cold_v"], places=11)
            self.assertAlmostEqual(result.summary["inward_current_a"]["cold"], current, places=10)
            self.assertLess(abs(result.summary["energy_residual_w"]), 1e-10)

    def test_temperature_dependent_resistivity_against_closed_solution(self):
        cold, hot, length, area, current = 300., 330., .001, 1e-6, 1.
        rho0, slope, kappa, seebeck = 1e-5, .01, 1.2, 180e-6
        rho = PropertyLaw(lambda t: rho0*(1+slope*(t-cold)), lambda t: np.full_like(t, rho0*slope))
        material = Material("linear rho", constant(kappa), constant(seebeck), rho)
        result = rectangular_leg(material, current, length_m=length, area_m2=area,
                                 cold_k=cold, hot_k=hot, transverse_cells=2, axial_cells=48).solve()
        lam = np.sqrt(current**2*rho0*slope/(kappa*area**2))
        c = 1/slope
        d = (hot-cold+c*(1-np.cos(lam*length)))/np.sin(lam*length)
        exact_cold = seebeck*current*cold-kappa*area*d*lam
        exact_hot = seebeck*current*hot-kappa*area*lam*(-c*np.sin(lam*length)+d*np.cos(lam*length))
        rho_integral = rho0*slope*(c*np.sin(lam*length)+d*(1-np.cos(lam*length)))/lam
        exact_voltage = -current*rho_integral/area-seebeck*(hot-cold)
        self.assertLess(abs(result.summary["inward_heat_w"]["cold"]-exact_cold), 2e-6)
        self.assertLess(abs(-result.summary["inward_heat_w"]["hot"]-exact_hot), 2e-6)
        self.assertLess(abs(result.summary["electrode_potential_v"]["hot"]-exact_voltage), 1e-7)
        self.assertLess(abs(result.summary["energy_residual_w"]), 1e-10)

    def test_thermal_only_region_has_no_spurious_electrical_equations(self):
        width, length = .001, .002
        mesh = MeshTet.init_tensor(np.linspace(0,width,3), np.linspace(0,width,3), np.linspace(0,length,9))
        center = np.mean(mesh.p[:,mesh.t], axis=1)
        ids = (center[0]>width/2).astype(int)
        materials = [Material("conductor", constant(2.), constant(200e-6), constant(1e-5)),
                     Material("insulator", constant(5.))]
        cold = np.flatnonzero(np.isclose(mesh.p[2],0,atol=1e-13))
        hot = np.flatnonzero(np.isclose(mesh.p[2],length,atol=1e-13))
        ec = cold[mesh.p[0,cold] <= width/2+1e-13]
        eh = hot[mesh.p[0,hot] <= width/2+1e-13]
        problem = CoupledFEM(mesh, materials, ids, {"cold":(cold,300.),"hot":(hot,330.)},
                             {"cold":Electrode(ec),"hot":Electrode(eh)}, "cold",
                             300+30*mesh.p[2]/length)
        result = problem.solve()
        exact_cold = -(2.+5.)*width**2/2/length*30
        self.assertAlmostEqual(result.summary["inward_heat_w"]["cold"], exact_cold, places=10)
        self.assertLess(result.summary["electrical_unknowns"], result.summary["thermal_unknowns"])
        self.assertLess(np.max(np.abs(result.temperature_k-(300+30*mesh.p[2]/length))),1e-8)

    def test_first_variations_against_polynomial_analytic_integral(self):
        for transfer_matched in (False, True):
            p,n,ap,an = constant_pair(seebeck_matched=False,transfer_matched=transfer_matched)
            kwargs = dict(current_a=1.,length_m=.001,p_area_m2=ap,n_area_m2=an,cold_k=300.,hot_k=330.)
            for mode in ("linear","endpoint_zero"):
                eps=.01
                plus = solve_pair(p,n,**kwargs,axial_cells=32,transverse_cells=2,epsilon=eps,mode=mode)
                minus = solve_pair(p,n,**kwargs,axial_cells=32,transverse_cells=2,epsilon=-eps,mode=mode)
                derivative=(plus["cold_heat_w"]-minus["cold_heat_w"])/(2*eps)
                exact=analytic_constant_pair_derivative(p,n,**kwargs,mode=mode)
                self.assertLess(abs(derivative-exact),max(2e-8,.002*abs(exact)))
                if transfer_matched:
                    self.assertLess(abs(derivative),2e-8)
                else:
                    self.assertGreater(abs(derivative),1e-5)

    def test_analytic_block_jacobian(self):
        rho=PropertyLaw(lambda t:1e-5*(1+.002*(t-300)),lambda t:np.full_like(t,2e-8))
        k=PropertyLaw(lambda t:1.2+.001*(t-300),lambda t:np.full_like(t,.001))
        s=PropertyLaw(lambda t:200e-6+2e-7*(t-300),lambda t:np.full_like(t,2e-7))
        problem=rectangular_leg(Material("nonlinear",k,s,rho),.8,length_m=.001,area_m2=1e-6,
                                cold_k=300.,hot_k=330.,transverse_cells=1,axial_cells=6)
        t=problem.initial_temperature.copy()
        e=np.linspace(0,-.02,problem.P.shape[1])
        r,jac,*_=problem.residual(t,e,jacobian=True)
        rng=np.random.default_rng(21)
        direction=np.r_[rng.normal(size=len(e))*.01,rng.normal(size=len(problem.free_t))]
        eps=1e-5
        shifted=t.copy()
        shifted[problem.free_t]+=eps*direction[len(e):]
        shifted_e=e+eps*direction[:len(e)]
        numerical=(problem.residual(shifted,shifted_e)[0]-r)/eps
        self.assertLess(np.linalg.norm(numerical-jac@direction)/np.linalg.norm(jac@direction),2e-6)

    def test_constant_common_shift_cancels_at_shared_cold_temperature(self):
        p,n,ap,an=constant_pair(seebeck_matched=False,transfer_matched=False)
        kwargs=dict(current_a=1.,length_m=.001,p_area_m2=ap,n_area_m2=an,cold_k=300.,hot_k=330.,
                    axial_cells=12,transverse_cells=1)
        base=solve_pair(p,n,**kwargs)
        shifted=solve_pair(p,n,**kwargs,mode="constant",epsilon=2.)
        self.assertAlmostEqual(base["cold_heat_w"],shifted["cold_heat_w"],places=11)
        self.assertAlmostEqual(base["voltage_v"],shifted["voltage_v"],places=11)

    def test_nonmonotone_profile_first_variation(self):
        p,n,ap,an=constant_pair(seebeck_matched=True,transfer_matched=False)
        kwargs=dict(current_a=2.,length_m=.001,p_area_m2=ap,n_area_m2=an,cold_k=300.,hot_k=330.)
        eps=.01
        plus=solve_pair(p,n,**kwargs,axial_cells=48,transverse_cells=2,epsilon=eps,mode="endpoint_zero")
        minus=solve_pair(p,n,**kwargs,axial_cells=48,transverse_cells=2,epsilon=-eps,mode="endpoint_zero")
        self.assertGreater(plus["legs"]["n"]["temperature_range_k"][1],330.)
        derivative=(plus["cold_heat_w"]-minus["cold_heat_w"])/(2*eps)
        exact=analytic_constant_pair_derivative(p,n,**kwargs,mode="endpoint_zero")
        self.assertLess(abs(derivative-exact),.003*abs(exact))

    def test_discrete_adjoint_and_total_peltier_reaction(self):
        p,_,_,_=constant_pair(seebeck_matched=True,transfer_matched=False)
        kwargs=dict(length_m=.001,area_m2=1e-6,cold_k=300.,hot_k=330.,axial_cells=16,transverse_cells=2)
        base_problem=rectangular_leg(p,.8,**kwargs)
        base=base_problem.solve()
        eps=.01
        for mode in ("constant","endpoint_zero"):
            plus=rectangular_leg(common_mode(p,eps,mode,300.,330.),.8,**kwargs)
            minus=rectangular_leg(common_mode(p,-eps,mode,300.,330.),.8,**kwargs)
            adjoint=discrete_adjoint_cold_response(base_problem,base,plus,minus,eps)
            derivative=(plus.solve().summary["inward_heat_w"]["cold"]-minus.solve().summary["inward_heat_w"]["cold"])/(2*eps)
            self.assertLess(abs(adjoint["derivative_w"]-derivative),2e-9)
            self.assertLess(adjoint["adjoint_equation_residual"],1e-10)
            if mode=="constant":
                self.assertAlmostEqual(adjoint["derivative_w"],.8*300*1e-5,places=9)

    def test_publication_property_laws_equal_existing_input_model(self):
        from scripts.analysis.analyze_bts_bst_endpoint_zero_common_mode import validate_sources_and_load_transport,build_legs
        transport,_=validate_sources_and_load_transport()
        original=build_legs(transport)
        independent=load_bts_bst()[:2]
        t=np.linspace(300.,523.,71)
        for old,new in zip(original,independent):
            for legacy,current in ((old.seebeck,new.seebeck),(old.electrical_resistivity,new.resistivity),
                                   (old.thermal_conductivity,new.kappa)):
                np.testing.assert_array_equal(legacy.evaluate(t),current.evaluate(t))
                np.testing.assert_array_equal(legacy.derivative(t),current.derivative(t))


if __name__ == "__main__":
    unittest.main()
