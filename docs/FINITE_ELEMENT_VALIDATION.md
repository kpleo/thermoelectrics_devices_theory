# Material matching and three-dimensional thermoelectric validation

## Physical question and comparison

The calculations distinguish equality of Seebeck magnitudes from equality of
the two legs' cold-side collection measures. A common perturbation is
`S_p,n(T) -> S_p,n(T) + epsilon m(T)`, with the electrical reference materials
unchanged. Reported responses are `dQc/d epsilon` at zero epsilon and fixed
total current. They are expressed in watts per p/n pair.

The four shapes are linear, parabolic, cold-skewed and hot-skewed. For
`u = (T-Tc)/(Th-Tc)`, they are `m/M = u`, `4u(1-u)`,
`(27/4)u(1-u)^2` and `(27/4)u^2(1-u)`, with `M = 10 microV/K`.
Normalization uses the prescribed material-end interval in the fixed-end
models and the external reservoir interval in the composite-device models.
The polynomials continue analytically through an internal temperature maximum.
An endpoint-zero profile at reservoir temperatures need not vanish at a
semiconductor end inside a package.

The controls keep the total semiconductor area, both leg lengths, material
curves, current and boundary temperatures fixed while changing `An/Ap`.
The constant-property pairs have `Sp/Sn = 200/-200` or `220/-180 microV/K`,
with the same differential thermopower. At `An/Ap = sqrt(3)`, equal
temperature profiles and adjoint weights give complete first-order matching.
For temperature-dependent materials, a zero of one tested mode does not imply
equality of the complete collection measures.

## Material inputs

- BTS/BST uses n-BTS+0.2%Cu and commercial p-BST figure-derived temperature-
  dependent Seebeck coefficient, resistivity and total thermal conductivity
  from DOI [10.1093/nsr/nwae448](https://doi.org/10.1093/nsr/nwae448).
  The fixed-end calculation uses 323/373 K, 3 A, 4.8 mm leg length and
  7.605 mm2 combined semiconductor area.
- PbSe/Cr uses p-PbSe+0.001Cr and n-PbSe+0.005Cr figure-derived curves from
  DOI [10.1126/science.aeg8963](https://doi.org/10.1126/science.aeg8963).
  The selected working point is approximately 310/363 K and 2.81 A,
  with 6 mm legs and 8 mm2 combined area.
- These are published-curve inputs, not newly supplied sample-resolved raw
  measurements. Their measurement-direction to fabricated-leg correspondence
  is not established by the source records. In three dimensions they are
  treated as scalar material laws.
- The original source identifiers, licenses and processed tables are described
  in [Data sources](DATA_SOURCES.md). Numerical source hashes identify the
  corresponding public input records; code is provided under descriptive
  scientific module names.

## Independent finite elements

The solver assembles the coupled electrical and thermal weak forms directly on
tetrahedral P1 elements. It solves

```
j = -sigma(T) [grad(phi) + S(T) grad(T)]
q = -kappa(T) grad(T) + S(T) T j
div(j) = 0
div(q) + j . grad(phi) = 0
```

Electrical terminal surfaces are equipotential, with a specified total current
and one reference potential. Conducting bodies retain finite electrical
resistance. Ceramic-only nodes have thermal but no electrical unknowns.
Cold heat is the total thermal boundary reaction; no separate Peltier term is
added to that reaction. A full coupled discrete adjoint is compared against
centered nonlinear perturbations.

The fixed-leg-end problems have laterally uniform continuous fields. Their
three-dimensional meshes test dimensional reduction and agreement with the
independent one-dimensional solution. They do not by themselves demonstrate
transverse physical spreading.

## Composite devices and electrical injection

The representative device contains two semiconductor legs, separate cold
copper pads, a hot copper bridge, four effective contact layers, and two
electrically insulating ceramic plates. External ceramic faces are held at
323/373 K, the current is 3 A, and other thermal surfaces are insulated.

Nominal package inputs are: depth 1.95 mm, leg length 4.8 mm, interleg gap
1 mm, copper thickness 0.2 mm, contact thickness 0.02 mm, and ceramic thickness
0.5 mm. Copper has `kappa = 400 W/(m K)`, `rho = 1.72e-8 ohm m` and
`S = 1.5 microV/K`; ceramic has `kappa = 25 W/(m K)`. Effective contacts
have areal electrical resistance `3.5e-10 ohm m2`, areal thermal resistance
`1e-6 m2 K/W`, and zero Seebeck coefficient. These are stated model assumptions,
not measured parameters of a reconstructed module.

The full-depth electrical terminals give a depth-invariant reference problem.
A second arrangement restricts the p terminal to the front half and the n
terminal to the back half of the device depth. The remaining electrical
boundary is insulating. This creates genuine three-dimensional current
spreading, mainly in the copper electrodes.

Both geometries use the same reservoir-anchored parabolic perturbation.
The selected area ratio 1.04240001 is the one-dimensional parabolic zero, not
a complete-device optimum. The fine-mesh finite-depth responses are 240.60
and 36.05 microW per pair for equal and adjusted areas. Their medium-to-fine
changes are 1.69 and 0.52 microW. Approximately 85% lower sensitivity is not
an 85% improvement in cooling efficiency.

In the finite-depth case, the depth-directed current-density RMS fraction is
27.1% over the conducting region. The same-mesh full-depth control gives
0.29% and decreases under refinement. Uncorrected maximum cross-depth
temperature differences also contain discretization error; the control and
the difference between the two fields are stored separately. Electrode-edge
peak values are not treated as converged physical maxima.

## Result locations

- `results/transport_validation/geometry/`: four-case controls, material
  area-ratio scans, measure profiles, mode roots, cooling heat/COP and
  centered-difference checks.
- `results/transport_validation/operations/operation_checks.json`: global,
  common-leg and differential-leg operations, zero-current control and
  constant-property complete matching.
- `results/transport_validation/fem_fixed_ends/`: coarse/medium fixed-end
  results and perturbation-step checks.
- `results/transport_validation/converged/fixed_end_fine.json`: fine fixed-end
  results for constant properties and BTS/BST.
- `results/transport_validation/fem_pbse/`: PbSe three-grid results.
- `results/transport_validation/converged/packaged/`: eight full-depth
  composite-device cases, including JSON values and NPZ field arrays.
- `results/transport_validation/fem_packaged/`: equal-area full-depth field
  controls used in identical-mesh comparisons.
- `results/transport_validation/fem_spreading/`: six finite-depth cases,
  three-mesh comparisons and depth-spreading indicators.
- `results/transport_validation/fem_comparison/`: consolidated fixed-end
  and full-depth-package comparisons. The finite-depth summary remains in
  `fem_spreading/summary.json`.

NPZ files contain numerical coordinates, material IDs, fields and boundary
sets, not images. JSON/CSV values retain computational precision for further
analysis; extra digits do not imply experimental precision.

## Reproduction

Use Python 3.10 and the pinned dependencies in `requirements.txt`. The focused
tests shown in the main README need only seconds. Commands below run from the
repository root; use a new output path if you want to preserve stored results.

```
uv run --python 3.10 python -W error scripts/analysis/geometry_matching.py
uv run --python 3.10 python -W error scripts/analysis/seebeck_operations.py
uv run --python 3.10 python -W error scripts/analysis/thermoelectric_fem.py --axial-cells 12 --transverse-cells 1 --output results/transport_validation/fem_fixed_ends/fixed_end_coarse.json
uv run --python 3.10 python -W error scripts/analysis/thermoelectric_fem.py --axial-cells 24 --transverse-cells 2 --output results/transport_validation/fem_fixed_ends/fixed_end_medium.json
uv run --python 3.10 python -W error scripts/analysis/thermoelectric_fem.py --axial-cells 48 --transverse-cells 3 --output results/transport_validation/converged/fixed_end_fine.json
uv run --python 3.10 python -W error scripts/analysis/pbse_fem.py --mesh all
uv run --python 3.10 python -W error scripts/analysis/run_packaged_study.py --output-dir results/transport_validation/converged/packaged
uv run --python 3.10 python -W error scripts/analysis/electrode_spreading.py
uv run --python 3.10 python -W error scripts/analysis/summarize_fem.py --converged-results results/transport_validation/converged
```

To rebuild the full-depth field controls, run `packaged_device.py` with
`--refinement 1`, `2` or `4` and write the equal-area results to
`fem_packaged/ratio1_refine1_adjoint.json`, `ratio1_refine2.json` or
`ratio1_refine4.json`, respectively. Then run `electrode_spreading.py
--summarize-only` to compare identical meshes. Existing finite-depth case files
are preserved; a new `--output-directory` is needed to force their recalculation.

The extra local geometry samples in `local_geometry_plot_scan.csv` can be
re-evaluated through `geometry_matching.evaluate_geometry(system, ratio)` at
the listed `system` and `area_ratio_n_over_p` values. This evaluates all four
response shapes without requiring figure-generation code.

Near cancellation, use absolute response errors. Mesh differences quantify
numerical convergence and must not be interpreted as material uncertainty.
