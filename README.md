# Thermoelectric device-theory reproducibility archive

This repository contains numerical models, processed input data, tests, and
machine-readable outputs for thermoelectric common-mode transport. The covered
calculations include branch-to-port transfer, split-pad endpoint topology,
one- and two-dimensional conservation tests, and source-constrained PbSe/Cr
and BTS/BST case studies. Independent three-dimensional finite elements now
test material matching, package feedback and finite-depth electrical terminals.

This repository does not include article source files or article graphics.
The analysis scripts can generate local diagnostic plots, which are excluded
from version control.

## Contents

- `data/processed/`: figure-derived transport coordinates used by the models.
- `data/raw/`: source records and licensed source objects used for validation.
- `scripts/analysis/`: one-dimensional, finite-volume and finite-element analyses.
- `scripts/tec_1d_solver/`: temperature-dependent thermoelectric solver.
- `results/scientific_analysis/`: machine-readable numerical results.
- `results/transport_validation/`: controlled geometry comparisons, published-
  property finite elements, composite-device fields and convergence data.
- `tests/`: focused numerical and reproducibility tests.
- `docs/`: data provenance and reproduction notes.

## Quick start

The reference environment used Python 3.10.5.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/reproduce_results.py --from-processed
python -m pytest -q
```

The processed-data calculation does not require publisher PDFs. If a legally
obtained local copy of the PbSe/Cr Supplementary Information is available, the
optional `PBSE_SCIENCE_SI_PDF` environment variable can point to that file. The
analysis then verifies it against the SHA-256 value in
`data/raw/source_records.csv`.

## Material matching and three-dimensional validation

[Finite-element validation](docs/FINITE_ELEMENT_VALIDATION.md) describes the
new calculations, their boundary conditions, source inputs, and reproduction
commands. The response is a fixed-current first derivative of cold-side heat,
not a finite efficiency change. Matching one specified common perturbation and
matching its entire temperature-dependent family are treated separately.

The focused validation suite can be run without regenerating stored results:

```bash
uv run --python 3.10 python -W error -m unittest tests.test_geometry_matching tests.test_seebeck_operations tests.test_thermoelectric_fem tests.test_packaged_device tests.test_electrode_spreading -v
```

The 28 tests cover analytic limits, nonlinear/adjoint derivatives, fixed
material inputs, conservation, insulating ceramics, and finite-depth terminal
constraints. The published numerical results include three mesh levels.

## Data provenance and copyright

Non-open PbSe/Cr publisher files are not redistributed. The repository contains
processed coordinates, source identifiers, DOI links, and source hashes.
The BTS/BST source objects included under `data/raw/nwae448_bts_bst/` are from an
open-access CC BY 4.0 article. See `THIRD_PARTY_NOTICES.md` and
`docs/DATA_SOURCES.md` for details.

## License and attribution

Code is released under the MIT License. Author-generated processed data are
released under CC BY 4.0, subject to the third-party source terms stated in
`THIRD_PARTY_NOTICES.md`. Source publications and licenses are listed in
`docs/DATA_SOURCES.md`.
