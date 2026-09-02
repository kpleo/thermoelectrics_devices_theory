# Reproducibility

## Numerical hierarchy

The repository separates three evidence levels:

1. exact analytical identities and constant-property reductions;
2. deterministic one- and two-dimensional numerical verification;
3. material-system response scales reconstructed from published figures.

The material reconstructions are source-constrained sensitivity analyses.

## Deterministic settings

- Reference Python: 3.10.5
- NumPy: 2.1.1
- SciPy: 1.14.1
- Matplotlib: 3.9.2
- `PYTHONHASHSEED=0`
- `SOURCE_DATE_EPOCH=1787745600`

Run `python scripts/reproduce_results.py --from-processed` from the repository root.
The command rebuilds the numerical records and local diagnostic plots. The
plots are intentionally excluded from version control. Some high-resolution
analyses take several minutes.

## Optional publisher-file verification

Set `PBSE_SCIENCE_SI_PDF` to a legally obtained copy of the PbSe/Cr
Supplementary Information. The analysis checks the file against the recorded
SHA-256 value before using its method statements. Numerical reproduction from
the supplied processed coordinates does not require this external file.
