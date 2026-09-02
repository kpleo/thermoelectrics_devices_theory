# Data sources

## PbSe/Cr

Processed transport coordinates are derived from figures in the article and
Supplementary Information identified by DOI `10.1126/science.aeg8963`.
`data/raw/source_records.csv` records the expected publisher-file hashes. The
non-open publisher files are external sources and are not distributed in this repository.

The four processed tables used by the calculations are:

- `data/processed/pbse_cr_figure1_transport_all_compositions.csv`
- `data/processed/material_thermal_conductivity_figure_s9.csv`
- `data/processed/device_cooling_curves_figure4.csv`
- `data/processed/device_conditions.csv`

The vector-route candidate and reconciliation record retain an independent
source-object check. They are figure-derived candidates, not author-supplied raw
measurements.

## BTS/BST

The second material system comes from the open-access article identified by DOI
`10.1093/nsr/nwae448`. Source acquisition URLs, hashes, and evidence locations are
recorded in `data/raw/nwae448_bts_bst/nwae448_source_records.json`. The included
source objects are distributed under CC BY 4.0.
