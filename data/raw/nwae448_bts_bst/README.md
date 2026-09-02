# Liu et al. `nwae448` source bundle

This directory contains the primary open-access source objects used for the
BTS+0.2%Cu/BST cross-material validation.  The article is:

> Dongrui Liu *et al.*, "Lattice plainification and band engineering lead to
> high thermoelectric cooling and power generation in n-type Bi2Te3 with mass
> production," *National Science Review* **12**, nwae448 (2025), DOI
> `10.1093/nsr/nwae448` (advance publication 6 December 2024).

Li-Dong Zhao is the corresponding author named in this article.  This source
must not be confused with papers having a different Zhao author.

## Reproducible acquisition

The article is CC BY 4.0.  On 2026-08-26 the PMC binary links presented a
proof-of-work download page, so the same PMC record was retrieved through the
Europe PMC REST service:

- full-text XML:
  `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11737397/fullTextXML`
- article figures and supplementary file bundle:
  `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11737397/supplementaryFiles`

`PMC11737397_fulltext.xml` and `PMC11737397_SupplementaryFiles.zip` are the
downloaded response bodies.  `europepmc_bundle/` is the extracted zip content.
`nwae448_supplement.pdf` is a byte-identical convenience copy of the PDF in the
bundle. All hashes and evidence locations used by the analysis are recorded in
`nwae448_source_records.json`.

## Digitization scope

`nwae448_digitized_transport.csv` records only the six visible marker centres
for the two legs needed here:

- n-type BTS+0.2%Cu: sigma and S from article Fig. 2(a,c), kappa_tot from
  article Fig. 5(a);
- p-type commercial BST: sigma, S, and kappa_tot from Supplementary Fig. S9
  (PDF page 17, printed S17).

Every row retains the pixel centre and axis calibration used to calculate the
tabulated value.  No missing raw table, error covariance, contact boundary
condition, or sub-300-K material property is inferred.  The digitized values
are figure-derived candidates, not author-supplied raw measurements.
