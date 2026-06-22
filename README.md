# Sensitivity Analysis of Malonyl-CoA Gene Circuit for Fatty Acid Biosynthesis in *E. coli*

## The Problem

Microbial fatty acid production is commercially promising but industrially unreliable.
Lab-scale titres routinely overpredict what fermentation processes deliver at scale —
because standard models optimize for ideal conditions, not for the kinetic drift and
parameter uncertainty that define real bioreactors.

Identifying *which* parameters actually drive titre variability is the prerequisite
for designing fermentation conditions that are robust, not just optimal.

## Approach

Global sensitivity analysis on a synthetic gene circuit controlling malonyl-CoA flux
in the *E. coli* fatty acid biosynthesis pathway.

- **Methods:** Sobol indices and Borgonovo sensitivity analysis for variance-based
  parameter ranking under uncertainty
- **Tools:** Python (UQLab)
- **Scope:** Uncertainty quantification across kinetic parameter space to identify
  rate-limiting steps under variable fermentation conditions

## Key Finding

A small subset of kinetic parameters drives the majority of titre variance.
Optimizing circuit design around those parameters — rather than the full system —
yields fermentation conditions projected to achieve a **16-fold titre improvement**
in microbial lipid production.

## Status

Ongoing MSc thesis, KU Leuven (Feb 2026 – present).
Projections are model-derived; experimental validation in progress.

## Skills Demonstrated

`Python` `SALib` `UQLab` `Sensitivity analysis` `Uncertainty quantification`
`Bioprocess modeling` `Experimental design` `Metabolic flux analysis`
A case study for sensitivity analysis of the gene circuit regulating Malonyl-CoA in E.coli
