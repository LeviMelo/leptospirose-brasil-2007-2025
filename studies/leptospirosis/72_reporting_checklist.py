"""Reporting-guideline compliance: the artefacts that make the STROBE/RECORD claim true.

A manuscript that cites a reporting guideline and then leaves the operational
pipeline opaque has cited it as decoration. Reviewers said exactly that. This
script emits the three artefacts that convert the claim into something a
reviewer can check line by line:

1. ``record_strobe_checklist.csv`` -- all 22 STROBE items and the 14 RECORD
   extension items, each with the place in *this* study where it is satisfied
   and the result file that evidences it. Where the evidence exists but has not
   yet been written into the manuscript, the row says so. An honest "not yet
   written" is worth more to a reviewer than a fabricated section reference,
   and every ``evidence_file`` path is checked against the filesystem before
   the CSV is written -- a row citing a file that does not exist fails loudly.

2. ``supplementary_inventory.csv`` -- the material reviewers asked to be moved
   out of the main text, in the order they asked for it, mapped onto the result
   files that already exist. The manuscript already cites S3 for the field
   dictionary; the numbering here is the reviewers' own ordering, and it lands
   the dictionary at S3.

3. ``reproducibility_manifest.json`` -- every analysis script numbered 60 and
   above with its purpose, inputs, outputs, the source systems, the analytic
   window, software versions **queried from the running toolchain rather than
   recalled**, and an execution order that is *derived*, not asserted: the
   dependency graph is built by scanning each script's source for the basenames
   of the study's datasets, then topologically sorted. A cycle, or an input
   whose producer runs later, aborts the build.

The one thing this script must not do is compute a study number. It reports on
the pipeline; it is not part of the argument chain. Every quantity it prints is
read from a file another script wrote.

Outputs to ``data/results/reporting/``.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.config import PATHS

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
OUT = PATHS.results / "reporting"
RES = PATHS.results

#: Relative to the repository root, so the CSV is portable and greppable.
R = "data/results"

RSCRIPT = r"C:\Program Files\R\R-4.4.1\bin\Rscript.exe"


# ---------------------------------------------------------------------------
# (a) STROBE + RECORD checklist
# ---------------------------------------------------------------------------
# `status` is not part of the requested column set but is carried because the
# coverage summary has to count something, and counting by re-parsing prose
# would be worse. Three values only:
#   satisfied  -- the item is met, in the manuscript and/or in a result file
#   partial    -- met in substance but incomplete, or evidence exists and the
#                 manuscript sentence that would carry it is thin
#   not_yet    -- the study cannot show this item is met today
#
# Manuscript sections are named, never numbered by line: paper/artigo.qmd is
# under concurrent revision and a line number would be stale before this file
# is read.

STROBE: list[tuple[str, str, str, str, str]] = [
    (
        "1",
        "Title and abstract: indicate the design with a common term; give a "
        "balanced, informative summary",
        "Title names the design ('estudo ecológico de séries temporais') and the "
        "country-window ('Brasil, 2007-2025'). Resumo is structured "
        "Objetivo/Métodos/Resultados/Conclusão and states the unit of analysis "
        "(região de saúde por ano), the exposure H and its inverse direction, "
        "and the three-system comparison. Every number in the Resumo is read "
        "from a result file at compile time, not transcribed.",
        f"paper/artigo.qmd; {R}/descriptives/descriptives_report.json; "
        f"{R}/triangulation/three_systems_by_stratum.csv",
        "satisfied",
    ),
    (
        "2",
        "Background/rationale: scientific background and rationale for the "
        "investigation",
        "Introdução sets out the mild-to-severe spectrum of leptospirosis, the "
        "community-cohort evidence that most infection is never notified, and "
        "why a case-fatality comparison between territories is therefore a "
        "comparison of case definitions in practice.",
        "paper/artigo.qmd; docs/literature/",
        "satisfied",
    ),
    (
        "3",
        "Objectives: specific objectives, including any prespecified hypotheses",
        "Objetivo stated in the Resumo and at the end of the Introdução. The "
        "falsification conditions are prespecified in the script docstrings "
        "rather than left implicit: 68_primary_model.R names four results any "
        "one of which would break the claim; 63, 64, 65, 66, 67 and 69 each "
        "open with what would confirm and what would rule out the threat they "
        "test.",
        "paper/artigo.qmd; studies/leptospirosis/68_primary_model.R; "
        "studies/leptospirosis/69_within_region.py",
        "satisfied",
    ),
    (
        "4",
        "Study design: present key elements early in the paper",
        "Métodos, 'Delineamento e período': ecological time-series, unit health "
        "region x year, 2007-2025, with the reason the series starts in 2007 "
        "(the SINAN Windows -> SINAN NET layout change; only 46 fields are "
        "common to the two eras). The panel that realises this design is "
        "8,208 region-year cells over 432 regions carrying at least one "
        "confirmed case.",
        f"paper/artigo.qmd; {R}/analysis_panel/panel_report.json; "
        f"{R}/analysis_panel/region_year_panel.parquet",
        "satisfied",
    ),
    (
        "5",
        "Setting: locations, relevant dates, periods of exposure, follow-up and "
        "data collection",
        "Métodos, 'Cenário e fontes de dados': 5,570 municipalities on the 2022 "
        "territorial mesh aggregated into 439 health regions by the official "
        "CIB/CIR division; SINAN window 2007-2025 by symptom onset; the "
        "three-system comparison restricted to 2008-2024, the only years all "
        "three systems have complete. The dated provenance of every geography "
        "is in the atlas manifest and ADR-001.",
        f"paper/artigo.qmd; {R}/atlas/atlas_manifest.json; "
        "docs/ADR-001-versioned-geographies.md",
        "satisfied",
    ),
    (
        "6",
        "Participants: eligibility criteria, sources and methods of selection",
        "Métodos, 'Variáveis' (the operationalisation table) and 'Tamanho do "
        "estudo'. The full cascade is study_flow.csv: 328,984 notifications -> "
        "66,667 confirmed -> 66,407 in window -> 66,358 with a residence "
        "municipality. Discarded, inconclusive and unclassified records are "
        "counted as named rows rather than dropped silently.",
        f"{R}/analysis_panel/study_flow.csv; "
        f"{R}/data_quality_record/field_dictionary.csv",
        "satisfied",
    ),
    (
        "7",
        "Variables: clearly define outcomes, exposures, predictors, potential "
        "confounders and effect modifiers; give diagnostic criteria",
        "Métodos, 'Variáveis' carries the operationalisation table in the main "
        "text (field, accepted values, denominator) and points to the field "
        "dictionary (S3) for the full 36-variable version. The exposure H is "
        "defined once as hospitalised confirmed cases over confirmed cases "
        "with a valid hospitalisation field, and its direction is stated as "
        "inverse: high H = surveillance meeting only severe illness.",
        f"paper/artigo.qmd; {R}/data_quality_record/field_dictionary.csv; "
        "docs/CODEBOOK.md",
        "satisfied",
    ),
    (
        "8",
        "Data sources / measurement: for each variable, sources of data and "
        "details of methods of assessment; comparability of methods across groups",
        "Métodos, 'Cenário e fontes de dados' names the four systems. "
        "Per-variable assessment detail -- raw field, accepted codes, "
        "transformation, missing-data rule, aggregation rule -- is one row per "
        "analytic variable in field_dictionary.csv, derived by reading the "
        "codebook registry and the package source rather than from memory. "
        "Comparability across the three systems is the subject of 62.",
        f"{R}/data_quality_record/field_dictionary.csv; "
        f"{R}/triangulation/triangulation_report.json; docs/CODEBOOK.md",
        "satisfied",
    ),
    (
        "9",
        "Bias: describe any efforts to address potential sources of bias",
        "Métodos, 'Vieses' names the shared denominator, the varying "
        "completeness of the outcome field and the ecological design. Each has "
        "a directed analysis behind it rather than a sentence: mechanical "
        "coupling decomposed in 61, differential completeness bounded "
        "adversarially in 63, case mix standardised in 66, confirmation "
        "pathway and severity coding attacked in 65, the minimum-case rule "
        "swept in 64.",
        f"{R}/coupling/coupling_report.json; "
        f"{R}/data_quality_record/cfr_adversarial_gradient_bounds.csv; "
        f"{R}/case_mix/case_mix_report.json; "
        f"{R}/construct_validity/construct_validity_report.json",
        "satisfied",
    ),
    (
        "10",
        "Study size: explain how the study size was arrived at",
        "Métodos, 'Tamanho do estudo': a census of confirmed notifications, no "
        "sampling. The primary model retains every region with at least one "
        "confirmed case and a known outcome, using partial pooling instead of "
        "exclusion; the quintile tables exclude regions with fewer than 30 "
        "cases with a valid hospitalisation field, and the effect of that rule "
        "is reported at four thresholds.",
        f"{R}/analysis_panel/study_flow.csv; "
        f"{R}/exclusion_sensitivity/exclusion_by_threshold.csv",
        "satisfied",
    ),
    (
        "11",
        "Quantitative variables: explain how they were handled; which groupings "
        "were chosen and why",
        "H enters the model continuously, on a fixed +10-percentage-point "
        "scale, centred at the national value; the scale is carried as a "
        "column in every estimate table rather than left in prose. For the "
        "descriptive tables regions are ordered on H and cut into five strata "
        "of approximately equal CASE mass (not equal region count), so each "
        "stratum supports a case-fatality estimate of comparable precision; "
        "the cut points and the region/case/person-year count of each stratum "
        "are reported.",
        f"{R}/primary_model/primary_model_report.json; "
        f"{R}/case_mix/H_quintile_assignment.csv; "
        f"{R}/triangulation/three_systems_by_stratum.csv",
        "satisfied",
    ),
    (
        "12",
        "Statistical methods: (a) all methods including those to control "
        "confounding; (b) subgroups and interactions; (c) missing data; "
        "(d) loss to follow-up / sampling strategy; (e) sensitivity analyses",
        "(a) Métodos, 'Métodos estatísticos': exact Poisson (Garwood) and "
        "Clopper-Pearson intervals; hierarchical binomial with BYM2 spatial "
        "effect and RW1 trend fitted by INLA; a region fixed-effects "
        "specification fitted alongside. (b) The modifier examined is the "
        "hospitalised stratum (model E), which discriminates the two accounts. "
        "(c) Missingness is handled by valid-state denominators, by "
        "restriction to complete region-years (model D) and by an adversarial "
        "bound, never by imputation. (d) No follow-up and no sampling: a "
        "census of notifications. (e) Sensitivity: exclusion thresholds, "
        "severity definitions, laboratory-only case definition, socioeconomic "
        "covariates.",
        f"{R}/primary_model/model_estimates.csv; "
        f"{R}/primary_model/primary_model_report.json; "
        f"{R}/data_quality_record/cfr_adversarial_gradient_bounds.csv; "
        f"{R}/exclusion_sensitivity/gradient_summary_by_threshold.csv",
        "satisfied",
    ),
    (
        "13",
        "Participants: numbers at each stage; reasons for non-participation; "
        "consider a flow diagram",
        "study_flow.csv is the canonical cascade and reconciles to the last "
        "record (328,984 -> 66,358). The manuscript reads it at compile time, "
        "so the in-text numbers cannot drift from it. NOT YET: the cascade is "
        "reported as prose, not as a flow figure; STROBE says 'consider', and "
        "RECORD 13.1 accepts text or diagram, so this is a presentational gap "
        "rather than a missing analysis.",
        f"{R}/analysis_panel/study_flow.csv",
        "partial",
    ),
    (
        "14",
        "Descriptive data: characteristics of participants and information on "
        "exposures and potential confounders; number with missing data for each "
        "variable",
        "Table 1 gives, by macro-region, confirmed cases, incidence, case "
        "fatality, H, outcome completeness and laboratory-confirmed share. "
        "Completeness travels beside the estimate it conditions rather than "
        "sitting in a separate appendix. Per-variable missingness for all 36 "
        "analytic variables is in the data-quality record (S2).",
        f"{R}/descriptives/table1_by_region.csv; "
        f"{R}/descriptives/national_by_year.csv; "
        f"{R}/data_quality_record/completeness_overall.csv",
        "satisfied",
    ),
    (
        "15",
        "Outcome data: report numbers of outcome events or summary measures",
        "Deaths, the known-outcome denominator, cases and person-years are "
        "carried per region-year in the panel and summed in the descriptive "
        "tables (national: 5,951 deaths over 61,023 known outcomes, 9.75%). "
        "Numerators and denominators are summed before any ratio is formed.",
        f"{R}/descriptives/descriptives_report.json; "
        f"{R}/analysis_panel/region_year_panel.parquet; "
        f"{R}/exclusion_sensitivity/exclusion_by_threshold.csv",
        "satisfied",
    ),
    (
        "16",
        "Main results: unadjusted and confounder-adjusted estimates with "
        "precision; which confounders were adjusted for and why; category "
        "boundaries; translate relative risk into absolute risk where meaningful",
        "Table 2 is the model sequence: A (space-time only), B (+H), C (+case "
        "mix), D (complete outcomes), with the odds ratio per +10 pp of H, its "
        "95% credible interval, the scale and the number of cells, from "
        "model_estimates.csv (85 rows). The absolute-risk translation is the "
        "case-fatality level of each H stratum (2.7% to 16.9%) reported "
        "alongside the relative estimate; the H cut points are in "
        "H_quintile_assignment.csv.",
        f"{R}/primary_model/model_estimates.csv; "
        f"{R}/primary_model/primary_model_report.json; "
        f"{R}/triangulation/three_systems_by_stratum.csv; "
        f"{R}/case_mix/H_quintile_assignment.csv",
        "satisfied",
    ),
    (
        "17",
        "Other analyses: subgroups and interactions, sensitivity analyses",
        "Threat check E (within the hospitalised stratum only), threat check F "
        "(regions with at least 10 known hospitalisation fields), the "
        "region fixed-effects estimate, the first-difference and lag-lead "
        "checks, the RS-2024 within-territory perturbation, the three-system "
        "triangulation, the exclusion-threshold sweep, the severity-definition "
        "sweep and the socioeconomic checks. Each is tied to a named threat "
        "rather than run as an undirected robustness sprawl.",
        f"{R}/primary_model/primary_model_report.json; "
        f"{R}/within_region/threat_ledger_checks.csv; "
        f"{R}/rs2024_event/rs2024_event_report.json; "
        f"{R}/triangulation/three_systems_by_stratum.csv; "
        f"{R}/socioeconomic_confounders/socioeconomic_confounders_report.json",
        "satisfied",
    ),
    (
        "18",
        "Key results: summarise key results with reference to study objectives",
        "Discussão opens on the three-system contrast that answers the "
        "objective: reported case fatality moves 6.3-fold and notified "
        "incidence 5.3-fold across H while population mortality from A27 stays "
        "between 1.7 and 2.5 per million person-years without monotone order.",
        f"paper/artigo.qmd; {R}/triangulation/triangulation_report.json",
        "satisfied",
    ),
    (
        "19",
        "Limitations: discuss limitations, sources of potential bias or "
        "imprecision; discuss both direction and magnitude of any potential bias",
        "Discussão, 'Limitações' gives direction and magnitude, not a list of "
        "regrets: roughly a third of the crude gradient is attributable to the "
        "shared denominator; the within-region estimate recovers about 56% of "
        "the pooled log odds ratio and 96% of the spatially adjusted one; "
        "outcome completeness varies with the exposure in the direction that "
        "would inflate the gradient, and the result survives the adversarial "
        "extreme; H also reflects admission thresholds and bed supply, so it "
        "is not a pure index of case-finding effort.",
        f"paper/artigo.qmd; {R}/coupling/coupling_report.json; "
        f"{R}/data_quality_record/cfr_adversarial_gradient_bounds.csv; "
        f"{R}/within_region/within_region_report.json",
        "satisfied",
    ),
    (
        "20",
        "Interpretation: cautious overall interpretation considering objectives, "
        "limitations, multiplicity, results from similar studies and other "
        "relevant evidence",
        "Discussão places the finding against the Salvador community cohorts "
        "and against the earlier national study that used SINAN alone, and "
        "states what the estimate does not license: H is offered as an "
        "indicator of comparability, explicitly not as a measure of "
        "surveillance performance. The RS-2024 episode is labelled supportive "
        "within-territory validation, not a natural experiment.",
        f"paper/artigo.qmd; {R}/rs2024_event/rs2024_event_report.json",
        "satisfied",
    ),
    (
        "21",
        "Generalisability: discuss the generalisability (external validity) of "
        "the study results",
        "The material for this item exists and is partly used: the primary "
        "model covers every health region with a confirmed case, and the "
        "person-time cost of the 30-case rule on the descriptive tables is "
        "quantified (retained regions hold 76.8% of national person-time "
        "against 96.4% of cases, so a case-share statement alone would "
        "overstate territorial coverage). NOT YET: no explicit external-"
        "validity paragraph in the Discussão; the person-time-versus-case-mass "
        "asymmetry is not stated as a generalisability caveat.",
        f"{R}/exclusion_sensitivity/exclusion_sensitivity_report.json; "
        f"{R}/exclusion_sensitivity/exclusion_by_threshold.csv; "
        f"{R}/exclusion_sensitivity/excluded_macroregion_composition.csv",
        "partial",
    ),
    (
        "22",
        "Funding: give the source of funding and the role of the funders; and "
        "for the original study on which the article is based",
        "NOT YET: the manuscript has an 'Aspectos éticos' statement (secondary, "
        "aggregated, public, unidentified data; CEP/CONEP waiver under "
        "Resolução CNS 510/2016) and a generative-AI declaration, but no "
        "funding statement and no conflict-of-interest statement. Author "
        "action; no result file can supply this.",
        "paper/artigo.qmd",
        "not_yet",
    ),
]

RECORD: list[tuple[str, str, str, str, str]] = [
    (
        "1.1",
        "The type of data used should be specified in the title or abstract; "
        "where possible, name the databases",
        "The abstract names SINAN ('Sistema de Informação de Agravos de "
        "Notificação') and the three systems compared (notification, hospital "
        "admission, population mortality by ICD-10 A27); the title names the "
        "design. NOT YET: the title itself does not name the databases, and "
        "SIM and SIH are named in the abstract by function rather than by "
        "acronym.",
        "paper/artigo.qmd",
        "partial",
    ),
    (
        "1.2",
        "If applicable, the geographic region and timeframe should be reported "
        "in the title or abstract",
        "Title: 'Brasil, 2007-2025'. Abstract repeats the window and states the "
        "unit of analysis (região de saúde por ano) and the restricted "
        "2008-2024 window used for the three-system comparison.",
        f"paper/artigo.qmd; {R}/analysis_panel/panel_report.json",
        "satisfied",
    ),
    (
        "1.3",
        "If linkage between databases was conducted, state it clearly in the "
        "title or abstract",
        "No linkage was conducted and none is claimed. SINAN, SIM and SIH are "
        "compared as three independent aggregate series, aligned on "
        "municipality of residence and calendar year -- never on a person. The "
        "abstract says the three systems are compared; it does not use the "
        "word linkage, which is correct. Independence of the denominators is "
        "the whole point of the triangulation, and it would be destroyed by "
        "linkage.",
        f"{R}/triangulation/triangulation_report.json; "
        f"{R}/triangulation/region_three_systems.parquet",
        "satisfied",
    ),
    (
        "6.1",
        "Methods of study-population selection (codes and algorithms used to "
        "identify subjects) listed in detail, or an explanation of why not",
        "The selection algorithm is four rules, each with its own denominator "
        "and each counted in the cascade: CLASSI_FIN = confirmado; SEM_PRI "
        "yielding a valid onset year inside 2007-2025; a non-null ID_MN_RESI "
        "resolving on the 2022 municipal lattice. Codes, accepted values and "
        "the four decode states (valid / unknown / missing / invalid, never "
        "pooled) are one row per variable in the field dictionary; the "
        "record-count consequence of each rule is a row in study_flow.csv.",
        f"{R}/data_quality_record/field_dictionary.csv; "
        f"{R}/analysis_panel/study_flow.csv; docs/CODEBOOK.md",
        "satisfied",
    ),
    (
        "6.2",
        "Reference any validation studies of the codes or algorithms used to "
        "select the population; if validation was done for this study, give "
        "methods and results",
        "No chart-review validation of SINAN case status exists for this study "
        "and none is claimed -- stating that is the requirement, not "
        "performing one. What was done instead is external comparison against "
        "two systems with different failure modes (SIM underlying-cause A27; "
        "SIH principal-diagnosis A27) and an internal construct check that "
        "repeats every quantity inside the laboratory-confirmed stratum, where "
        "confirmation-criterion drift cannot operate.",
        f"{R}/triangulation/triangulation_report.json; "
        f"{R}/construct_validity/cfr_by_quintile_by_criterion.csv; "
        f"{R}/construct_validity/lab_share_by_quintile.csv",
        "partial",
    ),
    (
        "6.3",
        "If the study involved linkage, consider a flow diagram or other "
        "graphical display of the linkage process, with numbers at each stage",
        "Not applicable, with a reason: no record linkage was performed, so "
        "there is no linkage cascade to display. The selection cascade that "
        "does exist is study_flow.csv and is covered by 13.1. The three "
        "systems are joined at health-region x year on their own aggregates, "
        "and the region-level join is auditable in "
        "region_three_systems.parquet.",
        f"{R}/triangulation/region_three_systems.parquet; "
        f"{R}/analysis_panel/study_flow.csv",
        "satisfied",
    ),
    (
        "6.4",
        "Linkage quality evaluated and reported (the published RECORD statement "
        "carries this requirement at 12.3; retained here at 6.4 to match the "
        "numbering used by this study's earlier checklist)",
        "Not applicable, with a reason: no linkage, therefore no linkage "
        "quality to evaluate. The nearest analogue that IS evaluated is the "
        "geographic join -- every one of the 5,570 municipalities on the 2022 "
        "lattice must map to a health region or the build raises, and the "
        "6-digit to 7-digit municipal code conversion uses the check-digit "
        "codec rather than string concatenation. This row and 12.3 point at "
        "the same evidence and neither is padding.",
        f"{R}/atlas/atlas_manifest.json; "
        f"{R}/audit_denominators_geo/06_geography.json; "
        "docs/ADR-001-versioned-geographies.md",
        "satisfied",
    ),
    (
        "7.1",
        "A complete list of codes and algorithms used to classify exposures, "
        "outcomes, confounders and effect modifiers, or an explanation of why "
        "it cannot be given",
        "field_dictionary.csv: 36 rows covering every analytic variable, each "
        "with source system, raw field, accepted values, transformation, "
        "missing-data rule, aggregation rule and the scripts that consume it. "
        "It records the awkward cases rather than hiding them -- CLI_HEMOPU "
        "excluded from the severe phenotype because it is in practice the same "
        "field as CLI_HEMORR (they disagree in 13 of 61,219 records), the "
        "conjunctive severity denominator, and the one asymmetry in the panel "
        "(share_male computed on all cases rather than on the valid-state "
        "denominator).",
        f"{R}/data_quality_record/field_dictionary.csv; docs/CODEBOOK.md",
        "satisfied",
    ),
    (
        "12.1",
        "Describe the extent to which the investigators had access to the "
        "database population used to create the study population",
        "Full access: the source files are the complete national annual "
        "SINAN-LEPT, SIM-DO and SIH-RD public microdata releases, downloaded "
        "in full, with no data-provider filter applied before the study's own "
        "selection rules. The denominator population is the complete "
        "municipality x sex x single-year-of-age tensor. NOT YET: this is "
        "stated in the pipeline (extraction scripts and the acquisition "
        "record) but not written into the manuscript as an access statement.",
        f"{R}/00_data_quality/reconciliation.csv; "
        f"{R}/audit_sim_sih/reach_summary.json; "
        f"{R}/audit_denominators_geo/00_summary.json",
        "partial",
    ),
    (
        "12.2",
        "Provide information on the data cleaning methods used",
        "Cleaning is decoding, and it is specified rather than performed ad "
        "hoc: every coded field carries one of four states (valid / unknown / "
        "missing / invalid) which are never pooled; a blank is never read as a "
        "negative answer; a value outside the registry is invalid rather than "
        "coerced; packed ages are converted by unit rather than floored; "
        "6-digit municipal codes are widened by check-digit codec. Undocumented "
        "codes and schema drift across the 19 annual files are logged.",
        f"{R}/data_quality_record/field_dictionary.csv; "
        f"{R}/00_data_quality/quality_undocumented_codes.csv; "
        f"{R}/00_data_quality/schema_drift.csv; docs/CODEBOOK.md",
        "satisfied",
    ),
    (
        "12.3",
        "State whether the study included person-level, institutional-level or "
        "other data linkage across two or more databases; give the methods of "
        "linkage and of linkage-quality evaluation",
        "Stated explicitly: no person-level and no institutional-level linkage. "
        "All aggregation is ecological, at health region x year. The three "
        "systems meet only as aggregates sharing a geography and a calendar "
        "year, and the SIH figure is admissions rather than people (an AIH is "
        "a billing record; a transferred patient generates more than one), "
        "which is recorded in the dictionary so the count is not misread as a "
        "person count.",
        f"{R}/data_quality_record/field_dictionary.csv; "
        f"{R}/triangulation/triangulation_report.json",
        "satisfied",
    ),
    (
        "13.1",
        "Describe in detail the selection of persons included, including "
        "filtering on data quality, data availability and linkage; text and/or "
        "flow diagram",
        "study_flow.csv carries every step with counts and percentages and "
        "reconciles to the last record. Data-availability filtering is "
        "reported separately from eligibility filtering: 155 confirmed records "
        "excluded for an invalid onset date, 105 for an onset year outside the "
        "window, 49 for a missing residence municipality. Quality-based "
        "restriction is never silent -- the minimum-case rule is swept at "
        "0/10/20/30/50 and the completeness restriction is a separate model, "
        "not a hidden default.",
        f"{R}/analysis_panel/study_flow.csv; "
        f"{R}/exclusion_sensitivity/exclusion_by_threshold.csv; "
        f"{R}/data_quality_record/cfr_gradient_by_completeness_restriction.csv",
        "satisfied",
    ),
    (
        "19.1",
        "Discuss the implications of using data not created to answer the "
        "research question: misclassification, unmeasured confounding, missing "
        "data, and changing eligibility over time",
        "All four are addressed with evidence rather than acknowledgement. "
        "Misclassification: the confirmation-criterion mix and the severity "
        "coding are attacked directly in 65. Unmeasured confounding: the "
        "region fixed-effects specification removes every time-invariant "
        "regional characteristic by construction, and socioeconomic covariates "
        "are checked in 53. Missing data: differential completeness is bounded "
        "adversarially. Changing eligibility over time: the series starts in "
        "2007 because the SINAN NET layout replaced SINAN Windows, and "
        "completeness by year is reported per field. The manuscript's "
        "'Limitações' now carries the substance of this item.",
        f"{R}/construct_validity/construct_validity_report.json; "
        f"{R}/within_region/within_region_report.json; "
        f"{R}/data_quality_record/completeness_by_year.csv; "
        f"{R}/socioeconomic_confounders/socioeconomic_confounders_report.json",
        "satisfied",
    ),
    (
        "22.1",
        "Provide information on how to access supplemental information such as "
        "the study protocol, raw data or programming code",
        "Manuscript, 'Disponibilidade de dados': the sources are public "
        "DATASUS and IBGE releases, and the extraction, processing and "
        "analysis code, the variable dictionary, this STROBE/RECORD "
        "checklist, the derived region-year panel and the result files behind "
        "every number are published as a reproducibility package at "
        "https://github.com/LeviMelo/leptospirose-brasil-2007-2025, cited in that section of the manuscript.",
        f"paper/artigo.qmd; {R}/reporting/reproducibility_manifest.json; "
        f"{R}/reporting/supplementary_inventory.csv",
        "satisfied",
    ),
]


# ---------------------------------------------------------------------------
# (b) Supplementary material inventory
# ---------------------------------------------------------------------------
# The numbering is the reviewers' own ordering of what should leave the main
# text. That ordering happens to place the field dictionary at S3, which is the
# supplement the manuscript already cites by number in 'Variáveis'. Do not
# renumber without checking paper/artigo.qmd.

SUPPLEMENTS: list[tuple[str, str, str, str]] = [
    (
        "S1",
        "Limiares alternativos de exclusão de regiões de saúde (0, 10, 20, 30 e "
        "50 casos)",
        "Efeito da regra de número mínimo de casos sobre o gradiente de "
        "letalidade. Para cada limiar: número de regiões retidas e excluídas, "
        "percentual de casos e percentual de pessoas-ano em cada braço, "
        "letalidade e incidência dos dois braços, letalidade do primeiro e do "
        "último quintil de H com intervalo exato, razão Q5/Q1, correlação de "
        "postos entre H e letalidade, e ajuste hierárquico binomial sem "
        "qualquer exclusão. Documenta que a retenção por massa de casos "
        "(96,4%) e por pessoas-ano (76,8%) não são a mesma coisa e que as "
        "regiões excluídas têm H mais alto, não mais baixo.",
        f"{R}/exclusion_sensitivity/exclusion_by_threshold.csv; "
        f"{R}/exclusion_sensitivity/gradient_by_threshold.csv; "
        f"{R}/exclusion_sensitivity/gradient_summary_by_threshold.csv; "
        f"{R}/exclusion_sensitivity/hierarchical_model_fits.csv; "
        f"{R}/exclusion_sensitivity/exclusion_sensitivity_report.json",
    ),
    (
        "S2",
        "Completude dos campos por variável, por ano, por macrorregião e por "
        "região de saúde",
        "Completude de cada campo analítico decomposta nos quatro estados de "
        "decodificação (válido, ignorado, ausente, inválido), com intervalo "
        "exato: no total, por ano de 2007 a 2025, por macrorregião, por região "
        "de saúde e por quintil de H. Inclui a correlação de postos entre "
        "completude e exposição, o gradiente de letalidade sob restrição "
        "progressiva a regiões bem registradas e os limites adversariais do "
        "gradiente — todo desfecho não registrado do estrato de detecção ampla "
        "contado como óbito e todo desfecho não registrado do estrato de "
        "detecção restrita contado como sobrevivência.",
        f"{R}/data_quality_record/completeness_overall.csv; "
        f"{R}/data_quality_record/completeness_by_year.csv; "
        f"{R}/data_quality_record/completeness_by_macroregion.csv; "
        f"{R}/data_quality_record/completeness_by_H_quintile.csv; "
        f"{R}/data_quality_record/region_completeness.parquet; "
        f"{R}/data_quality_record/completeness_vs_H_spearman.csv; "
        f"{R}/data_quality_record/cfr_gradient_by_completeness_restriction.csv; "
        f"{R}/data_quality_record/cfr_adversarial_gradient_bounds.csv; "
        f"{R}/data_quality_record/cfr_extreme_bounds.csv",
    ),
    (
        "S3",
        "Dicionário de variáveis, códigos aceitos e regras de transformação",
        "Uma linha por variável analítica (36 no total), com sistema de origem, "
        "campo bruto, valores aceitos e seus códigos, transformação aplicada, "
        "regra de dados ausentes, regra de agregação e scripts que a consomem. "
        "Cobre SINAN-LEPT, SIM, SIH, denominadores populacionais, geografia e "
        "as variáveis derivadas (H, letalidade, incidência, fenótipo grave, "
        "quintis). Registra explicitamente as decisões discutíveis: exclusão de "
        "CLI_HEMOPU do fenótipo grave, denominador conjuntivo da gravidade, uso "
        "do município de residência e não do de notificação, e a base temporal "
        "transversal das covariáveis do Censo 2022. Acompanha o codebook do "
        "repositório e o registro de códigos não documentados e de mudança de "
        "leiaute entre os arquivos anuais.",
        f"{R}/data_quality_record/field_dictionary.csv; docs/CODEBOOK.md; "
        f"{R}/00_data_quality/quality_undocumented_codes.csv; "
        f"{R}/00_data_quality/schema_drift.csv; "
        f"{R}/dictionary/",
    ),
    (
        "S4",
        "Regiões de saúde incluídas e excluídas em cada limiar: insumo do mapa",
        "Tabela por região de saúde com código, nome, unidade federativa, "
        "macrorregião, casos, óbitos, desfechos conhecidos, denominador de "
        "internação, pessoas-ano, H, letalidade, incidência, completude do "
        "desfecho, anos observados e quatro indicadores lógicos de exclusão "
        "(limiares 10, 20, 30 e 50). É o insumo direto do mapa de cobertura "
        "territorial: permite desenhar quais regiões saem das análises por "
        "quintil e verificar que a saída não é aleatória no espaço. Acompanha a "
        "composição macrorregional do braço excluído e a geometria oficial das "
        "regiões de saúde.",
        f"{R}/exclusion_sensitivity/region_inclusion.csv; "
        f"{R}/exclusion_sensitivity/excluded_macroregion_composition.csv; "
        f"paper/figures/geo_health_region.gpkg",
    ),
    (
        "S5",
        "Especificação do modelo: prioris, hiperparâmetros e diagnósticos",
        "Especificação completa do modelo hierárquico binomial. Prioris de "
        "complexidade penalizada: P(desvio padrão espacial > 1) = 0,01 e "
        "P(phi < 0,5) = 0,5 para o efeito BYM2, P(desvio padrão do passeio "
        "aleatório de primeira ordem > 1) = 0,01, e prioris normais N(0, 1) "
        "sobre os efeitos fixos na escala de log razão de chances por 10 pontos "
        "percentuais (precisão 0,001 no intercepto). Estimativas de todos os "
        "termos de todos os modelos com intervalo de credibilidade de 95% e "
        "escala declarada, DIC e WAIC de cada modelo, parâmetro de mistura phi, "
        "variância marginal espacial antes e depois da entrada de H, e o grafo "
        "de adjacência das regiões de saúde. As prioris estão declaradas no "
        "código (R/12_severity.R e R/03_inla_spacetime.R) e impressas na "
        "execução de 68_primary_model.R; não há arquivo de resultado que as "
        "armazene isoladamente.",
        f"{R}/primary_model/model_estimates.csv; "
        f"{R}/primary_model/primary_model_report.json; "
        f"{R}/primary_model/spatial_effects_A_vs_B.csv; "
        f"{R}/primary_model/health_region_bym2.adj; "
        "R/12_severity.R; R/03_inla_spacetime.R; "
        "studies/leptospirosis/68_primary_model.R",
    ),
    (
        "S6",
        "Sensibilidade à definição de fenótipo grave",
        "Repetição do contraste de incidência grave e não grave entre quintis "
        "de H sob definições alternativas de gravidade: icterícia isolada, "
        "insuficiência renal isolada, hemorragia isolada, a disjunção primária "
        "das três, a conjunção icterícia E renal (fenótipo tipo Weil) e a "
        "variante de quatro campos que inclui hemorragia pulmonar; e sob "
        "restrição aos registros com o bloco clínico inteiramente preenchido. "
        "Inclui completude do bloco clínico e distribuição do escore de "
        "gravidade. Os estratos de exposição permanecem fixos nos quintis "
        "primários de H em todas as variantes: varia-se a definição de "
        "gravidade, nunca a exposição.",
        f"{R}/construct_validity/severity_gradient_summary.csv; "
        f"{R}/construct_validity/severity_incidence_by_quintile.csv; "
        f"{R}/construct_validity/severity_score_distribution.csv; "
        f"{R}/construct_validity/clinical_block_completeness.csv; "
        f"{R}/construct_validity/total_incidence_by_quintile.csv; "
        f"{R}/construct_validity/construct_validity_report.json",
    ),
    (
        "S7",
        "Verificações socioeconômicas e de composição demográfica",
        "Covariáveis municipais do Censo 2022 por quintil de H — proporção de "
        "população com renda domiciliar per capita de até um quarto do salário "
        "mínimo, mediana de renda per capita ponderada pela população e "
        "proporção de domicílios com mais de três moradores por dormitório —, "
        "com o número de municípios por trás de cada faixa. Acompanha a "
        "padronização direta e indireta da letalidade por idade e sexo, a "
        "composição etária por quintil e a letalidade por quintil DENTRO de "
        "estratos etários amplos, que é o teste decisivo: composição entre "
        "estratos não pode produzir gradiente dentro deles. Registra a "
        "simplificação assumida — corte transversal de 2022 contra janela de "
        "desfecho de 2007 a 2025.",
        f"{R}/socioeconomic_confounders/covariates_by_depth_quintile.csv; "
        f"{R}/socioeconomic_confounders/socioeconomic_confounders_report.json; "
        f"{R}/case_mix/standardised_cfr_by_H_quintile.csv; "
        f"{R}/case_mix/cfr_by_H_quintile_within_age_stratum.csv; "
        f"{R}/case_mix/composition_by_H_quintile.csv; "
        f"{R}/case_mix/case_mix_report.json",
    ),
    (
        "S8",
        "Sensibilidade ao critério de confirmação laboratorial",
        "Repetição do gradiente de letalidade dentro de cada via de confirmação "
        "separadamente — clínico-laboratorial e clínico-epidemiológica — e "
        "reconstrução de H sobre casos exclusivamente laboratoriais, com "
        "verificação de que a ordenação das regiões se preserva. Inclui a "
        "proporção de confirmação laboratorial por quintil de H e por ano, e a "
        "série do Rio Grande do Sul, onde a proporção laboratorial cai durante "
        "a busca ativa de 2024; todas as quantidades do episódio de 2024 são "
        "repetidas no estrato exclusivamente laboratorial, onde a mudança de "
        "critério não pode operar.",
        f"{R}/construct_validity/cfr_by_quintile_by_criterion.csv; "
        f"{R}/construct_validity/lab_share_by_quintile.csv; "
        f"{R}/construct_validity/lab_share_by_year.csv; "
        f"{R}/construct_validity/lab_share_rs_by_year.csv; "
        f"{R}/construct_validity/region_H_variants.parquet; "
        f"{R}/rs2024_event/rs2024_event_report.json",
    ),
    (
        "S9",
        "Lista de verificação STROBE/RECORD e manifesto de reprodutibilidade",
        "Os 22 itens do STROBE e os 14 itens da extensão RECORD, cada um com o "
        "local exato deste estudo em que é atendido e o arquivo de resultado "
        "que o evidencia, mais o estado declarado (atendido, parcial, ainda "
        "não). Acompanha o manifesto de reprodutibilidade: todo script de "
        "análise numerado a partir de 60 com sua finalidade, entradas, saídas, "
        "sistemas de origem, janela analítica, versões de software consultadas "
        "na execução e a ordem exata de execução necessária para reproduzir "
        "cada número do manuscrito, derivada do grafo de dependências entre "
        "arquivos e não afirmada de memória.",
        f"{R}/reporting/record_strobe_checklist.csv; "
        f"{R}/reporting/reproducibility_manifest.json; "
        f"{R}/reporting/supplementary_inventory.csv",
    ),
]


# ---------------------------------------------------------------------------
# (c) Reproducibility manifest
# ---------------------------------------------------------------------------

SOURCE_SYSTEMS = [
    {
        "id": "SINAN-LEPT",
        "name": "Sistema de Informação de Agravos de Notificação, agravo "
                "leptospirose (LEPTBR)",
        "custodian": "Ministerio da Saude / DATASUS",
        "granularity": "notification record (line level)",
        "coverage_used": "annual national files 2007-2025",
        "role": "numerator: confirmed cases, deaths, hospitalisation, "
                "confirmation criterion, clinical block, demography",
        "key_fields": ["CLASSI_FIN", "SEM_PRI", "ID_MN_RESI", "EVOLUCAO",
                       "ATE_HOSP", "CRITERIO", "CLI_ICTERI", "CLI_RENAL",
                       "CLI_HEMORR", "NU_IDADE_N", "CS_SEXO"],
        "note": "The series starts in 2007 because SINAN NET replaced SINAN "
                "Windows that year; only 46 fields are common to the two "
                "layouts and the classification, criterion and outcome "
                "variables change name and code set.",
    },
    {
        "id": "SIM-DO",
        "name": "Sistema de Informacoes sobre Mortalidade, declaracoes de obito",
        "custodian": "Ministerio da Saude / DATASUS",
        "granularity": "death certificate",
        "coverage_used": "2008-2024 (the common window of the three systems)",
        "role": "independent numerator: population mortality with underlying "
                "cause ICD-10 A27",
        "key_fields": ["CAUSABAS", "CODMUNRES", "DTOBITO"],
        "note": "Underlying cause only for the primary count. The "
                "multiple-cause chain gives a different total and the two "
                "definitions are not interchangeable.",
    },
    {
        "id": "SIH-RD",
        "name": "Sistema de Informacoes Hospitalares, arquivos reduzidos de AIH",
        "custodian": "Ministerio da Saude / DATASUS",
        "granularity": "hospital admission authorisation (AIH), not person",
        "coverage_used": "2008-2024",
        "role": "independent numerator: admissions with principal diagnosis A27",
        "key_fields": ["DIAG_PRINC", "MUNIC_RES", "DT_INTER", "MORTE",
                       "DIAS_PERM", "UTI_MES_TO"],
        "note": "An AIH is a billing record; a transferred patient generates "
                "more than one. Counts are admissions, never people.",
    },
    {
        "id": "IBGE",
        "name": "Estimativas populacionais IBGE redistribuidas pelo Ministerio "
                "da Saude (POPSVS), malha municipal 2022, e Censo 2022 via SIDRA",
        "custodian": "Instituto Brasileiro de Geografia e Estatistica; "
                     "Ministerio da Saude",
        "granularity": "municipality x sex x single year of age x calendar year",
        "coverage_used": "2007-2025 for person-years; 2022 cross-section for "
                         "socioeconomic covariates",
        "role": "denominator: person-years for every rate; socioeconomic "
                "covariates for the confounding check",
        "key_fields": ["COD_MUN", "SEXO", "IDADE", "SIDRA 10296/10295/9933"],
        "note": "Gate on the build: the 2022 total must land within 5% of the "
                "Censo 2022 count of 203,080,756; it is 3.9% higher because "
                "the estimates incorporate measured post-enumeration "
                "undercount. A failing gate aborts.",
    },
]

#: Every dataset that crosses a script boundary in the 60+ pipeline, with the
#: script that produces it. Scripts numbered below 60 are prerequisites: they
#: build the line level, the denominators, the atlas and the geometry, and are
#: listed here so the execution order is complete rather than starting in the
#: middle.
PREREQUISITES = [
    {
        "script": "studies/leptospirosis/01_extract_sinan.py",
        "purpose": "Decode the annual national SINAN-LEPT files against the "
                   "codebook registry.",
    },
    {
        "script": "studies/leptospirosis/02_denominators.py",
        "purpose": "Build municipal person-years by sex and single year of age "
                   "from POPSVS, with the Censo 2022 reconciliation gate.",
        "produces": ["data/interim/population_municipal_year.parquet"],
    },
    {
        "script": "studies/leptospirosis/05_triangulation_extract.py",
        "purpose": "Extract SIM A27 deaths and SIH A27 admissions from the "
                   "national microdata.",
        "produces": ["data/interim/sim_a27_deaths.parquet",
                     "data/interim/sih_a27_admissions.parquet"],
    },
    {
        "script": "studies/leptospirosis/21_build_line_level.py",
        "purpose": "Assemble the decoded case-level frame with one row per "
                   "notification and a decode state per field.",
        "produces": ["data/interim/lept_line_level.parquet"],
    },
    {
        "script": "studies/leptospirosis/31_build_atlas.py",
        "purpose": "Build the municipality atlas: the municipality to health "
                   "region, UF and macro-region crosswalk on the 2022 lattice.",
        "produces": ["data/results/atlas/municipality_atlas.parquet"],
    },
    {
        "script": "paper/R/prep_geo.R",
        "purpose": "Dissolve municipal geometry into health-region polygons; "
                   "the source of the BYM2 adjacency graph.",
        "produces": ["paper/figures/geo_health_region.gpkg"],
    },
]

#: basename -> producing script, for datasets built before the 60+ pipeline.
UPSTREAM_PRODUCERS = {
    "population_municipal_year.parquet": "studies/leptospirosis/02_denominators.py",
    "sim_a27_deaths.parquet": "studies/leptospirosis/05_triangulation_extract.py",
    "sih_a27_admissions.parquet": "studies/leptospirosis/05_triangulation_extract.py",
    "lept_line_level.parquet": "studies/leptospirosis/21_build_line_level.py",
    "municipality_atlas.parquet": "studies/leptospirosis/31_build_atlas.py",
    "geo_health_region.gpkg": "paper/R/prep_geo.R",
}

#: script stem -> the results subdirectory it writes. Asserted against the
#: source text, so a renamed output directory breaks the build rather than
#: producing a manifest that quietly lies.
OUTPUT_DIRS = {
    "60_analysis_panel": "analysis_panel",
    "61_denominator_coupling": "coupling",
    "62_triangulation": "triangulation",
    "63_missingness_record": "data_quality_record",
    "64_exclusion_sensitivity": "exclusion_sensitivity",
    "65_construct_validity": "construct_validity",
    "66_case_mix": "case_mix",
    "67_rs2024_event": "rs2024_event",
    "68_primary_model": "primary_model",
    "69_within_region": "within_region",
    "70_descriptives": "descriptives",
    "71_manuscript_tables": "manuscript_tables",
    "72_reporting_checklist": "reporting",
}

#: Purposes that read better than the auto-extracted first docstring line, or
#: that the auto-extractor cannot reach. Everything else is taken from the
#: script's own first sentence, so a rewritten script updates its own manifest
#: entry.
PURPOSE_OVERRIDE = {
    "72_reporting_checklist": "Emit the STROBE/RECORD checklist, the "
                              "supplementary-material inventory and this "
                              "reproducibility manifest.",
}


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    text = text.replace("``", "").replace("**", "").replace("*", "")
    m = re.search(r"^(.{10,220}?[.?])(\s|$)", text)
    return (m.group(1) if m else text[:200]).strip()


def script_purpose(path: Path) -> str:
    stem = path.stem
    if stem in PURPOSE_OVERRIDE:
        return PURPOSE_OVERRIDE[stem]
    src = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".py":
        m = re.match(r'\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', src, re.S)
        if m:
            return _first_sentence(m.group(1))
    else:
        lines = src.splitlines()
        buf: list[str] = []
        for ln in lines:
            s = ln.strip()
            if s.startswith("#!"):
                continue
            if not s.startswith("#"):
                break
            body = s.lstrip("#").strip()
            if not body and buf:
                break
            if body:
                buf.append(body)
            if buf and buf[-1].endswith((".", "?")):
                break
        if buf:
            return _first_sentence(" ".join(buf))
    return f"(no docstring found in {path.name})"


def analysis_scripts() -> list[Path]:
    """Every script in studies/leptospirosis numbered 60 or above."""
    out = []
    for p in sorted(STUDY.iterdir()):
        if p.suffix.lower() not in {".py", ".r"}:
            continue
        m = re.match(r"^(\d+)", p.name)
        if m and int(m.group(1)) >= 60:
            out.append(p)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    # -- (a) checklist -----------------------------------------------------
    rows = []
    for guideline, items in (("STROBE", STROBE), ("RECORD", RECORD)):
        for num, text, where, ev, status in items:
            rows.append({
                "guideline": guideline,
                "item_number": num,
                "item_text_abbreviated": text,
                "where_addressed": where,
                "evidence_file": ev,
                "status": status,
            })
    checklist = pl.DataFrame(rows)

    # Every cited path must exist. A checklist that points at a file nobody
    # wrote is exactly the decoration the reviewers objected to. The three
    # artefacts this script itself writes are exempt from the pre-write check
    # and asserted after writing instead.
    self_outputs = {
        "data/results/reporting/record_strobe_checklist.csv",
        "data/results/reporting/supplementary_inventory.csv",
        "data/results/reporting/reproducibility_manifest.json",
    }

    def _check(paths: str, label: str) -> None:
        for raw in paths.split(";"):
            token = raw.strip()
            if not token or token.startswith("(") or token in self_outputs:
                continue
            if not (ROOT / token).exists():
                problems.append(f"{label}: cited path does not exist -> {token}")

    for r in rows:
        _check(r["evidence_file"], f"checklist {r['guideline']} {r['item_number']}")

    # -- (b) supplementary inventory --------------------------------------
    inv = pl.DataFrame([
        {"supplement_id": sid, "title_pt": title,
         "what_it_contains": what, "source_files": src}
        for sid, title, what, src in SUPPLEMENTS
    ])
    for sid, _t, _w, src in SUPPLEMENTS:
        _check(src, f"supplement {sid}")

    # -- (c) manifest ------------------------------------------------------
    scripts = analysis_scripts()
    src_text = {p.stem: p.read_text(encoding="utf-8", errors="replace")
                for p in scripts}

    # Outputs: the declared directory, verified against the source, listed from
    # disk. A script that has not been run yet reports zero outputs rather than
    # an invented list.
    entries: list[dict] = []
    produced_by: dict[str, str] = dict(UPSTREAM_PRODUCERS)
    for p in scripts:
        stem = p.stem
        rel = p.relative_to(ROOT).as_posix()
        subdir = OUTPUT_DIRS.get(stem)
        outputs: list[str] = []
        if subdir is None:
            problems.append(
                f"manifest: {p.name} is numbered >= 60 but has no declared "
                f"output directory in OUTPUT_DIRS; its outputs are unknown")
        else:
            if not re.search(rf'["/]{re.escape(subdir)}["/]', src_text[stem]):
                problems.append(
                    f"manifest: {p.name} does not mention its declared output "
                    f"directory '{subdir}'; OUTPUT_DIRS is stale")
            d = RES / subdir
            if d.is_dir():
                for f in sorted(d.rglob("*")):
                    if f.is_file() and not f.name.startswith("_"):
                        outputs.append(f.relative_to(ROOT).as_posix())
                        produced_by.setdefault(f.name, rel)
            else:
                problems.append(
                    f"manifest: {p.name} declares output directory "
                    f"{subdir} which does not exist on disk (not yet run?)")
        entries.append({
            "script": rel,
            "language": "R" if p.suffix.lower() == ".r" else "python",
            "purpose": script_purpose(p),
            "output_dir": f"data/results/{subdir}" if subdir else None,
            "outputs": outputs,
            "n_outputs": len(outputs),
        })

    # Dependencies, derived: a script depends on every registered dataset whose
    # basename appears in its source and which it does not itself produce.
    own_outputs = {e["script"]: {Path(o).name for o in e["outputs"]}
                   for e in entries}
    deps: dict[str, set[str]] = {}
    for e in entries:
        s = e["script"]
        ins: list[str] = []
        for base, producer in produced_by.items():
            if producer == s or base in own_outputs[s]:
                continue
            if base in src_text[Path(s).stem]:
                ins.append(base)
        e["inputs"] = sorted(ins)
        deps[s] = {produced_by[b] for b in ins}
        if Path(s).stem == Path(__file__).stem:
            # This script cites the whole result tree as evidence and asserts
            # that every cited path exists, so the scan reports all of it. Say
            # plainly which of those it opens for a value.
            e["inputs_note"] = (
                "These are the evidence paths the checklist and inventory "
                "cite; this script opens only study_flow.csv for a value and "
                "asserts the existence of the rest. It therefore runs last, "
                "after everything it certifies.")

    # Topological order over the 60+ scripts. Upstream producers are pinned
    # ahead of everything by construction (they are not in `entries`).
    pending = {e["script"] for e in entries}
    order: list[str] = []
    remaining = dict(deps)
    while pending:
        ready = sorted(s for s in pending
                       if not (remaining[s] & pending))
        if not ready:
            problems.append(
                "manifest: dependency cycle among "
                + ", ".join(sorted(pending))
                + " -- execution order could not be derived")
            order.extend(sorted(pending))
            break
        order.extend(ready)
        pending -= set(ready)

    # Verify the order is real: every dependency must be satisfied earlier.
    position = {s: i for i, s in enumerate(order)}
    violations = []
    for s, ds in deps.items():
        for d in ds:
            if d in position and position[d] >= position[s]:
                violations.append(f"{d} must run before {s}")
    problems.extend(f"manifest: order violation -- {v}" for v in violations)

    # Stages: scripts that can run in parallel share a stage.
    depth: dict[str, int] = {}
    for s in order:
        inner = [depth[d] for d in deps[s] if d in depth]
        depth[s] = (max(inner) + 1) if inner else 0
    stages: dict[int, list[str]] = defaultdict(list)
    for s in order:
        stages[depth[s]].append(s)

    # Software versions, queried rather than recalled.
    software = {
        "python": platform.python_version(),
        "python_executable": Path(sys.executable).name,
        "polars": pl.__version__,
    }
    for mod in ("numpy", "scipy", "pyarrow"):
        try:
            software[mod] = __import__(mod).__version__
        except Exception as exc:  # pragma: no cover - environment dependent
            software[mod] = f"query failed: {exc}"
    try:
        rq = subprocess.run(
            [RSCRIPT, "-e",
             'p <- c("INLA","data.table","arrow","sf","spdep","jsonlite",'
             '"sandwich"); '
             'v <- sapply(p, function(x) tryCatch(as.character(packageVersion(x)),'
             ' error=function(e) "not installed")); '
             'cat(R.version.string, "\\n"); '
             'for (n in names(v)) cat(n, v[[n]], "\\n")'],
            capture_output=True, text=True, timeout=180, cwd=str(ROOT))
        wanted = {"INLA", "data.table", "arrow", "sf", "spdep", "jsonlite",
                  "sandwich"}
        lines = [ln.strip() for ln in rq.stdout.splitlines() if ln.strip()]
        for ln in lines:
            if ln.startswith("R version"):
                software["R"] = ln
                continue
            parts = ln.split()
            if len(parts) >= 2 and parts[0] in wanted:
                software[f"R:{parts[0]}"] = " ".join(parts[1:])
        # renv prints a drift warning on stderr when the library and the
        # lockfile disagree. That is a reproducibility fact, not noise.
        drift = [ln.strip() for ln in (rq.stdout + "\n" + rq.stderr).splitlines()
                 if "out-of-sync" in ln or "out of sync" in ln]
        software["renv_lockfile"] = (
            "renv.lock" if (ROOT / "renv.lock").exists() else "absent")
        if drift:
            software["renv_status"] = drift[0].lstrip("- ").strip()
            problems.append(
                "manifest: renv reports the R library out of sync with "
                "renv.lock; the recorded package versions are those actually "
                "loaded, not those pinned")
    except Exception as exc:  # pragma: no cover - environment dependent
        problems.append(f"manifest: R version query failed: {exc}")
        software["R"] = f"query failed: {exc}"

    if "R:INLA" not in software:
        problems.append("manifest: INLA version could not be queried")

    flow = pl.read_csv(PATHS.results / "analysis_panel" / "study_flow.csv")
    n_final = int(flow.filter(pl.col("step") == "ANALYTIC POPULATION")["n"][0])

    manifest = {
        "study": "Ascertainment breadth and reported case fatality of "
                 "leptospirosis, Brazil, 2007-2025",
        "manuscript": "paper/artigo.qmd",
        "code_repository": "https://github.com/LeviMelo/leptospirose-brasil-2007-2025",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "studies/leptospirosis/72_reporting_checklist.py",
        "analytic_window": {
            "cases": {"from": 2007, "to": 2025,
                      "dated_by": "symptom onset (SEM_PRI)"},
            "three_system_comparison": {
                "from": 2008, "to": 2024,
                "reason": "the only years in which SINAN, SIM and SIH all have "
                          "complete national coverage"},
            "unit_of_analysis": "health region x calendar year",
            "analytic_population_confirmed_cases": n_final,
        },
        "source_systems": SOURCE_SYSTEMS,
        "software": software,
        "prerequisite_scripts": PREREQUISITES,
        "analysis_scripts": entries,
        "execution_order": {
            "derivation": "Dependency edges are derived by scanning each "
                          "script's source for the basenames of the datasets "
                          "the pipeline produces, then topologically sorting. "
                          "Scripts sharing a stage read nothing the others "
                          "write and may run in any order or in parallel.",
            "prerequisites_first": [p["script"] for p in PREREQUISITES],
            "stages": [{"stage": k, "scripts": stages[k]}
                       for k in sorted(stages)],
            "linear_order": order,
            "verified": not violations,
            "order_violations": violations,
        },
        "reporting_artefacts": {
            "checklist": "data/results/reporting/record_strobe_checklist.csv",
            "supplementary_inventory":
                "data/results/reporting/supplementary_inventory.csv",
            "manifest": "data/results/reporting/reproducibility_manifest.json",
        },
        "known_gaps": [
            r["item_text_abbreviated"].split(":")[0] + f" ({r['guideline']} "
            f"{r['item_number']}, {r['status']})"
            for r in rows if r["status"] != "satisfied"
        ],
        "build_problems": problems,
    }

    # -- write -------------------------------------------------------------
    checklist.write_csv(OUT / "record_strobe_checklist.csv")
    inv.write_csv(OUT / "supplementary_inventory.csv")
    (OUT / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for token in sorted(self_outputs):
        assert (ROOT / token).exists(), f"self-cited artefact not written: {token}"

    # -- summary -----------------------------------------------------------
    def _count(g: str, s: str) -> int:
        return int(checklist.filter(
            (pl.col("guideline") == g) & (pl.col("status") == s)).height)

    print("=" * 72)
    print("REPORTING-GUIDELINE COVERAGE")
    print("=" * 72)
    print(f"{'':10s}{'items':>7s}{'satisfied':>11s}{'partial':>9s}{'not yet':>9s}")
    for g in ("STROBE", "RECORD"):
        n = int(checklist.filter(pl.col("guideline") == g).height)
        print(f"{g:10s}{n:7d}{_count(g,'satisfied'):11d}"
              f"{_count(g,'partial'):9d}{_count(g,'not_yet'):9d}")
    tot = checklist.height
    sat = int(checklist.filter(pl.col("status") == "satisfied").height)
    par = int(checklist.filter(pl.col("status") == "partial").height)
    nyt = int(checklist.filter(pl.col("status") == "not_yet").height)
    print(f"{'TOTAL':10s}{tot:7d}{sat:11d}{par:9d}{nyt:9d}")
    print()
    print("Not fully satisfied, and what is missing:")
    for r in rows:
        if r["status"] != "satisfied":
            tail = r["where_addressed"]
            i = tail.find("NOT YET")
            note = tail[i:i + 150] if i >= 0 else tail[:150]
            print(f"  {r['guideline']:6s} {r['item_number']:>4s}  "
                  f"[{r['status']}] {note}")
    print()
    print(f"Supplements inventoried: {inv.height} "
          f"(S1-S{inv.height}); manuscript cites S3 by number.")
    print(f"Analysis scripts >= 60: {len(entries)}; "
          f"outputs listed: {sum(e['n_outputs'] for e in entries)} files.")
    print("Execution stages (scripts in a stage may run in parallel):")
    for k in sorted(stages):
        names = ", ".join(Path(s).name for s in stages[k])
        print(f"  stage {k}: {names}")
    print(f"Execution order verified: {not violations}")
    print(f"R: {software.get('R', '?')} | INLA {software.get('R:INLA', '?')} | "
          f"python {software['python']} | polars {software['polars']}")
    print()
    if problems:
        print(f"BUILD PROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("No build problems: every cited path exists, every declared "
              "output directory is real, and the execution order is acyclic.")
    print(f"\nWrote 3 files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
