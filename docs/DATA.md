# Data sources, vintages and known defects

**Study:** leptospirosis in Brazil, 2007–2025
**Scope of this document:** every dataset the study reads, where it comes from,
what it measures, how it is keyed, what is wrong with it, and what was done
about that.

This is the reference a reviewer, a co-author or a future maintainer should be
able to read *instead of* the code. Where a number appears here it was measured
from the artefacts on disk, not recalled; where a defect is described it is
because the defect was observed, not anticipated.

Two companion documents carry the parts that do not belong here:
`ADR-001-versioned-geographies.md` for how territorial change is handled, and
`CODEBOOK.md` for how coded values are translated.

---

## 0. Reading the layout

```
data/
  cache/      raw bytes exactly as retrieved, content-addressed, immutable
  derived/    decoded + code-translated source files, cache-keyed on inputs
  interim/    study-specific intermediates (line level, tensors, extracts)
  panel/      analysis-ready panels on the municipality-month spine
  results/    one directory per analysis stage
  export/     versioned manuscript bundles with manifests
```

`cache/` is never edited. Every file carries a sibling `.provenance.json` with
the retrieval URL, the HTTP `Last-Modified`, a SHA-256 and the fetch timestamp.
DATASUS files are mutable at source — the agency reposts a year without notice
— so the hash is the only durable identifier and is what a release points at.

---

## 1. Outcome: SINAN leptospirosis

| | |
|---|---|
| **Source** | Sistema de Informação de Agravos de Notificação (SINAN NET), Ministério da Saúde |
| **Access** | `ftp.datasus.gov.br/dissemin/publicos/SINAN/DADOS/{FINAIS,PRELIM}` |
| **Files** | `LEPTBR{YY}.dbc`, one per year, national |
| **Vintage** | Final releases 2007–2024; **2025 is preliminary** |
| **Cached** | 21 files, 30 MB, manifest `_manifest_sinan_LEPT_20260730.json` |
| **Grain** | One row per notification |
| **Volume** | **328,984 notifications**, 2007–2025, 120 raw columns |

### Case definition

`CLASSI_FIN == "1"` (*confirmado*). Measured composition of the full extract:

| Classification | n | % |
|---|---:|---:|
| Confirmed | 66,667 | 20.3 |
| Discarded | 238,353 | 72.4 |
| Inconclusive | 11,699 | 3.6 |
| Unclassified (field blank) | 12,265 | 3.7 |

The confirmed-to-notified ratio is not a nuisance: it is the quantity RQ3
interrogates, so the denominator categories are carried through the analysis
rather than filtered away at extraction.

### Timing and geography

- **Event time** is `DT_SIN_PRI` (first symptoms), not notification date.
  Notification month would shift the series by the reporting delay and would
  make the delay's own trend look like an epidemiological trend.
- **Event place** is municipality of probable infection (`COMUNINF`) where
  valid on the dated lattice, with residence (`ID_MN_RESI`) as the ordered
  fallback. 66,516 in-window confirmed records resolve as 56,330 by infection
  municipality, 10,150 by residence fallback and **36 unresolved**, which are
  retained in a named stratum rather than dropped.
- SINAN writes **six-digit** municipality codes (no check digit); IBGE and the
  panels use seven. The lattice carries both, so the mapping is exact.

### Known defects

1. **Outcome completeness drifts severely.** `EVOLUCAO` (case outcome) is valid
   for 45.3% of records in 2007 and 70.5% in 2025 — a 52.8 percentage-point
   swing, the largest of any field. Any case-fatality *time series* is
   confounded by this. RQ4 adjusts for the cell's completeness share; nothing
   else may report a CFR trend without doing likewise. (`ISSUE_LEDGER` BREPI-021.)
2. **Three fields are too incomplete to stratify on.** `TPAUTOCTO` 99.0%
   missing, `CON_AMBIEN` 79.1%, `DOENCA_TRA` 74.7%. `ID_OCUPA_N` is 50.2%
   missing plus 17.7% explicit-unknown, leaving 32.1% usable — which is why
   the occupational analysis is descriptive and not a rate model. (BREPI-022.)
3. **The 2020–2021 trough is a surveillance artefact, not an epidemiological
   one.** Notifications fall from ~16–21k/year to 10,438 and 8,962. Excluding
   the period is a prespecified RQ1 sensitivity and moves the primary estimate
   by 0.6%.
4. **2025 is preliminary** and subject to upward revision. Onset-to-notification
   delay (median 7 days, p95 34) shows the right edge is materially complete
   within about a month, which is what makes 2025 quotable at all — but it is
   flagged separately in every table.
5. **The published data dictionary (v5.0) does not match the files.** See
   `CODEBOOK.md`; the curated registry is authoritative and the PDF is a
   cross-check, because the PDF's own text layer is lossy.

### Derived artefacts

- `data/derived/sinan/lept/*.parquet` — one per year, decoded and
  code-translated, content-addressed on the source bytes plus the codebook
  version. A rerun costs a parquet read.
- `data/interim/lept_line_level.parquet` — 328,984 × 293, all years, the raw
  fields the study declares plus **every** column the codebook emits.

---

## 2. Triangulation outcomes: SIM and SIH

### SIM — mortality

| | |
|---|---|
| **Source** | Sistema de Informações sobre Mortalidade |
| **Access** | `ftp.datasus.gov.br/.../SIM/CID10/DORES/DO{UF}{YYYY}.dbc` |
| **Cached** | 486 files, 1.93 GB (state × year) |
| **Selection** | `CAUSABAS` in the A27 block (leptospirosis) |
| **Extract** | `data/interim/sim_a27_deaths.parquet`, **5,705 deaths** |

Underlying cause only. A death with leptospirosis as an associated cause is not
counted, which makes this a *lower* bound and is the conservative direction for
a burden-correction argument.

### SIH — hospital admissions

| | |
|---|---|
| **Source** | Sistema de Informações Hospitalares, AIH reduced files |
| **Access** | `ftp.datasus.gov.br/.../SIH/200801_/Dados/RD{UF}{YYMM}.dbc` |
| **Cached** | 1,298 files, 3.01 GB (state × month) |
| **Selection** | `DIAG_PRINC` in the A27 block |
| **Extract** | `data/interim/sih_a27_admissions.parquet`, **36,122 admissions**, 2008–2024 |
| **Coverage** | **2008 onward** — the SIH series begins 2008-01, so 2007 has no admissions data |

SIH counts **admissions, not people**: a transfer or a readmission produces a
second AIH. Any patient-level statement requires deduplication that the reduced
file does not support, so SIH is used as an *event* series.

### The linkage assumption, stated

SINAN, SIH and SIM are **not** linkable at the individual level in these public
files — there is no common person identifier. Every triangulation statement is
therefore about *system-level* overlap under declared assumptions, and
capture–recapture is only applied where independence is defensible and is
reported with its identification sensitivity. This is a limitation of the data,
not of the method, and is stated wherever a corrected burden appears.

---

## 3. Denominators: population

| | |
|---|---|
| **Sources** | IBGE censuses 2010 and 2022; intercensal municipal estimates; SIDRA age–sex structure |
| **Cached** | `data/cache/datasus/ibge_pop` (19 files), SIDRA tables |
| **Artefacts** | `population_municipal_year.parquet` (105,831 rows), `population_tensor_long.parquet` (3,598,254 rows) |
| **Tensor grain** | municipality × year × sex × 17 age groups (`00-04` … `80+`) |

The tensor is built by **iterative proportional fitting (raking)** of a census
age–sex seed to the official municipal totals, not by a cohort-component
projection. Raking guarantees the margins the study actually needs — municipal
totals match the official series exactly — and does not pretend to demographic
detail the inputs cannot support.

**Consequences to remember:**
- The tensor tops out at `80+` while the WHO World Standard runs to `85+`.
  `rebase_standard()` collapses the standard onto the tensor's grouping and
  conserves total weight; dropping the tail or matching labels textually would
  both bias the result.
- Person-time is the mid-year stock × 12 for a year, or × 1 for a month. This
  is the same offset the models use, so a descriptive rate and a fitted rate
  are on one scale.

---

## 4. Exposure: precipitation

Two products, spliced, because neither covers the period alone.

| Product | Role | Period used | Resolution | Cached |
|---|---|---|---|---|
| **BR-DWGD** (Xavier et al.) | primary | 2007-01 … **2024-03-20** | 0.1°, daily, gauge-interpolated | 2.16 GB |
| **ERA5-Land** (Copernicus) | tail | 2024-03-21 … 2025-12 | 0.1°, reanalysis | 3.57 GB |

Municipal values are **zonal means over the 2022 municipal mesh**, with derived
extremes: `rx1day` (wettest day), and counts of days above 1, 10, 20 and 50 mm.

### Known defects — all flagged in-panel, none silent

| Defect | Flag column | Extent |
|---|---|---|
| Product splice; March 2024 is a mixed-product month | `climate_product` | 1 month |
| ERA5-Land daily subset omits one day per month; totals calendar-prorated | `precip_prorated` | ERA5 period |
| Five municipalities created in 2013 absent from the source lattice; parent covariates used | `territorial_imputed` | 5 of 5,570 |
| Incomplete-month coverage | `climate_coverage`, `precip_missing_days` | per cell |

An unspliced sensitivity ending 2023-12 is available and is the check that the
splice is not driving the result. The five lineage-proxy municipalities are
`1504752`, `4212650`, `4220000`, `4314548`, `5006275` (BREPI-006);
the proration is BREPI-007.

### Exposure scale — the thing most easily misquoted

The RQ1 cross-basis uses a **unit-relative** exposure:
`(x − unit median) / unit IQR`. So a reported rate ratio is **per local IQR
above the local median**, *not* per millimetre. The health-region monthly IQR
is itself heterogeneous — p10 81 mm, p50 126 mm, p90 211 mm — which is exactly
why an absolute scale would compare an Amazonian wet month with a semi-arid one
and call them the same exposure. Every figure caption and table header must
carry the unit. (BREPI-014.)

---

## 5. Structural covariates: SIDRA / IBGE

| Variable | Source tables | Anchors | Interpolation |
|---|---|---|---|
| Sewer coverage | **1394** (2010), **6805** (2022) | 11,140 | nearest anchor 2007–09; linear 2010–22; nearest 2022–25 |
| Urban share | 202 (2010), 4709 (2022) | 11,140 | as above |
| Real GDP per capita | municipal accounts 2007–2023 + national deflator | 94,690 | as above; 2024–25 held at 2023 |

Tables **3154 and 9860 were rejected** for sanitation: their universes do not
match the estimand (share of permanent private households connected to the
public sewer or stormwater network). Margin checks against published totals
pass for both retained census years with **0 mismatches**.

**The comparability caveat that matters:** the 2010 and 2022 census questionnaire
wording for sewerage differs. The estimand is held constant and the wording
difference is carried as a documented sensitivity rather than assumed away.

**Identification caveat:** these are near-time-invariant municipal
characteristics competing with a BYM2 spatial field. Their *main effects* move
substantially with the spatial specification (`sanitation_sewer_share` from
−1.64 with no spatial field to −0.24 with one) and are reported as **not
separately identified**. The prespecified structural estimand is the
cross-basis × sanitation *interaction*, which is within-unit identified.
(BREPI-016.)

---

## 6. Treatment: disaster declarations

| | |
|---|---|
| **Source** | Atlas Digital de Desastres / S2iD, Ministério da Integração |
| **Cached** | 2 files, 0.09 GB |
| **Classification** | COBRADE |
| **Artefacts** | `flood_events.parquet` (21,644), `drought_events.parquet` (22,665), `flood_declarations.parquet` (984,504 municipality-months), `flood_first_treatment.parquet` (4,318 ever-treated municipalities) |

### The COBRADE selection, and why the obvious choice is wrong

The intuitive filter is the `1.2.x` hydrological block (floods, flash floods,
inundations). Applying it **drops the 2024 Rio Grande do Sul catastrophe from
467 municipalities to 47**, because 465 of 533 RS records were filed under
`1.3.2.1.4`. Declaration codes reflect the filing municipality's choice as much
as the hazard. The catalogue therefore uses the broader flood-related set and
reports the strict-code restriction as a sensitivity, not the reverse.

**Recurrence is the dominant design problem:** 73.3% of treated municipalities
are treated more than once, so an absorbing "first treatment" design discards
most of the variation. Both an absorbing and an episodic specification are
estimated and reported side by side.

Droughts are carried as a **placebo**: a drought declaration is an
administrative event of similar salience with no plausible leptospirosis
mechanism, so an effect there indicts the design.

---

## 7. Geography

| | |
|---|---|
| **Mesh** | IBGE 2022 municipal mesh, 5,570 municipalities |
| **Lattice API** | IBGE localidades, filtered to a dated lattice via `municipality_changes.yaml` |
| **Health regions** | 439 (DATASUS/CIR resolutions), contracted graph |
| **Graphs** | `data/panel/graphs/{municipality_all,municipality_endemic,health_region,immediate_region,microregion,uf}.adj` |

Health regions are a **SUS administrative division revised without a public
changelog**, so they are attached from a dated DATASUS extract rather than
inferred. Territorial change is handled as transfer algebra with conservation
invariants — extensive quantities conserve mass, intensive fields conserve
support-weighted means — never as join cleanup. See `ADR-001`.

---

## 8. ENSO

| | |
|---|---|
| **Source** | NOAA CPC — ONI (3-month running) and monthly Niño 3.4 (ERSSTv5, 1991–2020 base) |
| **Cached** | `data/cache/climate/enso/{oni.ascii.txt,nino34_monthly.ascii}` |
| **Coverage** | 1950 onward, so the whole study period with lags |
| **Role** | National temporal climate modulator; RQ1 sensitivity and heterogeneity |

ENSO is a **national** covariate: it varies in time only. It therefore cannot
be adjusted for alongside an unrestricted national temporal random walk — they
are the same degrees of freedom — and enters as a *modulator* of the rainfall
response rather than as an additive control.

---

## 9. Acquired but not used in a reported analysis

Honesty about what was fetched and then not used matters as much as
documenting what was.

| Source | Cached | Status |
|---|---|---|
| **CNES** establishments | 314 files, 0.07 GB | A 2024 benchmark adapter exists. Historical CNES is facility-row based with unstable taxonomy and geography; it is **optional** in the protocol and is not on the critical path. (BREPI-011.) |

## 10. Prescribed by the protocol, not acquired

| Source | Intended role | Why not |
|---|---|---|
| **MapBiomas** land use | RQ5 regime definition, effect heterogeneity | Large raster/area-statistic acquisition. RQ5 uses a **declared reduced typology** on burden, seasonality and structural features instead — which the protocol's own completion gate permits — and the reduction is stated in the results. |
| **Surface-water extent** | RQ2 hydrological dose | Same acquisition cost; RQ2 uses declaration events and rainfall as dose. |
| **ANA / HidroWeb** river stage | Hydrological validation | Station network is sparse relative to 5,570 municipalities; would support local sensitivity only. |
| **CEMADEN** high-frequency rainfall | Event validation | Coverage begins mid-period and is concentrated in monitored risk areas, so it cannot validate the national panel uniformly. |
| **Social vulnerability indices** | Regime description | GDP per capita, urban share and sanitation already carry the structural block; adding a composite index of the same census inputs would not be independent information. |

Each of these is a *stated omission with a reason*, not an oversight. Where an
omission bounds a conclusion, the bound is reported with the conclusion.

---

## 11. Panels

| Panel | Rows | Cols | Content |
|---|---:|---:|---|
| `lept_panel_municipality_month` | 1,269,960 | 26 | outcome + denominators on the complete spine |
| `lept_panel_rq1_municipality_month` | 1,269,960 | 41 | + climate |
| `lept_panel_rq1_structural_municipality_month` | 1,269,960 | 50 | + sanitation |
| `lept_panel_rq1_socioeconomic_municipality_month` | 1,269,960 | 69 | + urban share, GDP — **the analysis panel** |

5,570 municipalities × 228 months = 1,269,960 cells, with **no missing cells at
any stage**. A covariate that cannot be supplied for a cell is flagged, never
dropped: a source failure on the primary path stops the build rather than
silently reducing the population.

### The sparsity that drives the design

- 97.2% of municipality-months are zero
- median count among non-zero cells is **1**
- 2,106 of 5,570 municipalities never report a confirmed case
- Gini of case counts across municipalities **0.889**
- half the national case load arises in municipalities holding **12.3%** of the population

This is why the primary RQ1 fitting tier is health region × month (439 units)
rather than municipality: a distributed-lag surface is not identified against a
97% zero outcome. Data resolution and model resolution are deliberately
separate.

---

## 12. Reproducing the data layer

```bash
python studies/leptospirosis/00_build_geography.py
python studies/leptospirosis/01_extract_sinan.py
python studies/leptospirosis/02_denominators.py
python studies/leptospirosis/03_assemble_panel.py
python studies/leptospirosis/07_build_climate_panel.py
python studies/leptospirosis/08_build_sanitation_panel.py
python studies/leptospirosis/09_build_socioeconomic_panel.py
python studies/leptospirosis/14_build_disaster_panel.py
python studies/leptospirosis/21_build_line_level.py
```

Every stage is idempotent and cache-backed. With a warm cache the whole data
layer rebuilds without touching the network; the SINAN line level rebuilds from
19 cached `.dbc` files in **8.5 seconds**.

---

## 12b. The denominator is an estimate, not the census count — deliberately

This is the single most misreadable number in the study, so it is stated here
rather than left to be rediscovered.

| 2022 population | Value | What it is |
|---|---:|---|
| Censo 2022, first release (Jun 2023) | 203,062,512 | raw enumeration |
| Censo 2022, revised (Aug 2023) | 203,080,756 | enumeration after >12% of municipalities were corrected |
| **This study's denominator** | **210,862,983** | IBGE *Projeções e Estimativas, Revisão 2024*, via POPSVS |

The study's figure is **3.9% above the enumeration, and that is correct.** IBGE
raised the estimate in August 2024 because its **Pesquisa de Pós-Enumeração**
measured undercount in the census, and because vital-registration and migration
records — chiefly the international migration balance — estimate the population
better than the enumeration did. The census is a *count*; the population
estimate is an *estimate*, and a rate denominator wants the estimate.

The gap is not confined to 2022. Against the 2010 census the same series runs
**+2.09%**, and the excess grows across the window. That is the expected
signature of a projection series carrying an undercount correction that itself
grows, not of drift or error.

**Why POPSVS specifically.** `brepi/denominators/tensor.py` takes its margins
from POPSVS — the Ministry of Health's own denominators — so an incidence rate
computed here is numerically comparable to one published in a *Boletim
Epidemiológico*. A denominator that is privately better but publicly
incomparable is worse for an epidemiological paper.

**Consequences that must travel with any rate.**

- Methods must name the denominator as IBGE estimates via POPSVS, consistent
  with Revisão 2024, **not** the census enumeration, and give both 2022 values.
- Rate *levels* are not comparable with studies that used raw census counts —
  including Galan et al. 2021, which used the 2010 census and therefore reports
  rates on a denominator about 2% smaller.
- Rates on the raw enumeration would be about **3.9% higher** in 2022 and 2.1%
  higher in 2010. Nothing *within* the series changes: every territory and year
  uses one source, so no ranking, trend or contrast is affected.

## 13. Cached footprint, measured

Measured from `data/cache/` on 2026-08-14, excluding `.provenance.json`
sidecars. This is what a cold rebuild would have to re-fetch.

| Cache group | Files | GB |
|---|---:|---:|
| `datasus/sih` | 5,508 | 14.527 |
| `climate/era5land` | 10 | 3.575 |
| `climate/brdwgd` | 1 | 2.164 |
| `datasus/sim` | 486 | 1.935 |
| `ibge/meshes` | 1 | 0.204 |
| `disasters/atlas` | 2 | 0.086 |
| `datasus/cnes` | 314 | 0.074 |
| `datasus/ibge_pop` | 19 | 0.066 |
| `sidra/values` | 222 | 0.055 |
| `datasus/sinan` | 23 | 0.028 |
| `sidra/localidades` | 32 | 0.011 |
| everything else (geo, catalog, metadata, ENSO, manifests) | 79 | 0.007 |
| **Total** | **6,697** | **22.734** |

SIH dominates by two orders of magnitude because it is one file per state-month
across seventeen years. It is also the source with the most expensive first
decode (BREPI-018).

---

## 14. The DATASUS FTP compendium — a reference database, not a study input

`docs/sources/reference/datasus_compendium.sqlite` (55 MB) is a **complete scan
of `ftp.datasus.gov.br/dissemin/publicos`**, and it is deliberately broader than
this study. Nothing in the analysis reads it. It exists to answer, before any
adapter is written, three questions that are otherwise answered by trial and
error against a slow FTP: *what does DATASUS actually publish, in what file
families, and does the record layout change across the years I want?*

**Scan provenance.** Host `ftp.datasus.gov.br`, base path `/dissemin/publicos`,
303 directories walked, **124,810 files** and 334 directories inventoried.

**Coverage by subsystem** (file counts, whole FTP, not this study):

| System | Files | | System | Files |
|---|---:|---|---|---:|
| CNES | 54,645 | | SIM | 1,487 |
| SIHSUS | 39,021 | | SINAN | 1,105 |
| SISCAN | 10,873 | | SISPRENATAL | 945 |
| SIASUS | 6,116 | | SINASC | 901 |
| CIHA | 4,747 | | CIH | 871 |
| PNI | 1,591 | | DADOS_ABERTOS | 1,542 |

**Tables and what each one answers.**

| Table | Rows | Answers |
|---|---:|---|
| `inventory_files` | 124,810 | Every file on the FTP: path, extension, format family, inferred system, series prefix, geo code |
| `families` | 233 | File families grouped by prefix and partition scheme, with the observed time range and file count — *"does LEPTBR exist for the years I need?"* |
| `variable_profiles` | 12,487 | Per family-file-field: physical type, width, decimals, non-null and null counts, missingness percent |
| `value_frequencies` | 104,084 | Observed value distributions per field — the empirical complement to the published dictionary, and how untranslated code systems get discovered |
| `schema_presence` | 12,487 | Which fields appear in which schema signature, with field order |
| `schema_drift` | 233 | Per family: how many distinct schema signatures exist, the union field count, and how many fields are *sometimes* present — **the table that predicts a naive vertical concat failing** |
| `table_profiles` | 253 | Profiled files: format, size, rows profiled, column count, field names |
| `sample_plan` | 233 | Which file was profiled per family and why |
| `scan_summary` | 1 | Host, base path, method, counts, start and finish times |
| `listing_benchmark` | 6 | FTP listing method timings — why the scanner uses the method it does |
| `warnings` | 66 | Files and families where profiling failed or was partial |

**Why it earns its place in a study document.** The SINAN Windows → SINAN NET
break that fixes this study's window at 2007 is exactly a `schema_drift` fact:
two schema signatures for one series prefix, with most fields only *sometimes*
present. The compendium is where that class of problem is visible before it
becomes a silently mis-concatenated column. See `CODEBOOK.md` for how the
values are then translated, and `sources/DATASUS_COMPENDIUM.md` for the
narrative description of the subsystems.

**Querying it.**

```bash
python -c "
import sqlite3, pandas as pd
c = sqlite3.connect('docs/sources/reference/datasus_compendium.sqlite')
print(pd.read_sql('''
  select family_id, series_prefix, time_range_display, file_count
  from families where series_prefix = \"LEPTBR\"
''', c))
"
```

Companion files in the same directory: `datasus_codebook.yaml` (the curated
translation registry), `SINAN_LEPT_..._variable_profiles.csv` (the LEPT field
profile extracted from the compendium), and two codebase dumps retained for
provenance.

---

## 15. Acquired and wired, currently out of analytic scope

Distinct from §9, which is about sources fetched and then not used. These are
sources with **working adapters, tests and extracted outputs** that the
2026-08-14 scope restart placed outside the current manuscript. They are
documented here so a later study does not re-derive them, and so this document
does not imply they are unavailable.

| Source | Adapter | Extracted | Grain | Why out of scope |
|---|---|---|---|---|
| **PAM** crop area (SIDRA 1612) | `brepi/sources/agriculture` | `results/structural/agriculture_municipality_year.parquet` | municipality × year, 2007–2024 | Rice and sugarcane area index the flooded-field occupational route; belongs to the queued rainfall paper |
| **PPM** herd size (SIDRA 3939) | `brepi/sources/agriculture` | same file | municipality × year, 2007–2024 | Cattle and swine density is a rural reservoir proxy; same queued paper |
| **Censo 2022 favela block** (9883, 9887, 9888, 9892, 9893, 9894, 10344) | `brepi/sources/favelas` | `results/structural/favela_profile_municipality.parquet`, `favela_sewage_inside_outside.parquet` | municipality, 2022 | 656 municipalities with settlements, 16,349,896 residents. Table 10344 gives sewage **inside vs outside** settlements within one municipality — the within-municipality contrast that answers "that is not sanitation, it is poverty". Belongs to the queued sanitation/rainfall paper |
| **PNSB 2008 disease occurrence** (SIDRA 354, category 120933) | `brepi/sources/sanitation/pnsb` | `results/structural/pnsb_leptospirosis_2008.parquet` | municipality, 2008 | A municipal statement about leptospirosis made to a **sanitation** survey, not to the health system. Of 197 municipalities declaring occurrence, 111 (56.3%) confirmed zero cases to SINAN that year. Removed from the current manuscript by RES-012 as an under-specified second estimand |
| **PNSB service existence** (SIDRA 1238, 2000/2008; 7460/7461/7483/7500, 2017) | `brepi/sources/sanitation/pnsb` | `results/structural/pnsb_services.parquet` | municipality, 2000/2008/2017 | Measures *service existence* (binary) where the census measures *household coverage* (fraction). Used once, as validation: where PNSB 2008 reports no sewer network, the back-extrapolated census coverage has median 0.9% (p90 14.5%); where it reports one, 50.4%. That check is retained and reusable |

**These are all SIDRA tables, and SIDRA has its own single source of truth.**
`sources/SIDRA_COMPENDIUM.md` catalogues every table the project recognises with
its variables, classifications and traps, and `tests/test_sidra_compendium.py`
fails if a table is wired into the registry without being catalogued there.

---

## 16. Variable catalogue — what is actually in the line level

Sections 1–15 describe the *files*. This describes the **variables**, because a
data document that stops at the file level lets a study leave half its evidence
untouched — which is what happened here. As of 2026-08-14 the analysis had used
roughly eight fields of a **309-column** extract.

Generated by `studies/leptospirosis/46_variable_catalogue.py` into
`data/results/variable_catalogue/`:

| File | Contents |
|---|---|
| `variable_catalogue_confirmed.csv` | Every column, profiled on the **66,667 confirmed cases** — the analytic population |
| `variable_catalogue_all_notifications.csv` | The same over all 328,984 notifications |
| `block_summary.csv` | Per epidemiological block: variable count and median completeness |
| `usable_coded_variables.csv` | 49 coded variables ≥70% valid on confirmed cases |
| `poorly_recorded_variables.csv` | 19 coded variables <40% valid |

**Profile on confirmed cases, not on notifications.** The two differ enormously
— 79.7% of notifications are discarded — and conflating them has already
produced one wrong claim in this project (see `RESEARCH_JOURNAL.md` §8,
2026-08-14). Completeness figures below are on confirmed cases throughout.

The four decode states are never collapsed: **valid** (a code the dictionary
knows), **unknown** (an explicit *ignorado* sentinel — asked, not known),
**missing** (never filled), **invalid** (outside the dictionary). See
`CODEBOOK.md`.

### 16.1 Block overview

| Block | Vars | Median % non-null | Status in the current study |
|---|---:|---:|---|
| geography | 78 | 68.2 | used (residence; infection municipality available and mostly unused) |
| other / decoded | 56 | 88.5 | mixed |
| laboratory | 31 | 9.9 | **largely dark** — serology used descriptively only |
| exposure (`ANT_*`, `CON_AMBIEN`) | 16 | **95.3** | **dark — see 16.3** |
| clinical (`CLI_*`) | 15 | **96.6** | **dark — see 16.2** |
| date | 12 | 99.9 | used (onset, notification, digitisation) |
| demography | 8 | 100.0 | partly used (sex, age; race and schooling unused) |
| occupation | 3 | 53.1 | dark |
| care (`ATE_*`) | 2 | 83.0 | **used — `ATE_HOSP` carries the central result** |
| outcome | 2 | 93.9 | used |
| classification | 2 | 100.0 | used |

### 16.2 Clinical presentation — 15 signs, 91–98% recorded, essentially unused

Coded 1 = *sim*, 2 = *não*, 9 = *ignorado*. Prevalence is among confirmed cases
with the field recorded.

| Variable | % recorded | Present (*sim*) | Sign |
|---|---:|---:|---|
| `cli_febre` | 97.6 | 58,608 (90%) | fever |
| `cli_mialgi` | 96.7 | 54,709 (85%) | myalgia |
| `cli_cefale` | 95.6 | 47,981 (75%) | headache |
| `cli_icteri` | 95.4 | 31,984 (50%) | **jaundice** |
| `cli_vomito` | 95.2 | 34,208 (54%) | vomiting |
| `cli_prost` | 94.0 | 37,220 (59%) | prostration |
| `cli_pantur` | 93.8 | 37,667 (60%) | **calf pain** — the classic leptospirosis sign |
| `cli_diarre` | 93.9 | 20,359 (33%) | diarrhoea |
| `cli_respir` | 93.6 | 15,621 (25%) | respiratory involvement |
| `cli_renal` | 92.5 | 14,537 (24%) | **renal impairment** |
| `cli_conges` | 92.7 | 11,313 (18%) | conjunctival suffusion |
| `cli_hemorr` | 91.8 | 5,917 (10%) | **haemorrhage** |
| `cli_hemopu` | 91.8 | 5,903 (10%) | **pulmonary haemorrhage** |
| `cli_cardia` | 91.2 | 4,276 (7%) | cardiac involvement |
| `cli_mening` | 91.3 | 1,693 (3%) | meningismus |

**Why this matters.** Four of these (jaundice, renal, haemorrhage, pulmonary
haemorrhage) constitute a severe phenotype covering 54.9% of confirmed cases,
and that severity split is what allowed the depth-versus-severity separation in
`RESEARCH_JOURNAL.md` Link 3d. The other eleven signs remain unanalysed.

**Trap, and it bounds any use of this block.** The clinical form is completed
**at notification**, before deterioration. A non-severe-phenotype case dying is
therefore not a contradiction but a timing artefact, and the block's recording
completeness varies by territory (92.8% in the deepest-detecting quintile to
86.9% in the shallowest). Any severity adjustment built on it is a **lower
bound** on case-mix contribution, never a point estimate.

### 16.3 Antecedent exposures — 14 route variables, 81–90% recorded, unused

The `ANT_CB_*` block records self-reported contact in the 30 days before onset.
This is **individual-level transmission-route data on ~58,000 confirmed cases**
and the study has not touched it.

| Variable | % recorded | Reported (*sim*) | Exposure |
|---|---:|---:|---|
| `ant_cb_sin` | 88.9 | **36,304 (61%)** | signs of rodents |
| `ant_cb_lam` | 90.1 | 27,217 (45%) | mud |
| `ant_cb_cor` | 88.0 | 19,579 (33%) | stream / watercourse |
| `ant_cb_lix` | 88.0 | 19,561 (33%) | rubbish |
| `ant_cb_cri` | 87.8 | 19,303 (33%) | animal rearing |
| `ant_cb_roe` | 87.7 | **17,498 (30%)** | rodents seen |
| `ant_cb_ter` | 87.5 | 15,949 (27%) | vacant lot |
| `ant_cb_fos` | 87.7 | 11,222 (19%) | septic pit |
| `ant_cb_pla` | 87.3 | 9,507 (16%) | plantation |
| `ant_cb_gra` | 86.9 | 7,054 (12%) | grain / storage |
| `ant_cb_cai` | 87.1 | 6,246 (11%) | water tank |
| `ant_cb_out` | 81.2 | 5,357 (10%) | other |
| `ant_animai` | 60.0 | 483 | animal contact — **poorly recorded, do not use** |
| `ant_humano` | 63.7 | 3,559 | human contact — **poorly recorded, do not use** |

`con_ambien` (69.5% recorded) gives the **environment of probable infection**:
domiciliar 26,488, trabalho 11,342, outro 4,496, lazer (remainder). This is the
single most direct statement of transmission setting in the dataset.

`doenca_tra` (77.7% recorded) flags **work-relatedness**: 12,463 *sim* (24% of
recorded). Distinct from `con_ambien = trabalho` and worth cross-tabulating.

`TPAUTOCTO` (autochthonous case) is recorded for **0.63%** of confirmed cases.
**Unusable. Do not attempt an imported-case analysis with it.**

### 16.4 Laboratory — deep but sparse

| Variable | % recorded | Note |
|---|---:|---|
| `lab_elis_1` | 92.7 | ELISA, first sample; 50,426 *reagente*. The best-recorded laboratory field |
| `lab_elis_2` | 57.9 | ELISA, second sample |
| `lab_micr_1` | 56.9 | **MAT**, first sample; 8,326 *reagente* |
| `lab_micr_2` | 54.0 | MAT, second sample |
| `res_pcr` | 54.9 | PCR; 1,654 positive |
| `res_imuno` | 54.1 | immunohistochemistry; 323 positive |
| `res_isol` | 53.8 | culture isolation |
| `titre_s1_first` | **14.4** | MAT titre, 110 distinct values — a graded serological quantity, currently unused |

Serovar attribution reaches only 13.5% of confirmed cases and carries the
standing constraint RES-006 (`RESEARCH_JOURNAL.md` §5): MAT is serogroup-reliable
and serovar-presumptive, and Patoc is a saprophytic screening antigen.

### 16.5 Demography and occupation

| Variable | % recorded | Note |
|---|---:|---|
| `cs_sexo` | 99.99 | used — rate ratio male:female 4.19 (4.11–4.27) |
| `cs_gestant` | 98.0 | pregnancy status; 55,885 *não se aplica* |
| `cs_raca` | 89.8 | branca 29,815, parda 25,384, preta 3,930 — **well recorded and unused**. IBGE self-declaration and SINAN administrative record are *not* the same construct (`CODEBOOK.md`) |
| `cs_escol_n` | 61.8 | schooling; 11 levels — **too incomplete to carry a claim** |
| `id_ocupa_n` | ~32 | CBO occupation; usable only on the recorded subset, and that subset is not random |

### 16.6 What this catalogue changed

Three things were dark and are now named as such, with threads opened in
`RESEARCH_JOURNAL.md` §4: the clinical block beyond the four severity markers,
the entire antecedent-exposure block, and `con_ambien`. Two things were assumed
usable and are not: `TPAUTOCTO` (0.63%) and `cs_escol_n` (61.8%).
