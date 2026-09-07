# brepi — architecture of the epidemiology workbench

`brepi` is a study-agnostic workbench for reproducible epidemiology over
Brazilian public data. DATASUS, SIDRA, IBGE, climate and disaster systems are
reusable adapters. Diseases, causal questions, inclusion criteria, estimands
and manuscript claims live under `studies/`.

Study-agnostic does **not** mean statistically empty. Distributed lags,
spatiotemporal count models, standardisation, interrupted/event-study designs,
ascertainment models, multiple imputation, sensitivity analysis and diagnostic
contracts are recurring epidemiological motifs and belong in the harness.
A study selects and parameterises them; it does not reimplement them.

## Architectural boundary

```text
remote systems
  -> source adapters
  -> immutable raw artefacts + provenance manifests
  -> canonical long facts and versioned geographies
  -> validated analytic panel contracts
  -> reusable statistical engines
  -> study specification / target graph
  -> manuscript tables, figures and claim ledger
```

The layers are:

1. **Acquisition and provenance (Python).** Remote bytes, schemas, revisions,
   decoding and source-specific failure modes.
2. **Harmonisation (Python).** Codes, typed missingness, denominator tensors,
   versioned geography, conservative territorial transfer and panel joins.
3. **Statistical engines (primarily R; Python where useful).** General model
   primitives with explicit inputs, estimands, diagnostics and deterministic
   seeds.
4. **Study orchestration (`studies/<id>/`).** Case definitions, exposure
   choices, temporal/spatial scales, hypotheses, model formulae, robustness
   set and reporting.

Parquet is the Python–R boundary. A study target graph may call both languages,
but no model is allowed to reach back to a live remote source.

## Current package map

```text
brepi/
  config.py
  io/
    cache.py              logical-key cache, SHA-256 provenance sidecars,
                          frozen manifests and drift verification
    datasus_ftp.py        FTP / mirror transport
    dbc.py                DBC/DBF decoding
  sources/
    datasus/              SINAN, SIH, SIM, CNES, population
    sidra/                metadata registry, request planner, extraction
    climate/              BR-DWGD, ERA5-Land, ENSO, monthly/daily indices
    disasters/            Atlas/S2iD acquisition, episodes and treatment timing
    landuse/              planned
    sanitation/           census sanitation harmonisation; PNSB service and
                          disease-occurrence statements
    socioeconomic/        census urbanisation and municipal economy
    agriculture/          PAM crop area and PPM herd size as municipality-year
                          covariates, with the non-additive-category traps encoded
    favelas/              Censo 2022 favela block, including the within-municipality
                          sewage contrast and its restricted universe
  geo/
    lattice.py            dated municipal lattices, lineages, crosswalks, AMC
    health_regions.py     strict official membership loader
    transfer.py           conservative extensive/intensive/rate operators
  denominators/           census/estimate reconciliation and rates
  panel/
    spine.py              complete lattice and audited left joins
    aggregate.py          grain coarsening under the transform-after-aggregate
                          contract; refuses to sum a rate, share or transform
  analysis/
    atlas.py              declarative unit-level assembly with join auditing
    rates.py              exact Poisson and Clopper-Pearson intervals
  qa/
    evidence.py           declarative result assertions
    vocabulary.py         cross-file controlled-vocabulary consistency
    sinan_quality.py      source and panel quality diagnostics
R/
  00_io.R                 panel contract and aggregation
  00_analysis_spec.R      study-to-engine specification seam
  01_descriptive.R        reusable burden/seasonality/quality summaries
  02_crossbasis.R         grouped DLNM basis
  03_inla_spacetime.R     count/spatiotemporal engines
  04_multiscale.R         MAUP/scale sensitivity
  05_did_flood.R          staggered event-study engines
  06_ascertainment.R      multi-source/detection models
  07_figures.R            reproducible scientific graphics
  08_sensitivity.R        declarative robustness batteries
  09_exec.R               fit execution, timing and process budgets
  10_regimes.R            compositional clustering engines
  11_tables.R             exact-interval tables and standardisation
  12_severity.R           case-fatality and hospitalisation models
  13_figure_fusion.R      mass-chosen map zoom windows and panel fusion
studies/
  leptospirosis/          first consumer and current vertical slice
docs/                     see the documentation map below
```

`landuse` remains unimplemented. Climate, sanitation and socioeconomic
adjustment blocks are operational. Disaster primitives exist, but the
leptospirosis event-treatment vertical slice is incomplete.

## Seven non-negotiable contracts

1. **Source ownership.** URLs, encodings, layouts and revisions belong to the
   source adapter/configuration layer, not analysis scripts.
2. **Byte provenance.** Every remote artefact has URI, retrieval time, size,
   SHA-256, remote modification metadata and tool version. The present cache is
   key-addressed with content hashes; it must not be described as a
   content-addressed object store.
3. **Frozen input identity.** A paper points to manifests and panel hashes.
   Reanalysis verifies them before fitting.
4. **Typed missingness.** Zero, structural absence, suppression, unavailable
   vintage, incomplete month and failed join are distinct states.
5. **Versioned geography.** A code without a geography/version is incomplete.
   Counts conserve mass; fields conserve support-weighted means; rates are
   rebuilt; adjacency is recomputed. See ADR-001.
6. **Complete spine first.** Every data block left-joins onto an enumerated
   unit-time lattice and emits a coverage report. A covariate cannot redefine
   the population at risk.
7. **Stage honesty.** A primary-stage missing input or dependency stops.
   Optional secondary branches may be absent only if the run manifest records
   them as unavailable. A `NULL` is not a completed analysis.

## Statistical harness philosophy

### How study work becomes harness capability

The harness grows by absorbing what the active study forces us to build. The
test for promotion is not "was this useful once" but "is the *shape* of this
problem study-independent, with only the estimand and the inputs varying".
Three promotions from the leptospirosis RQ1/RQ2 work illustrate the rule:

* **Robustness batteries** (`R/08_sensitivity.R`). Every study runs one
  estimand through many defensible specifications and reports the movement.
  Only the perturbation list is study-specific, so the harness owns the
  declarative vocabulary (typed perturbation classes, mandatory rationale), the
  execution loop with per-run time budgets, the reference comparison against a
  pre-declared materiality threshold, and a verdict sentence. Crucially it owns
  the *failure* semantics: a specification that will not fit is recorded and
  reported, never dropped, because a battery that silently omits the
  disagreeing specification is worse than no battery.
* **Cost attribution before inference** (`studies/*/bench/`). Diagnosing why a
  model is slow is a recurring problem with a fixed method: run a nested
  structural ladder and a one-at-a-time diagnostic ladder as separate,
  timed, memory-sampled, timeout-bounded processes. The RQ1 blocker was
  misattributed for a full development pass because that decomposition did not
  exist.
* **Source-coding conventions are data, not truth** (`sources/disasters`). The
  COBRADE finding — that Brazilian civil-defence services file the 2024 Rio
  Grande do Sul flood under *chuvas intensas* rather than under any flood code
  — is exactly the kind of fact that belongs in the adapter's constants and
  docstrings, where the next study inherits it, rather than in one study's
  analysis script.

Three later promotions came out of the RQ2 correction and the atlas assembly.
All three share a property worth naming: each generalises a defect that
**raised no error**. A harness earns its keep on those, not on the failures that
announce themselves.

* **Transform-after-aggregate** (`panel/aggregate.py`). RQ2 coarsened a monthly
  `asinh(cases)` outcome to quarters by summing it. The sum of a transform is
  not the transform of a sum — it is not any quantity — and the event study had
  to be discarded and refitted. Every panel study coarsens a grain, and every
  one of them can make this mistake, so the rule is enforced instead of
  documented: a column whose name marks it as a rate, share, ratio, z-score,
  log or `asinh` cannot be placed in `sums=`, and derived quantities are
  evaluated after the aggregation, on the summed primitives. The same call
  verifies that columns declared constant within a group actually are, rather
  than taking `first` of something that varies.
* **Audited unit-level assembly** (`analysis/atlas.py`). A study finishes with
  its answers scattered one per stage directory, all keyed on the same unit, and
  a figure needs six joins. Three silent failure modes recur: a key that is a
  zero-padded string in the panel and an integer in every CSV that round-tripped
  through schema inference; a left join that drops layer rows without a word;
  and two layers both carrying `name`, producing `name_right`, with whichever a
  downstream script reads decided by luck. The module normalises the key once,
  reports per-layer match counts and out-of-spine rows, and raises on a
  collision or a duplicated key.
* **Cross-file vocabulary consistency** (`qa/vocabulary.py`). Fourteen result
  files here carried IBGE's Portuguese macro-regions and English ones in the
  same column name. Both individually correct; joined, they matched nothing.
  The check distinguishes disjoint vocabularies (fatal) from partial overlap
  (often a legitimate subset), and is wired into the study's claim audit so the
  fix cannot regress.

The harness owns recurring mechanics, including:

- construction and validation of offsets and rates;
- grouped lag bases that cannot leak across spatial units;
- negative-binomial, zero-inflated and Bayesian spatiotemporal model wrappers;
- spatial graph validation and scale sensitivity;
- event-time construction, staggered-treatment estimators and pretrend checks;
- multi-source ascertainment and detection models;
- missing-data and territorial-transfer ensembles;
- simulation/calibration, residual diagnostics, convergence checks and
  estimand-level tidy outputs.

The study owns:

- the outcome definition and cohort;
- causal diagram and adjustment set;
- exposure and contrast;
- lag window and prespecified alternatives;
- primary geography and time window;
- inclusion/exclusion rules;
- which reusable engine answers which hypothesis;
- the claim ledger linking every manuscript statement to a target.

For the active leptospirosis study, the machine-readable ledger is
`studies/leptospirosis/CLAIM_LEDGER.yaml`; `audit_claims.py` checks material
artifacts and writes `data/results/00_data_quality/claim_ledger_audit.json`. A validation fit
may prove that the numerical route executes, but cannot change a blocked claim
to a scientific result.

`R/00_analysis_spec.R` is the current seam. It is transitional: mature engines
still use canonical internal names such as `cases`, `precip_mm` and
`log_offset`, so the adapter materialises those aliases without mutating the
study panel. Engines should gradually accept arbitrary declared columns.

## Versioned geography

Territorial change is a first-class statistical problem, not a join cleanup.
Municipal splits are only one case; health-region membership, census sectors,
regional classifications and service catchments also change.

`geo.transfer` implements sparse transfer operators with executable invariants.
`geo.lattice.amc_groups()` produces exact stable unions when disaggregation is
not identified. Estimated areal/dasymetric transfers must carry method and
uncertainty and must be tested against AMC/native-geography analyses.

The RQ1 climate panel currently exposes two approximations rather than hiding
them:

- the published ERA5-Land tail omits one calendar day per month and is
  calendar-prorated, with observed totals and coverage retained;
- five 2013 municipalities absent from the zonal-product lattice use a
  parent-field proxy, marked for exclusion/pixel-based replacement.

## Reproducibility and environments

- Python runs in the `pegasus` conda environment and is tested with `pytest`.
- R uses the project-local `renv` scaffold. A committed `renv.lock` is required
  before a modelling result is submission-grade.
- Stochastic targets have explicit seeds.
- Large fitted objects and figures are derived artefacts; source manifests,
  code, specifications and compact result tables are the reproducibility core.
- `targets` should stop on failures in the primary path. Broad
  `error = "continue"` is suitable for exploration, not a paper build.

## Vertical slices and completion gates

Breadth is not completion. Work proceeds in vertical slices:

1. frozen outcome + denominator panel and descriptive reconciliation;
2. exposure-complete RQ1 panel and development model;
3. official health-region mapping, spatial graph and primary RQ1 model;
4. SIH/SIM triangulation and ascertainment results;
5. disaster-treatment panel and prespecified secondary event study;
6. robustness battery, frozen environment, figures and manuscript claim ledger.

A slice is complete only when its inputs are frozen, contracts pass, the target
executes, diagnostics are stored and the scientific output is interpretable.
Module count and syntax checks do not satisfy this gate.

## Documentation map

One file per purpose. If two files would answer the same question, one of them
is wrong.

| Question | Document |
|---|---|
| How is the codebase organised and why? | **this file** |
| What is the scientific scope, argument, and every verified finding? | [`RESEARCH_JOURNAL.md`](RESEARCH_JOURNAL.md) |
| What engineering defects are open? | [`ISSUE_LEDGER.md`](ISSUE_LEDGER.md) |
| What does each dataset contain and what is wrong with it? | [`DATA.md`](DATA.md) |
| How are DATASUS coded values translated? | [`CODEBOOK.md`](CODEBOOK.md) |
| How is territorial change handled? | [`ADR-001-versioned-geographies.md`](ADR-001-versioned-geographies.md) |
| Which SIDRA tables exist and what are their traps? | [`sources/SIDRA_COMPENDIUM.md`](sources/SIDRA_COMPENDIUM.md) |
| Which exact files support each manuscript claim? | `studies/leptospirosis/CLAIM_LEDGER.yaml` |
| Do the material numeric values still match the files? | `studies/leptospirosis/RESULT_ASSERTIONS.yaml`, run by `audit_claims.py` |
| How should an epidemiological analysis be conducted here? | `.claude/skills/epi-db-research/SKILL.md` |

## Current executable status (2026-08-14)

- Municipality-month outcome/denominator spine: built and reconciled. 66,516
  confirmed records conserve as 66,480 municipal assignments plus 36 explicitly
  unresolved.
- Climate RQ1 derivative: 1,269,960 rows, 100% join coverage, with source
  snapshot and quality report.
- Primary health-region spatiotemporal models: **fitted and complete.** The
  RQ1 DLNM/BYM2 blocker recorded in July was resolved by process-isolated
  execution with enforced budgets; the RQ4 case-fatality and SIH fatality models
  are fitted with 0 CPO failures.
- Analytic atlases: built at four grains and reconciled against the descriptive
  totals as a standing machine assertion.
- Territorial transfer core: extensive, intensive and rate invariants
  implemented and tested; uncertainty ensembles remain open (BREPI-012).
- SIH/SIM acquisition and cross-system comparison: complete, with residence
  alignment and admission-date correction applied.
- Structural adjustment blocks (sanitation, urbanisation, real GDP): complete.
  Agriculture, favela and PNSB blocks: extracted, currently out of scope.
- Surface-water and land-use extensions: not acquired.
- Test state: 286 Python tests and 4 R test files pass.

The current scientific milestone is defined in
[`RESEARCH_JOURNAL.md`](RESEARCH_JOURNAL.md) §4 — resolving the open threads
that gate the manuscript, not building another source adapter.

## Running

From `brepi/`:

```bash
python -m pytest tests -q
python studies/leptospirosis/07_build_climate_panel.py
Rscript --vanilla -e "targets::tar_make(script='R/_targets.R')"
```

`BREPI_DATA_ROOT` relocates cache, interim, panel and report directories.
