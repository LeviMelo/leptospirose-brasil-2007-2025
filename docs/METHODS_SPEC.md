# METHODS_SPEC — exact specification for reproduction

Required by FEEDBACK_2 §82E. This is the document another analyst needs in order
to rebuild model C without guessing any component. Where a required detail is not
recorded by the pipeline it is marked **UNRESOLVED** rather than filled with a
plausible value.

Software: R 4.4.1, R-INLA 24.12.11, Python 3.11.15, polars 1.41.2, scipy 1.17.1.
Geometry handled with `sf`; figures with `ggplot2`/`patchwork`/`ggspatial`.

---

## 1. Sources and provenance

| System | Files | Unit extracted | Geography | Date basis |
|---|---|---|---|---|
| SINAN-LEPT | Annual national LEPTBR files, 2007–2025 | Notification record | Municipality of **residence** (`ID_MN_RESI`), mapped to health region | Symptom onset (`SEM_PRI`) |
| SIM | Annual national mortality files | Death record, underlying cause **A27** (`CAUSABAS`) | Municipality of **residence** (`CODMUNRES`) | Date of death (`DTOBITO`) |
| SIH | Annual national AIH files | **Admission record (AIH)**, principal diagnosis A27 — *not* unique persons; no de-duplication or long-stay merging applied | Municipality of **residence** (`MUNIC_RES`) | Admission date (`DT_INTER`) |
| IBGE | Population estimates, Revisão 2024 | Municipality-year population | Municipality | Calendar year |

**UNRESOLVED — extraction dates.** The pipeline records file inventories but not
per-file download timestamps. Required for RECORD item 6.1.

**UNRESOLVED — health-region division version.** Municipalities are mapped to
439 health regions through `data/results/atlas/municipality_atlas.parquet`, and
geometry is dissolved from the municipal mesh by `paper/R/prep_geo.R`. The
release year of the health-region division is not recorded, and a single
contemporary geography is imposed retrospectively across 2007–2025 — boundary
changes within the period are **not** modelled.

## 2. Analytic population

From `studies/leptospirosis/60_analysis_panel.py`, asserted to reconcile to the
last record:

```
328.984 notificações
  − 238.353 descartadas − 11.699 inconclusivas − 12.265 sem classificação
  = 66.667 confirmadas
  − 155 com data de início de sintomas inválida
  − 105 com início fora de 2007–2025
  = 66.407 confirmadas na janela
  − 49 sem município de residência
  = 66.358 população analítica
```

## 3. Field operationalisation

Every decoded field carries a companion state in {valid, unknown, missing,
invalid}. **A blank is never read as a negative answer.** Each proportion uses
its own valid-field denominator, and those denominators are carried in the panel.

| Quantity | Field | Accepted | Denominator |
|---|---|---|---|
| Confirmed case | `CLASSI_FIN` | `confirmado` | notifications |
| Death | `EVOLUCAO` | `obito_por_leptospirose` only | `EVOLUCAO` valid |
| Hospitalised | `ATE_HOSP` | `sim` | `ATE_HOSP` valid |
| Laboratory | `CRITERIO` | `clinico_laboratorial` | `CRITERIO` valid |
| Age ≥60 | `NU_IDADE_N` (packed unit+value) | ≥60 years after decoding | age valid |
| Male | `CS_SEXO` | `masculino` | confirmed cases |

`obito_por_outras_causas` is a **known outcome that is not a case fatality**.

### Severity phenotype — exact algorithm (§68)

A record is classified only if **all three** of `CLI_ICTERI`, `CLI_RENAL`,
`CLI_HEMORR` decode validly. Given that, **severe** = any of the three is `sim`;
**non-severe** = all three are `nao`. A record with any undecodable component is
excluded from **both** numerator and denominator, so a missing symptom field
cannot manufacture a non-severe case. National completeness of the full block:
90,1%. `CLI_HEMOPU` is excluded from the canonical definition; it disagrees with
`CLI_HEMORR` in 13 of 58.695 doubly-decoded records (0,02%).

### Laboratory-confirmed exposure (§69)

*H*<sub>lab</sub> = hospitalised laboratory-confirmed cases ÷ laboratory-confirmed
cases with a valid `ATE_HOSP`. Both restrictions applied before the ratio.
Minimum 30 laboratory-confirmed cases with a valid hospitalisation field per
region; 196 regions qualify. Same window and geography as the primary exposure.

## 4. The exposure

*H*<sub>rt</sub> = hospitalised confirmed cases ÷ confirmed cases with a valid
hospitalisation field, in health region *r*, year *t*. **Inverse** indicator of
ascertainment breadth. Centred at the national case-weighted value (0,7219) and
scaled so every coefficient is **per +10 percentage points**. Centring changes
the intercept only.

*H* is an estimated proportion, noisy where the denominator is small; model F
restricts to cells with ≥10 cases behind it.

## 5. The models

Observation unit: health region × year. Cells require ≥1 known outcome **and** a
defined *H*: 4.941 cells, 426 regions, of 8.341 possible.

```
deaths_rt ~ Binomial(outcome_known_rt, p_rt)
logit(p_rt) = α + β·H10_rt + u_r + v_t + γ'X_rt
```

- `u_r` — **BYM2** spatial effect, Riebler *et al.* parameterisation, structured
  component scaled to unit generalised variance.
- `v_t` — **RW1** over year.
- `X_rt` — proportion of cases aged ≥60 and proportion male (models C, D, E, F).

**Adjacency**: queen contiguity over the dissolved health-region polygons
(`spdep::poly2nb`, `snap = 0.005`, simplified at `dTolerance = 0.002`), built
over all 439 regions so deleting a case-free region does not tear the
neighbourhood structure; 0 islands, cross-checked against the repository's own
graph by edge Jaccard (fit aborts below 0,95).

**Priors** (as in `23_rq4_lethality.R`, PC priors throughout):
- spatial sd: P(σ > 1) = 0,01
- BYM2 mixing φ: P(φ < 0,5) = 0,5
- RW1 sd: P(σ > 1) = 0,01
- fixed effects: N(0, 1) on the log odds ratio per 10 pp

**Fitting**: INLA, `int_strategy` and `strategy` from the repository's
`confirmatory` execution profile. Reported: DIC and WAIC (comparable across
A/B/C only, which share rows).

**Interpretation of φ**: the share of the *spatial random effect's* marginal
variance that is spatially structured — **not** the share of variation in case
fatality that is spatial.

### Fixed-effects specification (§15)

```
glm(cbind(deaths, outcome_known − deaths) ~ H10 + age60_10 + male_10
      + factor(health_region_code) + factor(year), family = binomial)
```
Cluster-robust covariance by health region (`sandwich::vcovCL`, HC0). Regions
contribute only with at least one death **and** one survivor across the series
and more than one year: 306 of 426 regions, 97,9% of cases. Removes
time-invariant regional confounding only.

## 6. Estimation of the descriptive strata

Health regions ordered by *H* and cut into five bands of approximately equal
**case mass** (≈20% of confirmed cases each, not 20% of regions), restricted to
regions with ≥30 confirmed cases carrying a valid hospitalisation field. Per-band
H median, regions, cases and person-years are in
`manuscript_tables/suppl_quintis_valores_absolutos.csv`.

## 7. Intervals

Exact throughout: Poisson (Garwood) for rates, Clopper–Pearson for proportions.
Bayesian models report 95% **credible** intervals; the fixed-effects model
reports a 95% **confidence** interval. Spearman ρ is reported as an ordinal
association, not an effect size.

## 8. Execution order

```
60_analysis_panel.py        panel + study flow      (all downstream depend on this)
61_denominator_coupling.py  arithmetic standardisation
62_triangulation.py         SINAN / SIH / SIM, 2008–2024
63_missingness_record.py    completeness + field dictionary
64_exclusion_sensitivity.py thresholds 0/10/20/30/50
65_construct_validity.py    confirmation pathway + severity definitions
66_case_mix.py              age–sex standardisation
67_rs2024_event.py          RS event series
68_primary_model.R          models A–F + fixed effects      (needs 66 optionally)
69_within_region.py         independent within-region cross-check
70_descriptives.py          Table 1 inputs
71_manuscript_tables.py     Tables 1 and 2                  (needs 68, 70)
72_reporting_checklist.py   STROBE/RECORD + supplement inventory
73_ledgers.py               RESULTS_LEDGER                  (needs all of the above)
paper/R/prep_geo.R, fig1_geography.R, fig2_association.R, fig3_rs2024.R
quarto render paper/artigo.qmd
```
