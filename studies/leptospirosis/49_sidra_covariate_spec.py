"""Which SIDRA covariates this study should acquire, and what each one buys.

Audit part 2. Part 1 inventories the SIDRA surface; this script is the
judgement: given the current thesis (case fatality governed by how far down the
severity distribution a territory's surveillance reaches) and the queued papers
(rainfall x sanitation DLNM, flood event studies, transmission-regime
clustering), which specific table/variable/classification/category tuples are
worth extracting, at what granularity, and do they actually return data at N6.

Nothing here is recommended on the strength of a table title. Every entry
carries the live category ids, and the priority set is *extracted* against the
API at N6 before being recommended: row counts, locality coverage, SIDRA value
statuses and, where the classification partitions a published Total, a margin
reconciliation.

Five blocks, each answering a question the current panel cannot:

1. ``health_capacity`` -- the reviewer's first objection. If case fatality falls
   where surveillance reaches deeper, is that not simply hospital availability?
   SIDRA's answer is thin and the script proves it thin (AMS is state-level;
   CEMPRE is a formal-enterprise register with confidentiality suppression);
   the mechanistic variable (dialysis capacity) turns out to exist in MUNIC 2021
   at N6, and beds exist only in CNES.
2. ``deprivation_income`` -- GDP per capita is production, not income. Median
   per-capita household income and the share under 1/4 minimum wage, at both
   census anchors.
3. ``crowding_housing`` -- residents per bedroom and precarious dwelling type.
   Classic leptospirosis covariates, entirely absent from the panel.
4. ``education_labour`` -- low-education share and the share of the occupied
   population working in agriculture (occupational exposure).
5. ``queued_papers`` -- what the rainfall/sanitation and flood papers need on
   top of the favela block and PNSB series they already have.

Outputs land in ``data/results/sidra_covariate_spec/``. Run with ``--validate``
to hit the API (cached; a rerun is offline). ``--spec-only`` writes the
specification without extracting.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.config import PATHS
from brepi.sources.sidra import api
from brepi.sources.sidra.extract import (
    Selection,
    check_margins,
    coverage_report,
    extract,
    resolve_localities,
    value_status_summary,
)

OUT = PATHS.results / "sidra_covariate_spec"
CATALOG = PATHS.cache / "sidra" / "catalog"
CNES_CACHE = PATHS.cache / "datasus" / "cnes"

# Tables already wired into brepi/sources/sidra/registry.yaml, and the subset of
# those whose values have actually been fetched. Both are read from disk in
# :func:`current_usage` rather than hardcoded here.
REGISTRY = Path(__file__).resolve().parents[2] / "brepi" / "sources" / "sidra" / "registry.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# The specification
# ---------------------------------------------------------------------------
#
# ``clsf`` maps classification id -> category ids to request. Every
# classification a table carries is pinned explicitly, including the ones held
# at Total: SIDRA returns unselected classifications pinned to their own Total
# anyway, and writing it down is what stops a sex-specific figure being read as
# a sex-total one three months later.
#
# ``derive`` states the estimand in category ids, so the covariate is defined by
# arithmetic on published cells rather than by a column name.

SPEC: list[dict[str, Any]] = [
    # -- 1. HEALTH-SERVICE CAPACITY ----------------------------------------
    {
        "id": "munic2021_nephrology_service",
        "block": "health_capacity",
        "priority": 1,
        "agregado": 9487,
        "table_name": "MUNIC 2021 - municipalities with specific services and bed types in public or SUS-contracted establishments",
        "research": "Pesquisa de Informações Básicas Municipais (MUNIC)",
        "variables": ["603"],
        "clsf": {
            "1623": ["59535", "59536", "59537"],
            "12446": ["47692"],
            "1619": ["59528"],
            "1624": ["59538"],
            "1625": ["59541"],
        },
        "clsf_names": {
            "1623": "Existência de serviço de nefrologia em estabelecimento público ou conveniado ao SUS",
            "12446": "Classe de tamanho da população do município (pinned Total)",
            "1619": "Disponibilidade de serviço de atendimento de emergência (pinned Total)",
            "1624": "Leitos de UTI neonatal (pinned Total)",
            "1625": "Leitos de cuidados intermediários (pinned Total)",
        },
        "level": "N6",
        "periods": ["2021"],
        "granularity": "municipality, cross-section 2021 only",
        "estimand": "binary: municipality has a nephrology (dialysis) service in a public or SUS-contracted establishment",
        "derive": {"kind": "flag", "variable": "603", "clsf": "1623", "yes": "59536", "total": "59535"},
        "buys": (
            "Severe leptospirosis kills through acute kidney injury; survival past the "
            "oliguric phase is dialysis-dependent. This is the mechanistic capacity "
            "variable for case fatality, and it is a different construct from "
            "'has a hospital'. It is the direct answer to the reviewer who says the "
            "hospitalised share is measuring hospital availability rather than "
            "surveillance depth: hospitalisation capacity and renal-replacement "
            "capacity can be adjusted for separately."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "MUNIC 2021 only -- one cross-section, mid-period. It is a time-invariant control, not a time-varying one.",
            "Self-reported by the municipal administration, and reports the service exists in the municipality, not that a leptospirosis patient reached it.",
            "Not catalogued in SIDRA_COMPENDIUM.md and not in registry.yaml; MUNIC is absent from the compendium entirely.",
        ],
    },
    {
        "id": "munic2021_emergency_referral",
        "block": "health_capacity",
        "priority": 2,
        "agregado": 9487,
        "table_name": "MUNIC 2021 - emergency (risk-of-life) service availability",
        "research": "MUNIC",
        "variables": ["603"],
        "clsf": {
            "1619": ["59528", "59529", "59530", "59531", "59532", "59533", "59534"],
            "12446": ["47692"],
            "1623": ["59535"],
            "1624": ["59538"],
            "1625": ["59541"],
        },
        "clsf_names": {
            "1619": "Disponibilidade de serviço de atendimento de emergência (Risco de Vida)",
        },
        "level": "N6",
        "periods": ["2021"],
        "granularity": "municipality, cross-section 2021 only",
        "estimand": (
            "ordered capacity: own public emergency service (59529) / other public "
            "service (59530) / SUS-contracted private (59531) / patient transported "
            "to another municipality (59532, 59533) / none (59534)"
        ),
        "derive": {"kind": "flag", "variable": "603", "clsf": "1619", "yes": "59532", "total": "59528"},
        "buys": (
            "Categories 59532/59533 name the municipality that has to send its "
            "emergencies elsewhere. That is referral dependence measured directly, "
            "which is the mechanism behind both under-hospitalisation and the "
            "residence-vs-infection municipality discordance already documented in "
            "the SINAN extract."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": ["Same 2021-only, self-report caveats as the nephrology item."],
    },
    {
        # Downgraded from priority 1 to 3 by its own validation: 99.5% of
        # municipalities answer 'Sim'. Kept because the 26 that answer 'none'
        # are a named stratum, not because the binary is a usable regressor.
        "id": "munic2021_health_surveillance",
        "block": "health_capacity",
        "priority": 3,
        "agregado": 9493,
        "table_name": "MUNIC 2021 - municipalities performing health surveillance services",
        "research": "MUNIC",
        "variables": ["603"],
        "clsf": {
            "1649": ["59630", "59631", "59632", "59633", "59634", "59635"],
            "12446": ["47692"],
            "1647": ["59627"],
        },
        "clsf_names": {
            "1649": "Realização de serviços de vigilância em saúde (Total / Sim / sanitária / epidemiológica / controle de endemias / não realiza nenhum)",
            "1647": "Busca ativa de nascidos vivos não registrados (pinned Total)",
        },
        "level": "N6",
        "periods": ["2021"],
        "granularity": "municipality, cross-section 2021 only",
        "estimand": "binary: municipality performs epidemiological surveillance (59633); and the complement, performs none of the three (59635)",
        "derive": {"kind": "flag", "variable": "603", "clsf": "1649", "yes": "59633", "total": "59630"},
        "buys": (
            "The thesis is that case fatality is governed by how far down the severity "
            "distribution surveillance reaches, indexed by the hospitalised share. "
            "That index is derived from the outcome data itself. This is an "
            "*external* municipal statement about whether an epidemiological "
            "surveillance service exists at all -- an instrument-like validator of "
            "the depth index that inherits none of SINAN's assumptions. It plays the "
            "same role as PNSB 354 already does, on a different axis and 13 years later."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "2021 only.",
            "MEASURED, and it is the reason this item was downgraded: 5,541 of 5,570 municipalities (99.5%) report performing some health surveillance, 5,346 (96.0%) epidemiological surveillance specifically, and only 26 (0.5%) report none. As a binary regressor it has almost no variance.",
            "Use it as a stratum definition (the 26 municipalities with no surveillance service at all) or as a falsification check, not as a covariate.",
            "Administrative existence, not performance.",
            "Categories 59632/59633/59634 are nivel-2 children of 59631; summing them with the parent double-counts.",
        ],
    },
    {
        # Upgraded to priority 1 by its own validation: the local-unit count is
        # never suppressed. Variable 707 (employed persons) was in this
        # selection and has been struck from the recommendation -- see caveats.
        "id": "cempre_hospital_activity_units",
        "block": "health_capacity",
        "priority": 1,
        "agregado": 6450,
        "table_name": "CEMPRE - local units and employed persons by CNAE 2.0 (series closed 2021)",
        "research": "Cadastro Central de Empresas",
        "variables": ["706", "707"],
        "clsf": {"12762": ["117811", "117812"]},
        "clsf_names": {
            "12762": "CNAE 2.0: 117811 = divisão 86 Atividades de atenção à saúde humana; 117812 = grupo 86.1 Atividades de atendimento hospitalar",
        },
        "level": "N6",
        "periods": ["2010", "2019"],
        "granularity": "municipality x year, annual 2006-2021 (continue with agregado 9528 for 2022-2024)",
        "estimand": "count of local units (706) and employed persons (707) in hospital-care activities (CNAE 86.1) per municipality per year",
        "derive": {"kind": "level", "variable": "706", "clsf": "12762", "category": "117812"},
        "buys": (
            "The only *annual* municipality-level health-capacity measure anywhere in "
            "SIDRA. If it survives suppression it lets health-service supply enter as "
            "a time-varying control rather than a census-anchored constant, which is "
            "what the current thesis needs to rule out capacity as the driver of the "
            "depth-fatality relationship."
        ),
        "status": "available_unused",
        "effort": "moderate",
        "caveats": [
            "EXTRACT VARIABLE 706 ONLY. Measured over 2010 and 2019 at CNAE 86.1: variable 706 (local units) has ZERO suppressed cells in either year at any municipality size; variable 707 (employed persons) is suppressed for 1,605 municipalities in 2010 and 1,648 in 2019, which is 64.5% of the municipalities that have any hospital activity at all in 2019. Employment at this CNAE depth is unusable; the establishment count is clean.",
            "In 2019, 2,556 municipalities have at least one CNAE 86.1 local unit and 3,014 are true zeros (SIDRA '-', absolute zero, not missing). Five 2010 cells are '...' -- the municipalities created after the 2010 lattice.",
            "CEMPRE is a register of formal enterprises and organisations. A municipal hospital run directly by the prefecture is plausibly registered under CNAE section O (public administration), not 86.1, so this undercounts public supply exactly where it matters most. Validate against CNES ST establishment counts at 2009 and 2024 before trusting the level; the year-to-year variation is the usable part.",
            "It counts establishments, never beds. It cannot substitute for CNES.",
            "Two agregados are needed and must be stitched: 6450 publishes 2006-2021, 9528 publishes 2022-2024 (verified live -- 2024 is now out, the compendium's '2022-2023' is stale). Together they cover 2007-2024 of the 2007-2025 study window; 2025 has no CEMPRE.",
        ],
    },
    {
        "id": "fasfil_nonprofit_hospitals",
        "block": "health_capacity",
        "priority": 3,
        "agregado": 6916,
        "table_name": "FASFIL - private foundations and nonprofits by detailed activity",
        "research": "FASFIL",
        "variables": ["2129", "2130"],
        "clsf": {"510": ["128557", "128524", "128525"]},
        "clsf_names": {"510": "128525 = Hospitais; 128524 = Saúde; 128557 = Total"},
        "level": "N6",
        "periods": ["2010", "2013", "2016"],
        "granularity": "municipality, 2010/2013/2016 only",
        "estimand": "count of nonprofit hospital entities and their employed persons",
        "derive": None,
        "buys": (
            "Santas Casas carry a large share of inpatient capacity in small Brazilian "
            "municipalities, so nonprofit hospital presence is a real component of the "
            "supply that CEMPRE's formal-enterprise view and CNES's establishment view "
            "each see only partly."
        ),
        "status": "available_unused",
        "effort": "moderate",
        "caveats": [
            "Nonprofit sector only. Not a measure of total hospital supply.",
            "Three irregular anchors; T4_LOW_PRIORITY in the compendium.",
            "Locs 5559 in the compendium, not 5570 -- lattice mismatch to check on join.",
        ],
    },
    # -- 2. DEPRIVATION AND INCOME -----------------------------------------
    {
        "id": "median_pc_household_income_2022",
        "block": "deprivation_income",
        "priority": 1,
        "agregado": 10295,
        "table_name": "Censo 2022 - per-capita household income, mean and median",
        "research": "Censo Demográfico",
        "variables": ["13534", "13431"],
        "clsf": {"2": ["6794"], "86": ["95251"], "58": ["95253"]},
        "clsf_names": {
            "2": "Sexo (pinned Total 6794)",
            "86": "Cor ou raça (pinned Total 95251)",
            "58": "Grupo de idade (pinned Total 95253)",
        },
        "level": "N6",
        "periods": ["2022"],
        "granularity": "municipality, census anchor 2022",
        "estimand": "13534 = valor do rendimento nominal MEDIANO mensal domiciliar per capita, in current 2022 R$; 13431 is the mean, kept only as a skew diagnostic",
        "derive": {"kind": "level", "variable": "13534", "clsf": None, "category": None},
        "buys": (
            "The direct replacement for GDP per capita. GDP per capita is production "
            "located at the plant: a municipality with a dam, a refinery or a mine "
            "reads rich and lives poor, and leptospirosis is a disease of living poor. "
            "The median rather than the mean because municipal income distributions "
            "are right-skewed and the mean tracks the top of the distribution."
        ),
        "status": "already_extracted",
        "effort": "already_extracted",
        "extracted_at": "data/interim/sidra/income_10295.parquet",
        "caveats": [
            "Nominal current reais at census date. Comparing 2010 with 2022 requires deflation by the same national index already used for real GDP per capita.",
            "The universe excludes pensionistas, empregados domésticos and their relatives -- correct for a per-capita household measure, but not identical to the resident population denominator.",
            "Already on disk (11,140 rows, 5,570 municipalities, all OK) and never joined to any panel.",
        ],
    },
    {
        "id": "median_pc_household_income_2010",
        "block": "deprivation_income",
        "priority": 1,
        "agregado": 3578,
        "table_name": "Censo 2010 - residents and per-capita household income by situation and income class",
        "research": "Censo Demográfico",
        "variables": ["2013", "2012"],
        "clsf": {"1": ["6795"], "386": ["9680"]},
        "clsf_names": {
            "1": "Situação do domicílio (pinned Total 6795)",
            "386": "Classes de rendimento per capita (pinned Total 9680)",
        },
        "level": "N6",
        "periods": ["2010"],
        "granularity": "municipality, census anchor 2010 (5,565 municipalities on the 2010 lattice)",
        "estimand": "2013 = valor do rendimento nominal MEDIANO mensal domiciliar per capita, current 2010 R$",
        "derive": {"kind": "level", "variable": "2013", "clsf": None, "category": None},
        "buys": (
            "The 2010 anchor that makes the 2022 median usable as a panel covariate "
            "rather than a cross-section. Person-weighted, matching 10295's definition; "
            "table 3563 gives the same statistic household-weighted and is the wrong "
            "one to pair with 10295."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "5,565 municipalities on the 2010 lattice; the five created since need the same transfer algebra ADR-001 already applies to urban share and sewer coverage.",
            "Nominal 2010 R$; deflate before differencing against 2022.",
            "There is NO per-capita median at 2000. Table 2035 publishes household-total median (var 848) only. This covariate has two anchors, not three.",
        ],
    },
    {
        "id": "share_below_quarter_mw_2022",
        "block": "deprivation_income",
        "priority": 1,
        "agregado": 10296,
        "table_name": "Censo 2022 - population by per-capita household income class",
        "research": "Censo Demográfico",
        "variables": ["13604"],
        "clsf": {
            "386": [
                "9680", "9681", "9682", "9683", "9684", "9685",
                "9686", "9687", "9688", "9689", "9690", "9692",
            ],
            "2": ["6794"],
            "86": ["95251"],
        },
        "clsf_names": {
            "386": "Classes de rendimento nominal mensal domiciliar per capita; 9680 Total, 9681 Até 1/4 SM, 9692 Sem rendimento",
            "2": "Sexo (pinned Total)",
            "86": "Cor ou raça (pinned Total)",
        },
        "level": "N6",
        "periods": ["2022"],
        "granularity": "municipality, census anchor 2022",
        "estimand": "share below 1/4 minimum wage = 13604[9681] / 13604[9680]",
        "derive": {"kind": "share", "variable": "13604", "clsf": "386", "numerator": ["9681"], "total": "9680"},
        "margin": {"clsf": "386", "total": "9680", "components": ["9681", "9682", "9683", "9684", "9685", "9686", "9687", "9688", "9689", "9690", "9692"]},
        "buys": (
            "Extreme-poverty headcount. Median income is a central-tendency measure and "
            "leptospirosis risk is concentrated in the lower tail: the fraction of the "
            "population living in the conditions that produce rodent contact is a "
            "different covariate from the middle of the income distribution, and the "
            "two are not collinear across 5,570 municipalities. Requesting all twelve "
            "classes also yields the whole distribution and permits a margin check."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "The line is relative to the minimum wage at census date. The real minimum wage rose substantially between 2010 and 2022, so '1/4 SM' is a HIGHER real line in 2022. This is not a fixed-real-line poverty rate and must not be described as one.",
            "9692 'Sem rendimento' is a separate class, not inside 9681. Whether zero-income persons belong in a deprivation numerator is a modelling decision that has to be made explicitly.",
        ],
    },
    {
        "id": "share_below_quarter_mw_2010",
        "block": "deprivation_income",
        "priority": 1,
        "agregado": 3578,
        "table_name": "Censo 2010 - residents by per-capita household income class",
        "research": "Censo Demográfico",
        "variables": ["2035"],
        "clsf": {
            "386": ["9680", "12009", "12010", "9682", "9683", "9684", "9685", "9686", "9687", "9691", "9692"],
            "1": ["6795"],
        },
        "clsf_names": {
            "386": "Classes de rendimento per capita; NOTE 2010 splits the lowest band into 12009 (até 1/8 SM) and 12010 (>1/8 a 1/4 SM)",
            "1": "Situação do domicílio (pinned Total 6795)",
        },
        "level": "N6",
        "periods": ["2010"],
        "granularity": "municipality, census anchor 2010 (5,565 municipalities)",
        "estimand": "share below 1/4 minimum wage = (2035[12009] + 2035[12010]) / 2035[9680]",
        "derive": {"kind": "share", "variable": "2035", "clsf": "386", "numerator": ["12009", "12010"], "total": "9680"},
        "margin": {"clsf": "386", "total": "9680", "components": ["12009", "12010", "9682", "9683", "9684", "9685", "9686", "9687", "9691", "9692"]},
        "buys": "The 2010 anchor for the extreme-poverty headcount.",
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "THE TRAP: 2010 has no single 'Até 1/4 SM' category. Category 9681 does not exist in this table's version of Clsf 386; the band is split into 12009 and 12010 and must be summed. A naive crosswalk by category id between 3578 and 10296 silently produces a 1/8-SM series for 2010 and a 1/4-SM series for 2022.",
            "The top band also differs (9691 '>10 SM' in 2010 vs 9688/9689/9690 in 2022); only the bottom of the distribution is like-for-like.",
        ],
    },
    # -- 3. CROWDING AND HOUSING -------------------------------------------
    {
        "id": "residents_per_bedroom_2022",
        "block": "crowding_housing",
        "priority": 1,
        "agregado": 9933,
        "table_name": "Censo 2022 - occupied households by residents per bedroom, tenure and dwelling type",
        "research": "Censo Demográfico",
        "variables": ["381"],
        "clsf": {
            "1975": ["73086", "73087", "73088", "73089", "73090"],
            "63": ["95826"],
            "125": ["2932"],
        },
        "clsf_names": {
            "1975": "Moradores por cômodo utilizado como dormitório; 73086 Total, 73089 >2 a 3, 73090 >3",
            "63": "Condição de ocupação (pinned Total 95826)",
            "125": "Tipo de domicílio (pinned Total 2932)",
        },
        "level": "N6",
        "periods": ["2022"],
        "granularity": "municipality, census anchor 2022",
        "estimand": "crowding share = (381[73089] + 381[73090]) / 381[73086] -- households with more than 2 residents per bedroom",
        "derive": {"kind": "share", "variable": "381", "clsf": "1975", "numerator": ["73089", "73090"], "total": "73086"},
        "margin": {"clsf": "1975", "total": "73086", "components": ["73087", "73088", "73089", "73090"]},
        "buys": (
            "Crowding is the classic household-level leptospirosis covariate and the "
            "panel has nothing like it. It is not redundant with sewer coverage: "
            "sewerage is a network attribute of the territory, crowding is an "
            "attribute of the dwelling, and the two dissociate sharply in dense "
            "peri-urban settlements with piped sewerage. It is also the covariate a "
            "reviewer expects to see when the paper claims a poverty gradient."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "2022 only. IBGE published residents-per-bedroom for the first time in 2022; there is no 2010 equivalent at municipality level, so this is a single-anchor covariate and cannot be time-varying.",
            "Clsf 63 has nivel-2 children (73126/4343 under 73554; 73127-73129 under 73553). Pinned to Total here, but any later use of the tenure axis must not sum parents with children.",
            "The 2010 near-analogue is mean household size, obtainable from agregado 9860 vars 382/381 at both 2010 and 2022 on one definition -- coarser, but two-anchor. See household_size_2010_2022 below.",
        ],
    },
    {
        "id": "household_size_2010_2022",
        "block": "crowding_housing",
        "priority": 2,
        "agregado": 9860,
        "table_name": "Censo 2010/2022 - double universe matrix, households and residents",
        "research": "Censo Demográfico",
        "variables": ["381", "382"],
        "clsf": {
            "125": ["2932"], "1821": ["72129"], "1817": ["72125"],
            "458": ["72117"], "11558": ["46292"], "2661": ["32776"],
        },
        "clsf_names": {"all": "every classification pinned to its Total; only the margin is wanted"},
        "level": "N6",
        "periods": ["2010", "2022"],
        "granularity": "municipality x {2010, 2022}",
        "estimand": "mean household size = 382 / 381, on one definition at both anchors",
        "derive": None,
        "buys": (
            "The two-anchor crowding proxy that residents-per-bedroom cannot be. "
            "Because 9860 publishes both censuses on the same universe it also gives a "
            "like-for-like household-count denominator for every other 2010-vs-2022 "
            "housing comparison the study makes."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "9860 is already in registry.yaml and its metadata is cached, but no values have been fetched. It is registered, not used.",
            "High-dimensional table; pin every classification to Total or the plan blows the cell budget.",
        ],
    },
    {
        "id": "precarious_dwelling_type_2022",
        "block": "crowding_housing",
        "priority": 2,
        "agregado": 6326,
        "table_name": "Censo 2022 - occupied households by dwelling type",
        "research": "Censo Demográfico",
        "variables": ["381"],
        "clsf": {"125": ["2932", "6815", "121264", "3247", "71975", "71976", "71977"]},
        "clsf_names": {
            "125": "Tipo de domicílio; 71975 casa de cômodos ou cortiço, 71977 estrutura residencial degradada ou inacabada",
        },
        "level": "N6",
        "periods": ["2022"],
        "granularity": "municipality, census anchor 2022",
        "estimand": "precarious share = (381[71975] + 381[71977]) / 381[2932]",
        "derive": {"kind": "share", "variable": "381", "clsf": "125", "numerator": ["71975", "71977"], "total": "2932"},
        "margin": {"clsf": "125", "total": "2932", "components": ["6815", "121264", "3247", "71975", "71976", "71977"]},
        "buys": (
            "The cortiço is the named urban leptospirosis habitat in the Brazilian "
            "literature, and 2022 is the first census to count it as a dwelling type "
            "nationally. Unlike the favela block this is a full municipal universe -- "
            "every municipality has a value -- so it complements 9883/9887 rather than "
            "duplicating them."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "2022 only; the 2010 type-of-dwelling classification has no cortiço or degraded-structure category.",
            "Counts are small in most municipalities; expect a zero-inflated covariate that behaves better as a binary or as a log1p rate.",
        ],
    },
    # -- 4. EDUCATION AND LABOUR -------------------------------------------
    {
        "id": "agriculture_employment_share_2010",
        "block": "education_labour",
        "priority": 1,
        "agregado": 1575,
        "table_name": "Censo 2010 - occupied persons 10+ by section of activity of the main job",
        "research": "Censo Demográfico",
        "variables": ["696"],
        "clsf": {"11805": ["0", "12640"]},
        "clsf_names": {"11805": "Seção de atividade do trabalho principal; 0 Total, 12640 Agricultura, pecuária, produção florestal, pesca e aquicultura"},
        "level": "N6",
        "periods": ["2010"],
        "granularity": "municipality, census anchor 2010",
        "estimand": "agricultural employment share = 696[12640] / 696[0]",
        "derive": {"kind": "share", "variable": "696", "clsf": "11805", "numerator": ["12640"], "total": "0"},
        "buys": (
            "The most direct occupational-exposure proxy available at municipal scale. "
            "Rice-paddy and livestock work is the canonical rural transmission route, "
            "and PAM/PPM (already extracted) measure the *land and the herd*, not the "
            "people exposed to them. A municipality can have large sugarcane area and "
            "almost no resident agricultural workforce, or the reverse; the two "
            "covariates are not substitutes."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "Agriculture-section employment counts formal and informal workers alike, which is the point -- CEMPRE's CNAE section A counts only formal establishments and understates rural work severely.",
            "Not catalogued in SIDRA_COMPENDIUM.md; the compendium has no census occupation-by-sector table at all.",
            "Sample-based (Resultados Gerais da Amostra); small municipalities carry sampling error that the universe tables do not.",
        ],
    },
    {
        "id": "agriculture_employment_share_2022",
        "block": "education_labour",
        "priority": 1,
        "agregado": 10262,
        "table_name": "Censo 2022 - occupied persons 14+ by position in occupation and section of activity",
        "research": "Censo Demográfico",
        "variables": ["4090"],
        "clsf": {"11805": ["95371", "12640"], "11913": ["96165"]},
        "clsf_names": {
            "11805": "Seção de atividade do trabalho principal; NOTE Total is 95371 here, not 0",
            "11913": "Posição na ocupação (pinned Total 96165)",
        },
        "level": "N6",
        "periods": ["2022"],
        "granularity": "municipality, census anchor 2022",
        "estimand": "agricultural employment share = 4090[12640] / 4090[95371]",
        "derive": {"kind": "share", "variable": "4090", "clsf": "11805", "numerator": ["12640"], "total": "95371"},
        "buys": "The 2022 anchor that makes the agricultural share a panel covariate.",
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "THE TRAP: the section-of-activity Total is category 0 in the 2010 table and 95371 in the 2022 table, while the agriculture member is 12640 in both. Crosswalking the Total by id fails silently.",
            "Age base differs: 10+ in 2010, 14+ in 2022. The SHARE is comparatively robust to this; the count is not.",
            "Not catalogued in SIDRA_COMPENDIUM.md.",
        ],
    },
    {
        "id": "low_education_share_2010_2022",
        "block": "education_labour",
        "priority": 2,
        "agregado": 3547,
        "table_name": "Censo 2010 - persons 25+ by sex and education (pair with 10141 for 2022)",
        "research": "Censo Demográfico",
        "variables": ["1643"],
        "clsf": {"1568": ["0", "9493", "9494", "9495", "99713", "11626"], "2": ["0"]},
        "clsf_names": {"1568": "Nível de instrução; 9493 Sem instrução e fundamental incompleto", "2": "Sexo (pinned Total)"},
        "level": "N6",
        "periods": ["2010"],
        "granularity": "municipality x {2010 via 3547, 2022 via 10141}",
        "estimand": "low-education share = 1643[9493] / 1643[Total]; Total is 0 in 3547 and 120704 in 10141",
        "derive": {"kind": "share", "variable": "1643", "clsf": "1568", "numerator": ["9493"], "total": "0"},
        "buys": (
            "A two-anchor deprivation axis that is not income and not sanitation. It "
            "matters specifically for the fatality thesis: education predicts time to "
            "care-seeking, and late presentation is the proximate cause of "
            "leptospirosis death independent of how good the hospital is."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "Total category id differs between anchors (0 in 3547, 120704 in 10141); the member ids 9493-99713 are shared.",
            "3547 carries an extra 'Não determinado' member (11626) that 10141 does not; the denominators are therefore not identically constructed.",
            "For 2022 pin Clsf 839 (deficiência) to 46583 as well.",
        ],
    },
    {
        "id": "labour_force_participation_2022",
        "block": "education_labour",
        "priority": 3,
        "agregado": 9517,
        "table_name": "Censo 2022 - persons 14+ by labour force status, sex, race and education",
        "research": "Censo Demográfico",
        "variables": ["1641"],
        "clsf": {"629": ["32385", "32386", "32387", "32446", "32447"], "2": ["6794"], "86": ["95251"], "1568": ["120704"]},
        "clsf_names": {"629": "Condição em relação à força de trabalho; 32446 desocupada"},
        "level": "N6",
        "periods": ["2022"],
        "granularity": "municipality, census anchor 2022",
        "estimand": "unemployment share = 1641[32446] / 1641[32386]",
        "derive": None,
        "buys": "Labour-market slack as a deprivation axis distinct from income level; weakest of the four in this block and listed for completeness.",
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": ["2022 only; the 2010 analogue (3572/616) uses a different activity concept and is not directly comparable."],
    },
    # -- 5. THE QUEUED PAPERS ----------------------------------------------
    {
        "id": "refuse_not_collected_2022",
        "block": "queued_papers",
        "priority": 1,
        "agregado": 6892,
        "table_name": "Censo 2022 - occupied households by refuse destination",
        "research": "Censo Demográfico",
        "variables": ["381"],
        "clsf": {"67": ["10972", "2520", "72120", "72121", "72122", "72123", "72124", "1091"]},
        "clsf_names": {"67": "Destino do lixo; 2520 Coletado is the PARENT of 72120+72121; 72124 jogado em terreno baldio"},
        "level": "N6",
        "periods": ["2022"],
        "granularity": "municipality, census anchor 2022",
        "estimand": "uncollected refuse share = (381[10972] - 381[2520]) / 381[10972]; dumped-in-public share = 381[72124] / 381[10972]",
        "derive": {"kind": "share", "variable": "381", "clsf": "67", "numerator": ["72124"], "total": "10972"},
        "margin": {"clsf": "67", "total": "10972", "components": ["72120", "72121", "72122", "72123", "72124", "1091"]},
        "buys": (
            "Uncollected refuse is the rodent-attractant covariate, and it is the one "
            "the rainfall x sanitation paper most obviously needs and does not have. "
            "Sewer coverage indexes contact with contaminated water once it is flowing; "
            "refuse indexes the reservoir density that makes the water infectious in "
            "the first place. They are different links in the same chain and the paper "
            "currently conflates them under one 'sanitation' modifier."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "6892 is in registry.yaml with cached metadata but no values fetched. Registered, not used.",
            "2520 Coletado is a parent of 72120 and 72121; summing all three double-counts. Compute uncollected as Total minus 2520.",
            "Three anchors exist (1439 for 2000, 3218 for 2010, 6892 for 2022) but the category sets differ in depth; only 'collected vs not' is comparable across all three.",
        ],
    },
    {
        "id": "refuse_not_collected_2010",
        "block": "queued_papers",
        "priority": 2,
        "agregado": 3218,
        "table_name": "Censo 2010 - water, sewage, refuse and electricity cross-matrix",
        "research": "Censo Demográfico",
        "variables": ["96"],
        "clsf": {"67": ["0", "2520", "1091"], "61": ["0"], "299": ["0"], "309": ["0"]},
        "clsf_names": {"67": "Destino do lixo (2010 depth: Total / Coletado / Outro destino)", "61": "pinned Total", "299": "pinned Total", "309": "pinned Total"},
        "level": "N6",
        "periods": ["2010"],
        "granularity": "municipality, census anchor 2010 (5,565 municipalities)",
        "estimand": "uncollected refuse share = (96[0] - 96[2520]) / 96[0]",
        "derive": {"kind": "share", "variable": "96", "clsf": "67", "numerator": ["1091"], "total": "0"},
        "buys": "The 2010 anchor that turns refuse collection into a time-varying (interpolated) modifier rather than a 2022 cross-section.",
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "3218 is registered with cached metadata and no fetched values.",
            "High-dimensional: pin 61, 299 and 309 to Total or the plan will not fit.",
        ],
    },
    {
        "id": "water_no_grid_connection_2022",
        "block": "queued_papers",
        "priority": 1,
        "agregado": 6803,
        "table_name": "Censo 2022 - water grid connection and main supply",
        "research": "Censo Demográfico",
        "variables": ["381"],
        "clsf": {"1821": ["72129", "72144", "72145", "72153", "72150", "72151", "72158", "72159"]},
        "clsf_names": {
            "1821": "Ligação à rede e principal forma; 72153 não possui ligação (parent of 72154-72160); 72150/72158 água de chuva; 72151/72159 rios, lagos, igarapés",
        },
        "level": "N6",
        "periods": ["2022"],
        "granularity": "municipality, census anchor 2022",
        "estimand": "no-grid share = 381[72153] / 381[72129]; surface/rainwater-sourced share = (381[72150]+381[72151]+381[72158]+381[72159]) / 381[72129]",
        "derive": {"kind": "share", "variable": "381", "clsf": "1821", "numerator": ["72153"], "total": "72129"},
        "buys": (
            "The rainfall paper's dose-response runs through water. Households whose "
            "main supply is rainwater or a river are the ones for whom a rainfall "
            "anomaly is an exposure event rather than a nuisance, and that is a "
            "sharper interaction term than sewer coverage. The panel currently has "
            "sewerage and no water variable at all."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "6803 registered, metadata cached, values never fetched.",
            "Three-level hierarchy: 72145 and 72153 are each parents of seven sub-forms. Summing parents with children double-counts. The rainwater/river members appear TWICE, once under 'has connection but uses another form' (72150/72151) and once under 'no connection' (72158/72159); both are needed for the source-based share.",
            "Companion 6804 crosses main supply with piping existence (Clsf 1817, 72128 = sem água canalizada) and is the better table if the estimand is in-dwelling piping.",
        ],
    },
    {
        "id": "favela_water_and_refuse_2022",
        "block": "queued_papers",
        "priority": 2,
        "agregado": 9894,
        "table_name": "Censo 2022 - favela households by water grid connection (9894); refuse by 9893; density by 9888",
        "research": "Censo Demográfico",
        "variables": ["9913"],
        "clsf": {"1821": ["72129", "72144", "72153"]},
        "clsf_names": {"1821": "as 6803, restricted to households in favelas"},
        "level": "N6",
        "periods": ["2022"],
        "granularity": "municipality (aggregate of settlements), 2022; municipalities without a settlement are ABSENT, not zero",
        "estimand": "within-favela no-grid water share; pairs with 6803 for the outside-favela contrast",
        "derive": None,
        "buys": (
            "The favela block is extracted only for count, population and sewage "
            "(9883, 9887, 10344). 9888 (density), 9892 (bathroom/sewage), 9893 "
            "(refuse) and 9894 (water) are registered and unfetched -- so the "
            "settlement profile the queued paper will lean on is a quarter built. "
            "9888's density in particular is the exposure gradient Reis and Hagan "
            "describe and it is one extraction away."
        ),
        "status": "available_unused",
        "effort": "cheap_extract",
        "caveats": [
            "9888/9892/9893/9894 are all registered in registry.yaml with cached metadata and zero fetched values.",
            "9888 var 13096 (density) is a ratio and not additive; recompute from 9612 and 9911 on any aggregation.",
            "Absence means no settlement, not zero households; the join must preserve the distinction.",
        ],
    },
    {
        "id": "subnormal_inside_outside_2010",
        "block": "queued_papers",
        "priority": 1,
        "agregado": 4011,
        "table_name": "Censo 2010 - households by per-capita income class and SECTOR TYPE (aglomerado subnormal vs other areas)",
        "research": "Censo Demográfico",
        "variables": ["96"],
        "clsf": {"386": ["0", "9681", "9682", "9683"], "11339": ["0", "110", "111"]},
        "clsf_names": {
            "386": "Classes de rendimento per capita",
            "11339": "Tipo do setor; 110 Aglomerados subnormais, 111 Outras áreas",
        },
        "level": "N6",
        "periods": ["2010"],
        "granularity": "municipality x sector type, 2010; N6 is the ONLY level this table offers",
        "estimand": "within-municipality contrast between subnormal-agglomerate households and other areas, on the same income classification",
        "derive": None,
        "buys": (
            "This is the 2010 counterpart of table 10344, and its existence changes "
            "what the queued sanitation paper can claim. 10344 gives the "
            "inside-vs-outside contrast at 2022 only, which makes the favela argument "
            "a cross-section. The 2010 'tipo do setor' family gives the same "
            "inside-vs-outside cut for income (4011), bedrooms (4001, 4002), water "
            "(4006), education (3990) and occupation by sector (3993) -- turning the "
            "central within-municipality contrast into a two-anchor difference, which "
            "is a far stronger design against the 'that is poverty, not sanitation' "
            "objection."
        ),
        "status": "available_unused",
        "effort": "moderate",
        "caveats": [
            "The 2010 aglomerado subnormal delimitation and the 2022 favela e comunidade urbana delimitation are DIFFERENT operational definitions on different mapped universes. A 2010-2022 difference is a difference in two related but non-identical constructs and must be argued, not assumed.",
            "Only municipalities with a mapped agglomerate carry a 110 row; the rest are absent, exactly as in the 2022 favela block.",
            "None of the 'tipo do setor' family is in SIDRA_COMPENDIUM.md or registry.yaml. Cataloguing them is a prerequisite, and tests/test_sidra_compendium.py enforces that ordering.",
        ],
    },
    {
        "id": "land_area_density_2022",
        "block": "queued_papers",
        "priority": 3,
        "agregado": 4714,
        "table_name": "Censo 2022 - land area, resident population and demographic density",
        "research": "Censo Demográfico",
        "variables": ["6318", "614"],
        "clsf": {},
        "clsf_names": {},
        "level": "N6",
        "periods": ["2022"],
        "granularity": "municipality, 2022",
        "estimand": "land area km2 (6318) and demographic density (614)",
        "derive": None,
        "buys": "Denominator for every density covariate the regime-clustering paper will build, including PPM herd density which is only a density once divided by area.",
        "status": "already_extracted",
        "effort": "already_extracted",
        "extracted_at": "data/interim/sidra/land_area.parquet, land_area_2022.parquet",
        "caveats": ["Already on disk; listed so the queued papers do not re-derive it."],
    },
]


# ---------------------------------------------------------------------------
# What the study currently uses, read from disk
# ---------------------------------------------------------------------------


def current_usage() -> dict[str, Any]:
    """Registered tables, fetched tables and landed extracts, from the tree."""
    registered: list[str] = []
    if REGISTRY.exists():
        for line in REGISTRY.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if (
                line.startswith("  '")
                and stripped.endswith("':")
                and stripped.strip("':").isdigit()
            ):
                registered.append(stripped.strip("':"))
    values_dir = PATHS.cache / "sidra" / "values"
    fetched = sorted(p.name for p in values_dir.iterdir() if p.is_dir()) if values_dir.exists() else []
    interim = PATHS.interim / "sidra"
    landed = sorted(p.name for p in interim.glob("*.parquet")) if interim.exists() else []
    return {
        "registered_in_registry_yaml": sorted(registered, key=int),
        "n_registered": len(registered),
        "values_fetched_at_least_once": fetched,
        "n_fetched": len(fetched),
        "registered_but_never_fetched": sorted(set(registered) - set(fetched), key=int),
        "landed_interim_parquet": landed,
    }


def health_capacity_verdict(*, probe: bool) -> dict[str, Any]:
    """Can SIDRA answer the hospital-availability objection, or must CNES?

    Probes the live territorial levels of the three AMS tables and of the CEMPRE
    and MUNIC candidates, then inventories the CNES cache on disk. No CNES
    download is performed.
    """
    sidra: dict[str, Any] = {}
    if probe:
        for agregado, note in [
            (211, "AMS: Estabelecimentos de saúde"),
            (215, "AMS: Estabelecimentos por tipo de atendimento e esfera"),
            (216, "AMS: Número de leitos para internação por esfera"),
            (6450, "CEMPRE por CNAE 2.0, 2006-2021"),
            (9528, "CEMPRE por CNAE 2.0, 2022-2023"),
            (9487, "MUNIC 2021 serviços e tipos de leitos"),
            (9493, "MUNIC 2021 vigilância em saúde"),
            (6916, "FASFIL por classificação de atividade"),
        ]:
            meta = api.get_metadata(agregado)
            levels = (meta.get("nivelTerritorial") or {}).get("Administrativo", [])
            sidra[str(agregado)] = {
                "name": meta.get("nome", "")[:160],
                "note": note,
                "administrative_levels": levels,
                "has_municipality_level": "N6" in levels,
                "periods": [str(p["id"]) for p in api.get_periods(agregado)],
                "variables": [
                    {"id": str(v["id"]), "name": v["nome"][:90], "unit": v.get("unidade")}
                    for v in meta.get("variaveis", [])
                ],
            }

    cnes: dict[str, Any] = {"cached": CNES_CACHE.exists(), "groups": {}}
    if CNES_CACHE.exists():
        total_bytes = 0
        for group_dir in sorted(p for p in CNES_CACHE.iterdir() if p.is_dir()):
            files = [p for p in group_dir.iterdir() if p.suffix.upper() == ".DBC"]
            competences = Counter(p.stem[4:] for p in files)
            ufs = sorted({p.stem[2:4] for p in files})
            size = sum(p.stat().st_size for p in files)
            total_bytes += size
            cnes["groups"][group_dir.name] = {
                "n_files": len(files),
                "n_ufs": len(ufs),
                "competences_yymm": dict(sorted(competences.items())),
                "bytes": size,
            }
        cnes["total_bytes"] = total_bytes
    return {"sidra": sidra, "cnes_cache": cnes}


# ---------------------------------------------------------------------------
# Selection construction and the exact call string
# ---------------------------------------------------------------------------


def selection_for(entry: dict[str, Any]) -> Selection:
    return Selection(
        agregado=entry["agregado"],
        periods=list(entry["periods"]),
        variables=list(entry["variables"]),
        level=entry["level"],
        classifications={k: list(v) for k, v in entry["clsf"].items()},
        label=f"spec_{entry['id']}",
    )


def call_string(entry: dict[str, Any]) -> str:
    """The literal Selection(...) an analyst should paste into a build script."""
    clsf = entry["clsf"]
    clsf_src = (
        "{}"
        if not clsf
        else "{\n        "
        + ",\n        ".join(
            f'"{k}": [{", ".join(chr(34) + c + chr(34) for c in v)}]' for k, v in clsf.items()
        )
        + ",\n    }"
    )
    return (
        "Selection(\n"
        f"    agregado={entry['agregado']},\n"
        f"    periods={entry['periods']!r},\n"
        f"    variables={list(entry['variables'])!r},\n"
        f'    level="{entry["level"]}",\n'
        f"    classifications={clsf_src},\n"
        f'    label="{entry["id"]}",\n'
        ")"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

#: Extracted live at N6 before being recommended. Chosen for priority and for
#: the questions they settle: does MUNIC really publish per-municipality health
#: capacity; does the 2010 income table survive its split lowest band; does
#: CEMPRE's hospital series survive confidentiality suppression.
VALIDATE = [
    "munic2021_nephrology_service",
    "munic2021_health_surveillance",
    "share_below_quarter_mw_2022",
    "share_below_quarter_mw_2010",
    "median_pc_household_income_2010",
    "residents_per_bedroom_2022",
    "agriculture_employment_share_2010",
    "agriculture_employment_share_2022",
    "cempre_hospital_activity_units",
]


def _axis_category(facts: pl.DataFrame, clsf: str) -> pl.DataFrame:
    """Attach the category id taken on one classification axis."""
    return (
        facts.with_row_index("_row")
        .explode(["classification_ids", "category_ids"])
        .filter(pl.col("classification_ids") == clsf)
        .select("_row", pl.col("category_ids").alias("_cat"))
    )


def derive_indicator(facts: pl.DataFrame, entry: dict[str, Any]) -> pl.DataFrame | None:
    """Compute the stated estimand, municipality by municipality."""
    spec = entry.get("derive")
    if not spec or facts.height == 0:
        return None
    var = spec["variable"]
    base = facts.filter(pl.col("variable_id") == var)
    if base.height == 0:
        return None
    name = entry["id"]

    if spec["kind"] == "level" and not spec.get("clsf"):
        return base.select(
            pl.col("locality_id").alias("munic_code"),
            pl.col("period").alias("year"),
            pl.lit(name).alias("indicator"),
            pl.col("value_numeric").alias("value"),
            pl.col("value_status"),
        )

    clsf = spec["clsf"]
    tagged = base.with_row_index("_row").join(_axis_category(base, clsf), on="_row", how="inner")

    if spec["kind"] == "level":
        return tagged.filter(pl.col("_cat") == spec["category"]).select(
            pl.col("locality_id").alias("munic_code"),
            pl.col("period").alias("year"),
            pl.lit(name).alias("indicator"),
            pl.col("value_numeric").alias("value"),
            pl.col("value_status"),
        )

    numerator = spec["numerator"] if spec["kind"] == "share" else [spec["yes"]]
    total = spec["total"]
    num = (
        tagged.filter(pl.col("_cat").is_in(numerator))
        .group_by(["locality_id", "period"])
        .agg(pl.col("value_numeric").sum().alias("num"))
    )
    den = (
        tagged.filter(pl.col("_cat") == total)
        .group_by(["locality_id", "period"])
        .agg(pl.col("value_numeric").sum().alias("den"))
    )
    joined = num.join(den, on=["locality_id", "period"], how="full", coalesce=True)
    return joined.select(
        pl.col("locality_id").alias("munic_code"),
        pl.col("period").alias("year"),
        pl.lit(name).alias("indicator"),
        pl.when(pl.col("den") > 0)
        .then(pl.col("num") / pl.col("den"))
        .otherwise(None)
        .alias("value"),
        pl.lit("DERIVED").alias("value_status"),
    )


def summarise(series: pl.Series) -> dict[str, Any]:
    clean = series.drop_nulls()
    if clean.len() == 0:
        return {"n": 0}
    return {
        "n": int(clean.len()),
        "n_null": int(series.len() - clean.len()),
        "min": float(clean.min()),
        "p10": float(clean.quantile(0.10)),
        "median": float(clean.median()),
        "p90": float(clean.quantile(0.90)),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
    }


def validate_entry(entry: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    selection = selection_for(entry)
    report: dict[str, Any] = {"id": entry["id"], "agregado": entry["agregado"]}
    try:
        localities = resolve_localities(selection)
        result = extract(selection, write=True, out_dir=out_dir)
    except Exception as exc:  # noqa: BLE001 -- the failure is the finding
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    facts = result.facts
    report["ok"] = True
    report["n_localities_offered_at_N6"] = len(localities)
    report["rows_returned"] = facts.height
    report["rows_planned"] = result.report["rows_planned"]
    report["row_surplus"] = result.report["row_surplus"]
    report["n_requests"] = result.report["plan"]["n_requests"]
    report["value_status"] = value_status_summary(facts)
    report["n_value_numeric_null"] = int(facts["value_numeric"].is_null().sum())
    report["coverage"] = coverage_report(facts, localities)
    report["parquet"] = str(result.parquet_path)

    margin = entry.get("margin")
    if margin:
        checked = check_margins(
            facts,
            margin["total"],
            margin["components"],
            classification=margin["clsf"],
        )
        summary = {
            k: v for k, v in checked.items() if k not in {"mismatches", "incomplete_examples"}
        }
        # A margin can fail on rounding alone. The absolute residual is what
        # separates "IBGE rounded the published cells" from "the category
        # selection is wrong", and only the second is a defect.
        summary["worst_absolute_differences"] = [
            {"locality_id": m["locality_id"], "total": m["total_value"], "difference": m["difference"]}
            for m in checked.get("mismatches", [])[:5]
        ]
        report["margin_check"] = summary

    derived = derive_indicator(facts, entry)
    if derived is not None and derived.height:
        report["derived"] = {
            "n_municipalities": int(derived["munic_code"].n_unique()),
            "summary": summarise(derived["value"]),
        }
        derived.write_parquet(out_dir / f"indicator_{entry['id']}.parquet")
        report["indicator_parquet"] = str(out_dir / f"indicator_{entry['id']}.parquet")
    return report


# ---------------------------------------------------------------------------
# Diagnostics: the questions the extraction was run to settle
# ---------------------------------------------------------------------------


def cempre_suppression(extracts: Path) -> dict[str, Any]:
    """How much of the CEMPRE hospital series survives confidentiality.

    An annual health-capacity control is only worth building if the cells exist.
    CEMPRE withholds counts that would identify an informant, and the smallest
    municipalities are exactly where a single hospital is decisive, so the
    suppression pattern determines whether this covariate is usable at all.
    """
    path = extracts / "spec_cempre_hospital_activity_units.parquet"
    if not path.exists():
        return {"available": False}
    facts = (
        pl.read_parquet(path)
        .with_row_index("_row")
        .explode(["classification_ids", "category_ids"])
        .filter(pl.col("classification_ids") == "12762")
    )
    by_cell = (
        facts.group_by(["period", "variable_id", "category_ids", "value_status"])
        .len()
        .sort(["period", "variable_id", "category_ids", "value_status"])
    )
    units_861 = facts.filter(
        (pl.col("variable_id") == "706") & (pl.col("category_ids") == "117812")
    )
    pop = (
        pl.read_parquet(PATHS.interim / "population_municipal_year.parquet")
        .filter(pl.col("year") == 2019)
        .select("munic_code", "population")
    )
    joined = units_861.filter(pl.col("period") == "2019").join(
        pop, left_on="locality_id", right_on="munic_code", how="left"
    )
    by_size = (
        joined.with_columns(
            pl.when(pl.col("population") < 10_000)
            .then(pl.lit("a_under_10k"))
            .when(pl.col("population") < 50_000)
            .then(pl.lit("b_10k_50k"))
            .when(pl.col("population") < 200_000)
            .then(pl.lit("c_50k_200k"))
            .otherwise(pl.lit("d_200k_plus"))
            .alias("size_band")
        )
        .group_by("size_band")
        .agg(
            pl.len().alias("n_municipalities"),
            (pl.col("value_status") == "SUPPRESSED").sum().alias("n_suppressed"),
            (pl.col("value_status") == "ABSOLUTE_ZERO").sum().alias("n_absolute_zero"),
            (pl.col("value_status") == "OK").sum().alias("n_ok"),
        )
        .sort("size_band")
    )
    return {
        "available": True,
        "cell_status_by_period_variable_category": by_cell.to_dicts(),
        "cnae_861_units_2019_by_population_band": by_size.to_dicts(),
        "reading": (
            "SUPPRESSED is SIDRA 'X' -- withheld to protect an informant. A "
            "suppressed cell is not a zero and imputing it as one manufactures "
            "the absence of a hospital."
        ),
    }


def munic_category_shares(extracts: Path) -> dict[str, Any]:
    """Prevalence of each MUNIC category, i.e. how much variance is on offer."""
    out: dict[str, Any] = {}
    for spec_id, clsf, labels in [
        (
            "munic2021_nephrology_service",
            "1623",
            {"59535": "Total", "59536": "Sim", "59537": "Não"},
        ),
        (
            "munic2021_health_surveillance",
            "1649",
            {
                "59630": "Total",
                "59631": "Sim (any)",
                "59632": "Vigilância sanitária",
                "59633": "Vigilância epidemiológica",
                "59634": "Controle de endemias",
                "59635": "Não realiza nenhum dos serviços",
            },
        ),
    ]:
        path = extracts / f"spec_{spec_id}.parquet"
        if not path.exists():
            continue
        facts = (
            pl.read_parquet(path)
            .with_row_index("_row")
            .explode(["classification_ids", "category_ids"])
            .filter(pl.col("classification_ids") == clsf)
        )
        n_mun = facts["locality_id"].n_unique()
        rows = (
            facts.group_by("category_ids")
            .agg(
                pl.col("value_numeric").sum().alias("n_municipalities_with"),
                pl.len().alias("n_cells"),
            )
            .sort("category_ids")
        )
        out[spec_id] = {
            "n_municipalities": int(n_mun),
            "categories": [
                {
                    "category_id": r["category_ids"],
                    "label": labels.get(r["category_ids"], r["category_ids"]),
                    "n_municipalities": int(r["n_municipalities_with"]),
                    "share": round(r["n_municipalities_with"] / n_mun, 4),
                }
                for r in rows.to_dicts()
            ],
        }
    return out


def relevance_probe(out_dir: Path) -> dict[str, Any]:
    """Is a recommended covariate distinguishable from what the study already has?

    Correlates each validated indicator with the study's own municipal
    surveillance-depth index (hospitalised share of confirmed cases) and with
    municipal case fatality, over municipalities with enough cases for the
    ratios to mean anything. A covariate that reproduces the depth index buys
    nothing; a covariate that predicts fatality but not depth is the one that
    settles the confounding question.
    """
    conf_path = PATHS.interim / "lept_confirmed_mun_month.parquet"
    ind_path = out_dir / "validated_indicators_long.parquet"
    if not (conf_path.exists() and ind_path.exists()):
        return {"available": False}

    min_cases = 30
    outcome = (
        pl.read_parquet(conf_path)
        .group_by("munic_code6")
        .agg(
            pl.col("cases").sum().alias("cases"),
            pl.col("deaths").sum().alias("deaths"),
            pl.col("hospitalised").sum().alias("hospitalised"),
        )
        .filter(pl.col("cases") >= min_cases)
        .with_columns(
            (pl.col("hospitalised") / pl.col("cases")).alias("depth_hosp_share"),
            (pl.col("deaths") / pl.col("cases")).alias("case_fatality"),
        )
    )
    # One value per municipality per indicator. The CEMPRE indicator is
    # extracted for two periods; stacking them would double every municipality
    # and quietly halve the effective sample.
    indicators = (
        pl.read_parquet(ind_path)
        .with_columns(pl.col("munic_code").str.slice(0, 6).alias("munic_code6"))
        .sort(["indicator", "munic_code6", "year"])
        .group_by(["indicator", "munic_code6"], maintain_order=True)
        .agg(pl.col("value").last(), pl.col("year").last())
    )

    results = []
    for name in sorted(indicators["indicator"].unique().to_list()):
        block = (
            indicators.filter(pl.col("indicator") == name)
            .select("munic_code6", "value")
            .drop_nulls()
        )
        merged = outcome.join(block, on="munic_code6", how="inner").drop_nulls(
            ["value", "depth_hosp_share", "case_fatality"]
        )
        if merged.height < 50:
            results.append({"indicator": name, "n": merged.height, "note": "too few municipalities"})
            continue
        results.append(
            {
                "indicator": name,
                "n_municipalities": merged.height,
                "spearman_vs_depth_hosp_share": round(
                    float(
                        merged.select(
                            pl.corr("value", "depth_hosp_share", method="spearman")
                        ).item()
                    ),
                    4,
                ),
                "spearman_vs_case_fatality": round(
                    float(
                        merged.select(
                            pl.corr("value", "case_fatality", method="spearman")
                        ).item()
                    ),
                    4,
                ),
            }
        )
    return {
        "available": True,
        "population": (
            f"municipalities with >= {min_cases} confirmed leptospirosis cases "
            "over 2007-2025, event municipality as resolved in "
            "lept_confirmed_mun_month.parquet"
        ),
        "n_municipalities_in_population": outcome.height,
        "depth_hosp_share_summary": summarise(outcome["depth_hosp_share"]),
        "case_fatality_summary": summarise(outcome["case_fatality"]),
        "correlations": results,
        "reading": (
            "|rho| near zero against depth_hosp_share means the covariate is not a "
            "restatement of the exposure of interest and can be adjusted for "
            "without absorbing it. Correlation with case_fatality here is "
            "unadjusted and descriptive; it sizes the covariate, it does not "
            "estimate an effect."
        ),
    }


def verify_already_extracted() -> dict[str, Any]:
    """The 2022 income table is on disk already; confirm it rather than refetch."""
    path = PATHS.interim / "sidra" / "income_10295.parquet"
    if not path.exists():
        return {"path": str(path), "present": False}
    df = pl.read_parquet(path)
    median = df.filter(pl.col("variable_id") == "13534")
    return {
        "path": str(path),
        "present": True,
        "rows": df.height,
        "n_municipalities": int(df["locality_id"].n_unique()),
        "variables": sorted(df["variable_id"].unique().to_list()),
        "value_status": {k: int(v) for k, v in df["value_status"].value_counts().iter_rows()},
        "median_pc_income_2022_summary": summarise(median["value_numeric"]),
        "used_in_any_panel": False,
        "note": "Extracted 2026-08-02, never joined to a panel. No refetch needed.",
    }


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-only", action="store_true", help="write the spec without hitting the API")
    parser.add_argument("--only", default=None, help="comma-separated spec ids to validate")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    extracts = OUT / "extracts"
    extracts.mkdir(parents=True, exist_ok=True)

    for entry in SPEC:
        entry["selection_call"] = call_string(entry)

    spec_doc = {
        "generated_at": now(),
        "study": "leptospirosis Brazil 2007-2025, 5570 municipalities, 66667 confirmed cases",
        "n_covariates": len(SPEC),
        "blocks": sorted({e["block"] for e in SPEC}),
        "current_usage": current_usage(),
        "already_extracted_check": verify_already_extracted(),
        "covariates": SPEC,
    }

    if not args.spec_only:
        spec_doc["health_capacity_verdict"] = health_capacity_verdict(probe=True)

    (OUT / "covariate_spec.json").write_text(
        json.dumps(spec_doc, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    flat = pl.DataFrame(
        [
            {
                "id": e["id"],
                "block": e["block"],
                "priority": e["priority"],
                "agregado": e["agregado"],
                "variables": ",".join(e["variables"]),
                "classifications": ";".join(
                    f"{k}={'+'.join(v)}" for k, v in e["clsf"].items()
                ),
                "level": e["level"],
                "periods": ",".join(e["periods"]),
                "granularity": e["granularity"],
                "estimand": e["estimand"],
                "status": e["status"],
                "effort": e["effort"],
                "n_caveats": len(e["caveats"]),
            }
            for e in SPEC
        ]
    )
    flat.write_csv(OUT / "covariate_spec.csv")
    print(f"[spec] {len(SPEC)} covariates -> {OUT / 'covariate_spec.json'}")

    if args.spec_only:
        return

    todo = VALIDATE if args.only is None else args.only.split(",")
    by_id = {e["id"]: e for e in SPEC}
    validations = []
    for spec_id in todo:
        entry = by_id[spec_id]
        print(f"[validate] {spec_id} (agregado {entry['agregado']}) ...", flush=True)
        rep = validate_entry(entry, extracts)
        status = "OK" if rep.get("ok") else f"FAILED {rep.get('error')}"
        print(
            f"           {status} rows={rep.get('rows_returned')} "
            f"munis={rep.get('coverage', {}).get('n_present')}",
            flush=True,
        )
        validations.append(rep)

    (OUT / "validation.json").write_text(
        json.dumps(
            {"validated_at": now(), "n_validated": len(validations), "results": validations},
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    indicators = sorted(extracts.glob("indicator_*.parquet"))
    frames = [pl.read_parquet(p) for p in indicators]
    if frames:
        panel = pl.concat(frames, how="vertical_relaxed").sort(["indicator", "munic_code"])
        panel.write_parquet(OUT / "validated_indicators_long.parquet")
        print(f"[panel] {panel.height} rows, {panel['indicator'].n_unique()} indicators")

    diagnostics = {
        "generated_at": now(),
        "cempre_suppression": cempre_suppression(extracts),
        "munic_category_shares": munic_category_shares(extracts),
        "relevance_probe": relevance_probe(OUT),
    }
    (OUT / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[diagnostics] {OUT / 'diagnostics.json'}")

    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
