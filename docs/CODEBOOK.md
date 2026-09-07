# The codebook: translating DATASUS coded values

DATASUS files are almost entirely coded. A record says `CS_SEXO = "M"`,
`EVOLUCAO = "2"`, `MICRO1_S1 = "10 - COPENHAGENI"`, `NU_IDADE_N = "4028"` and
`ID_MN_RESI = "431490"`. Translating those into *sex*, *died of leptospirosis*,
*serovar Copenhageni*, *28 years old* and *Porto Alegre* is not a convenience
step — it is the difference between an analysis and a pile of integers.

This document describes how that is done, what is authoritative, and what is
deliberately left untranslated.

---

## 1. The four decode states

Every translated value carries a companion `<name>_state` column with exactly
one of:

| State | Meaning |
|---|---|
| `missing` | the field is blank, null or whitespace |
| `valid` | the code was found and translated |
| `unknown` | the notifier entered an explicit "ignorado" code |
| `invalid` | the value is not blank and not in the dictionary |

These are **not interchangeable**, and collapsing them is the single most
consequential mistake available here. `missing` means nobody filled the field.
`unknown` means somebody actively recorded that the answer was not known.
`invalid` means the file contains a value the dictionary does not define —
which is a data-quality finding, not a null.

`EVOLUCAO` in this study is 28.3% missing and 7.1% unknown. A pipeline that
maps both to `NA` reports 35.4% missing and loses the fact that a quarter of
that is a different phenomenon.

A consumer cannot use a value without the state being available: both columns
are always emitted together.

---

## 2. Authority

The **curated registry** at `brepi/codebook/registry/datasus_codebook.yaml` is
authoritative. The published PDF data dictionary is a cross-check.

That ordering is deliberate and was forced by evidence:

- The v5.0 PDF's own **text layer is lossy**. The `EVOLUCAO` row extracts as
  `2-bito por leptospirose 3 bito por outras causas` — the hyphen after `3` and
  the leading "Ó" of "Óbito" are simply not in the text stream.
- A parser over that text **dropped the final category of every enumeration**,
  so `CLASSI_FIN` came out as `{1: Confirmado}` — losing *Descartado* entirely
  — and `ATE_HOSP` lost its `9`. That produced 31 spurious "clashes" between
  the dictionary and the registry, then 10 after a fix, then zero once the
  parse was abandoned as the authority.
- The dictionary describes SINAN NET v5.0. The files span 2007–2025 and their
  schema drifts within that.

The reconciliation report (`brepi/codebook/reconcile.py`) compares registry
against dictionary and is run as a check. It is not run as an import.

---

## 3. What "translated" means for each kind of code

An enum lookup handles the easy half. The rest of a DATASUS record is coded in
four other ways, each with its own decoder in
`brepi/codebook/transforms.py`.

### 3.1 Enumerations — `concepts` + `bindings`

The ordinary case. A concept declares `values` (code → canonical label) and
`unknown` (explicit ignorado codes); a binding attaches a concept to a
`system.field`. Concepts are deduplicated: `sinan_sim_nao_ign` is bound to all
29 clinical-sign and exposure-antecedent fields rather than being written 29
times.

Codes are matched verbatim **and** zero-stripped, because SINAN writes `01`
where the dictionary says `1` and SIH does the reverse.

### 3.2 Concept attributes — groupings that belong in one place

A concept may declare `attributes`: further mappings keyed by the same code,
emitted as extra columns. A serovar yields its serogroup and its maintenance
host in the same pass.

This exists so that taxonomy lives in the codebook, reviewable once, instead of
in whichever study script needs it next — where two scripts will eventually
disagree about which serogroup Hardjo belongs to.

### 3.3 Packed composites — `sinan_packed_age`

SINAN stores age as `UVVV`: a unit digit (1 hours, 2 days, 3 months, 4 years)
and a three-digit value. `4028` is 28 years; `3006` is six months. Read as a
number it is nonsense; read as a category it has 137 "levels".

Sub-year ages are **converted, not floored to zero**, and the unit is retained
alongside — "3 months" and "0 years" are the same number and different facts. A
unit digit outside 1–4 is `invalid`, never silently a newborn: SINAN files do
carry stray values here, and treating them as infants would inflate precisely
the age band where leptospirosis is rarest.

### 3.4 Structured identifiers — `epi_week`, `prefix_group`

- **`SEM_NOT` / `SEM_PRI`** are `YYYYWW`. Week 53 is legitimate — roughly one
  year in seven has one — and is kept; dropping it shortens those years in any
  weekly aggregation.
- **`ID_OCUPA_N`** is a six-digit CBO 2002 occupation code. The authoritative
  title list is a separate download, but the **first digit is the major
  occupational group** by definition of the classification. Decoding what the
  code structurally guarantees is honest partial translation and far better
  than leaving 698 six-digit numbers in the analysis. SINAN's `9999xx`
  sentinels are mapped to `unknown`, not to major group 9 — that would invent a
  workforce of maintenance workers.

### 3.5 Self-labelling codes — `labelled_code`

The MAT serovar fields hold `"10 - COPENHAGENI"` in some years and states and a
bare `"10"` in others, **in the same column**. Neither an enum on the raw
string nor a numeric cast handles both. Taking the leading integer normalises
them, after which it is an ordinary enum — and the file's label never has to be
trusted for spelling, which matters because the source writes `AUTRALIS` for
the serovar *Australis*.

The serovar table was **harvested from the data**: all 28 codes map to exactly
one label across 2007–2025. Canonical spellings are the codebook's.

> **Serovar 21 (Patoc) is *Leptospira biflexa* and saprophytic.** It is in the
> MAT panel as a broadly cross-reactive screening antigen. A Patoc-only
> reaction is not evidence of infection by a specific pathogenic serovar and
> must never be attributed to an animal reservoir. The registry records this in
> the concept's own notes and a test asserts it.

### 3.6 Large reference sets — `reference`

5,570 municipalities do not belong in a YAML enumeration, but they are no less
coded for that. Reference tables are built from the IBGE lattice the harness
already fetches and cached beside the registry.

The UF table keys on **both** the two-digit IBGE code and the alphabetic
abbreviation, because SINAN uses both in the same record: `SG_UF_NOT` holds
`43` while `COUFINF` holds `RS`. Keying on one and hoping is how half a field
silently becomes `invalid`.

### 3.7 Numbers stored as text — `numeric`

A MAT titre is the reciprocal of the highest reacting dilution — 100, 200, 400,
800 — and is a quantity, not a category. Leaving it as a string is as much a
translation failure as leaving a sex code untranslated; casting it blindly
hides the entries that are not numbers. A range bound makes an implausible
value `invalid` rather than an outlier that survives into a model.

---

## 4. Coverage: every column accounted for

`coverage_report()` classifies every column of a real file. The residual
category is **not** "we'll get to it":

### SINAN-LEPT, all 120 raw columns

| Status | n | What it means |
|---|---:|---|
| `coded` | 53 | enum concept bound |
| `date` | 26 | date field |
| `reference` | 16 | municipality / UF lookup |
| `ignored` | 11 | deliberately not translated, **with a stated reason** |
| `numeric` | 6 | typed and range-checked |
| `labelled_code` | 4 | serovar |
| `epi_week` | 2 | |
| `sinan_packed_age` | 1 | |
| `prefix_group` | 1 | occupation major group |
| **`UNBOUND`** | **0** | unaccounted |

Before this work, 72 of the 120 were unbound — and they were not the
unimportant ones. They included the serovar (the reservoir), the occupation
code (the exposure) and every geography key (the place).

A test asserts `UNBOUND == 0`, that every ignored field carries a reason longer
than a shrug, and that no field is bound in two ways at once. If a future
dictionary vintage adds a field, the suite fails until somebody either binds it
or writes down why not.

### The eleven ignored fields, and why

| Field(s) | Reason |
|---|---|
| `NU_LOTE_V/H/I` | SINAN transmission batch identifiers; operational keys with no analytic meaning |
| `MIGRADO_W`, `FLXRECEBI`, `CS_FLXRET` | internal record-flow flags; constant in this extract |
| `ID_UNIDADE` | CNES establishment **identifier** — a key, not a category; 5,203 distinct values are not an enumeration |
| `ID_REGIONA`, `ID_RG_RESI` | SINAN regional health-unit codes: a DATASUS division revised without a public changelog. The study uses IBGE-defined health regions instead (ADR-001) |
| `ANT_OU_DES`, `CLI_OTRDES` | free text (1,199 and 2,877 distinct strings); not codable without manual coding |

---

## 5. What the translation unlocked

Two variables that had never been analysed because they were unreadable:

**Serovar and reservoir** — 9,003 confirmed cases (13.5%) carry a MAT result:

| Reservoir class | Cases | Note |
|---|---:|---|
| *Rattus* (Icterohaemorrhagiae + Copenhageni) | 3,959 | urban rodent-borne |
| Swine / equine / hedgehog (Australis group) | 734 | |
| Dog (Canicola) | 552 | |
| Cattle (Sejroe group) | 569 | |
| Saprophyte, non-pathogenic (Patoc) | 431 | **not attributable** |

**Occupation** — 22,942 confirmed cases (34.4%) carry a usable CBO code;
agricultural, forestry and fishing workers are 6,358 of them.

---

## 6. Reproducing

```bash
python -m brepi.codebook.references          # rebuild reference tables
python -m pytest tests/test_codebook_transforms.py
python studies/leptospirosis/21_build_line_level.py
```

The materialisation cache is keyed on the source bytes **and** the codebook
version, so bumping the registry re-decodes every year automatically. There is
no manual invalidation step and no way to analyse a stale decode.
