"""Atlas Digital de Desastres no Brasil — officially recorded disaster events.

The Atlas (``atlasdigital.mdr.gov.br``, SEDEC/MIDR) is the public face of
**S2iD**, the system through which a Brazilian municipality registers a disaster
and requests recognition of *situacao de emergencia* (SE) or *estado de
calamidade publica* (ECP). It is the only national, municipality-resolved,
date-resolved register of disasters, which is what makes it usable as a
staggered treatment; it is also an administrative artefact of a bureaucratic
process, which is what makes the caveats below load-bearing rather than
decorative.

What the export actually is
---------------------------
One row per **municipality x disaster protocol**, not per meteorological event.
The May 2024 Rio Grande do Sul flood is ~470 rows, one per municipality that
filed. A single municipality can file twice for the same storm under different
COBRADE codes. ``Protocolo_S2iD`` is the row identity.

Two dates, and neither is the one you might assume
--------------------------------------------------
``Data_Evento`` is the date of occurrence declared by the municipality.
``Data_Registro`` is the date the record entered S2iD. They coincide for 68.6%
of records; the registration lag has a 90th percentile of 8 days and a maximum
of 461 days, and is *negative* for a minority (registration stamped before the
declared occurrence — a data-entry artefact, not a time machine).

**The date of the federal recognition portaria is not in this export.** The
``Status`` column says only whether recognition happened, not when. For a
Callaway-Sant'Anna design the event-time zero should be the *occurrence* date
(``Data_Evento``): it is the date the exposure actually happened, it is the date
that is causally prior to any health outcome, and it does not inherit the
administrative delay that would otherwise contaminate the treatment timing with
municipal bureaucratic capacity. :func:`disaster_events` therefore keys on
``Data_Evento`` and carries ``register_date`` alongside so a robustness check
against registration timing costs one argument.

The three tiers of officialdom, and the selection problem
---------------------------------------------------------
The Brazilian procedure (Lei 12.608/2012; Portaria MI 526/2012 and IN 1/2012,
which introduced the FIDE form and replaced CODAR with COBRADE; currently
Portaria MDR 260/2022 as amended by Portaria MDR 3.646/2022) has three steps,
and they are not the same event:

1. the municipality files a **FIDE** in S2iD — a *registration*;
2. the municipality or state **decrees** SE or ECP, valid 180 days from
   publication in the official gazette;
3. SEDEC **recognises** it federally by **portaria published in the DOU**.

``Status`` takes exactly two values in the export:

``Registro``      the municipality registered the disaster in S2iD. 40,583 rows.
``Reconhecido``   the federal government issued a recognition portaria. 35,608 rows.

State decrees are *not* separately represented: a state-level decree of SE/ECP
appears here only insofar as its municipalities each filed. So the export mixes
a low bar (self-registration) with a high one (federal recognition), and the
high one is reached disproportionately by municipalities with a functioning
Defesa Civil, a staff member who can assemble a FIDE form, and political access.
Restricting treatment to ``Reconhecido`` therefore buys severity validation at
the price of **selecting on municipal administrative capacity**, which is
plausibly correlated with health-system reporting capacity and hence with the
leptospirosis outcome itself. Restricting to ``Registro`` avoids that but admits
declarations of very different severity.

State decrees are therefore invisible as such: step 2 leaves no column. And the
export mixes step 1 with step 3 under a single ``Status`` field that **no
official document defines** — neither ``institucional.xhtml``,
``como-utilizar.xhtml`` nor the application manual's METADADOS chapter mentions
``Status`` or ``Data_Evento`` at all (the manual still documents a single
``data`` column). The reading above is INFERRED from the values themselves plus
the institucional page's statement that the base prioritised *"os documentos que
levaram ao reconhecimento federal por meio da publicação de Decretos e
Portarias"*.

Neither choice is safe by default, so :func:`disaster_events` does not choose:
``status=None`` returns both with the flag intact, and the DiD layer is expected
to run the design under both definitions. This is documented, not solved.

If the recognition *date* is genuinely needed, it is not here: it lives in the
S2iD "Serie Historica" (:data:`~brepi.config.S2ID_SERIES_PAGE`), which covers
federal SE/ECP recognitions from 2013 and is reachable only by driving a JSF
form. Acquiring it is not implemented; the argument for not needing it is in the
"Two dates" section above.

Mirrors, and why there are none
-------------------------------
Checked 2026-07-30, all negative:

* ``dadosabertos.mdr.gov.br`` (MIDR CKAN) does **not** host the Atlas. Its
  ``s2id_sedec`` package is an older, narrower product — one CSV per year,
  2013-2022 only — useful for its data dictionary and its copy of the COBRADE
  PDF, not as a substitute.
* ``dados.gov.br`` harvests that same package and now returns HTTP 401 to
  unauthenticated API calls, so it is not usable programmatically at all.
* **Base dos Dados** has a dataset with slug ``s2id`` (organisation ``midr``),
  but its table list is *empty*: it is a metadata stub linking out, not a
  BigQuery mirror. There is no ``br_mdr_s2id`` table to query.

The bulk CSV fetched here is the only complete machine-readable copy.

The COBRADE trap that decides the whole extraction
--------------------------------------------------
The intuitive hydrological filter is COBRADE ``1.2.x`` — inundacao, enxurrada,
alagamento. Applied to Rio Grande do Sul in April-June 2024, the largest flood
disaster in the state's history, it returns **47 municipalities**. The true
figure is ~470. The missing 435 filed under ``1.3.2.1.4 Chuvas Intensas``, which
sits in the *meteorological* subgroup of COBRADE but which the Atlas's own
``grupo_de_desastre`` column labels **Hidrologico**. Municipalities routinely
file the rainfall cause rather than the flooding consequence.

:data:`COBRADE_FLOOD` therefore includes ``1.3.2.1.4`` and is the default.
:data:`COBRADE_FLOOD_STRICT` is the narrow ``1.2.x`` reading, kept so the
sensitivity of every result to this single decision can be reported.

Manufactured recurrence
-----------------------
SEDEC's own methodology note warns that federal recognition is **renewed every
six months** for continuing disasters, and that each renewal is a row. For
drought (1.4.x) this inflates the record count enormously and would fabricate a
six-monthly "recurrent treatment" out of one continuous event. Floods are
short-lived and less exposed, but the same mechanism produces clusters of
protocols around one storm. This is why
:func:`brepi.sources.disasters.treatment.episodes` collapses filings within a
washout window instead of counting protocols, and why the washout length is an
argument rather than a constant.

Two further caveats from the same note, both load-bearing:

* the Atlas ingested ~96,900 raw submissions and publishes ~76,200 after
  de-duplication, and SEDEC states that records carrying federal recognition
  were *prioritised* in that reconciliation — so the recognised subset is
  cleaner as well as more selected;
* pre-S2iD series (before ~2012) "podem conter lacunas". The 2007-2011 stretch
  of this panel is retrospective data entry, not contemporaneous registration,
  which is a plausible reason the 2007-2009 treatment cohorts are so large.
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterable, Sequence

import httpx
import polars as pl

from brepi.config import (
    ATLAS_BULK_CSV_FALLBACK,
    ATLAS_DOWNLOADS_PAGE,
    ATLAS_ENCODING,
    ATLAS_SEPARATOR,
    HTTP_BACKOFF_SECONDS,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
)
from brepi.geo import lattice
from brepi.io import cache

TOOL_VERSION = "brepi.sources.disasters.atlas/1"
SOURCE_ID = "midr.atlas.s2id"

CACHE_KEY_BULK = "disasters/atlas/atlas_consolidado.csv"
CACHE_KEY_DOWNLOADS_PAGE = "disasters/atlas/downloads.xhtml"


class AtlasError(RuntimeError):
    """The Atlas export is absent, malformed, or not what this module expects."""


# --------------------------------------------------------------------------
# COBRADE
# --------------------------------------------------------------------------

#: COBRADE (Classificacao e Codificacao Brasileira de Desastres, IN MI 1/2012,
#: Anexo V) as it appears in the Atlas export. The export stores the code with
#: the dots removed ("12200"); this module normalises to the dotted form
#: everywhere, because the dotted form is what every official document, every
#: state Defesa Civil table and every reader uses.
COBRADE_LABELS: dict[str, str] = {
    # 1.1 Geologico
    "1.1.1.1.0": "Terremoto - tremor de terra",
    "1.1.1.2.0": "Terremoto - tsunami",
    "1.1.3.1.1": "Movimento de massa - quedas/tombamentos/rolamentos: blocos",
    "1.1.3.1.2": "Movimento de massa - quedas/tombamentos/rolamentos: lascas",
    "1.1.3.1.3": "Movimento de massa - quedas/tombamentos/rolamentos: matacoes",
    "1.1.3.1.4": "Movimento de massa - quedas/tombamentos/rolamentos: lajes",
    "1.1.3.2.1": "Movimento de massa - deslizamentos de solo e/ou rocha",
    "1.1.3.3.1": "Movimento de massa - corridas de massa: solo/lama",
    "1.1.3.3.2": "Movimento de massa - corridas de massa: rocha/detrito",
    "1.1.3.4.0": "Movimento de massa - subsidencias e colapsos",
    "1.1.4.1.0": "Erosao costeira/marinha",
    "1.1.4.2.0": "Erosao de margem fluvial",
    "1.1.4.3.1": "Erosao continental - laminar",
    "1.1.4.3.2": "Erosao continental - ravinas",
    "1.1.4.3.3": "Erosao continental - bocorocas",
    # 1.2 Hidrologico
    "1.2.1.0.0": "Inundacoes",
    "1.2.2.0.0": "Enxurradas",
    "1.2.3.0.0": "Alagamentos",
    # 1.3 Meteorologico
    "1.3.1.1.1": "Ciclones - ventos costeiros (mobilidade de dunas)",
    "1.3.1.1.2": "Ciclones - marés de tempestade (ressacas)",
    "1.3.1.2.0": "Frentes frias / zonas de convergencia",
    "1.3.2.1.1": "Tempestade local/convectiva - tornados",
    "1.3.2.1.2": "Tempestade local/convectiva - tempestade de raios",
    "1.3.2.1.3": "Tempestade local/convectiva - granizo",
    "1.3.2.1.4": "Tempestade local/convectiva - chuvas intensas",
    "1.3.2.1.5": "Tempestade local/convectiva - vendaval",
    "1.3.3.1.0": "Onda de calor",
    "1.3.3.2.1": "Onda de frio - friagem",
    "1.3.3.2.2": "Onda de frio - geadas",
    # 1.4 Climatologico
    "1.4.1.1.0": "Estiagem",
    "1.4.1.2.0": "Seca",
    "1.4.1.3.1": "Incendio florestal - em parques/APAs/APPs",
    "1.4.1.3.2": "Incendio florestal - em areas nao protegidas",
    "1.4.1.4.0": "Baixa umidade do ar",
    # 1.5 Biologico
    "1.5.1.1.0": "Epidemias - doencas infecciosas virais",
    "1.5.1.2.0": "Epidemias - doencas infecciosas bacterianas",
    "1.5.1.3.0": "Epidemias - doencas infecciosas parasiticas",
    "1.5.2.1.0": "Infestacoes - de animais",
    "1.5.2.3.0": "Infestacoes - de algas (mares vermelhas)",
}

#: The narrow, textbook hydrological reading: COBRADE subgroup 1.2.
#: Use as a *sensitivity* definition. As the module docstring explains, it
#: undercounts the RS 2024 disaster tenfold.
COBRADE_FLOOD_STRICT: tuple[str, ...] = ("1.2.",)

#: The default flood definition: subgroup 1.2 plus ``1.3.2.1.4`` (chuvas
#: intensas), which is where the majority of large Brazilian flood events are
#: actually filed and which the Atlas itself groups as ``Hidrologico``.
COBRADE_FLOOD: tuple[str, ...] = ("1.2.", "1.3.2.1.4")

#: Mass movement (landslides and debris flows), COBRADE 1.1.3.x. Co-occurs with
#: intense rain and is the other half of a "rain disaster"; the Atlas also files
#: it under ``grupo_de_desastre = Hidrologico``.
COBRADE_MASS_MOVEMENT: tuple[str, ...] = ("1.1.3.",)

#: Everything the Atlas itself calls hydrological.
COBRADE_HYDROLOGICAL: tuple[str, ...] = COBRADE_FLOOD + COBRADE_MASS_MOVEMENT

#: Droughts. The mirror image of flooding and a natural placebo/negative-control
#: treatment for a leptospirosis DiD: same bureaucratic pathway, opposite
#: hydrology, no plausible leptospirosis mechanism.
COBRADE_DROUGHT: tuple[str, ...] = ("1.4.1.1.0", "1.4.1.2.0")


def normalise_cobrade(code: str | int | None) -> str | None:
    """Render a COBRADE code in the canonical dotted form ``d.d.d.d.d``.

    Accepts the Atlas's dot-stripped five-character form (``"12200"``), the
    dotted form, and integers that have lost a leading zero in a spreadsheet.
    Returns ``None`` for anything that is not five digits, because a COBRADE
    code of another length is not a COBRADE code and guessing is worse than
    admitting it.

    >>> normalise_cobrade("12200")
    '1.2.2.0.0'
    >>> normalise_cobrade("1.1.3.2.1")
    '1.1.3.2.1'
    """
    if code is None:
        return None
    digits = re.sub(r"\D", "", str(code))
    if len(digits) != 5:
        return None
    return ".".join(digits)


def cobrade_expr(column: str = "Cod_Cobrade") -> pl.Expr:
    """Vectorised :func:`normalise_cobrade` for a polars column."""
    stripped = pl.col(column).cast(pl.Utf8).str.replace_all(r"\D", "")
    dotted = pl.concat_str(
        [stripped.str.slice(i, 1) for i in range(5)], separator="."
    )
    return pl.when(stripped.str.len_chars() == 5).then(dotted).otherwise(None)


def _matches_prefixes(column: str, prefixes: Sequence[str]) -> pl.Expr:
    """Whether a dotted-COBRADE column starts with any of ``prefixes``.

    Prefix matching on the *dotted* string is what makes ``"1.2."`` mean
    "subgroup 1.2" and not "anything beginning with the digits 12" — which
    would also catch ``1.2`` -vs- ``12.``-style collisions in a longer scheme.
    """
    if not prefixes:
        return pl.lit(True)
    expr = pl.lit(False)
    for prefix in prefixes:
        expr = expr | pl.col(column).str.starts_with(prefix)
    return expr


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------


def _http_get(url: str) -> tuple[bytes, dict[str, Any]]:
    """GET with the project's retry policy. Transient faults retry; 4xx raises."""
    last: Exception | None = None
    headers = {"User-Agent": f"brepi/{TOOL_VERSION} (academic research)"}
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            with httpx.Client(
                timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers
            ) as client:
                response = client.get(url)
        except httpx.HTTPError as exc:
            last = exc
        else:
            if response.status_code == 200:
                return response.content, {
                    "remote_modified": response.headers.get("last-modified"),
                    "params": {"url": str(response.url), "attempt": attempt},
                }
            if response.status_code < 500 and response.status_code != 429:
                raise AtlasError(f"{url} returned HTTP {response.status_code}")
            last = AtlasError(f"{url} returned HTTP {response.status_code}")
        if attempt < HTTP_MAX_RETRIES:
            time.sleep(HTTP_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise AtlasError(f"giving up on {url}") from last


_BULK_LINK = re.compile(r'href="(/arquivos/[^"]*Consolidado[^"]*\.csv)"', re.I)


def discover_bulk_url(*, refresh: bool = False) -> str:
    """Resolve the current bulk-CSV URL from the Atlas Downloads page.

    The filename encodes a version and a release date
    (``BD_Atlas_1991_2025_v1.0_2026.04.23_Consolidado.csv``) and changes at every
    release, so hard-coding it guarantees a silent 404 within months. The page
    is scraped, cached with provenance like any other remote byte, and falls
    back to :data:`~brepi.config.ATLAS_BULK_CSV_FALLBACK` if the markup changes
    — a stale-but-identified artefact beats a crash mid-pipeline.
    """
    from brepi.config import ATLAS_BASE

    try:
        path, _prov = cache.fetch(
            CACHE_KEY_DOWNLOADS_PAGE,
            lambda: _http_get(ATLAS_DOWNLOADS_PAGE),
            source=SOURCE_ID,
            uri=ATLAS_DOWNLOADS_PAGE,
            refresh=refresh,
            tool_version=TOOL_VERSION,
        )
        html = path.read_bytes().decode(ATLAS_ENCODING, errors="replace")
    except (AtlasError, httpx.HTTPError, OSError):
        return ATLAS_BULK_CSV_FALLBACK

    matches = _BULK_LINK.findall(html)
    if not matches:
        return ATLAS_BULK_CSV_FALLBACK
    # Several releases are occasionally listed; the lexicographically greatest
    # filename is the newest, because the date is embedded ISO-ish (YYYY.MM.DD).
    return ATLAS_BASE + sorted(matches)[-1]


def acquire(*, refresh: bool = False, url: str | None = None) -> tuple[Any, cache.Provenance]:
    """Fetch the consolidated Atlas export into the cache. Returns (path, provenance)."""
    resolved = url or discover_bulk_url(refresh=refresh)
    return cache.fetch(
        CACHE_KEY_BULK,
        lambda: _http_get(resolved),
        source=SOURCE_ID,
        uri=resolved,
        refresh=refresh,
        tool_version=TOOL_VERSION,
    )


def snapshot(name: str | None = None, *, refresh: bool = False) -> str:
    """Freeze the Atlas artefacts into a named manifest.

    The Atlas is *retroactively edited* — SEDEC publishes a corrections log
    precisely because rows for events years past are altered between releases.
    An analysis that does not pin the release cannot be reproduced, and the
    corrections are not distinguishable from an extraction bug after the fact.
    """
    from datetime import date

    acquire(refresh=refresh)
    key = name or f"atlas_disasters_{date.today():%Y%m%d}"
    cache.write_manifest(key, [CACHE_KEY_BULK, CACHE_KEY_DOWNLOADS_PAGE])
    return key


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------

#: Atlas column -> analysis column, for the impact measures we carry forward.
#: ``DH_`` is *danos humanos*, ``DM_`` *danos materiais*, ``PEPL_``/``PEPR_``
#: *prejuizos economicos publicos/privados*, in BRL.
#:
#: **The monetary columns are not on one price base.** The Atlas methodology
#: states that financial damages are available for 1995-2022 and were deflated
#: to December 2022 prices; records after 2022 carry the nominal figure as
#: filed. So a 2024 damage value and a 2010 damage value are not comparable and
#: a panel regression on the level of damages is measuring inflation at the
#: window's right edge. Use the *count* measures (deaths, homeless, displaced)
#: for severity, or restrict monetary work to <=2022 and say so.
_IMPACT_COLUMNS: dict[str, str] = {
    "DH_MORTOS": "deaths",
    "DH_FERIDOS": "injured",
    "DH_ENFERMOS": "ill",
    "DH_DESABRIGADOS": "homeless",
    "DH_DESALOJADOS": "displaced",
    "DH_DESAPARECIDOS": "missing",
    "DH_AFETADOS_SECA_ESTIAGEM": "affected_drought",
    "DH_total_danos_humanos_diretos": "affected_direct",
    "DH_OUTROS AFETADOS": "affected_other",
    "DM_total_danos_materiais": "damage_material_brl",
    "PEPL_total_publico": "damage_public_brl",
    "PEPR_total_privado": "damage_private_brl",
    "PE_PLePR": "damage_economic_brl",
}

_COUNT_COLUMNS = (
    "deaths",
    "injured",
    "ill",
    "homeless",
    "displaced",
    "missing",
    "affected_drought",
    "affected_direct",
    "affected_other",
)
_MONEY_COLUMNS = (
    "damage_material_brl",
    "damage_public_brl",
    "damage_private_brl",
    "damage_economic_brl",
)

EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "munic_code",
    "munic_code_raw",
    "munic_name",
    "uf_abbr",
    "region",
    "event_date",
    "register_date",
    "register_lag_days",
    "period",
    "year",
    "cobrade",
    "cobrade_label",
    "typology",
    "disaster_group",
    "status",
    "recognised",
    *_COUNT_COLUMNS,
    "affected_total",
    *_MONEY_COLUMNS,
)


def raw(*, refresh: bool = False) -> pl.DataFrame:
    """The consolidated Atlas export, wholly undecoded, every column ``Utf8``.

    Read as strings on purpose: the file carries Brazilian decimal commas in
    some vintages and dots in others, and letting a CSV reader infer per-column
    types across 76k rows is how a monetary column silently becomes null.
    """
    path, _prov = acquire(refresh=refresh)
    return pl.read_csv(
        path,
        separator=ATLAS_SEPARATOR,
        encoding=ATLAS_ENCODING.replace("iso-8859-1", "latin-1"),
        infer_schema_length=0,
        truncate_ragged_lines=True,
        quote_char='"',
    )


def _numeric(column: str) -> pl.Expr:
    """Parse an Atlas numeric field to Float64, tolerating pt-BR formatting.

    ``1.234.567,89`` and ``1234567.89`` both occur across releases. Thousands
    separators are stripped only when a decimal comma is present, so a plain
    ``1.234`` (a dot decimal) is not mangled into ``1234``.
    """
    text = pl.col(column).cast(pl.Utf8).str.strip_chars()
    has_comma = text.str.contains(",")
    ptbr = text.str.replace_all(r"\.", "").str.replace(",", ".")
    return (
        pl.when(has_comma).then(ptbr).otherwise(text).cast(pl.Float64, strict=False)
    )


def _project_codes(
    df: pl.DataFrame, *, lattice_year: int, source_column: str
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Normalise municipality codes to 7 digits on the ``lattice_year`` lattice.

    Three failure modes are handled explicitly because each is silent otherwise:

    * **6-digit codes.** Older Atlas vintages and every DATASUS-derived mirror
      carry the truncated code. Converted through
      :func:`brepi.geo.lattice.code6_to_code7`, which knows the nine
      hand-allocated codes the check-digit algorithm gets wrong.
    * **Codes that no longer exist.** Mapped forward through
      :func:`brepi.geo.lattice.build_crosswalk` where the mapping is resolvable.
    * **Codes that are not resolvable at all** (extinct without a recorded
      successor, or malformed). Dropped — but *counted and returned*, never
      silently discarded, since an unnoticed drop biases the never-treated set.
    """
    target = set(lattice.load_municipalities(lattice_year)["code7"].to_list())

    df = df.with_columns(
        pl.col(source_column).cast(pl.Utf8).str.strip_chars().alias("munic_code_raw")
    )
    width = pl.col("munic_code_raw").str.len_chars()
    df = df.with_columns(
        pl.when(width == 6)
        .then(lattice.code6_to_code7_expr("munic_code_raw"))
        .when(width == 7)
        .then(pl.col("munic_code_raw"))
        .otherwise(None)
        .alias("munic_code")
    )

    unresolved = set(
        df.filter(
            pl.col("munic_code").is_not_null() & ~pl.col("munic_code").is_in(list(target))
        )["munic_code"].unique().to_list()
    )
    remap: dict[str, str] = {}
    if unresolved:
        # Only build the (expensive) crosswalk when something actually needs it.
        walk = lattice.build_crosswalk(2000, lattice_year).filter(
            pl.col("resolvable") & pl.col("to_code7").is_not_null()
        )
        remap = {
            f: t
            for f, t in zip(walk["from_code7"].to_list(), walk["to_code7"].to_list())
            if f in unresolved and t in target
        }
        if remap:
            df = df.with_columns(
                pl.col("munic_code").replace(remap).alias("munic_code")
            )

    valid = pl.col("munic_code").is_not_null() & pl.col("munic_code").is_in(list(target))
    dropped = df.filter(~valid)
    report = {
        "lattice_year": lattice_year,
        "n_rows_in": df.height,
        "n_rows_dropped": dropped.height,
        "n_codes_remapped": len(remap),
        "remapped": remap,
        "dropped_codes": sorted(
            set(dropped["munic_code_raw"].drop_nulls().to_list())
        )[:50],
    }
    return df.filter(valid), report


def disaster_events(
    years: Iterable[int] | range | None = None,
    cobrade_prefixes: Sequence[str] | None = COBRADE_FLOOD,
    *,
    status: str | None = None,
    lattice_year: int = lattice.DEFAULT_YEAR,
    date_field: str = "event",
    refresh: bool = False,
    return_report: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, Any]]:
    """Long table of disaster events, one row per S2iD protocol.

    Parameters
    ----------
    years
        Calendar years to keep, filtered on ``date_field``. ``None`` keeps all
        (1991-2025 in the current release).
    cobrade_prefixes
        Dotted COBRADE prefixes, matched with ``startswith``. Defaults to
        :data:`COBRADE_FLOOD`, which **includes 1.3.2.1.4 chuvas intensas** —
        see the module docstring for why omitting it undercounts RS 2024 by a
        factor of ten. Pass ``None`` for every disaster type.
    status
        ``"Reconhecido"`` restricts to federally recognised events,
        ``"Registro"`` to registered-but-unrecognised ones, ``None`` (default)
        keeps both and lets the DiD layer run the design under each. Restricting
        to recognition selects on municipal administrative capacity; see module
        docstring.
    date_field
        ``"event"`` (default, ``Data_Evento``, the occurrence) or ``"register"``
        (``Data_Registro``). Occurrence is the right event-time zero; the other
        exists for the robustness check.

    Returns a frame with :data:`EVENT_COLUMNS`. ``munic_code`` is a 7-digit IBGE
    code on the ``lattice_year`` lattice and ``period`` is the first day of the
    event's month, so the result joins directly onto
    :func:`brepi.panel.spine.build_spine`.
    """
    if date_field not in ("event", "register"):
        raise ValueError("date_field must be 'event' or 'register'")

    df = raw(refresh=refresh)

    required = {"Protocolo_S2iD", "Cod_IBGE_Mun", "Cod_Cobrade", "Data_Evento", "Status"}
    missing = required - set(df.columns)
    if missing:
        raise AtlasError(
            f"the Atlas export is missing {sorted(missing)}. The schema changed; "
            "re-read the release notes rather than patching around it."
        )

    df, code_report = _project_codes(
        df, lattice_year=lattice_year, source_column="Cod_IBGE_Mun"
    )

    df = df.with_columns(
        pl.col("Data_Evento").str.strptime(pl.Date, "%d/%m/%Y", strict=False).alias("event_date"),
        pl.col("Data_Registro").str.strptime(pl.Date, "%d/%m/%Y", strict=False).alias("register_date"),
        cobrade_expr("Cod_Cobrade").alias("cobrade"),
    )

    n_unparsed = int(df.select(pl.col("event_date").is_null().sum()).item())
    if n_unparsed:
        # Deterministic failure: an unparseable date is a specification problem,
        # and a dropped event is a municipality wrongly counted as never-treated.
        raise AtlasError(
            f"{n_unparsed} rows have an unparseable Data_Evento. Expected %d/%m/%Y."
        )

    anchor = "event_date" if date_field == "event" else "register_date"
    df = df.with_columns(
        (pl.col("register_date") - pl.col("event_date")).dt.total_days().alias("register_lag_days"),
        pl.col(anchor).dt.truncate("1mo").alias("period"),
        pl.col(anchor).dt.year().cast(pl.Int32).alias("year"),
        pl.col("cobrade").replace_strict(COBRADE_LABELS, default=None).alias("cobrade_label"),
        (pl.col("Status") == "Reconhecido").alias("recognised"),
    )

    if cobrade_prefixes:
        df = df.filter(_matches_prefixes("cobrade", list(cobrade_prefixes)))
    if years is not None:
        df = df.filter(pl.col("year").is_in([int(y) for y in years]))
    if status is not None:
        df = df.filter(pl.col("Status") == status)

    df = df.with_columns(
        [_numeric(src).alias(dst) for src, dst in _IMPACT_COLUMNS.items() if src in df.columns]
    )
    for col in (*_COUNT_COLUMNS, *_MONEY_COLUMNS):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))

    df = df.with_columns(
        [pl.col(c).fill_null(0).cast(pl.Int64) for c in _COUNT_COLUMNS]
        + [pl.col(c).fill_null(0.0) for c in _MONEY_COLUMNS]
    ).with_columns(
        # "Affected" in the FIDE form is the union of several disjoint categories.
        # Drought-affected is kept separate because it is only ever populated for
        # 1.4.x and summing it into a flood exposure measure would be nonsense.
        (
            pl.col("homeless")
            + pl.col("displaced")
            + pl.col("injured")
            + pl.col("ill")
            + pl.col("affected_other")
        ).alias("affected_total")
    )

    out = df.rename(
        {
            "Protocolo_S2iD": "event_id",
            "Nome_Municipio": "munic_name",
            "Sigla_UF": "uf_abbr",
            "regiao": "region",
            "tipologia": "typology",
            "grupo_de_desastre": "disaster_group",
            "Status": "status",
        }
    )
    absent = [c for c in EVENT_COLUMNS if c not in out.columns]
    if absent:
        raise AtlasError(
            f"cannot emit {absent} — the Atlas export no longer carries the source "
            "columns they derive from. Re-read the release notes."
        )
    out = out.select(EVENT_COLUMNS).sort(["munic_code", "event_date", "cobrade"])

    if not return_report:
        return out
    report = {
        "codes": code_report,
        "n_events": out.height,
        "n_municipalities": out["munic_code"].n_unique(),
        "date_field": date_field,
        "cobrade_prefixes": list(cobrade_prefixes) if cobrade_prefixes else None,
        "status": status,
        "date_range": (out["event_date"].min(), out["event_date"].max()),
    }
    return out, report


def cobrade_summary(events: pl.DataFrame) -> pl.DataFrame:
    """Records and distinct municipalities by COBRADE code. The first thing to look at."""
    return (
        events.group_by("cobrade", "cobrade_label", "disaster_group")
        .agg(
            pl.len().alias("n_events"),
            pl.col("munic_code").n_unique().alias("n_municipalities"),
            pl.col("recognised").sum().alias("n_recognised"),
            pl.col("deaths").sum().alias("deaths"),
        )
        .sort("n_events", descending=True)
    )


__all__ = [
    "AtlasError",
    "COBRADE_LABELS",
    "COBRADE_FLOOD",
    "COBRADE_FLOOD_STRICT",
    "COBRADE_MASS_MOVEMENT",
    "COBRADE_HYDROLOGICAL",
    "COBRADE_DROUGHT",
    "EVENT_COLUMNS",
    "normalise_cobrade",
    "cobrade_expr",
    "discover_bulk_url",
    "acquire",
    "snapshot",
    "raw",
    "disaster_events",
    "cobrade_summary",
]
