"""The municipality lattice, its code systems, and its crosswalks over time.

This module is foundational: every join in the project -- SINAN numerators to
SIDRA denominators, denominators to sanitation covariates, covariates to
climate -- lands on the lattice defined here. Two things make that dangerous
enough to warrant a dedicated module.

**Two code systems.** IBGE identifies a municipality with seven digits
(``3550308``); DATASUS uses the first six (``355030``). The seventh is a check
digit. Truncation is therefore lossless in one direction and requires
recomputation in the other. Getting it wrong does not raise -- it produces a
join that silently drops or, worse, mismatches rows.

**A moving lattice.** Brazil had 5507 municipalities at the 2000 Census, 5560
from 2001, 5564 from 2004, 5565 at the 2010 Census and 5570 from 2013. A
leptospirosis case notified in 2005 under Santarem/PA covers territory that has
been Mojui dos Campos/PA since 2013. Comparing a 2005 rate to a 2015 rate at
face value compares two different places. :func:`build_crosswalk` states what is
resolvable and :func:`amc_groups` gives the merge sets for what is not.

Territorial facts live as data in ``municipality_changes.yaml`` with source
citations, not as literals in this file.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx
import polars as pl
import yaml

from brepi.config import HTTP_BACKOFF_SECONDS, HTTP_MAX_RETRIES, HTTP_TIMEOUT, IBGE_LOCALIDADES_BASE
from brepi.io.cache import fetch

TOOL_VERSION = "brepi.geo.lattice/1"
SOURCE_ID = "ibge.localidades.v1"

CHANGES_PATH = Path(__file__).resolve().with_name("municipality_changes.yaml")

#: The lattice the project models on.
DEFAULT_YEAR = 2022

#: Official municipality counts by year, used as an assertion target. A loader
#: that returns something else has picked up a lattice we did not ask for.
EXPECTED_COUNTS: dict[int, int] = {
    2000: 5507,
    2001: 5560,
    2004: 5564,
    2005: 5564,
    2010: 5565,
    2013: 5570,
    2022: 5570,
    2025: 5571,
}

_LEVEL_N6 = "N6"


class LatticeError(RuntimeError):
    """A municipality code or lattice operation is not well formed."""


# --------------------------------------------------------------------------
# Check digit
# --------------------------------------------------------------------------


def check_digit(code6: str | int) -> int:
    """IBGE municipality check digit for a six-digit code.

    The algorithm is a weighted modulus-10 sum: digits are weighted
    ``1,2,1,2,1,2`` from the left, each product above nine has its digits added
    together (equivalently, nine is subtracted), and the check digit is the
    complement of the running total to the next multiple of ten.

    >>> check_digit("355030")   # Sao Paulo
    8
    >>> check_digit("120040")   # Rio Branco
    1
    >>> check_digit("431490")   # Porto Alegre
    2
    """
    text = _digits(code6, 6, "code6")
    total = 0
    for position, char in enumerate(text):
        product = int(char) * (2 if position % 2 else 1)
        total += product - 9 if product > 9 else product
    return (10 - total % 10) % 10


#: Nine municipalities whose IBGE seven-digit code does not satisfy the check
#: digit above. This is not a bug in the algorithm and not a typo here: IBGE
#: assigned these codes and they are what appears in every IBGE and DATASUS
#: file. All nine were created in 1995-1997, when codes were allocated by hand.
#:
#: A naive ``code6 -> code7`` implementation silently produces a code that does
#: not exist for all nine, and the resulting join drops those municipalities
#: without error. Seven of the nine are in the semi-arid Northeast, where
#: leptospirosis incidence is low but nonzero; two are in Goias. Losing them is
#: a small bias, and an invisible one, which is worse.
#:
#: Verified against the live IBGE localidades lattice (5571 municipalities) on
#: 2026-07-30 by :func:`self_check`, which fails if this table stops matching.
CHECK_DIGIT_EXCEPTIONS: dict[str, str] = {
    "220191": "2201919",  # Bom Principio do Piaui (algorithm gives 2201911)
    "220198": "2201988",  # Brejo do Piaui (algorithm gives 2201986)
    "220225": "2202251",  # Canavieira/PI (algorithm gives 2202257)
    "261153": "2611533",  # Quixaba/PE (algorithm gives 2611531)
    "311783": "3117836",  # Conego Marinho/MG (algorithm gives 3117835)
    "315213": "3152131",  # Ponto Chique/MG (algorithm gives 3152139)
    "430587": "4305871",  # Coronel Barros/RS (algorithm gives 4305876)
    "520393": "5203939",  # Buriti de Goias (algorithm gives 5203930)
    "520396": "5203962",  # Buritinopolis/GO (algorithm gives 5203963)
}


def _digits(value: str | int, length: int, label: str) -> str:
    text = str(value).strip()
    if not text.isdigit():
        raise LatticeError(f"{label} must be numeric, got {value!r}")
    if len(text) != length:
        raise LatticeError(f"{label} must have {length} digits, got {value!r}")
    return text


def code6_to_code7(code6: str | int | Iterable[Any]) -> Any:
    """Recover the IBGE seven-digit code from a DATASUS six-digit code.

    Accepts a scalar or an iterable; returns the same shape. Six-digit codes are
    zero-padded first, because a code beginning with a low digit routinely
    survives a trip through a spreadsheet as an integer.

    :data:`CHECK_DIGIT_EXCEPTIONS` is consulted before the algorithm. Nine real
    municipalities have codes the algorithm does not reproduce.

    >>> code6_to_code7("355030")
    '3550308'
    >>> code6_to_code7("220191")   # exception, not 2201911
    '2201919'
    """
    if isinstance(code6, (str, int)):
        text = str(code6).strip().zfill(6)
        override = CHECK_DIGIT_EXCEPTIONS.get(text)
        if override is not None:
            return override
        return f"{text}{check_digit(text)}"
    return [code6_to_code7(item) for item in code6]


def code7_to_code6(code7: str | int | Iterable[Any], *, verify: bool = True) -> Any:
    """Drop the check digit from an IBGE seven-digit code.

    With ``verify`` the check digit is recomputed and a mismatch raises. That is
    the cheapest available integrity test on a municipality code column and it
    catches transposition and truncation before they become a bad join.
    """
    if isinstance(code7, (str, int)):
        text = str(code7).strip().zfill(7)
        _digits(text, 7, "code7")
        head, tail = text[:6], int(text[6])
        if CHECK_DIGIT_EXCEPTIONS.get(head) == text:
            return head
        if verify and check_digit(head) != tail:
            raise LatticeError(
                f"code7 {text} fails its check digit (expected {check_digit(head)}, got {tail})"
            )
        return head
    return [code7_to_code6(item, verify=verify) for item in code7]


def is_valid_code7(code7: str | int) -> bool:
    """Whether a seven-digit code is internally consistent."""
    try:
        code7_to_code6(code7, verify=True)
    except LatticeError:
        return False
    return True


def code6_to_code7_expr(column: str | pl.Expr) -> pl.Expr:
    """Vectorised :func:`code6_to_code7` for a polars string column.

    Applies :data:`CHECK_DIGIT_EXCEPTIONS` after the algorithm, so the vector
    and scalar paths cannot drift apart.
    """
    expr = pl.col(column) if isinstance(column, str) else column
    padded = expr.cast(pl.Utf8).str.strip_chars().str.pad_start(6, "0")
    total = pl.lit(0, dtype=pl.Int32)
    for position in range(6):
        digit = padded.str.slice(position, 1).cast(pl.Int32)
        product = digit * (2 if position % 2 else 1)
        total = total + pl.when(product > 9).then(product - 9).otherwise(product)
    dv = (10 - total % 10) % 10
    computed = padded + dv.cast(pl.Utf8)
    return (
        pl.when(padded.is_in(list(CHECK_DIGIT_EXCEPTIONS)))
        .then(padded.replace_strict(CHECK_DIGIT_EXCEPTIONS, default=None))
        .otherwise(computed)
    )


def code7_to_code6_expr(column: str | pl.Expr) -> pl.Expr:
    """Vectorised :func:`code7_to_code6` (no verification)."""
    expr = pl.col(column) if isinstance(column, str) else column
    return expr.cast(pl.Utf8).str.strip_chars().str.pad_start(7, "0").str.slice(0, 6)


def resolve_municipality_code(
    frame: pl.DataFrame,
    candidates: Sequence[tuple[str, str]],
    *,
    lattice_year: int = DEFAULT_YEAR,
    output_column: str = "munic_code6",
    source_column: str = "munic_code_source",
    status_column: str = "munic_code_status",
) -> pl.DataFrame:
    """Resolve ordered candidate fields against a real municipal lattice.

    Six digits do not prove that a value is a municipality: DATASUS tables
    contain state-level placeholders such as ``350000`` and other special
    codes. The first candidate present in the requested IBGE lattice wins.
    Rows with no valid candidate remain explicit as ``unresolved``; they are
    never assigned a plausible-looking check digit or silently discarded.

    Codes are returned in DATASUS' six-digit representation. Original fields
    remain available so callers can audit why a lower-priority field was used.
    """
    if not candidates:
        raise ValueError("at least one municipality-code candidate is required")
    missing = [
        column for _label, column in candidates if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"municipality-code candidate columns absent: {missing}")
    labels = [label for label, _column in candidates]
    if len(set(labels)) != len(labels):
        raise ValueError("municipality-code candidate labels must be unique")

    valid_codes = load_municipalities(lattice_year)["code6"].to_list()
    selected_codes: list[pl.Expr] = []
    selected_sources: list[pl.Expr] = []
    for label, column in candidates:
        clean = pl.col(column).cast(pl.Utf8).str.strip_chars()
        valid = (
            clean.str.contains(r"^\d{6}$").fill_null(False)
            & clean.is_in(valid_codes).fill_null(False)
        )
        selected_codes.append(pl.when(valid).then(clean))
        selected_sources.append(pl.when(valid).then(pl.lit(label)))
    resolved = pl.coalesce(selected_codes)
    return frame.with_columns(
        resolved.alias(output_column),
        pl.coalesce(selected_sources).alias(source_column),
        pl.when(resolved.is_not_null())
        .then(pl.lit("resolved"))
        .otherwise(pl.lit("unresolved"))
        .alias(status_column),
    )


# --------------------------------------------------------------------------
# Live lattice
# --------------------------------------------------------------------------


def _http_get(url: str) -> tuple[bytes, dict[str, Any]]:
    import time

    last: Exception | None = None
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
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
                raise LatticeError(f"{url} returned HTTP {response.status_code}")
            last = LatticeError(f"{url} returned HTTP {response.status_code}")
        if attempt < HTTP_MAX_RETRIES:
            time.sleep(HTTP_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise LatticeError(f"giving up on {url}") from last


def _fetch_localidades(resource: str, key: str, *, refresh: bool = False) -> Any:
    url = f"{IBGE_LOCALIDADES_BASE}/{resource}"
    path, _prov = fetch(
        key,
        lambda: _http_get(url),
        source=SOURCE_ID,
        uri=url,
        refresh=refresh,
        tool_version=TOOL_VERSION,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_municipalities(*, refresh: bool = False) -> list[dict[str, Any]]:
    return list(
        _fetch_localidades("municipios", "ibge/localidades/municipios.json", refresh=refresh)
    )


_MUNICIPALITY_SCHEMA: dict[str, pl.DataType] = {
    "code7": pl.Utf8,
    "code6": pl.Utf8,
    "name": pl.Utf8,
    "uf_code": pl.Utf8,
    "uf_abbr": pl.Utf8,
    "uf_name": pl.Utf8,
    "region": pl.Utf8,
    "region_abbr": pl.Utf8,
    "microregion": pl.Utf8,
    "microregion_name": pl.Utf8,
    "mesoregion": pl.Utf8,
    "mesoregion_name": pl.Utf8,
    "immediate_region": pl.Utf8,
    "immediate_region_name": pl.Utf8,
    "intermediate_region": pl.Utf8,
    "intermediate_region_name": pl.Utf8,
    "health_region": pl.Utf8,
    "health_region_name": pl.Utf8,
    "lattice_year": pl.Int32,
}


def load_municipalities(
    year: int = DEFAULT_YEAR,
    *,
    refresh: bool = False,
    assert_count: bool = True,
) -> pl.DataFrame:
    """The municipality lattice for ``year``.

    Sourced from ``/api/v1/localidades/municipios``, which serves the *current*
    lattice, then filtered to ``year`` using ``municipality_changes.yaml``: a
    municipality installed after ``year`` did not exist then and must not appear
    in a panel row for it.

    Columns: ``code7``, ``code6``, ``name``, ``uf_code``, ``uf_abbr``,
    ``region``, the legacy ``microregion``/``mesoregion`` divisions, the 2017
    ``immediate_region``/``intermediate_region`` divisions, and
    ``health_region``.

    ``health_region`` is emitted as null. Regioes de Saude are a SUS
    administrative division defined by CIR/CIB resolutions and published by
    DATASUS, not by IBGE, and they are revised without a public changelog.
    Populate the column with :func:`attach_health_regions` from a dated DATASUS
    extract rather than letting this module guess.

    The legacy micro/mesoregion divisions were formally superseded in 2017 by
    the immediate/intermediate regions. Both are kept because the literature
    splits across them: older leptospirosis papers stratify by mesoregion.
    """
    raw = _raw_municipalities(refresh=refresh)
    changes = load_changes()
    installed = {
        str(entry["code7"]): int(entry["effective_year"])
        for entry in changes.get("created", [])
    }

    rows: list[dict[str, Any]] = []
    for item in raw:
        code7 = str(item["id"])
        effective = installed.get(code7)
        if effective is not None and effective > year:
            continue
        micro = item.get("microrregiao") or {}
        meso = micro.get("mesorregiao") or {}
        imm = item.get("regiao-imediata") or {}
        inter = imm.get("regiao-intermediaria") or {}
        uf = meso.get("UF") or inter.get("UF") or {}
        region = uf.get("regiao") or {}
        rows.append(
            {
                "code7": code7,
                "code6": code7[:6],
                "name": item.get("nome"),
                "uf_code": str(uf["id"]) if uf.get("id") is not None else code7[:2],
                "uf_abbr": uf.get("sigla"),
                "uf_name": uf.get("nome"),
                "region": region.get("nome"),
                "region_abbr": region.get("sigla"),
                "microregion": str(micro["id"]) if micro.get("id") is not None else None,
                "microregion_name": micro.get("nome"),
                "mesoregion": str(meso["id"]) if meso.get("id") is not None else None,
                "mesoregion_name": meso.get("nome"),
                "immediate_region": str(imm["id"]) if imm.get("id") is not None else None,
                "immediate_region_name": imm.get("nome"),
                "intermediate_region": str(inter["id"]) if inter.get("id") is not None else None,
                "intermediate_region_name": inter.get("nome"),
                "health_region": None,
                "health_region_name": None,
                "lattice_year": year,
            }
        )

    df = pl.DataFrame(rows, schema=_MUNICIPALITY_SCHEMA).sort("code7")

    if assert_count:
        expected = EXPECTED_COUNTS.get(year)
        if expected is not None and df.height != expected:
            raise LatticeError(
                f"lattice for {year} has {df.height} municipalities, expected {expected}. "
                "Either IBGE changed the territorial division or "
                "municipality_changes.yaml is out of date; do not proceed on a "
                "lattice of unknown vintage."
            )
    return df


def attach_health_regions(
    df: pl.DataFrame,
    mapping: Mapping[str, tuple[str, str]] | pl.DataFrame,
) -> pl.DataFrame:
    """Fill ``health_region``/``health_region_name`` from a DATASUS extract.

    ``mapping`` is either ``{code7: (region_id, region_name)}`` or a frame with
    ``code7``, ``health_region`` and ``health_region_name``. Record the vintage
    of the extract alongside it: health regions are redrawn by state
    resolutions, so a mapping without a date is not reproducible.
    """
    if isinstance(mapping, pl.DataFrame):
        table = mapping.select("code7", "health_region", "health_region_name")
    else:
        table = pl.DataFrame(
            {
                "code7": list(mapping.keys()),
                "health_region": [v[0] for v in mapping.values()],
                "health_region_name": [v[1] for v in mapping.values()],
            }
        )
    return (
        df.drop("health_region", "health_region_name")
        .join(table, on="code7", how="left")
        .sort("code7")
    )


# --------------------------------------------------------------------------
# Territorial change data
# --------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _load_changes_cached(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def load_changes(path: Path = CHANGES_PATH) -> dict[str, Any]:
    """Read ``municipality_changes.yaml``."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; the lattice cannot be dated without it")
    return yaml.safe_load(_load_changes_cached(str(path)))


def created_since(year: int, path: Path = CHANGES_PATH) -> list[dict[str, Any]]:
    """Municipalities that entered the lattice after ``year``."""
    return [
        entry
        for entry in load_changes(path).get("created", [])
        if int(entry["effective_year"]) > int(year)
    ]


# --------------------------------------------------------------------------
# Crosswalks
# --------------------------------------------------------------------------

CROSSWALK_SCHEMA: dict[str, pl.DataType] = {
    "from_code7": pl.Utf8,
    "to_code7": pl.Utf8,
    "relation": pl.Utf8,
    "resolvable": pl.Boolean,
    "note": pl.Utf8,
}


def build_crosswalk(
    from_year: int,
    to_year: int,
    *,
    path: Path = CHANGES_PATH,
    refresh: bool = False,
) -> pl.DataFrame:
    """Map the ``from_year`` lattice onto the ``to_year`` lattice.

    Rows are ``(from_code7, to_code7, relation, resolvable, note)``.

    Going **backward** in time (``to_year < from_year``) every municipality has
    exactly one destination: a child created after ``to_year`` maps to the
    parent it was carved from, and that is exact -- the child's territory *was*
    part of the parent. ``relation = "child_to_parent"``, ``resolvable = True``.
    A child with several parents produces several rows; the split of its cases
    between them is not determined by this table and such rows are marked
    ``resolvable = False``.

    Going **forward** (``to_year > from_year``) a parent that has since split
    maps to itself, and the rows for its children are marked
    ``relation = "parent_to_children"`` with ``resolvable = False``. This is the
    honest answer: a case notified in 2005 under Santarem cannot be attributed
    to Santarem-or-Mojui, because the notification predates the distinction.
    Do not interpolate. Use :func:`amc_groups` and analyse the merged unit.
    """
    changes = load_changes(path)
    created = changes.get("created", [])

    from_codes = set(load_municipalities(from_year, refresh=refresh)["code7"].to_list())
    to_codes = set(load_municipalities(to_year, refresh=refresh)["code7"].to_list())

    parents_of: dict[str, list[str]] = {
        str(e["code7"]): [str(p) for p in (e.get("parents") or [])] for e in created
    }
    unknown_parents = {
        str(e["code7"]) for e in created if not (e.get("parents") or [])
    }

    rows: list[dict[str, Any]] = []
    backward = int(to_year) < int(from_year)

    for code in sorted(from_codes):
        if code in to_codes:
            rows.append(
                {
                    "from_code7": code,
                    "to_code7": code,
                    "relation": "identity",
                    "resolvable": True,
                    "note": None,
                }
            )
            continue

        # Present at from_year, absent at to_year: only possible going backward,
        # to before the municipality was installed.
        parents = [p for p in parents_of.get(code, []) if p in to_codes]
        if not parents:
            rows.append(
                {
                    "from_code7": code,
                    "to_code7": None,
                    "relation": "unmapped",
                    "resolvable": False,
                    "note": (
                        "no parent recorded in municipality_changes.yaml"
                        if code in unknown_parents
                        else "parent is itself absent from the target lattice"
                    ),
                }
            )
            continue
        for parent in parents:
            rows.append(
                {
                    "from_code7": code,
                    "to_code7": parent,
                    "relation": "child_to_parent",
                    "resolvable": len(parents) == 1,
                    "note": (
                        None
                        if len(parents) == 1
                        else f"carved from {len(parents)} municipalities; share not determined here"
                    ),
                }
            )

    if not backward:
        # Forward: flag the target-lattice municipalities that did not exist at
        # from_year, so a caller cannot mistake their absence for zero cases.
        for code in sorted(to_codes - from_codes):
            for parent in parents_of.get(code, []) or [None]:
                rows.append(
                    {
                        "from_code7": parent,
                        "to_code7": code,
                        "relation": "parent_to_children",
                        "resolvable": False,
                        "note": (
                            f"{code} did not exist in {from_year}; its territory was inside "
                            f"{parent or 'an unrecorded parent'}. Cases cannot be "
                            "disaggregated -- merge via amc_groups instead."
                        ),
                    }
                )

    return pl.DataFrame(rows, schema=CROSSWALK_SCHEMA).sort(["from_code7", "to_code7"])


# --------------------------------------------------------------------------
# Areas Minimas Comparaveis
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AMCReport:
    """Summary of an AMC construction."""

    from_year: int
    to_year: int
    n_units: int
    n_merged_units: int
    n_municipalities_merged: int
    unresolved: tuple[str, ...]


def amc_groups(
    from_year: int = 2000,
    to_year: int = DEFAULT_YEAR,
    *,
    path: Path = CHANGES_PATH,
    refresh: bool = False,
    return_report: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, AMCReport]:
    """Areas Minimas Comparaveis spanning ``from_year``..``to_year``.

    An AMC is the coarsest partition of the territory that is stable across the
    window: every municipality created inside the window is merged back with all
    of its parents, transitively. A parent that donated land to two different
    children ends up in one AMC with both, and if those children also had other
    parents, those parents join too -- the AMC is the connected component of the
    parent/child graph, which is why this is computed rather than listed.

    Returns a frame of ``code7``, ``amc_id``, ``amc_size``, ``lattice_year``,
    where ``amc_id`` is the smallest ``code7`` in the component, chosen because
    it is stable under recomputation.

    Municipalities created in the window whose parents are not recorded are
    reported and left as singletons; that is visibly wrong rather than quietly
    wrong.
    """
    lo, hi = (int(from_year), int(to_year)) if from_year <= to_year else (int(to_year), int(from_year))
    target = load_municipalities(hi, refresh=refresh)
    codes = target["code7"].to_list()
    code_set = set(codes)

    adjacency: dict[str, set[str]] = defaultdict(set)
    unresolved: list[str] = []
    for entry in load_changes(path).get("created", []):
        effective = int(entry["effective_year"])
        if not (lo < effective <= hi):
            continue
        child = str(entry["code7"])
        if child not in code_set:
            continue
        parents = [str(p) for p in (entry.get("parents") or []) if str(p) in code_set]
        if not parents:
            unresolved.append(child)
            continue
        for parent in parents:
            adjacency[child].add(parent)
            adjacency[parent].add(child)

    seen: set[str] = set()
    assignment: dict[str, str] = {}
    for code in codes:
        if code in seen:
            continue
        stack = [code]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, frozenset()) - component)
        amc_id = min(component)
        for node in component:
            assignment[node] = amc_id
        seen |= component

    sizes: dict[str, int] = defaultdict(int)
    for amc_id in assignment.values():
        sizes[amc_id] += 1

    df = pl.DataFrame(
        {
            "code7": codes,
            "amc_id": [assignment[c] for c in codes],
        }
    ).with_columns(
        pl.col("amc_id").replace_strict(sizes, return_dtype=pl.Int32).alias("amc_size"),
        pl.lit(hi, dtype=pl.Int32).alias("lattice_year"),
        pl.lit(lo, dtype=pl.Int32).alias("comparable_from"),
    ).sort(["amc_id", "code7"])

    if not return_report:
        return df

    merged = df.filter(pl.col("amc_size") > 1)
    report = AMCReport(
        from_year=lo,
        to_year=hi,
        n_units=df["amc_id"].n_unique(),
        n_merged_units=merged["amc_id"].n_unique(),
        n_municipalities_merged=merged.height,
        unresolved=tuple(sorted(unresolved)),
    )
    return df, report


def to_amc(
    df: pl.DataFrame,
    *,
    code_column: str = "code7",
    from_year: int = 2000,
    to_year: int = DEFAULT_YEAR,
    path: Path = CHANGES_PATH,
) -> pl.DataFrame:
    """Attach ``amc_id`` to a frame keyed by ``code_column``.

    Aggregate counts *and* denominators by ``amc_id`` afterwards. Aggregating
    only one of them is the classic way to produce an incidence rate that is
    wrong by a factor of the merge size.
    """
    groups = amc_groups(from_year, to_year, path=path)
    key = df[code_column].dtype
    right = groups.select(
        pl.col("code7").cast(key).alias(code_column), "amc_id", "amc_size"
    )
    return df.join(right, on=code_column, how="left")


def impute_created_unit_covariates(
    frame: pl.DataFrame,
    target_codes: Sequence[str],
    value_columns: Sequence[str],
    *,
    code_column: str = "munic_code",
    key_columns: Sequence[str] = ("period",),
    weight_column: str | None = None,
    source_lattice_year: int,
    include_same_year: bool = False,
    path: Path = CHANGES_PATH,
) -> pl.DataFrame:
    """Complete smooth covariates for units absent from an old lattice.

    Municipal products are often published once on a fixed boundary vintage.
    This function returns *only* the missing target-unit rows, copying a sole
    parent's covariate or averaging all recorded parents. When ``weight_column``
    is supplied, the latter is a support-weighted mean (for example households
    for a sanitation share); otherwise it is explicitly labelled unweighted.
    The method and parent codes are emitted on every row, so the approximation
    cannot be mistaken for a direct observation.

    This is restricted to smooth intensive covariates. Never use it to
    disaggregate counts, populations or events; use :func:`amc_groups` and
    analyse the stable merged geography for those quantities.

    ``include_same_year`` permits a recorded successor whose effective year
    equals the product year. It is intended for source-specific publication
    lag (for example, a municipal economic table that omits a newly introduced
    code even though the population lattice already carries it). The output
    remains a proxy and must not be relabelled observed.
    """
    required = {
        code_column,
        *key_columns,
        *value_columns,
        *([weight_column] if weight_column else []),
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise LatticeError(
            f"covariate frame lacks required columns {sorted(missing_columns)}"
        )
    numeric_columns = [*value_columns, *([weight_column] if weight_column else [])]
    non_numeric = [name for name in numeric_columns if not frame.schema[name].is_numeric()]
    if non_numeric:
        raise LatticeError(
            f"created-unit imputation accepts numeric covariates only: {non_numeric}"
        )
    keys = [code_column, *key_columns]
    duplicate_rows = frame.height - frame.select(keys).unique().height
    if duplicate_rows:
        raise LatticeError(
            f"covariate frame has {duplicate_rows} duplicate rows on {keys}"
        )

    target = {str(code) for code in target_codes}
    present = set(frame[code_column].cast(pl.Utf8).unique().to_list())
    absent = sorted(target - present)
    output_schema = {
        **{name: frame.schema[name] for name in keys},
        **{name: pl.Float64 for name in value_columns},
        "territorial_imputation": pl.Utf8,
        "territorial_source_codes": pl.Utf8,
        "territorial_source_count": pl.Int16,
    }
    if not absent:
        return pl.DataFrame(schema=output_schema)

    changes = {
        str(entry["code7"]): [str(code) for code in entry.get("parents") or []]
        for entry in load_changes(path).get("created", [])
        if int(entry["effective_year"]) > source_lattice_year
        or (
            include_same_year
            and int(entry["effective_year"]) == source_lattice_year
        )
    }
    unresolved = {
        child: changes.get(child, [])
        for child in absent
        if not changes.get(child)
        or any(parent not in present for parent in changes[child])
    }
    if unresolved:
        raise LatticeError(
            "target lattice has absent units without a complete recorded parent "
            f"set: {unresolved}"
        )

    expected_periods = frame.select(key_columns).unique().height
    blocks: list[pl.DataFrame] = []
    for child in absent:
        parents = sorted(changes[child])
        source = frame.filter(pl.col(code_column).cast(pl.Utf8).is_in(parents))
        if weight_column:
            bad_weights = source.filter(
                pl.col(weight_column).is_null()
                | ~pl.col(weight_column).is_finite()
                | (pl.col(weight_column) <= 0)
            )
            if bad_weights.height:
                raise LatticeError(
                    f"{child}: {bad_weights.height} invalid support weights in "
                    f"{weight_column}"
                )
            value_aggregations = [
                (
                    (pl.col(name) * pl.col(weight_column)).sum()
                    / pl.col(weight_column).sum()
                ).alias(name)
                for name in value_columns
            ]
        else:
            value_aggregations = [
                pl.col(name).mean().alias(name) for name in value_columns
            ]
        block = (
            source.group_by(list(key_columns))
            .agg(
                *value_aggregations,
                pl.col(code_column).n_unique().alias("_n_parents"),
            )
            .with_columns(
                pl.lit(child).alias(code_column),
                pl.lit(
                    "single_parent_copy"
                    if len(parents) == 1
                    else (
                        "parent_support_weighted_mean"
                        if weight_column
                        else "parent_mean_unweighted"
                    )
                ).alias("territorial_imputation"),
                pl.lit("+".join(parents)).alias("territorial_source_codes"),
                pl.lit(len(parents), dtype=pl.Int16).alias("territorial_source_count"),
            )
        )
        incomplete = block.filter(pl.col("_n_parents") != len(parents))
        if block.height != expected_periods or incomplete.height:
            raise LatticeError(
                f"{child}: parent covariates do not cover every key with all "
                f"{len(parents)} parent(s)"
            )
        blocks.append(
            block.drop("_n_parents").select(
                code_column,
                *key_columns,
                *value_columns,
                "territorial_imputation",
                "territorial_source_codes",
                "territorial_source_count",
            )
        )
    return pl.concat(blocks, how="vertical_relaxed").sort(keys)


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------

#: Municipalities used to pin the check-digit implementation. Chosen so the
#: three are in different states and exercise different weighted-sum paths.
CHECK_DIGIT_FIXTURES: tuple[tuple[str, str, str], ...] = (
    ("3550308", "355030", "Sao Paulo"),
    ("1200401", "120040", "Rio Branco"),
    ("4314902", "431490", "Porto Alegre"),
    # One of the nine irregular codes, pinned so a refactor that drops the
    # exception table fails here rather than in a join six modules away.
    ("2201919", "220191", "Bom Principio do Piaui"),
)


def self_check(*, refresh: bool = False) -> dict[str, Any]:
    """Assert the module's invariants against live data. Cheap; run it in CI.

    Verifies the check digit round-trips for the pinned fixtures, that it agrees
    with IBGE for every municipality in the live lattice, and that the 2022
    lattice has 5570 rows.
    """
    results: dict[str, Any] = {"fixtures": {}}
    for code7, code6, name in CHECK_DIGIT_FIXTURES:
        forward = code6_to_code7(code6)
        backward = code7_to_code6(code7)
        results["fixtures"][name] = {
            "code6_to_code7": forward,
            "code7_to_code6": backward,
            "ok": forward == code7 and backward == code6,
        }
    if not all(v["ok"] for v in results["fixtures"].values()):
        raise LatticeError(f"check-digit fixtures failed: {results['fixtures']}")

    df = load_municipalities(DEFAULT_YEAR, refresh=refresh)
    results["n_municipalities_2022"] = df.height
    if df.height != EXPECTED_COUNTS[DEFAULT_YEAR]:
        raise LatticeError(f"2022 lattice has {df.height} rows, expected 5570")

    # Every code in the live lattice must round-trip through code6 and back.
    roundtrip = df.with_columns(code6_to_code7_expr("code6").alias("_rebuilt"))
    bad = roundtrip.filter(pl.col("_rebuilt") != pl.col("code7"))
    results["n_check_digit_mismatches"] = bad.height
    results["check_digit_mismatches"] = bad.select("code7", "code6", "name", "_rebuilt").to_dicts()[:20]
    if bad.height:
        raise LatticeError(
            f"{bad.height} municipalities disagree with the check-digit algorithm: "
            f"{results['check_digit_mismatches']}"
        )

    # And the exception table must be exactly the set of disagreements -- no
    # more (a stale override would corrupt a valid code) and no fewer.
    raw = df.filter(
        pl.col("code6").map_elements(
            lambda c: str(check_digit(c)), return_dtype=pl.Utf8
        )
        != pl.col("code7").str.slice(6, 1)
    )
    observed = set(raw["code6"].to_list())
    declared = set(CHECK_DIGIT_EXCEPTIONS)
    results["n_check_digit_exceptions"] = len(observed)
    if observed != declared:
        raise LatticeError(
            "CHECK_DIGIT_EXCEPTIONS is out of date: live lattice disagrees on "
            f"{sorted(observed - declared)}, table declares stale "
            f"{sorted(declared - observed)}"
        )

    # code6 must stay injective, or DATASUS six-digit keys are ambiguous.
    results["n_code6_collisions"] = df.height - df["code6"].n_unique()
    if results["n_code6_collisions"]:
        raise LatticeError("two municipalities share a six-digit code")

    results["ok"] = True
    return results


__all__ = [
    "DEFAULT_YEAR",
    "EXPECTED_COUNTS",
    "CHANGES_PATH",
    "LatticeError",
    "check_digit",
    "code6_to_code7",
    "code7_to_code6",
    "code6_to_code7_expr",
    "code7_to_code6_expr",
    "resolve_municipality_code",
    "is_valid_code7",
    "load_municipalities",
    "attach_health_regions",
    "load_changes",
    "created_since",
    "build_crosswalk",
    "amc_groups",
    "to_amc",
    "impute_created_unit_covariates",
    "self_check",
]
