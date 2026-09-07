"""Thin, correct client for the IBGE aggregates API v3.

This module is the *only* place in the package that builds a SIDRA URL. Every
byte it retrieves goes through :func:`brepi.io.cache.fetch`, so a rerun against
a frozen cache is provably offline and every response carries a provenance
sidecar recording the full URL and the normalised query.

Response shape
--------------
The default (non-``flat``) view returns, per requested variable::

    [{"id": "93", "variavel": ..., "unidade": "Pessoas",
      "resultados": [
        {"classificacoes": [{"id": "2", "nome": "Sexo",
                             "categoria": {"4": "Homens"}}, ...],
         "series": [{"localidade": {"id": "3550308",
                                    "nivel": {"id": "N6", "nome": "Municipio"},
                                    "nome": "Sao Paulo (SP)"},
                     "serie": {"2022": "5386695"}}]}]}]

We use this view rather than ``view=flat`` because it is self-describing: the
category tuple is attached to the block, so nothing has to be recovered from a
header row. Note that classifications the caller did *not* select still appear,
pinned to their ``Total`` category -- that is the API's default and it must be
recorded, not discarded, or a later margin check will silently compare a
sex-specific figure with a sex-total one.

Typed missingness
-----------------
SIDRA does not return floats. It returns strings, some of which are sentinels
with genuinely different meanings. Collapsing them all to ``0`` inflates counts;
collapsing them all to null destroys the distinction between a true zero and a
suppressed cell. We keep ``value_raw`` verbatim, map the documented absolute-zero
symbol ``-`` to numeric zero, and record the reason in ``value_status``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import httpx
import polars as pl

from brepi.config import (
    HTTP_BACKOFF_SECONDS,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    SIDRA_API_BASE,
)
from brepi.io.cache import fetch, sha256_bytes

TOOL_VERSION = "brepi.sources.sidra.api/1"
SOURCE_ID = "ibge.sidra.v3"

#: HTTP statuses worth retrying. Everything else is a deterministic failure and
#: retrying it just wastes the server's patience and ours.
_RETRYABLE = frozenset({408, 425, 429, 500, 502, 503, 504})


class SidraError(RuntimeError):
    """A SIDRA request failed in a way that retrying will not fix."""


class SidraTransportError(RuntimeError):
    """A SIDRA request failed after exhausting the retry budget."""


class ValueStatus(str, Enum):
    """Why a cell has (or does not have) a number.

    The names are the contract; downstream code branches on these and never on
    the raw sentinel string.
    """

    OK = "OK"
    #: ``-`` -- SIDRA's documented absolute-zero symbol.
    ABSOLUTE_ZERO = "ABSOLUTE_ZERO"
    #: Numeric ``0`` -- zero resulting from calculation or unit rounding.
    ZERO_OR_ROUNDED = "ZERO_OR_ROUNDED"
    #: ``..`` -- the cell is not applicable to this combination of dimensions.
    NOT_APPLICABLE = "NOT_APPLICABLE"
    #: ``...`` -- the datum exists in principle but was not collected/published.
    NOT_AVAILABLE = "NOT_AVAILABLE"
    #: ``X`` -- withheld to protect the confidentiality of an informant.
    SUPPRESSED = "SUPPRESSED"
    #: Something non-numeric we have not seen before. Never silently dropped.
    UNKNOWN_SENTINEL = "UNKNOWN_SENTINEL"


#: Longest-first: ``...`` must be tested before ``..``.
_SENTINELS: tuple[tuple[str, ValueStatus], ...] = (
    ("...", ValueStatus.NOT_AVAILABLE),
    ("..", ValueStatus.NOT_APPLICABLE),
    ("-", ValueStatus.ABSOLUTE_ZERO),
    ("X", ValueStatus.SUPPRESSED),
)


def classify_value(raw: Any) -> tuple[str | None, float | None, ValueStatus]:
    """Split a SIDRA cell into ``(value_raw, value_numeric, value_status)``.

    SIDRA defines ``-`` as an absolute zero and numeric ``0`` as zero caused by
    calculation or rounding in the published unit. Both carry numeric value
    zero, but their statuses remain distinct.
    """
    if raw is None:
        return None, None, ValueStatus.NOT_AVAILABLE
    text = str(raw).strip()
    if text == "":
        return text, None, ValueStatus.NOT_AVAILABLE
    for token, status in _SENTINELS:
        if text == token:
            return (
                (text, 0.0, status)
                if status is ValueStatus.ABSOLUTE_ZERO
                else (text, None, status)
            )
    try:
        value = float(text)
        status = (
            ValueStatus.ZERO_OR_ROUNDED
            if value == 0
            else ValueStatus.OK
        )
        return text, value, status
    except ValueError:
        return text, None, ValueStatus.UNKNOWN_SENTINEL


# --------------------------------------------------------------------------
# Query normalisation and cache keys
# --------------------------------------------------------------------------


def _as_str_tuple(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(str(v) for v in values)


def normalise_classifications(
    classifications: Mapping[Any, Sequence[Any]] | None,
) -> dict[str, tuple[str, ...]]:
    """Canonicalise ``{classification: [categories]}``.

    Keys and values become strings and both are sorted, so that
    ``{86: [2777, 2776]}`` and ``{"86": ["2776", "2777"]}`` produce the same
    cache key and the same URL.
    """
    if not classifications:
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for clf, cats in classifications.items():
        cat_tuple = _as_str_tuple(cats)
        if not cat_tuple:
            raise SidraError(f"classification {clf!r} was given no categories")
        out[str(clf)] = tuple(sorted(cat_tuple, key=_sort_key))
    return {k: out[k] for k in sorted(out, key=_sort_key)}


def _sort_key(token: str) -> tuple[int, Any]:
    """Sort numerically when possible, lexically otherwise ("all", "allxp")."""
    return (0, int(token)) if token.lstrip("-").isdigit() else (1, token)


def classification_expr(classifications: Mapping[str, Sequence[str]]) -> str:
    """Render ``86[2776,2777]|2[4,5]`` from a normalised mapping."""
    return "|".join(f"{clf}[{','.join(cats)}]" for clf, cats in classifications.items())


def locality_expr(level: str, localities: str | Sequence[Any] = "all") -> str:
    """Render a SIDRA locality expression, always carrying its level.

    ``"all"`` means every locality the table supports at ``level`` and renders
    as the bare level (``N6``). A sequence renders as ``N6[3550308,1200401]``.
    A string that already contains ``[`` or ``]`` is passed through as a
    caller-supplied expression (``N6[N3[27]]``).
    """
    if isinstance(localities, str):
        if localities == "all":
            return level
        if "[" in localities or localities.startswith("N"):
            return localities
        return f"{level}[{localities}]"
    ids = _as_str_tuple(localities)
    if not ids:
        raise SidraError("empty locality selection")
    return f"{level}[{','.join(sorted(ids, key=_sort_key))}]"


def _canonical_query(
    agregado: int,
    periods: Sequence[str],
    variables: Sequence[str],
    level: str,
    localities: str | Sequence[Any],
    classifications: Mapping[str, Sequence[str]],
    view: str | None,
) -> dict[str, Any]:
    """A fully-normalised, order-independent description of one value query."""
    return {
        "endpoint": "values",
        "agregado": int(agregado),
        "periods": sorted(_as_str_tuple(periods), key=_sort_key),
        "variables": sorted(_as_str_tuple(variables), key=_sort_key),
        "level": level,
        "localities": locality_expr(level, localities),
        "classifications": {k: list(v) for k, v in classifications.items()},
        "view": view,
    }


def query_hash(query: Mapping[str, Any]) -> str:
    """Stable 16-hex digest of a normalised query.

    Two logically identical queries written differently must hit the same cache
    entry; that is the whole point of normalising before hashing.
    """
    blob = json.dumps(query, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(blob.encode("utf-8"))[:16]


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------


def _request_json(url: str, params: Mapping[str, str] | None = None) -> tuple[bytes, dict[str, Any]]:
    """GET ``url``, retrying only transient failures. Returns raw bytes."""
    last: Exception | None = None
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
                response = client.get(url, params=dict(params or {}))
        except httpx.HTTPError as exc:  # timeouts, resets, DNS
            last = exc
        else:
            if response.status_code == 200:
                meta = {
                    "remote_modified": response.headers.get("last-modified"),
                    "params": {
                        "url": str(response.url),
                        "request_url": url,
                        "query": dict(params or {}),
                        "status_code": response.status_code,
                        "attempt": attempt,
                    },
                }
                return response.content, meta
            if response.status_code not in _RETRYABLE:
                raise SidraError(
                    f"SIDRA rejected {response.url} with HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )
            last = SidraTransportError(
                f"HTTP {response.status_code} from {response.url}: {response.text[:200]}"
            )
        if attempt < HTTP_MAX_RETRIES:
            time.sleep(HTTP_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise SidraTransportError(f"giving up on {url} after {HTTP_MAX_RETRIES} attempts") from last


def _cached_json(
    key: str,
    url: str,
    params: Mapping[str, str] | None = None,
    *,
    refresh: bool = False,
) -> Any:
    path, _prov = fetch(
        key,
        lambda: _request_json(url, params),
        source=SOURCE_ID,
        uri=url if not params else f"{url}?{httpx.QueryParams(dict(params))}",
        refresh=refresh,
        tool_version=TOOL_VERSION,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _check_api_error(payload: Any, context: str) -> None:
    """SIDRA sometimes returns a 200 whose body is an error envelope."""
    if isinstance(payload, dict) and "statusCode" in payload:
        raise SidraError(f"{context}: SIDRA returned {payload}")


# --------------------------------------------------------------------------
# Metadata endpoints
# --------------------------------------------------------------------------


def get_metadata(agregado: int, *, refresh: bool = False) -> dict[str, Any]:
    """``GET /{agregado}/metadados``.

    The returned document is the authority on which variables, classifications,
    categories and territorial levels a table actually offers. Nothing in this
    package may assume a table's shape without consulting it.
    """
    url = f"{SIDRA_API_BASE}/{int(agregado)}/metadados"
    payload = _cached_json(f"sidra/metadados/{int(agregado)}.json", url, refresh=refresh)
    _check_api_error(payload, f"metadados/{agregado}")
    if not isinstance(payload, dict) or "id" not in payload:
        raise SidraError(f"agregado {agregado}: unexpected metadata payload {str(payload)[:200]}")
    return payload


def get_periods(agregado: int, *, refresh: bool = False) -> list[dict[str, Any]]:
    """``GET /{agregado}/periodos`` -- ``[{id, literals, modificacao}, ...]``."""
    url = f"{SIDRA_API_BASE}/{int(agregado)}/periodos"
    payload = _cached_json(f"sidra/periodos/{int(agregado)}.json", url, refresh=refresh)
    _check_api_error(payload, f"periodos/{agregado}")
    return list(payload)


def get_localities(agregado: int, level: str, *, refresh: bool = False) -> list[dict[str, Any]]:
    """``GET /{agregado}/localidades/{level}``.

    Coverage is table-specific: a Censo 2010 table offers 5565 municipalities
    and a Censo 2022 table offers 5570. Never assume; ask.
    """
    url = f"{SIDRA_API_BASE}/{int(agregado)}/localidades/{level}"
    payload = _cached_json(
        f"sidra/localidades/{int(agregado)}.{level}.json", url, refresh=refresh
    )
    _check_api_error(payload, f"localidades/{agregado}/{level}")
    return list(payload)


def get_catalog(*, refresh: bool = False, **filters: str) -> list[dict[str, Any]]:
    """``GET /agregados`` -- the catalogue, grouped by pesquisa.

    Used by :mod:`brepi.sources.sidra.registry` to discover table ids by name
    rather than transcribing them from a document.
    """
    params = {k: str(v) for k, v in sorted(filters.items()) if v is not None}
    suffix = query_hash({"endpoint": "catalog", "filters": params})
    payload = _cached_json(f"sidra/catalog/{suffix}.json", SIDRA_API_BASE, params, refresh=refresh)
    _check_api_error(payload, "catalog")
    return list(payload)


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------

FACT_SCHEMA: dict[str, pl.DataType] = {
    "agregado": pl.Int64,
    "variable_id": pl.Utf8,
    "variable_name": pl.Utf8,
    "unit": pl.Utf8,
    "classification_ids": pl.List(pl.Utf8),
    "category_ids": pl.List(pl.Utf8),
    "category_labels": pl.List(pl.Utf8),
    "locality_id": pl.Utf8,
    "locality_level": pl.Utf8,
    "locality_name": pl.Utf8,
    "period": pl.Utf8,
    "value_raw": pl.Utf8,
    "value_numeric": pl.Float64,
    "value_status": pl.Utf8,
}


def empty_facts() -> pl.DataFrame:
    """An empty frame with the canonical long-fact schema."""
    return pl.DataFrame(schema=FACT_SCHEMA)


@dataclass(frozen=True)
class ValueResponse:
    """A raw values payload plus the query that produced it."""

    payload: list[dict[str, Any]]
    query: dict[str, Any]
    url: str
    cache_key: str


def flatten_values(payload: Any, agregado: int) -> pl.DataFrame:
    """Flatten the nested ``resultados``/``series`` structure to long facts.

    One row per (agregado, variable, classification-combination, locality,
    period). The classification tuple is sorted by classification id so that
    two rows describing the same cell are byte-identical regardless of the order
    the API happened to list the dimensions in.
    """
    _check_api_error(payload, f"values/{agregado}")
    rows: list[dict[str, Any]] = []
    for block in payload or []:
        variable_id = str(block.get("id", ""))
        variable_name = block.get("variavel")
        unit = block.get("unidade")
        for result in block.get("resultados", []) or []:
            pairs: list[tuple[str, str, str]] = []
            for clf in result.get("classificacoes", []) or []:
                clf_id = str(clf.get("id", ""))
                # ``categoria`` is a single-entry {id: label} map per block.
                for cat_id, cat_label in (clf.get("categoria") or {}).items():
                    pairs.append((clf_id, str(cat_id), str(cat_label)))
            pairs.sort(key=lambda p: _sort_key(p[0]))
            clf_ids = [p[0] for p in pairs]
            cat_ids = [p[1] for p in pairs]
            cat_labels = [p[2] for p in pairs]
            for serie in result.get("series", []) or []:
                loc = serie.get("localidade") or {}
                loc_level = ((loc.get("nivel") or {}).get("id")) or ""
                for period, raw in (serie.get("serie") or {}).items():
                    value_raw, value_numeric, status = classify_value(raw)
                    rows.append(
                        {
                            "agregado": int(agregado),
                            "variable_id": variable_id,
                            "variable_name": variable_name,
                            "unit": unit,
                            "classification_ids": clf_ids,
                            "category_ids": cat_ids,
                            "category_labels": cat_labels,
                            "locality_id": str(loc.get("id", "")),
                            "locality_level": str(loc_level),
                            "locality_name": loc.get("nome"),
                            "period": str(period),
                            "value_raw": value_raw,
                            "value_numeric": value_numeric,
                            "value_status": status.value,
                        }
                    )
    if not rows:
        return empty_facts()
    return pl.DataFrame(rows, schema=FACT_SCHEMA)


def fetch_values(
    agregado: int,
    periods: Sequence[str],
    variables: Sequence[str],
    *,
    level: str = "N6",
    localities: str | Sequence[Any] = "all",
    classifications: Mapping[Any, Sequence[Any]] | None = None,
    view: str | None = None,
    refresh: bool = False,
) -> pl.DataFrame:
    """Retrieve one values request and return it as long facts.

    The cache key is derived from a stable hash of the fully-normalised query,
    so reordering periods, variables or categories -- or passing ints where
    strings were passed before -- still hits the same entry.

    This function does **not** size the request. Call it through
    :mod:`brepi.sources.sidra.extract`, which plans against the cell ceiling
    first; SIDRA answers an oversized request with an HTTP 500 that looks like a
    transient fault and is not one.
    """
    if not periods:
        raise SidraError("no periods requested")
    if not variables:
        raise SidraError("no variables requested")

    norm_clf = normalise_classifications(classifications)
    query = _canonical_query(
        agregado, periods, variables, level, localities, norm_clf, view
    )
    digest = query_hash(query)

    period_expr = "|".join(query["periods"])
    variable_expr = "|".join(query["variables"])
    url = (
        f"{SIDRA_API_BASE}/{int(agregado)}"
        f"/periodos/{period_expr}/variaveis/{variable_expr}"
    )
    params: dict[str, str] = {"localidades": query["localities"]}
    if norm_clf:
        params["classificacao"] = classification_expr(norm_clf)
    if view:
        params["view"] = view

    key = f"sidra/values/{int(agregado)}/{digest}.json"
    path, _prov = fetch(
        key,
        lambda: _annotated_loader(url, params, query),
        source=SOURCE_ID,
        uri=f"{url}?{httpx.QueryParams(params)}",
        refresh=refresh,
        tool_version=TOOL_VERSION,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return flatten_values(payload, agregado)


def _annotated_loader(
    url: str, params: Mapping[str, str], query: Mapping[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    """Retrieve and attach the normalised query to the provenance record."""
    payload, meta = _request_json(url, params)
    meta = dict(meta)
    meta["params"] = {**(meta.get("params") or {}), "normalised_query": dict(query)}
    return payload, meta


def values_cache_key(
    agregado: int,
    periods: Sequence[str],
    variables: Sequence[str],
    *,
    level: str = "N6",
    localities: str | Sequence[Any] = "all",
    classifications: Mapping[Any, Sequence[Any]] | None = None,
    view: str | None = None,
) -> str:
    """The cache key :func:`fetch_values` would use. For manifest assembly."""
    query = _canonical_query(
        agregado,
        periods,
        variables,
        level,
        localities,
        normalise_classifications(classifications),
        view,
    )
    return f"sidra/values/{int(agregado)}/{query_hash(query)}.json"


__all__ = [
    "SOURCE_ID",
    "TOOL_VERSION",
    "SidraError",
    "SidraTransportError",
    "ValueStatus",
    "FACT_SCHEMA",
    "classify_value",
    "normalise_classifications",
    "classification_expr",
    "locality_expr",
    "query_hash",
    "get_metadata",
    "get_periods",
    "get_localities",
    "get_catalog",
    "fetch_values",
    "flatten_values",
    "empty_facts",
    "values_cache_key",
]
