"""YAML-backed registry of the SIDRA tables this project uses.

The registry is *rebuilt from live ``/metadados``*, never transcribed from a
document. SIDRA_COMPENDIUM.md is a table-selection seed and nothing more: table
ids get retired, variables get renumbered between census rounds, and a category
list copied by hand in 2024 is a silent source of wrong joins in 2026.

Consequently :func:`refresh_registry` writes what the API says, and records the
seed's editorial intent (why we want the table, which variables are analytic)
alongside it rather than in place of it. A table that 404s is written with
``status: unverified`` and a note. It is never dropped -- a missing entry looks
like a table nobody wanted, which is a different and much worse claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from brepi.sources.sidra import api

REGISTRY_PATH = Path(__file__).resolve().with_name("registry.yaml")

#: Territorial level the project joins on. Everything else is context.
PRIMARY_LEVEL = "N6"


@dataclass(frozen=True)
class Seed:
    """Editorial intent for one table. Not a claim about its current shape."""

    agregado: int
    role: str
    why: str
    #: Variables we expect to use. Validated against live metadata on refresh.
    expect_variables: tuple[str, ...] = ()
    #: Territorial level we intend to extract at.
    level: str = PRIMARY_LEVEL
    #: ``modeling`` uses non-total categories; ``validation`` fetches totals so
    #: that :func:`brepi.sources.sidra.extract.check_margins` has something to
    #: reconcile against.
    purpose: str = "modeling"
    tags: tuple[str, ...] = ()
    note: str | None = None


#: Tables this study needs. Ids are seeds; every one is checked live on refresh.
SEEDS: tuple[Seed, ...] = (
    # ---------------------------------------------------------------- population
    Seed(
        6579,
        role="population.estimates.annual",
        why=(
            "Annual municipal population estimates, the denominator spine for "
            "inter-censal years. Note the series omits census years."
        ),
        expect_variables=("9324",),
        tags=("denominator", "population"),
    ),
    Seed(
        9514,
        role="population.census2022.age_sex",
        why="Censo 2022 population by sex and single/grouped age. Marginal check for 9606.",
        expect_variables=("93",),
        purpose="validation",
        tags=("denominator", "population", "census2022"),
    ),
    Seed(
        9606,
        role="population.census2022.age_sex_colour",
        why=(
            "Population by colour/race, sex and age for 2010 and 2022. The joint "
            "denominator tensor anchor. Race here is IBGE self-declaration and must "
            "not be silently equated with the administratively recorded race on a "
            "SINAN notification."
        ),
        expect_variables=("93",),
        tags=("denominator", "population", "census2010", "census2022"),
        note=(
            "The five colour/race categories do not sum to the race Total: in "
            "Censo 2022 they fall 400 short for men in Sao Paulo (5386295 against "
            "5386695), 15 in Porto Alegre, 2 in Rio Branco. The residual is "
            "undeclared race, which is inside the Total and outside every "
            "category. Race-specific denominators built from this table are "
            "therefore slightly smaller than the population they should cover, and "
            "a race-specific rate computed against them is correspondingly "
            "inflated. Reconcile with check_margins before use and decide "
            "explicitly how to treat the residual."
        ),
    ),
    Seed(
        4709,
        role="population.census2022.totals",
        why="Censo 2022 municipal totals and growth rate. Top-level margin check.",
        expect_variables=("93",),
        purpose="validation",
        tags=("denominator", "population", "census2022"),
    ),
    # ------------------------------------------------- census 2022 sanitation
    Seed(
        6803,
        role="sanitation.census2022.water_network",
        why=(
            "Households by connection to the general water distribution network. "
            "Piped-water deficit is a standing leptospirosis exposure proxy."
        ),
        expect_variables=("381",),
        tags=("sanitation", "water", "census2022", "exposure"),
    ),
    Seed(
        6804,
        role="sanitation.census2022.water_supply",
        why="Households by internal water canalisation and main supply source.",
        expect_variables=("381",),
        tags=("sanitation", "water", "census2022", "exposure"),
    ),
    Seed(
        6805,
        role="sanitation.census2022.sewage",
        why=(
            "Households by type of sewage disposal. Open sewage and rudimentary pits "
            "are the canonical municipal-level leptospirosis exposure covariate."
        ),
        expect_variables=("381",),
        tags=("sanitation", "sewage", "census2022", "exposure"),
    ),
    Seed(
        6806,
        role="sanitation.census2022.bathroom_sewage",
        why="Households cross-classified by bathroom availability and sewage type.",
        expect_variables=("381",),
        tags=("sanitation", "sewage", "census2022", "exposure"),
    ),
    Seed(
        6892,
        role="sanitation.census2022.refuse",
        why=(
            "Households by refuse destination. Uncollected refuse drives commensal "
            "rodent density and is the second standing exposure proxy."
        ),
        expect_variables=("381",),
        tags=("sanitation", "refuse", "census2022", "exposure"),
    ),
    Seed(
        9860,
        role="sanitation.census2022.household_panel",
        why=(
            "Households and residents by dwelling type, water, bathroom, sewage and "
            "urban/rural location. Metadata advertises 2010 and 2022, but the "
            "national extraction verified 2026-07-30 returned NOT_AVAILABLE for "
            "every requested 2010 municipal cell. Do not use it longitudinally."
        ),
        expect_variables=("381", "382"),
        tags=("sanitation", "census2010", "census2022", "exposure"),
    ),
    # ------------------------------------------------- census 2010 comparators
    Seed(
        1394,
        role="sanitation.census2010.sewage_final",
        why=(
            "Final 2010 Census household table with a direct public/stormwater "
            "sewer category. This replaces preliminary table 3154 in the "
            "longitudinal sanitation backbone."
        ),
        expect_variables=("96",),
        tags=("sanitation", "sewage", "census2010", "exposure"),
    ),
    Seed(
        2065,
        role="sanitation.census2010.water",
        why="Censo 2010 households by piped water and supply form. Pre-2022 comparator.",
        expect_variables=("96",),
        tags=("sanitation", "water", "census2010"),
    ),
    Seed(
        3154,
        role="sanitation.census2010.bathroom_sewage",
        why=(
            "Preliminary-universe table retained for provenance only. National "
            "margin checks failed in 119 municipalities; do not use as the "
            "confirmatory 2010 anchor."
        ),
        expect_variables=("96",),
        tags=("sanitation", "sewage", "census2010"),
    ),
    Seed(
        3218,
        role="sanitation.census2010.water_sewage_refuse_power",
        why=(
            "Censo 2010 households cross-classified by water, sewage, refuse and "
            "electricity. Category systems differ from 2022 and must be harmonised "
            "downstream, not assumed comparable."
        ),
        expect_variables=("96",),
        tags=("sanitation", "census2010"),
    ),
    # ------------------------------------- favelas e comunidades urbanas (2022)
    Seed(
        9883,
        role="favelas.census2022.count_by_municipality",
        why=(
            "Number of favelas and urban communities per municipality, Censo 2022. "
            "The entry point to the Favelas e Comunidades Urbanas family, which "
            "postdates SIDRA_COMPENDIUM.md and was located by name search against "
            "the live catalogue."
        ),
        expect_variables=("9910",),
        tags=("favela", "census2022", "exposure"),
        note=(
            "Covers only the 656 municipalities with at least one favela. A "
            "municipality absent from this table has none; fill zero, not null."
        ),
    ),
    Seed(
        9887,
        role="favelas.census2022.households_population",
        why="Dwellings and resident population per favela, by dwelling species.",
        tags=("favela", "census2022", "exposure"),
        note=(
            "Published at N6, but only for the 656 municipalities that have at "
            "least one favela. Absence from this table means 'no favela', not "
            "'missing'; joining it to the 5570-municipality lattice must fill zero, "
            "not null."
        ),
    ),
    Seed(
        9888,
        role="favelas.census2022.population_area_density",
        why="Population, area and demographic density per favela. Crowding proxy.",
        tags=("favela", "census2022", "exposure"),
        note=(
            "Published at N6 for the 655-656 municipalities that have a favela; "
            "absent means 'no favela', not 'missing'."
        ),
    ),
    Seed(
        9892,
        role="favelas.census2022.bathroom_sewage",
        why="Households in favelas by bathroom availability and sewage type.",
        tags=("favela", "sanitation", "sewage", "census2022", "exposure"),
        note=(
            "Published at N6 for the 655-656 municipalities that have a favela; "
            "absent means 'no favela', not 'missing'."
        ),
    ),
    Seed(
        9893,
        role="favelas.census2022.refuse",
        why="Households in favelas by refuse destination.",
        tags=("favela", "sanitation", "refuse", "census2022", "exposure"),
        note=(
            "Published at N6 for the 655-656 municipalities that have a favela; "
            "absent means 'no favela', not 'missing'."
        ),
    ),
    Seed(
        9894,
        role="favelas.census2022.water_network",
        why="Households in favelas by connection to the general water network.",
        tags=("favela", "sanitation", "water", "census2022", "exposure"),
        note=(
            "Published at N6 for the 655-656 municipalities that have a favela; "
            "absent means 'no favela', not 'missing'."
        ),
    ),
    Seed(
        10344,
        role="favelas.census2022.sewage_in_out",
        why=(
            "Households by sewage type inside versus outside favelas, for "
            "municipalities that have them. Gives the within-municipality contrast "
            "that the N6 aggregates cannot."
        ),
        tags=("favela", "sanitation", "sewage", "census2022", "exposure"),
    ),
    # --------------------------------------------------------------- livestock
    Seed(
        3939,
        role="livestock.ppm.herds",
        why=(
            "Municipal herd counts by species. Cattle and swine density is the "
            "standard rural leptospirosis reservoir proxy; the human-to-animal ratio "
            "needs the population denominator alongside it."
        ),
        expect_variables=("105",),
        tags=("livestock", "reservoir", "annual", "exposure"),
    ),
    # ------------------------------------------------------------------- crops
    Seed(
        1612,
        role="crops.pam.temporary",
        why=(
            "Planted/harvested area and output for temporary crops. Rice and "
            "sugarcane cultivation are occupational-exposure proxies for the "
            "flooded-field transmission route."
        ),
        expect_variables=("109", "216", "214"),
        tags=("agriculture", "occupational", "annual", "exposure"),
    ),
    Seed(
        5457,
        role="crops.pam.temporary_and_permanent",
        why="Consolidated PAM series covering temporary and permanent crops.",
        expect_variables=("8331", "216", "214"),
        tags=("agriculture", "occupational", "annual", "exposure"),
    ),
    # -------------------------------------------------------------------- PNSB
    Seed(
        354,
        role="pnsb.2008.sanitation_associated_disease",
        why=(
            "Municipalities reporting occurrence of sanitation-associated diseases, "
            "by disease type, PNSB 2008. Includes leptospirosis and is a rare "
            "municipality-level administrative statement about the disease that is "
            "independent of SINAN notification."
        ),
        expect_variables=("2597",),
        tags=("pnsb", "sanitation", "leptospirosis"),
    ),
)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def discover_agregados(*terms: str, refresh: bool = False) -> list[dict[str, Any]]:
    """Search the live catalogue for aggregates whose name contains all ``terms``.

    This is how the Favelas e Comunidades Urbanas ids were found: they postdate
    SIDRA_COMPENDIUM.md and are not in it. Inventing an id is worse than not
    having one, so ids are discovered, not guessed.
    """
    needles = [t.casefold() for t in terms]
    found: list[dict[str, Any]] = []
    for pesquisa in api.get_catalog(refresh=refresh):
        for ag in pesquisa.get("agregados", []) or []:
            name = str(ag.get("nome", "")).casefold()
            if all(n in name for n in needles):
                found.append(
                    {
                        "id": int(ag["id"]),
                        "name": ag.get("nome"),
                        "survey": pesquisa.get("nome"),
                    }
                )
    return sorted(found, key=lambda d: d["id"])


# --------------------------------------------------------------------------
# Refresh
# --------------------------------------------------------------------------


def _normalise_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    periodicity = meta.get("periodicidade") or {}
    levels = meta.get("nivelTerritorial") or {}
    return {
        "id": int(meta["id"]),
        "name": meta.get("nome"),
        "survey": meta.get("pesquisa"),
        "subject": meta.get("assunto"),
        "url": meta.get("URL"),
        "periodicity": {
            "frequency": periodicity.get("frequencia"),
            "start": periodicity.get("inicio"),
            "end": periodicity.get("fim"),
        },
        "geographic_levels": {
            group: list(codes) for group, codes in levels.items() if codes
        },
        "variables": [
            {
                "id": str(v["id"]),
                "name": v.get("nome"),
                "unit": v.get("unidade"),
            }
            for v in meta.get("variaveis", []) or []
        ],
        "classifications": [
            {
                "id": str(c["id"]),
                "name": c.get("nome"),
                "n_categories": len(c.get("categorias", []) or []),
                "categories": [
                    {
                        "id": str(cat["id"]),
                        "name": cat.get("nome"),
                        "unit": cat.get("unidade"),
                        "level": cat.get("nivel"),
                        # ``nivel: 0`` marks an aggregate (Total) member. It is
                        # not a normal category and must never be summed with
                        # its siblings.
                        "is_total": cat.get("nivel") == 0,
                    }
                    for cat in c.get("categorias", []) or []
                ],
            }
            for c in meta.get("classificacoes", []) or []
        ],
    }


def _seed_index() -> dict[int, Seed]:
    return {s.agregado: s for s in SEEDS}


def refresh_registry(
    agregados: Sequence[int] | None = None,
    *,
    path: Path = REGISTRY_PATH,
    refresh: bool = False,
    with_periods: bool = True,
    with_locality_counts: bool = True,
) -> dict[str, Any]:
    """Rebuild ``registry.yaml`` from live ``/metadados`` and write it.

    ``agregados`` defaults to every seeded table. Ids not in :data:`SEEDS` are
    accepted and recorded with an empty editorial block.

    Each entry gets ``status``:

    ``verified``
        metadata retrieved and every expected variable is present;
    ``verified_with_warnings``
        metadata retrieved but something the seed expected is missing;
    ``unverified``
        the table could not be retrieved. The id stays in the file with the
        error text so the gap is visible in review.
    """
    seeds = _seed_index()
    ids = [int(a) for a in (agregados if agregados is not None else sorted(seeds))]

    entries: dict[int, dict[str, Any]] = {}
    for agregado in ids:
        seed = seeds.get(agregado)
        entry: dict[str, Any] = {
            "id": agregado,
            "role": seed.role if seed else None,
            "why": seed.why if seed else None,
            "purpose": seed.purpose if seed else "modeling",
            "tags": list(seed.tags) if seed else [],
            "intended_level": seed.level if seed else PRIMARY_LEVEL,
            "expect_variables": list(seed.expect_variables) if seed else [],
            "note": seed.note if seed else None,
        }
        try:
            meta = api.get_metadata(agregado, refresh=refresh)
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            entry["status"] = "unverified"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["note"] = (
                (entry["note"] + " | " if entry["note"] else "")
                + "Live /metadados did not resolve this id. It may have been "
                "retired or renumbered; verify before use. Not dropped, so the "
                "gap stays visible."
            )
            entries[agregado] = entry
            continue

        entry.update(_normalise_metadata(meta))
        warnings: list[str] = []

        available = {v["id"] for v in entry["variables"]}
        missing = [v for v in entry["expect_variables"] if v not in available]
        if missing:
            warnings.append(f"expected variables absent from live metadata: {missing}")

        level = entry["intended_level"]
        admin = entry["geographic_levels"].get("Administrativo", [])
        special = entry["geographic_levels"].get("Especial", [])
        if level not in admin and level not in special:
            warnings.append(
                f"intended level {level} not offered; available "
                f"{entry['geographic_levels']}"
            )

        if with_periods:
            try:
                periods = api.get_periods(agregado, refresh=refresh)
                entry["periods"] = [str(p["id"]) for p in periods]
                entry["period_modified"] = {
                    str(p["id"]): p.get("modificacao")
                    for p in periods
                    if p.get("modificacao")
                }
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"/periodos failed: {type(exc).__name__}: {exc}")
                entry["periods"] = []

        if with_locality_counts and level in admin:
            try:
                entry["n_localities"] = len(api.get_localities(agregado, level, refresh=refresh))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"/localidades/{level} failed: {type(exc).__name__}: {exc}")

        entry["status"] = "verified_with_warnings" if warnings else "verified"
        if warnings:
            entry["warnings"] = warnings
        entries[agregado] = entry

    doc = {
        "schema": "brepi.sidra.registry/1",
        "rebuilt_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": api.SIDRA_API_BASE,
        "primary_level": PRIMARY_LEVEL,
        "n_tables": len(entries),
        "n_verified": sum(1 for e in entries.values() if e["status"] == "verified"),
        "n_unverified": sum(1 for e in entries.values() if e["status"] == "unverified"),
        "tables": {str(k): entries[k] for k in sorted(entries)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    return doc


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Read the registry. Raises if it has never been built."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run refresh_registry() to build it from live metadata"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_table(agregado: int, path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """One registry entry, by aggregate id."""
    doc = load_registry(path)
    try:
        return doc["tables"][str(int(agregado))]
    except KeyError as exc:
        raise KeyError(f"agregado {agregado} is not in the registry at {path}") from exc


def tables_with_tag(tag: str, path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    """Registry entries carrying ``tag``, e.g. ``"exposure"``."""
    doc = load_registry(path)
    return [e for e in doc["tables"].values() if tag in (e.get("tags") or [])]


def unverified(path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    """Entries that live metadata could not confirm. Review before every run."""
    doc = load_registry(path)
    return [
        e
        for e in doc["tables"].values()
        if e.get("status") in {"unverified", "verified_with_warnings"}
    ]


def _classification(agregado: int, classification: int | str, path: Path) -> dict[str, Any]:
    entry = get_table(agregado, path)
    for clf in entry.get("classifications", []) or []:
        if str(clf["id"]) == str(classification):
            return clf
    raise KeyError(f"agregado {agregado} has no classification {classification}")


def categories(
    agregado: int,
    classification: int | str,
    *,
    level: int | None = 1,
    include_total: bool = False,
    path: Path = REGISTRY_PATH,
) -> list[str]:
    """Category ids for a classification.

    SIDRA classifications are **hierarchies**, and ``nivel`` is the depth:
    ``0`` is the Total, ``1`` is the partition of that Total, ``2`` subdivides
    particular level-1 members. In table 6805 the level-1 member "Rede geral,
    rede pluvial ou fossa ligada a rede" contains the two level-2 members "Rede
    geral ou pluvial" and "Fossa septica ligada a rede". Requesting every
    non-total category and summing it therefore double-counts -- by 95 percent
    of the total, in Sao Paulo's case -- and the resulting share looks plausible
    enough to survive review.

    So the default is ``level=1``: the categories that actually partition the
    Total, and the only selection for which
    :func:`brepi.sources.sidra.extract.check_margins` can pass. Pass
    ``level=2`` for the finer breakdown, or ``level=None`` for everything, and
    reconcile those against their level-1 parent rather than against the Total.
    """
    clf = _classification(agregado, classification, path)
    return [
        c["id"]
        for c in clf["categories"]
        if (include_total or not c.get("is_total"))
        and (level is None or c.get("level") == level or (include_total and c.get("is_total")))
    ]


def category_levels(
    agregado: int, classification: int | str, *, path: Path = REGISTRY_PATH
) -> dict[str, int]:
    """``{category_id: nivel}`` for a classification. Inspect before selecting."""
    clf = _classification(agregado, classification, path)
    return {c["id"]: c.get("level") for c in clf["categories"]}


def total_category(
    agregado: int, classification: int | str, *, path: Path = REGISTRY_PATH
) -> str | None:
    """The Total category id of a classification, if it has one."""
    entry = get_table(agregado, path)
    for clf in entry.get("classifications", []) or []:
        if str(clf["id"]) == str(classification):
            for c in clf["categories"]:
                if c.get("is_total"):
                    return c["id"]
    return None


def seeded_ids() -> list[int]:
    return sorted(s.agregado for s in SEEDS)


def iter_seeds(tag: str | None = None) -> Iterable[Seed]:
    for seed in SEEDS:
        if tag is None or tag in seed.tags:
            yield seed


__all__ = [
    "REGISTRY_PATH",
    "PRIMARY_LEVEL",
    "Seed",
    "SEEDS",
    "seeded_ids",
    "iter_seeds",
    "discover_agregados",
    "refresh_registry",
    "load_registry",
    "get_table",
    "tables_with_tag",
    "unverified",
    "categories",
    "category_levels",
    "total_category",
]
