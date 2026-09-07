"""PNSB municipal statements about sanitation services and associated disease.

Two things from the Pesquisa Nacional de Saneamento Básico, both answered by the
municipal sanitation authority rather than by the health system:

``354``  whether the municipality reports occurrence of a sanitation-associated
         disease, by disease type, 2008
``1238`` whether the municipality has each basic sanitation service, 2000 and
         2008

Why table 354 is worth having. For any notifiable disease, a municipality
reporting zero cases is ambiguous: no disease and no surveillance look
identical inside the notification system. Adding a second *health* system only
partly helps, because a municipality with no health service is missing from
both. Table 354 is not a health system at all. A municipality that told IBGE it
has the disease and told the notification system nothing is a discrepancy that
inherits none of the assumptions of a capture-recapture design.

What it is not. "Ocorrência" is not a case definition, the respondent is the
sanitation authority, and the survey plainly under-declares -- for
leptospirosis, 197 of 5,564 municipalities. Disagreement runs both ways, and
that is the premise of a two-instrument comparison rather than a defect in one.
:func:`compare_with_notifications` therefore reports both directions and refuses
to name either source as truth.

Table 1238 measures *service existence*, a binary, where the census measures
*household coverage*, a fraction. They are different quantities and must not be
concatenated into one interpolated series. The use is validation: a coverage
fraction back-extrapolated to 2008 should be near zero where PNSB says there is
no network at all.
"""

from __future__ import annotations

import polars as pl

from brepi.sources.sidra.extract import Selection

__all__ = [
    "PNSBError",
    "DISEASES",
    "SERVICES",
    "compare_with_notifications",
    "disease_occurrence_selection",
    "service_existence_selection",
    "build_disease_flag",
    "build_service_flags",
]

TAB_DISEASE = 354
TAB_SERVICE = 1238

VAR_DISEASE = "2597"
VAR_SERVICE = "2613"

#: ``Clsf 12963``. ``total`` and ``all_municipalities`` are margins, not diseases.
DISEASES: dict[str, str] = {
    "leptospirosis": "120933",
    "dengue": "120937",
    "diarrhoea": "120932",
    "helminthiasis": "120934",
    "cholera": "120935",
    "diphtheria": "120936",
    "typhus": "120938",
    "malaria": "120939",
    "hepatitis": "120940",
    "yellow_fever": "120941",
    "dermatitis": "120942",
    "respiratory": "120943",
    "other": "120944",
}

#: ``Clsf 11969``. Excludes the two margin categories.
SERVICES: dict[str, str] = {
    "water_network": "98366",
    "sewer_network": "98367",
    "solid_waste": "121192",
    "stormwater": "121194",
}


class PNSBError(RuntimeError):
    """Raised when a PNSB request or frame is internally inconsistent."""


def disease_occurrence_selection(disease: str = "leptospirosis") -> Selection:
    """Municipal report of occurrence for one sanitation-associated disease."""
    if disease not in DISEASES:
        raise PNSBError(f"unknown disease {disease!r}. Known: {sorted(DISEASES)}")
    return Selection(
        agregado=TAB_DISEASE, periods=("2008",), variables=(VAR_DISEASE,),
        classifications={"12963": (DISEASES[disease],)},
        label=f"pnsb_2008_{disease}",
    )


def service_existence_selection(
    services: tuple[str, ...] = tuple(SERVICES),
    periods: tuple[str, ...] = ("2000", "2008"),
) -> Selection:
    """Whether each basic sanitation service exists in the municipality."""
    unknown = [s for s in services if s not in SERVICES]
    if unknown:
        raise PNSBError(f"unknown service: {sorted(unknown)}. Known: {sorted(SERVICES)}")
    return Selection(
        agregado=TAB_SERVICE, periods=periods, variables=(VAR_SERVICE,),
        classifications={"11969": tuple(SERVICES[s] for s in services)},
        label=f"pnsb_services_{'_'.join(periods)}",
    )


def _flagify(facts: pl.DataFrame, variable: str) -> pl.DataFrame:
    required = {"locality_id", "variable_id", "value_numeric"}
    missing = required - set(facts.columns)
    if missing:
        raise PNSBError(f"facts frame missing columns: {sorted(missing)}")
    return facts.filter(pl.col("variable_id") == variable).with_columns(
        pl.col("locality_id").cast(pl.Utf8).str.zfill(7).alias("munic_code")
    )


def build_disease_flag(facts: pl.DataFrame, *, disease: str = "leptospirosis") -> pl.DataFrame:
    """One row per municipality: did it report the disease in 2008?

    A null value stays null. At N6 the published count is 0 or 1, so anything
    else means the response is not what this function assumes and is refused
    rather than coerced to a boolean.
    """
    tidy = _flagify(facts, VAR_DISEASE)
    bad = tidy.filter(
        pl.col("value_numeric").is_not_null() & ~pl.col("value_numeric").is_in([0.0, 1.0])
    )
    if bad.height:
        raise PNSBError(
            f"table 354 at municipality level must be 0 or 1, got "
            f"{sorted(set(bad['value_numeric'].to_list()))[:5]} -- the request "
            "was probably not restricted to a single disease category"
        )
    return tidy.select(
        "munic_code",
        pl.when(pl.col("value_numeric").is_null())
        .then(None)
        .otherwise(pl.col("value_numeric") == 1.0)
        .alias(f"pnsb_{disease}_2008"),
    ).unique(subset="munic_code").sort("munic_code")


def build_service_flags(facts: pl.DataFrame) -> pl.DataFrame:
    """One row per municipality-year, one boolean column per service."""
    tidy = _flagify(facts, VAR_SERVICE)
    if "category_ids" not in tidy.columns:
        raise PNSBError("facts frame missing 'category_ids'")
    by_id = {cid: name for name, cid in SERVICES.items()}
    tidy = tidy.with_columns(
        pl.col("period").cast(pl.Int32).alias("year"),
        pl.col("category_ids").list.first().cast(pl.Utf8)
        .replace_strict(by_id, default=None).alias("_service"),
    )
    if tidy.filter(pl.col("_service").is_null()).height:
        raise PNSBError("response carries a service category that was not requested")
    wide = tidy.pivot(
        on="_service", index=["munic_code", "year"], values="value_numeric",
        aggregate_function="max",
    )
    service_cols = [c for c in wide.columns if c in SERVICES]
    return wide.with_columns(
        [(pl.col(c) == 1.0).alias(f"pnsb_has_{c}") for c in service_cols]
    ).drop(service_cols).sort(["munic_code", "year"])


def compare_with_notifications(
    flags: pl.DataFrame,
    notifications: pl.DataFrame,
    *,
    disease: str = "leptospirosis",
    count_column: str = "cases",
) -> dict[str, float | int]:
    """Cross the PNSB statement against notified cases, in both directions.

    ``notifications`` is one row per municipality with a case count for whatever
    window the caller chose. Returns raw counts alongside percentages so that a
    reader can reconstruct any of them, and reports the reverse discordance --
    municipalities denying occurrence that did notify -- because presenting only
    the forward direction would imply PNSB is the reference standard, which it
    is not.
    """
    flag_col = f"pnsb_{disease}_2008"
    if flag_col not in flags.columns:
        raise PNSBError(f"flags frame lacks {flag_col!r}")
    if count_column not in notifications.columns:
        raise PNSBError(f"notifications frame lacks {count_column!r}")
    joined = (
        flags.filter(pl.col(flag_col).is_not_null())
        .join(
            notifications.select(
                pl.col("munic_code").cast(pl.Utf8).str.zfill(7),
                pl.col(count_column).cast(pl.Float64).alias("_n"),
            ),
            on="munic_code", how="left",
        )
        .with_columns(pl.col("_n").fill_null(0.0))
    )
    declared = joined.filter(pl.col(flag_col))
    denied = joined.filter(~pl.col(flag_col))
    silent = declared.filter(pl.col("_n") == 0).height
    notified_anyway = denied.filter(pl.col("_n") > 0).height
    return {
        "declared": declared.height,
        "declared_with_no_notification": silent,
        "declared_silent_pct": (
            100.0 * silent / declared.height if declared.height else float("nan")
        ),
        "denied": denied.height,
        "denied_with_notification": notified_anyway,
        "denied_notified_pct": (
            100.0 * notified_anyway / denied.height if denied.height else float("nan")
        ),
    }
