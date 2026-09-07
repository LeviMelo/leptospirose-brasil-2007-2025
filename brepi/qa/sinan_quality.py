"""Surveillance-quality profiling for SINAN notification data.

Generic across agravos. Produces the completeness, delay and classification
tables that belong in a RECORD-compliant methods section, and that this study
additionally uses as *model covariates* rather than as caveats: detection
capacity is an estimand here, not a limitation to be noted in a discussion.

The measures implemented are the ones that turned out to carry signal on the
2007-2025 leptospirosis series:

``geography_completeness``
    Missingness of the probable-infection municipality among confirmed cases.
    For leptospirosis this runs 10-15% in 2007-2015 and rises to 20-22% after
    2020 — a COVID-era degradation that is non-random in time and therefore
    cannot be handled by complete-case analysis without bias.

``exposure_completeness``
    Share of confirmed cases with an informative (not blank, not "ignorado")
    answer in the risk-situation block. On leptospirosis this is 85-91%
    through the SINAN NET era, far better than the field's reputation
    suggests, which makes an internal individual-level validation of the
    ecological exposure model feasible.

``criterion_mix``
    Clinical-laboratory versus clinical-epidemiological confirmation. The
    clinical-epidemiological share rises sharply in flood years (23.7% in
    2024 against a ~12-15% baseline), which is a measurable ascertainment
    channel rather than noise.

``delay_profile``
    Onset to notification, notification to digitisation, notification to
    closure. ``DT_DIGITA`` and the ``DT_TRANS*`` block are almost never used
    in the literature and give a direct handle on reporting-chain latency,
    which is what justifies right-truncating the preliminary years.

``undocumented_codes``
    Values present in the data but absent from the official dictionary. On
    leptospirosis, ``CLASSI_FIN`` carries an undocumented ``8`` in every year
    since 2007 (11,732 records). Silently dropping or recoding these is a
    decision that must be visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import polars as pl

#: Fields whose disseminated content should be checked against the dictionary.
DEFAULT_CODED_FIELDS: tuple[str, ...] = (
    "CLASSI_FIN", "CRITERIO", "EVOLUCAO", "TPAUTOCTO", "CS_SEXO", "CS_RACA",
    "CS_GESTANT", "ATE_HOSP", "DOENCA_TRA", "CON_AMBIEN",
)

#: Sentinels that mean "no information", as distinct from a genuine category.
UNINFORMATIVE = frozenset({"", "9", "99", "999", "ignorado", "IGNORADO"})


def _clean(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars().fill_null("")


def _informative(col: str) -> pl.Expr:
    return ~_clean(col).is_in(list(UNINFORMATIVE))


@dataclass(frozen=True)
class QualityProfile:
    """Every table needed for the data-quality section of the paper."""

    by_year: pl.DataFrame
    delays: pl.DataFrame
    undocumented: pl.DataFrame
    field_completeness: pl.DataFrame

    def write(self, directory) -> None:
        from pathlib import Path

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.by_year.write_csv(d / "quality_by_year.csv")
        self.delays.write_csv(d / "quality_delays.csv")
        self.undocumented.write_csv(d / "quality_undocumented_codes.csv")
        self.field_completeness.write_csv(d / "quality_field_completeness.csv")


def _year(df: pl.DataFrame, date_col: str) -> pl.Expr:
    return pl.col(date_col).str.slice(0, 4).cast(pl.Int32, strict=False)


def profile(
    df: pl.DataFrame,
    *,
    year_col: str = "NU_ANO",
    confirmed_expr: pl.Expr | None = None,
    geography_fields: Sequence[str] = ("COMUNINF", "ID_MN_RESI", "ID_MUNICIP"),
    exposure_prefix: str = "ANT_CB_",
    coded_fields: Sequence[str] = DEFAULT_CODED_FIELDS,
    dictionary: Mapping[str, Iterable[str]] | None = None,
    date_pairs: Sequence[tuple[str, str, str]] = (
        ("DT_NOTIFIC", "DT_SIN_PRI", "onset_to_notification"),
        ("DT_DIGITA", "DT_NOTIFIC", "notification_to_digitisation"),
        ("DT_ENCERRA", "DT_NOTIFIC", "notification_to_closure"),
    ),
) -> QualityProfile:
    """Build the full quality profile of a decoded SINAN frame.

    ``dictionary`` maps a field to its officially documented category codes;
    anything observed outside that set is reported rather than dropped.
    """
    if year_col not in df.columns:
        raise ValueError(f"frame lacks {year_col!r}")
    d = df.with_columns(pl.col(year_col).cast(pl.Int32, strict=False).alias("_year"))
    conf = confirmed_expr if confirmed_expr is not None else (_clean("CLASSI_FIN") == "1")
    d = d.with_columns(conf.alias("_confirmed"))

    # -- per-year headline table ------------------------------------------
    aggs: list[pl.Expr] = [
        pl.len().alias("notified"),
        pl.col("_confirmed").sum().alias("confirmed"),
    ]
    if "CLASSI_FIN" in d.columns:
        aggs += [
            (_clean("CLASSI_FIN") == "2").sum().alias("discarded"),
            (_clean("CLASSI_FIN") == "").sum().alias("classification_open"),
        ]
    if "CRITERIO" in d.columns:
        aggs += [
            ((_clean("CRITERIO") == "1") & pl.col("_confirmed")).sum().alias("criterion_lab"),
            ((_clean("CRITERIO") == "2") & pl.col("_confirmed")).sum().alias("criterion_epi"),
        ]
    if "EVOLUCAO" in d.columns:
        aggs += [
            ((_clean("EVOLUCAO") == "2") & pl.col("_confirmed")).sum().alias("deaths_disease"),
            ((_clean("EVOLUCAO") == "3") & pl.col("_confirmed")).sum().alias("deaths_other"),
            ((_clean("EVOLUCAO").is_in(["1", "2", "3"])) & pl.col("_confirmed"))
            .sum()
            .alias("outcome_known"),
        ]
    if "ATE_HOSP" in d.columns:
        aggs.append(
            ((_clean("ATE_HOSP") == "1") & pl.col("_confirmed")).sum().alias("hospitalised")
        )

    for g in geography_fields:
        if g in d.columns:
            aggs.append(
                (pl.col("_confirmed") & ~_clean(g).str.contains(r"^\d{6}$"))
                .sum()
                .alias(f"{g}_missing_confirmed")
            )

    exposure_cols = [c for c in d.columns if c.startswith(exposure_prefix)]
    for c in exposure_cols:
        aggs.append(
            (pl.col("_confirmed") & _informative(c)).sum().alias(f"_inf_{c}")
        )

    by_year = d.group_by("_year").agg(aggs).sort("_year").rename({"_year": "year"})

    if exposure_cols:
        inf_cols = [f"_inf_{c}" for c in exposure_cols]
        by_year = by_year.with_columns(
            (pl.sum_horizontal(inf_cols) / (pl.col("confirmed") * len(exposure_cols)))
            .alias("exposure_block_informative_share")
        ).drop(inf_cols)

    for g in geography_fields:
        col = f"{g}_missing_confirmed"
        if col in by_year.columns:
            by_year = by_year.with_columns(
                (pl.col(col) / pl.col("confirmed")).alias(f"{g}_missing_share")
            )
    if "criterion_epi" in by_year.columns:
        by_year = by_year.with_columns(
            (
                pl.col("criterion_epi")
                / (pl.col("criterion_lab") + pl.col("criterion_epi"))
            ).alias("criterion_epi_share")
        )
    if "deaths_disease" in by_year.columns:
        by_year = by_year.with_columns(
            (pl.col("deaths_disease") / pl.col("outcome_known")).alias("case_fatality")
        )
    by_year = by_year.with_columns(
        (pl.col("confirmed") / pl.col("notified")).alias("confirmation_ratio")
    )

    # -- delays ------------------------------------------------------------
    delay_rows = []
    for later, earlier, label in date_pairs:
        if later not in d.columns or earlier not in d.columns:
            continue
        diff = (
            pl.col(later).str.to_date(strict=False)
            - pl.col(earlier).str.to_date(strict=False)
        ).dt.total_days()
        sub = (
            d.filter(pl.col("_confirmed"))
            .select(diff.alias("days"))
            .filter(pl.col("days").is_between(-30, 1500))
        )
        if sub.height == 0:
            continue
        delay_rows.append(
            {
                "interval": label,
                "n": sub.height,
                "median": float(sub["days"].median()),
                "p75": float(sub["days"].quantile(0.75)),
                "p95": float(sub["days"].quantile(0.95)),
                "share_over_30d": float((sub["days"] > 30).mean()),
            }
        )
    delays = pl.DataFrame(delay_rows) if delay_rows else pl.DataFrame(
        schema={"interval": pl.Utf8, "n": pl.Int64}
    )

    # -- undocumented codes -------------------------------------------------
    undoc_rows = []
    if dictionary:
        for field, allowed in dictionary.items():
            if field not in d.columns:
                continue
            allowed_set = set(allowed) | {""}
            obs = (
                d.select(_clean(field).alias("v"))
                .group_by("v")
                .agg(pl.len().alias("n"))
                .filter(~pl.col("v").is_in(list(allowed_set)))
                .sort("n", descending=True)
            )
            for r in obs.iter_rows(named=True):
                undoc_rows.append({"field": field, "value": r["v"], "n": r["n"]})
    undocumented = pl.DataFrame(undoc_rows) if undoc_rows else pl.DataFrame(
        schema={"field": pl.Utf8, "value": pl.Utf8, "n": pl.Int64}
    )

    # -- per-field completeness --------------------------------------------
    check = [c for c in list(coded_fields) + exposure_cols + list(geography_fields) if c in d.columns]
    comp = (
        d.filter(pl.col("_confirmed"))
        .select([_informative(c).mean().alias(c) for c in check])
        .transpose(include_header=True, header_name="field", column_names=["informative_share"])
        .sort("informative_share")
    )

    return QualityProfile(by_year, delays, undocumented, comp)


def reconcile(
    observed: pl.DataFrame,
    official: Mapping[int, int],
    *,
    year_col: str = "year",
    count_col: str = "confirmed",
) -> pl.DataFrame:
    """Compare an extraction against a published national series.

    This is the gate that decides whether an extraction is trustworthy at all.
    Report absolute and relative discrepancy per year; a systematic sign
    usually means the wrong date field (year of notification versus year of
    first symptoms), while scattered small differences usually mean the
    published snapshot predates the current file revision.
    """
    off = pl.DataFrame(
        {year_col: list(official.keys()), "official": list(official.values())}
    )
    out = off.join(observed.select(year_col, count_col), on=year_col, how="left")
    return out.with_columns(
        (pl.col(count_col) - pl.col("official")).alias("difference"),
        ((pl.col(count_col) - pl.col("official")) / pl.col("official")).alias("relative"),
    ).sort(year_col)


__all__ = ["QualityProfile", "profile", "reconcile", "DEFAULT_CODED_FIELDS", "UNINFORMATIVE"]
