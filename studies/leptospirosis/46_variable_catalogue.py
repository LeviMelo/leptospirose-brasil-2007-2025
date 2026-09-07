"""Complete variable catalogue for the analytic surface.

Written because the study had been running on roughly eight fields of a
309-column line-level extract, and nobody had ever enumerated what the other
three hundred contain. A source document that describes the *files* but not the
*variables* lets a study quietly leave half its evidence untouched.

For every column this records: its decode state distribution where the codebook
supplies one, completeness among **confirmed cases** (the analytic population,
not all notifications — the two differ enormously and conflating them has
already produced one wrong claim in this project), cardinality, and the modal
values. Variables are assigned to epidemiological blocks so that a reader can
see which blocks are exploited and which are dark.

The four decode states are not interchangeable and are never collapsed:
``valid`` (a code the dictionary knows), ``unknown`` (an explicit "ignorado"
sentinel — the respondent was asked and did not know), ``missing`` (never
filled), ``invalid`` (a code outside the dictionary).

Outputs to ``data/results/variable_catalogue/``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brepi.config import PATHS

OUT = PATHS.results / "variable_catalogue"
LINE = PATHS.interim / "lept_line_level.parquet"

#: Epidemiological blocks, applied in order; first match wins. Assignment is by
#: the SINAN field-naming convention, which is stable across the NET era.
BLOCKS: list[tuple[str, str, str]] = [
    ("identification", r"^(NU_NOTIFIC|ID_AGRAVO|TP_NOT|NDUPLIC|DT_DIGITA|NU_LOTE|CS_FLXRET|FLXRECEBI|MIGRADO)", "record identity and administrative flow"),
    ("date", r"(^DT_|_date$|^dt_|date$|epiweek|_year$|_month$)", "event and processing dates"),
    ("geography", r"(^ID_MUNICIP|^ID_MN_RESI|^ID_REGIONA|^ID_RG_RESI|^ID_UNIDADE|^SG_UF|^COMUNINF|^COUFINF|^COPAISINF|^ID_PAIS|^CO_MUN_R|^CO_UF_R|munic|uf_|region|health_region|state$|_code7$)", "where the case was notified, lived and was probably infected"),
    ("demography", r"^(CS_SEXO|CS_RACA|CS_ESCOL|CS_GESTANT|NU_IDADE|ANO_NASC|age|sex|race|schooling|education|pregnan)", "person characteristics"),
    ("occupation", r"^(ID_OCUPA|DOENCA_TRA|occupation|cbo)", "occupation and work-relatedness"),
    ("clinical", r"^(CLI_|clinical|symptom)", "presenting signs and symptoms"),
    ("exposure", r"^(ANT_|CON_AMBIEN|TPAUTOCTO|antecedent|exposure|contact)", "antecedent exposures and environment"),
    ("laboratory", r"^(LAB_|MICRO|DTMICRO|RES_|DTISOLA|DT_PCR|DTIMUNO|serovar|serogroup|reservoir|criterio|CRITERIO)", "laboratory tests, serology and confirmation criterion"),
    ("care", r"^(ATE_|hospital|admission)", "care pathway and hospitalisation"),
    ("outcome", r"^(EVOLUCAO|DT_OBITO|evolucao|outcome|death|died)", "outcome of the episode"),
    ("classification", r"^(CLASSI_FIN|classi_fin|CON_CLASSI)", "final case classification"),
    ("closure", r"^(DT_ENCERRA|DT_INVEST|DT_TRANS)", "investigation closure and transmission to higher tiers"),
]


def block_of(name: str) -> tuple[str, str]:
    for block, pattern, desc in BLOCKS:
        if re.search(pattern, name):
            return block, desc
    return "other", "unclassified"


def catalogue(df: pl.DataFrame, states: set[str]) -> pl.DataFrame:
    rows = []
    n = df.height
    for col in df.columns:
        if col.endswith("_state"):
            continue
        s = df[col]
        block, _ = block_of(col)
        partner = f"{col}_state"
        st = {}
        if partner in states:
            vc = df[partner].value_counts()
            st = {r[partner]: r["count"] for r in vc.iter_rows(named=True)}
        nonnull = int(s.is_not_null().sum())
        try:
            distinct = int(s.n_unique())
        except Exception:
            distinct = -1
        top = []
        if distinct != -1 and distinct <= 5000 and s.dtype in (pl.Utf8, pl.Boolean, pl.Int32, pl.Int64):
            vc = s.value_counts(sort=True).head(4)
            top = [f"{r[col]}={r['count']}" for r in vc.iter_rows(named=True)]
        rows.append({
            "variable": col,
            "block": block,
            "dtype": str(s.dtype),
            "n_nonnull": nonnull,
            "pct_nonnull": round(100 * nonnull / n, 2) if n else None,
            "n_distinct": distinct,
            "state_valid": st.get("valid"),
            "state_unknown": st.get("unknown"),
            "state_missing": st.get("missing"),
            "state_invalid": st.get("invalid"),
            "pct_valid": round(100 * st["valid"] / n, 2) if st.get("valid") is not None and n else None,
            "top_values": "; ".join(top),
        })
    schema = {
        "variable": pl.Utf8, "block": pl.Utf8, "dtype": pl.Utf8,
        "n_nonnull": pl.Int64, "pct_nonnull": pl.Float64, "n_distinct": pl.Int64,
        "state_valid": pl.Int64, "state_unknown": pl.Int64,
        "state_missing": pl.Int64, "state_invalid": pl.Int64,
        "pct_valid": pl.Float64, "top_values": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema).sort(["block", "variable"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_parquet(LINE)
    names = lf.collect_schema().names()
    states = {c for c in names if c.endswith("_state")}

    confirmed = lf.filter(pl.col("classi_fin") == "confirmado").collect()
    everything = lf.collect()
    print(f"line level: {everything.height} notifications, "
          f"{confirmed.height} confirmed, {len(names)} columns "
          f"({len(states)} decode-state partners)")

    cat_conf = catalogue(confirmed, states)
    cat_conf.write_csv(OUT / "variable_catalogue_confirmed.csv")
    cat_all = catalogue(everything, states)
    cat_all.write_csv(OUT / "variable_catalogue_all_notifications.csv")

    # Block summary: how much of each epidemiological block is actually usable
    # on the analytic population. A block that is 20% complete is not evidence.
    summ = (
        cat_conf.group_by("block").agg(
            pl.len().alias("variables"),
            pl.col("pct_valid").drop_nulls().median().alias("median_pct_valid"),
            pl.col("pct_nonnull").median().alias("median_pct_nonnull"),
        ).sort("variables", descending=True)
    )
    summ.write_csv(OUT / "block_summary.csv")
    print("\n=== blocks, on CONFIRMED cases ===")
    print(f"{'block':<16}{'vars':>6}{'median % valid':>16}{'median % non-null':>19}")
    for r in summ.iter_rows(named=True):
        mv = f"{r['median_pct_valid']:.1f}" if r["median_pct_valid"] is not None else "-"
        print(f"{r['block']:<16}{r['variables']:>6}{mv:>16}{r['median_pct_nonnull']:>19.1f}")

    # The headline the study needs: which coded variables are well recorded on
    # confirmed cases and therefore usable as evidence.
    usable = cat_conf.filter(
        pl.col("pct_valid").is_not_null() & (pl.col("pct_valid") >= 70)
    ).sort("pct_valid", descending=True)
    dark = cat_conf.filter(
        pl.col("pct_valid").is_not_null() & (pl.col("pct_valid") < 40)
    ).sort("pct_valid")
    usable.write_csv(OUT / "usable_coded_variables.csv")
    dark.write_csv(OUT / "poorly_recorded_variables.csv")
    print(f"\ncoded variables >=70% valid on confirmed cases: {usable.height}")
    print(f"coded variables <40% valid on confirmed cases:  {dark.height}")

    report = {
        "notifications": everything.height,
        "confirmed": confirmed.height,
        "columns": len(names),
        "decode_state_partners": len(states),
        "blocks": summ.to_dicts(),
        "usable_coded_variables": usable.height,
        "poorly_recorded_variables": dark.height,
    }
    (OUT / "variable_catalogue_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
