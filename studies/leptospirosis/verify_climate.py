"""Verification of the climate exposure layer against real data.

Run with the pegasus python. Checks, in order:

1. every product's declared coverage matches what the artefact actually holds;
2. a 20-municipality, five-macroregion monthly series is complete and
   non-negative;
3. the seasonal phase is right in each macroregion (Amazonian and Northeastern
   wet seasons are austral summer/autumn, the South rains year-round);
4. May 2024 in Porto Alegre and the Rio Grande do Sul valleys is a large
   positive anomaly against the municipal 1991-2020 climatology.

(4) is the load-bearing test. Everything upstream can be wrong in ways that
still produce a plausible-looking seasonal cycle; only a known event with a
known magnitude in a known place distinguishes a working exposure layer from a
well-formatted one.
"""

from __future__ import annotations

import argparse
from datetime import date

import polars as pl

from brepi.geo import lattice
from brepi.sources.climate import brdwgd, indices

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_formatting("ASCII_FULL")

#: Twenty municipalities, four per macroregion, capitals and large cities so
#: the expected seasonality is documented in any climatology textbook.
PANEL: dict[str, list[tuple[str, str]]] = {
    "North": [
        ("1302603", "Manaus/AM"),
        ("1200401", "Rio Branco/AC"),
        ("1501402", "Belem/PA"),
        ("1100205", "Porto Velho/RO"),
    ],
    "Northeast": [
        ("2304400", "Fortaleza/CE"),
        ("2611606", "Recife/PE"),
        ("2927408", "Salvador/BA"),
        ("2111300", "Sao Luis/MA"),
    ],
    "Southeast": [
        ("3550308", "Sao Paulo/SP"),
        ("3304557", "Rio de Janeiro/RJ"),
        ("3106200", "Belo Horizonte/MG"),
        ("3205309", "Vitoria/ES"),
    ],
    "South": [
        ("4314902", "Porto Alegre/RS"),
        ("4106902", "Curitiba/PR"),
        ("4205407", "Florianopolis/SC"),
        ("4209102", "Joinville/SC"),
    ],
    "Centre-West": [
        ("5300108", "Brasilia/DF"),
        ("5208707", "Goiania/GO"),
        ("5103403", "Cuiaba/MT"),
        ("5002704", "Campo Grande/MS"),
    ],
}

HEADLINE = ["3550308", "1200401", "4314902", "2304400", "1302603"]

#: Municipalities in the Taquari and Cai valleys and greater Porto Alegre that
#: were inundated in the RS catastrophe of late April / May 2024. Names are
#: resolved against the live lattice so a mistyped code fails loudly.
RS_FLOOD_NAMES = [
    "Porto Alegre",
    "Canoas",
    "Sao Leopoldo",
    "Novo Hamburgo",
    "Lajeado",
    "Estrela",
    "Mucum",
    "Roca Sales",
    "Encantado",
    "Sao Sebastiao do Cai",
    "Eldorado do Sul",
    "Guaiba",
    "Santa Maria",
    "Venancio Aires",
    "Cruzeiro do Sul",
]

#: A control set: Northeastern municipalities that were NOT flooded in May
#: 2024. If they also score +3 sigma, the anomaly is an artefact of the
#: climatology, not of the weather.
CONTROL = ["2304400", "2611606", "2927408", "1302603"]

CODES = [code for group in PANEL.values() for code, _ in group]
NAMES = {code: name for group in PANEL.values() for code, name in group}
REGION = {code: region for region, group in PANEL.items() for code, _ in group}


def _strip(text: str) -> str:
    import unicodedata

    return "".join(
        c
        for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    ).lower()


def resolve_rs_codes() -> dict[str, str]:
    """Look the RS flood municipalities up in the live lattice by name."""
    muni = lattice.load_municipalities()
    rs = muni.filter(pl.col("uf_abbr") == "RS")
    wanted = {_strip(n): n for n in RS_FLOOD_NAMES}
    hits = rs.with_columns(
        pl.col("name")
        .map_elements(_strip, return_dtype=pl.Utf8)
        .alias("_key")
    ).filter(pl.col("_key").is_in(list(wanted)))
    found = dict(zip(hits["code7"].to_list(), hits["name"].to_list()))
    missing = set(wanted) - set(hits["_key"].to_list())
    if missing:
        print(f"  [warn] RS names not resolved in the lattice: {sorted(missing)}")
    return found


# --------------------------------------------------------------------------


def check_coverage(products: list[str]) -> None:
    print("\n=== 1. declared vs actual coverage ===")
    print(brdwgd.available_products())
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET enable_progress_bar=false")
    for name in products:
        prod = brdwgd.PRODUCTS[name]
        try:
            src = brdwgd._sources_for(  # noqa: SLF001 - verification reaches in on purpose
                prod, "pr", prod.start, prod.end, download=False
            )
        except brdwgd.ClimateSourceError as exc:
            print(f"  {name}: not cached ({exc})")
            continue
        lo = hi = None
        for one in src:
            row = con.execute(
                f"SELECT min(date), max(date) FROM read_parquet('{one}') WHERE code_muni=4314902"
            ).fetchone()
            lo = row[0] if lo is None else min(lo, row[0])
            hi = row[1] if hi is None else max(hi, row[1])
        flag = "OK" if (lo, hi) == (prod.start, prod.end) else "MISMATCH"
        print(f"  {name}: declared {prod.start}..{prod.end}  actual {lo}..{hi}  [{flag}]")


def monthly_panel(
    codes: list[str], years: range, product: str, download: bool
) -> pl.DataFrame:
    daily = brdwgd.daily_municipal(
        codes,
        ["pr", "tmax", "tmin"],
        start=date(years[0], 1, 1),
        end=date(years[-1], 12, 31),
        product=product,
        stats=("mean",),
        download=download,
    )
    return indices.monthly_indices_from_daily(daily, variables=["pr", "tmax", "tmin"])


def check_panel(monthly: pl.DataFrame, codes: list[str], years: range) -> None:
    print("\n=== 2. panel integrity ===")
    expected = len(codes) * len(years) * 12
    print(f"  rows {monthly.height} / expected {expected}")
    assert monthly.height == expected, "monthly series is not complete"
    assert monthly["munic_code"].n_unique() == len(codes)
    assert (monthly["pr_total"] >= 0).all(), "negative precipitation"
    assert monthly["pr_total"].null_count() == 0, "null precipitation"
    assert (monthly["n_days"] == monthly["n_days_expected"]).all()
    assert (monthly["tmax_mean"] >= monthly["tmin_mean"]).all(), "tmax below tmin"
    tmax_lo, tmax_hi = monthly["tmax_mean"].min(), monthly["tmax_mean"].max()
    print(f"  tmax_mean range {tmax_lo:.1f} .. {tmax_hi:.1f} degC")
    assert -5 < tmax_lo and tmax_hi < 45, "temperature out of physical range (unit error?)"
    print(f"  pr_total range {monthly['pr_total'].min():.1f} .. {monthly['pr_total'].max():.1f} mm")
    print("  OK: complete, non-negative, physically plausible")


def check_seasonality(monthly: pl.DataFrame) -> None:
    """The seasonal phase must differ by macroregion in the documented way."""
    print("\n=== 3. seasonal phase by macroregion ===")
    tagged = monthly.with_columns(
        pl.col("munic_code").replace_strict(REGION, default=None).alias("region"),
        pl.col("munic_code").replace_strict(NAMES, default=None).alias("name"),
        pl.col("period").dt.month().alias("month"),
    )
    per_mun = (
        tagged.group_by(["region", "name", "month"])
        .agg(pl.col("pr_total").mean().alias("mm"))
        .sort(["region", "name", "month"])
    )
    rows = []
    for (region, name), block in per_mun.group_by(["region", "name"], maintain_order=True):
        block = block.sort("month")
        mm = block["mm"].to_list()
        wettest = int(block["month"][int(pl.Series(mm).arg_max())])
        driest = int(block["month"][int(pl.Series(mm).arg_min())])
        rows.append(
            {
                "region": region,
                "municipality": name,
                "wettest_month": wettest,
                "driest_month": driest,
                "wettest_mm": round(max(mm), 1),
                "driest_mm": round(min(mm), 1),
                "seasonality_ratio": round(max(mm) / max(min(mm), 0.1), 1),
            }
        )
    table = pl.DataFrame(rows).sort(["region", "municipality"])
    print(table)

    def wettest(code: str) -> int:
        return int(table.filter(pl.col("municipality") == NAMES[code])["wettest_month"][0])

    def ratio(code: str) -> float:
        return float(table.filter(pl.col("municipality") == NAMES[code])["seasonality_ratio"][0])

    problems: list[str] = []
    # Amazonian wet season is austral summer: peak Dec-Apr, dry trough Jul-Sep.
    for code in ("1302603", "1200401", "1100205"):
        if wettest(code) not in (12, 1, 2, 3, 4):
            problems.append(f"{NAMES[code]} wettest month {wettest(code)}, expected Dec-Apr")
    # Northern-coast Northeast (Fortaleza, Sao Luis, Belem) peaks Feb-May with
    # the ITCZ, and has a hard dry season -- a large seasonality ratio.
    for code in ("2304400", "2111300", "1501402"):
        if wettest(code) not in (1, 2, 3, 4, 5):
            problems.append(f"{NAMES[code]} wettest month {wettest(code)}, expected Feb-May")
    if ratio("2304400") < 5:
        problems.append(f"Fortaleza seasonality ratio {ratio('2304400')}, expected a hard dry season")
    # Eastern-coast Northeast (Recife, Salvador) is the counter-example: it
    # peaks in austral WINTER off the South Atlantic. If this comes out as a
    # summer peak, a whole-country seasonality has been imposed on the data.
    for code in ("2611606", "2927408"):
        if wettest(code) not in (4, 5, 6, 7, 8):
            problems.append(f"{NAMES[code]} wettest month {wettest(code)}, expected Apr-Aug")
    # Centre-West / Southeast: monsoonal, peak Nov-Mar, deep May-Aug trough.
    for code in ("5103403", "5208707", "5300108", "3550308", "3106200"):
        if wettest(code) not in (11, 12, 1, 2, 3):
            problems.append(f"{NAMES[code]} wettest month {wettest(code)}, expected Nov-Mar")
    if ratio("5103403") < 5:
        problems.append(f"Cuiaba seasonality ratio {ratio('5103403')}, expected a hard dry season")
    # South: rain all year. No dry season, so the ratio must be small -- this
    # is the discriminating test, not the peak month.
    for code in ("4314902", "4106902", "4205407", "4209102"):
        if ratio(code) > 3.5:
            problems.append(f"{NAMES[code]} seasonality ratio {ratio(code)}, expected year-round rain")

    if problems:
        raise AssertionError("seasonal phase wrong:\n  " + "\n  ".join(problems))
    print("  OK: Amazonian and NE-coast summer/autumn peaks, eastern-NE winter")
    print("      peak, monsoonal Centre-West, aseasonal South")


def print_headline_table(monthly: pl.DataFrame, year: int) -> None:
    print(f"\n=== monthly precipitation {year}, mm (areal mean daily depth summed) ===")
    sub = (
        monthly.filter(pl.col("period").dt.year() == year)
        .filter(pl.col("munic_code").is_in(HEADLINE))
        .with_columns(
            pl.col("munic_code").replace_strict(NAMES, default=None).alias("name"),
            pl.col("period").dt.month().alias("m"),
            pl.col("pr_total").round(1),
        )
    )
    print(sub.pivot(on="name", index="m", values="pr_total").sort("m"))


def rs_anomaly(
    climatology_product: str,
    event_product: str,
    *,
    download: bool,
    base: tuple[int, int] = indices.CLIMATOLOGY_BASE,
) -> pl.DataFrame:
    """May-2024 anomaly in the RS flood municipalities against 1991-2020."""
    print(f"\n=== 4. Rio Grande do Sul, May 2024 ===")
    rs = resolve_rs_codes()
    codes = sorted(set(rs) | set(CONTROL))
    print(f"  {len(rs)} RS municipalities + {len(CONTROL)} non-RS controls")
    print(f"  climatology: {climatology_product} {base[0]}-{base[1]}")
    print(f"  event month: {event_product} 2024-05")

    clim_prod = brdwgd.PRODUCTS[climatology_product]
    clim_end = min(date(base[1], 12, 31), clim_prod.end)
    clim_daily = brdwgd.daily_municipal(
        codes,
        ["pr"],
        start=date(base[0], 1, 1),
        end=clim_end,
        product=climatology_product,
        stats=("mean",),
        download=download,
    )
    clim = indices.monthly_indices_from_daily(clim_daily, variables=["pr"])
    n_base = clim.filter(pl.col("period").dt.month() == 5)["period"].dt.year().n_unique()
    print(f"  base years available for May: {n_base}")

    event_daily = brdwgd.daily_municipal(
        codes,
        ["pr"],
        start=date(2024, 5, 1),
        end=date(2024, 5, 31),
        product=event_product,
        stats=("mean",),
        download=download,
    )
    event = indices.monthly_indices_from_daily(event_daily, variables=["pr"])

    if climatology_product != event_product:
        print(
            f"  [note] CROSS-PRODUCT anomaly: {climatology_product} climatology vs "
            f"{event_product} event. Interpret the sigma as indicative."
        )

    combined = pl.concat(
        [
            clim.select("munic_code", "period", "pr_total", "rx1day", "rx5day", "r50", "cwd"),
            event.select("munic_code", "period", "pr_total", "rx1day", "rx5day", "r50", "cwd"),
        ]
    )
    with_z = indices.attach_anomalies(combined, base=base, min_base_years=20)
    out = (
        with_z.filter(pl.col("period") == date(2024, 5, 1))
        .with_columns(
            pl.col("munic_code")
            .replace_strict({**rs, **{c: NAMES.get(c, c) for c in CONTROL}}, default=None)
            .alias("municipality"),
            pl.col("munic_code").str.slice(0, 2).alias("uf"),
        )
        .select(
            "municipality",
            "munic_code",
            "uf",
            pl.col("pr_total").round(1).alias("may2024_mm"),
            pl.col("pr_total_clim_mean").round(1).alias("clim_mm"),
            pl.col("pr_total_clim_sd").round(1).alias("clim_sd"),
            pl.col("pr_total_anom_z").round(2).alias("anom_sigma"),
            pl.col("rx1day").round(1),
            pl.col("rx5day").round(1),
            "r50",
            "cwd",
        )
        .sort("anom_sigma", descending=True)
    )
    print(out)

    rs_rows = out.filter(pl.col("uf") == "43")
    ctrl_rows = out.filter(pl.col("uf") != "43")
    rs_median = float(rs_rows["anom_sigma"].median())
    ctrl_max = float(ctrl_rows["anom_sigma"].max()) if ctrl_rows.height else float("nan")
    poa = out.filter(pl.col("munic_code") == "4314902").row(0, named=True)
    print(
        f"\n  Porto Alegre 4314902: {poa['may2024_mm']} mm vs climatology "
        f"{poa['clim_mm']} +/- {poa['clim_sd']} mm  ->  {poa['anom_sigma']} sigma"
    )
    print(f"  RS median anomaly {rs_median:+.2f} sigma; non-RS control max {ctrl_max:+.2f} sigma")
    assert poa["anom_sigma"] > 2.0, "Porto Alegre May 2024 is not anomalous -- exposure layer suspect"
    assert rs_median > 1.5, "RS flood municipalities are not collectively anomalous"
    assert ctrl_max < rs_median, "controls are as anomalous as RS -- the anomaly is an artefact"
    print("  OK: the RS catastrophe is present, localised, and large")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--product",
        default="brdwgd_cdn",
        help="product for the seasonality panel (brdwgd_cdn needs no download)",
    )
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--headline-year", type=int, default=2019)
    parser.add_argument("--climatology-product", default="brdwgd_cdn")
    parser.add_argument("--event-product", default="era5land")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--only", default="", help="comma-separated: panel,rs")
    args = parser.parse_args()

    only = set(x for x in args.only.split(",") if x)
    download = not args.no_download
    years = range(args.start_year, args.end_year + 1)

    if not args.skip_coverage:
        check_coverage(["brdwgd_cdn", "brdwgd", "era5land"])

    if not only or "panel" in only:
        monthly = monthly_panel(CODES, years, args.product, download)
        check_panel(monthly, CODES, years)
        check_seasonality(monthly)
        print_headline_table(monthly, args.headline_year)

    if not only or "rs" in only:
        rs_anomaly(args.climatology_product, args.event_product, download=download)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
