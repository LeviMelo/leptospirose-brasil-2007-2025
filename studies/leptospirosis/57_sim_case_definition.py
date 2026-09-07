"""Should A27 "mention-only" deaths be counted, and what changes if they are?

Every SIM comparison in this study counts a death as leptospirosis when
``CAUSABAS`` -- the *underlying* cause selected by the coding system -- begins
A27. That yields 5,705 deaths over 2007-2024. The raw death certificates carry
A27 somewhere on 6,306 records. The 601-record difference is 10.5% of the
underlying-cause count, and it has never been adjudicated: the study inherited
"underlying cause only" as a default, not as a decision.

This script makes the decision explicitly. Four candidate case definitions are
carried side by side throughout:

``underlying``   ``CAUSABAS`` begins A27.  The current study definition.
``part1``        underlying, or A27 on a Part I line (``LINHAA``..``LINHAD``) --
                 the physician placed leptospirosis *in the causal sequence*
                 that led to death, and the ICD selection rules chose something
                 else as the underlying cause.
``certificate``  ``part1``, or A27 in Part II (``LINHAII``) -- a contributing
                 condition the physician judged relevant but outside the chain.
``any_mention``  ``certificate``, or A27 in ``CAUSABAS_O`` only -- the *original*
                 underlying cause, before the coding/investigation machinery
                 moved it. These records mention leptospirosis nowhere on the
                 certificate as it now stands.

The four are nested, so each supplement is attributable to one documentary act.

THE FALSE-POSITIVE GATE, first. ``LINHAA``..``LINHAII`` concatenate several
four-character CID-10 codes into one string, and ``brepi.sources.datasus.sim``
matches them by *substring* precisely because an anchored match would only ever
see the first code. A substring rule can manufacture hits across a token
boundary, and a manufactured hit here would inflate the very supplement being
adjudicated. The fields turn out to be ``*``-delimited; the script splits on the
delimiter and asserts that every A27-containing token *begins* A27 before any
count is formed. It aborts if one does not.

THE TREND THREAT, named once and then measured. ``LINHAII`` is filled on only
26-47% of A27 certificates and its fill rate *rises across the series*. Any
definition that draws on Part II therefore adds proportionally more deaths in
later years for documentary reasons. A mortality trend computed under such a
definition is contaminated by the completeness trend of the field that defines
it. The script quantifies the completeness trend and reports the mortality
AAPCs against it rather than beside it.

Outputs to ``data/results/sim_case_definition/``.

Run:
    PYTHONPATH=. python studies/leptospirosis/57_sim_case_definition.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.analysis.rates import binom_ci, poisson_ci
from brepi.config import PATHS
from brepi.geo import lattice

ROOT = Path(__file__).resolve().parents[2]
OUT = PATHS.results / "sim_case_definition"
SIM_FULL = PATHS.results / "audit_sim_sih" / "sim_a27_all_fields.parquet"
SIM_INTERIM = PATHS.interim / "sim_a27_deaths.parquet"
TRI = PATHS.panel / "triangulation_municipality_year.parquet"
HR_ATLAS = PATHS.results / "atlas" / "health_region_atlas.parquet"

RSCRIPT = r"C:/Program Files/R/R-4.4.1/bin/Rscript.exe"

#: SIM final release ends 2024; SINAN runs to 2025. Every SIM quantity here is
#: fitted inside 2007-2024, and the three-system window of 42_southeast_trend
#: (2008-2024) is reused verbatim for the Southeast AAPC so the number replaces
#: the published one rather than sitting alongside an incomparable variant.
SIM_YEARS = (2007, 2024)
COMMON = (2008, 2024)
Z = float(stats.norm.ppf(0.975))

#: From 40_ascertainment_depth.py; a health region with a handful of cases
#: contributes a hospitalisation share whose sampling error dwarfs the gradient.
MIN_CASES = 30

PART1 = ("LINHAA", "LINHAB", "LINHAC", "LINHAD")
PART2 = ("LINHAII",)
CHAIN = PART1 + PART2 + ("CAUSABAS_O",)

DEFS = ("underlying", "part1", "certificate", "any_mention")


def rule(s: str) -> None:
    print("\n" + "=" * 78 + "\n" + s)


# ---------------------------------------------------------------------------
# Estimation engines, lifted from 42_southeast_trend.py so that the Southeast
# AAPC recomputed here is produced by the same arithmetic as the number it is
# meant to replace. Validated against R below before anything is written.
# ---------------------------------------------------------------------------
def _irls_poisson(y: np.ndarray, X: np.ndarray, offset: np.ndarray,
                  tol: float = 1e-11, maxit: int = 100):
    mu = y + 0.1
    eta = np.log(mu)
    beta = np.zeros(X.shape[1])
    dev_old = np.inf
    for _ in range(maxit):
        w = mu
        z = eta - offset + (y - mu) / mu
        XtW = X.T * w
        XtWX = XtW @ X
        beta = np.linalg.solve(XtWX, XtW @ z)
        eta = np.clip(X @ beta + offset, -700, 700)
        mu = np.exp(eta)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(y / mu), 0.0)
        dev = 2.0 * np.sum(term - (y - mu))
        if abs(dev - dev_old) / (abs(dev) + 0.1) < tol:
            break
        dev_old = dev
    XtWX = (X.T * mu) @ X
    return beta, mu, XtWX


def mann_kendall(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return float("nan"), float("nan")
    d = x[:, None] - x[None, :]
    s = float(np.sum(np.sign(d[np.tril_indices(n, -1)])))
    _, counts = np.unique(x, return_counts=True)
    vs = (n * (n - 1) * (2 * n + 5)
          - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    z = (s - 1) / np.sqrt(vs) if s > 0 else ((s + 1) / np.sqrt(vs) if s < 0 else 0.0)
    return s / (0.5 * n * (n - 1)), float(2 * stats.norm.cdf(-abs(z)))


def qpois_aapc(years, events, person_time, *, conf: float = 0.95) -> dict:
    """AAPC of a rate from a log-linear quasi-Poisson model with a log offset."""
    yr = np.asarray(years, dtype=float)
    y = np.asarray(events, dtype=float)
    pt = np.asarray(person_time, dtype=float)
    keep = pt > 0
    yr, y, pt = yr[keep], y[keep], pt[keep]
    order = np.argsort(yr)
    yr, y, pt = yr[order], y[order], pt[order]
    n = yr.size
    out = {"n_years": int(n), "events": float(y.sum())}
    if n < 4 or y.sum() == 0:
        out.update(aapc=np.nan, aapc_lo=np.nan, aapc_hi=np.nan,
                   dispersion=np.nan, mk_tau=np.nan, mk_p=np.nan)
        return out
    X = np.column_stack([np.ones(n), yr - yr.mean()])
    beta, mu, XtWX = _irls_poisson(y, X, np.log(pt))
    dispersion = float(np.sum(((y - mu) / np.sqrt(mu)) ** 2) / (n - X.shape[1]))
    cov = dispersion * np.linalg.inv(XtWX)
    b, se = float(beta[1]), float(np.sqrt(cov[1, 1]))
    z = float(stats.norm.ppf(1 - (1 - conf) / 2))
    mk = mann_kendall(y / pt)
    out.update(aapc=100 * (np.exp(b) - 1),
               aapc_lo=100 * (np.exp(b - z * se) - 1),
               aapc_hi=100 * (np.exp(b + z * se) - 1),
               dispersion=dispersion, mk_tau=mk[0], mk_p=mk[1])
    return out


def qbinom_trend(years, success, total, *, conf: float = 0.95) -> dict:
    """Annual ODDS RATIO of a proportion from a quasi-binomial logistic model."""
    yr = np.asarray(years, dtype=float)
    k = np.asarray(success, dtype=float)
    m = np.asarray(total, dtype=float)
    keep = m > 0
    yr, k, m = yr[keep], k[keep], m[keep]
    order = np.argsort(yr)
    yr, k, m = yr[order], k[order], m[order]
    n = yr.size
    out = {"n_years": int(n), "success": float(k.sum()), "total": float(m.sum())}
    if n < 4:
        out.update(or_year=np.nan, or_lo=np.nan, or_hi=np.nan,
                   dispersion=np.nan, mk_tau=np.nan, mk_p=np.nan)
        return out
    X = np.column_stack([np.ones(n), yr - yr.mean()])
    p = (k + 0.5) / (m + 1.0)
    eta = np.log(p / (1 - p))
    beta = np.zeros(2)
    for _ in range(200):
        w = m * p * (1 - p)
        z = eta + (k / m - p) / (p * (1 - p))
        XtW = X.T * w
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-12:
            beta = beta_new
            break
        beta = beta_new
        eta = np.clip(X @ beta, -700, 700)
        p = 1.0 / (1.0 + np.exp(-eta))
    w = m * p * (1 - p)
    XtWX = (X.T * w) @ X
    dispersion = float(np.sum(((k - m * p) / np.sqrt(w)) ** 2) / (n - 2))
    cov = dispersion * np.linalg.inv(XtWX)
    b, se = float(beta[1]), float(np.sqrt(cov[1, 1]))
    z = float(stats.norm.ppf(1 - (1 - conf) / 2))
    mk = mann_kendall(k / m)
    out.update(or_year=float(np.exp(b)), or_lo=float(np.exp(b - z * se)),
               or_hi=float(np.exp(b + z * se)), dispersion=dispersion,
               mk_tau=mk[0], mk_p=mk[1])
    return out


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def check_ci_helpers() -> None:
    """The (estimate, lo, hi) contract. This repo has been bitten by it once."""
    est, lo, hi = poisson_ci([5705], [3.4e9], scale=1e5)
    assert lo[0] <= est[0] <= hi[0], (est, lo, hi)
    assert abs(est[0] - 5705 / 3.4e9 * 1e5) < 1e-12
    p, plo, phi = binom_ci([601], [6306])
    assert plo[0] <= p[0] <= phi[0] and abs(p[0] - 601 / 6306) < 1e-12
    print("  CI helpers return (estimate, lower, upper) - verified")


def validate_token_alignment(df: pl.DataFrame) -> dict:
    """THE FALSE-POSITIVE GATE.

    The multiple-cause fields concatenate four-character codes. The extraction
    rule matches them by unanchored substring, which could in principle fire on
    a boundary-spanning "A27" that is not a leptospirosis code at all. Every
    such hit would be a fabricated mention-only death.

    The fields are ``*``-delimited in practice. Split on the delimiter and
    require that every A27-containing token *begins* A27. An epidemiologically
    impossible number is a bug until proven otherwise; this is the proof.
    """
    tokens: dict[str, int] = {}
    misaligned: list[tuple[str, str, str]] = []
    undelimited = 0
    for row in df.select(*CHAIN).iter_rows(named=True):
        for col in CHAIN:
            v = (row[col] or "").strip()
            if "A27" not in v:
                continue
            if "*" in v:
                parts = [p for p in v.split("*") if p]
            else:
                undelimited += 1
                parts = [v[i:i + 4] for i in range(0, len(v), 4)]
            for p in parts:
                if "A27" not in p:
                    continue
                tokens[p] = tokens.get(p, 0) + 1
                if not p.startswith("A27"):
                    misaligned.append((col, v, p))
    if misaligned:
        raise AssertionError(
            f"{len(misaligned)} A27 substring hits are not code-aligned, e.g. "
            f"{misaligned[:5]}; the mention counts would be fabricated"
        )
    print(f"  A27-bearing tokens: "
          f"{', '.join(f'{k}={v}' for k, v in sorted(tokens.items()))}")
    print(f"  misaligned substring hits: 0 of {sum(tokens.values())} "
          f"-> the mention supplement is not a substring artefact")
    return {"tokens": tokens, "misaligned": 0,
            "undelimited_field_values": undelimited}


def reconcile(df: pl.DataFrame) -> dict:
    """The deep scan must reproduce the frozen extract exactly on underlying cause."""
    interim = pl.read_parquet(SIM_INTERIM)
    n_und = int(df.filter(pl.col("d_underlying")).height)
    assert n_und == interim.height, (
        f"deep scan has {n_und} underlying-cause A27 deaths but the frozen "
        f"extract has {interim.height}; the two populations have drifted"
    )
    assert set(df["_uf"].unique()) == set(interim["_src_uf"].unique())
    print(f"  deep scan underlying-cause subset = frozen extract exactly "
          f"({n_und} deaths, {df['_uf'].n_unique()} UFs, "
          f"{df['year'].min()}-{df['year'].max()})")
    # A death certificate that appeared twice would inflate the supplement.
    key = ["DTOBITO", "DTNASC", "CODMUNRES", "SEXO", "CAUSABAS"]
    dup = df.height - df.select(key).unique().height
    print(f"  near-duplicate certificates on {'+'.join(key)}: {dup} of {df.height}")
    return {"underlying_deep": n_und, "underlying_frozen": interim.height,
            "near_duplicates": dup}


def validate_against_r(series: pl.DataFrame) -> dict:
    """Refit the Southeast SIM series in R and demand agreement to < 1e-6."""
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "series.csv"
        series.write_csv(csv)
        rs = Path(td) / "fit.R"
        rs.write_text(
            'd <- read.csv("%s")\n'
            'f <- glm(events ~ year + offset(log(person_time)), '
            'family = quasipoisson(), data = d)\n'
            'co <- summary(f)$coefficients["year", ]\n'
            'cat(sprintf("%%.12f %%.12f %%.12f\\n", co[1], co[2], '
            'summary(f)$dispersion))\n' % csv.as_posix(), encoding="utf-8")
        res = subprocess.run([RSCRIPT, "--vanilla", str(rs)],
                             capture_output=True, text=True, cwd=str(ROOT))
        if res.returncode != 0:
            raise RuntimeError("R validation failed:\n" + res.stderr)
        b_r, se_r, disp_r = (float(v) for v in res.stdout.split())
    mine = qpois_aapc(series["year"], series["events"], series["person_time"])
    aapc_r = 100 * (np.exp(b_r) - 1)
    lo_r = 100 * (np.exp(b_r - Z * se_r) - 1)
    hi_r = 100 * (np.exp(b_r + Z * se_r) - 1)
    drift = max(abs(mine["aapc"] - aapc_r), abs(mine["aapc_lo"] - lo_r),
                abs(mine["aapc_hi"] - hi_r), abs(mine["dispersion"] - disp_r) / disp_r)
    print(f"  R glm(quasipoisson) AAPC {aapc_r:+.4f} ({lo_r:+.4f}, {hi_r:+.4f}); "
          f"python {mine['aapc']:+.4f} ({mine['aapc_lo']:+.4f}, {mine['aapc_hi']:+.4f})")
    if drift > 1e-6:
        raise AssertionError(f"python AAPC drifts from R by {drift:.2e}")
    print(f"  AAPC engine matches R to {drift:.1e} - proceeding")
    return {"r_aapc": aapc_r, "python_aapc": mine["aapc"], "max_abs_drift": drift}


# ---------------------------------------------------------------------------
# Load and label
# ---------------------------------------------------------------------------
def load() -> pl.DataFrame:
    """The 6,306 A27-bearing certificates, with the four definitions attached."""
    f = pl.read_parquet(SIM_FULL)
    code = lambda c: pl.col(c).cast(pl.Utf8).str.strip_chars()  # noqa: E731

    def has_a27(cols) -> pl.Expr:
        e = pl.lit(False)
        for c in cols:
            if c in f.columns:
                e = e | code(c).str.contains("A27", literal=True).fill_null(False)
        return e

    d = f.with_columns(
        code("CAUSABAS").str.starts_with("A27").fill_null(False).alias("d_underlying"),
        has_a27(PART1).alias("in_part1"),
        has_a27(PART2).alias("in_part2"),
        has_a27(("CAUSABAS_O",)).alias("in_causabas_o"),
        code("DTOBITO").str.slice(4, 4).cast(pl.Int32, strict=False).alias("year"),
        code("CAUSABAS").alias("cb"),
    )
    d = d.with_columns(
        (pl.col("d_underlying") | pl.col("in_part1")).alias("d_part1"),
        (pl.col("d_underlying") | pl.col("in_part1")
         | pl.col("in_part2")).alias("d_certificate"),
        pl.lit(True).alias("d_any_mention"),
    )
    # Position taxonomy for the records the current definition drops. The three
    # strata are different documentary acts and the recommendation turns on
    # telling them apart, so they are labelled once, here.
    d = d.with_columns(
        pl.when(pl.col("d_underlying")).then(pl.lit("0 underlying cause"))
        .when(pl.col("in_part1")).then(pl.lit("1 Part I causal sequence"))
        .when(pl.col("in_part2")).then(pl.lit("2 Part II contributing"))
        .otherwise(pl.lit("3 original underlying only (recoded away)"))
        .alias("position")
    )
    assert d.filter(pl.col("year").is_between(*SIM_YEARS)).height == d.height
    return d


# ---------------------------------------------------------------------------
# 1. What ARE the 601?
# ---------------------------------------------------------------------------
#: Ordered CID-10 rules, first match wins, matched on the leading three
#: characters of ``CAUSABAS``. The grouping is the clinical question -- "is this
#: the organ failure severe leptospirosis kills through?" -- not an ICD chapter.
#: Codes are the WHO ICD-10 blocks; the sequelae list is the one already used in
#: 47_audit_sim_sih.py's complication profile, widened to the full block.
CAUSE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sepsis or shock", ("A40", "A41", "A48", "R57", "R65")),
    ("acute renal failure", ("N17", "N19", "N99", "N28")),
    ("hepatic failure or jaundice", ("K70", "K71", "K72", "K74", "K76", "R17")),
    ("respiratory failure or pulmonary haemorrhage",
     ("J80", "J81", "J96", "J98", "R04", "R09")),
    ("pneumonia", ("J09", "J10", "J11", "J12", "J13", "J14", "J15", "J16",
                   "J17", "J18")),
    ("rhabdomyolysis or muscle", ("M62", "M60", "M63")),
    ("coagulopathy", ("D65", "D68", "D69")),
    ("pancreatitis or biliary", ("K80", "K81", "K82", "K83", "K85", "K86")),
    ("myocarditis or cardiac arrest", ("I40", "I46", "I42")),
    ("meningitis or cerebral oedema", ("G00", "G03", "G04", "G93")),
    ("maternal (ICD chapter XV takes priority)",
     ("O00", "O01", "O02", "O03", "O04", "O05", "O06", "O07", "O08", "O45",
      "O75", "O85", "O88", "O95", "O96", "O98", "O99")),
    ("dengue", ("A90", "A91")),
    ("other arboviral or viral haemorrhagic fever",
     ("A92", "A93", "A94", "A95", "A96", "A97", "A98", "A99", "A86", "A87")),
    ("COVID-19", ("B34", "U07", "U04")),
    ("viral hepatitis", ("B15", "B16", "B17", "B18", "B19")),
    ("HIV", ("B20", "B21", "B22", "B23", "B24")),
    ("tuberculosis", ("A15", "A16", "A17", "A18", "A19")),
    ("malaria or other parasitic", ("B50", "B51", "B52", "B53", "B54", "B55",
                                    "B57", "B65", "B67")),
    ("other infectious", ("A0", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
                          "B0", "B3", "B4", "B9", "G0")),
    ("neoplasm", ("C", "D0", "D1", "D2", "D3", "D4")),
    ("chronic circulatory", ("I0", "I1", "I2", "I3", "I5", "I6", "I7", "I8", "I9")),
    ("diabetes or metabolic", ("E1", "E8", "E4", "E5", "E6", "E7", "E87")),
    ("chronic respiratory", ("J4", "J6", "J7", "J3")),
    ("congenital", ("Q",)),
    ("external cause", ("V", "W", "X", "Y")),
    ("ill-defined", ("R99", "R68", "R54")),
)

#: Groups a physician-plus-coder would produce if leptospirosis really was the
#: initiating condition and the selection rules moved the underlying cause to
#: the terminal event. Stated once so the arithmetic below is auditable.
SEQUELAE_GROUPS = frozenset({
    "sepsis or shock", "acute renal failure", "hepatic failure or jaundice",
    "respiratory failure or pulmonary haemorrhage", "pneumonia",
    "rhabdomyolysis or muscle", "coagulopathy", "pancreatitis or biliary",
    "myocarditis or cardiac arrest", "meningitis or cerebral oedema",
    "maternal (ICD chapter XV takes priority)", "ill-defined",
})

#: Infections that share leptospirosis' acute febrile-jaundice-haemorrhage
#: presentation. When one of these is the underlying cause and A27 is merely
#: mentioned, the coder had two candidate killers and picked the other one.
COMPETING_INFECTION_GROUPS = frozenset({
    "dengue", "other arboviral or viral haemorrhagic fever", "COVID-19",
    "viral hepatitis", "HIV", "tuberculosis", "malaria or other parasitic",
    "other infectious",
})

#: The partition that decides the estimand. Three positions, and only the first
#: is a defect in the current count:
#:
#: ``downstream``  the underlying cause is a complication OF leptospirosis, so
#:      the selection rules demoted the infection below its own consequence.
#:      These are leptospirosis deaths that the underlying-cause rule loses.
#: ``competing``   another acute infection with an overlapping syndrome was
#:      selected. Attribution is genuinely contested and neither definition is
#:      obviously right.
#: ``antecedent``  the underlying cause is a chronic or unrelated condition
#:      that preceded the infection. ICD is doing exactly what it is designed
#:      to do: the antecedent initiated the sequence. Counting these does not
#:      fix an undercount, it changes the estimand from "died OF leptospirosis"
#:      to "died WITH leptospirosis on the certificate".
ESTIMAND_PARTITION = (
    ("downstream of leptospirosis (coding demotion)", SEQUELAE_GROUPS),
    ("competing acute infection (contested attribution)", COMPETING_INFECTION_GROUPS),
)


def partition() -> pl.Expr:
    e = pl.when(pl.col("cause_group").is_in(list(SEQUELAE_GROUPS))).then(
        pl.lit(ESTIMAND_PARTITION[0][0]))
    e = e.when(pl.col("cause_group").is_in(list(COMPETING_INFECTION_GROUPS))).then(
        pl.lit(ESTIMAND_PARTITION[1][0]))
    return e.otherwise(
        pl.lit("antecedent or unrelated condition (estimand change)")
    ).alias("estimand_class")


def classify(col: str = "cb") -> pl.Expr:
    e = pl.when(pl.lit(False)).then(pl.lit("x"))
    for label, prefixes in CAUSE_RULES:
        cond = pl.lit(False)
        for p in prefixes:
            cond = cond | pl.col(col).str.starts_with(p)
        e = e.when(cond).then(pl.lit(label))
    return e.otherwise(pl.lit("unclassified")).alias("cause_group")


def anatomy(d: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Position taxonomy crossed with the underlying cause of the dropped deaths."""
    mo = d.filter(~pl.col("d_underlying")).with_columns(classify())
    pos = mo.group_by("position").agg(pl.len().alias("n")).sort("position")

    grp = mo.group_by(["position", "cause_group"]).agg(pl.len().alias("n"))
    tot = mo.group_by("cause_group").agg(pl.len().alias("n")).with_columns(
        pl.lit("all mention-only").alias("position")
    ).select("position", "cause_group", "n")
    table = pl.concat([grp, tot]).with_columns(
        (100 * pl.col("n") / pl.col("n").sum().over("position")).alias("pct_of_position"),
        pl.col("cause_group").is_in(list(SEQUELAE_GROUPS)).alias("is_sequela"),
    ).sort(["position", "n"], descending=[False, True])

    summary: dict = {"n_mention_only": mo.height,
                     "distinct_underlying_codes": int(mo["cb"].n_unique())}
    for (p,), sub in mo.group_by(["position"]):
        s = sub.with_columns(classify())
        k = int(s.filter(pl.col("cause_group").is_in(list(SEQUELAE_GROUPS))).height)
        est, lo, hi = binom_ci([k], [float(sub.height)])
        assert lo[0] <= est[0] <= hi[0]
        summary[p] = {"n": sub.height, "sequela_n": k,
                      "sequela_pct": 100 * float(est[0]),
                      "sequela_lo": 100 * float(lo[0]),
                      "sequela_hi": 100 * float(hi[0])}
    k = int(mo.filter(pl.col("cause_group").is_in(list(SEQUELAE_GROUPS))).height)
    est, lo, hi = binom_ci([k], [float(mo.height)])
    assert lo[0] <= est[0] <= hi[0]
    summary["all mention-only"] = {"n": mo.height, "sequela_n": k,
                                   "sequela_pct": 100 * float(est[0]),
                                   "sequela_lo": 100 * float(lo[0]),
                                   "sequela_hi": 100 * float(hi[0])}
    summary["unclassified_n"] = int(
        mo.filter(pl.col("cause_group") == "unclassified").height)

    # The partition that decides the estimand, crossed with the position. The
    # two axes answer different questions: position says what documentary act
    # produced the mention, partition says what counting it would MEAN.
    p = mo.with_columns(partition())
    cross = p.group_by(["position", "estimand_class"]).agg(pl.len().alias("n"))
    marg = p.group_by("estimand_class").agg(pl.len().alias("n")).with_columns(
        pl.lit("all mention-only").alias("position")
    ).select("position", "estimand_class", "n")
    est_tab = pl.concat([cross, marg]).with_columns(
        (100 * pl.col("n") / pl.col("n").sum().over("position")).alias("pct_of_position")
    ).sort(["position", "n"], descending=[False, True])
    summary["estimand_partition"] = {
        r["estimand_class"]: {"n": r["n"], "pct": r["pct_of_position"]}
        for r in marg.join(est_tab.filter(pl.col("position") == "all mention-only")
                           .select("estimand_class", "pct_of_position"),
                           on="estimand_class").iter_rows(named=True)
    }
    return pos, table, est_tab, summary


def recode_flow(d: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Which way does the coding machinery move leptospirosis?

    ``CAUSABAS_O`` is the underlying cause the certificate carried *before* the
    selection/investigation machinery acted. Comparing it with ``CAUSABAS``
    gives the net flow. If A27 is moved OUT of the underlying position more
    often than INTO it, the study's count is systematically eroded by recoding
    and the 173 recoded-away deaths are a loss to be restored. If the flow runs
    the other way, those 173 are the counter-current of a process that has
    already *added* far more deaths than it removed, and restoring only the
    counter-current would be one-sided.
    """
    code = lambda c: pl.col(c).cast(pl.Utf8).str.strip_chars()  # noqa: E731
    g = d.with_columns(
        code("CAUSABAS_O").str.contains("A27", literal=True)
        .fill_null(False).alias("o_a27"),
        code("CAUSABAS_O").alias("cbo"),
    )
    tab = g.group_by(["d_underlying", "o_a27"]).agg(pl.len().alias("n")).sort(
        ["d_underlying", "o_a27"])
    into = g.filter(pl.col("d_underlying") & ~pl.col("o_a27"))
    away = g.filter(~pl.col("d_underlying") & pl.col("o_a27"))
    top_into = (into.with_columns(classify("cbo"))
                .group_by("cause_group").agg(pl.len().alias("n"))
                .with_columns(pl.lit("recoded INTO A27: original cause")
                              .alias("flow")))
    top_away = (away.with_columns(classify("cb"))
                .group_by("cause_group").agg(pl.len().alias("n"))
                .with_columns(pl.lit("recoded AWAY from A27: final cause")
                              .alias("flow")))
    flow = pl.concat([top_into, top_away]).with_columns(
        (100 * pl.col("n") / pl.col("n").sum().over("flow")).alias("pct_of_flow")
    ).sort(["flow", "n"], descending=[False, True])
    return flow, {
        "recoded_into_a27": into.height,
        "recoded_away_from_a27": away.height,
        "net_gain_to_the_underlying_count": into.height - away.height,
        "cross_tab": tab.to_dicts(),
    }


def concentration(d: pl.DataFrame) -> pl.DataFrame:
    """PLAUSIBILITY: is any single underlying cause a local coding habit?

    223 distinct codes over 601 deaths is a long tail, but a block that is
    16 deaths nationally and 14 of them in one state is a data-entry pattern,
    not an epidemiological signal, and must not be read as one.
    """
    mo = d.filter(~pl.col("d_underlying")).with_columns(
        pl.col("cb").str.slice(0, 3).alias("b3"))
    top = mo.group_by("b3").agg(pl.len().alias("n")).sort("n", descending=True).head(12)
    rows = []
    for b3 in top["b3"]:
        s = mo.filter(pl.col("b3") == b3)
        uf = s.group_by("_uf").agg(pl.len().alias("n")).sort("n", descending=True)
        yr = s.group_by("year").agg(pl.len().alias("n")).sort("n", descending=True)
        rows.append({
            "block": b3, "n": s.height,
            "top_uf": uf["_uf"][0], "top_uf_n": int(uf["n"][0]),
            "pct_in_top_uf": round(100 * int(uf["n"][0]) / s.height, 1),
            "n_ufs": uf.height,
            "top_year": int(yr["year"][0]), "top_year_n": int(yr["n"][0]),
            "cause_group": s.with_columns(classify())["cause_group"][0],
        })
    return pl.DataFrame(rows)


def completeness_threat(d: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """THE NAMED THREAT: the Part II field is filling up over the series.

    If ``LINHAII`` is filled on 33% of certificates in 2007 and 41% in 2024,
    then any definition that draws on Part II adds proportionally more deaths
    late than early, and the resulting mortality trend is partly the trend in
    multiple-cause recording. Measured on all 6,306 A27 certificates: their
    Part II fill rate is the exposure that the definition's supplement rides on.
    """
    code = lambda c: pl.col(c).cast(pl.Utf8).str.strip_chars()  # noqa: E731
    filled = lambda c: (code(c).is_not_null() & (code(c) != "")).cast(pl.Int32)  # noqa: E731
    y = d.group_by("year").agg(
        pl.len().alias("certificates"),
        filled("LINHAA").sum().alias("linhaa_filled"),
        filled("LINHAII").sum().alias("linhaii_filled"),
        (~pl.col("d_underlying")).sum().alias("mention_only"),
        (pl.col("d_certificate") & ~pl.col("d_underlying")).sum()
        .alias("supplement_certificate"),
    ).sort("year")
    trends = []
    for num, den, label in [
        ("linhaii_filled", "certificates",
         "Part II (LINHAII) filled, among A27 certificates"),
        ("linhaa_filled", "certificates",
         "Part I line a (LINHAA) filled, among A27 certificates"),
        ("mention_only", "certificates",
         "mention-only share of all A27 certificates"),
    ]:
        trends.append({"marker": label,
                       "measure": "odds ratio per calendar year "
                                  "(quasi-binomial logistic)",
                       "window": f"{SIM_YEARS[0]}-{SIM_YEARS[1]}",
                       **qbinom_trend(y["year"], y[num], y[den])})
    p, lo, hi = binom_ci(y["linhaii_filled"].to_numpy(),
                         y["certificates"].to_numpy().astype(float))
    assert np.all((lo <= p) & (p <= hi))
    y = y.with_columns(pl.Series("linhaii_share", p),
                       pl.Series("linhaii_lo", lo), pl.Series("linhaii_hi", hi))
    return y, {"trends": trends}


# ---------------------------------------------------------------------------
# 2. Geography: is the undercount differential across the depth gradient?
# ---------------------------------------------------------------------------
def bands() -> pl.DataFrame:
    """Case-weighted quintiles of hospitalisation share among confirmed cases.

    Copied verbatim from 54_sih_severity_depth.py, which copied it from
    40_ascertainment_depth.py. Q1 is the deepest surveillance (lowest
    hospitalisation share), Q5 the shallowest.
    """
    hr = pl.read_parquet(HR_ATLAS)
    hr = hr.filter(pl.col("cases") >= MIN_CASES).with_columns(
        (pl.col("hospitalised") / pl.col("cases")).alias("hosp_share")
    ).sort("hosp_share")
    hr = hr.with_columns(
        (pl.col("cases").cum_sum() / pl.col("cases").sum()).alias("_cw")
    ).with_columns(
        (pl.col("_cw") * 5).ceil().clip(1, 5).cast(pl.Int32).alias("quintile")
    )
    return hr.select("health_region_code", "quintile", "hosp_share")


def geolocate(d: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Attach municipality of residence and health region, exactly as 18 does.

    Residence, never occurrence: the study's estimand is residence-aligned and
    rescuing a missing residence code with the place of death would silently
    change it.
    """
    resolved = lattice.resolve_municipality_code(
        d, [("CODMUNRES", "CODMUNRES")], lattice_year=2022)
    n_unres = int((resolved["munic_code_status"] == "unresolved").sum())
    ok = resolved.filter(pl.col("munic_code_status") == "resolved").with_columns(
        lattice.code6_to_code7_expr(pl.col("munic_code6")).alias("munic_code"))
    geo = (pl.read_parquet(TRI)
           .select("munic_code", "health_region_code", "region", "uf_abbr")
           .unique(subset=["munic_code"]))
    out = ok.join(geo, on="munic_code", how="left")
    n_nogeo = int(out["health_region_code"].null_count())
    audit = {"records": d.height, "unresolved_municipality": n_unres,
             "resolved_but_no_health_region": n_nogeo,
             "placed": out.height - n_nogeo}
    print(f"  geography: {d.height} certificates -> {audit['placed']} placed "
          f"({n_unres} unresolvable CODMUNRES, {n_nogeo} outside the panel)")
    return out.drop_nulls("health_region_code"), audit


def mention_share_by_band(g: pl.DataFrame) -> pl.DataFrame:
    """Mention-only share of all A27 certificates, by depth band.

    The denominator is every A27-bearing certificate in the band, so the
    quantity is "what fraction of this territory's leptospirosis-labelled
    deaths the underlying-cause rule discards". If it rises with shallowness,
    the undercount is differential and biases the study's own comparison.
    """
    b = g.join(bands(), on="health_region_code", how="inner")
    q = b.group_by("quintile").agg(
        pl.len().alias("certificates"),
        pl.col("d_underlying").sum().alias("underlying"),
        (~pl.col("d_underlying")).sum().alias("mention_only"),
        (pl.col("d_part1") & ~pl.col("d_underlying")).sum().alias("add_part1"),
        (pl.col("d_certificate") & ~pl.col("d_part1")).sum().alias("add_part2"),
        (~pl.col("d_certificate")).sum().alias("add_recoded_away"),
        pl.col("hosp_share").median().alias("band_median_hosp_share"),
    ).sort("quintile")
    p, lo, hi = binom_ci(q["mention_only"].to_numpy(),
                         q["certificates"].to_numpy().astype(float))
    assert np.all((lo <= p) & (p <= hi))
    return q.with_columns(
        pl.Series("mention_only_pct", 100 * p),
        pl.Series("mention_only_lo", 100 * lo),
        pl.Series("mention_only_hi", 100 * hi),
    )


def band_gradient_within_region(g: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """THREAT: the band gradient in mention-only share is macro-region composition.

    The depth bands are not regionally balanced (Q1 holds no Nordeste at all,
    Q5 almost no Sul), and multiple-cause recording is a state-level clerical
    practice. If the whole band contrast were regional composition, it would
    vanish inside a single macro-region. Sudeste is the only region present in
    all five bands, so it is the only place the test can be run, and its 2,506
    certificates are enough to run it on.
    """
    b = g.join(bands(), on="health_region_code", how="inner")
    rows = []
    for reg in ["Sudeste", "Sul", "Nordeste", "Norte"]:
        s = b.filter(pl.col("region") == reg)
        q = s.group_by("quintile").agg(
            pl.len().alias("certificates"),
            (~pl.col("d_underlying")).sum().alias("mention_only"),
        ).sort("quintile")
        if q.height < 4:
            rows.append({"region": reg, "bands_present": q.height,
                         "note": "present in too few bands to fit a gradient"})
            continue
        p, lo, hi = binom_ci(q["mention_only"].to_numpy(),
                             q["certificates"].to_numpy().astype(float))
        assert np.all((lo <= p) & (p <= hi))
        tr = qbinom_trend(q["quintile"], q["mention_only"], q["certificates"])
        rows.append({"region": reg, "bands_present": q.height,
                     "certificates": int(q["certificates"].sum()),
                     "mention_only": int(q["mention_only"].sum()),
                     "deepest_band_present": int(q["quintile"][0]),
                     "shallowest_band_present": int(q["quintile"][-1]),
                     "deepest_band_pct": 100 * float(p[0]),
                     "shallowest_band_pct": 100 * float(p[-1]),
                     "or_per_band_step": tr["or_year"],
                     "or_lo": tr["or_lo"], "or_hi": tr["or_hi"],
                     "dispersion": tr["dispersion"],
                     # A quasi-binomial fit on five bands has three residual
                     # degrees of freedom. When the five proportions happen to
                     # sit near a logistic line the estimated dispersion falls
                     # below one and the interval becomes NARROWER than the
                     # binomial interval -- which is not a statement about
                     # sampling error that anyone should believe. Flagged, not
                     # silently rescaled.
                     "interval_credible": bool(tr["dispersion"] >= 1.0),
                     "mk_p": tr["mk_p"]})
    tab = pl.DataFrame(rows, infer_schema_length=None)
    return tab, {"within_region": rows}


def mention_share_by_region(g: pl.DataFrame) -> pl.DataFrame:
    r = g.group_by("region").agg(
        pl.len().alias("certificates"),
        (~pl.col("d_underlying")).sum().alias("mention_only"),
    ).sort("region")
    p, lo, hi = binom_ci(r["mention_only"].to_numpy(),
                         r["certificates"].to_numpy().astype(float))
    assert np.all((lo <= p) & (p <= hi))
    return r.with_columns(pl.Series("mention_only_pct", 100 * p),
                          pl.Series("lo", 100 * lo), pl.Series("hi", 100 * hi))


# ---------------------------------------------------------------------------
# 3. Recompute the study's SIM quantities under every definition
# ---------------------------------------------------------------------------
def person_years() -> tuple[pl.DataFrame, float]:
    """Person-years 2007-2024 by health region, from the panel SIM is joined to.

    Taken from the triangulation panel rather than the health-region atlas
    because the atlas spans 2007-2025 while SIM's final release ends 2024; a
    denominator a year wider than the numerator would deflate every rate by
    about 5%.
    """
    tri = pl.read_parquet(TRI).filter(
        pl.col("year").is_between(*SIM_YEARS) & pl.col("sim_covered"))
    by_hr = tri.group_by("health_region_code").agg(
        pl.col("population").sum().cast(pl.Float64).alias("person_years"))
    return by_hr, float(tri["population"].sum())


def national_rates(g: pl.DataFrame, pt_national: float) -> pl.DataFrame:
    rows = []
    for name in DEFS:
        n = int(g.filter(pl.col(f"d_{name}")).height)
        est, lo, hi = poisson_ci([float(n)], [pt_national], scale=1e5)
        assert lo[0] <= est[0] <= hi[0]
        rows.append({
            "definition": name, "deaths": n,
            "person_years": pt_national,
            "measure": "A27 mortality RATE per 100,000 person-years",
            "window": f"{SIM_YEARS[0]}-{SIM_YEARS[1]}",
            "rate_per_100k_py": float(est[0]),
            "rate_lo": float(lo[0]), "rate_hi": float(hi[0]),
        })
    base = rows[0]["deaths"]
    for r in rows:
        r["pct_above_underlying"] = 100 * (r["deaths"] - base) / base
    return pl.DataFrame(rows)


def rates_by_band(g: pl.DataFrame, pt_hr: pl.DataFrame) -> pl.DataFrame:
    b = g.join(bands(), on="health_region_code", how="inner")
    pt = (bands().join(pt_hr, on="health_region_code", how="left")
          .group_by("quintile").agg(pl.col("person_years").sum()))
    rows = []
    for name in DEFS:
        cnt = b.filter(pl.col(f"d_{name}")).group_by("quintile").agg(
            pl.len().alias("deaths"))
        m = pt.join(cnt, on="quintile", how="left").with_columns(
            pl.col("deaths").fill_null(0)).sort("quintile")
        est, lo, hi = poisson_ci(m["deaths"].to_numpy().astype(float),
                                 m["person_years"].to_numpy(), scale=1e5)
        assert np.all((lo <= est) & (est <= hi))
        for i, r in enumerate(m.iter_rows(named=True)):
            rows.append({
                "definition": name, "quintile": r["quintile"],
                "deaths": int(r["deaths"]), "person_years": r["person_years"],
                "measure": "A27 mortality RATE per 100,000 person-years",
                "rate_per_100k_py": float(est[i]),
                "rate_lo": float(lo[i]), "rate_hi": float(hi[i]),
            })
    out = pl.DataFrame(rows)
    base = out.filter(pl.col("definition") == "underlying").select(
        "quintile", pl.col("rate_per_100k_py").alias("rate_underlying"),
        pl.col("deaths").alias("deaths_underlying"))
    return out.join(base, on="quintile").with_columns(
        (pl.col("rate_per_100k_py") / pl.col("rate_underlying")).alias("ratio_to_underlying")
    ).sort(["definition", "quintile"])


def supplement_trend(g: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Is the AAPC shift epidemiology, or the Part II field filling up?

    Widening the definition moves the Southeast AAPC toward zero. Two things
    could do that: leptospirosis mortality really is falling more slowly than
    the underlying-cause series says, or the supplement is growing because the
    multiple-cause fields are being completed more often each year.

    The ratio model separates them. Offsetting the supplement by the count of
    underlying-cause deaths makes the slope the annual change in *supplement
    per true-positive death* -- a pure recording quantity, zero under the null
    that mentions are a fixed fraction of leptospirosis deaths.

    The test has a sharp prediction. Part I recording (``LINHAA``) is flat
    across the series; Part II recording (``LINHAII``) rises at 3.0%/yr. If the
    drift is documentary it must be concentrated in the Part II component and
    largely absent from the Part I component. If instead both rise together,
    the recording explanation fails and the supplement is telling us something
    about mortality.
    """
    tri = pl.read_parquet(TRI).filter(pl.col("sim_covered"))
    rows = []
    for terr, sel in [("Sudeste", pl.col("region") == "Sudeste"),
                      ("Brazil", pl.lit(True))]:
        s = g.filter(sel & pl.col("year").is_between(*COMMON))
        y = s.group_by("year").agg(
            pl.col("d_underlying").sum().alias("underlying"),
            (pl.col("d_part1") & ~pl.col("d_underlying")).sum().alias("sup_part1"),
            (pl.col("d_certificate") & ~pl.col("d_part1")).sum().alias("sup_part2"),
            (~pl.col("d_certificate")).sum().alias("sup_recoded"),
            (pl.col("d_certificate") & ~pl.col("d_underlying")).sum()
            .alias("sup_certificate"),
            (~pl.col("d_underlying")).sum().alias("sup_all"),
        ).sort("year")
        pop = (tri.filter(sel & pl.col("year").is_between(*COMMON))
               .group_by("year").agg(pl.col("population").sum().cast(pl.Float64)
                                     .alias("pt")).sort("year"))
        y = y.join(pop, on="year")
        for col, label, flat in [
            ("sup_part1", "supplement from Part I lines", "flat (LINHAA)"),
            ("sup_part2", "supplement from Part II only", "rising 3.0%/yr (LINHAII)"),
            ("sup_certificate", "supplement, Part I + Part II", "mixed"),
            ("sup_recoded", "supplement from recoded-away only", "n/a"),
            ("sup_all", "supplement, all sources", "mixed"),
        ]:
            rows.append({
                "territory": terr, "component": label,
                "source_field_completeness_trend": flat,
                "measure": "AAPC of the supplement per underlying-cause death "
                           "(quasi-Poisson ratio model)",
                "window": f"{COMMON[0]}-{COMMON[1]}",
                **qpois_aapc(y["year"], y[col], y["underlying"].cast(pl.Float64)),
            })
        # THREAT: the recoded-away drift is COVID. In 2020-21 the coding rules
        # put U07.1/B34.2 above everything else, which would move A27 out of the
        # underlying position for two years and manufacture an upward slope in a
        # series that ends in 2024. Refit without those two years.
        yx = y.filter(~pl.col("year").is_in([2020, 2021]))
        rows.append({
            "territory": terr,
            "component": "supplement from recoded-away only, excl. 2020-21",
            "source_field_completeness_trend": "n/a",
            "measure": "AAPC of the supplement per underlying-cause death "
                       "(quasi-Poisson ratio model)",
            "window": f"{COMMON[0]}-{COMMON[1]} excl. 2020-21",
            **qpois_aapc(yx["year"], yx["sup_recoded"],
                         yx["underlying"].cast(pl.Float64))})
        rows.append({
            "territory": terr, "component": "underlying-cause deaths (reference)",
            "source_field_completeness_trend": "n/a",
            "measure": "AAPC of the A27 mortality rate per 100,000 person-years",
            "window": f"{COMMON[0]}-{COMMON[1]}",
            **qpois_aapc(y["year"], y["underlying"], y["pt"])})
    tab = pl.DataFrame(rows)
    se = tab.filter((pl.col("territory") == "Sudeste"))
    grab = lambda c: se.filter(pl.col("component") == c).to_dicts()[0]  # noqa: E731
    return tab, {
        "sudeste_supplement_part1": grab("supplement from Part I lines"),
        "sudeste_supplement_part2": grab("supplement from Part II only"),
        "sudeste_supplement_certificate": grab("supplement, Part I + Part II"),
        "sudeste_supplement_recoded": grab("supplement from recoded-away only"),
        "sudeste_supplement_recoded_excl_covid": grab(
            "supplement from recoded-away only, excl. 2020-21"),
        "sudeste_supplement_all": grab("supplement, all sources"),
    }


def southeast_trend(g: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """The Link-7 number: the Southeast SIM AAPC, 2008-2024, under each definition."""
    tri = pl.read_parquet(TRI).filter(pl.col("sim_covered"))
    territories = {
        "Sudeste": pl.col("region") == "Sudeste",
        "Brazil": pl.lit(True),
        "Brazil excl. Sudeste": pl.col("region") != "Sudeste",
    }
    rows, series = [], []
    for terr, sel in territories.items():
        pop = (tri.filter(sel & pl.col("year").is_between(*COMMON))
               .group_by("year").agg(pl.col("population").sum()
                                     .cast(pl.Float64).alias("person_time"))
               .sort("year"))
        sub = g.filter(sel & pl.col("year").is_between(*COMMON))
        s = pop
        for name in DEFS:
            cnt = (sub.filter(pl.col(f"d_{name}")).group_by("year")
                   .agg(pl.len().alias(name)))
            s = s.join(cnt, on="year", how="left").with_columns(
                pl.col(name).fill_null(0))
        s = s.sort("year")
        series.append(s.with_columns(pl.lit(terr).alias("territory")))
        for name in DEFS:
            rows.append({"territory": terr, "definition": name,
                         "measure": "AAPC of the A27 mortality rate per 100,000 "
                                    "person-years (log-linear quasi-Poisson)",
                         "window": f"{COMMON[0]}-{COMMON[1]}",
                         **qpois_aapc(s["year"], s[name], s["person_time"])})
    trends = pl.DataFrame(rows)
    se = [r for r in rows if r["territory"] == "Sudeste"]
    base = next(r for r in se if r["definition"] == "underlying")
    verdict = {
        "published_underlying_aapc": -1.945113,
        "recomputed_underlying_aapc": base["aapc"],
        "definitions": {r["definition"]: {
            "aapc": r["aapc"], "aapc_lo": r["aapc_lo"], "aapc_hi": r["aapc_hi"],
            "events": r["events"], "mk_p": r["mk_p"],
            "interval_excludes_zero": bool(r["aapc_hi"] < 0 or r["aapc_lo"] > 0),
        } for r in se},
    }
    return trends, pl.concat(series), verdict


# ---------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    rule("GUARDS")
    check_ci_helpers()
    d = load()
    report["reconciliation"] = reconcile(d)
    report["token_alignment"] = validate_token_alignment(d)

    rule("1. WHAT ARE THE 601? Position on the certificate, and underlying cause")
    pos, table, est_tab, summ = anatomy(d)
    pos.write_csv(OUT / "01_mention_only_position.csv")
    table.write_csv(OUT / "02_mention_only_cause_by_position.csv")
    est_tab.write_csv(OUT / "02b_estimand_partition.csv")
    report["anatomy"] = summ
    print(f"\n  {summ['n_mention_only']} deaths mention A27 but are not coded to "
          f"it; {summ['distinct_underlying_codes']} distinct underlying causes")
    for r in pos.iter_rows(named=True):
        print(f"    {r['position']:<44} n={r['n']:>4}")
    print(f"\n  share whose underlying cause is a recognised leptospirosis "
          f"sequela or an ill-defined code:")
    for k in ["1 Part I causal sequence", "2 Part II contributing",
              "3 original underlying only (recoded away)", "all mention-only"]:
        v = summ[k]
        print(f"    {k:<44} {v['sequela_n']:>4}/{v['n']:<4} = "
              f"{v['sequela_pct']:>5.1f}% ({v['sequela_lo']:.1f}-{v['sequela_hi']:.1f})")
    print(f"    unclassified by the rule table: {summ['unclassified_n']}")
    print("\n  top underlying-cause groups among all mention-only deaths:")
    for r in table.filter(pl.col("position") == "all mention-only").head(12).iter_rows(named=True):
        flag = "sequela" if r["is_sequela"] else "       "
        print(f"    {r['cause_group']:<46} {r['n']:>4}  {r['pct_of_position']:>5.1f}%  {flag}")

    print("\n  THE PARTITION THAT DECIDES THE ESTIMAND:")
    for r in est_tab.filter(pl.col("position") == "all mention-only").iter_rows(named=True):
        print(f"    {r['estimand_class']:<54} {r['n']:>4}  {r['pct_of_position']:>5.1f}%")
    print("\n  ...crossed with where the mention sits:")
    print(f"    {'position':<44}{'downstrm':>9}{'competing':>10}{'antecedent':>11}")
    for p in ["1 Part I causal sequence", "2 Part II contributing",
              "3 original underlying only (recoded away)"]:
        cells = []
        for cls, _ in list(ESTIMAND_PARTITION) + [
                ("antecedent or unrelated condition (estimand change)", None)]:
            f = est_tab.filter((pl.col("position") == p)
                               & (pl.col("estimand_class") == cls))
            cells.append(int(f["n"][0]) if f.height else 0)
        print(f"    {p:<44}{cells[0]:>9}{cells[1]:>10}{cells[2]:>11}")

    rule("1b. Which way does the recoding machinery move leptospirosis?")
    flow, fsum = recode_flow(d)
    flow.write_csv(OUT / "03_recode_flow.csv")
    report["recode_flow"] = fsum
    print(f"  recoded INTO A27 (CAUSABAS=A27, original was not): "
          f"{fsum['recoded_into_a27']}")
    print(f"  recoded AWAY from A27 (original was A27, CAUSABAS is not): "
          f"{fsum['recoded_away_from_a27']}")
    print(f"  NET gain to the underlying-cause count: "
          f"{fsum['net_gain_to_the_underlying_count']:+d}")
    print("\n  what the machinery converts INTO A27 (original cause):")
    for r in flow.filter(pl.col("flow").str.starts_with("recoded INTO")).head(8).iter_rows(named=True):
        print(f"    {r['cause_group']:<46} {r['n']:>4}  {r['pct_of_flow']:>5.1f}%")

    rule("1c. PLAUSIBILITY: is any block a local coding habit rather than a signal?")
    conc = concentration(d)
    conc.write_csv(OUT / "04_mention_only_concentration.csv")
    print(f"  {'block':<7}{'n':>5}{'UFs':>5}{'top UF':>8}{'n':>5}{'%':>7}   cause group")
    for r in conc.iter_rows(named=True):
        print(f"  {r['block']:<7}{r['n']:>5}{r['n_ufs']:>5}{r['top_uf']:>8}"
              f"{r['top_uf_n']:>5}{r['pct_in_top_uf']:>7.1f}   {r['cause_group']}")

    rule("1d. NAMED THREAT: the Part II field is filling up across the series")
    yr, tsum = completeness_threat(d)
    yr.write_csv(OUT / "05_field_completeness_by_year.csv")
    report["completeness_threat"] = tsum
    print(f"  {'year':>5}{'certs':>7}{'LINHAII filled':>16}{'mention-only':>14}")
    for r in yr.iter_rows(named=True):
        print(f"  {r['year']:>5}{r['certificates']:>7}"
              f"{100 * r['linhaii_share']:>15.1f}%{r['mention_only']:>14}")
    for t in tsum["trends"]:
        print(f"\n  {t['marker']}\n    OR/yr {t['or_year']:.4f} "
              f"({t['or_lo']:.4f}, {t['or_hi']:.4f})  MK p={t['mk_p']:.4f}")

    rule("2. IS THE UNDERCOUNT DIFFERENTIAL ACROSS THE DEPTH GRADIENT?")
    g, geo_audit = geolocate(d)
    report["geography_audit"] = geo_audit
    band_tab = mention_share_by_band(g)
    band_tab.write_csv(OUT / "06_mention_only_by_depth_band.csv")
    print(f"\n  {'Q':>2}{'certs':>8}{'underly':>9}{'ment-only':>11}"
          f"{'%':>8}{'95% CI':>16}   {'+PartI':>7}{'+PartII':>8}{'+recoded':>9}")
    for r in band_tab.iter_rows(named=True):
        ci = "({:.2f}-{:.2f})".format(r["mention_only_lo"], r["mention_only_hi"])
        print(f"  {r['quintile']:>2}{r['certificates']:>8}{r['underlying']:>9}"
              f"{r['mention_only']:>11}{r['mention_only_pct']:>8.2f}{ci:>16}"
              f"   {r['add_part1']:>7}{r['add_part2']:>8}{r['add_recoded_away']:>9}")
    q1 = band_tab.filter(pl.col("quintile") == 1).to_dicts()[0]
    q5 = band_tab.filter(pl.col("quintile") == 5).to_dicts()[0]
    overlap = (q1["mention_only_lo"] <= q5["mention_only_hi"]
               and q5["mention_only_lo"] <= q1["mention_only_hi"])
    tr = qbinom_trend(band_tab["quintile"], band_tab["mention_only"],
                      band_tab["certificates"])
    print(f"\n  Q1 (deepest) {q1['mention_only_pct']:.2f}% vs "
          f"Q5 (shallowest) {q5['mention_only_pct']:.2f}%; intervals overlap: {overlap}")
    print(f"  odds ratio per band step (quasi-binomial): {tr['or_year']:.4f} "
          f"({tr['or_lo']:.4f}, {tr['or_hi']:.4f})")
    report["band_gradient"] = {"table": band_tab.to_dicts(),
                               "q1_vs_q5_intervals_overlap": overlap,
                               "or_per_band_step": tr}
    reg = mention_share_by_region(g)
    reg.write_csv(OUT / "07_mention_only_by_region.csv")
    print("\n  by macro-region:")
    for r in reg.iter_rows(named=True):
        print(f"    {r['region']:<14} {r['mention_only']:>4}/{r['certificates']:<6} "
              f"{r['mention_only_pct']:>5.2f}% ({r['lo']:.2f}-{r['hi']:.2f})")

    wr, wrsum = band_gradient_within_region(g)
    wr.write_csv(OUT / "07b_band_gradient_within_region.csv")
    report["band_gradient_within_region"] = wrsum
    print("\n  THREAT: the band contrast is macro-region composition. Refit "
          "inside each region:")
    for r in wr.iter_rows(named=True):
        if r.get("note"):
            print(f"    {r['region']:<14} bands={r['bands_present']}  {r['note']}")
            continue
        print(f"    {r['region']:<14} bands={r['bands_present']} "
              f"n={r['certificates']:>5}  Q{r['deepest_band_present']} "
              f"{r['deepest_band_pct']:>5.2f}% -> Q{r['shallowest_band_present']} "
              f"{r['shallowest_band_pct']:>5.2f}%  "
              f"OR/step {r['or_per_band_step']:.3f} "
              f"({r['or_lo']:.3f}, {r['or_hi']:.3f})"
              f"{'' if r['interval_credible'] else '  <- dispersion %.2f < 1, interval not credible' % r['dispersion']}")

    rule("3. RECOMPUTED SIM QUANTITIES UNDER EVERY DEFINITION")
    pt_hr, pt_nat = person_years()
    nat = national_rates(g, pt_nat)
    nat.write_csv(OUT / "08_national_rate_by_definition.csv")
    report["national_rate"] = nat.to_dicts()
    print(f"\n  national A27 mortality rate, {SIM_YEARS[0]}-{SIM_YEARS[1]}, "
          f"{pt_nat / 1e6:.1f}M person-years")
    for r in nat.iter_rows(named=True):
        print(f"    {r['definition']:<14} deaths={r['deaths']:>5} "
              f"({r['pct_above_underlying']:+5.1f}%)  "
              f"rate={r['rate_per_100k_py']:.4f} "
              f"({r['rate_lo']:.4f}-{r['rate_hi']:.4f}) per 100k py")

    bandr = rates_by_band(g, pt_hr)
    bandr.write_csv(OUT / "09_rate_by_band_and_definition.csv")
    report["rate_by_band"] = bandr.to_dicts()
    print("\n  A27 mortality rate per 100k person-years by depth band:")
    print(f"  {'Q':>2}", end="")
    for name in DEFS:
        print(f"{name:>21}", end="")
    print()
    for q in range(1, 6):
        print(f"  {q:>2}", end="")
        for name in DEFS:
            r = bandr.filter((pl.col("definition") == name)
                             & (pl.col("quintile") == q)).to_dicts()[0]
            cell = "{:.3f} (n={})".format(r["rate_per_100k_py"], r["deaths"])
            print(f"{cell:>21}", end="")
        print()
    print("\n  what the definition change does INSIDE each band "
          "(rate ratio vs underlying-cause):")
    print(f"  {'Q':>2}", end="")
    for name in DEFS[1:]:
        print(f"{name:>16}", end="")
    print()
    for q in range(1, 6):
        print(f"  {q:>2}", end="")
        for name in DEFS[1:]:
            r = bandr.filter((pl.col("definition") == name)
                             & (pl.col("quintile") == q)).to_dicts()[0]
            print(f"{r['ratio_to_underlying']:>16.4f}", end="")
        print()

    print("\n  Q5/Q1 rate ratio (aggregate primitives, then the ratio):")
    for name in DEFS:
        s = bandr.filter(pl.col("definition") == name).sort("quintile")
        r1, r5 = s["rate_per_100k_py"][0], s["rate_per_100k_py"][4]
        print(f"    {name:<14} {r5 / r1:.3f}")
        report.setdefault("q5_q1_rate_ratio", {})[name] = float(r5 / r1)

    rule("3b. THE LINK-7 NUMBER: Southeast SIM AAPC 2008-2024")
    se_series = (pl.read_parquet(TRI)
                 .filter(pl.col("sim_covered") & (pl.col("region") == "Sudeste")
                         & pl.col("year").is_between(*COMMON))
                 .group_by("year").agg(pl.col("sim_a27_deaths").sum().alias("events"),
                                       pl.col("population").sum().cast(pl.Float64)
                                       .alias("person_time")).sort("year"))
    report["r_validation"] = validate_against_r(se_series)

    trends, series, verdict = southeast_trend(g)
    trends.write_csv(OUT / "10_aapc_by_definition.csv")
    series.write_csv(OUT / "11_annual_series_by_definition.csv")
    report["southeast_trend"] = verdict
    print(f"\n  published Sudeste SIM AAPC (underlying cause): -1.95%/yr "
          f"(-3.44, -0.43)")
    for terr in ["Sudeste", "Brazil", "Brazil excl. Sudeste"]:
        print(f"\n  {terr}:")
        for r in trends.filter(pl.col("territory") == terr).iter_rows(named=True):
            excl = "excludes 0" if (r["aapc_hi"] < 0 or r["aapc_lo"] > 0) else "covers 0"
            print(f"    {r['definition']:<14} deaths={int(r['events']):>5}  "
                  f"AAPC {r['aapc']:+6.2f}%/yr ({r['aapc_lo']:+6.2f}, "
                  f"{r['aapc_hi']:+6.2f})  disp={r['dispersion']:.2f}  "
                  f"MK p={r['mk_p']:.4f}  {excl}")

    rule("3c. Is the AAPC shift epidemiology, or the Part II field filling up?")
    sup, supsum = supplement_trend(g)
    sup.write_csv(OUT / "12_supplement_ratio_trends.csv")
    report["supplement_trend"] = supsum
    print("  ratio model: annual change in the supplement PER underlying-cause "
          "death.\n  Zero under the null that mentions are a fixed fraction of "
          "leptospirosis deaths.\n")
    for terr in ["Sudeste", "Brazil"]:
        print(f"  {terr}:")
        for r in sup.filter(pl.col("territory") == terr).iter_rows(named=True):
            excl = "excludes 0" if (r["aapc_hi"] < 0 or r["aapc_lo"] > 0) else "covers 0"
            print(f"    {r['component']:<38} n={int(r['events']):>4}  "
                  f"{r['aapc']:+6.2f}%/yr ({r['aapc_lo']:+6.2f}, {r['aapc_hi']:+6.2f})"
                  f"  {excl:<11} source field: "
                  f"{r['source_field_completeness_trend']}")
        print()

    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
