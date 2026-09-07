#!/usr/bin/env Rscript
# Descriptive tables for the manuscript.
#
# Every number the paper states about the study population, its burden, its
# demography, its seasonality, its concentration and its data quality is
# computed here, from two inputs and nothing else:
#
#   data/interim/lept_line_level.parquet   328,984 notifications, 2007-2025
#   data/interim/population_tensor_long.parquet  age x sex x municipality x year
#
# The estimation is done by R/11_tables.R, which knows nothing about
# leptospirosis. This script is the study-specific part: which records are
# cases, which field means death, how SINAN encodes an age.
#
# Usage: Rscript studies/leptospirosis/22_paper_descriptives.R

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({ library(data.table); library(arrow) })
for (f in c("00_io.R", "01_descriptive.R", "11_tables.R")) source(file.path("R", f))

OUT <- "data/results/01_descriptive"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
w <- function(x, name) {
  fwrite(x, file.path(OUT, name)); cat("  wrote", name, "-", nrow(x), "rows\n")
}
rule <- function(s) cat("\n", strrep("-", 70), "\n", s, "\n", sep = "")

# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------
# CLASSI_FIN == confirmado is the case definition throughout the study; it is
# stated here once so every table below shares it. `inconclusivo` and the
# 12,265 unclassified notifications are NOT cases, but they are counted and
# reported, because the ratio of confirmed to notified is itself the
# ascertainment signal RQ3 pursues.
rule("Study population")
line <- as.data.table(read_parquet("data/interim/lept_line_level.parquet"))
# Geography is resolved by the codebook against the IBGE lattice, with the
# notification UF as the ordered fallback when residence is absent.
line[, uf_code := substr(ID_MN_RESI, 1, 2)]
line[is.na(uf_code) | uf_code == "", uf_code := substr(SG_UF_NOT, 1, 2)]
line[, region := municipality_residence_region]
line[is.na(region), region := uf_notification_region]
line[, year := as.integer(src_year)]
line[, onset := as.IDate(DT_SIN_PRI)]
line[, notified := as.IDate(DT_NOTIFIC)]
line[, digitised := as.IDate(DT_DIGITA)]

# Age comes from the codebook, which decodes SINAN's packed UVVV form and
# reports a state. It is deliberately NOT re-parsed here: a study script that
# reimplements a decode will disagree with the codebook the moment either
# changes, and the disagreement will be silent.
line[, age := age_years]
TENSOR_GROUPS <- c("00-04", "05-09", "10-14", "15-19", "20-24", "25-29",
                   "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
                   "60-64", "65-69", "70-74", "75-79", "80+")
line[, age_group := cut(age, breaks = c(seq(0, 80, by = 5), Inf),
                        right = FALSE, labels = TENSOR_GROUPS)]

# The analytic window is defined on ONSET, not on the file year, so that the
# descriptive tables and the modelling panel count the same events. A case
# notified in January 2007 with onset in December 2006 belongs to neither.
WINDOW <- as.IDate(c("2007-01-01", "2025-12-31"))
line[, in_window := !is.na(onset) & onset >= WINDOW[1] & onset <= WINDOW[2]]
line[is.na(onset), in_window := !is.na(notified) &
       notified >= WINDOW[1] & notified <= WINDOW[2]]
line[, is_case := !is.na(classi_fin) & classi_fin == "confirmado" & in_window]
line[, is_death := is_case & !is.na(evolucao) & evolucao == "obito_por_leptospirose"]
line[, outcome_known := is_case & evolucao_state == "valid"]
line[, hospitalised := is_case & !is.na(ate_hosp) & ate_hosp == "sim"]

# The flow must CLOSE, and the earlier version did not (BREPI-034). It applied
# the analytic-window filter to the confirmed row but not to discarded,
# inconclusive or unclassified, so the four classification classes summed to
# 328,833 against 328,984 notifications -- a 151-record hole that a reviewer
# adding up the column would find. The window exclusion is now its own explicit
# cascade row, the classification classes are reported unfiltered so they
# partition the notifications exactly, and both identities are asserted below.
n_confirmed_all <- sum(line$classi_fin == "confirmado", na.rm = TRUE)
n_discarded     <- sum(line$classi_fin == "descartado", na.rm = TRUE)
n_inconclusive  <- sum(line$classi_fin == "inconclusivo", na.rm = TRUE)
n_unclassified  <- sum(is.na(line$classi_fin))
n_out_of_window <- n_confirmed_all - sum(line$is_case)

flow <- data.table(
  step = c("notifications in the SINAN files",
           "  discarded",
           "  inconclusive",
           "  no final classification",
           "  confirmed (CLASSI_FIN = confirmado)",
           "confirmed, excluded: onset outside 2007-2025",
           "CONFIRMED CASES IN THE ANALYTIC WINDOW",
           "  of which outcome known",
           "  of which died of leptospirosis",
           "  of which hospitalised",
           "  of which a usable age",
           "  of which a residence municipality"),
  n = c(nrow(line),
        n_discarded, n_inconclusive, n_unclassified, n_confirmed_all,
        n_out_of_window,
        sum(line$is_case),
        sum(line$outcome_known), sum(line$is_death), sum(line$hospitalised),
        sum(line$is_case & !is.na(line$age_group)),
        sum(line$is_case & !is.na(line$uf_code) & line$uf_code != "")))
flow[, pct_of_notifications := round(100 * n / nrow(line), 2)]

# Two identities, both fatal if violated. A study-flow table that does not
# reconcile is a STROBE defect, and silently shipping one is worse than failing.
stopifnot(
  n_discarded + n_inconclusive + n_unclassified + n_confirmed_all == nrow(line),
  n_confirmed_all - n_out_of_window == sum(line$is_case)
)
cat(sprintf(
  "study flow reconciles: %d + %d + %d + %d = %d notifications; %d - %d = %d in window\n",
  n_discarded, n_inconclusive, n_unclassified, n_confirmed_all, nrow(line),
  n_confirmed_all, n_out_of_window, sum(line$is_case)))
print(flow)
w(flow, "T1_study_flow.csv")

conf <- line[is_case == TRUE]
cat("\nconfirmation ratio overall:",
    sprintf("%.1f%%", 100 * nrow(conf) / nrow(line)), "\n")

# ---------------------------------------------------------------------------
# Denominators
# ---------------------------------------------------------------------------
pop <- as.data.table(read_parquet("data/interim/population_tensor_long.parquet"))
pop[, uf_code := substr(munic_code, 1, 2)]
pop[, region := macro_region(uf_code)]
pop <- pop[year %in% unique(line$year)]
# The tensor is a mid-year stock; person-months over a calendar year is that
# stock times twelve, which is the same offset the models use.
pop[, person_months := population * 12]

STANDARD <- rebase_standard(WHO_WORLD_STANDARD, TENSOR_GROUPS)
cat("\nstandard population: WHO World Standard, collapsed to the tensor's",
    length(TENSOR_GROUPS), "age groups\n")

# ---------------------------------------------------------------------------
# Burden: national, by region, by year
# ---------------------------------------------------------------------------
rule("Burden")
num_reg <- conf[!is.na(region), .(cases = .N,
                                   deaths = sum(is_death),
                                   outcome_known = sum(outcome_known),
                                   hospitalised = sum(hospitalised)),
                 by = .(region, year)]
den_reg <- pop[!is.na(region), .(person_months = sum(person_months)),
               by = .(region, year)]
burden_reg <- merge(num_reg, den_reg, by = c("region", "year"), all = TRUE)
burden_reg[is.na(cases), `:=`(cases = 0, deaths = 0, outcome_known = 0,
                              hospitalised = 0)]

# Sudeste (Southeast) is the reference: it holds the largest share of the
# population and is where the metropolitan-flood transmission archetype is
# conventionally described, so the contrast a reader wants is "relative to the
# Southeast". The literal is the CANONICAL IBGE label; presentation may
# translate it via `region_en`.
rt_region <- rate_table(burden_reg, by = "region", numerator = "cases",
                        reference = "Sudeste")
w(rt_region, "T2a_incidence_by_region.csv")
print(rt_region[, .(region, events, rate = round(rate, 2),
                    lo = round(lo, 2), hi = round(hi, 2),
                    RR = round(rr, 2), rr_lo = round(rr_lo, 2),
                    rr_hi = round(rr_hi, 2))])

rt_year <- rate_table(burden_reg, by = "year", numerator = "cases")
w(rt_year, "T2b_incidence_by_year.csv")

cfr_region <- proportion_table(burden_reg, by = "region", numerator = "deaths",
                               denominator = "outcome_known")
cfr_all_region <- proportion_table(burden_reg, by = "region",
                                   numerator = "deaths", denominator = "cases")
cfr <- merge(cfr_region, cfr_all_region[, .(region, p_all = p)], by = "region")
cfr[, outcome_known_share := outcome_known / (cfr_all_region$cases[match(region, cfr_all_region$region)])]
w(cfr, "T2c_cfr_by_region.csv")
cat("\ncase fatality by region (known-outcome denominator, and all-case):\n")
print(cfr[, .(region, deaths, outcome_known,
              cfr_known = round(100 * p, 2), lo = round(100 * lo, 2),
              hi = round(100 * hi, 2), cfr_all = round(100 * p_all, 2),
              known_share = round(100 * outcome_known_share, 1))])

hosp <- proportion_table(burden_reg, by = "region", numerator = "hospitalised",
                         denominator = "cases")
w(hosp, "T2d_hospitalisation_by_region.csv")

# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------
rule("Trend")
trend_nat <- trend_apc(burden_reg, numerator = "cases")
trend_reg <- trend_apc(burden_reg, by = "region", numerator = "cases")
trend_excl <- trend_apc(burden_reg[!year %in% c(2020L, 2021L)], numerator = "cases")
trend <- rbind(cbind(scope = "national", trend_nat),
               cbind(scope = "national, excluding 2020-21", trend_excl),
               cbind(scope = paste("region:", trend_reg$region),
                     trend_reg[, !"region"]))
w(trend, "T3_trend.csv")
print(trend[, .(scope, aapc = round(aapc, 2), lo = round(aapc_lo, 2),
                hi = round(aapc_hi, 2), dispersion = round(dispersion, 1),
                mk_tau = round(mk_tau, 3), mk_p = signif(mk_p, 2))])

# ---------------------------------------------------------------------------
# Age and sex
# ---------------------------------------------------------------------------
rule("Age and sex")
line[, sex := fifelse(cs_sexo == "masculino", "male",
                      fifelse(cs_sexo == "feminino", "female", NA_character_))]
num_as <- line[is_case == TRUE & !is.na(age_group) & !is.na(sex),
               .(cases = .N, deaths = sum(is_death)), by = .(sex, age_group, year)]
den_as <- pop[, .(person_months = sum(person_months)), by = .(sex, age_group, year)]
as_dt <- merge(num_as, den_as, by = c("sex", "age_group", "year"), all.y = TRUE)
as_dt[is.na(cases), `:=`(cases = 0, deaths = 0)]

rt_age <- rate_table(as_dt, by = c("sex", "age_group"), numerator = "cases")
w(rt_age, "T4a_rate_by_age_sex.csv")

rt_sex <- rate_table(as_dt, by = "sex", numerator = "cases", reference = "female")
w(rt_sex, "T4b_rate_by_sex.csv")
cat("\nincidence by sex (per 100,000 person-years):\n")
print(rt_sex[, .(sex, events, rate = round(rate, 2), lo = round(lo, 2),
                 hi = round(hi, 2), RR = round(rr, 2),
                 rr_lo = round(rr_lo, 2), rr_hi = round(rr_hi, 2))])

asr_year <- age_standardise(as_dt, age_col = "age_group", by = "year",
                            standard = STANDARD)
w(asr_year, "T4c_asr_by_year.csv")
cat("\nage-standardised incidence, first and last five years:\n")
print(rbind(head(asr_year, 5), tail(asr_year, 5))[
  , .(year, events, crude = round(crude, 2), asr = round(asr, 2),
      lo = round(lo, 2), hi = round(hi, 2))])

# Region needs the region on both sides of the join.
num_asr <- line[is_case == TRUE & !is.na(age_group) & !is.na(region),
                .(cases = .N), by = .(region, age_group)]
den_asr <- pop[!is.na(region), .(person_months = sum(person_months)),
               by = .(region, age_group)]
asr_in <- merge(num_asr, den_asr, by = c("region", "age_group"), all.y = TRUE)
asr_in[is.na(cases), cases := 0]
asr_region <- age_standardise(asr_in, age_col = "age_group", by = "region",
                              standard = STANDARD)
w(asr_region, "T4d_asr_by_region.csv")
cat("\nage-standardised vs crude incidence by region:\n")
print(asr_region[, .(region, crude = round(crude, 2), asr = round(asr, 2),
                     lo = round(lo, 2), hi = round(hi, 2),
                     ratio = round(asr / crude, 3))])

# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------
rule("Seasonality")
conf[, month := as.integer(format(onset, "%m"))]
conf[is.na(month), month := as.integer(format(notified, "%m"))]
seas_nat <- circular_season(conf[!is.na(month)], numerator = NULL)
seas_reg <- circular_season(conf[!is.na(month) & !is.na(region)],
                            by = "region", numerator = NULL)
seas <- rbind(cbind(scope = "national", seas_nat),
              cbind(scope = seas_reg$region, seas_reg[, !"region"]))
w(seas, "T5_seasonality.csv")
print(seas[, .(scope, peak_month, peak_frac = round(peak_month_frac, 2),
               r = round(resultant_r, 4), cases)])
cat("\n`r` is the interpretable quantity: 0 is a flat year, 1 is one month.\n")

# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------
rule("Concentration of burden across municipalities")
panel <- read_panel(
  "data/panel/lept_panel_rq1_socioeconomic_municipality_month.parquet",
  validate = TRUE, prepare = TRUE)
conc <- concentration(panel, unit = "munic_code")
w(conc$summary, "T6a_concentration.csv")
w(conc$lorenz, "T6b_lorenz.csv")
print(conc$summary[1, .(units, units_with_zero, zero_share = round(zero_share, 4),
                        gini = round(gini_cases, 4))])
print(conc$summary[, .(metric, value = round(value, 4))])

# ---------------------------------------------------------------------------
# Data quality: completeness of the coded fields, over time
# ---------------------------------------------------------------------------
rule("Field completeness")
# Assessed on the RAW codes, not the decoded labels: a decoded label cannot
# distinguish a blank field from a code the dictionary does not contain, and
# that distinction is the point.
QUALITY_FIELDS <- c("CLASSI_FIN", "CRITERIO", "EVOLUCAO", "ATE_HOSP",
                    "CS_SEXO", "CS_RACA", "CS_ESCOL_N", "ID_OCUPA_N",
                    "TPAUTOCTO", "CON_AMBIEN", "DOENCA_TRA")
UNKNOWN <- list(
  CLASSI_FIN = character(), CRITERIO = character(),
  EVOLUCAO = "9", ATE_HOSP = "9", CS_SEXO = "I", CS_RACA = "9",
  CS_ESCOL_N = c("9", "09", "10"), ID_OCUPA_N = c("999991", "999992", "999993"),
  TPAUTOCTO = "3", CON_AMBIEN = "9", DOENCA_TRA = "9"
)
comp_year <- completeness_table(line, fields = QUALITY_FIELDS, by = "year",
                               unknown_values = UNKNOWN)
w(comp_year, "T7a_completeness_by_year.csv")

comp_all <- completeness_table(line, fields = QUALITY_FIELDS,
                               unknown_values = UNKNOWN)
w(comp_all, "T7b_completeness_overall.csv")
cat("\ncompleteness over the whole period (% of 328,984 notifications):\n")
print(comp_all[order(-share_missing),
               .(field, valid = round(100 * share_valid, 1),
                 missing = round(100 * share_missing, 1),
                 unknown = round(100 * share_unknown, 1))])

# Drift is the dangerous property: a field whose completeness moves will
# manufacture a trend in anything stratified by it.
drift <- comp_year[, .(first = share_valid[which.min(year)],
                       last = share_valid[which.max(year)],
                       min = min(share_valid), max = max(share_valid)),
                   by = field]
drift[, swing_pp := round(100 * (max - min), 1)]
setorder(drift, -swing_pp)
w(drift, "T7c_completeness_drift.csv")
cat("\ncompleteness drift, 2007 to 2025 (percentage points of swing):\n")
print(drift[, .(field, first = round(100 * first, 1), last = round(100 * last, 1),
                swing_pp)])

# ---------------------------------------------------------------------------
# Clinical and exposure profile of confirmed cases
# ---------------------------------------------------------------------------
rule("Clinical syndrome and exposure antecedents (confirmed cases)")
# Denominator is records where the field was actually answered. Dividing by all
# confirmed cases would report "42% had jaundice" when the truth is "42% of the
# 70% who were asked".
profile_block <- function(prefix, label) {
  cols <- grep(paste0("^", prefix), names(conf), value = TRUE)
  cols <- cols[!grepl("_state$", cols)]
  cols <- cols[vapply(cols, function(c) is.character(conf[[c]]), logical(1))]
  rbindlist(lapply(cols, function(c) {
    st <- paste0(c, "_state")
    answered <- if (st %in% names(conf)) conf[[st]] == "valid" else !is.na(conf[[c]])
    yes <- answered & conf[[c]] == "sim"
    ci <- binom_ci(sum(yes, na.rm = TRUE), sum(answered, na.rm = TRUE))
    data.table(block = label, item = c, answered = sum(answered, na.rm = TRUE),
               yes = sum(yes, na.rm = TRUE),
               answered_share = sum(answered, na.rm = TRUE) / nrow(conf),
               p = ci$p, lo = ci$lo, hi = ci$hi)
  }))
}
prof <- rbind(profile_block("cli_", "clinical sign"),
              profile_block("ant_", "exposure antecedent"))
setorder(prof, block, -p)
w(prof, "T8_clinical_exposure_profile.csv")
print(prof[, .(block, item, answered, yes, pct = round(100 * p, 1),
               lo = round(100 * lo, 1), hi = round(100 * hi, 1),
               asked_pct = round(100 * answered_share, 1))])

# ---------------------------------------------------------------------------
# Confirmation criterion
# ---------------------------------------------------------------------------
rule("Confirmation criterion among confirmed cases")
crit <- conf[!is.na(criterio), .N, by = .(year, criterio)]
crit_w <- dcast(crit, year ~ criterio, value.var = "N", fill = 0)
crit_w[, total := clinico_laboratorial + clinico_epidemiologico]
cb <- binom_ci(crit_w$clinico_laboratorial, crit_w$total)
crit_w[, `:=`(lab_share = cb$p, lab_lo = cb$lo, lab_hi = cb$hi)]
w(crit_w, "T9_confirmation_criterion.csv")
print(crit_w[, .(year, lab = clinico_laboratorial, epi = clinico_epidemiologico,
                 lab_pct = round(100 * lab_share, 1))])

# ---------------------------------------------------------------------------
# Notification delay
# ---------------------------------------------------------------------------
rule("Notification delay")
dl <- line[!is.na(onset) & !is.na(notified) & is_case == TRUE]
dl[, delay_days := as.numeric(notified - onset)]
dl <- dl[delay_days >= 0 & delay_days <= 365]
delays <- dl[, .(munic_code = ID_MN_RESI,
                 date = as.Date(paste0(format(onset, "%Y-%m"), "-01")),
                 delay_days)]
nd <- notification_delay(delays)
w(nd$overall, "T10a_delay_overall.csv")
w(nd$by_year, "T10b_delay_by_year.csv")
cat("\nonset-to-notification delay (days), confirmed cases:\n")
print(nd$overall)
cat("\nby year:\n"); print(nd$by_year)
cat("\ndropped as out of range (negative or > 365 days):",
    nrow(line[is_case == TRUE & !is.na(onset) & !is.na(notified)]) - nrow(dl), "\n")

dg <- line[!is.na(notified) & !is.na(digitised) & is_case == TRUE]
dg[, lag_days := as.numeric(digitised - notified)]
dg <- dg[lag_days >= 0 & lag_days <= 365]
dig <- dg[, .(p50 = median(lag_days), p90 = quantile(lag_days, 0.9),
              p95 = quantile(lag_days, 0.95), n = .N), by = year]
setorder(dig, year)
w(dig, "T10c_digitisation_lag_by_year.csv")

# ---------------------------------------------------------------------------
# Serovar, serogroup and inferred reservoir
# ---------------------------------------------------------------------------
rule("Microscopic agglutination test: serovar and reservoir")
# The MAT serovar is the only field in the record that speaks to the
# TRANSMISSION SOURCE. It was untranslated until the codebook learned to read
# the self-labelling form, so this is the first time it enters the study.
#
# Three caveats travel with every number below and belong in the caption:
#  1. MAT identifies the serogroup reliably and the serovar only presumptively;
#     cross-reaction within a serogroup is the rule, not the exception.
#  2. The reservoir is an INFERENCE from the serovar's usual maintenance host,
#     not an observation of an animal.
#  3. Patoc is Leptospira biflexa and saprophytic -- a screening antigen. A
#     Patoc-only reaction is NOT evidence of a specific pathogenic serovar and
#     is reported separately rather than attributed to any reservoir.
sero <- conf[!is.na(serovar_s1_first)]
cat("confirmed cases with a MAT serovar result:", nrow(sero),
    sprintf("(%.1f%% of %s confirmed)", 100 * nrow(sero) / nrow(conf),
            format(nrow(conf), big.mark = ",")), "\n")

sv <- sero[, .(cases = .N), by = .(serovar = serovar_s1_first,
                                   serogroup = serovar_s1_first_serogroup,
                                   reservoir = serovar_s1_first_reservoir)]
ci <- binom_ci(sv$cases, nrow(sero))
sv[, `:=`(share = ci$p, lo = ci$lo, hi = ci$hi)]
setorder(sv, -cases)
w(sv, "T11a_serovar_profile.csv")
print(sv[, .(serovar, serogroup, cases, pct = round(100 * share, 1),
             lo = round(100 * lo, 1), hi = round(100 * hi, 1))])

res <- sero[, .(cases = .N), by = .(reservoir = serovar_s1_first_reservoir)]
rci <- binom_ci(res$cases, nrow(sero))
res[, `:=`(share = rci$p, lo = rci$lo, hi = rci$hi)]
setorder(res, -cases)
w(res, "T11b_reservoir_attribution.csv")
cat("\ninferred maintenance host (presumptive; see caveats):\n")
print(res[, .(reservoir, cases, pct = round(100 * share, 1),
              lo = round(100 * lo, 1), hi = round(100 * hi, 1))])

# Does the reservoir mix differ by region? This is the testable form of the
# metropolitan-rat versus rural-livestock contrast that RQ5 clusters on
# outcomes alone.
sero_reg <- sero[!is.na(region), .(cases = .N),
                 by = .(region, reservoir = serovar_s1_first_reservoir)]
sero_reg[, share := cases / sum(cases), by = region]
w(sero_reg, "T11c_reservoir_by_region.csv")
rat <- sero[!is.na(region), .(
  n = .N, rattus = sum(serovar_s1_first_reservoir == "rodent_rattus")), by = region]
rc <- binom_ci(rat$rattus, rat$n)
rat[, `:=`(share = rc$p, lo = rc$lo, hi = rc$hi)]
setorder(rat, -share)
w(rat, "T11d_rattus_share_by_region.csv")
cat("\nshare of serotyped cases attributed to a Rattus reservoir, by region:\n")
print(rat[, .(region, n, rattus, pct = round(100 * share, 1),
              lo = round(100 * lo, 1), hi = round(100 * hi, 1))])

# Serovar profile over time: a shift would indicate changing transmission
# ecology rather than changing incidence.
sero_yr <- sero[, .(n = .N,
                    rattus = sum(serovar_s1_first_reservoir == "rodent_rattus"),
                    livestock = sum(serovar_s1_first_reservoir %in%
                                    c("cattle", "cattle_rodent", "swine",
                                      "swine_cattle", "swine_equine_hedgehog"))),
                by = year]
setorder(sero_yr, year)
sero_yr[, `:=`(rattus_share = rattus / n, livestock_share = livestock / n)]
w(sero_yr, "T11e_reservoir_by_year.csv")

# ---------------------------------------------------------------------------
# Occupation
# ---------------------------------------------------------------------------
rule("Occupational profile")
# CBO major group only. The authoritative six-digit title list is a separate
# download; the first digit is the major group by definition of the
# classification, so this is what the code structurally guarantees and no more.
# The denominator is cases with a USABLE occupation: 50.2% of the field is
# blank and a further 17.7% is an explicit "not informed" sentinel, so a
# percentage over all confirmed cases would be a statement about the form, not
# about the workforce.
occ <- conf[!is.na(occupation_major_group)]
cat("confirmed cases with a usable occupation:", nrow(occ),
    sprintf("(%.1f%%)", 100 * nrow(occ) / nrow(conf)), "\n")
cat("occupation field state among confirmed cases:\n")
print(conf[, .N, by = occupation_major_group_state][order(-N)])

ot <- occ[, .(cases = .N), by = .(major_group = occupation_major_group)]
oci <- binom_ci(ot$cases, nrow(occ))
ot[, `:=`(share = oci$p, lo = oci$lo, hi = oci$hi)]
setorder(ot, -cases)
w(ot, "T12a_occupation_major_group.csv")
print(ot[, .(major_group, cases, pct = round(100 * share, 1),
             lo = round(100 * lo, 1), hi = round(100 * hi, 1))])

# Case fatality by occupational group, on the known-outcome denominator.
occ_cfr <- occ[outcome_known == TRUE, .(known = .N, deaths = sum(is_death)),
               by = .(major_group = occupation_major_group)]
occ_cfr <- occ_cfr[known >= 100]
ccc <- binom_ci(occ_cfr$deaths, occ_cfr$known)
occ_cfr[, `:=`(cfr = ccc$p, lo = ccc$lo, hi = ccc$hi)]
setorder(occ_cfr, -cfr)
w(occ_cfr, "T12b_cfr_by_occupation.csv")
cat("\ncase fatality by occupational major group (>=100 known outcomes):\n")
print(occ_cfr[, .(major_group, known, deaths, cfr_pct = round(100 * cfr, 2),
                  lo = round(100 * lo, 2), hi = round(100 * hi, 2))])

# Rural occupations against the rest, as a single interpretable contrast.
occ[, rural := occupation_major_group ==
      "trabalhadores_agropecuarios_florestais_e_da_pesca"]
rural_reg <- occ[!is.na(region), .(n = .N, rural = sum(rural)), by = region]
rr <- binom_ci(rural_reg$rural, rural_reg$n)
rural_reg[, `:=`(share = rr$p, lo = rr$lo, hi = rr$hi)]
setorder(rural_reg, -share)
w(rural_reg, "T12c_rural_occupation_by_region.csv")
cat("\nshare of occupationally-coded cases in agriculture/forestry/fishing:\n")
print(rural_reg[, .(region, n, rural, pct = round(100 * share, 1),
                    lo = round(100 * lo, 1), hi = round(100 * hi, 1))])

rule("done")
cat("all tables written to", OUT, "\n")
