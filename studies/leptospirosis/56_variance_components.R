# Variance components and hospital-stratified conditional logistic for
# 56_hospital_vs_territory.py. Called with (analysis_frame.parquet, out_dir).
#
# Two model families:
#   * crossed / nested random-intercept logistic (lme4), to say where the
#     variance in in-hospital fatality and ICU use sits -- between hospitals or
#     between health regions. Reported as the intercept variance on the logit
#     scale and as a median odds ratio, which is the odds ratio between two
#     randomly chosen clusters and is readable without a scale in one's head.
#   * conditional logistic stratified by treating hospital (survival::clogit),
#     which is the decisive test: the hospital contributes a nuisance parameter
#     that the conditional likelihood removes, so nothing about the building
#     can carry the estimate.

suppressPackageStartupMessages({
  library(arrow); library(lme4); library(survival); library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
d <- as.data.frame(read_parquet(args[1]))
outdir <- args[2]

d$died <- as.integer(d$died)
d$any_icu <- as.integer(d$any_icu)
d$CNES <- factor(d$CNES)
d$hr_res <- factor(d$hr_res)
d$hr_mov <- factor(d$hr_mov)
d$band_res_f <- factor(d$band_res)
d$age_z <- as.numeric(scale(d$age_years))
d$female <- as.integer(d$female)
d$year_c <- d$year - 2016
d$referred_hr <- as.integer(d$referred_hr)
d$referred_munic <- as.integer(d$referred_munic)
d$log_vol <- log(d$hosp_volume)
d$hosp_ever_icu <- as.integer(d$hosp_ever_icu)
# depth on the natural 0-1 scale, so a coefficient is the log odds ratio across
# the whole national range of hospitalisation share
d$hs_res <- as.numeric(d$hs_res)
d$munic_hosp_share <- as.numeric(d$munic_hosp_share)
d$hs_res_surv <- as.numeric(d$hs_res_surv)
d$munic_hosp_share_surv <- as.numeric(d$munic_hosp_share_surv)

MOR <- function(v) exp(sqrt(2 * v) * qnorm(0.75))

vc <- function(fit) {
  v <- as.data.frame(VarCorr(fit))
  setNames(lapply(seq_len(nrow(v)), function(i) {
    list(variance = v$vcov[i], sd = v$sdcor[i], median_odds_ratio = MOR(v$vcov[i]))
  }), v$grp)
}

fit_glmer <- function(f, data) {
  # nAGQ = 0 keeps a two-way crossed model with ~1,700 hospital levels tractable;
  # the variance components it returns are the quantity of interest, not the
  # fixed effects, and it is stable where Laplace stalls.
  glmer(f, data = data, family = binomial, nAGQ = 0,
        control = glmerControl(optimizer = "bobyqa",
                               optCtrl = list(maxfun = 3e5)))
}

res <- list()
res$n <- nrow(d)
res$hospitals <- nlevels(d$CNES)
res$home_regions <- nlevels(d$hr_res)

for (y in c("died", "any_icu")) {
  cat("\n--", y, "--\n")
  # crossed: the patient's home region and the hospital they reached
  m_cross <- fit_glmer(as.formula(paste(y, "~ 1 + (1|hr_res) + (1|CNES)")), d)
  # nested: hospitals inside the region where they sit
  m_nest <- fit_glmer(as.formula(paste(y, "~ 1 + (1|hr_mov) + (1|CNES)")), d)
  # does the depth band absorb the home-region variance?
  m_band <- fit_glmer(
    as.formula(paste(y, "~ band_res_f + age_z + female + year_c + (1|hr_res) + (1|CNES)")), d)
  # hospital-only and region-only, for the share each level explains alone
  m_hosp <- fit_glmer(as.formula(paste(y, "~ 1 + (1|CNES)")), d)
  m_reg <- fit_glmer(as.formula(paste(y, "~ 1 + (1|hr_res)")), d)

  res[[y]] <- list(
    crossed_home_region_and_hospital = vc(m_cross),
    nested_hospital_in_treatment_region = vc(m_nest),
    crossed_adjusted_for_band = vc(m_band),
    band_fixed_effects = {
      s <- summary(m_band)$coefficients
      lapply(rownames(s), function(r) list(term = r, estimate = s[r, 1],
                                           se = s[r, 2], z = s[r, 3]))
    },
    hospital_only = vc(m_hosp),
    home_region_only = vc(m_reg)
  )
  v <- res[[y]]$crossed_home_region_and_hospital
  # share of total variance on the latent scale; the binomial residual variance
  # on the logit scale is pi^2/3, which is what makes these comparable at all
  tot <- v$CNES$variance + v$hr_res$variance + pi^2 / 3
  res[[y]]$variance_share_pct <- list(
    hospital = 100 * v$CNES$variance / tot,
    home_health_region = 100 * v$hr_res$variance / tot,
    residual = 100 * (pi^2 / 3) / tot,
    hospital_to_region_ratio = v$CNES$variance / v$hr_res$variance
  )
  cat(sprintf("  variance share: hospital %.1f%%, home region %.1f%%, ratio %.2f\n",
              res[[y]]$variance_share_pct$hospital,
              res[[y]]$variance_share_pct$home_health_region,
              res[[y]]$variance_share_pct$hospital_to_region_ratio))
  cat(sprintf("  crossed: hospital var %.4f (MOR %.2f) | home region var %.4f (MOR %.2f)\n",
              v$CNES$variance, v$CNES$median_odds_ratio,
              v$hr_res$variance, v$hr_res$median_odds_ratio))
  vb <- res[[y]]$crossed_adjusted_for_band
  cat(sprintf("  after band:  hospital var %.4f | home region var %.4f (region var explained by band: %.1f%%)\n",
              vb$CNES$variance, vb$hr_res$variance,
              100 * (1 - vb$hr_res$variance / v$hr_res$variance)))
}

# ---- the decisive test: conditional logistic, stratified by hospital --------
clog <- function(f, data, label) {
  fit <- try(clogit(f, data = data), silent = TRUE)
  if (inherits(fit, "try-error")) return(list(label = label, error = as.character(fit)))
  s <- summary(fit)$coefficients
  ci <- try(confint(fit), silent = TRUE)
  out <- list(label = label, n = fit$n, events = fit$nevent,
              terms = lapply(rownames(s), function(r) {
                list(term = r, beta = s[r, 1], or = exp(s[r, 1]), se = s[r, 3],
                     z = s[r, 4], p = s[r, 5],
                     or_lo = exp(s[r, 1] - 1.96 * s[r, 3]),
                     or_hi = exp(s[r, 1] + 1.96 * s[r, 3]))
              }))
  out
}

# the same model without hospital strata, on the same rows: the between-hospital
# estimate the stratified one has to be compared against. Reporting only the
# stratified coefficient invites the reader to compare it with a number from a
# different sample.
plain <- function(f, data, label) {
  fit <- try(glm(f, data = data, family = binomial), silent = TRUE)
  if (inherits(fit, "try-error")) return(list(label = label, error = as.character(fit)))
  s <- summary(fit)$coefficients
  # geographic dummies are nuisance parameters; keeping several hundred of them
  # in the report would bury the three coefficients anyone reads
  rn <- setdiff(rownames(s)[-1], grep("^factor\\(", rownames(s), value = TRUE))
  list(label = label, n = nrow(fit$model), events = sum(fit$y),
       terms = lapply(rn, function(r) {
         list(term = r, beta = s[r, 1], or = exp(s[r, 1]), se = s[r, 2],
              p = s[r, 4], or_lo = exp(s[r, 1] - 1.96 * s[r, 2]),
              or_hi = exp(s[r, 1] + 1.96 * s[r, 2]))
       }))
}

res$conditional_logistic <- list()
for (thr in c(1, 30, 50, 100)) {
  sub <- d[d$hosp_volume >= thr, ]
  # keep only hospitals that actually saw more than one home health region;
  # a single-region hospital contributes no within-stratum contrast
  keep <- names(which(tapply(as.character(sub$hr_res), sub$CNES,
                             function(z) length(unique(z))) > 1))
  sub <- droplevels(sub[as.character(sub$CNES) %in% keep, ])
  cat(sprintf("\nclogit threshold %d: %d hospitals, %d admissions, %d deaths\n",
              thr, nlevels(sub$CNES), nrow(sub), sum(sub$died)))

  key <- paste0("volume_ge_", thr)
  msub <- sub[!is.na(sub$munic_hosp_share) & sub$munic_cases >= 30, ]
  res$conditional_logistic[[key]] <- list(
    hospitals = nlevels(sub$CNES), admissions = nrow(sub), deaths = sum(sub$died),
    died_home_depth = clog(
      died ~ hs_res + age_z + female + year_c + referred_hr + strata(CNES),
      sub, "in-hospital death ~ home health-region hospitalisation share"),
    died_home_depth_pooled = plain(
      died ~ hs_res + age_z + female + year_c + referred_hr, sub,
      "same model, hospital NOT held fixed"),
    died_home_depth_unadjusted = clog(
      died ~ hs_res + strata(CNES), sub, "unadjusted"),
    icu_home_depth = clog(
      any_icu ~ hs_res + age_z + female + year_c + referred_hr + strata(CNES),
      sub, "ICU use ~ home health-region hospitalisation share"),
    icu_home_depth_pooled = plain(
      any_icu ~ hs_res + age_z + female + year_c + referred_hr, sub,
      "same model, hospital NOT held fixed"),
    died_home_depth_municipal = clog(
      died ~ munic_hosp_share + age_z + female + year_c + referred_munic + strata(CNES),
      msub, "in-hospital death ~ home MUNICIPALITY hospitalisation share (>=30 cases)"),
    died_home_depth_municipal_pooled = plain(
      died ~ munic_hosp_share + age_z + female + year_c + referred_munic, msub,
      "same model, hospital NOT held fixed"),
    icu_home_depth_municipal = clog(
      any_icu ~ munic_hosp_share + age_z + female + year_c + referred_munic + strata(CNES),
      msub, "ICU use ~ home MUNICIPALITY hospitalisation share (>=30 cases)"),
    icu_home_depth_municipal_pooled = plain(
      any_icu ~ munic_hosp_share + age_z + female + year_c + referred_munic, msub,
      "same model, hospital NOT held fixed"),
    # sensitivity: depth recomputed among survivors, so a territory's own deaths
    # cannot inflate its hospitalisation share
    died_home_depth_municipal_survivor = clog(
      died ~ munic_hosp_share_surv + age_z + female + year_c + referred_munic +
        strata(CNES),
      sub[!is.na(sub$munic_hosp_share_surv) & sub$surv_cases >= 30, ],
      "in-hospital death ~ SURVIVOR-ONLY home municipality hospitalisation share"),
    died_home_depth_municipal_survivor_pooled = plain(
      died ~ munic_hosp_share_surv + age_z + female + year_c + referred_munic,
      sub[!is.na(sub$munic_hosp_share_surv) & sub$surv_cases >= 30, ],
      "same model, hospital NOT held fixed"),
    icu_home_depth_municipal_survivor = clog(
      any_icu ~ munic_hosp_share_surv + age_z + female + year_c + referred_munic +
        strata(CNES),
      sub[!is.na(sub$munic_hosp_share_surv) & sub$surv_cases >= 30, ],
      "ICU ~ SURVIVOR-ONLY home municipality hospitalisation share"),
    icu_home_depth_municipal_survivor_pooled = plain(
      any_icu ~ munic_hosp_share_surv + age_z + female + year_c + referred_munic,
      sub[!is.na(sub$munic_hosp_share_surv) & sub$surv_cases >= 30, ],
      "same model, hospital NOT held fixed")
  )
  for (nm in c("died_home_depth", "icu_home_depth",
               "died_home_depth_municipal", "icu_home_depth_municipal",
               "died_home_depth_municipal_survivor",
               "icu_home_depth_municipal_survivor")) {
    o <- res$conditional_logistic[[key]][[nm]]
    p <- res$conditional_logistic[[key]][[paste0(nm, "_pooled")]]
    if (!is.null(o$terms)) {
      t1 <- o$terms[[1]]; t0 <- p$terms[[1]]
      cat(sprintf("  %-28s pooled OR %6.2f (%5.2f-%7.2f) | hospital-fixed OR %6.2f (%5.2f-%7.2f)  [n=%d, ev=%d]\n",
                  nm, t0$or, t0$or_lo, t0$or_hi, t1$or, t1$or_lo, t1$or_hi,
                  o$n, o$events))
    }
  }
}

# ---- the mirror test: hospital properties, within home territory ------------
# If the gradient were a hospital property, hospital characteristics should
# predict outcome once the patient's own territory is held fixed. Fitted with
# health-region dummies rather than a conditional likelihood: 233 strata of a
# few hundred observations each make the incidental-parameter bias negligible,
# and clogit does not converge at this stratum size.
res$hospital_properties_within_home_region <- plain(
  died ~ log_vol + hosp_ever_icu + hosp_inflow_share + age_z + female + year_c +
    factor(hr_res), d, "in-hospital death ~ hospital characteristics | home region")
o <- res$hospital_properties_within_home_region
if (!is.null(o$terms)) {
  cat("\nhospital characteristics within home health region:\n")
  for (t in o$terms) {
    if (grepl("^factor", t$term)) next
    cat(sprintf("  %-20s OR %.3f (%.3f-%.3f) p=%.3g\n",
                t$term, t$or, t$or_lo, t$or_hi, t$p))
  }
}

# and the same for the territory, holding the place of treatment fixed:
res$depth_within_treatment_region <- plain(
  died ~ hs_res + age_z + female + year_c + factor(hr_mov), d,
  "in-hospital death ~ home depth | treatment region")
o <- res$depth_within_treatment_region
if (!is.null(o$terms)) {
  t <- o$terms[[1]]
  cat(sprintf("\nhome depth within treatment region: OR %.2f (%.2f-%.2f) p=%.3g\n",
              t$or, t$or_lo, t$or_hi, t$p))
}

write_json(res, file.path(outdir, "r_models.json"),
           auto_unbox = TRUE, digits = 8, null = "null")
cat("\nwrote r_models.json\n")
