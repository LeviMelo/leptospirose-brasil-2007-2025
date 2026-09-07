
suppressMessages(library(INLA))
suppressMessages(library(jsonlite))

args   <- commandArgs(trailingOnly = TRUE)
f_reg  <- args[1]; f_ry <- args[2]; f_out <- args[3]
reg    <- read.csv(f_reg, stringsAsFactors = FALSE)
ry     <- read.csv(f_ry,  stringsAsFactors = FALSE)

# Vague priors on the fixed effects; the iid precision keeps INLA's default
# log-gamma. Nothing here is trying to be informative -- the point of the model
# is partial pooling of small regions, not prior-driven shrinkage of the slope.
CTRL_FIXED <- list(mean = 0, prec = 0.01, mean.intercept = 0, prec.intercept = 0.01)

qsum <- function(m, name, scale) {
  mg <- m$marginals.fixed[[name]]
  q  <- inla.qmarginal(c(0.025, 0.5, 0.975), mg)
  # exp() is monotone, so posterior quantiles of the odds ratio are the
  # exponentiated quantiles of the log-odds coefficient.
  list(or = exp(scale * q[2]), or_lo = exp(scale * q[1]), or_hi = exp(scale * q[3]))
}

sd_of <- function(m, hyp) {
  mg <- m$marginals.hyperpar[[hyp]]
  q  <- inla.qmarginal(c(0.025, 0.5, 0.975), mg)
  # precision -> sd is monotone decreasing, so the quantiles swap ends.
  list(sd = 1/sqrt(q[2]), sd_lo = 1/sqrt(q[3]), sd_hi = 1/sqrt(q[1]))
}

fit_region <- function(df, label) {
  df$id <- seq_len(nrow(df))
  m <- inla(deaths ~ H + f(id, model = "iid"),
            family = "binomial", Ntrials = df$outcome_known, data = df,
            control.fixed = CTRL_FIXED,
            control.compute = list(dic = TRUE, waic = TRUE))
  per1  <- qsum(m, "H", 1.0)
  per10 <- qsum(m, "H", 0.1)
  s     <- sd_of(m, "Precision for id")
  list(model = "region-level, iid region intercept", subset = label,
       n_rows = nrow(df), n_regions = nrow(df),
       cases = sum(df$cases), deaths = sum(df$deaths),
       outcome_known = sum(df$outcome_known),
       person_years = sum(df$person_years),
       intercept = unname(m$summary.fixed["(Intercept)", "mean"]),
       beta_H = unname(m$summary.fixed["H", "mean"]),
       beta_H_sd = unname(m$summary.fixed["H", "sd"]),
       or_per_unit_H = per1$or, or_lo = per1$or_lo, or_hi = per1$or_hi,
       or_per_10pp_H = per10$or, or_10pp_lo = per10$or_lo, or_10pp_hi = per10$or_hi,
       region_sd = s$sd, region_sd_lo = s$sd_lo, region_sd_hi = s$sd_hi,
       dic = m$dic$dic, waic = m$waic$waic)
}

fit_region_year <- function(df, label, xvar) {
  # One row per region-year. Two exposures are available and they answer
  # different questions:
  #
  #   H_region : the region's exposure over the whole window, constant within
  #              region. Because the region intercept is also constant within
  #              region and the binomial is closed under summation at constant
  #              p, this model is algebraically the region-level model plus a
  #              year effect -- it is run once as a lossless-aggregation check,
  #              not as independent evidence.
  #   H_year   : the region-year exposure. This one does not collapse: the
  #              region intercept is now identified from repeated observations
  #              rather than acting as pure binomial overdispersion, and the
  #              slope is driven by within-region movement in H over time.
  df$rid <- as.integer(factor(df$health_region_code))
  df$yid <- as.integer(factor(df$year))
  df$xx  <- df[[xvar]]
  df     <- df[!is.na(df$xx), ]
  m <- inla(deaths ~ xx + f(rid, model = "iid") + f(yid, model = "iid"),
            family = "binomial", Ntrials = df$outcome_known, data = df,
            control.fixed = CTRL_FIXED,
            control.compute = list(dic = TRUE, waic = TRUE))
  per1  <- qsum(m, "xx", 1.0)
  per10 <- qsum(m, "xx", 0.1)
  s     <- sd_of(m, "Precision for rid")
  list(model = sprintf("region-year (%s), iid region + iid year", xvar),
       subset = label,
       n_rows = nrow(df), n_regions = length(unique(df$health_region_code)),
       cases = sum(df$cases), deaths = sum(df$deaths),
       outcome_known = sum(df$outcome_known),
       person_years = NA,
       intercept = unname(m$summary.fixed["(Intercept)", "mean"]),
       beta_H = unname(m$summary.fixed["xx", "mean"]),
       beta_H_sd = unname(m$summary.fixed["xx", "sd"]),
       or_per_unit_H = per1$or, or_lo = per1$or_lo, or_hi = per1$or_hi,
       or_per_10pp_H = per10$or, or_10pp_lo = per10$or_lo, or_10pp_hi = per10$or_hi,
       region_sd = s$sd, region_sd_lo = s$sd_lo, region_sd_hi = s$sd_hi,
       dic = m$dic$dic, waic = m$waic$waic)
}

fits <- list()
fits[[length(fits) + 1]] <- fit_region(reg, "no exclusion (all regions with >=1 case)")
for (t in c(10, 20, 30, 50)) {
  fits[[length(fits) + 1]] <- fit_region(reg[reg$cases >= t, ],
                                         sprintf("cases >= %d", t))
}
fits[[length(fits) + 1]] <- fit_region_year(
  ry, "no exclusion (all regions with >=1 case)", "H_region")
fits[[length(fits) + 1]] <- fit_region_year(
  ry, "no exclusion (all regions with >=1 case)", "H_year")
fits[[length(fits) + 1]] <- fit_region_year(
  ry[ry$cases >= 30, ], "cases >= 30", "H_year")

write(toJSON(fits, auto_unbox = TRUE, digits = 10, na = "null"), f_out)
cat("fits written:", length(fits), "\n")
