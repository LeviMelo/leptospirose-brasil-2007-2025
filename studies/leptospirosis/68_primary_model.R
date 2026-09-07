#!/usr/bin/env Rscript
# The primary model: is reported case fatality a function of how far
# surveillance reaches into the clinical spectrum?
#
# H = hospitalised confirmed cases / confirmed cases with a valid hospitalisation
# field. H is an INVERSE indicator of ascertainment breadth. A territory that
# records 90% of its confirmed cases as hospitalised is finding almost only
# severe illness -- NARROW ascertainment. One recording 50% is also finding mild
# illness -- BROAD ascertainment. The thesis is that reported case fatality is
# partly a property of where the case definition is drawn in practice rather than
# a property of the organism or of the care received, so higher H should predict
# higher reported case fatality after the persistent differences between places
# and the secular trend have been removed.
#
# WHAT WOULD FALSIFY IT. Four results, any of which would break the claim:
#   (i)  beta for H is null or negative once space and time are in the model;
#   (ii) beta collapses once demographic case mix enters, i.e. H is a proxy for
#        an older or otherwise frailer notified population and nothing more;
#   (iii) beta collapses when the analysis is confined to region-years whose
#        outcome field is essentially complete, i.e. the gradient is an artefact
#        of who gets an outcome recorded rather than of who gets notified;
#   (iv) beta survives cross-sectionally but vanishes WITHIN regions over time.
#        (iv) would not falsify the thesis outright, but it would demote it from
#        a statement about surveillance behaviour to a statement about the kind
#        of place that has narrow surveillance, and it must be reported as such
#        rather than buried -- so the within-region estimate is fitted here and
#        printed next to the Bayesian one, not relegated to a sensitivity file.
#
# The earlier draft of this study collapsed nineteen years into five grouped
# points and used a spatial model only to demonstrate that "space matters". Both
# are corrected here: H enters continuously at the observation unit, and the
# panel structure -- 426 health regions x 19 years -- is what identifies the
# spatial and temporal components separately from it.
#
# Usage: Rscript studies/leptospirosis/68_primary_model.R

suppressWarnings(source("renv/activate.R"))
suppressPackageStartupMessages({
  library(data.table); library(arrow); library(INLA)
  library(sf); library(spdep); library(jsonlite); library(sandwich)
})
for (f in c("00_io.R", "01_descriptive.R", "03_inla_spacetime.R",
            "09_exec.R", "11_tables.R", "12_severity.R")) {
  source(file.path("R", f))
}

OUT <- "data/results/primary_model"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
PANEL   <- "data/results/analysis_panel/region_year_panel.parquet"
GEOM    <- "paper/figures/geo_health_region.gpkg"
REPOADJ <- "data/panel/graphs/health_region.adj"   # read-only cross-check
GRAPH   <- file.path(OUT, "health_region_bym2.adj")
THREADS <- Sys.getenv("BREPI_BENCH_THREADS", "6:1")

# The exposure is reported per TEN percentage points of H, not per unit. A unit
# change in H is the whole range of the variable and produces an odds ratio no
# reader can hold in their head; ten percentage points is roughly a third of the
# interquartile range across health regions and is a movement a surveillance
# system could plausibly make. Every coefficient in every table below is on this
# scale, and the scale is carried in the output as a column rather than left in
# a caption.
PP <- 0.10

# Model D's threshold. 95% is not a conventional cut-off chosen for roundness:
# below it, the case-fatality denominator is missing enough of its cases that
# selective recording of fatal outcomes can move the numerator materially, and
# the whole point of D is to ask whether the gradient survives where that
# mechanism cannot operate.
COMPLETE_MIN <- 0.95

# Model F's threshold. H computed on fewer than ten cases is a coarse fraction
# (a single case moves it by >= 10 pp) and, in the extreme, a one-case cell has
# H in {0,1} and case fatality in {0,1} simultaneously -- the two are then not
# independent measurements at all. Ten is the smallest denominator at which H
# has finer resolution than the coefficient's own reporting scale.
MIN_HOSP_KNOWN <- 10L

w <- function(x, name) { fwrite(x, file.path(OUT, name))
                         cat("  wrote", name, "-", nrow(x), "rows\n") }
rule <- function(s) cat("\n", strrep("-", 74), "\n", s, "\n", sep = "")

# ---------------------------------------------------------------------------
# The analysis set
# ---------------------------------------------------------------------------
rule("Assembling the region-year analysis set")
panel <- as.data.table(read_parquet(PANEL))
cat("panel:", nrow(panel), "region-years,", uniqueN(panel$health_region_code),
    "regions,", uniqueN(panel$year), "years\n")

# A cell enters the analysis if it can supply BOTH sides of the regression: at
# least one case with a recorded outcome (else the binomial cell has Ntrials = 0
# and contributes no likelihood) and at least one case with a recorded
# hospitalisation field (else H is undefined). Nothing else is deleted --
# regions with a single case-year stay in, because partial pooling, not
# case-count filtering, is what protects the estimate from them.
dt <- panel[outcome_known > 0 & hosp_known > 0]
cat("dropped for a zero outcome denominator:",
    nrow(panel[outcome_known <= 0]), "cell(s) holding",
    sum(panel[outcome_known <= 0]$cases), "cases\n")
cat("dropped for an undefined H:",
    nrow(panel[outcome_known > 0 & hosp_known <= 0]), "cell(s) holding",
    sum(panel[outcome_known > 0 & hosp_known <= 0]$cases), "cases\n")
cat("analysis set:", nrow(dt), "cells,", uniqueN(dt$health_region_code),
    "regions,", uniqueN(dt$year), "years\n")

stopifnot(all(dt$deaths <= dt$outcome_known),
          all(dt$hospitalised <= dt$hosp_known),
          all(dt$d_hosp <= dt$hk_hosp))

crude_cfr <- sum(dt$deaths) / sum(dt$outcome_known)
national_H <- sum(dt$hospitalised) / sum(dt$hosp_known)
national_a60 <- sum(dt$age60) / sum(dt$age_known)
national_male <- sum(dt$male) / sum(dt$cases)
cat(sprintf("crude case fatality: %.2f%% (%d deaths / %d outcome-known cases)\n",
            100 * crude_cfr, sum(dt$deaths), sum(dt$outcome_known)))
cat(sprintf("national H: %.4f (%d hospitalised / %d hosp-known cases)\n",
            national_H, sum(dt$hospitalised), sum(dt$hosp_known)))

# Plausibility gate. Leptospirosis case fatality in Brazil is documented around
# 9-11%; anything an order of magnitude away is a plumbing bug, not a finding.
if (crude_cfr < 0.04 || crude_cfr > 0.20) {
  stop("crude case fatality of ", sprintf("%.3f", crude_cfr), " is outside the ",
       "range this disease takes anywhere. Stop and find the bug.", call. = FALSE)
}

# Centring at the national CASE-WEIGHTED value, not the unweighted mean of the
# cell-level ratios: the intercept is then the log odds of death in a territory
# whose ascertainment breadth is the national one, which is a quantity with a
# meaning. (Centring changes only the intercept; the slope is per 10 pp either
# way. It is done so the intercept is reportable, not to change beta.)
dt[, H10      := (H - national_H) / PP]
dt[, age60_10 := (share_age60 - national_a60) / PP]
dt[, male_10  := (share_male - national_male) / PP]

# ---------------------------------------------------------------------------
# The adjacency graph
# ---------------------------------------------------------------------------
# Built over ALL 439 national health regions rather than only the 426 that carry
# a case, because deleting a region tears a hole in the neighbourhood structure
# of the regions around it. The 13 case-free regions simply contribute no
# likelihood; their effect is prior- and neighbour-driven, which is the correct
# behaviour and not an imputation of data.
rule("Adjacency graph")
if (!file.exists(GRAPH)) {
  sf::sf_use_s2(FALSE)
  geo <- st_read(GEOM, quiet = TRUE)
  geo$health_region_code <- as.character(geo$health_region_code)
  geo <- geo[order(geo$health_region_code), ]
  # Simplification at ~200 m is for tractability only; it is checked below
  # against the repository's own health-region graph, which was built
  # independently, so a simplification that moved a boundary would show up as
  # disagreement rather than pass silently.
  geo <- suppressWarnings(st_make_valid(
    st_simplify(geo, dTolerance = 0.002, preserveTopology = TRUE)))
  nb <- poly2nb(geo, row.names = geo$health_region_code, queen = TRUE,
                snap = 0.005)
  nb2INLA(GRAPH, nb)
  hr_codes <- geo$health_region_code
  saveRDS(hr_codes, file.path(OUT, "graph_node_codes.rds"))
  cat("built:", length(nb), "nodes, mean", round(mean(card(nb)), 2),
      "neighbours, islands:", sum(card(nb) == 0),
      ", connected components:", n.comp.nb(nb)$nc, "\n")
  stopifnot(sum(card(nb) == 0) == 0)
} else {
  hr_codes <- readRDS(file.path(OUT, "graph_node_codes.rds"))
  cat("reusing", GRAPH, "-", length(hr_codes), "nodes\n")
}

# Cross-check against the graph the rest of the repository already uses. The
# node ordering convention (sorted health_region_code) is what makes id_space
# mean the same thing here as in 23_rq4_lethality.R; a mismatch would silently
# assign every region its neighbour's random effect.
if (file.exists(REPOADJ)) {
  old <- readLines(REPOADJ)
  if (as.integer(old[1]) == length(hr_codes)) {
    mine <- readLines(GRAPH)
    parse_adj <- function(lines) {
      lapply(lines[-1], function(ln) {
        v <- as.integer(strsplit(trimws(ln), "[[:space:]]+")[[1]])
        if (length(v) >= 2 && v[2] > 0) v[3:(2 + v[2])] else integer(0)
      })
    }
    a <- parse_adj(old); b <- parse_adj(mine)
    jac <- sum(mapply(function(x, y) length(intersect(x, y)), a, b)) /
           sum(mapply(function(x, y) length(union(x, y)), a, b))
    cat("agreement with", REPOADJ, "(Jaccard over edges):",
        sprintf("%.4f", jac), "\n")
    if (jac < 0.95) {
      stop("the geometry-derived graph disagrees with the repository graph. ",
           "One of them is wrong; do not fit until it is resolved.", call. = FALSE)
    }
  }
}

dt[, id_space := match(health_region_code, hr_codes)]
dt[, id_time  := year - min(year) + 1L]
stopifnot(!any(is.na(dt$id_space)))
N_SPACE <- length(hr_codes)
cat("id_space spans", min(dt$id_space), "to", max(dt$id_space),
    "over a graph of", N_SPACE, "nodes;", uniqueN(dt$id_space),
    "of them carry data\n")

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
# BYM2 (Riebler et al. 2016): u = (1/sqrt(tau)) * (sqrt(1-phi) v + sqrt(phi) u*),
# with the structured component scaled to unit generalised variance. The field's
# MARGINAL variance is therefore 1/tau, and phi is the share of THAT variance
# carried by the spatially structured component.
#
# Read phi correctly. phi = 0.9 says nine tenths of the SPATIAL RANDOM EFFECT's
# variance is structured rather than independent noise between neighbours. It
# says nothing whatever about how much of the variation in case fatality overall
# is spatial -- most of that is binomial sampling variation in small cells plus
# the fixed effects. A previous draft of this study wrote "almost all the
# residual variation is in the spatially structured component" from phi = 0.875.
# That sentence is wrong and must not reappear.
marginal_variance <- function(fit, hyper = "Precision for id_space") {
  m <- fit$marginals.hyperpar[[hyper]]
  if (is.null(m)) return(list(mean = NA_real_, lo = NA_real_, hi = NA_real_))
  v <- INLA::inla.tmarginal(function(x) 1 / x, m)
  z <- INLA::inla.zmarginal(v, silent = TRUE)
  list(mean = z$mean, lo = z$quant0.025, hi = z$quant0.975)
}

# The variance of the posterior-mean region effects across the regions that
# actually carry data. This is the empirical companion to 1/tau: 1/tau is what
# the model believes the spatial field's variance is, this is how far apart the
# fitted regions actually ended up. They answer the absorption question from two
# directions and are reported together because either alone is arguable.
spatial_effect <- function(fit, ids) {
  sp <- fit$summary.random$id_space
  if (is.null(sp)) return(NULL)
  # BYM2 returns 2n rows: the combined effect first, then the structured part.
  data.table(id_space = sp$ID[seq_len(N_SPACE)],
             u = sp$mean[seq_len(N_SPACE)])[id_space %in% ids]
}

hyper_row <- function(fit, name) {
  h <- fit$summary.hyperpar
  if (!name %in% rownames(h)) return(c(NA_real_, NA_real_, NA_real_))
  c(h[name, "mean"], h[name, "0.025quant"], h[name, "0.975quant"])
}

EST <- list()   # accumulates the estimates table
add <- function(model, term, estimate, lower, upper, scale) {
  EST[[length(EST) + 1L]] <<- data.table(
    model = model, term = term, estimate = estimate,
    lower = lower, upper = upper, scale = scale)
}

# Harvest everything reportable from one fit into the estimates table.
harvest <- function(res, model, ids, note) {
  fit <- res$fit
  od <- res$odds
  for (i in seq_len(nrow(od))) {
    tm <- od$term[i]
    add(model, tm, od$or[i], od$or_lo[i], od$or_hi[i],
        if (tm == "(Intercept)")
          "odds of death at the national mean of every centred covariate"
        else "odds ratio per +10 percentage points")
  }
  phi <- hyper_row(fit, "Phi for id_space")
  add(model, "phi (BYM2 mixing)", phi[1], phi[2], phi[3],
      "share of the SPATIAL EFFECT's marginal variance that is spatially structured")
  mv <- marginal_variance(fit)
  add(model, "spatial marginal variance", mv$mean, mv$lo, mv$hi,
      "variance of the BYM2 field on the logit scale (= 1/precision)")
  add(model, "spatial marginal sd", sqrt(mv$mean), sqrt(mv$lo), sqrt(mv$hi),
      "sd of the BYM2 field on the logit scale")
  tp <- hyper_row(fit, "Precision for id_time")
  add(model, "rw1 year precision", tp[1], tp[2], tp[3], "precision")
  u <- spatial_effect(fit, ids)
  add(model, "empirical sd of region effects", sd(u$u), NA_real_, NA_real_,
      "sd across regions with data, of posterior-mean BYM2 effects (logit scale)")
  sc <- res$scores
  add(model, "DIC",  sc$dic,  NA_real_, NA_real_, "information criterion")
  add(model, "WAIC", sc$waic, NA_real_, NA_real_, "information criterion")
  add(model, "cells", nrow(res$data_used), NA_real_, NA_real_, "region-years")
  add(model, "deaths", res$data_dim$deaths, NA_real_, NA_real_, "count")
  add(model, "trials", res$data_dim$trials, NA_real_, NA_real_, "count")
  cat("\n", model, " -- ", note, "\n", sep = "")
  print(od[, .(term, or = round(or, 4), lo = round(or_lo, 4),
               hi = round(or_hi, 4))])
  cat(sprintf("  phi %.3f [%.3f, %.3f] | spatial marginal var %.4f [%.4f, %.4f]",
              phi[1], phi[2], phi[3], mv$mean, mv$lo, mv$hi),
      sprintf("| empirical sd %.4f\n", sd(u$u)))
  cat(sprintf("  DIC %.1f | WAIC %.1f | cells %d\n", sc$dic, sc$waic,
              nrow(res$data_used)))
  invisible(list(fit = fit, phi = phi, mv = mv, u = u, scores = sc))
}

# ---------------------------------------------------------------------------
# The model sequence
# ---------------------------------------------------------------------------
rule("Fitting the sequence")
ctl <- model_execution_profile("confirmatory", num_threads = THREADS)$controls
cat("integration:", ctl$control.inla$int_strategy %||% "ccd",
    "| strategy:", ctl$control.inla$strategy, "\n")
cat("PC priors (as in 23_rq4_lethality.R): P(sd_spatial > 1) = 0.01;",
    "P(phi < 0.5) = 0.5; P(sd_rw1 > 1) = 0.01\n")
cat("fixed effects: N(0, 1) on the log odds ratio per 10 pp -- weakly",
    "informative,\n  allowing a sevenfold odds ratio at two prior sd\n")

fitseq <- function(data, covariates, label) {
  res <- fit_severity(data, covariates = covariates, graph = GRAPH,
                      control = ctl, adjust_completeness = FALSE,
                      num_threads = THREADS)
  res$data_used <- data
  res
}

# A -- space and time only. This is the reference against which absorption is
# measured, so it is fitted on EXACTLY the rows model B uses. Fitting A on the
# full panel and B on the H-defined subset would make the variance comparison an
# artefact of the different row sets.
mA <- fitseq(dt, character(), "A")
hA <- harvest(mA, "A_space_time", unique(dt$id_space),
              "space + time only; no exposure")

# B -- the exposure enters.
mB <- fitseq(dt, "H10", "B")
hB <- harvest(mB, "B_plus_H", unique(dt$id_space),
              "A + H, continuous, per +10 percentage points")

# C -- demographic case mix. Threat: H is a proxy for an older or more male
# notified population, and age is the real driver of death. If beta for H
# collapses here, the ascertainment reading is wrong.
mC <- fitseq(dt, c("H10", "age60_10", "male_10"), "C")
hC <- harvest(mC, "C_plus_case_mix", unique(dt$id_space),
              "B + proportion aged 60+ and proportion male")

# D -- outcome completeness. Threat: where the outcome field is poorly filled,
# deaths are recorded preferentially over recoveries, inflating the observed
# case-fatality proportion; if completeness is also lower where H is high, that
# alone would produce the gradient. Restriction rather than adjustment, because
# a covariate cannot repair a selectively observed denominator.
dtD <- dt[outcome_completeness >= COMPLETE_MIN]
cat("\nD retains", nrow(dtD), "of", nrow(dt), "cells",
    sprintf("(%.1f%%)", 100 * nrow(dtD) / nrow(dt)), "and",
    uniqueN(dtD$id_space), "regions\n")
mD <- fitseq(dtD, c("H10", "age60_10", "male_10"), "D")
hD <- harvest(mD, "D_complete_outcomes", unique(dtD$id_space),
              sprintf("C restricted to outcome completeness >= %.0f%%",
                      100 * COMPLETE_MIN))

# ---------------------------------------------------------------------------
# Threat checks that are not part of the reported sequence but answer the two
# objections a reviewer raises first
# ---------------------------------------------------------------------------
rule("Directed checks on the two strongest objections")

# E -- mechanical composition. Hospitalised cases die more often than
# non-hospitalised ones, so a territory with a higher hospitalised SHARE has a
# higher case-fatality proportion by arithmetic, with no ascertainment story
# needed. The discriminating analysis holds the mix fixed: case fatality among
# HOSPITALISED cases only, against the same H. If the composition objection were
# the whole story, beta here would be zero. Note the direction of the prior: on
# a pure-composition account, admitting a larger share of cases should if
# anything DILUTE the hospitalised pool with milder patients and push this beta
# negative, so a positive residual beta is the harder result to explain away.
dtE <- dt[hk_hosp > 0]
dtE <- copy(dtE)[, `:=`(deaths_all = deaths, outcome_known_all = outcome_known)]
dtE[, `:=`(deaths = d_hosp, outcome_known = hk_hosp)]
cat("E: ", nrow(dtE), " cells, ", sum(dtE$deaths), " deaths among ",
    sum(dtE$outcome_known), " hospitalised outcome-known cases",
    sprintf(" (within-hospitalised case fatality %.2f%%)\n",
            100 * sum(dtE$deaths) / sum(dtE$outcome_known)), sep = "")
mE <- fitseq(dtE, c("H10", "age60_10", "male_10"), "E")
hE <- harvest(mE, "E_within_hospitalised", unique(dtE$id_space),
              "C, but the outcome is case fatality among HOSPITALISED cases only")

# F -- resolution of H. Threat: in a cell with one or two cases, H and the
# case-fatality proportion are computed from overlapping handfuls of the same
# people and are not independent measurements. If the gradient is manufactured
# by those cells it will not survive restriction to cells where H rests on at
# least ten cases.
dtF <- dt[hosp_known >= MIN_HOSP_KNOWN]
cat("\nF retains", nrow(dtF), "of", nrow(dt), "cells",
    sprintf("(%.1f%%)", 100 * nrow(dtF) / nrow(dt)), "holding",
    sprintf("%.1f%%", 100 * sum(dtF$cases) / sum(dt$cases)), "of the cases\n")
mF <- fitseq(dtF, c("H10", "age60_10", "male_10"), "F")
hF <- harvest(mF, "F_hosp_known_ge10", unique(dtF$id_space),
              sprintf("C restricted to cells with >= %d cases behind H",
                      MIN_HOSP_KNOWN))

# ---------------------------------------------------------------------------
# The quantity the model exists for: how much spatial variance H absorbs
# ---------------------------------------------------------------------------
rule("How much of model A's spatial variance does H absorb?")
# If the persistent differences in case fatality between health regions are, in
# part, differences in where each territory draws its case definition, then
# putting H in the linear predictor should take variance OUT of the spatial
# random effect. If H were unrelated to the between-region pattern, the field
# would be unchanged and only the fixed effect would move.
#
# Two readings, deliberately both reported. The first compares the posterior
# means of the field's marginal variance (1/tau); the second compares the
# empirical spread of the fitted region effects. The first is what the model
# believes, the second is what it produced. Neither carries a credible interval
# for the DIFFERENCE, because A and B are separate fits and the difference of
# two posteriors from separate fits is not itself a posterior -- reported as
# point comparisons, and described that way.
absorb_marginal  <- 1 - hB$mv$mean / hA$mv$mean
uAB <- merge(hA$u[, .(id_space, uA = u)], hB$u[, .(id_space, uB = u)],
             by = "id_space")
absorb_empirical <- 1 - var(uAB$uB) / var(uAB$uA)
cat(sprintf("marginal variance of the BYM2 field: A %.4f -> B %.4f  (%.1f%% absorbed)\n",
            hA$mv$mean, hB$mv$mean, 100 * absorb_marginal))
cat(sprintf("empirical variance of region effects: A %.4f -> B %.4f  (%.1f%% absorbed)\n",
            var(uAB$uA), var(uAB$uB), 100 * absorb_empirical))
cat(sprintf("correlation between the two fields: %.3f\n",
            cor(uAB$uA, uAB$uB)))
add("A_vs_B", "spatial marginal variance absorbed by H", absorb_marginal,
    NA_real_, NA_real_,
    "proportion; point comparison of posterior means from two separate fits")
add("A_vs_B", "empirical spatial variance absorbed by H", absorb_empirical,
    NA_real_, NA_real_,
    "proportion; variance of posterior-mean region effects, A versus B")
uAB[, health_region_code := hr_codes[id_space]]
w(uAB[, .(health_region_code, id_space, u_model_A = uA, u_model_B = uB,
          or_A = exp(uA), or_B = exp(uB))],
  "spatial_effects_A_vs_B.csv")

# ---------------------------------------------------------------------------
# The within-region estimate
# ---------------------------------------------------------------------------
rule("Within-region (fixed-effects) estimate")
# Every model above is identified partly by comparing one health region with
# another, so any time-invariant regional property that raises both H and case
# fatality -- poverty, distance to a reference hospital, a laboratory that only
# ever tests inpatients, a different circulating serovar -- is confounding it.
# A region fixed effect removes all of them by construction, at the price of
# discarding the between-region contrast entirely. beta is then identified only
# by a region moving its own H from one year to the next.
#
# A region can only contribute if it has both a death and a survival somewhere
# in its series; one with no deaths at all is perfectly predicted by its own
# fixed effect and its likelihood contribution to beta is degenerate. These are
# removed explicitly and counted, rather than left for glm to send to +/-Inf.
reg <- dt[, .(d = sum(deaths), n = sum(outcome_known), yrs = .N),
          by = health_region_code]
usable <- reg[d > 0 & d < n & yrs > 1, health_region_code]
fedt <- dt[health_region_code %in% usable]
cat("regions usable for a within estimate:", length(usable), "of",
    uniqueN(dt$health_region_code), "\n")
cat("  removed:", reg[d == 0, .N], "with no death in any year;",
    reg[d > 0 & d == n, .N], "with no survivor in any year;",
    reg[yrs <= 1 & d > 0 & d < n, .N], "observed in a single year\n")
cat("  they held", sum(dt$cases) - sum(fedt$cases), "cases of", sum(dt$cases),
    sprintf("(%.1f%%)\n", 100 * (1 - sum(fedt$cases) / sum(dt$cases))))
cat("within-estimate rows:", nrow(fedt), "| deaths", sum(fedt$deaths),
    "| trials", sum(fedt$outcome_known), "\n")

fedt[, hr := factor(health_region_code)]
fedt[, yr := factor(year)]

fit_fe <- function(rhs, label) {
  f <- stats::as.formula(paste("cbind(deaths, outcome_known - deaths) ~",
                               rhs, "+ hr + yr"))
  m <- stats::glm(f, family = stats::binomial(), data = fedt)
  # Cluster-robust by health region. The binomial likelihood assumes the cases
  # in a region-year are independent draws; they are not (shared outbreak,
  # shared hospital, shared notifier), and the model-based standard error is
  # correspondingly too small. Clustering on the region is the honest interval.
  V <- sandwich::vcovCL(m, cluster = fedt$hr, type = "HC0")
  keep <- grep("^(H10|age60_10|male_10)$", names(stats::coef(m)), value = TRUE)
  b <- stats::coef(m)[keep]; se <- sqrt(diag(V))[keep]
  z <- stats::qnorm(0.975)
  out <- data.table(term = keep, or = exp(b),
                    lo = exp(b - z * se), hi = exp(b + z * se),
                    se_cluster = se,
                    se_model = sqrt(diag(stats::vcov(m)))[keep])
  cat("\n", label, "  (", length(usable), " region effects, ",
      nlevels(fedt$yr), " year effects)\n", sep = "")
  print(out[, .(term, or = round(or, 4), lo = round(lo, 4), hi = round(hi, 4),
                se_cluster = round(se_cluster, 4),
                se_model = round(se_model, 4))])
  for (i in seq_len(nrow(out))) {
    add(label, out$term[i], out$or[i], out$lo[i], out$hi[i],
        "odds ratio per +10 percentage points; cluster-robust by health region")
  }
  out
}

fe1 <- fit_fe("H10", "FE_region_year_H")
fe2 <- fit_fe("H10 + age60_10 + male_10", "FE_region_year_H_case_mix")

or_B  <- mB$odds[term == "H10", or]
or_C  <- mC$odds[term == "H10", or]
or_fe <- fe2[term == "H10", or]
cat("\nBetween versus within:\n")
cat(sprintf("  partially pooled, case mix adjusted (model C): OR %.4f per +10 pp\n", or_C))
cat(sprintf("  within region, case mix adjusted (FE):         OR %.4f per +10 pp\n", or_fe))
cat(sprintf("  the within estimate is %.0f%% of the log odds ratio of model C\n",
            100 * log(or_fe) / log(or_C)))

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
rule("Writing results")
est <- rbindlist(EST)
w(est, "model_estimates.csv")

pull <- function(m, t) {
  r <- est[model == m & term == t]
  if (!nrow(r)) return(NULL)
  list(estimate = r$estimate[1], lower = r$lower[1], upper = r$upper[1],
       scale = r$scale[1])
}
hrow <- function(m) pull(m, "H10")

report <- list(
  question = paste(
    "Across health regions and years, how does reported case fatality change",
    "with the hospitalisation proportion H, after accounting for persistent",
    "spatial differences, temporal trend and demographic case mix?"),
  construct = paste(
    "H = hospitalised confirmed cases / confirmed cases with a valid",
    "hospitalisation field. H is an INVERSE indicator of ascertainment breadth:",
    "a high H means the surveillance system is finding almost only severe",
    "illness (narrow ascertainment)."),
  measure = paste(
    "Odds ratio for the case-fatality proportion, per +10 percentage points of",
    "H, from a binomial logit model with a BYM2 spatial field and an RW1 over",
    "year, fitted by INLA."),
  exposure_scale = list(
    unit = "10 percentage points of H",
    centred_at = national_H,
    centring_note = paste(
      "H centred at the national case-weighted value so the intercept is the",
      "odds of death at national ascertainment breadth; centring does not",
      "affect the slope.")),
  data = list(
    panel = PANEL,
    panel_cells = nrow(panel),
    analysis_cells = nrow(dt),
    regions = uniqueN(dt$health_region_code),
    graph_nodes = N_SPACE,
    years = list(min = min(dt$year), max = max(dt$year)),
    deaths = sum(dt$deaths),
    outcome_known = sum(dt$outcome_known),
    cases = sum(dt$cases),
    crude_case_fatality = crude_cfr,
    national_H = national_H,
    national_share_age60 = national_a60,
    national_share_male = national_male,
    excluded_zero_outcome_denominator = nrow(panel[outcome_known <= 0]),
    excluded_undefined_H = nrow(panel[outcome_known > 0 & hosp_known <= 0])),
  models = list(
    A_space_time = list(
      description = "BYM2 space + RW1 year, no exposure",
      phi = pull("A_space_time", "phi (BYM2 mixing)"),
      spatial_marginal_variance = pull("A_space_time", "spatial marginal variance"),
      spatial_marginal_sd = pull("A_space_time", "spatial marginal sd"),
      dic = est[model == "A_space_time" & term == "DIC", estimate],
      waic = est[model == "A_space_time" & term == "WAIC", estimate],
      cells = nrow(dt)),
    B_plus_H = list(
      description = "A + H (continuous)",
      or_H_per_10pp = hrow("B_plus_H"),
      phi = pull("B_plus_H", "phi (BYM2 mixing)"),
      spatial_marginal_variance = pull("B_plus_H", "spatial marginal variance"),
      dic = est[model == "B_plus_H" & term == "DIC", estimate],
      waic = est[model == "B_plus_H" & term == "WAIC", estimate],
      cells = nrow(dt)),
    C_plus_case_mix = list(
      description = "B + proportion aged 60+ and proportion male",
      or_H_per_10pp = hrow("C_plus_case_mix"),
      or_age60_per_10pp = pull("C_plus_case_mix", "age60_10"),
      or_male_per_10pp = pull("C_plus_case_mix", "male_10"),
      phi = pull("C_plus_case_mix", "phi (BYM2 mixing)"),
      spatial_marginal_variance = pull("C_plus_case_mix", "spatial marginal variance"),
      dic = est[model == "C_plus_case_mix" & term == "DIC", estimate],
      waic = est[model == "C_plus_case_mix" & term == "WAIC", estimate],
      cells = nrow(dt)),
    D_complete_outcomes = list(
      description = sprintf("C restricted to outcome completeness >= %.0f%%",
                            100 * COMPLETE_MIN),
      or_H_per_10pp = hrow("D_complete_outcomes"),
      phi = pull("D_complete_outcomes", "phi (BYM2 mixing)"),
      spatial_marginal_variance = pull("D_complete_outcomes",
                                       "spatial marginal variance"),
      dic = est[model == "D_complete_outcomes" & term == "DIC", estimate],
      waic = est[model == "D_complete_outcomes" & term == "WAIC", estimate],
      cells = nrow(dtD),
      cell_share_retained = nrow(dtD) / nrow(dt))),
  threat_checks = list(
    E_within_hospitalised = list(
      threat = paste(
        "Hospitalised cases die more often, so a higher hospitalised share",
        "raises the case-fatality proportion by arithmetic alone."),
      discriminates = paste(
        "Case fatality among hospitalised cases only, against the same H. A",
        "null beta would mean the gradient is purely compositional."),
      or_H_per_10pp = hrow("E_within_hospitalised"),
      within_hospitalised_case_fatality =
        sum(dtE$deaths) / sum(dtE$outcome_known),
      cells = nrow(dtE)),
    F_hosp_known_ge10 = list(
      threat = paste(
        "In cells with a handful of cases, H and the case-fatality proportion",
        "are computed from overlapping people and are not independent",
        "measurements."),
      discriminates = sprintf(
        "Restriction to cells with at least %d cases behind H.", MIN_HOSP_KNOWN),
      or_H_per_10pp = hrow("F_hosp_known_ge10"),
      cells = nrow(dtF),
      case_share_retained = sum(dtF$cases) / sum(dt$cases))),
  spatial_variance_absorbed_by_H = list(
    marginal_variance_A = hA$mv$mean,
    marginal_variance_B = hB$mv$mean,
    proportion_absorbed_marginal = absorb_marginal,
    empirical_variance_A = var(uAB$uA),
    empirical_variance_B = var(uAB$uB),
    proportion_absorbed_empirical = absorb_empirical,
    correlation_of_fields = cor(uAB$uA, uAB$uB),
    note = paste(
      "A and B are separate fits, so these are point comparisons of posterior",
      "summaries, not a posterior for a difference. Both fits use exactly the",
      "same rows.")),
  within_region = list(
    specification = paste(
      "glm(cbind(deaths, outcome_known - deaths) ~ H10 [+ case mix] + region +",
      "year, binomial), standard errors clustered on health region."),
    regions_contributing = length(usable),
    regions_total = uniqueN(dt$health_region_code),
    case_share_retained = sum(fedt$cases) / sum(dt$cases),
    or_H_per_10pp = hrow("FE_region_year_H"),
    or_H_per_10pp_case_mix = hrow("FE_region_year_H_case_mix"),
    ratio_of_log_or_within_to_model_C = log(or_fe) / log(or_C)),
  interpretation_discipline = list(
    phi = paste(
      "phi is the share of the SPATIAL RANDOM EFFECT's marginal variance that",
      "is spatially structured rather than independent between neighbours. It",
      "is NOT the share of variation in case fatality that is spatial."),
    ecological = paste(
      "The observation unit is a health region-year. Every estimate is an",
      "ecological association between a territory's ascertainment breadth and",
      "its reported case fatality. It does not license the individual-level",
      "statement that hospitalising a patient raises that patient's risk of",
      "death."),
    dic_waic = paste(
      "DIC and WAIC are comparable across A, B and C, which are fitted to",
      "identical rows. They are NOT comparable to D, E or F, which are fitted",
      "to different rows or a different outcome.")),
  provenance = list(
    script = "studies/leptospirosis/68_primary_model.R",
    graph = GRAPH,
    graph_source = GEOM,
    inla_version = as.character(utils::packageVersion("INLA")),
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    run_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"))
)
write_json(report, file.path(OUT, "primary_model_report.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 8, na = "null")
cat("  wrote primary_model_report.json\n")

if (exists("assert_no_strays")) assert_no_strays(kill = TRUE)
rule("done")
cat("wrote", OUT, "\n")
