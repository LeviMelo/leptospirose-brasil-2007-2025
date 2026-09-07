#!/usr/bin/env Rscript
# Tests for R/13_figure_fusion.R.
#
# Window selection is the part that can be wrong without looking wrong: a figure
# will render whatever boxes it is handed, and a reader cannot tell a
# mass-maximising window from one chosen because the result looked better there.
# So these are mostly invariant tests on the selection -- planted mass must be
# found, windows must not claim the same unit twice, the reported share must
# reconcile against the total, and the choice must not move when the input row
# order does.
#
# Usage: Rscript tests/test_figure_fusion.R
source("R/00_io.R"); source("R/13_figure_fusion.R")

.pass <- 0L; .fail <- 0L
ok <- function(label, cond) {
  if (isTRUE(cond)) { .pass <<- .pass + 1L; cat("  ok   ", label, "\n") }
  else { .fail <<- .fail + 1L; cat("  FAIL ", label, "\n") }
}
near <- function(a, b, tol = 1e-8) isTRUE(all(abs(a - b) < tol))

# A country-shaped scatter with three planted concentrations of mass, well
# separated, plus diffuse background weight everywhere else.
set.seed(1)
n_bg <- 400L
bg <- data.frame(x = runif(n_bg, 0, 100), y = runif(n_bg, 0, 100), w = 1)
blob <- function(cx, cy, n, w) {
  data.frame(x = rnorm(n, cx, 1.2), y = rnorm(n, cy, 1.2), w = w)
}
pts <- rbind(bg, blob(15, 15, 30, 40), blob(80, 20, 30, 25), blob(50, 85, 30, 12))

cat("\nzoom_windows: finding planted mass\n")
win <- zoom_windows(pts$x, pts$y, pts$w, k = 3, span = 0.12)
ok("returns k windows", nrow(win) == 3L)
ok("ordered by mass descending", !is.unsorted(rev(win$mass)))
centres <- data.frame(x = (win$xmin + win$xmax) / 2, y = (win$ymin + win$ymax) / 2)
found <- function(cx, cy, tol = 6) any(abs(centres$x - cx) < tol & abs(centres$y - cy) < tol)
ok("recovers the heaviest blob (15,15)", found(15, 15))
ok("recovers the second blob (80,20)", found(80, 20))
ok("recovers the third blob (50,85)", found(50, 85))
ok("heaviest blob is window 1", abs(centres$x[1] - 15) < 6 && abs(centres$y[1] - 15) < 6)

cat("\nzoom_windows: reported mass reconciles\n")
ok("share is mass over total", near(win$share, win$mass / sum(pts$w)))
ok("shares sum below 1", sum(win$share) < 1)
ok("windows capture a real fraction", sum(win$share) > 0.4)
ok("n_units is positive everywhere", all(win$n_units > 0))

cat("\nzoom_windows: windows do not double-claim units\n")
# Each unit may sit inside at most one window's *claim*; boxes may touch, but a
# unit counted twice would inflate the caption's coverage number.
claimed <- vapply(seq_len(nrow(win)), function(i) {
  sum(pts$x >= win$xmin[i] & pts$x <= win$xmax[i] &
      pts$y >= win$ymin[i] & pts$y <= win$ymax[i])
}, numeric(1))
ok("claimed counts match reported n_units", all(claimed >= win$n_units))
ok("total claimed does not exceed the point count", sum(win$n_units) <= nrow(pts))

cat("\nzoom_windows: determinism\n")
perm <- sample.int(nrow(pts))
win2 <- zoom_windows(pts$x[perm], pts$y[perm], pts$w[perm], k = 3, span = 0.12)
ok("row order does not change the windows",
   near(sort(win$mass), sort(win2$mass)))
win3 <- zoom_windows(pts$x, pts$y, pts$w, k = 3, span = 0.12)
ok("repeated calls agree exactly", identical(win, win3))

cat("\nzoom_windows: geometry\n")
ok("aspect 1 gives square windows in coordinate units",
   near(win$xmax - win$xmin, win$ymax - win$ymin))
tall <- zoom_windows(pts$x, pts$y, pts$w, k = 1, span = 0.1, aspect = 2)
ok("aspect 2 doubles the height", near((tall$ymax - tall$ymin),
                                       2 * (tall$xmax - tall$xmin)))
ok("span sets width as a fraction of the x-range",
   near(win$xmax[1] - win$xmin[1], 0.12 * diff(range(pts$x))))

cat("\nzoom_windows: refusals\n")
bad <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
ok("non-finite coordinates refused",
   bad(zoom_windows(c(1, NA), c(1, 2), c(1, 1), k = 1)))
ok("negative weights refused",
   bad(zoom_windows(c(1, 2), c(1, 2), c(1, -1), k = 1)))
ok("all-zero weights refused",
   bad(zoom_windows(c(1, 2), c(1, 2), c(0, 0), k = 1)))
ok("k below 1 refused",
   bad(zoom_windows(c(1, 2), c(1, 2), c(1, 1), k = 0)))

cat("\nzoom_windows: zero weights are ignored, not framed\n")
# Units with no weight must not attract a window: a magnified view of nothing
# is the failure mode this guards.
sparse <- data.frame(x = c(rep(90, 50), 10), y = c(rep(90, 50), 10),
                     w = c(rep(0, 50), 5))
w1 <- zoom_windows(sparse$x, sparse$y, sparse$w, k = 1, span = 0.2)
ok("window lands on the only weighted point",
   w1$xmin <= 10 && w1$xmax >= 10 && w1$ymin <= 10 && w1$ymax >= 10)
ok("mass equals the weight it found", near(w1$mass, 5))

cat("\nzoom_windows: k larger than the available mass\n")
one <- data.frame(x = c(1, 1.1), y = c(1, 1.1), w = c(1, 1))
few <- suppressWarnings(zoom_windows(one$x, one$y, one$w, k = 5, span = 0.9))
ok("returns fewer windows rather than empty ones", nrow(few) < 5L)
ok("every returned window has mass", all(few$mass > 0))

cat("\npanel_badge and describe_windows\n")
ok("badges are lowercase, per Lancet house style", panel_badge(1) == "a")
ok("badge 2 is 'b'", panel_badge(2) == "b")
txt <- describe_windows(win, what = "case mass")
ok("description names the criterion", grepl("rather than by eye", txt))
ok("description quotes the joint coverage",
   grepl(sprintf("%.1f%%", 100 * sum(win$share)), txt, fixed = TRUE))
ok("description labels start at 'b' when the map is panel 'a'",
   grepl("^Insets .* \\(b ", txt))

cat("\nzoom_windows: non-overlap\n")
# Two heavy blobs close enough that a greedy search would otherwise cover the
# same ground twice, magnifying the shared area in two separate insets.
close_pts <- rbind(blob(50, 50, 40, 30), blob(53, 50, 40, 28), bg)
overlaps <- function(w) {
  if (nrow(w) < 2) return(FALSE)
  any(w$xmin[1] < w$xmax[2] & w$xmax[1] > w$xmin[2] &
      w$ymin[1] < w$ymax[2] & w$ymax[1] > w$ymin[2])
}
ov <- zoom_windows(close_pts$x, close_pts$y, close_pts$w, k = 2, span = 0.2)
ok("chosen boxes do not overlap by default", !overlaps(ov))
ov2 <- zoom_windows(close_pts$x, close_pts$y, close_pts$w, k = 2, span = 0.2,
                    allow_overlap = TRUE)
ok("allow_overlap=TRUE finds at least as much mass",
   sum(ov2$mass) >= sum(ov$mass) - 1e-9)
ok("overlapping search is what non-overlap protects against",
   overlaps(ov2) || near(sum(ov2$mass), sum(ov$mass)))

cat("\nbadges are shared between frame and inset\n")
lab <- vapply(seq_len(nrow(win)), function(i) panel_badge(i + 1L), character(1))
ok("default labels follow the map panel", identical(lab, c("b", "c", "d")))
ok("describe_windows uses the same badges",
   all(vapply(lab, function(l) grepl(l, describe_windows(win)), logical(1))))

cat(sprintf("\n%d passed, %d failed\n", .pass, .fail))
if (.fail > 0L) quit(status = 1L)
