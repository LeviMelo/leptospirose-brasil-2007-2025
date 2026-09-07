#!/usr/bin/env Rscript
# Run every R test file and exit non-zero if any fails.
#
# The R side has no testthat harness, deliberately: these are plain scripts that
# print a line per assertion and quit(status = 1) on failure, so they run
# identically under Rscript, under CI, and under a human reading the output.
# What was missing was a single entry point, which meant "the R tests pass" was
# a claim about whichever files somebody remembered to run.
#
# Usage: Rscript tests/run_all.R
files <- sort(list.files("tests", pattern = "^test_.*\\.R$", full.names = TRUE))
if (!length(files)) stop("no test files found under tests/", call. = FALSE)

rscript <- file.path(R.home("bin"), "Rscript")
results <- vapply(files, function(f) {
  cat("\n", strrep("=", 70), "\n", f, "\n", strrep("=", 70), "\n", sep = "")
  status <- system2(rscript, shQuote(f), stdout = "", stderr = "")
  as.integer(status)
}, integer(1))

cat("\n", strrep("=", 70), "\n", sep = "")
for (i in seq_along(files)) {
  cat(sprintf("  %-40s %s\n", basename(files[i]),
              if (results[i] == 0L) "PASS" else sprintf("FAIL (exit %d)", results[i])))
}
failed <- sum(results != 0L)
cat(sprintf("\n%d file(s), %d failed\n", length(files), failed))
if (failed > 0L) quit(status = 1L)
