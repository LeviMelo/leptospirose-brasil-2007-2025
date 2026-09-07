# Dissolve the municipal mesh to health regions once, and cache it.
#
# The paper maps health regions (439 units), which is the modelling tier; the
# only geometry on disk is municipal (5,570 polygons). Dissolving is slow, so it
# is done once here and cached rather than repeated inside every figure script.

suppressPackageStartupMessages({library(sf); library(arrow); library(data.table)})

OUT <- "paper/figures"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
target <- file.path(OUT, "geo_health_region.gpkg")
outline <- file.path(OUT, "geo_brasil.gpkg")

if (file.exists(target) && file.exists(outline)) {
  cat("cached geometry already present\n"); quit(save = "no")
}

geo <- st_read("data/panel/geo_municipality.gpkg", quiet = TRUE)
atl <- as.data.table(read_parquet("data/results/atlas/municipality_atlas.parquet"))

# Simplify BEFORE dissolving: the union of 5,570 full-resolution polygons is the
# expensive step, and at print size the tolerance is invisible.
geo <- st_simplify(geo, dTolerance = 0.008, preserveTopology = TRUE)
geo <- merge(geo, atl[, .(munic_code, health_region_code)], by = "munic_code")
stopifnot(!any(is.na(geo$health_region_code)))

hr <- aggregate(geo["health_region_code"],
                by = list(health_region_code = geo$health_region_code),
                FUN = function(x) x[1], do_union = TRUE)
hr <- hr[, "health_region_code"]
cat(sprintf("dissolved %d municipalities into %d health regions\n",
            nrow(geo), nrow(hr)))
stopifnot(nrow(hr) == 439)
st_write(hr, target, delete_dsn = TRUE, quiet = TRUE)

# The national outline is a union of the dissolved regions, and s2's spherical
# validity check rejects it: simplification leaves self-touching edges at shared
# borders. Planar geometry is the right tool for a single national silhouette --
# it is a drawing, not a measurement -- so s2 is disabled for this step only.
old_s2 <- sf_use_s2()
sf_use_s2(FALSE)
br <- st_union(st_make_valid(st_geometry(hr)))
sf_use_s2(old_s2)
st_write(st_sf(geometry = br), outline, delete_dsn = TRUE, quiet = TRUE)
cat("wrote", target, "and", outline, "\n")
