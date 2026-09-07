"""Global configuration and path resolution for the brepi core.

Every path used anywhere in the package resolves through :data:`PATHS`. Nothing
constructs a path or a URL inline; that rule is what makes a run reproducible
from the manifest alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ENV_ROOT = "BREPI_DATA_ROOT"


def _default_root() -> Path:
    env = os.environ.get(_ENV_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "data").resolve()


@dataclass(frozen=True)
class Paths:
    """Canonical directory layout.

    ``cache`` holds byte-identical copies of remote artefacts plus their
    provenance sidecars. ``interim`` holds decoded but un-joined tables.
    ``panel`` holds assembled analysis-ready products.
    """

    root: Path = field(default_factory=_default_root)

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def panel(self) -> Path:
        return self.root / "panel"

    @property
    def derived(self) -> Path:
        return self.root / "derived"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def export(self) -> Path:
        return self.root / "export"

    @property
    def reports(self) -> Path:
        """Deprecated alias for :attr:`results`.

        Kept so existing call sites keep working, pointing at the new location
        rather than recreating the old flat directory beside it. See
        :mod:`brepi.paths` for the stage-aware accessor that should be used
        instead of composing a path from this.
        """
        return self.results

    def ensure(self) -> "Paths":
        for p in (self.cache, self.interim, self.panel, self.derived,
                  self.results, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths()

# --------------------------------------------------------------------------
# Source endpoints. Declared once, here, so that a mirror switch is one edit.
# --------------------------------------------------------------------------

#: DATASUS public FTP. Egress is geo-restricted to Brazil; from outside Brazil
#: the connection is *refused* at the TCP layer, which presents as an outage
#: rather than an access error. Verified 2026-07-30.
DATASUS_FTP_HOST = "ftp.datasus.gov.br"
DATASUS_FTP_ROOT = "/dissemin/publicos"

#: Unauthenticated S3 mirror of the same tree, minus the ``dissemin/publicos``
#: prefix, rclone-synced daily. Use when Brazilian egress is unavailable.
DATASUS_MIRROR_BASE = "https://datasus-ftp-mirror.nyc3.digitaloceanspaces.com"

#: IBGE SIDRA aggregate API.
SIDRA_API_BASE = "https://servicodados.ibge.gov.br/api/v3/agregados"
IBGE_LOCALIDADES_BASE = "https://servicodados.ibge.gov.br/api/v1/localidades"

#: Versioned official territorial meshes. Keep the vintage in the path: the
#: current IBGE mesh is not interchangeable with the 2022 analytic lattice.
IBGE_MUNICIPAL_MESH_BASE = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
    "malhas_municipais"
)

#: Ministry of Health open-data API. Unlike the historical DATASUS FTP
#: territory tables, this publishes the current municipality -> health-region
#: and macroregion membership as a paginated JSON/CSV resource.
OPEN_DATASUS_API_BASE = "https://apidadosabertos.saude.gov.br"

#: Atlas Digital de Desastres no Brasil (SEDEC/MIDR), the public face of S2iD.
#: There is no API: the JSF/PrimeFaces dashboard is server-rendered and issues
#: no JSON calls. The only programmatic route is the static bulk export linked
#: from the Downloads page, whose filename carries a version and a release date
#: and therefore *moves*. ``ATLAS_DOWNLOADS_PAGE`` is scraped for the current
#: link; ``ATLAS_BULK_CSV_FALLBACK`` pins the release this code was written
#: against so that a layout change degrades to a stale-but-known artefact
#: rather than to an exception. Verified 2026-07-30.
ATLAS_BASE = "https://atlasdigital.mdr.gov.br"
ATLAS_DOWNLOADS_PAGE = f"{ATLAS_BASE}/paginas/downloads.xhtml"
ATLAS_BULK_CSV_FALLBACK = (
    f"{ATLAS_BASE}/arquivos/BD_Atlas_1991_2025_v1.0_2026.04.23_Consolidado.csv"
)
#: Companion workbook listing the corrections SEDEC applied to the base. The
#: Atlas is retroactively edited; this is its only changelog.
ATLAS_CORRECTION_LOG = f"{ATLAS_BASE}/arquivos/2026.04-atlas-log-de-correcoes.xlsx"

#: The application manual, whose METADADOS chapter is the only official column
#: dictionary for the export. VERIFIED 2026-07-30 that it is *stale*: it
#: documents a single ``data`` column and knows nothing of ``Data_Evento`` or
#: ``Status``, both of which the current file carries.
ATLAS_MANUAL_PDF = f"{ATLAS_BASE}/arquivos/Atlas_Digital_Desastres_Manual_Aplicacao.pdf"

#: S2iD "Serie Historica": the federal *recognition portarias* (SE and ECP)
#: issued by SEDEC since 2013 — i.e. the one date the Atlas export omits. It is
#: a JSF form with no static file URL, so acquiring it means driving a POST.
#: Not implemented here; noted so that a robustness check on recognition timing
#: knows where to go.
S2ID_SERIES_PAGE = "https://s2id.mi.gov.br/paginas/series/"

#: MIDR CKAN open-data portal. VERIFIED 2026-07-30: it does NOT carry the Atlas
#: base. Its ``s2id_sedec`` package is a different, older product (one CSV per
#: year, 2013-2022 only) plus a data dictionary and the COBRADE PDF. Use it for
#: the dictionary, not as an Atlas mirror.
MDR_OPENDATA_BASE = "https://dadosabertos.mdr.gov.br"
MDR_OPENDATA_S2ID_DATASET = f"{MDR_OPENDATA_BASE}/dataset/s2id_sedec"

#: The Atlas export is Latin-1 and semicolon-delimited, like the DATASUS files.
ATLAS_ENCODING = "iso-8859-1"
ATLAS_SEPARATOR = ";"

# --------------------------------------------------------------------------
# Climate. See sources/climate/brdwgd.py for what each of these actually
# covers; the coverage windows are load-bearing and are asserted there.
# --------------------------------------------------------------------------

#: CDN mirror (rfsaldanha / brclimr) of BR-DWGD municipal zonal statistics.
#: Daily, all 5570 municipalities. VERIFIED 2026-07-30: 1961-01-01..2020-07-31
#: ONLY -- the objects carry a Last-Modified of 2023-03-03 and have not been
#: refreshed since. Free of rate limits and supports parquet predicate
#: pushdown, so it is the cheapest route for the pre-2020 climatology.
BRCLIMR_BRDWGD_CDN_BASE = "https://brdwgd.nyc3.cdn.digitaloceanspaces.com/parquet"

#: Same mirror, TerraClimate product. VERIFIED 2026-07-30: monthly,
#: 1958-01..2021-12 only. Too stale to carry this panel.
BRCLIMR_TERRACLIMATE_CDN_BASE = "https://terraclimate.nyc3.cdn.digitaloceanspaces.com/parquet"

#: Zenodo record API. ``{base}/{record}/files/{filename}/content`` is the
#: byte-stream endpoint. Zenodo rate-limits aggressively (HTTP 429 after a
#: few dozen requests), so parquet range-reads are NOT viable there: climate
#: artefacts must be downloaded whole into the cache and queried locally.
ZENODO_RECORD_BASE = "https://zenodo.org/api/records"

#: BR-DWGD v3.2.3 municipal zonal statistics, daily, 1961-01-01..2024-03-20.
#: Concept DOI 10.5281/zenodo.7824919; this is the current version record.
BRDWGD_ZONAL_ZENODO_RECORD = "13906834"

#: ERA5-Land municipal zonal statistics, daily, published in one historical
#: record plus one record per recent year. This is the only municipal-level
#: product that reaches 2025 and therefore the only way to close the panel.
ERA5LAND_ZONAL_ZENODO_RECORDS: dict[str, str] = {
    "1950-2022": "10036212",
    "2023": "10947952",
    "2024": "15748125",
    "2025": "18257037",
}

#: NOAA CPC Oceanic Nino Index: 3-month running SST anomaly in Nino 3.4,
#: relative to centred 30-year base periods. VERIFIED 2026-07-30, runs to
#: MAM 2026.
NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

#: NOAA CPC monthly (not 3-month-smoothed) Nino region SSTs and anomalies,
#: 1991-2020 base. Use when a genuinely monthly ENSO covariate is wanted.
NOAA_NINO_MONTHLY_URL = (
    "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
)


def zenodo_file_uri(record: str, filename: str) -> str:
    """The byte-stream URI of one file in a Zenodo record."""
    return f"{ZENODO_RECORD_BASE}/{record}/files/{filename}/content"


#: Network etiquette. DATASUS in particular throttles aggressive clients.
HTTP_TIMEOUT = 120.0
HTTP_MAX_RETRIES = 4
HTTP_BACKOFF_SECONDS = 2.0
FTP_TIMEOUT = 120.0

#: DATASUS DBF/CSV payloads are Latin-1. They are never UTF-8.
DATASUS_ENCODING = "iso-8859-1"

__all__ = [
    "PATHS",
    "Paths",
    "DATASUS_FTP_HOST",
    "DATASUS_FTP_ROOT",
    "DATASUS_MIRROR_BASE",
    "SIDRA_API_BASE",
    "IBGE_LOCALIDADES_BASE",
    "IBGE_MUNICIPAL_MESH_BASE",
    "ATLAS_BASE",
    "ATLAS_DOWNLOADS_PAGE",
    "ATLAS_BULK_CSV_FALLBACK",
    "ATLAS_CORRECTION_LOG",
    "ATLAS_MANUAL_PDF",
    "ATLAS_ENCODING",
    "ATLAS_SEPARATOR",
    "S2ID_SERIES_PAGE",
    "MDR_OPENDATA_BASE",
    "MDR_OPENDATA_S2ID_DATASET",
    "HTTP_TIMEOUT",
    "HTTP_MAX_RETRIES",
    "HTTP_BACKOFF_SECONDS",
    "FTP_TIMEOUT",
    "DATASUS_ENCODING",
    "BRCLIMR_BRDWGD_CDN_BASE",
    "BRCLIMR_TERRACLIMATE_CDN_BASE",
    "ZENODO_RECORD_BASE",
    "BRDWGD_ZONAL_ZENODO_RECORD",
    "ERA5LAND_ZONAL_ZENODO_RECORDS",
    "NOAA_ONI_URL",
    "NOAA_NINO_MONTHLY_URL",
    "zenodo_file_uri",
]
