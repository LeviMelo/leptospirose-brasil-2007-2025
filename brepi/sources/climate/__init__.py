"""Climate exposure: municipal daily weather, derived indices, ENSO.

    from brepi.sources.climate import brdwgd, indices, enso

    brdwgd.available_products()                  # what covers which years
    brdwgd.ensure_local("brdwgd", ["pr"], ...)   # download once, hashed
    monthly = brdwgd.municipal_climate(range(2007, 2026))
    monthly = indices.attach_anomalies(monthly)
    monthly = indices.attach_spi(monthly)
    panel.add(monthly, name="climate")           # onto the spine

The one thing to read before using this: no single published municipal-level
product spans 2007-2025. BR-DWGD stops 2024-03-20 and ERA5-Land carries the
rest, so the exposure layer is spliced and every row says which product it came
from. See :mod:`brepi.sources.climate.brdwgd` for the coverage table and
:func:`brepi.sources.climate.brdwgd.bias_adjustment` for the correction.
"""

from brepi.sources.climate import brdwgd, enso, indices

__all__ = ["brdwgd", "enso", "indices"]
