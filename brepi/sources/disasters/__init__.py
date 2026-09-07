"""Officially recorded disaster events as a staggered treatment.

``atlas`` acquires and decodes the Atlas Digital de Desastres (SEDEC/MIDR, built
on S2iD); ``treatment`` converts its events into a Callaway-Sant'Anna design.
The split is the usual one in this package: ``atlas`` touches the network and is
the only place that knows the file format, ``treatment`` is pure and testable.

Two things a caller must read before using either:

* the default flood definition includes COBRADE ``1.3.2.1.4`` (chuvas intensas),
  because the strict ``1.2.x`` reading misses ~90% of the municipalities in the
  Rio Grande do Sul 2024 disaster;
* Brazilian municipalities flood repeatedly, so the absorbing-treatment
  assumption behind Callaway-Sant'Anna is violated. Both the absorbing and the
  recurrent definitions are implemented, and
  ``treatment.did_feasibility_report`` measures how much the distinction costs.

>>> from brepi.sources.disasters import atlas, treatment
>>> from brepi.geo import lattice
>>> events = atlas.disaster_events(range(2007, 2026))
>>> codes = lattice.load_municipalities(2022)["code7"].to_list()
>>> g = treatment.first_treatment_period(events, municipalities=codes)
>>> print(treatment.did_feasibility_report(events, municipalities=codes).summary())
"""

from brepi.sources.disasters import atlas, treatment

__all__ = ["atlas", "treatment"]
