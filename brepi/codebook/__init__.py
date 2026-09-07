"""The DATASUS categorical codebook: one translation authority.

DATASUS microdata is *coded*. ``CS_RACA = '4'`` means *parda*, ``EVOLUCAO =
'2'`` means death from the notified disease, ``CLASSI_FIN = '1'`` means
confirmed. None of that is in the files; it is in the official data
dictionaries, one per agravo and per system, published as PDFs. Any analysis
that does not go through those dictionaries is guessing, and guessing wrong is
silent.

This module is the single place where code becomes meaning.

Structure of ``registry/datasus_codebook.yaml``
-----------------------------------------------
``concepts``
    Deduplicated by *concept*, not by field. ``yes_no_strict`` (1=sim, 2=nao,
    9=ignorado) appears in dozens of fields across SINAN, SIM and SIH; it is
    defined once. A concept declares ``values`` (code to canonical meaning) and
    ``unknown`` (sentinels that mean "genuinely unknown", which is *not* the
    same as absent).
``bindings``
    ``system -> field -> concept``. This is the only per-system content, and
    it is a mapping, not logic.
``lookups``
    ``system -> field -> reference table`` for high-cardinality codes
    (municipality, occupation/CBO, ICD) that are resolved against an external
    table rather than an inline dictionary.

Four states, never three
------------------------
The usual failure is collapsing everything non-informative to ``NA``, which
destroys the distinction the surveillance data actually carries:

``missing``
    The field is blank. Nobody answered.
``valid``
    The code is in the concept's dictionary; the canonical value is returned.
``unknown``
    The code is an explicit "ignorado" sentinel. Somebody answered, and the
    answer was that it is not known. On leptospirosis, ``CS_RACA = '9'`` runs
    at 8% of confirmed cases and varies by state and year; treating it as
    missing hides a measurable surveillance-quality gradient.
``invalid``
    The code is not in the dictionary at all. On leptospirosis,
    ``CLASSI_FIN = '8'`` appears 11,699 times and is absent from the v5.0
    dictionary. Silently dropping it would be a decision nobody recorded.

Both a record-level :func:`translate` and a vectorised
:func:`categorical_exprs` are provided, and they resolve through the same
dictionary so the two paths cannot drift.
"""

from brepi.codebook.codebook import (
    CodebookError,
    DecodeState,
    binding_for,
    concept,
    concepts_for_system,
    categorical_exprs,
    decode_frame,
    load_codebook,
    systems,
    translate,
    coverage_report,
)

__all__ = [
    "CodebookError",
    "DecodeState",
    "binding_for",
    "concept",
    "concepts_for_system",
    "categorical_exprs",
    "decode_frame",
    "load_codebook",
    "systems",
    "translate",
    "coverage_report",
]
