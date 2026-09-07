"""Ingest the official leptospirosis dictionary and reconcile it with the files.

Source: *Dicionario de Dados - SINAN NET - versao 5.0*, agravo Leptospirose
(`DIC_DADOS_Leptospirose_v5.pdf`). The PDF is the only published description of
the form; it is not versioned per file vintage, so disagreement with any given
year's data is expected and is the output of interest rather than an error.

The parser is deliberately narrow. The PDF is a table rendered to text, and its
one reliable structural feature is that every row ends with the DBF column name
in the final cell. Fields are anchored on that, categories are recovered from
the `N-label` enumerations in the row, and anything not confidently parsed is
reported as unparsed rather than guessed at.

Run:
    python studies/leptospirosis/16_reconcile_dictionary.py [year]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from brepi.codebook.codebook import concept, concepts_for_system
from brepi.codebook.reconcile import DictionaryField, reconcile, write_reconciliation
from brepi.config import PATHS
from brepi.sources.datasus import sinan

PDF = Path(__file__).resolve().parents[3] / "DIC_DADOS_Leptospirose_v5.pdf"
OUT = PATHS.reports / "dictionary_lept_v5"

#: DBF names appear as the trailing cell of each dictionary row.
_DBF = re.compile(r"\b([A-Z][A-Z0-9_]{2,11})\b\s*$")
#: Category enumerations look like "1-Sim", "2- Nao", "9 - Ignorado".
#: Category enumerations run together on one line: "1- Confirmado 2- Descartado".
#: A regex capturing the label with a bounded character class is greedy across
#: the *next* code marker and returns a single category, which then makes every
#: other code look undocumented. Instead the markers are located and each label
#: is the span between consecutive markers, cut where explanatory prose starts.
_CAT_MARK = re.compile(r"(?<![\w.,])(\d{1,2})\s*[-–]\s*(?=[A-Za-zÀ-ÿ])")
_CAT_STOP = re.compile(
    r"(CAMPO|Campo|Se\s|Quando|Deve|Informar|Indica|Preencher|Habilitado|VARCHAR|NUMBER|DATE)"
)


def _categories(flat: str) -> dict[str, str]:
    marks = list(_CAT_MARK.finditer(flat))
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(flat)
        label = flat[start:end]
        stop = _CAT_STOP.search(label)
        if stop:
            label = label[: stop.start()]
        label = re.sub(r"[\s;.,–-]+$", "", label).strip()
        if label and len(label) <= 60:
            out.setdefault(m.group(1), label)
    return out


def extract_text(pdf: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise SystemExit(
            "pypdf is required to ingest the dictionary PDF:\n"
            "  python -m pip install pypdf"
        )
    reader = PdfReader(str(pdf))
    raw = "\n".join((p.extract_text() or "") for p in reader.pages)
    # The PDF is Latin-1 mis-declared as UTF-8; repair before matching accents.
    try:
        raw = raw.encode("latin-1", "ignore").decode("utf-8", "ignore")
    except Exception:
        pass
    return raw


def parse_fields(text: str) -> tuple[list[DictionaryField], list[str]]:
    """Recover (dbf_name, label, categories) triples from the dictionary text."""
    # Rows wrap across lines; rejoin on the numbered-field marker "31." etc.
    # Rows wrap across lines AND run together on one line: the PDF renders
    # "... CLASSI_FIN 62. Criterio de confirmacao ..." with no break, so a
    # newline-anchored split merges consecutive fields and the merged block is
    # attributed to whichever DBF name happens to end it -- which is how
    # CLASSI_FIN ended up with only half its categories. Flatten first, then
    # split on the numbered-field marker wherever it occurs.
    text = re.sub(r"\s+", " ", text)
    blocks = re.split(r"(?=\b\d{1,3}\.\s+[A-Za-zÀ-ÿ])", text)
    fields: dict[str, DictionaryField] = {}
    unparsed: list[str] = []
    for block in blocks:
        flat = re.sub(r"\s+", " ", block).strip()
        if not flat:
            continue
        m = _DBF.search(flat)
        if not m:
            unparsed.append(flat[:90])
            continue
        name = m.group(1)
        # Reject prose words that happen to be uppercase.
        if name in {"CAMPO", "OBRIGATORIO", "DBF", "SINAN", "NET", "DATE", "VARCHAR"}:
            unparsed.append(flat[:90])
            continue
        label = re.split(r"\s(?:VARCHAR|NUMBER|DATE)", re.sub(r"^\s*\d{1,3}\.\s*", "", flat))[0][:90]
        cats = _categories(flat)
        required = "OBRIGAT" in flat.upper()
        prev = fields.get(name)
        if prev is None or (not prev.categories and cats):
            fields[name] = DictionaryField(
                name=name, label=label, categories=cats or None, required=required
            )
    return list(fields.values()), unparsed


def main(year: int = 2024) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[dict] parsing {PDF.name}")
    declared, unparsed = parse_fields(extract_text(PDF))
    with_cats = [d for d in declared if d.categories]
    print(f"[dict] recovered {len(declared)} fields, {len(with_cats)} with categories; "
          f"{len(unparsed)} blocks unparsed")
    pl.DataFrame(
        [{"field": d.name, "label": d.label, "required": d.required,
          "n_categories": len(d.categories or {}),
          "categories": "; ".join(f"{k}={v}" for k, v in (d.categories or {}).items())}
         for d in declared]
    ).sort("field").write_csv(OUT / "dictionary_declared_fields.csv")
    pl.DataFrame({"block": unparsed}).write_csv(OUT / "dictionary_unparsed_blocks.csv")

    # The PDF text layer is lossy in ways that cannot be parsed around: the
    # EVOLUCAO row renders as "2-bito por leptospirose 3 bito por outras
    # causas", having lost the hyphen after "3", and enumerations that end a
    # table cell reliably drop their final category. A parser tuned until those
    # cases pass would be fitting the noise of one PDF render.
    #
    # So the curated registry in brepi/codebook -- hand-read from this same
    # dictionary -- is the authority, and the PDF parse is demoted to a
    # cross-check against it. Where they disagree, the disagreement is
    # reported for human review rather than silently resolved either way.
    curated = {
        field: concept(concept_id)
        for field, concept_id in concepts_for_system("SINAN-LEPT").items()
    }
    merged: list[DictionaryField] = []
    disagreements: list[dict] = []
    for d in declared:
        spec = curated.get(d.name)
        if spec is None:
            merged.append(d)
            continue
        codes_curated = {str(k) for k in (spec.get("values") or {})} | {
            str(u) for u in (spec.get("unknown") or [])
        }
        codes_pdf = set((d.categories or {}))
        if codes_pdf and codes_pdf != codes_curated:
            disagreements.append({
                "field": d.name,
                "only_in_pdf": ",".join(sorted(codes_pdf - codes_curated)) or None,
                "only_in_curated": ",".join(sorted(codes_curated - codes_pdf)) or None,
            })
        merged.append(
            DictionaryField(
                name=d.name, label=d.label, required=d.required,
                categories={c: str(c) for c in codes_curated},
                note="categories from curated registry",
            )
        )
    pl.DataFrame(
        disagreements,
        schema={"field": pl.Utf8, "only_in_pdf": pl.Utf8, "only_in_curated": pl.Utf8},
    ).write_csv(OUT / "dictionary_pdf_vs_curated.csv")
    print(f"[dict] curated registry covers {len(curated)} fields; "
          f"{len(disagreements)} disagree with the PDF parse (see "
          f"dictionary_pdf_vs_curated.csv)")

    print(f"[dict] loading LEPTBR{year % 100:02d}")
    frame, _ = sinan.fetch_year("LEPT", year)
    result = reconcile(frame, merged, sample_note=f"LEPTBR{year % 100:02d}")
    write_reconciliation(result, OUT)

    print("\n=== reconciliation summary ===")
    print(result["summary"].to_pandas().to_string(index=False))

    print("\n=== documented but ABSENT from the disseminated file ===")
    print(result["documented_absent"].select("field", "label").head(15)
          .to_pandas().to_string(index=False))

    print("\n=== observed codes NOT in the dictionary (version clashes) ===")
    print(result["undocumented_code"].head(20).to_pandas().to_string(index=False))

    print(f"\n[dict] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024))
