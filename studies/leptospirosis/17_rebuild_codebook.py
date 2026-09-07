"""Regenerate the SINAN-LEPT codebook from the parsed dictionary, authoritatively.

Supersedes the PDF-text parser in ``16_reconcile_dictionary.py``. That parser
had to infer table structure from a lossy text layer and got it wrong twice
(dropping the final category of every enumeration; the PDF itself had lost the
hyphen in "3 - Obito por outras causas"). The markdown rendering has an
explicit ``Categorias`` cell and an explicit ``DBF`` cell, so the codes are
read rather than reconstructed.

Two codes occur in the data and in no version of the dictionary. Both are
resolved here with their evidence, because a code that reaches the analysis
without a recorded meaning is a silent assumption:

``CLASSI_FIN = 8`` -> *inconclusivo*
    11,699 records, 2007-2025. Documented for SINAN NET generally, not in the
    leptospirosis dictionary. Assigned **automatically by the system when the
    investigation deadline lapses**, so it is an administrative timeout rather
    than a clinical judgement. It therefore belongs to the ascertainment
    analysis, not to the case definition, and is excluded from the numerator.

``LAB_MICR_1/2 = 4`` -> *nao realizado*
    699 records, 2018-2025. Not in the v5.0 dictionary, which documents
    1/2/3/9 for the MAT fields. Resolved **empirically**: 97.0% of these
    records carry no MAT collection date, against 98.4% for the documented
    "3 - Nao Realizada" and 3.5% / 14.7% for "1 - Reagente" / "2 - Nao
    Reagente". It duplicates code 3 and almost certainly appeared when the
    form harmonised the MAT scale onto the four-code ELISA scale.

Run:
    python studies/leptospirosis/17_rebuild_codebook.py [--write]
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MD = Path(__file__).resolve().parents[3] / "DIC_DADOS_Leptospirose_v5_parsed.md"
REGISTRY = Path(__file__).resolve().parents[2] / "brepi/codebook/registry/datasus_codebook.yaml"

#: "1 - Sim <br> 2 - Nao <br> 9 - Ignorado"
_CODE = re.compile(r"(\d{1,2})\s*[-–]\s*([^<|]+)")
#: Codes meaning "answered, but not known" rather than "not answered".
_UNKNOWN_WORDS = ("ignorado", "ignorada")

#: Codes present in the data and absent from every dictionary version. See the
#: module docstring for the evidence behind each.
UNDOCUMENTED = {
    "CLASSI_FIN": {"8": "inconclusivo"},
    "LAB_MICR_1": {"4": "nao_realizado"},
    "LAB_MICR_2": {"4": "nao_realizado"},
}


def slug(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", text)


def parse(md: Path) -> dict[str, dict[str, str]]:
    """DBF field -> {code: canonical meaning}, from the markdown table."""
    out: dict[str, dict[str, str]] = {}
    for line in md.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("| :---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        categories, dbf = cells[4], cells[-1]
        dbf = re.sub(r"[^A-Z0-9_]", "", dbf.upper())
        if not dbf or categories in {"", "—", "-"}:
            continue
        codes = {c: slug(lbl) for c, lbl in _CODE.findall(categories) if slug(lbl)}
        if codes:
            out.setdefault(dbf, {}).update(codes)
    return out


def main(write: bool = False) -> int:
    parsed = parse(MD)
    print(f"[codebook] parsed {len(parsed)} categorical fields from {MD.name}")

    for field, extra in UNDOCUMENTED.items():
        if field in parsed:
            parsed[field].update(extra)

    book = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    concepts, bindings = book["concepts"], book["bindings"]

    # Fields sharing an identical code set share a concept, so the eleven
    # risk-situation flags and sixteen symptom flags collapse to one entry
    # rather than twenty-seven near-duplicates.
    by_signature: dict[tuple, str] = {}
    new_bindings: dict[str, str] = {}
    added = 0
    for field, codes in sorted(parsed.items()):
        values = {c: v for c, v in codes.items()
                  if not any(w in v for w in _UNKNOWN_WORDS)}
        unknown = sorted(c for c, v in codes.items()
                         if any(w in v for w in _UNKNOWN_WORDS))
        sig = (tuple(sorted(values.items())), tuple(unknown))
        if sig in by_signature:
            new_bindings[field] = by_signature[sig]
            continue
        if sig == ((("1", "sim"), ("2", "nao")), ("9",)):
            cid = "sinan_sim_nao_ign"          # already in the registry
        else:
            cid = f"lept_{slug(field)}"
        concepts[cid] = {"dtype": "string", "values": values, "unknown": unknown}
        by_signature[sig] = cid
        new_bindings[field] = cid
        added += 1

    prev = set(bindings.get("SINAN-LEPT", {}))
    bindings["SINAN-LEPT"] = dict(sorted({**bindings.get("SINAN-LEPT", {}), **new_bindings}.items()))
    book["version"] = int(book.get("version", 1)) + 1

    print(f"[codebook] {added} distinct concepts; "
          f"{len(bindings['SINAN-LEPT'])} bound fields "
          f"({len(set(bindings['SINAN-LEPT']) - prev)} new)")
    for f in ("CLASSI_FIN", "EVOLUCAO", "LAB_MICR_1", "CON_AMBIEN"):
        cid = bindings["SINAN-LEPT"].get(f)
        if cid:
            print(f"    {f:<12} -> {cid:<28} {concepts[cid]['values']} "
                  f"unknown={concepts[cid]['unknown']}")

    if write:
        REGISTRY.write_text(
            yaml.safe_dump(book, allow_unicode=True, sort_keys=False, width=100),
            encoding="utf-8")
        print(f"[codebook] wrote {REGISTRY} (version {book['version']})")
    else:
        print("[codebook] dry run; pass --write to persist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--write" in sys.argv))
