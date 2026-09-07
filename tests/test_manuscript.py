"""Structural checks on the RESS manuscript source.

These guard the failure modes that a render does not catch. A reference listed
but never cited still renders; a citation numbered out of order still renders;
an interpolation that resolves to nothing renders as a grammatical sentence with
a hole in it. Each of those shipped at least once during drafting.

The checks read `paper/artigo.qmd` as text rather than the rendered PDF, so they
run without Quarto, R or a LaTeX/Typst toolchain.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

QMD = Path(__file__).resolve().parents[1] / "paper" / "artigo.qmd"

#: RESS limits, from `docs/literature/journals/RESS_submission_rules.txt`.
MAX_REFERENCES = 30
MAX_ILLUSTRATIONS = 5


@pytest.fixture(scope="module")
def source() -> str:
    return QMD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def body_and_refs(source: str) -> tuple[str, str]:
    head, _, tail = source.partition("## Referências")
    assert tail, "manuscript has no reference section"
    return head, tail


def _citations(text: str) -> list[int]:
    """Every reference number appearing in a `^n^` or `^n,m^` superscript."""
    out: list[int] = []
    for marker in re.finditer(r"\^([0-9][0-9,\s]*)\^", text):
        out.extend(int(n) for n in marker.group(1).split(","))
    return out


def _listed(refs: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"^(\d+)\. ", refs, re.M)]


def test_every_listed_reference_is_cited(body_and_refs):
    body, refs = body_and_refs
    assert not set(_listed(refs)) - set(_citations(body))


def test_every_citation_is_listed(body_and_refs):
    body, refs = body_and_refs
    assert not set(_citations(body)) - set(_listed(refs))


def test_references_are_numbered_in_order_of_first_appearance(body_and_refs):
    body, _ = body_and_refs
    seen: list[int] = []
    for n in _citations(body):
        if n not in seen:
            seen.append(n)
    assert seen == sorted(seen), f"citations appear out of order: {seen}"


def test_reference_list_is_contiguous_from_one(body_and_refs):
    _, refs = body_and_refs
    listed = _listed(refs)
    assert listed == list(range(1, len(listed) + 1))


def test_reference_count_within_journal_limit(body_and_refs):
    _, refs = body_and_refs
    assert len(_listed(refs)) <= MAX_REFERENCES


def test_illustration_count_within_journal_limit(source: str):
    # Titles, not in-text mentions: those start a line with the bold label.
    titles = set(re.findall(r"^\*\*(Figura|Tabela) (\d)\.\*\*", source, re.M))
    assert len(titles) <= MAX_ILLUSTRATIONS, sorted(titles)


def test_every_figure_file_referenced_exists(source: str):
    for rel in re.findall(r"!\[\]\((figures/[^)]+?\.png)\)", source):
        assert (QMD.parent / rel).exists(), rel


def test_flow_lookup_asserts_a_unique_match(source: str):
    """The study-flow lookup must fail loudly rather than interpolate a blank.

    `fread` strips the indentation the flow table uses to mark sub-steps, so an
    untrimmed exact-match lookup silently returns nothing and the value renders
    as an empty string inside an otherwise grammatical sentence.
    """
    fl = re.search(r"fl\s*<- function\(k\)\s*\{(.+?)\n\}", source, re.S)
    assert fl, "the study-flow lookup helper is no longer recognisable"
    assert "trimws" in fl.group(1)
    assert "stopifnot" in fl.group(1)


def test_typst_only_blocks_are_balanced(source: str):
    """Unbreakable-block wrappers must open and close in pairs.

    An unclosed `#block(breakable: false)[` swallows the rest of the document
    into one block, which Typst renders as a single overflowing page.
    """
    opens = len(re.findall(r"#block\(breakable: false\)\[", source))
    closes = len(re.findall(r"```\{=typst\}\n\]\n```", source))
    assert opens == closes, f"{opens} opened, {closes} closed"
