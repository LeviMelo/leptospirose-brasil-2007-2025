"""The research journal's evidence index must point at files that exist.

The journal is the study's audit trail: §10 maps every link in the argument
chain to the script that produced it and the generated files that hold its
numbers. An index whose paths have rotted is worse than no index, because it
looks auditable and is not — a reader who cannot resolve a citation has no way
to distinguish a moved file from an invented result.

This checks the citations resolve. It cannot check that the numbers in the prose
match the numbers in the files; that is what `audit_claims.py` and
`RESULT_ASSERTIONS.yaml` are for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "docs" / "RESEARCH_JOURNAL.md"

#: Where a cited basename may legitimately live.
SEARCH_ROOTS = ("data/results", "studies/leptospirosis", "docs", "scripts",
                "tests", "brepi")

#: Backticked tokens ending in an analysis-artefact extension.
CITATION = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|R|csv|parquet|json|yaml|md))`")


def _text() -> str:
    return JOURNAL.read_text(encoding="utf-8")


def _index() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for base in SEARCH_ROOTS:
        b = ROOT / base
        if not b.exists():
            continue
        for f in b.rglob("*"):
            if f.is_file():
                out.setdefault(f.name, []).append(f)
    return out


def _evidence_section() -> str:
    t = _text()
    marker = "## 10. Evidence index"
    assert marker in t, "the journal has lost its evidence index (§10)"
    return t[t.index(marker):]


def test_journal_has_its_structural_sections() -> None:
    """Guard the guard: a truncated journal must fail loudly."""
    t = _text()
    for section in ("## 1. Scope", "## 2. Argument chain", "## 3. Threat ledger",
                    "## 4. Open threads", "## 10. Evidence index"):
        assert section in t, f"journal is missing {section!r}"
    assert len(t) > 40_000, f"journal is only {len(t)} chars — probably truncated"


def test_argument_chain_is_contiguously_numbered() -> None:
    """Links must run 1..N with no gaps, so a reader can follow the argument.

    Letter-suffixed links (9b) are permitted — they mark a claim inserted into
    an existing position — but a *gap* in the integer sequence means a link was
    deleted and its cross-references are now dangling.
    """
    nums = [int(m) for m in re.findall(r"(?m)^### Link (\d+)[a-z]? — ", _text())]
    assert nums, "no links found"
    expected = list(range(1, max(nums) + 1))
    missing = sorted(set(expected) - set(nums))
    assert not missing, f"gaps in the link sequence: {missing}"


def test_every_evidence_citation_resolves() -> None:
    section = _evidence_section()
    index = _index()
    unresolved: list[str] = []
    for token in sorted(set(CITATION.findall(section))):
        if "*" in token or (ROOT / token).exists():
            continue
        hits = index.get(token.split("/")[-1], [])
        if not hits:
            unresolved.append(token)
        elif "/" in token and not any(
            str(h).replace("\\", "/").endswith(token) for h in hits
        ):
            unresolved.append(f"{token} (basename exists elsewhere)")
    assert not unresolved, (
        "evidence index cites files that do not exist: "
        + ", ".join(unresolved)
        + ". Either the analysis was not run, or a file moved and the index "
        "was not updated. An unresolvable citation is not an audit trail."
    )


@pytest.mark.parametrize(
    "directory",
    ["ascertainment_depth", "depth_vs_severity", "exposure_routes",
     "socioeconomic_confounders", "completeness_stratification",
     "sih_severity_depth", "rs2024", "southeast_trend", "temporal_threads",
     "clinical_profile", "clinical_block_timing", "hospital_vs_territory",
     "sim_case_definition", "variable_catalogue"],
)
def test_current_scope_result_directory_exists(directory: str) -> None:
    """Every results directory the current argument depends on must be present."""
    d = ROOT / "data" / "results" / directory
    assert d.is_dir() and any(d.iterdir()), (
        f"data/results/{directory} is missing or empty; the link that cites it "
        "cannot be audited"
    )
