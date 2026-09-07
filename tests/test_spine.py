from datetime import date
import polars as pl, pytest
from brepi.panel.spine import SpineSpec, build_spine, attach, PanelBuild, sparsity_report

MUNS = ("3550308", "1200401", "4314902")

def spec(grain="month"):
    return SpineSpec(municipalities=MUNS, start=date(2007,1,1), end=date(2008,12,1), grain=grain)

def test_spine_is_complete_and_indexed():
    s = build_spine(spec())
    assert s.height == 3*24 == spec().n_rows
    assert s["time_index"].max() == 23
    assert s.select("munic_code","period").unique().height == s.height

def test_spine_rejects_6digit_codes():
    with pytest.raises(ValueError, match="7-digit"):
        build_spine(SpineSpec(municipalities=("355030",), start=date(2007,1,1), end=date(2007,2,1)))

def test_attach_preserves_zeros_and_reports_coverage():
    s = build_spine(spec())
    src = pl.DataFrame({"munic_code":["3550308"],"period":[date(2007,3,1)],"cases":[7]})
    out, rep = attach(s, src, name="sinan", fill={"cases":0})
    assert out.height == s.height
    assert out["cases"].sum() == 7
    assert out.filter(pl.col("cases")==0).height == s.height-1
    assert rep.matched_rows == 1 and rep.unmatched_source_keys == 0

def test_attach_detects_key_convention_mismatch():
    s = build_spine(spec())
    src = pl.DataFrame({"munic_code":["355030"],"period":[date(2007,3,1)],"cases":[7]})
    _, rep = attach(s, src, name="bad")
    assert rep.matched_rows == 0 and rep.unmatched_source_keys == 1

def test_attach_rejects_duplicates():
    s = build_spine(spec())
    src = pl.DataFrame({"munic_code":["3550308"]*2,"period":[date(2007,3,1)]*2,"cases":[1,2]})
    with pytest.raises(ValueError, match="duplicate rows"):
        attach(s, src, name="dup")

def test_panelbuild_min_coverage_gate():
    s = build_spine(spec())
    src = pl.DataFrame({"munic_code":["355030"],"period":[date(2007,3,1)],"x":[1.0]})
    with pytest.raises(ValueError, match="coverage"):
        PanelBuild(s).add(src, name="cov", min_coverage=0.5)

def test_sparsity_report():
    s = build_spine(spec())
    src = pl.DataFrame({"munic_code":["3550308","3550308"],"period":[date(2007,3,1),date(2007,4,1)],"cases":[10,2]})
    out,_ = attach(s, src, name="n", fill={"cases":0})
    r = sparsity_report(out)
    assert r["municipalities_ever_reporting"] == 1
    assert r["municipalities_never_reporting"] == 2
    assert r["nonzero_cells"] == 2
    assert r["total_count"] == 12
