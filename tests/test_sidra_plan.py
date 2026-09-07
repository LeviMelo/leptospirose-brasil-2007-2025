from __future__ import annotations

from brepi.sources.sidra.plan import RequestSpec, plan_requests


def test_planner_chunks_localities_even_when_cells_fit():
    spec = RequestSpec(
        agregado=9860,
        periods=("2010",),
        variables=("381",),
        localities=tuple(f"{i:07d}" for i in range(5570)),
        classifications={"11558": tuple(str(i) for i in range(8))},
    )
    requests = plan_requests(spec)
    assert len(requests) == 14
    assert max(len(request.localities) for request in requests) == 400
    assert sum(request.estimated_cells for request in requests) == 5570 * 8


def test_planner_packs_periods_inside_each_locality_chunk():
    spec = RequestSpec(
        agregado=1,
        periods=("2010", "2022"),
        variables=("1",),
        localities=tuple(str(i) for i in range(500)),
        classifications={"2": ("a", "b")},
    )
    requests = plan_requests(spec, max_localities=400)
    assert len(requests) == 2
    assert all(request.periods == ("2010", "2022") for request in requests)
