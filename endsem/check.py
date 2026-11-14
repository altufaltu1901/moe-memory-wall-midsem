"""Check transfer scheduling and validate outputs from the experiment matrix."""

import csv
import json
import math
from pathlib import Path

from replay import Link, Platform, replay


def check_link():
    platform = Platform(bandwidth=10, startup_ns=2, weight_bytes=100, cache_bytes=200)
    link = Link(platform)
    link.demand((1, 1), {})
    link.demand((1, 2), {})
    staged = link.prefetch([(2, 1), (2, 2)], 17)
    assert staged == {(2, 1): 12, (2, 2): 5}
    assert len(link.cache) + len(staged) == 2
    link.demand((2, 1), staged)
    link.demand((2, 2), staged)
    assert link.stats["stall_ns"] == 31
    assert link.stats["useful_prefetch_bytes"] == 130
    assert link.stats["demand_bytes"] == 270
    assert link.stats["ready_prefetches"] == 1
    assert link.stats["partial_prefetches_used"] == 1
    assert len(link.cache) == 2 and not staged

    # With no overlap or HBM cache, prefetching cannot change latency or traffic.
    batches = [[(0, {("seq", 0): {1}}), (1, {("seq", 0): {2}})]]
    baseline = replay(batches, [20], [0], platform, shared=False)
    no_overlap = replay(batches, [20], [0], platform, "oracle",
                        window_fraction=0, shared=False)
    assert baseline[0]["latency_ns"] == no_overlap[0]["latency_ns"]
    assert baseline[0]["traffic_bytes"] == no_overlap[0]["traffic_bytes"]


def check_outputs():
    out = Path(__file__).resolve().parent / "artifacts"
    with (out / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 77
    for row in rows:
        assert 0 <= float(row["recall"]) <= 1
        assert 0 <= float(row["precision"]) <= 1
        assert float(row["wasted_gb_per_token"]) >= -1e-10
        assert 0 <= float(row["exposed_stall_ms_per_step"]) <= float(row["mean_step_ms"])
    series = json.loads((out / "main_steps.json").read_text(encoding="utf-8"))
    for scenario in series:
        for row in scenario["steps"]:
            assert math.isclose(row["latency_ns"], row["compute_ns"] + row.get("stall_ns", 0))
            assert math.isclose(row["traffic_bytes"],
                                row.get("demand_bytes", 0) + row.get("prefetch_bytes", 0))
            assert row.get("useful_prefetch_bytes", 0) <= row.get("prefetch_bytes", 0) + 1e-5
    print("Transfer scheduling and all experiment results passed consistency checks.")


if __name__ == "__main__":
    check_link()
    check_outputs()
