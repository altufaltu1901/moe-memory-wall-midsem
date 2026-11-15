"""Compare and validate the three matched GLM scheduling policies."""

import csv
import json
import math
from pathlib import Path

from run_glm_coupled import check


OUT = Path(__file__).resolve().parent / "artifacts"


def read_pair(path, shared):
    saved = json.loads((path / "summary.json").read_text())
    spec = saved["trace"]
    routes = (path / "input_routes.txt").read_text().splitlines()
    checked = check(path, routes, spec["moe_layers"], spec["sample_ratio"],
                    spec["chunk_bytes"], shared)
    for policy, row in checked.items():
        assert all(row[key] == value for key, value in saved[policy].items())
    return checked


def main():
    rows = []
    for offset in (700, 760):
        bounded_path = OUT / f"glm_bounded_{offset}"
        shared_path = OUT / f"glm_shared_{offset}"
        assert (bounded_path / "input_routes.txt").read_bytes() == (
            shared_path / "input_routes.txt").read_bytes()
        bounded = read_pair(bounded_path, shared=False)
        combined = read_pair(shared_path, shared=True)
        assert bounded["on_demand"] == combined["on_demand"]
        base = bounded["on_demand"]
        for name, run in (("on_demand", base), ("bounded", bounded["spatial"]),
                          ("bounded_and_shared", combined["spatial"])):
            saved_ms = base["latency_ms"] - run["latency_ms"]
            added_gb = run["traffic_gb_per_token"] - base["traffic_gb_per_token"]
            # Incremental offload energy. Core dynamic energy cancels for identical routes.
            modeled_energy_saved_j = saved_ms * 1e-3 * 100 - added_gb * 8e9 * 20e-12
            assert math.isclose(saved_ms, base["exposed_stall_ms"] - run["exposed_stall_ms"],
                                abs_tol=0.02)
            rows.append(dict(trace_offset=offset, policy=name,
                             latency_ms=run["latency_ms"], ms_saved=saved_ms,
                             percent_saved=100 * saved_ms / base["latency_ms"],
                             traffic_gb_per_token=run["traffic_gb_per_token"],
                             added_traffic_percent=100 * added_gb / base["traffic_gb_per_token"],
                             energy_saved_j_per_token_at_100w_20pj=modeled_energy_saved_j,
                             shared_prefetches=run["shared_prefetches"]))
        assert combined["spatial"]["shared_prefetches"] > 0
    with (OUT / "glm_schedule_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Verified both route windows, repeated baselines, transfer accounting and stalls")


if __name__ == "__main__":
    main()
