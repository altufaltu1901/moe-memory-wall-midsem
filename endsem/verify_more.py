"""Cross-check timing, traffic, and energy deltas across matching runs."""

import csv
import json
import math
from pathlib import Path


OUT = Path(__file__).resolve().parent / "artifacts"


def main():
    with (OUT / "summary.csv").open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    keys = ("workload", "bandwidth_gbps", "cache_gib", "transfer_pj_per_bit", "stall_watts")
    baseline = {tuple(r[k] for k in keys): r for r in all_rows if r["policy"] == "on_demand"}
    verified = []
    for row in all_rows:
        if row["policy"] == "on_demand":
            continue
        base = baseline[tuple(row[k] for k in keys)]
        stall_saved_ms = float(base["exposed_stall_ms_per_step"]) - float(row["exposed_stall_ms_per_step"])
        latency_saved_ms = float(base["mean_step_ms"]) - float(row["mean_step_ms"])
        assert math.isclose(latency_saved_ms, stall_saved_ms, abs_tol=1e-8)
        assert stall_saved_ms <= float(row["overlap_window_ms_per_step"]) + 1e-8
        extra_bits = (float(row["traffic_gb_per_token"]) -
                      float(base["traffic_gb_per_token"])) * 8e9
        stall_saved_s = stall_saved_ms * 1e-3 * int(row["steps"]) / int(row["tokens"])
        energy_saved = float(base["energy_j_per_token"]) - float(row["energy_j_per_token"])
        predicted = stall_saved_s * float(row["stall_watts"]) - extra_bits * float(
            row["transfer_pj_per_bit"]) * 1e-12
        assert math.isclose(energy_saved, predicted, abs_tol=1e-10)
        break_even_watts = extra_bits * float(row["transfer_pj_per_bit"]) * 1e-12 / stall_saved_s if (
            extra_bits > 0 and stall_saved_s > 0) else None
        verified.append(dict(workload=row["workload"], policy=row["policy"], sweep=row["sweep"],
                             bandwidth_gbps=float(row["bandwidth_gbps"]),
                             cache_gib=float(row["cache_gib"]), budget=int(row["budget"]),
                             window_fraction=float(row["window_fraction"]),
                             stall_saved_ms=stall_saved_ms, extra_bits_per_token=extra_bits,
                             energy_saved_j_per_token=energy_saved, break_even_watts=break_even_watts))
    (OUT / "crosschecks.json").write_text(json.dumps(verified, indent=2) + "\n", encoding="utf-8")
    print(f"Verified timing, overlap limit and energy balance for {len(verified)} policy comparisons")


if __name__ == "__main__":
    main()
