"""Verify the paired in-process LLMSimulator experiments and their accounting."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent / "artifacts" / "coupled"
CASES = ("control", "ram1", "ram1_prefetch", "ram4", "ram4_prefetch", "skew4", "skew4_prefetch")


def load_case(name, root):
    root = root / name
    assert (root / "exit_code.txt").read_text().strip() == "0"
    with (root / "metrics.csv").open(newline="") as handle:
        metrics = list(csv.DictReader(handle))
    assert len(metrics) == 5
    demand_bytes = speculative_bytes = 0
    stalls = [0.0] * len(metrics)
    dram_reads = 0
    formatted_stages = 0
    dram_bytes = 0
    predictions = used = rejected = 0
    for line in (root / "stdout.log").read_text().splitlines():
        fields = line.split(",")
        if fields[0] == "EXPERT_DRAM":
            _, _, size, issued, completed, cycles, clock = fields
            expected = math.ceil(int(size) / (64 * 1024))
            assert int(issued) == int(completed) == expected
            assert int(cycles) >= expected / 8 and float(clock) > 0
            dram_reads += int(completed)
            dram_bytes += int(size)
        elif fields[0] == "OFFCHIP_LOAD":
            formatted_stages += 1
            stalls[int(fields[1])] += float(fields[-1])
            demand_bytes += int(fields[-2])
        elif fields[0] == "EXPERT_PREFETCH":
            predictions += 1
            speculative_bytes += int(fields[2])
        elif fields[0] == "EXPERT_PREFETCH_USED":
            used += 1
        elif fields[0] == "EXPERT_PREFETCH_UNUSED":
            rejected += 1
    assert (dram_reads > 0) == (name != "control")
    if dram_reads:
        assert dram_bytes == demand_bytes + speculative_bytes
    assert predictions == used + rejected
    if not name.endswith("_prefetch"):
        assert predictions == 0
    elif name.startswith("ram"):
        assert predictions > 0
    latency = [float(row["latency"]) for row in metrics]
    assert all(0 <= stall <= span for stall, span in zip(stalls, latency))
    tokens = sum(int(row["batchsize"]) for row in metrics)
    core_j = sum(float(row["total_energy"]) * 1e-9 for row in metrics)
    traffic = demand_bytes + speculative_bytes
    assert traffic > 0 and tokens == 20 and formatted_stages == 570
    assert all(int(row["OOM"]) == 0 for row in metrics)
    return {
        "latency_ms_per_step": mean(latency) / 1e6,
        "latency_ms_per_token_amortized": sum(latency) / tokens / 1e6,
        "stall_ms_per_step": mean(stalls) / 1e6,
        "traffic_gb_per_token": traffic / tokens / 1e9,
        "transfer_energy_j_per_token_at_20pj_per_bit": traffic * 8 * 20e-12 / tokens,
        "stall_energy_j_per_token_at_100w": sum(stalls) * 1e-9 * 100 / tokens,
        "core_energy_j_per_token": core_j / tokens,
        "modeled_energy_j_per_token": (core_j + traffic * 8 * 20e-12 +
                                        sum(stalls) * 1e-9 * 100) / tokens,
        "dram_reads": dram_reads,
        "prefetches": predictions,
        "used_prefetches": used,
        "prefetch_precision": used / predictions if predictions else None,
        "unused_prefetches": rejected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=ROOT)
    root = parser.parse_args().root
    summary = {name: load_case(name, root) for name in CASES}
    for base in ("ram1", "ram4", "skew4"):
        prefetch = base + "_prefetch"
        assert (root / base / "routes.csv").read_bytes() == (root / prefetch / "routes.csv").read_bytes()
        baseline, speculative = summary[base], summary[prefetch]
        latency_delta = speculative["latency_ms_per_step"] - baseline["latency_ms_per_step"]
        stall_delta = speculative["stall_ms_per_step"] - baseline["stall_ms_per_step"]
        # The C++ log writes six significant digits, so the summed stall has
        # more rounding error than the high-precision latency CSV.
        assert math.isclose(latency_delta, stall_delta, abs_tol=0.03)
    assert (root / "control/routes.csv").read_bytes() == (root / "ram4/routes.csv").read_bytes()
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Verified all cases, exact matched routes, completed requests, traffic and latency accounting")


if __name__ == "__main__":
    main()
