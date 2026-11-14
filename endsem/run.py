"""Run the endsem experiment matrix from the repository root."""

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean

from replay import Platform, replay


HERE = Path(__file__).resolve().parent
MIDSEM = HERE.parent / "midsem" / "artifacts"
OUT = HERE / "artifacts"


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def synthetic(tag, platform):
    directory = MIDSEM / tag
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for row in csv_rows(directory / "routes.csv"):
        step = int(row["token_position"])
        layer = int(row["layer"])
        groups[step][layer][(row["sequence_id"], step)].add(int(row["expert_id"]))
    batches = [sorted(layers.items()) for layers in groups.values()]
    metrics = csv_rows(next(directory.glob("*decode.csv")))
    logged = defaultdict(Counter)
    order = defaultdict(list)
    for line in (directory / "stdout.log").read_text(encoding="utf-8").splitlines():
        if not line.startswith("OFFCHIP_LOAD,"):
            continue
        _, step, key, hits, misses, evictions, size, ns = line.split(",")
        step = int(step)
        layer = int(re.search(r"MoE_decoder_(\d+)", key)[1])
        order[step].append((layer, key.endswith(":shared")))
        logged[step].update(hits=int(hits), misses=int(misses), evictions=int(evictions),
                            traffic_bytes=int(size), stall_ns=float(ns))
    expected_order = [[(layer, shared) for layer, _ in batch for shared in (False, True)]
                      for batch in batches]
    if list(order.values()) != expected_order or len(batches) != len(metrics):
        raise ValueError("LLMSimulator routes and load stages do not align")
    for batch, metric in zip(batches, metrics):
        if int(metric["batchsize"]) != len(batch[0][1]):
            raise ValueError("Unexpected number of decoded tokens")
    compute = [float(row["latency"]) - logged[i]["stall_ns"] for i, row in enumerate(metrics)]
    energy = [float(row["total_energy"]) * 1e-9 for row in metrics]
    baseline = replay(batches, compute, energy, platform)
    for i, (row, expected) in enumerate(zip(baseline, logged.values())):
        for field in ("hits", "misses", "evictions", "traffic_bytes", "stall_ns"):
            if not math.isclose(row.get(field, 0), expected[field], rel_tol=3e-6):
                raise ValueError(f"Baseline {tag}, step {i}, {field}: "
                                 f"{row.get(field, 0)} != {expected[field]}")
        if not math.isclose(row["latency_ns"], float(metrics[i]["latency"]), rel_tol=3e-6):
            raise ValueError("Replayed latency differs from LLMSimulator")
    return batches, compute, energy


def glm():
    file = MIDSEM / "real_trace" / "tokens.jsonl"
    samples = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
    if len({row["task_id"] for row in samples}) != 1:
        raise ValueError("Expected a single GLM trajectory")
    if [row["generated_position"] for row in samples] != list(range(len(samples))):
        raise ValueError("Expected consecutive token positions")
    return [[(layer, {(row["task_id"], row["generated_position"]): set(experts)})
             for layer, experts in enumerate(row["route_selected"])]
            for row in samples]


def summarize(rows):
    total = Counter()
    for row in rows:
        total.update(row)
    tokens = total["tokens"]
    latency_ms = mean(row["latency_ns"] for row in rows) / 1e6
    energy_j = total["energy_j"] / tokens
    return dict(steps=len(rows), tokens=tokens, recall=total["overlap"] / total["actual"],
                precision=total["overlap"] / max(1, total["predicted"]),
                predictions_per_layer=total["predicted"] / total["route_events"],
                mean_step_ms=latency_ms,
                p95_step_ms=sorted(row["latency_ns"] for row in rows)[math.ceil(.95 * len(rows)) - 1] / 1e6,
                throughput_tokens_s=tokens / (total["latency_ns"] * 1e-9),
                amortized_ms_per_token=total["latency_ns"] / tokens / 1e6,
                overlap_window_ms_per_step=total["overlap_window_ns"] / len(rows) / 1e6,
                exposed_stall_ms_per_step=total["stall_ns"] / len(rows) / 1e6,
                expert_load_service_ms_per_step=(total["demand_service_ns"] +
                                                 total["prefetch_service_ns"]) / len(rows) / 1e6,
                traffic_gb_per_token=total["traffic_bytes"] / tokens / 1e9,
                speculative_gb_per_token=total["prefetch_bytes"] / tokens / 1e9,
                wasted_gb_per_token=(total["prefetch_bytes"] - total["useful_prefetch_bytes"]) / tokens / 1e9,
                ready_prefetches=int(total["ready_prefetches"]),
                partial_prefetches_used=int(total["partial_prefetches_used"]),
                energy_j_per_token=energy_j, edp_j_ms_per_token=energy_j * latency_ms,
                core_energy_j_per_token=total["core_energy_j"] / tokens,
                transfer_energy_j_per_token=total["transfer_energy_j"] / tokens,
                stall_energy_j_per_token=total["stall_energy_j"] / tokens)


def main():
    platform = Platform()
    primary = synthetic("offchip", platform)
    skew = synthetic("cache_check", platform)
    output = []
    scenarios = []

    def evaluate(workload, inputs, hardware, policy, budget=8, window=1.0,
                 shared=True, score_from=0, sweep="main"):
        rows = replay(*inputs, hardware, policy, budget, window, shared, score_from)
        summary = summarize(rows)
        output.append(dict(workload=workload, sweep=sweep, policy=policy, budget=budget,
                           bandwidth_gbps=hardware.bandwidth, cache_gib=hardware.cache_bytes / 2**30,
                           window_fraction=window, transfer_pj_per_bit=hardware.transfer_pj_per_bit,
                           stall_watts=hardware.stall_watts, **summary))
        if sweep == "main":
            scenarios.append(dict(workload=workload, policy=policy, steps=rows))

    policies = ("on_demand", "popularity", "temporal", "spatial", "hybrid", "adaptive", "oracle")
    for label, inputs in [("deepseek_uniform", primary), ("deepseek_skew", skew)]:
        for policy in policies:
            evaluate(label, inputs, platform, policy)
        for bandwidth in (16, 64):
            for policy in ("on_demand", "adaptive", "oracle"):
                evaluate(label, inputs, replace(platform, bandwidth=bandwidth), policy, sweep="bandwidth")
        for cache in (8, 16, 48):
            for policy in ("on_demand", "adaptive"):
                evaluate(label, inputs, replace(platform, cache_bytes=cache * 2**30), policy, sweep="cache")
        for budget in (1, 4, 16, 32):
            evaluate(label, inputs, platform, "hybrid", budget=budget, sweep="budget")
        for window in (.25, .50):
            for policy in ("adaptive", "oracle"):
                evaluate(label, inputs, platform, policy, window=window, sweep="overlap")
        for transfer, watts in ((10, 50), (10, 200), (40, 50), (40, 200)):
            for policy in ("on_demand", "adaptive"):
                evaluate(label, inputs, replace(platform, transfer_pj_per_bit=transfer,
                                               stall_watts=watts), policy, sweep="energy")

    public = glm()
    # A proxy used to demonstrate replay behavior under observed GLM routing.
    # GLM prediction scores use real route IDs. Its timing and energy are illustrative.
    glm_compute_ns = mean(primary[1]) / len(primary[0][0]) * len(public[0])
    inputs = public, [glm_compute_ns] * len(public), [0] * len(public)
    for policy in policies:
        evaluate("glm_proxy", inputs, platform, policy, shared=False, score_from=700)
    OUT.mkdir(exist_ok=True)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    (OUT / "main_steps.json").write_text(json.dumps(scenarios, separators=(",", ":")) + "\n",
                                         encoding="utf-8")
    print(f"Validated both LLMSimulator baselines. Wrote {len(output)} comparisons.")


if __name__ == "__main__":
    main()
