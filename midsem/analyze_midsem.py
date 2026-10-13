"""Reproduce the midsem plots from the single-GPU simulator artifacts.

Usage: python analyze_midsem.py
All times in the simulator CSV and OFFCHIP_LOAD records are nanoseconds.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts" / "offchip"
OUT = ROOT / "artifacts" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

with (SOURCE / "routes.csv").open(newline="", encoding="utf-8") as f:
    routes = list(csv.DictReader(f))

route_sets: dict[tuple[int, int, int], set[int]] = defaultdict(set)
source_counts = Counter()
for row in routes:
    key = (int(row["sequence_id"]), int(row["token_position"]), int(row["layer"]))
    route_sets[key].add(int(row["expert_id"]))
    source_counts[row["source"]] += 1

layers = sorted({key[2] for key in route_sets})
num_experts = max(int(row["expert_id"]) for row in routes) + 1
top_k_counts = Counter(map(len, route_sets.values()))
assert len(top_k_counts) == 1, f"Inconsistent top-k: {top_k_counts}"
top_k = next(iter(top_k_counts))

frequency = np.zeros((len(layers), num_experts), dtype=int)
layer_index = {layer: i for i, layer in enumerate(layers)}
for (_, _, layer), selected in route_sets.items():
    for expert in selected:
        frequency[layer_index[layer], expert] += 1

temporal_overlap = []
for (seq, token, layer), selected in route_sets.items():
    previous = route_sets.get((seq, token - 1, layer))
    if previous is not None:
        temporal_overlap.append(len(previous & selected))

expected_overlap = top_k * top_k / num_experts
observed_overlap = mean(temporal_overlap)

load_by_step: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
with (SOURCE / "stdout.log").open(encoding="utf-8") as f:
    for line in f:
        if not line.startswith("OFFCHIP_LOAD,"):
            continue
        _, step, layer, hits, misses, evictions, bytes_loaded, ns = line.strip().split(",")
        s = load_by_step[int(step)]
        s["hits"] += int(hits)
        s["misses"] += int(misses)
        s["evictions"] += int(evictions)
        s["bytes"] += int(bytes_loaded)
        s["load_ns"] += float(ns)
        s["moe_layers"] += 1

metric_file = next(SOURCE.glob("*decode.csv"))
with metric_file.open(newline="", encoding="utf-8") as f:
    metrics = list(csv.DictReader(f))
assert len(metrics) == len(load_by_step), "Load records and simulator rows differ"

iterations = []
for step, row in enumerate(metrics):
    load = load_by_step[step]
    expert_ffn_ns = float(row["expert_ffn"])
    latency_ns = float(row["latency"])
    compute_ns = expert_ffn_ns - load["load_ns"]
    assert compute_ns >= 0, "Load exceeds expert-FFN total; check time accounting"
    iterations.append({
        "step": step,
        "latency_ms": latency_ns / 1e6,
        "expert_load_ms": load["load_ns"] / 1e6,
        "expert_compute_ms": compute_ns / 1e6,
        "expert_hits": int(load["hits"]),
        "expert_misses": int(load["misses"]),
        "evictions": int(load["evictions"]),
        "bytes_loaded": int(load["bytes"]),
        "moe_layers": int(load["moe_layers"]),
    })

summary = {
    "routing_source": dict(source_counts),
    "num_layers": len(layers),
    "num_experts": num_experts,
    "top_k": top_k,
    "route_groups": len(route_sets),
    "temporal_pairs": len(temporal_overlap),
    "mean_temporal_overlap": observed_overlap,
    "random_topk_expected_overlap": expected_overlap,
    "mean_load_ms": mean(x["expert_load_ms"] for x in iterations),
    "mean_expert_compute_ms": mean(x["expert_compute_ms"] for x in iterations),
    "mean_total_latency_ms": mean(x["latency_ms"] for x in iterations),
    "mean_load_to_compute_ratio": mean(x["expert_load_ms"] for x in iterations) /
        mean(x["expert_compute_ms"] for x in iterations),
    "mean_load_fraction_of_token_latency": mean(x["expert_load_ms"] / x["latency_ms"]
                                                 for x in iterations),
    "cache_hit_rate": sum(x["expert_hits"] for x in iterations) /
        sum(x["expert_hits"] + x["expert_misses"] for x in iterations),
    "iterations": iterations,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

plt.figure(figsize=(11, 5))
plt.imshow(frequency, aspect="auto", interpolation="nearest", cmap="viridis")
plt.colorbar(label="Selections among token observations per layer")
plt.xlabel("Expert ID (local to each layer)")
plt.ylabel("MoE layer index")
plt.title("Synthetic DeepSeekV3 routing: expert frequency")
plt.tight_layout()
plt.savefig(OUT / "expert_frequency_heatmap.png", dpi=180)
plt.close()

observed = Counter(temporal_overlap)
denominator = math.comb(num_experts, top_k)
expected_prob = [math.comb(top_k, k) * math.comb(num_experts - top_k, top_k - k) /
                 denominator for k in range(top_k + 1)]
xs = np.arange(top_k + 1)
plt.figure(figsize=(8, 5))
plt.bar(xs - 0.2, [observed[k] / len(temporal_overlap) for k in xs],
        width=0.4, label="Observed synthetic trace")
plt.bar(xs + 0.2, expected_prob, width=0.4,
        label="Independent uniform top-k expectation")
plt.xlim(-0.6, min(4.6, top_k + 0.6))
plt.xlabel("Shared experts at same layer in consecutive tokens")
plt.ylabel("Fraction of token pairs")
plt.title("Temporal overlap against the random-routing null")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "temporal_overlap.png", dpi=180)
plt.close()

first_layer = layers[0]
layer_routes = [selected for (_, _, layer), selected in route_sets.items()
                if layer == first_layer]
common = [expert for expert, _ in Counter(
    expert for selected in layer_routes for expert in selected).most_common(20)]
coaccess = np.zeros((len(common), len(common)), dtype=int)
for selected in layer_routes:
    for i, a in enumerate(common):
        for j, b in enumerate(common):
            if a != b and a in selected and b in selected:
                coaccess[i, j] += 1
plt.figure(figsize=(8, 7))
plt.imshow(coaccess, cmap="magma", interpolation="nearest")
plt.colorbar(label="Tokens selecting both experts")
plt.xticks(range(len(common)), common, rotation=90)
plt.yticks(range(len(common)), common)
plt.xlabel("Expert ID")
plt.ylabel("Expert ID")
plt.title(f"Within-layer co-access, MoE layer {first_layer} (synthetic)")
plt.tight_layout()
plt.savefig(OUT / "coaccess_heatmap.png", dpi=180)
plt.close()

steps = [x["step"] for x in iterations]
loads = [x["expert_load_ms"] for x in iterations]
computes = [x["expert_compute_ms"] for x in iterations]
plt.figure(figsize=(8, 5))
plt.bar(steps, loads, label="Expert load from off-chip DRAM")
plt.bar(steps, computes, bottom=loads, label="Expert FFN execution")
plt.xlabel("Decode iteration")
plt.ylabel("Simulated time (ms)")
plt.title("Single-GPU MoE: expert loading versus execution")
plt.xticks(steps)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "load_vs_compute.png", dpi=180)
plt.close()

# Hold model, routes, and cache fixed; vary only the analytical off-chip link.
sweep = []
for bandwidth, folder in [(16, "offchip_bw16"), (32, "offchip"),
                          (64, "offchip_bw64")]:
    path = ROOT / "artifacts" / folder
    if not path.exists():
        continue
    with next(path.glob("*decode.csv")).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    log_lines = (path / "stdout.log").read_text(encoding="utf-8").splitlines()
    loads_ns = [float(line.rsplit(",", 1)[1]) for line in log_lines
                if line.startswith("OFFCHIP_LOAD,")]
    assert len(loads_ns) == 57 * 5
    sweep.append({
        "bandwidth_GBps": bandwidth,
        "mean_token_latency_ms": mean(float(row["latency"]) for row in rows) / 1e6,
        "mean_expert_load_ms": sum(loads_ns) / 5e6,
    })
if sweep:
    (OUT / "bandwidth_sweep.json").write_text(json.dumps(sweep, indent=2) + "\n",
                                               encoding="utf-8")
    plt.figure(figsize=(7, 5))
    plt.plot([x["bandwidth_GBps"] for x in sweep],
             [x["mean_token_latency_ms"] for x in sweep], marker="o",
             label="Total token latency")
    plt.plot([x["bandwidth_GBps"] for x in sweep],
             [x["mean_expert_load_ms"] for x in sweep], marker="s",
             label="Expert loading")
    plt.xlabel("Modeled off-chip bandwidth (GB/s)")
    plt.ylabel("Mean simulated time per decode iteration (ms)")
    plt.title("Single-GPU bandwidth sensitivity (same synthetic routes)")
    plt.xticks([x["bandwidth_GBps"] for x in sweep])
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "bandwidth_sweep.png", dpi=180)
    plt.close()

print(json.dumps({k: v for k, v in summary.items() if k != "iterations"}, indent=2))
