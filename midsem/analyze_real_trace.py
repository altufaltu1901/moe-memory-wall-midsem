"""Analyze a separate public GLM routing sample; never merge with DeepSeek timing."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "artifacts" / "real_trace"
OUT = SRC / "analysis"
OUT.mkdir(exist_ok=True)

rows = [json.loads(line) for line in (SRC / "tokens.jsonl").read_text(encoding="utf-8").splitlines()]
assert rows and all(row["task_id"] == rows[0]["task_id"] for row in rows)
positions = [row["generated_position"] for row in rows]
assert positions == list(range(positions[0], positions[0] + len(rows)))
routes = [[set(layer) for layer in row["route_selected"]] for row in rows]
layer_count = len(routes[0])
top_k = len(routes[0][0])
assert all(len(token) == layer_count and all(len(x) == top_k for x in token) for token in routes)
expert_count = 256  # dataset card states 256 layer-local expert IDs
assert all(0 <= expert < expert_count for token in routes for layer in token for expert in layer)
n = len(routes)

frequency = np.zeros((layer_count, expert_count), dtype=int)
observed = []
frequency_null = []
for layer in range(layer_count):
    for token in routes:
        for expert in token[layer]:
            frequency[layer, expert] += 1
    observed.append(mean(len(routes[t - 1][layer] & routes[t][layer]) for t in range(1, n)))
    # Expected overlap after permuting token order within this layer. This
    # preserves its exact expert frequencies while removing time order.
    frequency_null.append(
        sum(int(c) * (int(c) - 1) for c in frequency[layer]) / (n * (n - 1))
    )

first_layer = 0  # dataset layers are 3..77; list index 0 corresponds to layer 3
common = [int(e) for e in np.argsort(frequency[first_layer])[::-1][:20]]
coaccess = np.zeros((len(common), len(common)), dtype=int)
for token in routes:
    active = token[first_layer]
    for i, a in enumerate(common):
        for j, b in enumerate(common):
            if i != j and a in active and b in active:
                coaccess[i, j] += 1

summary = {
    "source": "mistrjirka/Predicting-Experts-for-MOE-Suite",
    "model_scope": "GLM-5.2 variant routing; separate from DeepSeekV3 timing model",
    "task_id": rows[0]["task_id"],
    "tokens": n,
    "moe_layers": layer_count,
    "layer_ids": "3..77",
    "experts_per_layer": expert_count,
    "top_k": top_k,
    "adjacent_token_pairs": (n - 1) * layer_count,
    "mean_adjacent_overlap": mean(observed),
    "mean_frequency_preserving_null": mean(frequency_null),
    "independent_uniform_topk_null": top_k * top_k / expert_count,
    "layers_above_frequency_null": sum(a > b for a, b in zip(observed, frequency_null)),
    "mean_overlap_by_layer": observed,
    "frequency_null_by_layer": frequency_null,
    "first_layer_top_20_experts": common,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

plt.figure(figsize=(12, 6))
plt.imshow(frequency, aspect="auto", interpolation="nearest", cmap="viridis")
plt.colorbar(label="Selections in 1,000 tokens")
plt.xlabel("Expert ID (local to each layer)")
plt.ylabel("MoE layer ID")
plt.yticks(range(0, layer_count, 10), [3 + x for x in range(0, layer_count, 10)])
plt.title("Public GLM routing sample: expert frequency by layer")
plt.tight_layout()
plt.savefig(OUT / "expert_frequency_heatmap.png", dpi=180)
plt.close()

plt.figure(figsize=(11, 5))
xs = np.arange(3, 3 + layer_count)
plt.plot(xs, observed, label="Observed adjacent-token overlap", linewidth=1.5)
plt.plot(xs, frequency_null, label="Frequency-preserving shuffled-order expectation", linewidth=1.5)
plt.axhline(top_k * top_k / expert_count, color="gray", linestyle="--",
            label="Uniform top-8/256 expectation")
plt.xlabel("MoE layer ID")
plt.ylabel("Mean shared experts out of 8")
plt.title("Public GLM routing sample: temporal locality")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "temporal_overlap_by_layer.png", dpi=180)
plt.close()

plt.figure(figsize=(8, 7))
plt.imshow(coaccess, cmap="magma", interpolation="nearest")
plt.colorbar(label="Tokens selecting both experts")
plt.xticks(range(len(common)), common, rotation=90)
plt.yticks(range(len(common)), common)
plt.xlabel("Expert ID")
plt.ylabel("Expert ID")
plt.title("Public GLM routing sample: within-layer co-access (layer 3)")
plt.tight_layout()
plt.savefig(OUT / "coaccess_heatmap.png", dpi=180)
plt.close()

print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))
