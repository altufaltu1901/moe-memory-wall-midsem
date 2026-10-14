"""Test adjacent-layer routing association in the public GLM sample.

Expert IDs are local to each layer, so this learns source-to-target expert
associations instead of comparing equal-numbered IDs across layers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "artifacts" / "real_trace"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)
rows = [json.loads(line) for line in (ROOT / "tokens.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(rows) == 1000
assert len({row["task_id"] for row in rows}) == 1
positions = [row["generated_position"] for row in rows]
assert positions == list(range(positions[0], positions[0] + len(rows)))

layers = len(rows[0]["route_selected"])
experts = 256
top_k = 8
train_n = 700
routes = np.asarray([row["route_selected"] for row in rows], dtype=np.int16)
assert routes.shape == (len(rows), layers, top_k)
assert np.all((routes >= 0) & (routes < experts))

source_recall = []
shuffled_recall = []
popularity_recall = []
rng = np.random.default_rng(20261014)
for layer in range(layers - 1):
    source = np.zeros((len(rows), experts), dtype=np.float64)
    target = np.zeros_like(source)
    source[np.arange(len(rows))[:, None], routes[:, layer, :]] = 1
    target[np.arange(len(rows))[:, None], routes[:, layer + 1, :]] = 1

    x_train, x_test = source[:train_n], source[train_n:]
    y_train, y_test = target[:train_n], target[train_n:]
    prior = y_train.mean(axis=0)
    # Shrink each source-expert conditional toward the target-layer popularity.
    source_count = x_train.sum(axis=0)
    coaccess = x_train.T @ y_train
    strength = 20.0
    conditional = (coaccess + strength * prior[None, :]) / (source_count[:, None] + strength)
    association = conditional - prior[None, :]

    def recall(x: np.ndarray) -> float:
        scores = prior[None, :] + (x @ association) / top_k
        predicted = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
        return float(np.take_along_axis(y_test, predicted, axis=1).sum() / (len(x) * top_k))

    source_recall.append(recall(x_test))
    shuffled_recall.append(recall(x_test[rng.permutation(len(x_test))]))
    popular = np.argpartition(prior, -top_k)[-top_k:]
    popularity_recall.append(float(y_test[:, popular].sum() / (len(x_test) * top_k)))

summary = {
    "source": "public GLM-5.2 variant routing sample; not DeepSeekV3",
    "method": "Adjacent layer l to l+1, same token; popularity-shrunk source-to-target expert associations",
    "training_tokens": train_n,
    "test_tokens": len(rows) - train_n,
    "transitions": layers - 1,
    "top_k": top_k,
    "metric": "Mean fraction of eight actual target-layer experts recovered",
    "source_conditioned_recall": float(np.mean(source_recall)),
    "shuffled_source_recall": float(np.mean(shuffled_recall)),
    "target_popularity_recall": float(np.mean(popularity_recall)),
    "transitions_above_popularity": int(np.sum(np.asarray(source_recall) > popularity_recall)),
    "transitions_above_shuffled": int(np.sum(np.asarray(source_recall) > shuffled_recall)),
    "source_conditioned_recall_by_transition": source_recall,
    "shuffled_source_recall_by_transition": shuffled_recall,
    "target_popularity_recall_by_transition": popularity_recall,
    "caveat": "One coding-task trajectory; this is held-out association evidence, not DeepSeekV3 routing or simulator latency.",
}
(OUT / "cross_layer_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

xs = np.arange(3, 3 + layers - 1)
plt.figure(figsize=(12, 5))
plt.plot(xs, source_recall, label="Source-layer experts -> next-layer experts", linewidth=1.5)
plt.plot(xs, shuffled_recall, label="Shuffled source tokens", linewidth=1.2)
plt.plot(xs, popularity_recall, label="Target-layer popularity only", linewidth=1.2)
plt.xlabel("Source MoE layer ID")
plt.ylabel("Held-out top-8 expert recall")
plt.title("Public GLM routing sample: adjacent-layer association")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "cross_layer_recall.png", dpi=180)
plt.close()
print(json.dumps({key: value for key, value in summary.items() if not key.endswith("_by_transition")}, indent=2))
