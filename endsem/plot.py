"""Render submission figures from endsem/artifacts/summary.csv."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "artifacts"


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    with (OUT / "summary.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    main_rows = {(r["workload"], r["policy"]): r for r in rows if r["sweep"] == "main"}
    workloads = ["deepseek_uniform", "deepseek_skew", "glm_proxy"]
    titles = ["DeepSeekV3 uniform synthetic", "DeepSeekV3 skewed synthetic", "GLM public routing trace"]
    policies = ["popularity", "temporal", "spatial", "hybrid", "adaptive", "oracle"]
    colors = ["#5b6da8", "#338677", "#ab6a3b", "#644d80", "#00729c", "#777777"]
    x = np.arange(len(policies))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True, layout="constrained")
    for ax, workload, title in zip(axes, workloads, titles):
        selected = [main_rows[workload, p] for p in policies]
        for i, (color, r) in enumerate(zip(colors, selected)):
            ax.bar(i - .18, 100 * float(r["recall"]), .36, color=color)
            ax.bar(i + .18, 100 * float(r["precision"]), .36, color=color, alpha=.42)
        ax.set_xticks(x, policies, rotation=45, ha="right")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Expert overlap (%)")
    fig.suptitle("Prediction overlap: recall (solid), precision (light)")
    save(fig, "prediction_overlap.png")

    metrics = [("mean_step_ms", "Step latency saved"),
               ("exposed_stall_ms_per_step", "Exposed stall saved"),
               ("traffic_gb_per_token", "Extra traffic"),
               ("energy_j_per_token", "Estimated energy saved")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, (metric, title) in zip(axes.flat, metrics):
        sign = 1 if metric == "traffic_gb_per_token" else -1
        for index, workload in enumerate(workloads):
            scale = float(main_rows[workload, "on_demand"][metric])
            values = [sign * 100 * (float(main_rows[workload, p][metric]) / scale - 1)
                      for p in policies]
            ax.bar(x + (index - 1) * .24, values, .24, label=titles[index])
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set_title(title)
        ax.set_ylabel("Change vs on demand (%)")
        ax.set_xticks(x, policies, rotation=30, ha="right")
        ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(fontsize=7)
    save(fig, "system_tradeoffs.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True, layout="constrained")
    for ax, workload, title in zip(axes, workloads, titles):
        baseline = main_rows[workload, "on_demand"]
        selected = [main_rows[workload, "hybrid"]] + [
            r for r in rows if r["workload"] == workload and r["sweep"] == "budget"]
        selected.sort(key=lambda r: int(r["budget"]))
        for r in selected:
            latency = 100 * (1 - float(r["mean_step_ms"]) / float(baseline["mean_step_ms"]))
            traffic = 100 * (float(r["traffic_gb_per_token"]) /
                             float(baseline["traffic_gb_per_token"]) - 1)
            ax.scatter(traffic, latency, color="#00729c")
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.axvline(0, color="#555555", linewidth=.8)
        ax.set_title(title)
        ax.set_xlabel("Extra traffic vs on demand (%)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Latency saved vs on demand (%)")
    fig.suptitle("Fixed hybrid budgets (1, 4, 8, 16, 32 candidates; points can coincide)")
    save(fig, "budget_tradeoff.png")


if __name__ == "__main__":
    main()
