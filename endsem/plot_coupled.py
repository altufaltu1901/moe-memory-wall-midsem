"""Plot the verified matched native LLMSimulator comparisons."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from check_coupled import ROOT


def main():
    rows = json.loads((ROOT / "summary.json").read_text())
    cases = ("ram1", "ram4", "skew4")
    labels = ["Uniform, 1 channel", "Uniform, 4 channels", "Skewed, 4 channels"]
    metrics = (
        ("latency_ms_per_step", "Decode latency"),
        ("traffic_gb_per_token", "Expert traffic"),
        ("modeled_energy_j_per_token", "Modeled energy"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout="constrained")
    for axis, (field, title) in zip(axes, metrics):
        changes = [100 * (rows[key + "_prefetch"][field] / rows[key][field] - 1)
                   for key in cases]
        axis.bar(labels, changes, color=["#406780", "#447f77", "#8d7661"])
        axis.axhline(0, linewidth=.8, color="black")
        axis.set_title(title)
        axis.set_ylabel("Change with prefetch (%)")
        axis.tick_params(axis="x", rotation=25)
    fig.savefig(ROOT / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
