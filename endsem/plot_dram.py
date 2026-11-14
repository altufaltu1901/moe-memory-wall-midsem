"""Plot DRAM read throughput and its effect on the replay."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent / "artifacts" / "ramulator"


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    full = [r for r in read("measurements.csv") if int(r["size"]) == 44_040_192
            and int(r["offset"]) == 0]
    replay = [r for r in read("replay.csv") if r["topology"] == "pipelined"
              and int(r["offset"]) == 0 and r["workload"] == "offchip"]
    fig, (raw, result) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for mapper, label in [("RoBaRaCoCh", "Channel interleaved"),
                          ("ChRaBaRoCo", "Channel contiguous")]:
        rows = sorted((r for r in full if r["mapper"] == mapper),
                      key=lambda r: int(r["channels"]))
        channels = [int(r["channels"]) for r in rows]
        raw.plot(channels, [float(r["gbps"]) for r in rows], "o-", label=label)
        baseline = {r["channels"]: float(r["mean_step_ms"]) for r in replay
                    if r["mapper"] == mapper and r["policy"] == "on_demand"}
        for policy in ("spatial", "oracle"):
            selected = {r["channels"]: float(r["mean_step_ms"]) for r in replay
                        if r["mapper"] == mapper and r["policy"] == policy}
            savings = [100 * (1 - selected[str(c)] / baseline[str(c)]) for c in channels]
            result.plot(channels, savings, "o-" if policy == "oracle" else "o--",
                        label=f"{label}, {policy}")
    raw.axhline(32, color="gray", linestyle=":", label="Host link")
    raw.set(ylabel="Completed read throughput (GB/s)", title="Ramulator full expert read")
    result.set(ylabel="Decode time saved (%)", title="Calibrated DeepSeekV3 replay")
    for ax in (raw, result):
        ax.set(xlabel="DDR4 channels", xticks=[1, 2, 4])
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(ROOT / "dram_sensitivity.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
