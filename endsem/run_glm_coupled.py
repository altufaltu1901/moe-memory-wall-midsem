"""Run recorded GLM routes through the coupled simulator's DeepSeek timing proxy."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
import subprocess

from replay import History
from run import glm


def prepare(batches, offset, steps, output):
    history = History()
    serialized_routes, serialized_plan = [], []
    for index, batch in enumerate(batches[:offset + steps]):
        context = {}
        for layer, routes in batch:
            history.observe(layer, routes, context)
            context = routes
            if index >= offset:
                choices = sorted(next(iter(routes.values())))
                serialized_routes.append(" ".join(map(str, choices)))
                # Only the current layer and previously observed targets are visible.
                next_layer = layer + 1
                predictions = history.predict(next_layer, routes, context, "spatial", 8, 8, set()) if (
                    next_layer < len(batch)
                ) else []
                serialized_plan.append(" ".join(map(str, predictions)))
    for name, rows in (("input_routes.txt", serialized_routes),
                       ("prefetch_plan.txt", serialized_plan)):
        (output / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return serialized_routes


def run(executable, output, name, steps, layers, ratio):
    root = Path(__file__).resolve().parents[1]
    case = output / name
    case.mkdir()
    data = executable.parent.parent / "data" / f"glm_{output.name}_{name}"
    data.mkdir(parents=True, exist_ok=True)
    if list(data.glob("*.csv")):
        raise ValueError(f"Old metrics in {data}")
    text = (root / "config_offchip.yaml").read_text()
    settings = f"\n    ramulator_channels: 4\n    ramulator_sample_ratio: {ratio}"
    settings += f"\n    prefetch: {str(name == 'spatial').lower()}"
    text = text.replace("  offchip_experts:", "  offchip_experts:" + settings)
    text = text.replace("  model_name: deepseekV3", f"  model_name: deepseekV3\n  recorded_moe_layers: {layers}")
    for key, value in (("max_batch_size", 1), ("iter", steps), ("output_len", steps + 32),
                       ("output_directory", f"../data/glm_{output.name}_{name}")):
        text = re.sub(rf"(?m)^  {key}:.*$", f"  {key}: {value}", text)
    config = case / "config.yaml"
    config.write_text(text)
    env = dict(os.environ, MOE_INPUT_ROUTES=str(output / "input_routes.txt"),
               MOE_TRACE_PATH=str(case / "routes.csv"))
    env.pop("MOE_PREFETCH_PLAN", None)
    if name == "spatial":
        env["MOE_PREFETCH_PLAN"] = str(output / "prefetch_plan.txt")
    with (case / "stdout.log").open("wb") as log, (case / "stderr.log").open("wb") as err:
        result = subprocess.run([str(executable), str(config)], cwd=executable.parent,
                                env=env, stdout=log, stderr=err, timeout=1800)
    (case / "exit_code.txt").write_text(f"{result.returncode}\n")
    result.check_returncode()
    metric, = data.glob("*.csv")
    (case / "metrics.csv").write_bytes(metric.read_bytes())
    print(f"Completed {name}", flush=True)


def check(output, expected, layers, ratio):
    summary = {}
    for name in ("on_demand", "spatial"):
        case = output / name
        routes = {}
        with (case / "routes.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                assert row["source"] == "recorded"
                key = int(row["token_position"]), int(row["layer"])
                routes.setdefault(key, []).append(int(row["expert_id"]))
        assert [" ".join(map(str, sorted(v))) for _, v in sorted(routes.items())] == expected
        with (case / "metrics.csv").open(newline="") as handle:
            metrics = list(csv.DictReader(handle))
        assert len(metrics) * layers == len(expected)
        assert all(int(m["batchsize"]) == 1 and int(m["OOM"]) == 0 for m in metrics)
        demand = prefetch = dram = useful = rejected = requests = 0
        stall = 0
        for line in (case / "stdout.log").read_text().splitlines():
            fields = line.split(",")
            if fields[0] == "EXPERT_DRAM":
                size, issued, done, cycles = map(int, fields[2:6])
                assert issued == done == math.ceil(size / (64 * ratio))
                assert cycles > 0
                dram += size
                requests += done
            elif fields[0] == "OFFCHIP_LOAD":
                demand += int(fields[-2])
                stall += float(fields[-1])
            elif fields[0] == "EXPERT_PREFETCH":
                prefetch += int(fields[2])
            elif fields[0] == "EXPERT_PREFETCH_USED":
                useful += 1
            elif fields[0] == "EXPERT_PREFETCH_UNUSED":
                rejected += 1
        assert dram == demand + prefetch and requests > 0
        assert prefetch == (useful + rejected) * 44_040_192
        assert (prefetch > 0) == (name == "spatial")
        assert (case / "exit_code.txt").read_text().strip() == "0"
        summary[name] = dict(
            latency_ms=mean(float(m["latency"]) for m in metrics) / 1e6,
            exposed_stall_ms=stall / len(metrics) / 1e6,
            traffic_gb_per_token=dram / len(metrics) / 1e9,
            completed_dram_reads=requests, useful_prefetches=useful,
            unused_prefetches=rejected,
        )
    baseline, predicted = summary["on_demand"], summary["spatial"]
    assert math.isclose(
        baseline["latency_ms"] - predicted["latency_ms"],
        baseline["exposed_stall_ms"] - predicted["exposed_stall_ms"],
        abs_tol=0.02,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sample-ratio", type=int, default=2048)
    args = parser.parse_args()
    batches = glm()
    offset = 700
    if not 2 <= args.steps <= len(batches) - offset or args.sample_ratio < 1:
        parser.error("Invalid trace length or sample ratio")
    output = args.output.resolve()
    output.mkdir()
    expected = prepare(batches, offset, args.steps, output)
    layers = len(batches[0])
    for name in ("on_demand", "spatial"):
        run(args.executable.resolve(strict=True), output, name, args.steps, layers, args.sample_ratio)
    summary = check(output, expected, layers, args.sample_ratio)
    summary["trace"] = dict(offset=offset, steps=args.steps, moe_layers=layers,
                            sample_ratio=args.sample_ratio, model="DeepSeek-shaped timing proxy")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
