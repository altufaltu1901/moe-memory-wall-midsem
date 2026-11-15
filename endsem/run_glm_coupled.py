"""Run recorded GLM routes through the coupled decode simulator."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
import shutil
import subprocess

from replay import History
from run import glm


def prepare(batches, offset, steps, output, oracle=False):
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
                if oracle and next_layer < len(batch):
                    predictions = sorted(next(iter(batch[next_layer][1].values())))
                serialized_plan.append(" ".join(map(str, predictions)))
    for name, rows in (("input_routes.txt", serialized_routes),
                       ("prefetch_plan.txt", serialized_plan)):
        (output / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
    return serialized_routes


def packed_sizes(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["layer"]): (int(row["routed_bytes"]), int(row["shared_bytes"]))
                for row in csv.DictReader(handle)}


def run(executable, output, name, steps, layers, ratio, chunk_bytes, shared, geometry,
        oracle, input_len, exact_context):
    root = Path(__file__).resolve().parents[1]
    case = output / name
    case.mkdir()
    run_id = hashlib.sha256(output.name.encode()).hexdigest()[:10] + "_" + name[0]
    data = executable.parent.parent / "data" / run_id
    data.mkdir(parents=True, exist_ok=True)
    if list(data.glob("*.csv")):
        raise ValueError(f"Old metrics in {data}")
    text = (root / "config_offchip.yaml").read_text()
    settings = f"\n    ramulator_channels: 4\n    ramulator_sample_ratio: {ratio}"
    settings += f"\n    prefetch: {str(name == 'spatial').lower()}"
    if shared and name == "spatial":
        settings += "\n    prefetch_shared: true"
    if chunk_bytes:
        settings += f"\n    prefetch_chunk_bytes: {chunk_bytes}"
    if (chunk_bytes or oracle) and name == "on_demand":
        # Prefetching reserves one full expert slot.
        reserve = (42 if geometry == "proxy" else 72) * 2**20
        if geometry == "glm5_2_packed":
            reserve = max(max(sizes) for sizes in packed_sizes(output / "packed_profile.csv").values())
        text = text.replace("expert_cache_GiB: 32 ", f"expert_cache_GiB: {32 - reserve / 2**30} ")
    text = text.replace("  offchip_experts:", "  offchip_experts:" + settings)
    if geometry != "proxy":
        assert layers == 75
        text = text.replace("model_name: deepseekV3", "model_name: glm5_2")
        text = text.replace("precision_byte: 1", "precision_byte: 2")
        if geometry == "glm5_2_packed":
            text = text.replace("  model_name: glm5_2",
                                "  model_name: glm5_2\n  packed_expert_profile: ../packed_profile.csv")
    else:
        text = text.replace("  model_name: deepseekV3",
                            f"  model_name: deepseekV3\n  recorded_moe_layers: {layers}")
    output_len = min(steps + 32, 2048 - input_len) if geometry != "proxy" else steps + 32
    for key, value in (("max_batch_size", 1), ("iter", steps), ("input_len", input_len),
                       ("output_len", output_len),
                       ("output_directory", f"../data/{run_id}")):
        text = re.sub(rf"(?m)^  {key}:.*$", f"  {key}: {value}", text)
    config = case / "config.yaml"
    config.write_text(text)
    env = dict(os.environ, MOE_INPUT_ROUTES=str(output / "input_routes.txt"),
               MOE_TRACE_PATH=str(case / "routes.csv"))
    env.pop("MOE_PREFETCH_PLAN", None)
    env.pop("MOE_EXACT_CONTEXT", None)
    if exact_context:
        env["MOE_EXACT_CONTEXT"] = "1"
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


def check(output, expected, layers, ratio, chunk_bytes=0, shared=False, expert_bytes=44_040_192):
    summary = {}
    packed = packed_sizes(output / "packed_profile.csv") if (output / "packed_profile.csv").is_file() else {}
    def weight_bytes(key):
        if not packed:
            return expert_bytes
        layer = int(re.search(r"_decoder_(\d+)", key)[1])
        return packed[layer][":shared:" in key]

    for name in ("on_demand", "spatial"):
        case = output / name
        routes = {}
        with (case / "routes.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                assert row["source"] == "recorded"
                key = int(row["token_position"]), int(row["layer"])
                routes.setdefault(key, []).append(int(row["expert_id"]))
        assert [" ".join(map(str, sorted(v))) for _, v in sorted(routes.items())] == expected
        assert len({position for position, _ in routes}) * layers == len(routes)
        with (case / "metrics.csv").open(newline="") as handle:
            metrics = list(csv.DictReader(handle))
        assert len(metrics) * layers == len(expected)
        assert all(int(m["batchsize"]) == 1 and int(m["OOM"]) == 0 for m in metrics)
        demand = prefetch = dram = useful = rejected = requests = 0
        stall = 0
        shared_issued = shared_used = issued_prefetches = 0
        for line in (case / "stdout.log").read_text().splitlines():
            fields = line.split(",")
            if fields[0] == "EXPERT_DRAM":
                size, issued, done, cycles = map(int, fields[2:6])
                effective_ratio = min(ratio, max(1, size // (64 * 128)))
                assert issued == done == math.ceil(size / (64 * effective_ratio))
                assert cycles > 0
                full = weight_bytes(fields[1])
                assert size in {full, chunk_bytes, full - chunk_bytes}
                dram += size
                requests += done
            elif fields[0] == "OFFCHIP_LOAD":
                demand += int(fields[-2])
                stall += float(fields[-1])
            elif fields[0] == "EXPERT_PREFETCH":
                speculative = int(fields[2])
                full = weight_bytes(fields[1])
                assert speculative == (full if ":shared:" in fields[1] or not chunk_bytes else min(chunk_bytes, full))
                prefetch += speculative
                issued_prefetches += 1
                shared_issued += ":shared:" in fields[1]
            elif fields[0] == "EXPERT_PREFETCH_USED":
                useful += 1
                shared_used += ":shared:" in fields[1]
            elif fields[0] == "EXPERT_PREFETCH_UNUSED":
                rejected += 1
        assert dram == demand + prefetch and requests > 0
        assert shared_issued == shared_used
        assert shared or shared_issued == 0
        assert issued_prefetches == useful + rejected
        assert (prefetch > 0) == (name == "spatial")
        assert (case / "exit_code.txt").read_text().strip() == "0"
        summary[name] = dict(
            transferred_bytes=dram,
            latency_ms=mean(float(m["latency"]) for m in metrics) / 1e6,
            exposed_stall_ms=stall / len(metrics) / 1e6,
            traffic_gb_per_token=dram / len(metrics) / 1e9,
            completed_dram_reads=requests, useful_prefetches=useful,
            unused_prefetches=rejected,
            shared_prefetches=shared_used,
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
    parser.add_argument("--chunk-mib", type=int, default=0)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--geometry", choices=("proxy", "glm5_2", "glm5_2_packed"), default="glm5_2")
    parser.add_argument("--oracle", action="store_true",
                        help="Use next-layer ground truth as a hindsight scheduling reference")
    parser.add_argument("--offset", type=int, default=700)
    parser.add_argument("--input-len", type=int,
                        help="Override modeled prompt length (default: match packed capture context)")
    args = parser.parse_args()
    batches = glm()
    offset = args.offset
    chunk_bytes = args.chunk_mib * 2**20
    exact_context = args.input_len is None and args.geometry == "glm5_2_packed"
    input_len = (1255 + offset if exact_context else args.input_len or 1024)
    if (not 0 <= offset < len(batches) or not 2 <= args.steps <= len(batches) - offset
            or input_len < 1 or (args.geometry != "proxy" and input_len + args.steps + 5 > 2048)
            or args.sample_ratio < 1 or not 0 <= args.chunk_mib < (42 if args.geometry == "proxy" else 72)
            or chunk_bytes % (64 * args.sample_ratio) or (args.shared and not (chunk_bytes or args.oracle))):
        parser.error("Invalid trace range, chunk size or sample ratio")
    output = args.output.resolve()
    output.mkdir()
    if args.geometry == "glm5_2_packed":
        profile = Path(__file__).with_name("glm52_gguf_expert_bytes.csv")
        shutil.copyfile(profile, output / "packed_profile.csv")
    expected = prepare(batches, offset, args.steps, output, args.oracle)
    layers = len(batches[0])
    for name in ("on_demand", "spatial"):
        run(args.executable.resolve(strict=True), output, name, args.steps, layers,
            args.sample_ratio, chunk_bytes, args.shared, args.geometry, args.oracle, input_len, exact_context)
    expert_bytes = (42 if args.geometry == "proxy" else 72) * 2**20
    summary = check(output, expected, layers, args.sample_ratio, chunk_bytes, args.shared,
                    expert_bytes)
    if exact_context:
        for case in ("on_demand", "spatial"):
            with (output / case / "routes.csv").open(newline="") as handle:
                positions = {int(row["token_position"]) for row in csv.DictReader(handle)}
            assert positions == set(range(input_len, input_len + args.steps))
    if args.oracle:
        assert summary["spatial"]["unused_prefetches"] == 0
    summary["trace"] = dict(offset=offset, steps=args.steps, moe_layers=layers,
                            input_len=input_len, exact_context=exact_context,
                            sample_ratio=args.sample_ratio, chunk_bytes=chunk_bytes, shared_prefetch=args.shared, oracle=args.oracle, logical_expert_bytes=expert_bytes,
                            model={
                                "glm5_2": "GLM-5.2 publisher geometry, BF16",
                                "glm5_2_packed": "GLM-5.2 publisher geometry, GGUF packed expert bytes",
                                "proxy": "DeepSeek-shaped timing proxy",
                            }[args.geometry])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
