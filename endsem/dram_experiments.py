"""Measure DDR4 read completion and replay both DeepSeekV3 workloads."""

import argparse
import csv
import math
import re
from dataclasses import replace
from pathlib import Path
import subprocess

from replay import Platform, replay
from run import summarize, synthetic


def measure(executable, size, channels, mapper, offset, output):
    process = subprocess.run(
        [str(executable), str(size), str(channels), mapper, str(offset)],
        capture_output=True, text=True, check=True, timeout=300,
    )
    name = f"{size}_{channels}_{mapper}_{offset}"
    (output / f"{name}.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    return verify_log(process.stdout, size, channels, mapper, offset)


def verify_log(output_text, size, channels, mapper, offset):
    line, = (line for line in output_text.splitlines() if line.startswith("DRAM_PROBE,"))
    _, logged_size, logged_channels, logged_mapper, logged_offset, issued, done, cycles, tck, first = line.split(",")
    if (int(logged_size), int(logged_channels), logged_mapper, int(logged_offset)) != (
        size, channels, mapper, offset
    ) or int(issued) != int(done) or int(done) * 64 != size:
        raise ValueError(f"Invalid request accounting: {line}")
    cycles, first, tck = int(cycles), int(first), float(tck)
    for stat, expected in (("total_num_read_requests", int(done)), ("memory_system_cycles", cycles)):
        matches = re.findall(rf"^  {stat}: (\d+)$", output_text, re.MULTILINE)
        if matches != [str(expected)]:
            raise ValueError(f"Ramulator statistic {stat} disagrees with callback accounting")
    if not (0 < first <= cycles and math.isfinite(tck) and tck > 0):
        raise ValueError(f"Invalid completion timing: {line}")
    duration = cycles * tck
    return dict(size=size, channels=channels, mapper=mapper, offset=offset,
                completed_reads=int(done), cycles=cycles, tck_ns=tck,
                first_read_ns=first * tck, duration_ns=duration, gbps=size / duration)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--from-logs", action="store_true", help="Recheck saved raw logs and rebuild CSVs")
    args = parser.parse_args()
    output = args.output
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    platform = Platform()
    scenarios = [
        (size, channels, mapper, 0)
        for size in (64, 2**20, platform.weight_bytes)
        for channels in (1, 2, 4)
        for mapper in ("RoBaRaCoCh", "ChRaBaRoCo")
    ]
    scenarios.extend((platform.weight_bytes, 4, "RoBaRaCoCh", offset)
                     for offset in (4096, 64 * 2**20))
    if args.from_logs:
        measured = []
        for size, channels, mapper, offset in scenarios:
            log = logs / f"{size}_{channels}_{mapper}_{offset}.log"
            measured.append(verify_log(log.read_text(encoding="utf-8"), size, channels, mapper, offset))
    else:
        executable = args.probe.resolve(strict=True)
        measured = [measure(executable, *scenario, logs) for scenario in scenarios]
    write_csv(output / "measurements.csv", measured)
    inputs = {tag: synthetic(tag, platform) for tag in ("offchip", "cache_check")}
    rows = []
    for measurement in measured:
        if measurement["size"] != platform.weight_bytes:
            continue
        # Separate the independent host/link setup from completion on the DDR4 side.
        # An ideal pipeline is limited by its slower stage. Also report serial service.
        link_ns = platform.weight_bytes / platform.bandwidth
        dram_ns = measurement["duration_ns"]
        for mode, payload_ns in (("pipelined", max(link_ns, dram_ns)),
                                 ("serial", link_ns + dram_ns)):
            hardware = replace(platform, bandwidth=platform.weight_bytes / payload_ns)
            for workload, data in inputs.items():
                for policy in ("on_demand", "spatial", "adaptive", "oracle"):
                    result = summarize(replay(*data, hardware, policy=policy))
                    rows.append(dict(workload=workload, policy=policy, topology=mode,
                                     channels=measurement["channels"], mapper=measurement["mapper"],
                                     offset=measurement["offset"], ddr4_ns=dram_ns,
                                     effective_gbps=hardware.bandwidth, **result))
    key_fields = ("workload", "topology", "channels", "mapper", "offset")
    baseline = {tuple(row[k] for k in key_fields): row for row in rows
                if row["policy"] == "on_demand"}
    for row in rows:
        base = baseline[tuple(row[k] for k in key_fields)]
        saved = base["mean_step_ms"] - row["mean_step_ms"]
        stall_saved = base["exposed_stall_ms_per_step"] - row["exposed_stall_ms_per_step"]
        if not math.isclose(saved, stall_saved, abs_tol=1e-8):
            raise ValueError("Replay latency and stall differences disagree")
        if saved > row["overlap_window_ms_per_step"] + 1e-8:
            raise ValueError("Replay hides more time than the compute window")
    write_csv(output / "replay.csv", rows)
    print(f"Verified {len(measured)} DRAM completion probes and ran {len(rows)} replays")


if __name__ == "__main__":
    main()
