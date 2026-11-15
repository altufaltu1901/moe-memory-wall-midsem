"""Compute ideal overlap ceilings from on-demand LLMSimulator logs."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
NS_PER_MS = 1e6


@dataclass(frozen=True)
class Load:
    stage: str
    misses: int
    size_bytes: int
    service_ns: float


def read_baseline(root):
    if (root / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError(f"{root}: baseline failed")
    with (root / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = [(float(row["latency"]), int(row["batchsize"])) for row in csv.DictReader(handle)]
    if not metrics or any(not math.isfinite(latency) or latency <= 0 or tokens <= 0
                          for latency, tokens in metrics):
        raise ValueError(f"{root}: expected completed decode steps")

    loads, stamps = defaultdict(list), defaultdict(list)
    with (root / "stdout.log").open(encoding="utf-8") as log:
        for line in log:
            fields = line.strip().split(",")
            if fields[0] == "OFFCHIP_LOAD":
                _, step, stage, _, misses, _, size, duration = fields
                load = Load(stage, int(misses), int(size), float(duration))
                if not math.isfinite(load.service_ns) or min(
                    load.misses, load.size_bytes, load.service_ns
                ) < 0 or bool(load.misses) != bool(load.size_bytes):
                    raise ValueError(f"{root}: inconsistent load counters for {stage}")
                loads[int(step)].append(load)
            elif fields[0] == "EXPERT_STAGE":
                _, step, stage, start, finish = fields
                start, finish = float(start), float(finish)
                if not math.isfinite(start) or not math.isfinite(finish) or not 0 <= start <= finish:
                    raise ValueError(f"{root}: invalid timing interval for {stage}")
                stamps[int(step)].append((stage, start, finish))
            elif fields[0] == "EXPERT_PREFETCH":
                raise ValueError(f"{root}: ceiling requires an on-demand run")

    if set(loads) != set(range(len(metrics))) or (stamps and set(stamps) != set(loads)):
        raise ValueError(f"{root}: load events do not span the same decode steps as metrics")
    return metrics, loads, stamps


def local_overlap(loads, stamps):
    """Sum disjoint compute windows, each eligible for just the following load."""
    if [load.stage for load in loads] != [stage for stage, _, _ in stamps]:
        raise ValueError("Timing stages do not match load-stage ordering")
    by_kind = {"routed_to_shared": 0.0, "shared_to_routed": 0.0}
    eligible_windows = 0
    for prev, nxt in zip(stamps, stamps[1:]):
        prev_name, _, prev_end = prev
        next_name, next_start, next_end = nxt
        gap = next_start - prev_end
        if gap < -1e-3:
            raise ValueError("Unexpected overlap or reordered on-demand stages")
        if prev_name.endswith(":routed") and next_name.endswith(":shared"):
            if prev_name.rsplit(":", 1)[0] != next_name.rsplit(":", 1)[0]:
                raise ValueError("Expected shared load from the same MoE layer")
            kind = "routed_to_shared"
        elif prev_name.endswith(":shared") and next_name.endswith(":routed"):
            kind = "shared_to_routed"
        else:
            # The implementation has no scheduling hook for other phase pairs.
            continue
        if gap > 0 and next_end > next_start:
            by_kind[kind] += min(gap, next_end - next_start)
            eligible_windows += 1
    return by_kind, eligible_windows


def analyze(root, case):
    metrics, loads, stamps = read_baseline(root)
    phase_totals = defaultdict(float)
    eligible_windows = 0
    total = stall = tokens = transfer_bytes = perfect_overlap = 0
    for step, (latency, token_count) in enumerate(metrics):
        listed_service = sum(load.service_ns for load in loads[step])
        service = listed_service
        if stamps:
            exact_ns = [finish - start for _, start, finish in stamps[step]]
            if len(exact_ns) != len(loads[step]) or any(
                not math.isclose(precise, load.service_ns, rel_tol=6e-6, abs_tol=1e-3)
                for load, precise in zip(loads[step], exact_ns)
            ):
                raise ValueError(f"{root}: precise and rounded service durations disagree")
            service = sum(exact_ns)
            stage_span = stamps[step][-1][2] - stamps[step][0][1]
            if stage_span > latency + 1e-3:
                raise ValueError(f"{root}: timed stage span exceeds step latency")
            overlap, windows = local_overlap(loads[step], stamps[step])
            for kind, amount in overlap.items():
                phase_totals[kind] += amount
            eligible_windows += windows
        if service > latency:
            raise ValueError(f"{root}: serialized service exceeds step latency")
        total += latency
        stall += service
        tokens += token_count
        transfer_bytes += sum(load.size_bytes for load in loads[step])
        perfect_overlap += min(service, latency - service)

    if stamps and sum(phase_totals.values()) > perfect_overlap + 1e-3:
        raise ValueError(f"{root}: phase overlap exceeds the global overlap ceiling")

    def per_token(ns):
        return ns / tokens / NS_PER_MS

    def ceiling(savings):
        return {
            "saved_ms_per_token": per_token(savings),
            "latency_floor_ms_per_token": per_token(total - savings),
            "max_latency_reduction_pct": 100 * savings / total,
            "max_speedup_x": total / (total - savings) if savings < total else None,
        }

    return {
        "case": case,
        "decode_steps": len(metrics),
        "tokens": tokens,
        "on_demand_ms_per_token": per_token(total),
        "serialized_expert_service_ms_per_token": per_token(stall),
        "non_expert_transfer_ms_per_token": per_token(total - stall),
        "expert_traffic_gb_per_token": transfer_bytes / tokens / 1e9,
        "global_fixed_traffic_overlap": ceiling(perfect_overlap),
        "phase_local_fixed_traffic_overlap": (
            {
                **ceiling(sum(phase_totals.values())),
                "eligible_windows": eligible_windows,
                "routed_to_shared_ms_per_token": per_token(phase_totals["routed_to_shared"]),
                "shared_to_routed_ms_per_token": per_token(phase_totals["shared_to_routed"]),
            }
            if stamps else None
        ),
        "zero_expert_transfer_what_if": ceiling(stall),
    }


def measured_policy(path, baseline, oracle=False):
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if bool(data["trace"].get("oracle")) != oracle:
        raise ValueError(f"{path}: mislabeled practical or oracle experiment")
    baseline_metrics = data["on_demand"]
    if not math.isclose(baseline_metrics["latency_ms"], baseline["on_demand_ms_per_token"],
                        abs_tol=1e-9):
        raise ValueError(f"{path}: policy summary uses a different on-demand latency")
    if not math.isclose(baseline_metrics["traffic_gb_per_token"],
                        baseline["expert_traffic_gb_per_token"], abs_tol=1e-9):
        raise ValueError(f"{path}: policy summary uses different on-demand traffic")
    if data["trace"]["steps"] != baseline["decode_steps"]:
        raise ValueError(f"{path}: policy summary uses a different number of steps")

    spatial = path.parent / "spatial"
    if (spatial / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError(f"{path}: policy simulation failed")
    if (path / "routes.csv").read_bytes() != (spatial / "routes.csv").read_bytes():
        raise ValueError(f"{path}: policy changed routes")
    with (spatial / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    total_ns = sum(float(row["latency"]) for row in metrics)
    tokens = sum(int(row["batchsize"]) for row in metrics)
    if len(metrics) != baseline["decode_steps"] or tokens != baseline["tokens"]:
        raise ValueError(f"{path}: policy ran a different number of tokens")
    latency = total_ns / tokens / NS_PER_MS
    transfer_bytes = stalls = issued = used = unused = 0
    with (spatial / "stdout.log").open(encoding="utf-8") as log:
        for line in log:
            fields = line.strip().split(",")
            if fields[0] == "OFFCHIP_LOAD":
                transfer_bytes += int(fields[-2])
                stalls += float(fields[-1])
            elif fields[0] == "EXPERT_PREFETCH":
                transfer_bytes += int(fields[2])
                issued += 1
            elif fields[0] == "EXPERT_PREFETCH_USED":
                used += 1
            elif fields[0] == "EXPERT_PREFETCH_UNUSED":
                unused += 1
    if issued != used + unused:
        raise ValueError(f"{path}: missing prefetch outcomes")
    traffic = transfer_bytes / tokens / 1e9
    stall_ms = stalls / tokens / NS_PER_MS
    published = data["spatial"]
    for key, actual in (("latency_ms", latency), ("traffic_gb_per_token", traffic),
                        ("exposed_stall_ms", stall_ms), ("useful_prefetches", used),
                        ("unused_prefetches", unused)):
        if not math.isclose(published[key], actual, abs_tol=1e-9):
            raise ValueError(f"{path}: raw policy logs disagree with summary for {key}")
    if oracle and (unused or not math.isclose(traffic, baseline["expert_traffic_gb_per_token"],
                                              abs_tol=1e-9)):
        raise ValueError(f"{path}: hindsight oracle altered fixed-traffic assumptions")
    saved = baseline["on_demand_ms_per_token"] - latency
    return {
        "latency_ms_per_token": latency,
        "saved_ms_per_token": saved,
        "latency_reduction_pct": 100 * saved / baseline["on_demand_ms_per_token"],
        "speedup_x": baseline["on_demand_ms_per_token"] / latency,
        "exposed_stall_ms_per_token": stall_ms,
        "expert_traffic_gb_per_token": traffic,
        "used_prefetches": used,
        "unused_prefetches": unused,
        "shared_prefetches": published.get("shared_prefetches", 0),
    }


def evaluated_cases(paths):
    analysis = {
        path: analyze(path, path.parent.name if path.name == "on_demand" else "coupled_" + path.name)
        for path in paths
    }
    merged = set()
    for path, case in analysis.items():
        if path.name != "on_demand":
            continue
        if "_oracle_" in case["case"]:
            case["measured_hindsight_oracle"] = measured_policy(path, case, oracle=True)
            continue
        case["measured_practical_policy"] = measured_policy(path, case)
        family, separator, offset = case["case"].rpartition("_")
        if not separator:
            continue
        reference = path.parent.with_name(f"{family}_oracle_{offset}") / "on_demand"
        if reference not in analysis or not (reference.parent / "summary.json").is_file():
            continue
        # Independent on-demand runs must match the entire baseline route log,
        # metric CSV and packed profile before any cross-policy comparison.
        for filename in ("routes.csv", "metrics.csv"):
            if (path / filename).read_bytes() != (reference / filename).read_bytes():
                raise ValueError(f"{reference}: {filename} differs from the practical baseline")
        actual_profile = path.parent / "packed_profile.csv"
        oracle_profile = reference.parent / "packed_profile.csv"
        if (actual_profile.is_file() != oracle_profile.is_file() or
                (actual_profile.is_file() and actual_profile.read_bytes() != oracle_profile.read_bytes())):
            raise ValueError(f"{reference}: expert byte profile differs from the practical baseline")
        case["measured_hindsight_oracle"] = measured_policy(reference, case, oracle=True)
        merged.add(reference)
    return [case for path, case in analysis.items() if path not in merged]


def default_cases():
    directories = [
        ARTIFACTS / "coupled" / "ram1",
        ARTIFACTS / "coupled" / "ram4",
        ARTIFACTS / "coupled" / "skew4",
        ARTIFACTS / "glm_shared_700" / "on_demand",
        ARTIFACTS / "glm_shared_760" / "on_demand",
    ]
    directories.extend(sorted(
        case for case in ARTIFACTS.glob("glm52_*/on_demand")
        if all((case / name).is_file() for name in ("metrics.csv", "exit_code.txt", "stdout.log"))
    ))
    return directories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", type=Path, help="On-demand directories containing logs and metrics")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "ideal_ceiling.json")
    args = parser.parse_args()
    cases = evaluated_cases(args.cases or default_cases())
    report = {
        "interpretation": {
            "global_fixed_traffic_overlap": "Theoretical ceiling for the measured workload and unchanged serialized expert service. With no traffic moved between decode steps, an ideal GPU and link could at best overlap min(non-transfer time, expert service) per step. No routing barrier, cache visibility constraint, or scheduling overhead is charged.",
            "phase_local_fixed_traffic_overlap": "Tighter analytical ceiling with perfect knowledge and only existing scheduling hooks. Each gap from a routed load to its shared load or a shared load to the next routed load can hide at most min(gap, following load service). Uses exact stage timestamps when recorded. The estimate assumes unchanged service time, ignores staging capacity limits, and does not permit transfers across decode steps.",
            "measured_hindsight_oracle": "Separate simulator run that knows the next layer's true route set. The practical policy uses prior observations only. Measured policies are verified against raw route and transfer logs, with identical paired on-demand runs before combining comparisons.",
            "zero_expert_transfer_what_if": "Counterfactual with every off-chip expert transfer eliminated and unchanged non-transfer work. This requires different memory/traffic assumptions and is not a prefetch result.",
        },
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        ideal = case["global_fixed_traffic_overlap"]
        phase_local = case["phase_local_fixed_traffic_overlap"]
        message = (f'{case["case"]}: baseline {case["on_demand_ms_per_token"]:.3f} ms/token, '
                   f'ideal global <= {ideal["max_latency_reduction_pct"]:.3f}% reduction')
        if phase_local:
            message += f', phase-local <= {phase_local["max_latency_reduction_pct"]:.3f}%'
        print(message)


if __name__ == "__main__":
    main()

