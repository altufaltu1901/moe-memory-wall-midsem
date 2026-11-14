"""Serialized expert-link replay driven by observed router decisions."""

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Platform:
    bandwidth: float = 32.0  # bytes/ns, numerically equal to decimal GB/s
    startup_ns: float = 2000.0
    weight_bytes: int = 44_040_192
    cache_bytes: int = 32 * 2**30
    transfer_pj_per_bit: float = 20.0
    stall_watts: float = 100.0

    def __post_init__(self):
        if self.bandwidth <= 0 or self.weight_bytes <= 0:
            raise ValueError("Bandwidth and expert size must be positive")
        if min(self.startup_ns, self.cache_bytes, self.transfer_pj_per_bit,
               self.stall_watts) < 0:
            raise ValueError("Platform costs and capacity must be nonnegative")


class History:
    """Causal popularity, adjacent-layer counts and preceding-token reuse."""

    def __init__(self, experts=256):
        self.popularity = defaultdict(lambda: np.zeros(experts, dtype=np.int64))
        self.samples = Counter()
        self.source_support = defaultdict(lambda: np.zeros(experts, dtype=np.int64))
        self.associations = defaultdict(lambda: np.zeros((experts, experts), dtype=np.int64))
        self.previous = {}
        self.experts = experts

    def predict(self, layer, tokens, context, policy, budget, adaptive_budget, resident):
        scores = np.zeros(self.experts)
        for sequence, position in tokens:
            prior = self.popularity[layer] / max(1, self.samples[layer])
            score = prior.copy()
            if policy in {"spatial", "hybrid", "adaptive"}:
                source = list(context.get((sequence, position), ()))
                if source:
                    observed = self.associations[layer][source]
                    support = self.source_support[layer][source, None]
                    score += ((observed - support * prior) / (support + 20)).mean(axis=0)
            if policy in {"temporal", "hybrid", "adaptive"}:
                previous = self.previous.get((layer, sequence))
                if previous and previous[0] + 1 == position:
                    score[list(previous[1])] += 0.08 if policy == "adaptive" else 0.45
            scores += np.clip(score, 0, 1)
        scores /= len(tokens)
        if policy == "adaptive":
            budget = min(budget, adaptive_budget)
            scores[list(resident)] = 0
            eligible = np.flatnonzero(scores >= 0.15)
        else:
            eligible = np.flatnonzero(scores > 0)
        ranked = sorted(eligible, key=lambda i: (-scores[i], i))
        return [int(i) for i in ranked[:budget]]

    def observe(self, layer, routes, context):
        for (sequence, position), selected in routes.items():
            self.popularity[layer][list(selected)] += 1
            self.samples[layer] += 1
            self.previous[layer, sequence] = (position, selected)
            for expert in context.get((sequence, position), ()):
                self.source_support[layer][expert] += 1
                self.associations[layer][expert, list(selected)] += 1


@dataclass
class Link:
    platform: Platform
    cache: OrderedDict = field(default_factory=OrderedDict)
    stats: Counter = field(default_factory=Counter)

    @property
    def transfer_ns(self):
        return self.platform.startup_ns + self.platform.weight_bytes / self.platform.bandwidth

    def insert(self, key, reserved):
        slots = self.platform.cache_bytes // self.platform.weight_bytes
        if slots <= reserved:
            return
        if len(self.cache) + reserved >= slots:
            self.cache.popitem(last=False)
            self.stats["evictions"] += 1
        self.cache[key] = None

    def demand(self, key, staged):
        p = self.platform
        if key in self.cache:
            self.cache.move_to_end(key)
            self.stats["hits"] += 1
            return
        elapsed = staged.pop(key, 0)
        transferred = max(0, elapsed - p.startup_ns) * p.bandwidth
        self.stats["stall_ns"] += self.transfer_ns - elapsed
        self.stats["demand_bytes"] += p.weight_bytes - transferred
        self.stats["misses"] += 1
        self.stats["demand_service_ns"] += self.transfer_ns - elapsed
        self.stats["useful_prefetch_bytes"] += transferred
        if elapsed >= self.transfer_ns:
            self.stats["ready_prefetches"] += 1
        elif transferred:
            self.stats["partial_prefetches_used"] += 1
        self.insert(key, len(staged))

    def prefetch(self, candidates, window_ns):
        """Stage weights during compute. A partial final transfer can resume on demand."""
        staged = {}
        p = self.platform
        slots = p.cache_bytes // p.weight_bytes
        # Reserve staging space within the same HBM budget. Demand can reclaim it.
        for key in candidates:
            if window_ns <= p.startup_ns or len(staged) >= slots:
                break
            if key in self.cache:
                continue
            if len(self.cache) + len(staged) >= slots:
                self.cache.popitem(last=False)
                self.stats["evictions"] += 1
            elapsed = min(window_ns, self.transfer_ns)
            staged[key] = elapsed
            self.stats["prefetch_service_ns"] += elapsed
            self.stats["prefetch_bytes"] += (elapsed - p.startup_ns) * p.bandwidth
            window_ns -= elapsed
        return staged


def replay(batches, compute, core_energy, platform, policy="on_demand", budget=8,
           window_fraction=1.0, shared=True, score_from=0):
    """Each batch is an ordered list of (layer, per-token route sets).

    Transfers for layer l+1 can use only the compute interval after layer l's
    routed and shared demand. No transfer uses future routing in deployable policies.
    """
    history = History()
    link = Link(platform)
    rows = []
    for step, layers in enumerate(batches):
        snapshot = link.stats.copy()
        context, staged, ranked = {}, {}, []
        # The input supplies a whole-step compute duration. Each layer receives
        # an equal share, or an explicit list of compute intervals.
        intervals = compute[step]
        if isinstance(intervals, (int, float)):
            intervals = [intervals / len(layers)] * len(layers)
        if len(intervals) != len(layers):
            raise ValueError("Compute intervals must match layer count")
        for index, (layer, routes) in enumerate(layers):
            actual = sorted(set().union(*routes.values()))
            link.stats["actual"] += len(actual)
            link.stats["route_events"] += 1
            if index + 1 < len(layers):
                link.stats["overlap_window_ns"] += intervals[index] * window_fraction
            link.stats["predicted"] += len(ranked)
            link.stats["overlap"] += len(set(actual).intersection(ranked))
            # Reclaim wrong speculative entries before servicing demands.
            staged = {key: elapsed for key, elapsed in staged.items() if key[1] in actual}
            for expert in actual:
                link.demand((layer, expert), staged)
            if shared:
                link.demand((layer, -1), {})
            if policy not in {"on_demand", "oracle"}:
                history.observe(layer, routes, context)
            context = routes
            ranked, staged = [], {}
            if policy == "on_demand" or index + 1 == len(layers):
                continue
            target_layer, target_routes = layers[index + 1]
            window = intervals[index] * window_fraction
            if policy == "oracle":
                ranked = [expert for expert in sorted(set().union(*target_routes.values()))
                          if (target_layer, expert) not in link.cache][:budget]
            else:
                # Sequence identifiers and positions come from current routes.
                adaptive_budget = int(np.ceil(window / link.transfer_ns))
                resident = {expert for owner, expert in link.cache if owner == target_layer and expert >= 0} if policy == "adaptive" else set()
                ranked = history.predict(target_layer, routes, context, policy, budget,
                                         adaptive_budget, resident)
            staged = link.prefetch(((target_layer, e) for e in ranked), window)

        delta = {key: value - snapshot[key] for key, value in link.stats.items()}
        stall = delta.get("stall_ns", 0)
        demand = delta.get("demand_bytes", 0)
        prefetch = delta.get("prefetch_bytes", 0)
        transfer_energy = (demand + prefetch) * 8 * platform.transfer_pj_per_bit * 1e-12
        stall_energy = stall * 1e-9 * platform.stall_watts
        tokens = len(layers[0][1])
        if step >= score_from:
            rows.append(dict(delta, step=step, tokens=tokens, compute_ns=sum(intervals),
                             latency_ns=sum(intervals) + stall, traffic_bytes=demand + prefetch,
                             transfer_energy_j=transfer_energy, stall_energy_j=stall_energy,
                             core_energy_j=core_energy[step],
                             energy_j=core_energy[step] + transfer_energy + stall_energy))
    return rows
