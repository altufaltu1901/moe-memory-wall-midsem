# Midsem files

See the [main README](../README.md) for the full setup, run, and plot commands.

## Reproducing the supplied Duplex configuration

The original supplied [`config.yaml`](../config.yaml) uses **one node and eight B100 GPUs**. Its unmodified LLMSimulator run completed with [exit code 0](artifacts/baseline/exit_code.txt). The [saved CSV](artifacts/baseline/) contains four recorded metric rows even though the configuration requested five decode iterations; the upstream exporter omitted the final row. The exact source and build environment are recorded in [`versions.txt`](artifacts/baseline/versions.txt): LLMSimulator `419252761fbdb95b789778a02256d458a5537ec7` and Ramulator2 `b5262869b24790e5ec62d9d9974fca9bbb4f54e0`.

After building, the original configuration can be run from `build/` with `./run ../config.yaml`. The modified experiment instead uses **one H100 GPU**, batch size four, and the added off-chip expert path. Its latency is not directly comparable to the original eight-B100 run because the hardware, batch size, and memory path differ.

## Diagnostic checks cited in the report

The [skewed-routing cache check](artifacts/cache_check/) uses [`config_cache_check.yaml`](../config_cache_check.yaml) and records 2,353 cache hits and 823 misses. The [offloading-disabled timing check](artifacts/compute_only/) uses [`config_compute_only.yaml`](../config_compute_only.yaml) and averages 27.47 ms per iteration. The latter bypasses the capacity check and is a timing-decomposition check, not a deployable one-H100 configuration.

- [Timing plots and numbers](artifacts/analysis/)
- [Routing-correlation plots](artifacts/real_trace/analysis/)
- [Adjacent-layer association plot](artifacts/real_trace/analysis/cross_layer_recall.png) and [numbers](artifacts/real_trace/analysis/cross_layer_summary.json)
- [Reproduction script](reproduce.sh)
