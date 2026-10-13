# MoE Memory Wall: midsem simulator experiments

This branch extends [LLMSimulator](https://github.com/scale-snu/LLMSimulator) at upstream commit `419252761fbdb95b789778a02256d458a5537ec7`. It keeps the original eight-device example and adds a single-GPU DeepSeekV3 configuration with routed expert weights in off-chip DRAM and a 32 GiB HBM LRU expert cache. The transfer model charges `2,000 ns + bytes / bandwidth` on a cache miss before expert execution. No physical GPU or model weights were used.

The original supplied example ran successfully. The modified single-H100 run disables Logic-PIM and expert sharding, passes its 80 GiB HBM capacity check, and records all five decode iterations. An upstream final-CSV-row omission is corrected on this branch. The original eight-B100 example and single-H100 experiment use different hardware and batching, so their latencies are not direct performance comparators.

At a modeled **32 GB/s** off-chip bandwidth and independent synthetic top-8/256 routing, mean expert loading is **2,400.37 ms**, expert execution excluding loading is **23.68 ms**, and total decode-iteration latency is **2,427.84 ms**. The same routes at 16 and 64 GB/s give **4,824.73 ms** and **1,229.39 ms** respectively. The synthetic workload has zero expert-cache hits. A separate deliberately skewed routing check produces 2,170 hits and 1,500 misses, verifying the reuse path. `artifacts/analysis/` contains the JSON summaries and plots; each experiment's raw CSV/logs are under `artifacts/`.

The real-routing plots use 1,000 consecutive tokens from one task in [Predicting-Experts-for-MOE-Suite](https://huggingface.co/datasets/mistrjirka/Predicting-Experts-for-MOE-Suite), a **different** low-bit GLM-5.2 model. Adjacent tokens share a mean 1.374 of eight experts at the same layer, compared with 0.742 after randomizing time order while preserving per-layer expert frequencies. These measurements cannot be attributed to the simulated DeepSeekV3 workload or generalized beyond this task. The dataset's [MIT license and model/task details](https://huggingface.co/datasets/mistrjirka/Predicting-Experts-for-MOE-Suite) are maintained by its publisher; `artifacts/real_trace/provenance.json` records the revision and sampled split. Expert IDs are local to each layer.

## Reproduce

On Ubuntu/WSL 2, install `build-essential cmake clang git python3 python3-venv`. Clone this repository with `--recurse-submodules`, then from the repository root run:

```bash
bash midsem/reproduce.sh
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
python midsem/analyze_midsem.py
python midsem/analyze_real_trace.py
```

`reproduce.sh` applies the upstream Ramulator patch plus the GCC 15 `<cstdint>` compatibility patch, builds the simulator, and runs the five configurations. CMake 4 needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` for the pinned dependency. The original baseline's raw run and build versions are preserved in `artifacts/baseline/`. `midsem/patches/midsem-offchip.patch` is the source/config diff against the pinned upstream commit, offered for review; it is already applied on this branch. `fetch_public_trace.py` can refresh the bounded dataset sample, but the checked-in sample is enough to regenerate its plots.

These are simulator results from an analytical fixed-bandwidth link. They do not model transfer contention, energy, or transfer/compute overlap. The compute-only run is an intentionally infeasible reference: it bypasses the 80 GiB HBM limit for a 614.46 GiB logical model state. It isolates the added load delay and is not a deployable baseline. The course requires students to write the report and slides themselves; this repository provides code, data, and reproducible measurements.
