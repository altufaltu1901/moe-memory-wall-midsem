# MoE Memory Wall end evaluation

From the repository root:

    python -m pip install -r endsem/requirements.txt
    python endsem/run.py
    python endsem/check.py
    python endsem/plot.py

Python 3.11 or newer can run these scripts on Windows or Linux using the checked-in inputs. The root README covers the C++ simulator setup separately.

## Experiment inputs

- DeepSeekV3 uniform and skewed synthetic routing use saved one-H100 LLMSimulator runs. The replay verifies every decode step's hits, misses, evictions, bytes, and load duration against the original log. Compute time is the original latency minus logged loading.
- The public GLM trace supplies one coding-task trajectory. Predictor metrics score the last 300 tokens after a 700-token warm-up, with causal history updates throughout. The GLM system model uses proxy timings and expert sizes, with shared experts and core energy omitted.
- Policies include on-demand LRU, popularity, previous-token reuse, adjacent-layer association, fixed hybrid, adaptive hybrid, and a next-layer oracle. The adaptive policy masks resident experts, thresholds candidate scores, and caps staging by the transfer window. The oracle knows next-layer routes and bounds savings for this one-layer lookahead and overlap model.

Prefetches share one serialized link and reserve space within the finite HBM cache. Staging can use only the estimated compute window between demand stages. Demand resumes correct partial transfers, discards wrong predictions, and executes the observed expert routes. All staged bytes count toward traffic. Each transfer pays its startup cost once. The replay divides measured non-transfer duration evenly across layers and sweeps the usable portion. This is a trace-driven extension to the saved LLMSimulator timing model.

## Outputs

- artifacts/summary.csv contains experiment-level metrics, including recall, precision, average and p95 step time, throughput, exposed stall, transfer service, traffic, energy, and energy-delay product. Compare scenarios with the same workload and hardware.
- artifacts/main_steps.json contains per-step timing, traffic, and energy counts.
- artifacts/prediction_overlap.png shows recall and precision. Recall includes the first layer, for which no early prediction is made. Inputs are deduplicated across the batch.
- artifacts/system_tradeoffs.png shows normalized latency, stall, traffic, and estimated energy. artifacts/budget_tradeoff.png compares the fixed budget sensitivity.

Step latency is inter-token latency for each sequence in the lockstep batch. Token throughput counts all completed tokens across the batch. Expert link service includes demand and speculative work, while stall counts only service on the demand path.

Energy adds saved LLMSimulator dynamic energy to off-chip transfer bits times a cost and exposed stall seconds times a waiting power. The defaults of 20 pJ/bit and 100 W are scenario assumptions. The grid varies both parameters. Predictor overhead, HBM contention, and DRAM command timing are outside this model. Each DeepSeekV3 run contains five iterations, making p95 descriptive of a short trace.
Design references: https://arxiv.org/abs/2606.15453 and https://arxiv.org/abs/2608.11688
