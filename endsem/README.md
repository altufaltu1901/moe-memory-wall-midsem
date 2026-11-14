# MoE Memory Wall end evaluation

From the repository root:

    python -m pip install -r endsem/requirements.txt
    python endsem/run.py
    python endsem/check.py
    python endsem/verify_more.py
    python endsem/plot.py

Python 3.11 or newer can run these scripts on Windows or Linux using the checked-in inputs. The root README covers the C++ simulator setup separately.

## Experiment inputs

- DeepSeekV3 uniform and skewed synthetic routing use saved one-H100 LLMSimulator runs. The replay verifies every decode step's hits, misses, evictions, bytes, and load duration against the original log. Compute time is the original latency minus logged loading.
- The public GLM trace supplies one coding-task trajectory. Predictor metrics score the last 300 tokens after a 700-token warm-up, with causal history updates throughout. The GLM system model uses proxy timings and expert sizes, with shared experts and core energy omitted.
- Policies include on-demand LRU, popularity, previous-token reuse, adjacent-layer association, fixed hybrid, adaptive hybrid, and a next-layer oracle. The adaptive policy masks resident experts, thresholds candidate scores, and caps staging by the transfer window. The oracle knows next-layer routes and skips resident weights. It is a reference policy, since a different cache insertion or eviction order can change performance. The reported overlap window bounds directly hidden link time per step.

Prefetches share one serialized link and reserve space within the finite HBM cache. Staging can use only the estimated compute window between demand stages. Demand resumes correct partial transfers, discards wrong predictions, and executes the observed expert routes. All staged bytes count toward traffic. Each transfer pays its startup cost once. The replay divides measured non-transfer duration evenly across layers and sweeps the usable portion. This is a trace-driven extension to the saved LLMSimulator timing model.

## Outputs

- artifacts/summary.csv contains experiment-level metrics, including recall, precision, average and p95 step time, throughput, exposed stall, transfer service, traffic, energy, and energy-delay product. Compare scenarios with the same workload and hardware.
- artifacts/main_steps.json contains per-step timing, traffic, and energy counts.
- artifacts/prediction_overlap.png shows recall and precision. Recall includes the first layer, for which no early prediction is made. Inputs are deduplicated across the batch.
- artifacts/system_tradeoffs.png shows normalized latency, stall, traffic, and estimated energy. artifacts/budget_tradeoff.png compares the fixed budget sensitivity.

Step latency is inter-token latency for each sequence in the lockstep batch. Token throughput counts all completed tokens across the batch. Expert link service includes demand and speculative work, while stall counts only service on the demand path.

Energy adds the saved LLMSimulator dynamic energy to off-chip transfer bits times a cost and exposed stall seconds times a waiting power. The defaults of 20 pJ/bit and 100 W are scenario assumptions. The grid varies both parameters. Predictor overhead, HBM contention, and DRAM command timing are outside this model. Each DeepSeekV3 run contains five iterations, making p95 descriptive of a short trace.
Design references: https://arxiv.org/abs/2606.15453 and https://arxiv.org/abs/2608.11688

## Ramulator probe and calibrated replay

Build from a package with the pinned Ramulator and bundled C++ dependencies.
Use Git, CMake, Ninja, and a C++20 compiler. On Windows the probe was developed
with LLVM-MinGW, with its bin directory on PATH.

    python endsem/build_dram_probe.py PATH_TO_SOURCE_ZIP NEW_BUILD_DIRECTORY
    python endsem/dram_experiments.py NEW_BUILD_DIRECTORY/build/dram_probe endsem/artifacts/ramulator
    python endsem/plot_dram.py

On Windows, use dram_probe.exe and pass --compiler PATH_TO_CLANG++ if needed.
The build works in a separate extracted source directory. It applies the same
PIM source patch as LLMSimulator and adapts compiler and startup behavior for the
standalone target. The experiment uses ordinary DDR4 reads and the generic
FR-FCFS controller.

To recheck the saved probe logs and regenerate the CSVs without rebuilding,
run the same experiment command with --from-logs. The probe path is ignored
in that mode.

The probe uses 64-byte transactions and stops after the final read callback.
It varies transfer size, channel count, address map and address alignment.
The replay uses the measured full-expert DDR4 transfer times and a separate
32 GB/s host link. It describes two assumed configurations: ideal pipelining
of the slower stage and serial DDR4 plus link time. Raw probe logs and derived
CSV inputs are in artifacts/ramulator. The DDR4 measurement models a contiguous
expert read; the replay applies the resulting service cost uniformly to experts.
It leaves DRAM contention across transfers, host/HBM power, and controller energy
outside the model. See VERIFICATION.md for the requirement-by-requirement audit.

## Coupled LLMSimulator runs

Build the patched full simulator with the same helper used for the probe.
The source ZIP must contain this revision of the C++ code.

    python endsem/build_dram_probe.py PATH_TO_SOURCE_ZIP NEW_BUILD_DIRECTORY --llmsimulator
    python endsem/run_coupled.py NEW_BUILD_DIRECTORY/build/run NEW_RESULTS_DIRECTORY
    python endsem/check_coupled.py NEW_RESULTS_DIRECTORY

Use run.exe on Windows. The seven saved cases, each with five decode steps,
are checked by running python endsem/check_coupled.py with no arguments.

The original HBM GPU timing remains at its configured analytical setting.
The new off-chip controller is independent of that setting. The expert-load
module calls the patched Ramulator GenericDRAM DDR4 controller and waits for
actual completion callbacks to obtain a service duration. The configured
sample ratio of 1,024 uses 672 requests per 42 MiB expert and scales the
measured batch time to the full weight. A persistent controller preserves
state across sampled transfers. It advances during transfer service only.
The effective transfer is the slower of the DRAM and host-link payload
durations, plus the configured startup latency.

Prefetch uses one reserved HBM slot. After the shared load stage it selects an
uncached expert from the next layer's previous decode step and schedules its
completion on the serialized expert link. Future routing still determines
which experts execute. The GPU clock can advance during the scheduled
transfer. Demand waits only for any remaining time, including contention
from unused prefetches. The coupled test uses this simple temporal policy.
The broader predictor ablations and GLM evaluation remain in the Python
replay. Energy adds modeled transfer cost and waiting power to logged core
energy. These are estimates rather than a calibrated DDR4 power model.

Linux and Windows can generate different synthetic routes because C++ library
distributions differ. Compare matched cases from the same executable. The
analytical control preserves the original timing model and expert inventory.

## Public GLM routes in the coupled simulator

    python endsem/run_glm_coupled.py PATH_TO_RUN_EXECUTABLE NEW_RESULTS_DIRECTORY

The matched batch-one cases use 12 consecutive tokens starting at position
700 of the public trace. Each token uses the recorded top-eight choices at
all 75 MoE layers. The route logger is checked against the original input.
The spatial policy generates its candidate list from earlier observed routes
and the current source layer. LLMSimulator issues at most one candidate that
is absent from HBM. The first token initializes next-layer relationships.

GPU dimensions, expert weight sizes and shared experts come from the
DeepSeekV3 timing proxy. Expert transfers use four Ramulator DDR4 channels
with one 64-byte request per 2,048 bursts sampled and scaled to full weight.
These are simulated results for GLM routing with proxy compute and memory.
