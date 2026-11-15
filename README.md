# MoE Memory Wall

A one-GPU simulation of expert offloading for mixture-of-experts models. Expert weights live in off-chip DDR4 and are staged into a finite HBM cache. The project extends [LLMSimulator](https://github.com/scale-snu/LLMSimulator) with a training-free predictor, a shared-expert prefetch schedule, and Ramulator 2.0 transfer timing. You can run the experiments without a physical GPU or model weights.

## Endsem results

Both packed-weight GLM-5.2 windows use exactly the recorded expert routes, the published model dimensions, and per-layer packed sizes read from the public GGUF headers. The practical policy stages up to 1 MiB of a predicted routed expert and overlaps the known shared-expert load with routed compute.

- **Practical policy:** 180.74 → 178.58 ms/token and 190.46 → 188.10 ms/token, a **1.20–1.24% latency reduction** with **0.55–0.64% extra traffic**.
- **Hindsight oracle:** 171.19 and 180.91 ms/token, a **5.02–5.29% latency reduction** with unchanged traffic. The oracle knows the next layer’s actual routes and uses the same single staging slot.
- **Analytical limits:** 5.42–5.71% at the existing scheduling points. A looser 12.38–13.05% ceiling assumes perfect overlap across the decode step at fixed traffic.

These results cover two 12-token windows from one public trace. GPU timing is analytical, and Ramulator cycles are sampled and scaled to weight size. The timings exclude quantized GPU kernels and unpacking overhead. [Verified results and oracle calculations](endsem/artifacts/ideal_ceiling.json) · [Trace and weight provenance](endsem/GLM_TRACE_PROVENANCE.md)

## Verify the saved runs

With Python 3.11 or newer, clone the repository and verify the checked-in evidence. This step uses only the Python standard library.

```bash
git clone https://github.com/altufaltu1901/moe-memory-wall-midsem.git
cd moe-memory-wall-midsem
python endsem/ideal_ceiling.py
python endsem/check_coupled.py
```

The first command recomputes the GLM results and ceilings from the saved logs. The second checks the matched DeepSeekV3 runs. Use `python3` if that is the name of your Python executable.

## Build and run

On Ubuntu or WSL 2, install the build tools and initialize the pinned Ramulator submodule from the repository root.

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-venv
git submodule update --init --recursive
bash midsem/reproduce.sh --build-only
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r endsem/requirements.txt
mkdir -p results
python endsem/run_glm_coupled.py build/run results/glm-packed --geometry glm5_2_packed --chunk-mib 1 --shared --sample-ratio 1024
```

Use a fresh output directory for each run. For the second trace window add `--offset 760`. For the hindsight comparison replace `--chunk-mib 1` with `--oracle`. The [endsem guide](endsem/README.md) has the full experiment commands, the native Windows build procedure using the bundled source archive, and the plotting commands.

## Midsem evidence

- The supplied [Duplex configuration](config.yaml) used one node with eight B100 GPUs, GPU execution, and analytical memory timing. The [reproduction evidence](midsem/artifacts/baseline/) records exit code 0, pinned versions, and four CSV rows from the five requested iterations.
- In the modified one-H100 DeepSeekV3 run, off-chip expert loading averaged **2,478.93 ms** and expert execution averaged **23.68 ms** per decode iteration. The [plots](midsem/artifacts/analysis/) cover loading versus compute, bandwidth, and routing locality.
- The [midsem guide](midsem/README.md) documents the original setup and diagnostic runs. To rerun the modified midsem cases, use `bash midsem/reproduce.sh`.

[Course requirements audit](endsem/VERIFICATION.md) · [Research notes](endsem/RESEARCH_NOTES.md) · [Worked report example](endsem/EXAMPLE_REPORT.md) · [Upstream license](LICENSE)
