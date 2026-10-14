# MoE Memory Wall — midsem simulator

This is the [LLMSimulator](https://github.com/scale-snu/LLMSimulator) codebase extended for the Computer Architecture MoE Memory Wall project. It runs a **simulated** single H100 GPU with routed and shared expert weights in off-chip DRAM and a 32 GiB HBM expert cache. No physical GPU or model weights are needed.

**Repository URL for the report PDF:** https://github.com/altufaltu1901/moe-memory-wall-midsem

## Set up and run

Use Ubuntu Linux or Ubuntu on WSL 2. On a Windows machine without WSL, open PowerShell and run `wsl --install -d Ubuntu`, restart if prompted, and finish Ubuntu's first-run user setup. Then run these commands **inside Ubuntu**:

```bash
sudo apt update
sudo apt install -y build-essential cmake clang git python3 python3-venv
git clone --recurse-submodules https://github.com/altufaltu1901/moe-memory-wall-midsem.git
cd moe-memory-wall-midsem
bash midsem/reproduce.sh
```

`reproduce.sh` applies the repository's Ramulator patch and a GCC 15 header fix, builds with CMake, and runs the main 32 GB/s off-chip model plus 16 and 64 GB/s bandwidth checks. It saves each run's log, expert routes, and simulator CSV under `midsem/artifacts/`. The original supplied eight-GPU configuration is `config.yaml`; to run it after the build:

```bash
cd build
./run ../config.yaml > original_config.log 2>&1
cd ..
```

The unmodified upstream run already captured for this project is in [`midsem/artifacts/baseline/`](midsem/artifacts/baseline/). Its metric CSV has four rows because upstream omitted the final row; the modified branch fixes that omission. The experiment configurations are [`config_offchip.yaml`](config_offchip.yaml), [`config_offchip_bw16.yaml`](config_offchip_bw16.yaml), and [`config_offchip_bw64.yaml`](config_offchip_bw64.yaml).

## Regenerate the plots

After the simulator run, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
python midsem/analyze_midsem.py
python midsem/analyze_real_trace.py
```

The timing [plots and JSON summaries](midsem/artifacts/analysis/) are generated from the simulator logs and CSVs. The [routing-correlation plots](midsem/artifacts/real_trace/analysis/) use the checked-in 1,000-token [public GLM routing sample](https://huggingface.co/datasets/mistrjirka/Predicting-Experts-for-MOE-Suite), with [provenance](midsem/artifacts/real_trace/provenance.json). That trace belongs to a **different model** from the DeepSeekV3 timing simulation. To fetch the sample again, run `python midsem/fetch_public_trace.py` before the two analysis commands; the checked-in copy is sufficient for normal reproduction.

## Midsem evidence

- **Working provided configuration:** [`config.yaml`](config.yaml) is the supplied setup; [`baseline/`](midsem/artifacts/baseline/) has its exit code, metric CSV, and build versions.
- **Single GPU, no Logic-PIM, all experts off-chip:** [`config_offchip.yaml`](config_offchip.yaml) and the [source changes](src/module/expert.cpp) model DRAM-to-HBM loads with a finite cache. The simulated 80 GiB HBM capacity check passes.
- **Loading versus execution:** at 32 GB/s, mean expert loading is **2,478.93 ms** and expert execution is **23.68 ms** per decode iteration; total latency is **2,506.40 ms**. See [load-versus-compute plot](midsem/artifacts/analysis/load_vs_compute.png), [bandwidth sweep](midsem/artifacts/analysis/bandwidth_sweep.png), and [full numbers](midsem/artifacts/analysis/summary.json).
- **Temporal and spatial access patterns:** [synthetic-routing plots](midsem/artifacts/analysis/) and [separate real-routing plots](midsem/artifacts/real_trace/analysis/) show frequency, adjacent-token overlap, and within-layer co-access.

The off-chip link is a fixed-bandwidth analytical model without transfer contention or transfer/compute overlap. The independent synthetic DeepSeekV3 routes do not demonstrate real routing predictability; the GLM trace provides a separate correlation example.

**Report PDF reminder:** include the repository URL above directly in your PDF. The project brief requires students to write the report themselves. The brief does not specify a particular PDF filename or upload layout here; follow the course submission portal's instructions for those details.

The original LLMSimulator license is in [`LICENSE`](LICENSE).
