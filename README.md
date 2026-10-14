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

The experiment configurations are [`config_offchip.yaml`](config_offchip.yaml), [`config_offchip_bw16.yaml`](config_offchip_bw16.yaml), and [`config_offchip_bw64.yaml`](config_offchip_bw64.yaml).

### Supplied Duplex configuration: reproduction evidence

The supplied [`config.yaml`](config.yaml) specifies **one node with eight B100 GPUs**, a DeepSeekV3 model configuration, and five decode iterations. We ran this original configuration before modifying the simulator. Its saved [exit code is 0](midsem/artifacts/baseline/exit_code.txt), and the [metric CSV](midsem/artifacts/baseline/) records the completed simulation. The [versions file](midsem/artifacts/baseline/versions.txt) pins LLMSimulator `419252761fbdb95b789778a02256d458a5537ec7` and Ramulator2 `b5262869b24790e5ec62d9d9974fca9bbb4f54e0`, along with the build environment. The upstream CSV contains four metric rows although five iterations were requested; the original exporter omitted the final row.

The modified experiment uses **one H100 GPU**, batch size four, and the added off-chip expert path. The original eight-B100 run demonstrates that the supplied configuration works; its latency is not a direct performance comparison with the modified run because the hardware, batch size, and memory path differ.

The report's additional [skewed-routing cache check](midsem/artifacts/cache_check/) and [offloading-disabled timing check](midsem/artifacts/compute_only/) have their configurations and saved results in the repository. These are diagnostic checks; the offloading-disabled model exceeds the simulated H100 capacity.

### Run from the submitted ZIP

The submission ZIP includes the Ramulator2 source under `src/dram/ramulator2`, so no submodule download is needed after extraction. On Ubuntu or WSL 2, install the packages above, then from the extracted ZIP directory run `bash midsem/reproduce.sh`. To regenerate the plots, run the Python commands below. The supplied original configuration is `config.yaml`; its saved upstream reproduction evidence is in `midsem/artifacts/baseline/`.

## Regenerate the plots

After the simulator run, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
python midsem/analyze_midsem.py
python midsem/analyze_real_trace.py
python midsem/analyze_cross_layer.py
```

The timing [plots and JSON summaries](midsem/artifacts/analysis/) are generated from the simulator logs and CSVs. The [routing-correlation plots](midsem/artifacts/real_trace/analysis/) use the checked-in 1,000-token [public GLM routing sample](https://huggingface.co/datasets/mistrjirka/Predicting-Experts-for-MOE-Suite), with [provenance](midsem/artifacts/real_trace/provenance.json). The [adjacent-layer plot](midsem/artifacts/real_trace/analysis/cross_layer_recall.png) tests whether a token's experts at one GLM layer help identify its experts at the next layer, using the first 700 tokens for association counts and the last 300 for evaluation. That trace belongs to a **different model** from the DeepSeekV3 timing simulation. To fetch the sample again, run `python midsem/fetch_public_trace.py` before the analysis commands; the checked-in copy is sufficient for normal reproduction.

## Midsem evidence

- **Working provided configuration:** [`config.yaml`](config.yaml) is the supplied setup; [`baseline/`](midsem/artifacts/baseline/) has its exit code, metric CSV, and build versions.
- **Single GPU, no Logic-PIM, all experts off-chip:** [`config_offchip.yaml`](config_offchip.yaml) and the [source changes](src/module/expert.cpp) model DRAM-to-HBM loads with a finite cache. The simulated 80 GiB HBM capacity check passes.
- **Loading versus execution:** at 32 GB/s, mean expert loading is **2,478.93 ms** and expert execution is **23.68 ms** per decode iteration; total latency is **2,506.40 ms**. See [load-versus-compute plot](midsem/artifacts/analysis/load_vs_compute.png), [bandwidth sweep](midsem/artifacts/analysis/bandwidth_sweep.png), and [full numbers](midsem/artifacts/analysis/summary.json).
- **Temporal and spatial access patterns:** [synthetic-routing plots](midsem/artifacts/analysis/) and [separate real-routing plots](midsem/artifacts/real_trace/analysis/) show frequency, adjacent-token overlap, within-layer co-access, and held-out adjacent-layer association.

The off-chip link is a fixed-bandwidth analytical model without transfer contention or transfer/compute overlap. The independent synthetic DeepSeekV3 routes do not demonstrate real routing predictability; the GLM trace provides a separate correlation example.

**Report PDF reminder:** include the repository URL above directly in your PDF. The project brief requires students to write the report themselves. The brief does not specify a particular PDF filename or upload layout here; follow the course submission portal's instructions for those details.

The original LLMSimulator license is in [`LICENSE`](LICENSE).
