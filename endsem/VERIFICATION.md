# Course brief verification

Scope: Project 2, pages 5Ã¢â‚¬â€œ6 of the course project list. This is a reproducibility checklist for the code and experiment artifacts.

## Mid evaluation

1. **Run the supplied Duplex configuration.** The eight-B100 configuration is in config.yaml. Its recorded exit code, metric CSV and pinned versions are under midsem/artifacts/baseline. The supplied run selected GPU execution and disabled Ramulator. It establishes reproduction of the provided configuration. Reproduction of the paper's Logic-PIM experiments has no evidence here.
2. **Use one GPU, disable sharding and Logic-PIM, and place all expert weights off chip.** config_offchip.yaml selects one H100 with GPU execution and expert parallel degree one. The capacity log inventories approximately 600.84 GiB of routed and shared weights off chip and reserves 32 GiB of HBM for the expert cache. The saved timing breakdown reports loads and expert execution separately.
3. **Explain execution and the loading bottleneck.** The midsem results include a load-versus-compute breakdown and three bandwidth settings. The final report should describe router selection, HBM cache lookup, demand loads on misses and execution of selected experts, with the team's interpretation.
4. **Analyze temporal and spatial expert access.** The midsem artifacts contain expert frequencies, within-layer co-access, consecutive-token overlap and adjacent-layer association. The public GLM route trace and synthetic DeepSeekV3 runs identify their data sources separately.

## Final evaluation

1. **Training-free prediction and overlap.** replay.py updates historical frequencies and co-occurrence counts only after routing is observed. The 77-case endsem matrix includes previous-token and adjacent-layer ablations. GLM overlap uses the last 300 tokens after a 700-token warm-up. The five-step DeepSeek experiments use synthetic routes and provide a timing and cache sanity check.
2. **Prefetch from off-chip DRAM into GPU HBM.** replay.py models one serialized link, shared cache capacity, eviction, wrong predictions and partially completed transfers. Original routes determine executed experts. The coupled C++ experiment additionally stages a previous-token prediction inside LLMSimulator and services expert requests using a patched Ramulator DDR4 controller.
3. **Compare on-demand and prefetching runs.** The replay first reproduces the saved on-demand counts, transferred bytes and load time. The output records total latency, amortized token time, throughput, link service, exposed stall and traffic. Match the workload and configuration before comparing.
4. **Evaluate energy and extra traffic.** The endsem results separate saved LLMSimulator dynamic energy, energy per transferred bit and incremental waiting energy. Sensitivity to transfer and waiting costs is included. Off-chip energy is estimated, with no DRAM power model or hardware energy measurements.
5. **Final report, complete codebase, one submission.** The source package includes the runnable code, pinned dependencies and evidence. The final student-authored report remains required. Package the final materials once for the group.
6. **Simulator and cited methods.** LLMSimulator supplies routes, GPU timings and component energy estimates. The temporal policy also runs in the coupled executable. The wider set of policies runs in replay. The analysis uses adjacent-layer and temporal history inspired by ST-MoE and varies prefetch budgets in the spirit of APEX. The SSD-offloading framework is optional in the brief.

## Added verification

- check.py checks causal prediction, partially completed transfers, the oracle's
  treatment of resident experts, and reported energy components.
- verify_more.py independently checks the latency/stall identity, the available
  overlap window and energy balance for 56 matched policy comparisons. It also
  computes break-even waiting power where a policy saves time but adds traffic.
- The Ramulator probe executes pinned, patched Ramulator 2.0. Its raw logs show
  accepted and completed reads, first completion, total cycles and DRAM clock.
  Full-weight response times feed the pipelined and serial replay assumptions.
  This tests the DRAM side of the memory model separately from the LLMSimulator
  executable. A coupled run with Ramulator handling each expert request and the
  GPU scheduler executing prefetch events remains an extension.

- check_coupled.py checks exit codes, exact routed expert identity between each
  matched pair, completed DDR4 request counts, aggregate bytes, outcomes for
  issued prefetches, and the latency/stall accounting.
