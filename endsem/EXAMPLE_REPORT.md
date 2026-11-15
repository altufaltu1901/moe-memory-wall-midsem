# Example report structure: overlapping weight transfers

Worked example for practice. This uses a toy architecture and invented numbers. Use the evidence guide below to prepare your own project report.

## Abstract example

In a two-stage decoder simulator, one worker starts loading a weight block while another worker computes. The on-demand schedule takes 10 ms per step. The overlapped schedule takes 9 ms with the same 8 MiB of traffic. This experiment illustrates how a known dependency can create a small but repeatable latency saving even when the transfer dominates execution. The numbers describe the toy simulator.

## Method example

Assume the first stage computes for 2 ms. The second stage uses a weight block that takes 5 ms to transfer and then computes for 3 ms. In the on-demand schedule, the stages take 2 + 5 + 3 = 10 ms. If a dependency signal makes the block available for transfer 1 ms before the first compute stage ends, 1 ms of the transfer can overlap with compute. The remaining 4 ms is exposed and the step takes 9 ms. Both schedules fetch the block once.

A guessed block has a different cost. If a wrong guess occupies the link for 2 ms after demand arrives, the same 1 ms of overlap may be outweighed by the extra waiting. Report elapsed time, exposed transfer time, and bytes separately.

## Evidence guide for your own report

1. **Problem and baseline** â€” identify the one-GPU offload design and distinguish the original eight-B100 reproduction. Use the pinned versions and original exit code in [baseline evidence](../midsem/artifacts/baseline/).
2. **Routing evidence** â€” separate full candidate-set recall from the precision of transfers actually issued. Explain how the earlier observed tokens supply the counts and how the current source layer predicts the next layer. Use [prediction results](artifacts/summary.csv).
3. **Design and simulator** â€” sketch the routed-load, routed-compute, and shared-compute stages. Explain what is sampled in Ramulator and which GPU quantities are proxy values. Use the [reproduction guide](README.md) and [GLM trace and weight provenance](GLM_TRACE_PROVENANCE.md).
4. **Matched comparison** â€” report absolute milliseconds per token alongside relative change. Include on demand, full-weight speculation, bounded speculation, and the shared-expert scheduling experiment. Report traffic, exposed stalls, and modeled energy, with the assumptions and matching checks. Use the [oracle and analytical ceiling evidence](artifacts/ideal_ceiling.json) to separate perfect route knowledge from perfect overlap. The [matched scheduling CSV](artifacts/glm_schedule_comparison.csv) is a starting point.
5. **Limitations** â€” identify the number of tokens and trajectories, cache and link assumptions, and the distinction between visibility of a required weight and how much transfer actually finished in time.
6. **Reproduction** â€” include your repository URL, simulator commit, exact run commands, and artifact locations.
