# Prefetch scheduling research notes

- [Duplex, MICRO 2024](https://doi.org/10.1109/MICRO61859.2024.00105) separates LLM work across conventional compute and Logic-PIM. This project studies a single-GPU offload variant. The measured DDR4 expert service and analytical GPU timing are separate parts of that variant.
- [ST-MoE, §4.2](https://arxiv.org/html/2606.15453v1) maintains cross-layer association and confidence using observed routes. Our causal counts apply that insight without a trained router. The router trace still determines the requested experts.
- [APEX, §IV](https://arxiv.org/html/2608.11688v1) uses a trained confidence model to adapt how many experts it prefetches. We use a fixed byte cap and a known shared expert because trace-only inputs provide route IDs and no activations for a learned prefetch router.
- [Pre-gated MoE, §IV–V](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf) advances expert selection through an architecture change so transfers can overlap with existing computation. The experiment here advances only an already-known dependency. The shared expert runs every layer and needs no prediction or model change.
- [SpecPrefetch](https://arxiv.org/abs/2607.24787) couples next-layer prediction to an overlap budget using a learned adapter. Its system design motivates measuring elapsed time and transferred bytes separately from prediction quality.
- [Fast Inference of MoE with Offloading](https://arxiv.org/abs/2312.17238) describes offloading and caching experts to exploit temporal reuse. We hold effective cache capacity constant within each paired comparison so space reserved by the prefetcher does not distort the result.

## Design tested

1. Give a known shared expert an earlier start at its layer’s routed-compute stage. A shared expert already in HBM needs no transfer.
2. Keep the 1 MiB speculative prefix for the next routed layer. A wrong prediction spends at most the prefix bytes. If the guess is right, demand loads the rest.
3. Use a single link and reserved staging slot. Any transfer that is still active when demanded imposes the remaining wait; the scheduler does not assume it was fully hidden.

A useful next experiment would learn a per-layer cutoff from earlier tokens and freeze it before testing a different task. Evaluate whether fewer wrong speculative prefixes also reduce queue interference and energy. Papers with trained routers, model modifications, different links, or different cache budgets are design references; their reported speedups are not modeled by this code.
