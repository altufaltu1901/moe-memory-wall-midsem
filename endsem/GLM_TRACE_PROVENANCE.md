# GLM-5.2 trace and expert sizes

The [pinned route dataset manifest](https://huggingface.co/datasets/mistrjirka/Predicting-Experts-for-MOE-Suite/raw/110677e41b102126aa98f820cbf7b2190c35a3d6/dataset_manifest.json) lists `zai-org/GLM-5.2` and the `GLM-5.2-UD-IQ2_XXS GGUF` capture variant. Its route labels are from that quantized run. The dataset manifest gives the variant name and does not record the hash of the GGUF used for capture, so the file-level link to the public conversion is inferred from the matching name and geometry.

Z.ai's [configuration at cf457fa](https://huggingface.co/zai-org/GLM-5.2/raw/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json) specifies 78 base layers (three dense and 75 MoE), hidden size 6144, dense FFN width 12288, expert FFN width 2048, 256 routed experts, one shared expert and top-8 routing. It also specifies 64 attention heads, query and KV ranks 2048 and 512, Q/K non-RoPE and RoPE dimensions 192 and 64, value head dimension 256 and vocabulary size 154880. Its attention uses dynamic sparse attention with a top-2048 index limit. The publisher's BF16 expert occupies 72 MiB from three matrices. The GGUF sizes below describe packed weights in the named conversion.

The published [token rows at offset 700](https://datasets-server.huggingface.co/rows?dataset=mistrjirka%2FPredicting-Experts-for-MOE-Suite&config=token_cube&split=train&offset=700&length=12) and [offset 760](https://datasets-server.huggingface.co/rows?dataset=mistrjirka%2FPredicting-Experts-for-MOE-Suite&config=token_cube&split=train&offset=760&length=12) match the saved expert IDs for all 24 tokens. Their absolute context positions are 1955–1966 and 2015–2026, respectively. The pinned source is [the task's token_cube parquet](https://huggingface.co/datasets/mistrjirka/Predicting-Experts-for-MOE-Suite/blob/110677e41b102126aa98f820cbf7b2190c35a3d6/data/token_cube/train/corpus-02-cpp-framed-rpc-parser.parquet), identified in the manifest by SHA-256 `548998630dfb101d73012625f0655752bda80e4cde385d68e982570be9f36878`. Both windows are below the top-2048 cutoff. The matched-context runs set input length to 1955 or 2015 and bypass synthetic queue aging. The exported route positions are verified against the recorded window. The BF16 experiment uses a fixed simulator prompt length of 1024. The [official Transformers model implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py) computes the index scores before taking the smaller of the cutoff and the available key count. Layers configured to share the index reuse the previous index results.

## Reproduce the packed sizes

1. Enumerate the [six pinned public GGUF shards](https://huggingface.co/unsloth/GLM-5.2-GGUF/tree/f4a6ccbfacca9d45556d3e838cc67f91965acf01/UD-IQ2_XXS). For the metadata-only shard 1 read its 9,423,744 bytes. For shards 2–6 send `Range: bytes=0-524287` and require HTTP 206. No tensor payload is required for calculating the sizes.
2. Parse the GGUF v3 tensor directory after the metadata. For each base layer 3–77 identify `blk.N.ffn_{gate,up,down}_{exps,shexp}.weight`. Gate and up have matrix dimensions 6144 × 2048 and down has 2048 × 6144. Routed tensors have a third dimension of 256. Shared tensors have one matrix per projection.
3. Use the [GGML block-size definitions](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py) to compute bytes per expert as matrix entries divided by elements per block, times bytes per block. Sum gate, up and down. Validate each full tensor size against the difference to the next tensor's GGUF data offset.
4. Write one row per layer to [glm52_gguf_expert_bytes.csv](glm52_gguf_expert_bytes.csv). This includes 75 base MoE layers and excludes the separate MTP layer. SHA-256 with LF line endings is `5fd7d91f60c2840afb29612b8cd01840b05a1d35ff2bad87064a07d182073289`.

Typical packed sizes are 11,304,960 bytes per routed expert and 27,623,424 bytes per shared expert. Layer 8 uses 14,745,600 routed and 34,013,184 shared bytes. Layers 75–77 use 13,172,736 routed bytes and the typical shared size. The aggregate pools are 219,370,487,808 routed bytes and 2,078,146,560 shared bytes. These are weight payload bytes, before any runtime allocation or unpacking overhead.

Hashes of the input metadata file and shard prefixes:

```text
1  9423744  7bf96eeabbe887e58b6c44364962731ddc9dc5bf46fec8d097c1dff64bea4a18
2   524288  8c6d337566110268fc6cb0128dada47781d5799a7360a04735f08cd7b2782945
3   524288  52a244784b06060d72a6d432bf3b17b7064b00af90d261ea9ce9df34f90a3679
4   524288  dd52676315fd0b558aec28946d3443dce665a7b8e72ba1fa64f49712659bb840
5   524288  b08283eebd484c6d78dba2bfa74e6ce0e8cdec96ff7367481d00c3448d25410c
6   524288  dca501865732727186b5cf5ddb4e50d6b9ef947468f2c90ba52f54b908dadfde
```
