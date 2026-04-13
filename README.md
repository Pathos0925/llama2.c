## llama2.c with Qwen3.5 architecture

> **Architecture update:** This fork has been migrated from the original Llama 2 architecture to **Qwen3.5**. Both Python training and C inference (fp32 + int8 quantized) are fully working. See [Architecture changes](#architecture-changes) below.

Train a Qwen3.5 hybrid-attention LLM in PyTorch, then inference it in pure C. Based on Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), updated with the Qwen3.5 architecture: hybrid linear/full attention layers, gated Q projections, partial RoPE, and Gated Delta Net linear attention.

Small LLMs can have surprisingly strong performance if you make the domain narrow enough (ref: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) paper). This repo is a "fullstack" train + inference solution with focus on minimalism and simplicity.

### quick start

```bash
# Train a model
python tinystories.py download
python tinystories.py pretokenize
python train.py config/qwen3_suggested.py

# Or train on SimpleStories with a custom 4K tokenizer
python simplestories.py download
python simplestories.py train_vocab --vocab_size=4096
python simplestories.py pretokenize --vocab_size=4096
python train.py config/qwen3_simplestories_4k.py

# Export and run in C (fp32)
python export.py out/model.bin --version 3 --checkpoint out/ckpt.pt
gcc -O2 -o run run.c -lm
./run out/model.bin -i "Once upon a time" -n 256

# Or quantized (int8, ~3.4x smaller, ~25% faster)
python export.py out/model_q8.bin --version 4 --checkpoint out/ckpt.pt
gcc -O2 -o runq runq.c -lm
./runq out/model_q8.bin -i "Once upon a time" -n 256

# Python inference (for testing/comparison)
python inference.py --checkpoint out/ckpt.pt --prompt "Once upon a time"
```

## models

| model | dim | n_layers | n_heads | n_kv_heads | max context length | parameters | val loss | download
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 260K | 64 | 5 | 8 | 4 | 512 | 260K | 1.297 | [stories260K](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K)
| OG | 288 | 6 | 6 | 6 | 256 | 15M | 1.072 | [stories15M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin) |
| 42M| 512 | 8 | 8 | 8 | 1024 | 42M | 0.847 | [stories42M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin) |
| 110M| 768 | 12 | 12 | 12 | 1024 | 110M | 0.760 | [stories110M.bin](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin) |

## architecture changes

The model architecture has been updated from Llama 2 to Qwen3.5. Key differences:

| Feature | Llama 2 (original) | Qwen3.5 (current) |
|---------|--------------------|--------------------|
| **Normalization** | RMSNorm, weight init=1 | RMSNorm, zero-init weight, `x * (1 + weight)` |
| **RoPE** | Full head_dim, theta=10K | Partial rotary (25%), theta=10M |
| **Full attention** | Standard Q projection | Gated Q (2x width + sigmoid gate), QK norm |
| **Linear attention** | N/A | Gated Delta Net with causal depthwise conv |
| **Layer pattern** | All identical | Hybrid: 3 linear + 1 full attention, repeating |
| **FFN** | SwiGLU | SwiGLU (unchanged) |

The hybrid attention design alternates between lightweight linear attention layers (Gated Delta Net) and full softmax attention layers. Linear attention layers use a delta rule recurrence instead of the quadratic attention matrix, making them more efficient for long sequences.

### Block Attention Residuals

Optional block-level attention residuals from [Moonshot AI (arXiv:2603.15031)](https://arxiv.org/abs/2603.15031). Instead of standard residual connections, sub-layers are grouped into blocks and an attention mechanism over completed blocks produces the input for each sub-layer. Enabled by setting `attnres_block_size > 0` (counts sub-layers: attention + MLP = 2 per transformer layer).

### export formats

| Version | Format | Command | C binary |
|---------|--------|---------|----------|
| v3 | fp32 | `python export.py out/model.bin --version 3 --checkpoint out/ckpt.pt` | `run` |
| v4 | int8 Q8_0 | `python export.py out/model_q8.bin --version 4 --checkpoint out/ckpt.pt` | `runq` |

Older v0/v1/v2 formats from the original llama2.c are incompatible with the new Qwen3.5 model.

## Memory Caching (MC)

The linear attention layers optionally support **Memory Caching** from ["Memory Caching: RNNs with Growing Memory"](https://arxiv.org/abs/2602.24281). MC segments the sequence and caches the Gated Delta Net state at segment boundaries, allowing each token to query compressed memories from earlier segments. This gives linear attention layers a growing memory that interpolates between the fixed state of RNNs and the full KV cache of Transformers.

We use the **Sparse Selective Caching (SSC)** aggregation strategy: a learned router selects the top-k most relevant cached segment memories per token, then softmax-weights their outputs with the current segment's online output.

MC is disabled by default (`mc_segment_size=0`) with zero overhead. To enable it:

```python
# In a config file or via command line
mc_segment_size = 128      # segment size in tokens (0=disabled)
mc_ssc_top_k = 2           # number of past segments to select per token
mc_detach_cached_states = True  # detach cached states from grad graph (saves memory)
```

```bash
python train.py config/qwen3_simplestories_4k_mc.py
```

See `MEMORY_CACHING_PLAN.md` for detailed design documentation.

## LoopLM (Loop-Cached Reasoning)

The model optionally supports **LoopLM** — weight-tied layer looping with adaptive exit — based on ["Scaling Latent Reasoning via Looped Language Models"](https://arxiv.org/abs/2501.xxxxx) (Zhu et al., 2025). LoopLM reuses the same transformer layers T times, giving the model more depth of computation without adding parameters. A learned exit gate predicts when to stop looping.

Combined with Memory Caching, this creates **Loop-Cached Reasoning**: after each loop iteration, the linear attention memory states are cached and made available to subsequent loops via SSC. This gives the model a structured, queryable record of what each pass of reasoning discovered — preventing "reasoning regression" where later loops overwrite useful features from earlier loops.

### how it works

1. **Weight-tied looping**: The same layer stack runs T times. Each loop refines the hidden states.
2. **Exit gate**: A learned gate predicts per-token exit probability at each step. At inference, tokens exit early once the cumulative exit probability exceeds a threshold.
3. **Loop-axis Memory Caching** (optional): After each loop, the final recurrent state of each linear attention layer is cached. Subsequent loops query these cached states via SSC with a separate router projection, blending insights from prior loops with the current loop's output.
4. **Two-stage training**:
   - **Stage I**: Joint training with entropy-regularized expected loss. The exit distribution weights per-step losses, and an entropy bonus prevents collapse to always-max-loops.
   - **Stage II**: Freeze the LM, train only the exit gate on a per-step improvement signal (does this loop step still help?).

### configuration

LoopLM is disabled by default (`loop_max_steps=1`) with zero overhead — no exit gate or extra projections are allocated.

```python
# LoopLM
loop_max_steps = 4          # number of loop iterations (1=disabled)
loop_kl_beta = 0.1          # entropy regularization coefficient (Stage I)
loop_exit_threshold = 0.9   # CDF threshold for early exit at inference
loop_training_stage = 1     # 1=joint training, 2=gate-only (Stage II)

# Loop-axis Memory Caching (requires mc_segment_size > 0)
loop_mc_enabled = True      # cache linear-attn states across loop iterations
loop_mc_top_k = 2           # top-k prior loop iterations to select via SSC
loop_mc_detach = True       # detach loop-cached states from grad graph
```

```bash
python train.py config/qwen3_simplestories_4k_loop_mc.py
```

### training notes

LoopLM requires careful training. The config uses these stability measures:

- **Conservative learning rate** (5e-4 vs. 1e-3 for non-looped) — recurrent architectures need smaller LR
- **Gradient clipping** at 1.0 — essential when gradients flow through T iterations
- **KL coefficient annealing** — `loop_kl_beta` linearly decays to 50% over the second half of training
- **Reduced batch size** (16 vs. 32) with increased gradient accumulation (8 vs. 4) — T loop iterations per step consume more memory
- **`compile=False`** — the loop with dynamic exit and MC segment loops cause graph breaks

See `LOOPLM_IMPLEMENTATION_PLAN.md` for detailed design documentation.

## wandb logging

Training metrics are optionally logged to [Weights & Biases](https://wandb.ai). Enable with:

```python
wandb_log = True
wandb_project = "llamac"
wandb_run_name = "my_run"
```

Logged metrics include train/val loss, learning rate, MFU, and iteration time. When LoopLM is enabled, per-loop-step losses, exit probability distribution, mean exit step, and KL beta are also logged.

## datasets

Two datasets are supported:

- **TinyStories** — synthetic children's stories generated by GPT-3.5/4. Uses the Llama 2 tokenizer (32K vocab) by default.
- **SimpleStories** — a more diverse dataset from HuggingFace. Supports custom tokenizer training for smaller vocabularies (e.g., 4K tokens).

```bash
# TinyStories
python tinystories.py download
python tinystories.py pretokenize

# SimpleStories with custom 4K tokenizer
python simplestories.py download
python simplestories.py train_vocab --vocab_size=4096
python simplestories.py pretokenize --vocab_size=4096
```

## config files

Training configs live in the `config/` directory:

| Config | Description |
|--------|-------------|
| `qwen3_original.py` | Original Qwen3.5 defaults on TinyStories |
| `qwen3_suggested.py` | Tuned hyperparameters (recommended baseline) |
| `qwen3_simplestories_4k.py` | SimpleStories with 4K custom tokenizer |
| `qwen3_simplestories_4k_mc.py` | + Memory Caching (SSC) |
| `qwen3_simplestories_4k_loop_mc.py` | + LoopLM + Loop-Cached Reasoning |

You can also override individual params: `python train.py config/qwen3_suggested.py --max_iters=50000`

## unsorted todos

- add support in run.c of reading version 1+ files from export, later deprecate "version 0"
- run.cu (CUDA) investigate and merge
- add more tests inside [test.c](test.c)
- add Engine class for use in sample.py that does efficient inference in PyTorch, e.g. KV cache keeping
- C inference support for LoopLM (loop unrolling + exit gate export)
- C inference support for Memory Caching (segment loop export)

## License

MIT
