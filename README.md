# Chat with a 95M parameter model.

<p align="center">
  <img src="assets/llamathink.jpg" width="430" height="300" alt="Llama Think">
</p>

A 95.7M parameter hybrid-attention language model pretrained on [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (10B tokens) and finetuned on simple conversations. Built on a Qwen3.5 backbone with DeepSeek V4 architectural innovations, trained with the Muon optimizer on 4x NVIDIA B300 GPUs.

## Model architecture

| | |
|---|---|
| **Parameters** | 95.7M |
| **Dimensions** | 576 |
| **Layers** | 16 (hybrid pattern) |
| **Attention heads** | 9 (3 KV heads, GQA) |
| **Head dim** | 64 |
| **Context length** | 1024 tokens |
| **Vocabulary** | 32K (Llama 2 tokenizer) |
| **RoPE** | Partial (50%), theta=20K |

### Hybrid attention layers

The model uses a repeating 4-layer pattern: **2 linear + 1 compressed + 1 full attention**, tiled 4 times.

```
linear → linear → CSA → full  (×4 = 16 layers)
```

- **Linear attention** (Gated Delta Net) — O(n) recurrence with causal depthwise conv, learned gating, and delta rule updates. 9 key/value heads, 112-dim keys and values.
- **Compressed Sparse Attention (CSA)** — Dual-path design from DeepSeek V4. A learned stride-4 compressor produces compressed KV for long-range context, while a 64-token local window preserves fine-grained detail. A learned gate blends the two paths per-token.
- **Full attention** — Standard softmax causal attention with gated Q projections, QK normalization, and grouped-query attention (9 heads, 3 KV heads).

### Hyper-Connections

Every layer uses learned per-dimension residual scaling (zero-initialized). Instead of `h = h + sublayer(h)`, each sublayer computes:

```
h = (1 + residual_scale) * h + (1 + sublayer_scale) * sublayer(h)
```

This gives the model fine-grained control over information flow without adding meaningful parameter count.

## Training

### Optimizer: Muon

[Muon](https://github.com/KellerJordan/Muon) applies Newton-Schulz orthogonalization to the momentum buffer, normalizing all singular values toward 1 for faster convergence. Embeddings and the output head fall back to AdamW.

| | |
|---|---|
| **Learning rate** | 0.02 (Muon) / 0.002 (AdamW fallback) |
| **Momentum** | 0.95 (Nesterov) |
| **NS iterations** | 5 |
| **Weight decay** | 0.01 |
| **Gradient clip** | 1.0 |
| **Warmup** | 500 steps |
| **Schedule** | Cosine decay to 0 |

### Pretraining data: FineWeb

Pretrained on [FineWeb sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb) — 10 billion tokens of cleaned web text from Common Crawl, curated by HuggingFace.

| | |
|---|---|
| **Dataset** | FineWeb sample-10BT |
| **Documents** | 14.9M |
| **Tokens** | ~10B |
| **Tokenizer** | Llama 2 SentencePiece (32K vocab) |
| **Epochs** | ~2 |
| **Batch size** | 64 × 8 grad accum × 4 GPUs = 2.1M tokens/iter |
| **Total iters** | 10,000 |

### Hardware

4x NVIDIA B300 SXM6 (275 GB HBM3e each). With `torch.compile`, steady-state iteration time is ~2.35s at 23% MFU. Full pretraining takes approximately 7 hours.

## Quick start

### Pretrain from scratch

```bash
# Download and tokenize FineWeb (10B tokens, ~28 GB download)
python fineweb.py download
python fineweb.py pretokenize

# Train on 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py config/fineweb_95m_dsv4_4xB300.py
```

### Export and run in C

```bash
python export.py out_fineweb_95m_dsv4/model.bin --version 3 --checkpoint out_fineweb_95m_dsv4/ckpt.pt
gcc -O2 -o run run.c -lm
./run out_fineweb_95m_dsv4/model.bin -i "The history of" -n 256
```

### Monitor training

```bash
tail -f train.log                          # live output
grep "| loss" train.log | tail -20         # recent loss values
nvidia-smi                                 # GPU utilization
```

Training metrics (loss, LR, MFU, samples) are logged to [Weights & Biases](https://wandb.ai). Checkpoints are saved every 1000 steps and uploaded to Cloudflare R2.

## Flash Linear Attention

The linear attention layers automatically use fused Triton kernels from [`flash-linear-attention`](https://github.com/sustcsonglin/flash-linear-attention) when installed, providing roughly 4x throughput improvement over the naive PyTorch fallback.

```bash
pip install flash-linear-attention
```

## License

MIT
