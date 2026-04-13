# LoopLM + Memory Caching Implementation Plan

## Context

We have a Qwen3.5-style hybrid architecture (3 linear + 1 full attention layers, repeating) with Memory Caching (SSC) already integrated on linear attention layers. The goal is to add **LoopLM** — weight-tied layer looping with adaptive exit — and combine it with **loop-axis Memory Caching**, creating what the speculation doc calls "Loop-Cached Reasoning."

### What LoopLM Gives Us
LoopLM reuses the same transformer layers T times, giving the model more compute depth without more parameters. An exit gate learns when to stop looping. This improves reasoning and knowledge manipulation without increasing model size.

### What Loop-Axis MC Gives Us
By caching the memory state after each loop iteration (treating loops as "segments" on the depth axis), the model retains explicit access to what it discovered in prior iterations. This prevents "reasoning regression" — where later loops overwrite useful features from earlier loops.

---

## Files to Modify

| File | Changes |
|------|---------|
| `model.py` | ModelArgs (new fields), ExitGate class, Transformer.forward (loop logic + loop-axis MC), per-step loss collection |
| `train.py` | New config vars, two-stage training support, loss computation changes, wandb metrics |

No changes needed to: `LinearAttention`, `TransformerBlock`, `Attention`, `FeedForward`, `export.py`, `inference.py`, data loading.

---

## 1. ModelArgs Additions (`model.py`)

After the existing MC fields, add:

```python
# LoopLM -- weight-tied layer looping with adaptive exit
loop_max_steps: int = 1           # 1 = no looping (standard model); >1 = LoopLM enabled
loop_kl_beta: float = 0.1         # entropy regularization coefficient (Stage I)
loop_exit_threshold: float = 0.9  # CDF threshold q for early exit at inference

# Loop-axis Memory Caching (only active when both loop_max_steps>1 and mc_segment_size>0)
loop_mc_enabled: bool = False     # cache memory states across loop iterations
loop_mc_top_k: int = 2            # top-k past loop iterations to select via SSC
loop_mc_detach: bool = True       # detach loop-cached states from grad graph
```

**Design note**: `loop_max_steps=1` makes the model identical to the current non-looped model. Zero overhead, zero code path changes. This mirrors the `mc_segment_size=0` pattern.

---

## 2. ExitGate Class (`model.py`, new class)

A small module that predicts per-token exit probability at each loop step.

```python
class ExitGate(nn.Module):
    """LoopLM exit gate: predicts probability of exiting at each loop step."""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1, bias=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, D] -> lambda_t: [B, T] in (0, 1)"""
        return torch.sigmoid(self.proj(h).squeeze(-1))
```

This is the `lambda_t(x) = sigma(Linear(h^(t)))` from LoopLM Section 3.2. One instance, shared across all loop steps.

**Placement**: Initialized in `Transformer.__init__` when `loop_max_steps > 1`.

---

## 3. Transformer.__init__ Changes (`model.py`)

When `loop_max_steps > 1`:

```python
# Exit gate
self.exit_gate = ExitGate(params.dim)

# Loop-axis MC: router projection for selecting relevant prior-loop memories
# Only needed when both looping and loop-MC are enabled
if params.loop_mc_enabled and params.mc_segment_size > 0:
    # One router per linear attention layer, projecting dim -> num_v_heads * head_k_dim
    # We can reuse the existing mc_u_proj for sequence-axis routing,
    # so we need a SEPARATE projection for loop-axis routing.
    # Store these on the LinearAttention layers.
    for layer in self.layers:
        if layer.layer_type == "linear":
            la = layer.attention
            la.loop_mc_top_k = params.loop_mc_top_k
            la.loop_mc_detach = params.loop_mc_detach
            la.loop_mc_u_proj = nn.Linear(
                params.dim, la.num_v_heads * la.head_k_dim, bias=False
            )
```

**Why separate router projections?** The existing `mc_u_proj` routes across sequence segments. Loop-axis routing asks a different question ("which prior loop's memory is relevant?") so it needs its own learned projection. The paper's finding that sharing gate projection with query causes training collapse (Section 5.6) reinforces keeping these separate.

---

## 4. Transformer.forward Changes — The Core Loop (`model.py`)

This is the main change. The current forward pass runs each layer once:

```python
# CURRENT (non-looped)
for layer in self.layers:
    blocks, partial_block = layer(blocks, partial_block, freqs_cos, freqs_sin)
```

With LoopLM, we run the full layer stack T times:

```python
# NEW (looped)
if self.params.loop_max_steps <= 1:
    # Standard non-looped path (unchanged)
    for layer in self.layers:
        blocks, partial_block = layer(blocks, partial_block, freqs_cos, freqs_sin)
    h = self.norm(partial_block)
else:
    # LoopLM path
    loop_losses = []          # per-step CE losses for training
    exit_lambdas = []         # per-step exit probabilities
    loop_memory_cache = {}    # layer_idx -> list of cached states per loop

    for t in range(self.params.loop_max_steps):
        # Reset blocks for each loop iteration (fresh block attention context)
        blocks_t = [h]  # h is the current hidden state
        partial_block_t = h

        for layer in self.layers:
            blocks_t, partial_block_t = layer(
                blocks_t, partial_block_t, freqs_cos, freqs_sin
            )

        h_out = self.norm(partial_block_t)

        # Exit gate
        lambda_t = self.exit_gate(h_out)  # [B, T]
        exit_lambdas.append(lambda_t)

        # Compute per-step loss (needed for training)
        if targets is not None:
            logits_t = self.output(h_out)
            loss_t = F.cross_entropy(
                logits_t.view(-1, logits_t.size(-1)),
                targets.view(-1), ignore_index=-1, reduction='none'
            ).view(tokens.shape)  # [B, T]
            loop_losses.append(loss_t)

        # Update hidden state for next iteration
        # The refined output becomes input to the next loop
        h = h_out
```

### 4.1 Loop-Axis Memory Caching Integration

The key insight from the speculation doc: after each loop iteration, cache the linear attention memory states, and let subsequent loops query them via SSC.

This requires modifying how `LinearAttention.forward` is called within the loop. We need to:

1. **After each loop**: Extract the final memory state from each linear attention layer and cache it
2. **Before each loop (t>0)**: Provide the cached loop-iteration memories to each linear attention layer

**Approach — extend LinearAttention with loop-axis cache methods:**

```python
# On LinearAttention, add:
def cache_loop_state(self, state, context):
    """Called after a loop iteration to save this loop's final memory state."""
    # state: [B, H, k, v], context: [B, H, k]
    ...

def get_loop_augmented_output(self, online_out, query, u_loop, loop_cache):
    """SSC aggregation across cached loop-iteration memories."""
    # Reuse the same _mc_ssc_aggregate logic with loop_mc_top_k
    ...
```

However, this creates a problem: `LinearAttention.forward` currently handles the full sequence-axis MC internally and returns the final output. We need access to intermediate states.

**Cleaner approach — post-hoc loop-axis aggregation:**

Rather than modifying `LinearAttention.forward`, we handle loop-axis MC at the `Transformer` level:

1. After each loop iteration t, for each linear attention layer, capture its per-segment final states (which are already computed during sequence-axis MC). The "loop memory" for layer `l` at loop `t` is the **concatenation of all segment states** — effectively a summary of what that layer learned across the full sequence during loop t.
2. For simplicity, use the **last segment's final state** as the loop-iteration memory (it incorporates information from all prior segments via checkpoint initialization).
3. Before loop t+1, inject these cached loop states as additional context.

**Implementation detail**: Add an optional `loop_cached_states` parameter to `LinearAttention.forward`:

```python
def forward(self, x, loop_cached_states=None, loop_cached_contexts=None):
    # ... existing code ...
    # After computing core_out (whether MC-enabled or not):
    # If loop_cached_states provided, do SSC aggregation across loop iterations
    if loop_cached_states is not None and len(loop_cached_states) > 0:
        u_loop = self.loop_mc_u_proj(x).reshape(bsz, seq_len, self.num_v_heads, self.head_k_dim)
        core_out = self._loop_mc_aggregate(core_out, query, u_loop,
                                            loop_cached_states, loop_cached_contexts)
    # ... rest of forward (norm, out_proj) ...
```

The `_loop_mc_aggregate` method is structurally identical to `_mc_ssc_aggregate` but operates on loop-cached states instead of segment-cached states. We can factor out a shared `_ssc_aggregate` helper.

**What gets cached per loop iteration per layer:**
- `final_state`: `[B, num_v_heads, head_k_dim, head_v_dim]` — the memory matrix after the last segment of this loop
- `context`: `[B, num_v_heads, head_k_dim]` — sum of all keys across the full sequence in this loop (for routing)

---

## 5. Loss Computation — Training Stage I (`train.py` + `model.py`)

LoopLM Stage I uses an entropy-regularized expected loss:

```
L = sum_t p(t|x) * L^(t) - beta * H(p(·|x))
```

Where `p(t|x)` is the normalized exit distribution derived from the exit gate lambdas.

**In Transformer.forward, after the loop completes:**

```python
if targets is not None and self.params.loop_max_steps > 1:
    # Compute exit distribution from lambdas
    # lambda_t: list of [B, T] tensors
    # p(t|x) = lambda_t * prod_{j<t}(1 - lambda_j)  (for t < T_max)
    # p(T_max|x) = prod_{j<T_max}(1 - lambda_j)     (forced exit)

    survival = torch.ones_like(exit_lambdas[0])  # [B, T]
    exit_probs = []
    for t in range(len(exit_lambdas)):
        if t < len(exit_lambdas) - 1:
            p_t = exit_lambdas[t] * survival
            exit_probs.append(p_t)
            survival = survival * (1 - exit_lambdas[t])
        else:
            exit_probs.append(survival)  # forced exit at T_max

    exit_probs = torch.stack(exit_probs, dim=0)  # [T_max, B, T]

    # Expected loss: sum_t p(t) * L^(t)
    # loop_losses: list of [B, T] tensors
    losses_stack = torch.stack(loop_losses, dim=0)  # [T_max, B, T]
    expected_loss = (exit_probs * losses_stack).sum(dim=0).mean()

    # Entropy bonus: H(p) = -sum_t p(t) log p(t)
    log_probs = torch.log(exit_probs + 1e-8)
    entropy = -(exit_probs * log_probs).sum(dim=0).mean()

    self.last_loss = expected_loss - self.params.loop_kl_beta * entropy

    # Store per-step losses for Stage II and logging
    self.loop_step_losses = [l.mean().item() for l in loop_losses]
    self.loop_exit_distribution = exit_probs.mean(dim=(1, 2)).detach().cpu().tolist()
```

**For non-looped models (loop_max_steps=1)**: Loss computation is unchanged.

---

## 6. Inference — Early Exit (`model.py`, Transformer.forward)

At inference time (`targets is None`), use the CDF-based early exit:

```python
# Inside the loop, after computing lambda_t:
if targets is None:  # inference
    # CDF: cumulative exit probability
    # Exit when CDF >= threshold for ALL tokens in the batch
    cdf = 1.0 - survival_prob  # after updating survival
    if cdf.min() >= self.params.loop_exit_threshold:
        break  # early exit
```

For generation, this means "easy" tokens exit after 1-2 loops while "hard" tokens use all T loops. The `loop_exit_threshold` parameter (default 0.9) controls the tradeoff.

---

## 7. Training Stage II — Adaptive Gate Training (`train.py`)

Stage II freezes the LM parameters and trains only the exit gate on a per-step improvement signal.

**Implementation in train.py:**

```python
# New config variable
loop_training_stage = 1  # 1 = joint training (default), 2 = gate-only training

# In the training loop, when stage == 2:
if loop_training_stage == 2:
    # Freeze everything except exit gate
    for name, param in raw_model.named_parameters():
        param.requires_grad = 'exit_gate' in name
    # Recreate optimizer with only gate params
    gate_params = [p for n, p in raw_model.named_parameters() if 'exit_gate' in n]
    optimizer = torch.optim.AdamW(gate_params, lr=learning_rate, betas=(beta1, beta2))
```

**Stage II loss (computed in Transformer.forward when stage==2):**

```python
# Improvement metric: I^(t) = max(0, L^(t-1) - L^(t))
# Target: w^(t) = sigmoid(k * (I^(t) - gamma))  with k=50, gamma=0.005
# Loss: binary cross-entropy between (1-lambda_t) and w^(t)

if self.training_stage == 2:
    adaptive_losses = []
    for t in range(1, len(loop_losses)):
        with torch.no_grad():
            improvement = torch.clamp(loop_losses[t-1] - loop_losses[t], min=0)
            w_t = torch.sigmoid(50.0 * (improvement - 0.005))
        lambda_t = exit_lambdas[t]
        # BCE: w_t*log(1-lambda_t) + (1-w_t)*log(lambda_t)
        bce = w_t * torch.log(1 - lambda_t + 1e-8) + (1 - w_t) * torch.log(lambda_t + 1e-8)
        adaptive_losses.append(-bce.mean())
    self.last_loss = torch.stack(adaptive_losses).mean()
```

---

## 8. Training Configuration (`train.py`)

New config variables:

```python
# LoopLM
loop_max_steps = 1             # 1 = disabled; >1 = number of loop iterations
loop_kl_beta = 0.1             # entropy regularization (Stage I)
loop_exit_threshold = 0.9      # CDF threshold for inference early exit
loop_training_stage = 1        # 1 = joint train, 2 = gate-only

# Loop-axis Memory Caching
loop_mc_enabled = False
loop_mc_top_k = 2
loop_mc_detach = True
```

Add all to `model_args` dict and resume forced-keys list.

---

## 9. Training Stability Measures

These are critical — LoopLM is known to be unstable without them.

### 9.1 Start with Fewer Loop Steps
- Begin training with `loop_max_steps=4` (not 8). The LoopLM paper found 8 caused loss spikes; they reduced to 4.
- For our tiny model, even `loop_max_steps=3` may be sufficient initially.

### 9.2 Conservative Learning Rate
- Use a lower LR than non-looped training. The LoopLM paper emphasizes this.
- Suggestion: `learning_rate=5e-4` (vs. `1e-3` for non-looped MC config).

### 9.3 KL Coefficient Annealing
- Start with `loop_kl_beta=0.1`, reduce to `0.05` partway through training.
- Implement as a simple schedule in train.py:
```python
if loop_max_steps > 1 and iter_num > max_iters // 2:
    raw_model.params.loop_kl_beta = loop_kl_beta * 0.5
```

### 9.4 Gradient Clipping
- Already have `grad_clip=1.0`. Keep this — essential for looped architectures where gradients flow through T iterations.

### 9.5 torch.compile Disabled
- The loop with dynamic exit and memory caching creates graph breaks. Must use `compile=False` (already the case for MC configs).

---

## 10. TransformerBlock — No Changes Needed

The key LoopLM insight is that **weight tying is implicit**: we literally reuse the same `self.layers` ModuleList in each loop iteration. No duplication, no special weight-sharing logic. The existing `TransformerBlock` and its sub-modules work as-is.

The only question is **block attention residuals across loops**. Current behavior: `blocks` accumulates across layers within a single forward pass. With looping, we reset `blocks` at the start of each loop iteration (each loop gets a fresh depth-residual context). This is the right default — each loop is a fresh "pass" over the input.

---

## 11. Memory and Compute Overhead

### Parameters Added
- **ExitGate**: `dim + 1` = 289 params (negligible)
- **Loop-axis MC router projections** (per linear attention layer): `dim * num_v_heads * head_k_dim` = `288 * 4 * 144` = ~166K per layer, ~1.24M total for ~7.5 linear layers. Small vs. total model size.

### Compute
- Each loop iteration ≈ one full forward pass through all layers.
- `loop_max_steps=4` means ~4x the compute per training step.
- Loop-axis MC adds minor overhead (one SSC aggregation per linear layer per loop).
- **Effective throughput**: ~4x slower per iteration, but LoopLM achieves better loss-per-parameter, so total tokens-to-target-loss may be comparable.

### Memory
- Must store intermediate activations for all T loop steps during training (for backprop through the loop).
- With `loop_mc_detach=True`, loop-cached states don't contribute to the backward graph.
- Rough estimate: T * (normal activation memory). With T=4 and batch_size=32, this may require reducing batch size to 16 or using more gradient accumulation steps.

---

## 12. Example Config

```python
# config/qwen3_simplestories_4k_loop_mc.py
# LoopLM + Memory Caching (Loop-Cached Reasoning)

# data
dataset = "simplestories"
batch_size = 16                    # reduced from 32 due to T loop steps in memory
max_seq_len = 512
vocab_size = 4096
vocab_source = "custom"

# model (same architecture as MC config)
dim = 288
n_layers = 10
n_heads = 4
n_kv_heads = 2
head_dim = None
multiple_of = 32
dropout = 0.0
rope_base = 10_000.0
partial_rotary_factor = 0.5
layer_types = None
linear_num_key_heads = 4
linear_num_value_heads = 4
linear_key_head_dim = 144
linear_value_head_dim = 144
linear_conv_kernel_dim = 4
attnres_block_size = 4

# Sequence-axis Memory Caching (existing)
mc_segment_size = 128
mc_ssc_top_k = 2
mc_detach_cached_states = True

# LoopLM
loop_max_steps = 4                 # 4 loop iterations
loop_kl_beta = 0.1                 # entropy regularization
loop_exit_threshold = 0.9          # inference early exit CDF threshold
loop_training_stage = 1            # Stage I: joint training

# Loop-axis Memory Caching (novel combination)
loop_mc_enabled = True
loop_mc_top_k = 2                  # top-2 prior loop iterations
loop_mc_detach = True

# optimizer (conservative for looped architecture)
learning_rate = 5e-4               # lower than non-looped MC config
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
gradient_accumulation_steps = 8    # increased to compensate for smaller batch

# lr schedule
decay_lr = True
warmup_iters = 1000               # longer warmup for stability
max_iters = 100000

# I/O
out_dir = "out_simplestories_4k_loop_mc"
eval_interval = 500
wandb_log = True
wandb_project = "llamac"
wandb_run_name = "loop_mc_4step"

compile = False
```

---

## 13. Implementation Order

### Phase 1: Basic LoopLM (no loop-MC)
1. Add ModelArgs fields for LoopLM (`loop_max_steps`, `loop_kl_beta`, `loop_exit_threshold`)
2. Add `ExitGate` class
3. Modify `Transformer.__init__` to create exit gate when looping enabled
4. Modify `Transformer.forward` with the loop logic + per-step loss + exit distribution
5. Update `train.py` with new config vars and model_args
6. **Verify**: `loop_max_steps=1` produces identical results to current code
7. **Test**: Train with `loop_max_steps=4`, `loop_mc_enabled=False` — confirm loss decreases

### Phase 2: Loop-Axis Memory Caching
8. Add ModelArgs fields for loop-MC (`loop_mc_enabled`, `loop_mc_top_k`, `loop_mc_detach`)
9. Add `loop_mc_u_proj` to LinearAttention layers
10. Add `_loop_mc_aggregate` method to LinearAttention (reuse SSC logic)
11. Extend `LinearAttention.forward` with `loop_cached_states` parameter
12. Wire up loop-axis caching in `Transformer.forward` loop
13. **Verify**: `loop_mc_enabled=False` is identical to Phase 1
14. **Test**: Train with both sequence-MC and loop-MC enabled

### Phase 3: Stage II Gate Training
15. Add `loop_training_stage` config and parameter freezing logic in `train.py`
16. Implement adaptive gate loss in `Transformer.forward`
17. **Test**: Train Stage I, then switch to Stage II and verify gate learns meaningful exit patterns

### Phase 4: Evaluation and Tuning
18. Compare perplexity: baseline vs. MC-only vs. LoopLM-only vs. Loop-Cached Reasoning
19. Log exit distributions to wandb — verify the gate assigns more loops to harder sequences
20. Tune `loop_max_steps`, `loop_kl_beta`, `loop_mc_top_k`
21. Test inference early exit with different thresholds

---

## 14. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Training instability (loss spikes) | High | Conservative LR, grad clip=1.0, start with T=3-4, KL annealing |
| OOM from T loop activations | Medium | Reduce batch_size, increase grad_accum, use `loop_mc_detach=True` |
| Exit gate collapses to always-T_max | Medium | Entropy regularization (beta=0.1), Stage II refinement |
| Loop-axis MC interferes with sequence-axis MC | Low | Separate router projections, independent top-k selections |
| Negligible benefit at this model scale | Medium | LoopLM paper tested at 1.4B+; our model is ~15M. Benefits may be smaller but the architecture validates the approach for later scaling |

---

## 15. Wandb Metrics to Log

```python
# Per training step
"loop/step_losses": list of per-loop-step losses
"loop/exit_distribution": probability mass per loop step
"loop/mean_exit_step": weighted average loop step
"loop/entropy": exit distribution entropy

# Per eval
"loop/val_loss_step_1": loss using only 1 loop iteration
"loop/val_loss_step_T": loss using all T iterations
"loop/val_loss_adaptive": loss with exit gate early stopping
```

---

## 16. What Does NOT Change

- `chunk_gated_delta_rule()` — already supports initial_state and output_final_state
- `Attention` class (full softmax) — no MC or loop awareness needed
- `FeedForward` — unchanged
- Sequence-axis MC in `LinearAttention` — continues working as before
- `export.py` / C inference — future work (would need loop unrolling)
- Data loading, tokenizer, simplestories.py
