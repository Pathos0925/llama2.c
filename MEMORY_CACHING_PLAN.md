# Memory Caching (SSC) Integration Plan

## Context

The project trains small LLM models using a Qwen3.5-style hybrid architecture (3 linear attention + 1 full attention layers, repeating). The linear attention layers use Gated Delta Net, which maintains a matrix-valued state `[B, H, k_dim, v_dim]`. The paper "Memory Caching: RNNs with Growing Memory" (arXiv:2602.24281) proposes caching snapshots of this state at segment boundaries and querying them to improve recall on long sequences.

We're integrating this with:
- **Sparse Selective Caching (SSC)** aggregation strategy
- **Checkpoint** state initialization (each segment continues from previous segment's final state)
- **Constant** segmentation (fixed segment size)

## Files to Modify

- **`model.py`** -- ModelArgs, LinearAttention class (core changes)
- **`train.py`** -- config variables, model_args dict, resume forced-keys

No changes needed to `inference.py` (loads model_args from checkpoint automatically), `export.py`, or C code.

---

## 1. ModelArgs additions (`model.py`, after `attnres_block_size`)

```python
# Memory Caching (MC) -- only affects LinearAttention layers
mc_segment_size: int = 0       # 0=disabled; >0=MC segment size in tokens
mc_ssc_top_k: int = 2          # SSC: how many past segments to select per token
mc_detach_cached_states: bool = True  # detach cached states from grad graph
```

- `mc_segment_size=0` disables MC entirely -- identical code path, zero overhead.
- Should be a multiple of 64 (the internal chunk_size of the gated delta rule).

## 2. LinearAttention.__init__ -- new SSC router projection

When `mc_segment_size > 0`, add one new learnable projection:

```python
self.mc_u_proj = nn.Linear(hidden_size, key_dim, bias=False)
```

This maps hidden state → same dimension as keys so we can compute dot-product routing scores against segment contexts (sum of keys per segment). This is the `u_t = x_t W_u` from the paper (Equation 10/16).

Store `mc_segment_size`, `mc_ssc_top_k`, `mc_detach` as instance attributes.

## 3. LinearAttention.forward -- core MC logic

### When MC disabled (`segment_size==0` or `seq_len <= segment_size`)

Current code runs unchanged, zero overhead. This is the default.

### When MC enabled

All projections (QKV + conv1d, z, beta, a, g) are computed on the **full sequence first** -- this is critical because the causal conv1d must see the full sequence for continuity across segment boundaries. The new `u` projection is also computed on the full sequence.

Then:

### Step A: Segment loop

Split the sequence into MC segments. For each segment `i` (start → end):

1. Slice `query, key, value, g, beta` for positions `[start:end]`
2. Call `chunk_gated_delta_rule(..., initial_state=prev_final_state, output_final_state=True)`
   - `initial_state` is `None` for segment 0, previous segment's `final_state` for later segments
3. Get back `(online_output, final_state)`
   - `online_output`: `[B, H, seg_len, v_dim]` -- per-token outputs from the delta rule
   - `final_state`: `[B, H, k_dim, v_dim]` -- the memory matrix at segment end
4. Cache:
   - `final_state` (detached from grad graph if `mc_detach=True`)
   - `seg_context = key_seg.sum(dim=time_dim)` → `[B, H, k_dim]` (sum of L2-normed keys, per head)

### Step B: SSC aggregation (for segments `i > 0` with cached history)

For segment 0, output is just the online_output (no cached states exist).

For segment `i > 0`:

1. **Cached outputs** -- query each cached state with current segment's queries:
   ```
   cached_out_j = einsum("bhkv, bhtk -> bhtv", cached_state_j, query_seg * scale)
   ```
   where `scale = 1/sqrt(k_dim)` (matching the scale applied inside `chunk_gated_delta_rule`).
   
   Batched: stack all cached states → `[B, H, num_cached, k_dim, v_dim]`, one einsum.

2. **Router scores** -- dot product of `u_t` with each segment's context:
   ```
   cached_scores = einsum("bhtd, bhcd -> bhtc", u_seg, ctx_stack)
   ```
   → `[B, H, seg_len, num_cached]`

3. **Top-k selection** -- select top-k cached segments per token, mask rest to `-inf`

4. **Online score** -- `u_seg` dot-product with current segment's context → `[B, H, seg_len]`

5. **Softmax** over `(masked cached scores, online score)` → weights

6. **Weighted sum**:
   ```
   output = online_weight * online_output + sum(cached_weight_j * cached_out_j)
   ```

### Step C: Reassemble

Concatenate segment outputs along the sequence dimension, then continue to gated RMSNorm + `out_proj` as before (unchanged).

## 4. train.py additions

**Config variables** (after `attnres_block_size = 0`):
```python
mc_segment_size = 0
mc_ssc_top_k = 2
mc_detach_cached_states = True
```

**model_args dict**: add all three fields.

**Resume forced-keys list**: add all three to ensure architecture consistency when resuming.

## 5. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Aggregation | SSC | User preference; efficient via top-k sparsity |
| State init | Checkpoint | Each segment continues from previous; paper default |
| Segmentation | Constant | Fixed segment size, simple and predictable |
| Segment context | Sum of keys | Paper's MeanPooling; sum is equivalent for routing since softmax normalizes |
| Query scale for cached states | `1/sqrt(k_dim)` | Match the scale applied inside `chunk_gated_delta_rule` |
| Detach cached states | True (default) | Prevents memory blowup; gradients only flow through current segment |
| Conv1d scope | Full sequence | Run causal conv1d on full seq BEFORE segmenting, preserving cross-segment continuity |
| Online context per segment | Full segment context | Use segment total context (not per-token cumulative) for efficient batched computation |

## 6. What Does NOT Change

- `chunk_gated_delta_rule()` function -- already supports `initial_state` and `output_final_state` params
- `Attention` class (full softmax attention) -- MC only applies to linear attention layers
- `TransformerBlock`, `Transformer` -- no changes needed
- `inference.py` -- loads model_args from checkpoint dict, works automatically
- `export.py` / C inference code -- future work

## 7. Memory & Compute Overhead

**Parameters**: One `mc_u_proj` per linear attention layer: `hidden_size * key_dim` params each.
For default TinyStories config (dim=288, key_dim=288): ~83K params per layer, ~498K total for 6 linear layers. Negligible vs model size.

**Cached states during forward pass**: `num_segments * B * H * k_dim * v_dim` floats.
Example: `mc_segment_size=128`, `seq_len=512` → 4 segments → 4 matrices of `[B, H, 72, 72]`.
With B=32, H=4: ~9.5 MB. Manageable.

**Compute**: One extra linear projection + per-segment cached-state queries + routing. Minor overhead for small models.

## 8. Verification Plan

1. **MC disabled (default)**: `mc_segment_size=0` should produce bit-identical results to current code.
2. **Single segment**: `mc_segment_size >= seq_len` → one segment, no cached states, output should match non-MC path.
3. **Training test**: Config with `mc_segment_size=128` on SimpleStories, verify loss decreases normally.
4. **Param count**: Check that MC adds only the expected `mc_u_proj` parameters.

## 9. Example Config

```python
# config/qwen3_simplestories_4k_mc.py
# Memory Caching variant of the SimpleStories 4K config

# ... (same base config as qwen3_simplestories_4k_v3.py) ...

# Memory Caching
mc_segment_size = 128      # split sequence into 128-token segments
mc_ssc_top_k = 2           # select top-2 past segments per token
mc_detach_cached_states = True
```
