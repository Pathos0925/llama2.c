import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None  # explicit head dim; if None, uses dim // n_heads
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-6
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_base: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    # layer type pattern: "full" or "linear", repeating. None = all full attention.
    layer_types: Optional[Tuple[str, ...]] = None
    # linear attention (Gated Delta Net) params
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # Block Attention Residuals: 0 = disabled (standard residuals)
    # block_size counts sub-layers (attn+mlp = 2 per transformer layer)
    attnres_block_size: int = 0
    # Memory Caching (MC) -- only affects LinearAttention layers
    # 0 = disabled; >0 = segment size in tokens for memory caching
    mc_segment_size: int = 0
    mc_ssc_top_k: int = 2          # SSC: how many past segments to select per token
    mc_detach_cached_states: bool = True  # detach cached states from computation graph


class RMSNorm(torch.nn.Module):
    """Qwen3.5-style RMSNorm: zero-init weight, forward is x_norm * (1 + weight)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * (1.0 + self.weight)


class RMSNormGated(torch.nn.Module):
    """Gated RMSNorm for linear attention layers: norm(x) * weight * SiLU(gate)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, gate):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.to(torch.float32))
        return x.to(input_dtype)


class BlockAttnRes(nn.Module):
    """Block Attention Residual: softmax attention over block-level representations."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.proj = nn.Linear(dim, 1, bias=False)
        self.norm = RMSNorm(dim, eps=eps)

    def forward(self, blocks: list, partial_block: torch.Tensor) -> torch.Tensor:
        # blocks: list of [B, T, D], partial_block: [B, T, D]
        V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
        K = self.norm(V)
        logits = torch.einsum('d, n b t d -> n b t', self.proj.weight.squeeze(), K)
        weights = torch.softmax(logits, dim=0)
        h = torch.einsum('n b t, n b t d -> b t d', weights, V)
        return h


def precompute_freqs_cis(head_dim: int, end: int, theta: float = 10_000_000.0, partial_rotary_factor: float = 0.25):
    """Precompute cos/sin for partial RoPE. Only rotary_dim dimensions are rotated."""
    rotary_dim = int(head_dim * partial_rotary_factor)
    rotary_dim = max(2, rotary_dim - (rotary_dim % 2))  # ensure even
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(end)
    freqs = torch.outer(t, inv_freq).float()
    # duplicate for paired rotation: [cos(f0), cos(f1), ..., cos(f0), cos(f1), ...]
    freqs = torch.cat([freqs, freqs], dim=-1)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply partial RoPE. freqs have rot_dim columns; remaining dims pass through."""
    # xq, xk: (bs, n_heads, seqlen, head_dim) -- already transposed
    seq_len = xq.shape[2]
    head_dim = xq.shape[-1]
    rot_dim = freqs_cos.shape[-1]

    cos = freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, rot_dim)
    sin = freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)

    def _rotate(x):
        x_rot = x[..., :rot_dim].float()
        x_pass = x[..., rot_dim:]
        half = rot_dim // 2
        x1, x2 = x_rot[..., :half], x_rot[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        x_rot = (x_rot * cos + rotated * sin).type_as(x)
        return torch.cat([x_rot, x_pass], dim=-1)

    return _rotate(xq), _rotate(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    """Qwen3.5-style full attention with gated Q projection and QK norm."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim if args.head_dim is not None else args.dim // args.n_heads
        self.d_out = self.n_heads * self.head_dim

        # Gated Q: projects to 2x for (query, gate) split
        self.wq = nn.Linear(args.dim, self.d_out * 2, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.d_out, args.dim, bias=False)

        # QK norm (per-head RMSNorm, applied before RoPE)
        self.q_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.norm_eps)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # Gated Q projection: split into queries and gate
        q_and_gate = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim * 2)
        xq, gate = torch.chunk(q_and_gate, 2, dim=-1)  # each (bsz, seqlen, n_heads, head_dim)
        gate = gate.reshape(bsz, seqlen, self.d_out)  # flatten heads for later gating

        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # QK norm before RoPE (operates on last dim = head_dim, broadcasts over heads)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # transpose to (bs, heads, seqlen, head_dim) for RoPE and attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # RoPE (partial rotary)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # GQA: expand KV heads
        xk = repeat_kv(xk.transpose(1, 2), self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv.transpose(1, 2), self.n_rep).transpose(1, 2)

        # attention
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # concat heads -> (bsz, seqlen, d_out)
        context = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_out)

        # apply sigmoid gate from the Q projection
        context = context * torch.sigmoid(gate)

        output = self.wo(context)
        output = self.resid_dropout(output)
        return output


def l2norm(x, dim=-1, eps=1e-6):
    """L2 normalize along dim."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_causal_conv1d(hidden_states, weight, bias=None):
    """Naive causal 1D depthwise conv with SiLU activation.
    hidden_states: (batch, channels, seq_len)
    weight: (channels, 1, kernel_size) -- depthwise
    """
    channels, _, kernel_size = weight.shape
    padding = kernel_size - 1
    out = F.conv1d(hidden_states, weight, bias, padding=padding, groups=channels)
    out = F.silu(out[..., :hidden_states.shape[-1]])
    return out


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                           initial_state=None, output_final_state=False):
    """Chunked gated delta rule -- naive PyTorch implementation.
    All inputs: (batch, heads, seq_len, dim) except g, beta: (batch, heads, seq_len).
    L2 norm on Q, K is applied by caller.
    """
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_len = seq_len + pad_size
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = (
        torch.zeros(batch_size, num_heads, k_dim, v_dim, device=value.device, dtype=value.dtype)
        if initial_state is None else initial_state.to(value)
    )
    core_out = torch.zeros_like(value)
    mask2 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask2, 0)
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        core_out[:, :, i] = attn_inter + attn_i @ v_new
        state = (
            state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    final_state = state if output_final_state else None
    core_out = core_out.reshape(core_out.shape[0], core_out.shape[1], -1, core_out.shape[-1])
    core_out = core_out[:, :, :seq_len]
    return core_out.transpose(1, 2).contiguous().to(query.dtype), final_state


class LinearAttention(nn.Module):
    """Qwen3.5 Gated Delta Net linear attention layer, with optional Memory Caching (MC)."""
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.hidden_size = args.dim
        self.num_k_heads = args.linear_num_key_heads
        self.num_v_heads = args.linear_num_value_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.layer_idx = layer_idx

        # Combined QKV projection
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)

        # Depthwise causal conv on QKV
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
            kernel_size=self.conv_kernel_size, groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Gate projections for delta rule
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        # Learnable time-step and decay parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Gated RMSNorm after attention
        self.norm = RMSNormGated(self.head_v_dim, eps=args.norm_eps)

        # Output projection
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # Memory Caching (MC) -- SSC aggregation over cached segment states
        self.mc_segment_size = args.mc_segment_size
        if self.mc_segment_size > 0:
            self.mc_ssc_top_k = args.mc_ssc_top_k
            self.mc_detach = args.mc_detach_cached_states
            # After head expansion, keys have num_v_heads heads.  Router u_t must
            # match: project hidden_size -> num_v_heads * head_k_dim, then reshape
            # to [B, T, num_v_heads, head_k_dim] for per-head routing scores.
            self.mc_u_proj = nn.Linear(self.hidden_size, self.num_v_heads * self.head_k_dim, bias=False)

    def _mc_ssc_aggregate(self, online_out, query_seg, u_seg, seg_context,
                          cached_states, cached_contexts):
        """Sparse Selective Caching aggregation for one MC segment.

        Args:
            online_out: [B, T_seg, num_v_heads, head_v_dim] -- output from chunk_gated_delta_rule
            query_seg:  [B, T_seg, num_v_heads, head_k_dim] -- L2-normed queries for this segment
            u_seg:      [B, T_seg, num_v_heads, head_k_dim] -- router vectors for this segment
            seg_context:[B, num_v_heads, head_k_dim] -- sum of keys in this segment
            cached_states:   list of [B, num_v_heads, head_k_dim, head_v_dim] tensors
            cached_contexts: list of [B, num_v_heads, head_k_dim] tensors
        Returns:
            aggregated output: [B, T_seg, num_v_heads, head_v_dim]
        """
        num_cached = len(cached_states)
        if num_cached == 0:
            return online_out

        bsz, seg_len, num_heads, head_k = query_seg.shape
        scale = 1.0 / (head_k ** 0.5)

        # Compute cached outputs: query each cached state with all tokens in segment
        # Stack cached states: [B, num_cached, H, k, v]
        all_states = torch.stack(cached_states, dim=1)
        # query_seg scaled: [B, T, H, k] -> [B, H, T, k]
        q_scaled = query_seg.transpose(1, 2) * scale
        # all_states: [B, num_cached, H, k, v] -> [B, H, num_cached, k, v]
        all_states = all_states.transpose(1, 2)
        # Batched einsum: [B, H, T, k] x [B, H, num_cached, k, v] -> [B, H, num_cached, T, v]
        cached_outs = torch.einsum("bhtk,bhckv->bhctv", q_scaled, all_states)

        # Router scores for cached segments
        # u_seg: [B, T, H, k] -> [B, H, T, k]
        u_seg_t = u_seg.transpose(1, 2)
        # cached_contexts stacked: [B, num_cached, H, k] -> [B, H, num_cached, k]
        ctx_stack = torch.stack(cached_contexts, dim=1).transpose(1, 2)
        # scores: [B, H, T, num_cached]
        cached_scores = torch.einsum("bhtk,bhck->bhtc", u_seg_t, ctx_stack)

        # Top-k selection
        k = min(self.mc_ssc_top_k, num_cached)
        topk_indices = torch.topk(cached_scores, k=k, dim=-1).indices
        mask = torch.zeros_like(cached_scores, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        masked_scores = cached_scores.masked_fill(~mask, float("-inf"))

        # Online score: u_seg dot current segment context
        # seg_context: [B, H, k] -- broadcast across T
        online_scores = torch.einsum("bhtk,bhk->bht", u_seg_t, seg_context)

        # Softmax over (masked cached scores, online score)
        # all_scores: [B, H, T, num_cached + 1]
        all_scores = torch.cat([masked_scores, online_scores.unsqueeze(-1)], dim=-1)
        all_weights = torch.softmax(all_scores, dim=-1)

        cached_weights = all_weights[..., :-1]   # [B, H, T, num_cached]
        online_weight = all_weights[..., -1:]     # [B, H, T, 1]

        # Weighted combination
        # online_out: [B, T, H, v] -> [B, H, T, v]
        online_t = online_out.transpose(1, 2)
        # weighted online: [B, H, T, v]
        result = online_weight * online_t
        # weighted cached: einsum [B, H, T, num_cached] x [B, H, num_cached, T, v] -> [B, H, T, v]
        result = result + torch.einsum("bhtc,bhctv->bhtv", cached_weights, cached_outs)

        # Back to [B, T, H, v]
        return result.transpose(1, 2)

    def forward(self, x: torch.Tensor):
        bsz, seq_len, _ = x.shape

        # QKV projection + causal conv (full sequence for cross-segment continuity)
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)  # (bsz, conv_dim, seq_len)
        mixed_qkv = torch_causal_conv1d(mixed_qkv, self.conv1d.weight, self.conv1d.bias)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # (bsz, seq_len, conv_dim)

        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(bsz, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)

        # Gate projections
        z = self.in_proj_z(x).reshape(bsz, seq_len, self.num_v_heads, self.head_v_dim)
        beta = self.in_proj_b(x).sigmoid()  # (bsz, seq_len, num_v_heads)
        a = self.in_proj_a(x)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # Expand K heads to match V heads if needed
        if self.num_v_heads // self.num_k_heads > 1:
            rep = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        # L2 norm Q, K before delta rule
        query = l2norm(query, dim=-1)
        key = l2norm(key, dim=-1)

        mc_enabled = self.mc_segment_size > 0 and seq_len > self.mc_segment_size

        if not mc_enabled:
            # Standard path: no memory caching
            core_out, _ = chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=None, output_final_state=False,
            )
        else:
            # Memory Caching path: segment, cache states, SSC aggregate
            mc_seg = self.mc_segment_size
            # Router projection on full sequence, reshaped to per-head
            u = self.mc_u_proj(x).reshape(bsz, seq_len, self.num_v_heads, self.head_k_dim)

            # Build segment spans (last segment may be shorter)
            spans = []
            pos = 0
            while pos < seq_len:
                end = min(pos + mc_seg, seq_len)
                spans.append((pos, end))
                pos = end

            cached_states = []    # list of [B, num_v_heads, head_k_dim, head_v_dim]
            cached_contexts = []  # list of [B, num_v_heads, head_k_dim]
            prev_state = None
            segment_outputs = []

            for start, end in spans:
                q_seg = query[:, start:end]
                k_seg = key[:, start:end]
                v_seg = value[:, start:end]
                g_seg = g[:, start:end]
                beta_seg = beta[:, start:end]

                # Run gated delta rule on this segment
                online_out, final_state = chunk_gated_delta_rule(
                    q_seg, k_seg, v_seg, g=g_seg, beta=beta_seg,
                    initial_state=prev_state, output_final_state=True,
                )
                # online_out: [B, seg_len, num_v_heads, head_v_dim]
                # final_state: [B, num_v_heads, head_k_dim, head_v_dim]

                # Segment context: sum of L2-normed keys per head
                # k_seg: [B, seg_len, H, k] -> [B, H, k] via sum over time dim
                seg_ctx = k_seg.sum(dim=1)  # [B, num_v_heads, head_k_dim]

                # SSC aggregation
                u_seg = u[:, start:end]
                online_out = self._mc_ssc_aggregate(
                    online_out, q_seg, u_seg, seg_ctx,
                    cached_states, cached_contexts,
                )

                segment_outputs.append(online_out)

                # Cache this segment's state and context for future segments
                state_to_cache = final_state.detach().clone() if self.mc_detach else final_state.clone()
                cached_states.append(state_to_cache)
                cached_contexts.append(seg_ctx.detach().clone() if self.mc_detach else seg_ctx.clone())

                # Checkpoint: next segment starts from this segment's final state
                prev_state = final_state

            core_out = torch.cat(segment_outputs, dim=1)

        # Gated RMSNorm: norm(attn_out) * SiLU(z)
        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_out = self.norm(core_out, z)
        core_out = core_out.reshape(bsz, seq_len, -1)

        return self.out_proj(core_out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, layer_type: str = "full"):
        super().__init__()
        self.layer_type = layer_type
        self.layer_id = layer_id
        self.attnres_block_size = args.attnres_block_size

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)

        if layer_type == "full":
            self.attention = Attention(args)
        else:
            self.attention = LinearAttention(args, layer_idx=layer_id)

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        if args.attnres_block_size > 0:
            self.attn_res = BlockAttnRes(args.dim, eps=args.norm_eps)
            self.mlp_res = BlockAttnRes(args.dim, eps=args.norm_eps)

    def forward(self, blocks, partial_block, freqs_cos, freqs_sin):
        if self.attnres_block_size > 0:
            # Block boundary: snapshot partial_block and start fresh
            sublayer_idx = self.layer_id * 2
            if sublayer_idx > 0 and sublayer_idx % self.attnres_block_size == 0:
                blocks = blocks + [partial_block]
                # Attend over completed blocks to produce input for attention
                # (partial_block is the last completed block, already in blocks)
                h = self.attn_res(blocks, blocks[-1])
            else:
                # Before attention: attend over blocks + current partial
                h = self.attn_res(blocks, partial_block)

            # Attention sub-layer
            if self.layer_type == "full":
                attn_out = self.attention(self.attention_norm(h), freqs_cos, freqs_sin)
            else:
                attn_out = self.attention(self.attention_norm(h))

            # Start new partial_block at boundary, or accumulate
            if sublayer_idx > 0 and sublayer_idx % self.attnres_block_size == 0:
                partial_block = attn_out
            else:
                partial_block = partial_block + attn_out

            # Before MLP: attend over depth again
            h = self.mlp_res(blocks, partial_block)

            # MLP sub-layer
            mlp_out = self.feed_forward(self.ffn_norm(h))
            partial_block = partial_block + mlp_out

            return blocks, partial_block
        else:
            # Standard residual path (attnres disabled)
            if self.layer_type == "full":
                h = partial_block + self.attention(self.attention_norm(partial_block), freqs_cos, freqs_sin)
            else:
                h = partial_block + self.attention(self.attention_norm(partial_block))
            out = h + self.feed_forward(self.ffn_norm(h))
            return blocks, out


class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # Resolve head_dim
        head_dim = params.head_dim if params.head_dim is not None else params.dim // params.n_heads

        # Build layer type pattern: default is 3 linear + 1 full, repeating
        if params.layer_types is not None:
            layer_types = list(params.layer_types)
        else:
            pattern = ("linear", "linear", "linear", "full")
            layer_types = [pattern[i % len(pattern)] for i in range(params.n_layers)]

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, layer_type=layer_types[layer_id]))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # precompute RoPE with partial rotary and configurable theta
        freqs_cos, freqs_sin = precompute_freqs_cis(
            head_dim, params.max_seq_len,
            theta=params.rope_base, partial_rotary_factor=params.partial_rotary_factor)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight') or pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Token embedding is the first "block" and the initial partial sum
        blocks = [h]
        partial_block = h

        for layer in self.layers:
            blocks, partial_block = layer(blocks, partial_block, freqs_cos, freqs_sin)
        h = self.norm(partial_block)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
