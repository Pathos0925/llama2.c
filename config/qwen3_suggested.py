# Qwen3.5 TinyStories config -- suggested improvements
# Usage: python train.py config/qwen3_suggested.py
#
# Changes from original:
#   - partial_rotary_factor 0.25 -> 0.5 (36 rotary dims, better positional signal)
#   - linear key/value head dim 72 -> 144 (2x model dim ratio, matching Qwen3.5-0.8B)
#   - learning_rate 5e-4 -> 1e-3 (standard for <100M models)
#   - beta2 0.95 -> 0.99 (better gradient variance tracking)
#   - warmup_iters 1000 -> 500 (shorter warmup with higher LR)

# model
dim = 288
n_layers = 8
n_heads = 4
n_kv_heads = 2
head_dim = None  # dim // n_heads = 72
multiple_of = 32
dropout = 0.0
rope_base = 10_000_000.0
partial_rotary_factor = 0.5         # was 0.25 -- more rotary dims for better position encoding
layer_types = None  # default: 3 linear + 1 full, repeating
linear_num_key_heads = 4
linear_num_value_heads = 4
linear_key_head_dim = 144           # was 72 -- linear layers need more capacity (2x head_dim)
linear_value_head_dim = 144         # was 72
linear_conv_kernel_dim = 4
attnres_block_size = 4              # block attention residuals (4 sub-layers per block = 2 layers)

# data
dataset = "tinystories"
batch_size = 64
max_seq_len = 256
vocab_size = 32000
vocab_source = "llama2"

# optimizer
learning_rate = 1e-3                # was 5e-4 -- small models tolerate higher LR
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99                        # was 0.95 -- more standard, better variance estimates
grad_clip = 1.0
gradient_accumulation_steps = 4

# lr schedule
decay_lr = True
warmup_iters = 500                  # was 1000 -- shorter warmup with higher LR
max_iters = 100000

# I/O
out_dir = "out_qwen3_suggested"
eval_interval = 2000
