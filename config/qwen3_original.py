# Qwen3.5 TinyStories config -- original defaults
# Usage: python train.py config/qwen3_original.py

# model
dim = 288
n_layers = 8
n_heads = 4
n_kv_heads = 2
head_dim = None  # dim // n_heads = 72
multiple_of = 32
dropout = 0.0
rope_base = 10_000_000.0
partial_rotary_factor = 0.25
layer_types = None  # default: 3 linear + 1 full, repeating
linear_num_key_heads = 4
linear_num_value_heads = 4
linear_key_head_dim = 72
linear_value_head_dim = 72
linear_conv_kernel_dim = 4
attnres_block_size = 0  # disabled

# data
batch_size = 64
max_seq_len = 256
vocab_size = 32000
vocab_source = "llama2"

# optimizer
learning_rate = 5e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
gradient_accumulation_steps = 4

# lr schedule
decay_lr = True
warmup_iters = 1000
max_iters = 100000

# I/O
out_dir = "out_qwen3_original"
eval_interval = 2000
