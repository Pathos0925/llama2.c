# Qwen3.5 SimpleStories config with Memory Caching (SSC)
# Based on qwen3_simplestories_4k_v3.py with MC enabled
#
# Usage: python train.py config/qwen3_simplestories_4k_mc.py
#
# Prepare data first:
#   python simplestories.py download
#   python simplestories.py train_vocab --vocab_size=4096
#   python simplestories.py pretokenize --vocab_size=4096

# data
dataset = "simplestories"
batch_size = 32
max_seq_len = 512
vocab_size = 4096
vocab_source = "custom"

# model
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

# Memory Caching
mc_segment_size = 128      # split sequence into 128-token segments (512/128 = 4 segments)
mc_ssc_top_k = 2           # select top-2 past segments per token
mc_detach_cached_states = True

# optimizer
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
gradient_accumulation_steps = 4

# lr schedule
decay_lr = True
warmup_iters = 500
max_iters = 100000

# I/O
out_dir = "out_simplestories_4k_mc"
eval_interval = 500

# MC uses dynamic segment loops that cause graph breaks with torch.compile
compile = False
