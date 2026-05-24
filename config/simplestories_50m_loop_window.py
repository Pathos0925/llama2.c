# 50M param LoopLM + Windowed Attention on SimpleStories
# Usage: python train.py config/simplestories_50m_loop_window.py
#
# Architecture: 12 physical layers (linear, linear, window, full) x3
# With loop_max_steps=4, effective depth = 48 layers
# No memory caching (mc_segment_size=0, loop_mc_enabled=False)
#
# Prepare data first:
#   python simplestories.py download
#   python simplestories.py train_vocab --vocab_size=4096
#   python simplestories.py pretokenize --vocab_size=4096

# data
dataset = "simplestories"
batch_size = 24
max_seq_len = 1024
vocab_size = 4096
vocab_source = "custom"

# model
dim = 512
n_layers = 12
n_heads = 8
n_kv_heads = 4
head_dim = None  # dim // n_heads = 64
multiple_of = 32
dropout = 0.0
rope_base = 10_000.0
partial_rotary_factor = 0.5

# layer pattern: 2 linear + 1 window + 1 full, repeating
layer_types = ("linear", "linear", "window", "full",
               "linear", "linear", "window", "full",
               "linear", "linear", "window", "full")
window_size = 128

# linear attention
linear_num_key_heads = 8
linear_num_value_heads = 8
linear_key_head_dim = 128
linear_value_head_dim = 128
linear_conv_kernel_dim = 4

# block attention residuals
attnres_block_size = 4

# memory caching -- disabled
mc_segment_size = 0

# LoopLM -- 4 loop iterations, no caching
loop_max_steps = 2
loop_kl_beta = 0.1
loop_exit_threshold = 0.9
loop_training_stage = 1
loop_mc_enabled = False

# optimizer
learning_rate = 1e-3
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
gradient_accumulation_steps = 6

# lr schedule
decay_lr = True
warmup_iters = 500
max_iters = 100000

# I/O
out_dir = "out_simplestories_50m_loop_window"
eval_interval = 2000
eval_iters = 50

# system
compile = True
num_workers = 4

# wandb
wandb_log = False
wandb_project = "loop-cached-reasoning"
wandb_run_name = "50m-loop4-window128-simplestories"
