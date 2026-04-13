# Qwen3.5 SimpleStories config with LoopLM + Memory Caching (Loop-Cached Reasoning)
# Based on qwen3_simplestories_4k_mc.py with LoopLM enabled
#
# Usage: python train.py config/qwen3_simplestories_4k_loop_mc.py
#
# Prepare data first:
#   python simplestories.py download
#   python simplestories.py train_vocab --vocab_size=4096
#   python simplestories.py pretokenize --vocab_size=4096

# data
dataset = "simplestories"
batch_size = 16                    # reduced from 32 due to T loop steps in memory
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

# Sequence-axis Memory Caching (existing)
mc_segment_size = 128              # split sequence into 128-token segments
mc_ssc_top_k = 2                  # select top-2 past segments per token
mc_detach_cached_states = True

# LoopLM
loop_max_steps = 4                 # 4 loop iterations
loop_kl_beta = 0.1                 # entropy regularization (anneals to 0.05)
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

# LoopLM loop + MC dynamic segments cause graph breaks with torch.compile
compile = False
