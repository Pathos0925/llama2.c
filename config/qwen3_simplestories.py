# Qwen3.5 SimpleStories config
# Usage: python train.py config/qwen3_simplestories.py
#
# Uses the SimpleStories dataset with tuned hyperparameters.
# Prepare data first:
#   python simplestories.py download
#   python simplestories.py pretokenize

# data
dataset = "simplestories"
batch_size = 64
max_seq_len = 512               # SimpleStories reference uses 512
vocab_size = 32000
vocab_source = "llama2"

# model
dim = 288
n_layers = 8
n_heads = 4
n_kv_heads = 2
head_dim = None
multiple_of = 32
dropout = 0.0
rope_base = 10_000_000.0
partial_rotary_factor = 0.5
layer_types = None
linear_num_key_heads = 4
linear_num_value_heads = 4
linear_key_head_dim = 144
linear_value_head_dim = 144
linear_conv_kernel_dim = 4
attnres_block_size = 4

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
out_dir = "out_simplestories"
eval_interval = 2000
