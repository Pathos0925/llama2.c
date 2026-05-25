# 50M param model with DeepSeek V4 innovations on SimpleStories
# - Compressed Sparse Attention (CSA) replaces windowed attention
# - Hyper-Connections (learned residual scaling)
# - Muon optimizer (Newton-Schulz gradient orthogonalization)
# No looping (loop_max_steps=1)
#
# Prepare data first:
#   python simplestories.py download
#   python simplestories.py train_vocab --vocab_size=4096
#   python simplestories.py pretokenize --vocab_size=4096

# data
dataset = "simplestories"
batch_size = 64
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

# layer pattern: 2 linear + 1 CSA + 1 full, repeating
layer_types = ("linear", "linear", "csa", "full",
               "linear", "linear", "csa", "full",
               "linear", "linear", "csa", "full")

# Compressed Attention params
csa_stride = 4
csa_local_window = 32

# Hyper-connections (learned residual scaling)
hyper_connections = True

# linear attention
linear_num_key_heads = 8
linear_num_value_heads = 8
linear_key_head_dim = 128
linear_value_head_dim = 128
linear_conv_kernel_dim = 4

# block attention residuals -- disabled (incompatible with hyper-connections)
attnres_block_size = 0

# memory caching -- disabled
mc_segment_size = 0

# No looping
loop_max_steps = 1

# Muon optimizer
optimizer_type = "muon"
learning_rate = 0.02
muon_momentum = 0.95
muon_ns_iters = 5
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
gradient_accumulation_steps = 32

# lr schedule
decay_lr = True
warmup_iters = 500
max_iters = 17000

# I/O
out_dir = "out_simplestories_50m_dsv4"
eval_interval = 2000
eval_iters = 50

# system
compile = True
num_workers = 8

# checkpointing
save_interval = 500
r2_upload = True
r2_endpoint = "https://9b4c7692e09c1fcfe021662409d4e695.r2.cloudflarestorage.com"
r2_bucket = "jarvis"
r2_prefix = "loop-cached-reasoning-dsv4/"

# wandb
wandb_log = True
wandb_project = "loop-cached-reasoning"
wandb_run_name = "50m-dsv4-csa-hc-muon-8xH100"
