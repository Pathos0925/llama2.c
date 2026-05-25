# 95M param model with DeepSeek V4 innovations on FineWeb
# - Compressed Sparse Attention (CSA) replaces windowed attention
# - Hyper-Connections (learned residual scaling)
# - Muon optimizer (Newton-Schulz gradient orthogonalization)
# - No looping (loop_max_steps=1)
#
# Hardware: 4x NVIDIA B300 SXM6 (275 GB each)
#
# Prepare data first:
#   python fineweb.py download
#   python fineweb.py pretokenize
#
# Train (4 GPU DDP):
#   torchrun --standalone --nproc_per_node=4 train.py config/fineweb_95m_dsv4_4xB300.py

# data
dataset = "fineweb"
batch_size = 64
max_seq_len = 1024
vocab_size = 32000
vocab_source = "llama2"

# model -- 95.7M params
dim = 576
n_layers = 16
n_heads = 9
n_kv_heads = 3
head_dim = None  # dim // n_heads = 64
multiple_of = 32
dropout = 0.0
rope_base = 20_000.0
partial_rotary_factor = 0.5

# layer pattern: 2 linear + 1 CSA + 1 full, repeating (x4 = 16 layers)
layer_types = ("linear", "linear", "csa", "full",
               "linear", "linear", "csa", "full",
               "linear", "linear", "csa", "full",
               "linear", "linear", "csa", "full")

# Compressed Attention params
csa_stride = 4
csa_local_window = 64

# Hyper-connections (learned residual scaling)
hyper_connections = True

# linear attention
linear_num_key_heads = 9
linear_num_value_heads = 9
linear_key_head_dim = 112
linear_value_head_dim = 112
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
# 32 total, /4 GPUs = 8 per GPU
# tokens/iter = 8 * 4 * 64 * 1024 = 2,097,152 (~2.1M tokens/iter)
gradient_accumulation_steps = 32

# lr schedule
decay_lr = True
warmup_iters = 500
max_iters = 10000  # ~21B tokens (~2 epochs of sample-10BT)

# I/O
out_dir = "out_fineweb_95m_dsv4"
eval_interval = 2000
eval_iters = 50

# system
compile = True
num_workers = 8

# checkpointing
save_interval = 1000
r2_upload = True
r2_endpoint = "https://9b4c7692e09c1fcfe021662409d4e695.r2.cloudflarestorage.com"
r2_bucket = "jarvis"
r2_prefix = "loop-cached-reasoning-fineweb-95m/"

# wandb
wandb_log = True
wandb_project = "loop-cached-reasoning"
wandb_run_name = "95m-dsv4-fineweb-4xB300"
