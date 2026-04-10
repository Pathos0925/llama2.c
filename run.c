/* Inference for Qwen3.5 hybrid Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    // Qwen3.5 additions
    int head_dim;
    float rope_base;
    float partial_rotary_factor;
    float norm_eps;
    // linear attention params
    int linear_num_key_heads;
    int linear_num_value_heads;
    int linear_key_head_dim;
    int linear_value_head_dim;
    int linear_conv_kernel_dim;
    int attnres_block_size;  // 0 = disabled
    // flags
    int shared_weights;
    unsigned char* layer_types; // 0=linear, 1=full, length n_layers
    int n_full_layers;
    int n_linear_layers;
} Config;

typedef struct {
    float* token_embedding_table; // (vocab_size, dim)
    // Per-layer pointers (arrays of n_layers pointers)
    float** rms_att_weight;       // [layer] -> (dim,)
    // Full attention layer weights (NULL for linear layers)
    float** wq;                   // [layer] -> (dim, n_heads * head_dim * 2) gated Q
    float** wk;                   // [layer] -> (dim, n_kv_heads * head_dim)
    float** wv;                   // [layer] -> (dim, n_kv_heads * head_dim)
    float** wo;                   // [layer] -> (n_heads * head_dim, dim)
    float** q_norm_weight;        // [layer] -> (head_dim,)
    float** k_norm_weight;        // [layer] -> (head_dim,)
    // Linear attention layer weights (NULL for full layers)
    float** in_proj_qkv;          // [layer] -> (dim, conv_dim)
    float** in_proj_z;            // [layer] -> (dim, value_dim)
    float** in_proj_b;            // [layer] -> (dim, num_v_heads)
    float** in_proj_a;            // [layer] -> (dim, num_v_heads)
    float** conv_weight;          // [layer] -> (conv_dim, 1, kernel_size)
    float** dt_bias;              // [layer] -> (num_v_heads,)
    float** A_log;                // [layer] -> (num_v_heads,)
    float** gated_norm_weight;    // [layer] -> (value_head_dim,)
    float** out_proj;             // [layer] -> (value_dim, dim)
    // Block AttnRes weights (NULL if disabled)
    float** attn_res_proj;    // [layer] -> (dim,)
    float** attn_res_norm;    // [layer] -> (dim,)
    float** mlp_res_proj;     // [layer] -> (dim,)
    float** mlp_res_norm;     // [layer] -> (dim,)
    // FFN weights (all layers)
    float** rms_ffn_weight;       // [layer] -> (dim,)
    float** w1;                   // [layer] -> (hidden_dim, dim)
    float** w2;                   // [layer] -> (dim, hidden_dim)
    float** w3;                   // [layer] -> (hidden_dim, dim)
    // Final
    float* rms_final_weight;      // (dim,)
    float* wcls;                  // (vocab_size, dim)
} TransformerWeights;

typedef struct {
    // shared activation buffers
    float *x;       // (dim,)
    float *xb;      // (dim,)
    float *xb2;     // (dim,)
    float *hb;      // (hidden_dim,)
    float *hb2;     // (hidden_dim,)
    float *logits;  // (vocab_size,)
    // full attention buffers
    float *q;       // (n_heads * head_dim * 2) raw gated Q output
    float *k;       // (n_kv_heads * head_dim)
    float *v;       // (n_kv_heads * head_dim)
    float *att;     // (n_heads, seq_len)
    float *key_cache;   // (n_full_layers, seq_len, n_kv_heads * head_dim)
    float *value_cache; // (n_full_layers, seq_len, n_kv_heads * head_dim)
    // linear attention buffers
    float *lin_qkv;     // (conv_dim) combined QKV
    float *lin_z;       // (value_dim) z gate
    float *lin_b;       // (num_v_heads) beta raw
    float *lin_a;       // (num_v_heads) alpha raw
    float *lin_out;     // (value_dim) output before out_proj
    // linear attention persistent state
    float **conv_state;       // [lin_layer_idx] -> (conv_dim, kernel_size - 1)
    float **recurrent_state;  // [lin_layer_idx] -> (num_v_heads, key_head_dim, value_head_dim)
    // block attnres state (single-token inference: each block rep is dim floats)
    float **block_reps;       // array of block representation vectors (dim,)
    int n_blocks;             // current number of completed blocks
    int max_blocks;           // max blocks allocated
    float *partial_block;     // (dim,) current partial block sum
    float *attnres_out;       // (dim,) scratch for attnres output
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

// allocate per-layer pointer arrays
static void alloc_layer_pointers(TransformerWeights *w, int n_layers) {
    w->rms_att_weight = calloc(n_layers, sizeof(float*));
    w->wq = calloc(n_layers, sizeof(float*));
    w->wk = calloc(n_layers, sizeof(float*));
    w->wv = calloc(n_layers, sizeof(float*));
    w->wo = calloc(n_layers, sizeof(float*));
    w->q_norm_weight = calloc(n_layers, sizeof(float*));
    w->k_norm_weight = calloc(n_layers, sizeof(float*));
    w->in_proj_qkv = calloc(n_layers, sizeof(float*));
    w->in_proj_z = calloc(n_layers, sizeof(float*));
    w->in_proj_b = calloc(n_layers, sizeof(float*));
    w->in_proj_a = calloc(n_layers, sizeof(float*));
    w->conv_weight = calloc(n_layers, sizeof(float*));
    w->dt_bias = calloc(n_layers, sizeof(float*));
    w->A_log = calloc(n_layers, sizeof(float*));
    w->gated_norm_weight = calloc(n_layers, sizeof(float*));
    w->out_proj = calloc(n_layers, sizeof(float*));
    w->attn_res_proj = calloc(n_layers, sizeof(float*));
    w->attn_res_norm = calloc(n_layers, sizeof(float*));
    w->mlp_res_proj = calloc(n_layers, sizeof(float*));
    w->mlp_res_norm = calloc(n_layers, sizeof(float*));
    w->rms_ffn_weight = calloc(n_layers, sizeof(float*));
    w->w1 = calloc(n_layers, sizeof(float*));
    w->w2 = calloc(n_layers, sizeof(float*));
    w->w3 = calloc(n_layers, sizeof(float*));
}

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = p->n_kv_heads * p->head_dim;
    int key_dim = p->linear_num_key_heads * p->linear_key_head_dim;
    int value_dim = p->linear_num_value_heads * p->linear_value_head_dim;
    int conv_dim = key_dim * 2 + value_dim;

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->n_heads * p->head_dim * 2, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));

    // KV cache only for full attention layers
    if (p->n_full_layers > 0) {
        s->key_cache = calloc((long long)p->n_full_layers * p->seq_len * kv_dim, sizeof(float));
        s->value_cache = calloc((long long)p->n_full_layers * p->seq_len * kv_dim, sizeof(float));
    } else {
        s->key_cache = NULL;
        s->value_cache = NULL;
    }

    // linear attention buffers
    if (p->n_linear_layers > 0) {
        s->lin_qkv = calloc(conv_dim, sizeof(float));
        s->lin_z = calloc(value_dim, sizeof(float));
        s->lin_b = calloc(p->linear_num_value_heads, sizeof(float));
        s->lin_a = calloc(p->linear_num_value_heads, sizeof(float));
        s->lin_out = calloc(value_dim, sizeof(float));

        s->conv_state = calloc(p->n_linear_layers, sizeof(float*));
        s->recurrent_state = calloc(p->n_linear_layers, sizeof(float*));
        int conv_state_size = conv_dim * (p->linear_conv_kernel_dim - 1);
        int recurrent_state_size = p->linear_num_value_heads * p->linear_key_head_dim * p->linear_value_head_dim;
        for (int i = 0; i < p->n_linear_layers; i++) {
            s->conv_state[i] = calloc(conv_state_size, sizeof(float));
            s->recurrent_state[i] = calloc(recurrent_state_size, sizeof(float));
        }
    } else {
        s->lin_qkv = NULL; s->lin_z = NULL; s->lin_b = NULL;
        s->lin_a = NULL; s->lin_out = NULL;
        s->conv_state = NULL; s->recurrent_state = NULL;
    }

    // Block AttnRes state
    if (p->attnres_block_size > 0) {
        // max blocks = n_layers * 2 / block_size + 2 (generous upper bound)
        s->max_blocks = (p->n_layers * 2) / p->attnres_block_size + 2;
        s->block_reps = calloc(s->max_blocks, sizeof(float*));
        for (int i = 0; i < s->max_blocks; i++)
            s->block_reps[i] = calloc(p->dim, sizeof(float));
        s->partial_block = calloc(p->dim, sizeof(float));
        s->attnres_out = calloc(p->dim, sizeof(float));
        s->n_blocks = 0;
    } else {
        s->block_reps = NULL; s->partial_block = NULL;
        s->attnres_out = NULL; s->max_blocks = 0; s->n_blocks = 0;
    }

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s, Config* p) {
    free(s->x); free(s->xb); free(s->xb2);
    free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
    free(s->lin_qkv); free(s->lin_z); free(s->lin_b);
    free(s->lin_a); free(s->lin_out);
    if (s->conv_state) {
        for (int i = 0; i < p->n_linear_layers; i++) {
            free(s->conv_state[i]);
            free(s->recurrent_state[i]);
        }
        free(s->conv_state);
        free(s->recurrent_state);
    }
    if (s->block_reps) {
        for (int i = 0; i < s->max_blocks; i++) free(s->block_reps[i]);
        free(s->block_reps);
    }
    free(s->partial_block); free(s->attnres_out);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr) {
    int head_dim = p->head_dim;
    int key_dim = p->linear_num_key_heads * p->linear_key_head_dim;
    int value_dim = p->linear_num_value_heads * p->linear_value_head_dim;
    int conv_dim = key_dim * 2 + value_dim;

    // token embeddings
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;

    // per-layer weights, matching version3_export order
    for (int l = 0; l < p->n_layers; l++) {
        // attention norm
        w->rms_att_weight[l] = ptr; ptr += p->dim;

        if (p->layer_types[l] == 1) { // full attention
            w->wq[l] = ptr; ptr += p->dim * (p->n_heads * head_dim * 2);
            w->wk[l] = ptr; ptr += p->dim * (p->n_kv_heads * head_dim);
            w->wv[l] = ptr; ptr += p->dim * (p->n_kv_heads * head_dim);
            w->wo[l] = ptr; ptr += (p->n_heads * head_dim) * p->dim;
            w->q_norm_weight[l] = ptr; ptr += head_dim;
            w->k_norm_weight[l] = ptr; ptr += head_dim;
        } else { // linear attention
            w->in_proj_qkv[l] = ptr;      ptr += conv_dim * p->dim;
            w->in_proj_z[l] = ptr;         ptr += value_dim * p->dim;
            w->in_proj_b[l] = ptr;         ptr += p->linear_num_value_heads * p->dim;
            w->in_proj_a[l] = ptr;         ptr += p->linear_num_value_heads * p->dim;
            w->conv_weight[l] = ptr;       ptr += conv_dim * 1 * p->linear_conv_kernel_dim;
            w->dt_bias[l] = ptr;           ptr += p->linear_num_value_heads;
            w->A_log[l] = ptr;             ptr += p->linear_num_value_heads;
            w->gated_norm_weight[l] = ptr; ptr += p->linear_value_head_dim;
            w->out_proj[l] = ptr;          ptr += value_dim * p->dim;
        }

        // AttnRes weights (if enabled)
        if (p->attnres_block_size > 0) {
            w->attn_res_proj[l] = ptr; ptr += p->dim;
            w->attn_res_norm[l] = ptr; ptr += p->dim;
            w->mlp_res_proj[l] = ptr;  ptr += p->dim;
            w->mlp_res_norm[l] = ptr;  ptr += p->dim;
        }

        // FFN
        w->rms_ffn_weight[l] = ptr; ptr += p->dim;
        w->w1[l] = ptr; ptr += p->hidden_dim * p->dim;
        w->w2[l] = ptr; ptr += p->dim * p->hidden_dim;
        w->w3[l] = ptr; ptr += p->hidden_dim * p->dim;
    }

    // final norm + classifier
    w->rms_final_weight = ptr; ptr += p->dim;
    w->wcls = p->shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }

    // read magic and version
    unsigned int magic;
    int version;
    if (fread(&magic, sizeof(unsigned int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&version, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);

    if (magic == 0x616b3432 && version == 3) {
        // v3 Qwen3.5 format
        if (fread(&config->dim, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&config->hidden_dim, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&config->n_layers, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&config->n_heads, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&config->n_kv_heads, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&config->vocab_size, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&config->seq_len, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
        // Qwen3.5 params
        if (fread(&config->head_dim, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&config->rope_base, sizeof(float), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&config->partial_rotary_factor, sizeof(float), 1, file) != 1) exit(EXIT_FAILURE);
        if (fread(&config->norm_eps, sizeof(float), 1, file) != 1) exit(EXIT_FAILURE);
        // linear attention params
        int linear_params[6];
        if (fread(linear_params, sizeof(int), 6, file) != 6) exit(EXIT_FAILURE);
        config->linear_num_key_heads = linear_params[0];
        config->linear_num_value_heads = linear_params[1];
        config->linear_key_head_dim = linear_params[2];
        config->linear_value_head_dim = linear_params[3];
        config->linear_conv_kernel_dim = linear_params[4];
        config->attnres_block_size = linear_params[5];
        // shared classifier flag
        unsigned char shared_flag;
        if (fread(&shared_flag, sizeof(unsigned char), 1, file) != 1) exit(EXIT_FAILURE);
        config->shared_weights = (int)shared_flag;
        // layer types
        config->layer_types = malloc(config->n_layers);
        if (fread(config->layer_types, 1, config->n_layers, file) != (size_t)config->n_layers) exit(EXIT_FAILURE);
    } else {
        // try legacy v0 format (no magic, raw Config struct)
        fseek(file, 0, SEEK_SET);
        int header[7];
        if (fread(header, sizeof(int), 7, file) != 7) exit(EXIT_FAILURE);
        config->dim = header[0]; config->hidden_dim = header[1];
        config->n_layers = header[2]; config->n_heads = header[3];
        config->n_kv_heads = header[4]; config->vocab_size = header[5];
        config->seq_len = header[6];
        config->shared_weights = config->vocab_size > 0 ? 1 : 0;
        config->vocab_size = abs(config->vocab_size);
        config->head_dim = config->dim / config->n_heads;
        config->rope_base = 10000.0f;
        config->partial_rotary_factor = 1.0f;
        config->norm_eps = 1e-5f;
        // all full attention (legacy)
        config->layer_types = malloc(config->n_layers);
        memset(config->layer_types, 1, config->n_layers);
        config->linear_num_key_heads = 0;
        config->linear_num_value_heads = 0;
        config->linear_key_head_dim = 0;
        config->linear_value_head_dim = 0;
        config->linear_conv_kernel_dim = 0;
        config->attnres_block_size = 0;
    }

    // count layer types
    config->n_full_layers = 0;
    config->n_linear_layers = 0;
    for (int l = 0; l < config->n_layers; l++) {
        if (config->layer_types[l] == 1) config->n_full_layers++;
        else config->n_linear_layers++;
    }

    fclose(file);

    // memory map the weights
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    fseek(fopen(checkpoint, "rb"), 0, SEEK_END); // hacky but works
    {
        FILE *f = fopen(checkpoint, "rb");
        fseek(f, 0, SEEK_END);
        *file_size = ftell(f);
        fclose(f);
    }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

    // weights start after the 512-byte header for v3, or after 7 ints for legacy
    int header_size = (magic == 0x616b3432 && version == 3) ? 512 : 7 * (int)sizeof(int);
    float* weights_ptr = (float*)((char*)*data + header_size);

    alloc_layer_pointers(weights, config->n_layers);
    memory_map_weights(weights, config, weights_ptr);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    free_run_state(&t->state, &t->config);
    free(t->config.layer_types);
    // free weight pointer arrays
    TransformerWeights *w = &t->weights;
    free(w->rms_att_weight); free(w->wq); free(w->wk); free(w->wv); free(w->wo);
    free(w->q_norm_weight); free(w->k_norm_weight);
    free(w->in_proj_qkv); free(w->in_proj_z); free(w->in_proj_b); free(w->in_proj_a);
    free(w->conv_weight); free(w->dt_bias); free(w->A_log);
    free(w->gated_norm_weight); free(w->out_proj);
    free(w->attn_res_proj); free(w->attn_res_norm);
    free(w->mlp_res_proj); free(w->mlp_res_norm);
    free(w->rms_ffn_weight); free(w->w1); free(w->w2); free(w->w3);
}

// ----------------------------------------------------------------------------
// neural net blocks

void rmsnorm(float* o, float* x, float* weight, int size, float eps) {
    // Qwen3.5 RMSNorm: (1 + weight) * (x / rms(x))
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
    ss /= size;
    ss += eps;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = (1.0f + weight[j]) * (ss * x[j]);
    }
}

void rmsnorm_gated(float* o, float* x, float* gate, float* weight, int size, float eps) {
    // Gated RMSNorm for linear attention: weight * norm(x) * SiLU(gate)
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
    ss /= size;
    ss += eps;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        float g = gate[j];
        float silu_g = g / (1.0f + expf(-g));
        o[j] = weight[j] * (ss * x[j]) * silu_g;
    }
}

void l2norm(float* o, float* x, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
    ss += 1e-6f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) { o[j] = x[j] * ss; }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) { x[i] /= sum; }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void reset_linear_state(RunState* s, Config* p) {
    int key_dim = p->linear_num_key_heads * p->linear_key_head_dim;
    int value_dim = p->linear_num_value_heads * p->linear_value_head_dim;
    int conv_dim = key_dim * 2 + value_dim;
    for (int i = 0; i < p->n_linear_layers; i++) {
        memset(s->conv_state[i], 0,
               conv_dim * (p->linear_conv_kernel_dim - 1) * sizeof(float));
        memset(s->recurrent_state[i], 0,
               p->linear_num_value_heads * p->linear_key_head_dim
               * p->linear_value_head_dim * sizeof(float));
    }
}

void block_attn_res(float* out, float** block_reps, int n_blocks,
                    float* partial_block, float* proj_weight,
                    float* norm_weight, int dim, float eps) {
    // Attend over (n_blocks completed blocks + 1 partial) to produce output
    int total = n_blocks + 1;
    float logits[64]; // max blocks (generous)

    for (int n = 0; n < total; n++) {
        float* v = (n < n_blocks) ? block_reps[n] : partial_block;
        // RMSNorm the value, then dot with proj_weight
        float ss = 0.0f;
        for (int j = 0; j < dim; j++) ss += v[j] * v[j];
        ss = 1.0f / sqrtf(ss / dim + eps);
        float dot = 0.0f;
        for (int j = 0; j < dim; j++) {
            float normed = (1.0f + norm_weight[j]) * (ss * v[j]);
            dot += proj_weight[j] * normed;
        }
        logits[n] = dot;
    }

    // softmax over block dimension
    float max_val = logits[0];
    for (int n = 1; n < total; n++) if (logits[n] > max_val) max_val = logits[n];
    float sum = 0.0f;
    for (int n = 0; n < total; n++) { logits[n] = expf(logits[n] - max_val); sum += logits[n]; }
    for (int n = 0; n < total; n++) logits[n] /= sum;

    // weighted sum
    memset(out, 0, dim * sizeof(float));
    for (int n = 0; n < total; n++) {
        float* v = (n < n_blocks) ? block_reps[n] : partial_block;
        float w = logits[n];
        for (int j = 0; j < dim; j++) out[j] += w * v[j];
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int head_dim = p->head_dim;
    int kv_dim = p->n_kv_heads * head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    float eps = p->norm_eps;

    // RoPE params
    int rotary_dim = (int)(head_dim * p->partial_rotary_factor);
    rotary_dim = rotary_dim - (rotary_dim % 2);
    if (rotary_dim < 2) rotary_dim = 2;

    // linear attention dims
    int lin_key_dim = p->linear_num_key_heads * p->linear_key_head_dim;
    int lin_value_dim = p->linear_num_value_heads * p->linear_value_head_dim;
    int lin_conv_dim = lin_key_dim * 2 + lin_value_dim;
    int ksize = p->linear_conv_kernel_dim;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));

    // Initialize block attnres state: token embedding is first block and partial
    int use_attnres = p->attnres_block_size > 0;
    if (use_attnres) {
        memcpy(s->block_reps[0], x, dim * sizeof(float));
        s->n_blocks = 1;
        memcpy(s->partial_block, x, dim * sizeof(float));
    }

    int full_idx = 0;   // index into full attention KV cache layers
    int lin_idx = 0;     // index into linear attention state layers

    for (int l = 0; l < p->n_layers; l++) {

        if (use_attnres) {
            int sublayer_idx = l * 2;
            int is_boundary = (sublayer_idx > 0 && sublayer_idx % p->attnres_block_size == 0);

            if (is_boundary) {
                // snapshot partial_block as new completed block
                memcpy(s->block_reps[s->n_blocks], s->partial_block, dim * sizeof(float));
                s->n_blocks++;
                // attend over all completed blocks (partial = last completed)
                block_attn_res(s->xb, s->block_reps, s->n_blocks,
                               s->block_reps[s->n_blocks - 1],
                               w->attn_res_proj[l], w->attn_res_norm[l], dim, eps);
            } else {
                // attend over completed blocks + current partial
                block_attn_res(s->xb, s->block_reps, s->n_blocks,
                               s->partial_block,
                               w->attn_res_proj[l], w->attn_res_norm[l], dim, eps);
            }
            // xb now has the attnres output; use it as input to attention norm
            rmsnorm(s->xb, s->xb, w->rms_att_weight[l], dim, eps);
        } else {
            // standard path: just norm the running hidden state
            rmsnorm(s->xb, x, w->rms_att_weight[l], dim, eps);
        }

        if (p->layer_types[l] == 1) {
            // ==================== FULL ATTENTION ====================
            int d_out = p->n_heads * head_dim;

            // gated Q matmul: output is 2x (queries + gate)
            matmul(s->q, s->xb, w->wq[l], dim, d_out * 2);
            matmul(s->k, s->xb, w->wk[l], dim, kv_dim);
            matmul(s->v, s->xb, w->wv[l], dim, kv_dim);

            // split Q into queries (xb) and gate -- reuse xb as temp for queries
            // s->q has [head0_q, head0_gate, head1_q, head1_gate, ...]
            // interleaved as (n_heads, head_dim*2) in row-major
            // We need to deinterleave into separate q and gate arrays
            // Use xb for queries, xb2 for gate temporarily
            for (int h = 0; h < p->n_heads; h++) {
                for (int i = 0; i < head_dim; i++) {
                    s->xb[h * head_dim + i] = s->q[h * head_dim * 2 + i];
                    s->xb2[h * head_dim + i] = s->q[h * head_dim * 2 + head_dim + i];
                }
            }
            // xb now has queries (d_out), xb2 has gate (d_out)
            // copy queries back to q for RoPE processing
            memcpy(s->q, s->xb, d_out * sizeof(float));
            // save gate in the upper part of the q buffer (we have 2x space)
            float* gate = s->q + d_out; // reuse second half of q buffer
            memcpy(gate, s->xb2, d_out * sizeof(float));

            // QK norm (per-head RMSNorm before RoPE)
            for (int h = 0; h < p->n_heads; h++) {
                rmsnorm(s->q + h * head_dim, s->q + h * head_dim,
                        w->q_norm_weight[l], head_dim, eps);
            }
            for (int h = 0; h < p->n_kv_heads; h++) {
                rmsnorm(s->k + h * head_dim, s->k + h * head_dim,
                        w->k_norm_weight[l], head_dim, eps);
            }

            // partial RoPE on Q (split-half style: pairs dim i with dim i+half)
            int half = rotary_dim / 2;
            for (int h = 0; h < p->n_heads; h++) {
                float* qh = s->q + h * head_dim;
                for (int i = 0; i < half; i++) {
                    float freq = 1.0f / powf(p->rope_base, (float)(i * 2) / (float)rotary_dim);
                    float val = pos * freq;
                    float fcr = cosf(val);
                    float fci = sinf(val);
                    float v0 = qh[i], v1 = qh[i + half];
                    qh[i]        = v0 * fcr - v1 * fci;
                    qh[i + half] = v1 * fcr + v0 * fci;
                }
            }
            // partial RoPE on K (split-half style)
            for (int h = 0; h < p->n_kv_heads; h++) {
                float* kh = s->k + h * head_dim;
                for (int i = 0; i < half; i++) {
                    float freq = 1.0f / powf(p->rope_base, (float)(i * 2) / (float)rotary_dim);
                    float val = pos * freq;
                    float fcr = cosf(val);
                    float fci = sinf(val);
                    float v0 = kh[i], v1 = kh[i + half];
                    kh[i]        = v0 * fcr - v1 * fci;
                    kh[i + half] = v1 * fcr + v0 * fci;
                }
            }

            // store K, V in cache for this full-attention layer
            long long loff = (long long)full_idx * p->seq_len * kv_dim;
            float* kcache = s->key_cache + loff + pos * kv_dim;
            float* vcache = s->value_cache + loff + pos * kv_dim;
            memcpy(kcache, s->k, kv_dim * sizeof(float));
            memcpy(vcache, s->v, kv_dim * sizeof(float));

            // multihead attention
            int h;
            #pragma omp parallel for private(h)
            for (h = 0; h < p->n_heads; h++) {
                float* qh = s->q + h * head_dim;
                float* atth = s->att + h * p->seq_len;
                for (int t = 0; t <= pos; t++) {
                    float* kk = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                    float score = 0.0f;
                    for (int i = 0; i < head_dim; i++) {
                        score += qh[i] * kk[i];
                    }
                    score /= sqrtf(head_dim);
                    atth[t] = score;
                }
                softmax(atth, pos + 1);
                // weighted sum of values
                float* xbh = s->xb + h * head_dim;
                memset(xbh, 0, head_dim * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float* vv = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                    float a = atth[t];
                    for (int i = 0; i < head_dim; i++) {
                        xbh[i] += a * vv[i];
                    }
                }
            }

            // apply sigmoid gate
            for (int i = 0; i < d_out; i++) {
                float g = 1.0f / (1.0f + expf(-gate[i]));
                s->xb[i] *= g;
            }

            // output projection
            matmul(s->xb2, s->xb, w->wo[l], d_out, dim);
            full_idx++;

        } else {
            // ==================== LINEAR ATTENTION ====================

            // 1) QKV projection
            matmul(s->lin_qkv, s->xb, w->in_proj_qkv[l], dim, lin_conv_dim);

            // 2) Causal depthwise conv (update state, apply kernel + SiLU)
            // conv_state holds the previous ksize-1 values per channel.
            // Full window is [state[0], state[1], ..., state[ksize-2], current_input].
            float* cs = s->conv_state[lin_idx];
            int cs_len = ksize - 1;
            for (int c = 0; c < lin_conv_dim; c++) {
                // dot product: state values with kernel[0..ksize-2], current with kernel[ksize-1]
                float val = 0.0f;
                for (int k = 0; k < cs_len; k++) {
                    val += cs[c * cs_len + k] * w->conv_weight[l][c * ksize + k];
                }
                val += s->lin_qkv[c] * w->conv_weight[l][c * ksize + ksize - 1];
                // shift state left, insert current value (for next timestep)
                for (int k = 0; k < cs_len - 1; k++) {
                    cs[c * cs_len + k] = cs[c * cs_len + k + 1];
                }
                cs[c * cs_len + cs_len - 1] = s->lin_qkv[c];
                // SiLU activation, store result
                s->lin_qkv[c] = val / (1.0f + expf(-val));
            }

            // 3) Split into Q, K, V
            float* lq = s->lin_qkv;
            float* lk = s->lin_qkv + lin_key_dim;
            float* lv = s->lin_qkv + lin_key_dim * 2;

            // 4) Gate projections
            matmul(s->lin_z, s->xb, w->in_proj_z[l], dim, lin_value_dim);
            matmul(s->lin_b, s->xb, w->in_proj_b[l], dim, p->linear_num_value_heads);
            matmul(s->lin_a, s->xb, w->in_proj_a[l], dim, p->linear_num_value_heads);

            // beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
            float beta_buf[256], g_buf[256]; // max heads
            for (int h = 0; h < p->linear_num_value_heads; h++) {
                beta_buf[h] = 1.0f / (1.0f + expf(-s->lin_b[h]));
                float a_val = s->lin_a[h] + w->dt_bias[l][h];
                float sp = a_val > 20.0f ? a_val : logf(1.0f + expf(a_val));
                g_buf[h] = -expf(w->A_log[l][h]) * sp;
            }

            // 5) Expand K heads to V heads if needed
            int kv_rep = p->linear_num_value_heads / p->linear_num_key_heads;
            // If kv_rep > 1, we need to expand Q and K in-place or use index mapping
            // For simplicity, use index mapping in the recurrence

            // 6) L2 norm Q and K per head
            for (int h = 0; h < p->linear_num_key_heads; h++) {
                l2norm(lq + h * p->linear_key_head_dim,
                       lq + h * p->linear_key_head_dim, p->linear_key_head_dim);
                l2norm(lk + h * p->linear_key_head_dim,
                       lk + h * p->linear_key_head_dim, p->linear_key_head_dim);
            }

            // 7) Recurrent delta rule (single timestep)
            float* rs = s->recurrent_state[lin_idx];
            int kd = p->linear_key_head_dim;
            int vd = p->linear_value_head_dim;
            float scale = 1.0f / sqrtf((float)kd);

            for (int h = 0; h < p->linear_num_value_heads; h++) {
                float* state = rs + h * kd * vd;
                int kh_idx = h / kv_rep; // map V head to K head
                float* q_h = lq + kh_idx * kd;
                float* k_h = lk + kh_idx * kd;
                float* v_h = lv + h * vd;
                float decay = expf(g_buf[h]);

                // decay state
                for (int i = 0; i < kd * vd; i++) state[i] *= decay;

                // kv_mem = state^T @ k -> (vd,)
                float kv_mem[512]; // max vd
                for (int j = 0; j < vd; j++) {
                    kv_mem[j] = 0.0f;
                    for (int i = 0; i < kd; i++) {
                        kv_mem[j] += state[i * vd + j] * k_h[i];
                    }
                }

                // delta = (v - kv_mem) * beta
                float delta[512]; // max vd
                for (int j = 0; j < vd; j++) {
                    delta[j] = (v_h[j] - kv_mem[j]) * beta_buf[h];
                }

                // state += outer(k, delta)
                for (int i = 0; i < kd; i++) {
                    for (int j = 0; j < vd; j++) {
                        state[i * vd + j] += k_h[i] * delta[j];
                    }
                }

                // output = state^T @ q * scale -> (vd,)
                float* out_h = s->lin_out + h * vd;
                for (int j = 0; j < vd; j++) {
                    out_h[j] = 0.0f;
                    for (int i = 0; i < kd; i++) {
                        out_h[j] += state[i * vd + j] * q_h[i];
                    }
                    out_h[j] *= scale;
                }
            }

            // 8) Gated RMSNorm per head
            for (int h = 0; h < p->linear_num_value_heads; h++) {
                rmsnorm_gated(s->lin_out + h * vd,
                              s->lin_out + h * vd,
                              s->lin_z + h * vd,
                              w->gated_norm_weight[l], vd, eps);
            }

            // 9) Output projection
            matmul(s->xb2, s->lin_out, w->out_proj[l], lin_value_dim, dim);
            lin_idx++;
        }

        if (use_attnres) {
            // attn residual: accumulate into partial_block
            int sublayer_idx = l * 2;
            int is_boundary = (sublayer_idx > 0 && sublayer_idx % p->attnres_block_size == 0);
            if (is_boundary) {
                // partial_block was reset; start fresh from attn output
                memcpy(s->partial_block, s->xb2, dim * sizeof(float));
            } else {
                for (int i = 0; i < dim; i++) s->partial_block[i] += s->xb2[i];
            }

            // Before MLP: attend over depth again
            block_attn_res(s->xb, s->block_reps, s->n_blocks,
                           s->partial_block,
                           w->mlp_res_proj[l], w->mlp_res_norm[l], dim, eps);
            // FFN with attnres input
            rmsnorm(s->xb, s->xb, w->rms_ffn_weight[l], dim, eps);
        } else {
            // standard residual
            for (int i = 0; i < dim; i++) { x[i] += s->xb2[i]; }
            rmsnorm(s->xb, x, w->rms_ffn_weight[l], dim, eps);
        }

        // FFN
        matmul(s->hb, s->xb, w->w1[l], dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3[l], dim, hidden_dim);
        // SwiGLU
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        matmul(s->xb, s->hb, w->w2[l], hidden_dim, dim);

        if (use_attnres) {
            for (int i = 0; i < dim; i++) s->partial_block[i] += s->xb[i];
        } else {
            for (int i = 0; i < dim; i++) { x[i] += s->xb[i]; }
        }
    }

    // final rmsnorm
    if (use_attnres) {
        rmsnorm(x, s->partial_block, w->rms_final_weight, dim, eps);
    } else {
        rmsnorm(x, x, w->rms_final_weight, dim, eps);
    }

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) { str_len = 0; }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) { continue; }
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < (int)str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) { break; }
        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) { return i; }
    }
    return n - 1;
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) { last_idx = i; break; }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) { return probindex[i].index; }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // reset linear attention state for fresh generation
    if (transformer->config.n_linear_layers > 0) {
        reset_linear_state(&transformer->state, &transformer->config);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;
        if (next == 1) { break; }
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    int8_t user_turn = 1;
    int next;
    int token;
    int prev_token;
    int pos = 0;
    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt == NULL) {
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            if (pos == 0 && cli_user_prompt != NULL) {
                strcpy(user_prompt, cli_user_prompt);
            } else {
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            printf("Assistant: ");
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        if (token == 2) { user_turn = 1; }

        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece);
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    char *mode = "generate";
    char *system_prompt = NULL;

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
