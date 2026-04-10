/* Inference for Qwen3.5 hybrid Transformer model in pure C, int8 quantized forward pass. */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
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
// Globals
int GS = 0; // group size global for quantization of the weights

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
    int head_dim;
    float rope_base;
    float partial_rotary_factor;
    float norm_eps;
    int linear_num_key_heads;
    int linear_num_value_heads;
    int linear_key_head_dim;
    int linear_value_head_dim;
    int linear_conv_kernel_dim;
    int attnres_block_size;
    int shared_weights;
    unsigned char* layer_types;
    int n_full_layers;
    int n_linear_layers;
} Config;

typedef struct {
    int8_t* q;    // quantized values
    float* s;     // scaling factors
} QuantizedTensor;

typedef struct {
    float* token_embedding_table;   // dequantized (vocab_size, dim)
    QuantizedTensor *q_tokens;      // quantized token embeddings

    // Per-layer fp32 weights
    float** rms_att_weight;
    float** q_norm_weight;
    float** k_norm_weight;
    float** conv_weight;
    float** dt_bias;
    float** A_log;
    float** gated_norm_weight;
    // Block AttnRes weights (fp32, NULL if disabled)
    float** attn_res_proj;
    float** attn_res_norm;
    float** mlp_res_proj;
    float** mlp_res_norm;

    float** rms_ffn_weight;
    float* rms_final_weight;

    // Per-layer quantized weights -- full attention
    QuantizedTensor **wq;
    QuantizedTensor **wk;
    QuantizedTensor **wv;
    QuantizedTensor **wo;

    // Per-layer quantized weights -- linear attention
    QuantizedTensor **in_proj_qkv;
    QuantizedTensor **in_proj_z;
    QuantizedTensor **in_proj_b;
    QuantizedTensor **in_proj_a;
    QuantizedTensor **out_proj;

    // Per-layer quantized weights -- FFN
    QuantizedTensor **w1;
    QuantizedTensor **w2;
    QuantizedTensor **w3;

    // Classifier
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    float *x, *xb, *xb2;
    float *hb, *hb2;
    float *logits;
    // quantized activation buffers
    QuantizedTensor xq;     // (dim)
    QuantizedTensor hq;     // (hidden_dim)
    QuantizedTensor lin_xq; // (dim) for linear attention projections
    QuantizedTensor lin_oq; // (value_dim) for linear out_proj
    // full attention
    float *q, *k, *v;
    float *att;
    float *key_cache, *value_cache;
    // linear attention
    float *lin_qkv, *lin_z, *lin_b, *lin_a, *lin_out;
    float **conv_state, **recurrent_state;
    // block attnres state
    float **block_reps;
    int n_blocks, max_blocks;
    float *partial_block;
    float *attnres_out;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) wmax = val;
        }
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale;
            qx->q[group * GS + i] = (int8_t) round(quant_value);
        }
    }
}

// Map a single quantized tensor from the mmap'd data
static void map_quantized_tensor(QuantizedTensor *qt, void **ptr, int size) {
    qt->q = (int8_t*)*ptr;
    *ptr = (int8_t*)*ptr + size;
    qt->s = (float*)*ptr;
    *ptr = (float*)*ptr + size / GS;
}

// ----------------------------------------------------------------------------
// Memory allocation and weight mapping

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = p->n_kv_heads * p->head_dim;
    int key_dim = p->linear_num_key_heads * p->linear_key_head_dim;
    int value_dim = p->linear_num_value_heads * p->linear_value_head_dim;
    int conv_dim = key_dim * 2 + value_dim;
    int d_out = p->n_heads * p->head_dim;

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));

    // quantized activation buffers
    s->xq = (QuantizedTensor){ .q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim / GS + 1, sizeof(float)) };
    s->hq = (QuantizedTensor){ .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim / GS + 1, sizeof(float)) };

    // full attention
    s->q = calloc(d_out * 2, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    if (p->n_full_layers > 0) {
        s->key_cache = calloc((long long)p->n_full_layers * p->seq_len * kv_dim, sizeof(float));
        s->value_cache = calloc((long long)p->n_full_layers * p->seq_len * kv_dim, sizeof(float));
    } else {
        s->key_cache = NULL; s->value_cache = NULL;
    }

    // linear attention
    if (p->n_linear_layers > 0) {
        s->lin_qkv = calloc(conv_dim, sizeof(float));
        s->lin_z = calloc(value_dim, sizeof(float));
        s->lin_b = calloc(p->linear_num_value_heads, sizeof(float));
        s->lin_a = calloc(p->linear_num_value_heads, sizeof(float));
        s->lin_out = calloc(value_dim, sizeof(float));
        s->lin_xq = (QuantizedTensor){ .q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim / GS + 1, sizeof(float)) };
        s->lin_oq = (QuantizedTensor){ .q = calloc(value_dim, sizeof(int8_t)), .s = calloc(value_dim / GS + 1, sizeof(float)) };

        s->conv_state = calloc(p->n_linear_layers, sizeof(float*));
        s->recurrent_state = calloc(p->n_linear_layers, sizeof(float*));
        int conv_state_size = conv_dim * (p->linear_conv_kernel_dim - 1);
        int rec_state_size = p->linear_num_value_heads * p->linear_key_head_dim * p->linear_value_head_dim;
        for (int i = 0; i < p->n_linear_layers; i++) {
            s->conv_state[i] = calloc(conv_state_size, sizeof(float));
            s->recurrent_state[i] = calloc(rec_state_size, sizeof(float));
        }
    } else {
        s->lin_qkv = NULL; s->lin_z = NULL; s->lin_b = NULL;
        s->lin_a = NULL; s->lin_out = NULL;
        s->lin_xq = (QuantizedTensor){0}; s->lin_oq = (QuantizedTensor){0};
        s->conv_state = NULL; s->recurrent_state = NULL;
    }

    if (p->attnres_block_size > 0) {
        s->max_blocks = (p->n_layers * 2) / p->attnres_block_size + 2;
        s->block_reps = calloc(s->max_blocks, sizeof(float*));
        for (int i = 0; i < s->max_blocks; i++) s->block_reps[i] = calloc(p->dim, sizeof(float));
        s->partial_block = calloc(p->dim, sizeof(float));
        s->attnres_out = calloc(p->dim, sizeof(float));
        s->n_blocks = 0;
    } else {
        s->block_reps = NULL; s->partial_block = NULL;
        s->attnres_out = NULL; s->max_blocks = 0; s->n_blocks = 0;
    }

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->logits) {
        fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s, Config* p) {
    free(s->x); free(s->xb); free(s->xb2);
    free(s->hb); free(s->hb2); free(s->logits);
    free(s->xq.q); free(s->xq.s); free(s->hq.q); free(s->hq.s);
    free(s->q); free(s->k); free(s->v); free(s->att);
    free(s->key_cache); free(s->value_cache);
    free(s->lin_qkv); free(s->lin_z); free(s->lin_b);
    free(s->lin_a); free(s->lin_out);
    free(s->lin_xq.q); free(s->lin_xq.s);
    free(s->lin_oq.q); free(s->lin_oq.s);
    if (s->conv_state) {
        for (int i = 0; i < p->n_linear_layers; i++) {
            free(s->conv_state[i]); free(s->recurrent_state[i]);
        }
        free(s->conv_state); free(s->recurrent_state);
    }
    if (s->block_reps) {
        for (int i = 0; i < s->max_blocks; i++) free(s->block_reps[i]);
        free(s->block_reps);
    }
    free(s->partial_block); free(s->attnres_out);
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr) {
    int head_dim = p->head_dim;
    int key_dim = p->linear_num_key_heads * p->linear_key_head_dim;
    int value_dim = p->linear_num_value_heads * p->linear_value_head_dim;
    int conv_dim = key_dim * 2 + value_dim;
    int n = p->n_layers;

    // alloc pointer arrays
    w->rms_att_weight = calloc(n, sizeof(float*));
    w->q_norm_weight = calloc(n, sizeof(float*));
    w->k_norm_weight = calloc(n, sizeof(float*));
    w->conv_weight = calloc(n, sizeof(float*));
    w->dt_bias = calloc(n, sizeof(float*));
    w->A_log = calloc(n, sizeof(float*));
    w->gated_norm_weight = calloc(n, sizeof(float*));
    w->rms_ffn_weight = calloc(n, sizeof(float*));
    w->wq = calloc(n, sizeof(QuantizedTensor*));
    w->wk = calloc(n, sizeof(QuantizedTensor*));
    w->wv = calloc(n, sizeof(QuantizedTensor*));
    w->wo = calloc(n, sizeof(QuantizedTensor*));
    w->in_proj_qkv = calloc(n, sizeof(QuantizedTensor*));
    w->in_proj_z = calloc(n, sizeof(QuantizedTensor*));
    w->in_proj_b = calloc(n, sizeof(QuantizedTensor*));
    w->in_proj_a = calloc(n, sizeof(QuantizedTensor*));
    w->out_proj = calloc(n, sizeof(QuantizedTensor*));
    w->attn_res_proj = calloc(n, sizeof(float*));
    w->attn_res_norm = calloc(n, sizeof(float*));
    w->mlp_res_proj = calloc(n, sizeof(float*));
    w->mlp_res_norm = calloc(n, sizeof(float*));
    w->w1 = calloc(n, sizeof(QuantizedTensor*));
    w->w2 = calloc(n, sizeof(QuantizedTensor*));
    w->w3 = calloc(n, sizeof(QuantizedTensor*));

    // Token embeddings (quantized)
    w->q_tokens = malloc(sizeof(QuantizedTensor));
    map_quantized_tensor(w->q_tokens, &ptr, p->vocab_size * p->dim);
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    // Per-layer weights
    for (int l = 0; l < n; l++) {
        // attention norm (fp32)
        w->rms_att_weight[l] = (float*)ptr;
        ptr = (float*)ptr + p->dim;

        if (p->layer_types[l] == 1) { // full attention
            w->wq[l] = malloc(sizeof(QuantizedTensor));
            map_quantized_tensor(w->wq[l], &ptr, p->dim * (p->n_heads * head_dim * 2));
            w->wk[l] = malloc(sizeof(QuantizedTensor));
            map_quantized_tensor(w->wk[l], &ptr, p->dim * (p->n_kv_heads * head_dim));
            w->wv[l] = malloc(sizeof(QuantizedTensor));
            map_quantized_tensor(w->wv[l], &ptr, p->dim * (p->n_kv_heads * head_dim));
            w->wo[l] = malloc(sizeof(QuantizedTensor));
            map_quantized_tensor(w->wo[l], &ptr, (p->n_heads * head_dim) * p->dim);
            w->q_norm_weight[l] = (float*)ptr; ptr = (float*)ptr + head_dim;
            w->k_norm_weight[l] = (float*)ptr; ptr = (float*)ptr + head_dim;
        } else { // linear attention
            w->in_proj_qkv[l] = malloc(sizeof(QuantizedTensor));
            map_quantized_tensor(w->in_proj_qkv[l], &ptr, conv_dim * p->dim);
            w->in_proj_z[l] = malloc(sizeof(QuantizedTensor));
            map_quantized_tensor(w->in_proj_z[l], &ptr, value_dim * p->dim);
            w->in_proj_b[l] = malloc(sizeof(QuantizedTensor));
            map_quantized_tensor(w->in_proj_b[l], &ptr, p->linear_num_value_heads * p->dim);
            w->in_proj_a[l] = malloc(sizeof(QuantizedTensor));
            map_quantized_tensor(w->in_proj_a[l], &ptr, p->linear_num_value_heads * p->dim);
            // small params: fp32
            w->conv_weight[l] = (float*)ptr;
            ptr = (float*)ptr + conv_dim * 1 * p->linear_conv_kernel_dim;
            w->dt_bias[l] = (float*)ptr; ptr = (float*)ptr + p->linear_num_value_heads;
            w->A_log[l] = (float*)ptr; ptr = (float*)ptr + p->linear_num_value_heads;
            w->gated_norm_weight[l] = (float*)ptr; ptr = (float*)ptr + p->linear_value_head_dim;
            w->out_proj[l] = malloc(sizeof(QuantizedTensor));
            map_quantized_tensor(w->out_proj[l], &ptr, value_dim * p->dim);
        }

        // AttnRes weights (fp32)
        if (p->attnres_block_size > 0) {
            w->attn_res_proj[l] = (float*)ptr; ptr = (float*)ptr + p->dim;
            w->attn_res_norm[l] = (float*)ptr; ptr = (float*)ptr + p->dim;
            w->mlp_res_proj[l] = (float*)ptr;  ptr = (float*)ptr + p->dim;
            w->mlp_res_norm[l] = (float*)ptr;  ptr = (float*)ptr + p->dim;
        }

        // FFN
        w->rms_ffn_weight[l] = (float*)ptr; ptr = (float*)ptr + p->dim;
        w->w1[l] = malloc(sizeof(QuantizedTensor));
        map_quantized_tensor(w->w1[l], &ptr, p->hidden_dim * p->dim);
        w->w2[l] = malloc(sizeof(QuantizedTensor));
        map_quantized_tensor(w->w2[l], &ptr, p->dim * p->hidden_dim);
        w->w3[l] = malloc(sizeof(QuantizedTensor));
        map_quantized_tensor(w->w3[l], &ptr, p->hidden_dim * p->dim);
    }

    // Final norm (fp32)
    w->rms_final_weight = (float*)ptr; ptr = (float*)ptr + p->dim;

    // Classifier
    if (!p->shared_weights) {
        w->wcls = malloc(sizeof(QuantizedTensor));
        map_quantized_tensor(w->wcls, &ptr, p->dim * p->vocab_size);
    } else {
        w->wcls = w->q_tokens;
    }
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }

    uint32_t magic;
    int version;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&version, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (magic != 0x616b3432 || version != 4) {
        fprintf(stderr, "Bad magic/version. Expected v4 quantized format, got magic=0x%x version=%d\n", magic, version);
        exit(EXIT_FAILURE);
    }

    if (fread(&config->dim, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->hidden_dim, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->n_layers, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->n_heads, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->n_kv_heads, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->vocab_size, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->seq_len, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->head_dim, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->rope_base, sizeof(float), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->partial_rotary_factor, sizeof(float), 1, file) != 1) exit(EXIT_FAILURE);
    if (fread(&config->norm_eps, sizeof(float), 1, file) != 1) exit(EXIT_FAILURE);
    int linear_params[6];
    if (fread(linear_params, sizeof(int), 6, file) != 6) exit(EXIT_FAILURE);
    config->linear_num_key_heads = linear_params[0];
    config->linear_num_value_heads = linear_params[1];
    config->linear_key_head_dim = linear_params[2];
    config->linear_value_head_dim = linear_params[3];
    config->linear_conv_kernel_dim = linear_params[4];
    config->attnres_block_size = linear_params[5];

    unsigned char shared_flag;
    if (fread(&shared_flag, sizeof(unsigned char), 1, file) != 1) exit(EXIT_FAILURE);
    config->shared_weights = (int)shared_flag;
    config->layer_types = malloc(config->n_layers);
    if (fread(config->layer_types, 1, config->n_layers, file) != (size_t)config->n_layers) exit(EXIT_FAILURE);

    int group_size;
    if (fread(&group_size, sizeof(int), 1, file) != 1) exit(EXIT_FAILURE);
    GS = group_size;

    config->n_full_layers = 0; config->n_linear_layers = 0;
    for (int l = 0; l < config->n_layers; l++) {
        if (config->layer_types[l] == 1) config->n_full_layers++;
        else config->n_linear_layers++;
    }

    fclose(file);

    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    {
        FILE *f = fopen(checkpoint, "rb");
        fseek(f, 0, SEEK_END);
        *file_size = ftell(f);
        fclose(f);
    }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

    void* weights_ptr = (char*)*data + 512; // v4 header is 512 bytes
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
    // free quantized tensor structs and pointer arrays
    TransformerWeights *w = &t->weights;
    free(w->token_embedding_table);
    free(w->q_tokens);
    for (int l = 0; l < t->config.n_layers; l++) {
        if (t->config.layer_types[l] == 1) {
            free(w->wq[l]); free(w->wk[l]); free(w->wv[l]); free(w->wo[l]);
        } else {
            free(w->in_proj_qkv[l]); free(w->in_proj_z[l]);
            free(w->in_proj_b[l]); free(w->in_proj_a[l]); free(w->out_proj[l]);
        }
        free(w->w1[l]); free(w->w2[l]); free(w->w3[l]);
    }
    if (w->wcls != w->q_tokens) free(w->wcls);
    free(w->rms_att_weight); free(w->q_norm_weight); free(w->k_norm_weight);
    free(w->conv_weight); free(w->dt_bias); free(w->A_log);
    free(w->gated_norm_weight); free(w->rms_ffn_weight);
    free(w->wq); free(w->wk); free(w->wv); free(w->wo);
    free(w->in_proj_qkv); free(w->in_proj_z); free(w->in_proj_b);
    free(w->in_proj_a); free(w->out_proj);
    free(w->attn_res_proj); free(w->attn_res_norm);
    free(w->mlp_res_proj); free(w->mlp_res_norm);
    free(w->w1); free(w->w2); free(w->w3);
}

// ----------------------------------------------------------------------------
// neural net blocks

void rmsnorm(float* o, float* x, float* weight, int size, float eps) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
    ss /= size; ss += eps;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) { o[j] = (1.0f + weight[j]) * (ss * x[j]); }
}

void rmsnorm_gated(float* o, float* x, float* gate, float* weight, int size, float eps) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
    ss /= size; ss += eps;
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
    for (int i = 1; i < size; i++) { if (x[i] > max_val) max_val = x[i]; }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) { x[i] /= sum; }
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,), both inputs quantized
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }
        xout[i] = val;
    }
}

void reset_linear_state(RunState* s, Config* p) {
    int key_dim = p->linear_num_key_heads * p->linear_key_head_dim;
    int value_dim = p->linear_num_value_heads * p->linear_value_head_dim;
    int conv_dim = key_dim * 2 + value_dim;
    for (int i = 0; i < p->n_linear_layers; i++) {
        memset(s->conv_state[i], 0, conv_dim * (p->linear_conv_kernel_dim - 1) * sizeof(float));
        memset(s->recurrent_state[i], 0,
               p->linear_num_value_heads * p->linear_key_head_dim
               * p->linear_value_head_dim * sizeof(float));
    }
}

void block_attn_res(float* out, float** block_reps, int n_blocks,
                    float* partial_block, float* proj_weight,
                    float* norm_weight, int dim, float eps) {
    int total = n_blocks + 1;
    float logits[64];
    for (int n = 0; n < total; n++) {
        float* v = (n < n_blocks) ? block_reps[n] : partial_block;
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
    float max_val = logits[0];
    for (int n = 1; n < total; n++) if (logits[n] > max_val) max_val = logits[n];
    float sum = 0.0f;
    for (int n = 0; n < total; n++) { logits[n] = expf(logits[n] - max_val); sum += logits[n]; }
    for (int n = 0; n < total; n++) logits[n] /= sum;
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

    int rotary_dim = (int)(head_dim * p->partial_rotary_factor);
    rotary_dim = rotary_dim - (rotary_dim % 2);
    if (rotary_dim < 2) rotary_dim = 2;
    int half = rotary_dim / 2;

    int lin_key_dim = p->linear_num_key_heads * p->linear_key_head_dim;
    int lin_value_dim = p->linear_num_value_heads * p->linear_value_head_dim;
    int lin_conv_dim = lin_key_dim * 2 + lin_value_dim;
    int ksize = p->linear_conv_kernel_dim;

    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

    int use_attnres = p->attnres_block_size > 0;
    if (use_attnres) {
        memcpy(s->block_reps[0], x, dim * sizeof(float));
        s->n_blocks = 1;
        memcpy(s->partial_block, x, dim * sizeof(float));
    }

    int full_idx = 0;
    int lin_idx = 0;

    for (int l = 0; l < p->n_layers; l++) {

        if (use_attnres) {
            int sublayer_idx = l * 2;
            int is_boundary = (sublayer_idx > 0 && sublayer_idx % p->attnres_block_size == 0);
            if (is_boundary) {
                memcpy(s->block_reps[s->n_blocks], s->partial_block, dim * sizeof(float));
                s->n_blocks++;
                block_attn_res(s->xb, s->block_reps, s->n_blocks,
                               s->block_reps[s->n_blocks - 1],
                               w->attn_res_proj[l], w->attn_res_norm[l], dim, eps);
            } else {
                block_attn_res(s->xb, s->block_reps, s->n_blocks,
                               s->partial_block,
                               w->attn_res_proj[l], w->attn_res_norm[l], dim, eps);
            }
            rmsnorm(s->xb, s->xb, w->rms_att_weight[l], dim, eps);
        } else {
            rmsnorm(s->xb, x, w->rms_att_weight[l], dim, eps);
        }

        if (p->layer_types[l] == 1) {
            // ==================== FULL ATTENTION (quantized matmuls) ====================
            int d_out = p->n_heads * head_dim;

            quantize(&s->xq, s->xb, dim);
            matmul(s->q, &s->xq, w->wq[l], dim, d_out * 2);
            matmul(s->k, &s->xq, w->wk[l], dim, kv_dim);
            matmul(s->v, &s->xq, w->wv[l], dim, kv_dim);

            // split gated Q
            for (int h = 0; h < p->n_heads; h++) {
                for (int i = 0; i < head_dim; i++) {
                    s->xb[h * head_dim + i] = s->q[h * head_dim * 2 + i];
                    s->xb2[h * head_dim + i] = s->q[h * head_dim * 2 + head_dim + i];
                }
            }
            memcpy(s->q, s->xb, d_out * sizeof(float));
            float* gate = s->q + d_out;
            memcpy(gate, s->xb2, d_out * sizeof(float));

            // QK norm
            for (int h = 0; h < p->n_heads; h++)
                rmsnorm(s->q + h*head_dim, s->q + h*head_dim, w->q_norm_weight[l], head_dim, eps);
            for (int h = 0; h < p->n_kv_heads; h++)
                rmsnorm(s->k + h*head_dim, s->k + h*head_dim, w->k_norm_weight[l], head_dim, eps);

            // partial RoPE (split-half)
            for (int h = 0; h < p->n_heads; h++) {
                float* qh = s->q + h * head_dim;
                for (int i = 0; i < half; i++) {
                    float freq = 1.0f / powf(p->rope_base, (float)(i*2) / (float)rotary_dim);
                    float val = pos * freq;
                    float fcr = cosf(val), fci = sinf(val);
                    float v0 = qh[i], v1 = qh[i+half];
                    qh[i] = v0*fcr - v1*fci;
                    qh[i+half] = v1*fcr + v0*fci;
                }
            }
            for (int h = 0; h < p->n_kv_heads; h++) {
                float* kh = s->k + h * head_dim;
                for (int i = 0; i < half; i++) {
                    float freq = 1.0f / powf(p->rope_base, (float)(i*2) / (float)rotary_dim);
                    float val = pos * freq;
                    float fcr = cosf(val), fci = sinf(val);
                    float v0 = kh[i], v1 = kh[i+half];
                    kh[i] = v0*fcr - v1*fci;
                    kh[i+half] = v1*fcr + v0*fci;
                }
            }

            // KV cache
            long long loff = (long long)full_idx * p->seq_len * kv_dim;
            memcpy(s->key_cache + loff + pos*kv_dim, s->k, kv_dim*sizeof(float));
            memcpy(s->value_cache + loff + pos*kv_dim, s->v, kv_dim*sizeof(float));

            // multihead attention
            int h;
            #pragma omp parallel for private(h)
            for (h = 0; h < p->n_heads; h++) {
                float* qh = s->q + h*head_dim;
                float* atth = s->att + h*p->seq_len;
                for (int t = 0; t <= pos; t++) {
                    float* kk = s->key_cache + loff + t*kv_dim + (h/kv_mul)*head_dim;
                    float score = 0.0f;
                    for (int i = 0; i < head_dim; i++) score += qh[i]*kk[i];
                    atth[t] = score / sqrtf(head_dim);
                }
                softmax(atth, pos+1);
                float* xbh = s->xb + h*head_dim;
                memset(xbh, 0, head_dim*sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float* vv = s->value_cache + loff + t*kv_dim + (h/kv_mul)*head_dim;
                    float a = atth[t];
                    for (int i = 0; i < head_dim; i++) xbh[i] += a*vv[i];
                }
            }

            // sigmoid gate
            for (int i = 0; i < d_out; i++)
                s->xb[i] *= 1.0f / (1.0f + expf(-gate[i]));

            // output projection (quantized)
            quantize(&s->xq, s->xb, d_out);
            matmul(s->xb2, &s->xq, w->wo[l], d_out, dim);
            full_idx++;

        } else {
            // ==================== LINEAR ATTENTION (quantized projections) ====================

            // 1) QKV (quantized)
            quantize(&s->lin_xq, s->xb, dim);
            matmul(s->lin_qkv, &s->lin_xq, w->in_proj_qkv[l], dim, lin_conv_dim);

            // 2) Causal depthwise conv (fp32)
            float* cs = s->conv_state[lin_idx];
            int cs_len = ksize - 1;
            for (int c = 0; c < lin_conv_dim; c++) {
                float val = 0.0f;
                for (int k = 0; k < cs_len; k++)
                    val += cs[c*cs_len+k] * w->conv_weight[l][c*ksize+k];
                val += s->lin_qkv[c] * w->conv_weight[l][c*ksize+ksize-1];
                for (int k = 0; k < cs_len-1; k++)
                    cs[c*cs_len+k] = cs[c*cs_len+k+1];
                cs[c*cs_len+cs_len-1] = s->lin_qkv[c];
                s->lin_qkv[c] = val / (1.0f + expf(-val));
            }

            // 3) Split Q, K, V
            float* lq = s->lin_qkv;
            float* lk = s->lin_qkv + lin_key_dim;
            float* lv = s->lin_qkv + lin_key_dim * 2;

            // 4) Gate projections (quantized)
            matmul(s->lin_z, &s->lin_xq, w->in_proj_z[l], dim, lin_value_dim);
            matmul(s->lin_b, &s->lin_xq, w->in_proj_b[l], dim, p->linear_num_value_heads);
            matmul(s->lin_a, &s->lin_xq, w->in_proj_a[l], dim, p->linear_num_value_heads);

            float beta_buf[256], g_buf[256];
            for (int h = 0; h < p->linear_num_value_heads; h++) {
                beta_buf[h] = 1.0f / (1.0f + expf(-s->lin_b[h]));
                float a_val = s->lin_a[h] + w->dt_bias[l][h];
                float sp = a_val > 20.0f ? a_val : logf(1.0f + expf(a_val));
                g_buf[h] = -expf(w->A_log[l][h]) * sp;
            }

            // 5) L2 norm Q and K
            for (int h = 0; h < p->linear_num_key_heads; h++) {
                l2norm(lq + h*p->linear_key_head_dim, lq + h*p->linear_key_head_dim, p->linear_key_head_dim);
                l2norm(lk + h*p->linear_key_head_dim, lk + h*p->linear_key_head_dim, p->linear_key_head_dim);
            }

            // 6) Recurrent delta rule
            float* rs = s->recurrent_state[lin_idx];
            int kd = p->linear_key_head_dim;
            int vd = p->linear_value_head_dim;
            float scale = 1.0f / sqrtf((float)kd);
            int kv_rep = p->linear_num_value_heads / p->linear_num_key_heads;

            for (int h = 0; h < p->linear_num_value_heads; h++) {
                float* state = rs + h*kd*vd;
                int kh_idx = h / kv_rep;
                float* q_h = lq + kh_idx*kd;
                float* k_h = lk + kh_idx*kd;
                float* v_h = lv + h*vd;
                float decay = expf(g_buf[h]);
                for (int i = 0; i < kd*vd; i++) state[i] *= decay;
                float kv_mem[512], delta[512];
                for (int j = 0; j < vd; j++) {
                    kv_mem[j] = 0.0f;
                    for (int i = 0; i < kd; i++) kv_mem[j] += state[i*vd+j] * k_h[i];
                }
                for (int j = 0; j < vd; j++)
                    delta[j] = (v_h[j] - kv_mem[j]) * beta_buf[h];
                for (int i = 0; i < kd; i++)
                    for (int j = 0; j < vd; j++)
                        state[i*vd+j] += k_h[i] * delta[j];
                float* out_h = s->lin_out + h*vd;
                for (int j = 0; j < vd; j++) {
                    out_h[j] = 0.0f;
                    for (int i = 0; i < kd; i++) out_h[j] += state[i*vd+j] * q_h[i];
                    out_h[j] *= scale;
                }
            }

            // 7) Gated RMSNorm
            for (int h = 0; h < p->linear_num_value_heads; h++)
                rmsnorm_gated(s->lin_out + h*vd, s->lin_out + h*vd,
                              s->lin_z + h*vd, w->gated_norm_weight[l], vd, eps);

            // 8) Output projection (quantized)
            quantize(&s->lin_oq, s->lin_out, lin_value_dim);
            matmul(s->xb2, &s->lin_oq, w->out_proj[l], lin_value_dim, dim);
            lin_idx++;
        }

        if (use_attnres) {
            int sublayer_idx = l * 2;
            int is_boundary = (sublayer_idx > 0 && sublayer_idx % p->attnres_block_size == 0);
            if (is_boundary) {
                memcpy(s->partial_block, s->xb2, dim * sizeof(float));
            } else {
                for (int i = 0; i < dim; i++) s->partial_block[i] += s->xb2[i];
            }
            block_attn_res(s->xb, s->block_reps, s->n_blocks,
                           s->partial_block,
                           w->mlp_res_proj[l], w->mlp_res_norm[l], dim, eps);
            rmsnorm(s->xb, s->xb, w->rms_ffn_weight[l], dim, eps);
        } else {
            for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
            rmsnorm(s->xb, x, w->rms_ffn_weight[l], dim, eps);
        }

        // FFN (quantized matmuls)
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1[l], dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3[l], dim, hidden_dim);
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2[l], hidden_dim, dim);

        if (use_attnres) {
            for (int i = 0; i < dim; i++) s->partial_block[i] += s->xb[i];
        } else {
            for (int i = 0; i < dim; i++) x[i] += s->xb[i];
        }
    }

    if (use_attnres) {
        rmsnorm(x, s->partial_block, w->rms_final_weight, dim, eps);
    } else {
        rmsnorm(x, x, w->rms_final_weight, dim, eps);
    }
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
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
    unsigned char byte_pieces[512];
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
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab); free(t->vocab_scores); free(t->sorted_vocab);
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
    if (piece == NULL) return;
    if (piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
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
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) { tokens[(*n_tokens)++] = id; }
        else { for (int i=0; i<(int)str_len; i++) tokens[(*n_tokens)++] = (unsigned char)str_buffer[i]+3; }
        str_len = 0;
    }
    while (1) {
        float best_score = -1e10; int best_id = -1; int best_idx = -1;
        for (int i=0; i<(*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) { best_score = t->vocab_scores[id]; best_id = id; best_idx = i; }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) tokens[i] = tokens[i+1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler

typedef struct { float prob; int index; } ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0; float max_p = probabilities[0];
    for (int i = 1; i < n; i++) { if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; } }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) { cdf += probabilities[i]; if (coin < cdf) return i; }
    return n - 1;
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*)a; ProbIndex* b_ = (ProbIndex*)b;
    if (a_->prob > b_->prob) return -1; if (a_->prob < b_->prob) return 1; return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) { probindex[n0].index = i; probindex[n0].prob = probabilities[i]; n0++; }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);
    float cumulative_prob = 0.0f; int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) { cumulative_prob += probindex[i].prob; if (cumulative_prob > topp) { last_idx = i; break; } }
    float r = coin * cumulative_prob; float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) { cdf += probindex[i].prob; if (r < cdf) return probindex[i].index; }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size; sampler->temperature = temperature;
    sampler->topp = topp; sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}
void free_sampler(Sampler* sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12; *state ^= *state << 25; *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) { next = sample_argmax(logits, sampler->vocab_size); }
    else {
        for (int q=0; q<sampler->vocab_size; q++) logits[q] /= sampler->temperature;
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) next = sample_mult(logits, sampler->vocab_size, coin);
        else next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
    return next;
}

// ----------------------------------------------------------------------------
long time_in_ms() {
    struct timespec time; clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) prompt = empty_prompt;
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) { fprintf(stderr, "expected at least 1 prompt token\n"); exit(EXIT_FAILURE); }
    if (transformer->config.n_linear_layers > 0)
        reset_linear_state(&transformer->state, &transformer->config);
    long start = 0; int next; int token = prompt_tokens[0]; int pos = 0;
    while (pos < steps) {
        float* logits = forward(transformer, token, pos);
        if (pos < num_prompt_tokens - 1) next = prompt_tokens[pos+1];
        else next = sample(sampler, logits);
        pos++;
        if (next == 1) break;
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); fflush(stdout);
        token = next;
        if (start == 0) start = time_in_ms();
    }
    printf("\n");
    if (pos > 1) { long end = time_in_ms(); fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000); }
    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len-1] == '\n') buffer[len-1] = '\0';
    }
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {
    char system_prompt[512]; char user_prompt[512]; char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx; int8_t user_turn = 1; int next; int token; int pos = 0;
    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt == NULL) read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                else strcpy(system_prompt, cli_system_prompt);
            }
            if (pos == 0 && cli_user_prompt != NULL) strcpy(user_prompt, cli_user_prompt);
            else read_stdin("User: ", user_prompt, sizeof(user_prompt));
            if (pos == 0 && system_prompt[0] != '\0') sprintf(rendered_prompt, "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]", system_prompt, user_prompt);
            else sprintf(rendered_prompt, "[INST] %s [/INST]", user_prompt);
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; user_turn = 0; printf("Assistant: ");
        }
        if (user_idx < num_prompt_tokens) token = prompt_tokens[user_idx++];
        else token = next;
        if (token == 2) user_turn = 1;
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits); pos++;
        if (user_idx >= num_prompt_tokens && next != 2) { char* piece = decode(tokenizer, token, next); safe_printf(piece); fflush(stdout); }
        if (next == 2) printf("\n");
    }
    printf("\n"); free(prompt_tokens);
}

// ----------------------------------------------------------------------------
#ifndef TESTING
void error_usage() {
    fprintf(stderr, "Usage:   runq <checkpoint> [options]\n");
    fprintf(stderr, "Example: runq model.bin -n 256 -i \"Once upon a time\"\n");
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
    char *checkpoint_path = NULL; char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f; float topp = 0.9f; int steps = 256;
    char *prompt = NULL; unsigned long long rng_seed = 0;
    char *mode = "generate"; char *system_prompt = NULL;
    if (argc >= 2) checkpoint_path = argv[1]; else error_usage();
    for (int i = 2; i < argc; i+=2) {
        if (i+1 >= argc) error_usage(); if (argv[i][0] != '-') error_usage(); if (strlen(argv[i]) != 2) error_usage();
        if (argv[i][1] == 't') temperature = atof(argv[i+1]);
        else if (argv[i][1] == 'p') topp = atof(argv[i+1]);
        else if (argv[i][1] == 's') rng_seed = atoi(argv[i+1]);
        else if (argv[i][1] == 'n') steps = atoi(argv[i+1]);
        else if (argv[i][1] == 'i') prompt = argv[i+1];
        else if (argv[i][1] == 'z') tokenizer_path = argv[i+1];
        else if (argv[i][1] == 'm') mode = argv[i+1];
        else if (argv[i][1] == 'y') system_prompt = argv[i+1];
        else error_usage();
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
    if (strcmp(mode, "generate") == 0) generate(&transformer, &tokenizer, &sampler, prompt, steps);
    else if (strcmp(mode, "chat") == 0) chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    else { fprintf(stderr, "unknown mode: %s\n", mode); error_usage(); }
    free_sampler(&sampler); free_tokenizer(&tokenizer); free_transformer(&transformer);
    return 0;
}
#endif
