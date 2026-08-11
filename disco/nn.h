/* nn.h -- tiny reverse-mode autodiff + instrumented inference for architecture search.
 *
 * Design contract (this is what makes the evaluator trustworthy):
 *   ONE forward function per architecture serves BOTH
 *     (a) full-sequence training  (g_train=1, tape recorded, grads allocated)
 *     (b) single-step decoding with KV/state cache (g_train=0, tape off)
 *   so the model that is measured is bit-for-bit the model that was trained.
 *   Weight-byte traffic is COUNTED inside the ops during (b), never estimated.
 */
#ifndef NN_H
#define NN_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXDIM 4
#define MAXTAPE 200000
#define MAXPARAM 4096

typedef struct Tensor {
  int shape[MAXDIM], ndim, n;
  float *d, *g;
  int is_param;
  float bpe;          /* bytes per element in the DEPLOYED representation (4=fp32, 1=int8, .25=2bit) */
  struct Tensor *src_param;  /* if this activation is a transform of a param, bill I/O to it */
  float *qbuf; int qvalid;   /* inference-time cache for quantised weights */
  unsigned char *touch;  /* per-row touched flags for the current decode step */
  unsigned char *touchw; /* same, but cleared once per WINDOW of tokens */
  int nrow, rowelem, tany, tfull, wany, wfull;
  char name[48];
} Tensor;

/* ---------------- global state ---------------- */
extern int   g_train;         /* 1 = record tape + accumulate grads */
extern long long g_wbytes;    /* weight bytes read (counted in ops)  */
extern long long g_sbytes;    /* state/cache bytes read              */
extern long long g_flops;     /* multiply-accumulates                */

/* ---------------- arena ---------------- */
void  arena_init(size_t bytes);
void  arena_reset(void);
void *arena_alloc(size_t bytes);
size_t arena_used(void);

/* ---------------- tensors ---------------- */
Tensor *T_new(int ndim, int a, int b, int c, int dd);        /* activation (arena) */
Tensor *P_new(const char *name, int ndim, int a, int b, int c, int dd);  /* parameter */
extern Tensor *g_params[MAXPARAM];
extern int g_nparams;
long long params_count(void);
double   params_bytes(void);
void     params_init_normal(Tensor *t, float std);
void     params_init_zeros(Tensor *t);
void     params_init_ones(Tensor *t);

/* Two distinct I/O quantities, both measured, neither estimated:
 *   g_wbytes   raw read TRAFFIC  (every read event counted, re-reads included)
 *   touch[]    UNIQUE bytes touched in one decode step (what a cache actually fetches)
 * A layer-recurrent model has small unique / large traffic; a dense model has them equal.
 * That gap is the whole point of the second axis.  */
static inline Tensor *io_owner(Tensor *t){ return t->is_param ? t : t->src_param; }
static inline void touch_all_(Tensor *t){
  if(g_train||!t->is_param) return;
  if(!t->tfull){ memset(t->touch,1,t->nrow); t->tfull=1; t->tany=1; }
  if(!t->wfull){ memset(t->touchw,1,t->nrow); t->wfull=1; t->wany=1; }
}
static inline void touch_row_(Tensor *t,int r){
  if(g_train||!t->is_param) return;
  t->touch[r]=1; t->tany=1;
  t->touchw[r]=1; t->wany=1;
}
static inline void count_read_(Tensor *t, long long nelem) {
  if (g_train) return;
  if (t->is_param){ g_wbytes += (long long)(nelem * t->bpe); }
  else             g_sbytes += (long long)(nelem * 4);
}
/* public forms follow src_param so quantised / derived weight views are billed
   to the stored parameter at its real bytes-per-element */
static inline void touch_all(Tensor *t){ Tensor*a=io_owner(t); if(a) touch_all_(a); }
static inline void touch_row(Tensor *t,int r){ Tensor*a=io_owner(t); if(a) touch_row_(a,r); }
static inline void count_read(Tensor *t,long long n){ Tensor*a=io_owner(t); if(a) count_read_(a,n); else count_read_(t,n); }
long long wbytes_unique_step(void);   /* sum touched rows, then clear for next token */
/* Unique bytes touched over a WINDOW of consecutive tokens, divided by the window.
   For a dense model this equals wbytes_unique_step().  For conditional computation
   it is LARGER, because successive tokens visit different experts and the cache
   sees their union.  Discovered because hashffn has identical MACs and identical
   per-token unique bytes to the baseline yet decodes 17% slower. */
long long wbytes_window_flush(void);

/* ---------------- ops ---------------- */
/* out[M,O] = x[M,I] @ W[O,I]^T */
Tensor *op_linear (Tensor *x, Tensor *W);
/* out[M,N] = a[M,K] @ b[K,N]   (b may be an activation: hypernets, factorization) */
Tensor *op_matmul (Tensor *a, Tensor *b);
Tensor *op_emb    (Tensor *W, int *idx, int M);
Tensor *op_rmsnorm(Tensor *x, Tensor *w);
Tensor *op_add    (Tensor *a, Tensor *b);
Tensor *op_mul    (Tensor *a, Tensor *b);
Tensor *op_addbias(Tensor *x, Tensor *b);            /* broadcast b[D] over rows */
Tensor *op_scale  (Tensor *x, float s);
#define ACT_SILU 0
#define ACT_GELU 1
#define ACT_RELU 2
#define ACT_SIGM 3
#define ACT_TANH 4
#define ACT_SQR  5
Tensor *op_act    (Tensor *x, int kind);
Tensor *op_softmax(Tensor *x);                       /* row-wise over last dim */
Tensor *op_slice  (Tensor *x, int off, int len);     /* last-dim slice */
Tensor *op_concat (Tensor *a, Tensor *b);
/* Write src into dst's columns [off, off+src_width).  Building a wide tensor by
   chaining op_concat is O(K^2) in memory -- at K=8 heads the intermediates came
   to 303MB and exhausted the arena.  Allocate once, place K times. */
void    op_place  (Tensor *dst, Tensor *src, int off);
Tensor *op_gather (Tensor *W, int *idx, int M);      /* like emb, distinct name for MoE/hash */
Tensor *op_ste_quant(Tensor *x, int levels);         /* straight-through uniform quantizer */
Tensor *op_ternary(Tensor *W);                       /* per-row-scaled {-s,0,s}, STE, cached at infer */
/* generic per-row symmetric quantiser: 2*levels+1 states.  levels==1 uses a
   mean-absolute scale (ternary); higher levels use absmax.  STE backward. */
Tensor *op_qrow(Tensor *W, int levels);
float   qrow_bpe(int levels);                        /* bytes/element of the packed form */
Tensor *op_rows   (Tensor *x, int *idx, int n);      /* gather activation rows */
Tensor *op_scatter(Tensor *src, int *idx, int n, int M);  /* scatter rows into an [M,D] zero tensor */
Tensor *T_view(int R, int C, float *data);           /* header over existing storage, no copy */
Tensor *op_rope   (Tensor *x, int B, int T, int nh, int hd, int pos0, float theta);
/* fused causal attention.  q[M,nh*hd] k,v[M,nkv*hd].  cache!=NULL => streaming step. */
typedef struct { float *k, *v; int cap, len, nkv, hd; } KVCache;
Tensor *op_attn(Tensor *q, Tensor *k, Tensor *v, int B, int T, int nh, int nkv, int hd,
                KVCache *cache, float softcap);
double  op_ce_loss(Tensor *logits, int *targets, int M);   /* returns mean NLL, seeds grads */
/* --- token mixers with O(1) decode state (contrast: attention's growing KV) --- */
/* depthwise causal conv, width W, per-channel kernel[D,W]. hist!=NULL => streaming. */
Tensor *op_dwconv(Tensor *x, Tensor *kern, int B, int T, int W, float *hist);
/* per-channel exponential moving average, decay = sigmoid(a[D]). st!=NULL => streaming. */
Tensor *op_ema(Tensor *x, Tensor *a, int B, int T, float *st);

/* ---------------- tape ---------------- */
void tape_reset(void);
void tape_backward(void);

/* ---------------- optimizer ---------------- */
typedef struct { float *m, *v; } AdamSlot;
void opt_init(void);
void opt_step(float lr, float beta1, float beta2, float eps, float wd, int t, float clip);
void opt_zero_grad(void);
void params_invalidate_q(void);
void opt_state_save(FILE *f,int step);
int  opt_state_load(FILE *f);

/* ---------------- rng ---------------- */
extern uint64_t g_rng;
static inline uint32_t rnd_u32(void){ g_rng^=g_rng>>12; g_rng^=g_rng<<25; g_rng^=g_rng>>27; return (uint32_t)((g_rng*0x2545F4914F6CDD1DULL)>>32); }
static inline float rnd_f(void){ return (rnd_u32()>>8)/16777216.0f; }
float rnd_normal(void);

/* ---------------- profiling ---------------- */
extern double g_prof[24]; extern long long g_profn[24]; extern int g_prof_on;
void prof_dump(const char *tag);

/* ---------------- misc ---------------- */
long peak_rss_kb(void);
double now_sec(void);

#endif
