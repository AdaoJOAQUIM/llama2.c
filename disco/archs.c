/* archs.c -- architecture registry.
 * Every arch exposes ONE forward used for both training and cached decoding.
 */
#include "model.h"

float cfg_get(Cfg *c,const char*k,float d){ for(int i=0;i<c->nknob;i++) if(!strcmp(c->kname[i],k)) return c->kval[i]; return d; }
int   cfg_geti(Cfg *c,const char*k,int d){ return (int)cfg_get(c,k,(float)d); }

/* ================================================================= *
 *  A0: baseline llama2 (rmsnorm, GQA+rope, SwiGLU, tied embeddings)
 * ================================================================= */
#define MAXL 24
typedef struct {
  Tensor *emb,*outw,*fno;
  Tensor *an[MAXL],*wq[MAXL],*wk[MAXL],*wv[MAXL],*wo[MAXL];
  Tensor *fn[MAXL],*w1[MAXL],*w2[MAXL],*w3[MAXL];
  KVCache kv[MAXL];
} Llama;
static Llama L;

static void nm(char*b,const char*p,int i){ sprintf(b,"%s.%d",p,i); }
/* [1,D] of ones as a plain activation: lets op_matmul broadcast a per-row scalar */
static Tensor *op_ones_row(int D){ Tensor*t=T_new(2,1,D,0,0); for(int i=0;i<D;i++) t->d[i]=1.0f; return t; }

static void llama_build(Cfg*c){
  int D=c->dim,H=c->hidden_dim,V=c->vocab,hd=D/c->n_heads,kvd=c->n_kv_heads*hd;
  char b[48];
  L.emb=P_new("emb",2,V,D,0,0); params_init_normal(L.emb,0.02f);
  for(int l=0;l<c->n_layers;l++){
    nm(b,"an",l);  L.an[l]=P_new(b,1,D,0,0,0); params_init_ones(L.an[l]);
    nm(b,"wq",l);  L.wq[l]=P_new(b,2,D,D,0,0);   params_init_normal(L.wq[l],0.02f);
    nm(b,"wk",l);  L.wk[l]=P_new(b,2,kvd,D,0,0); params_init_normal(L.wk[l],0.02f);
    nm(b,"wv",l);  L.wv[l]=P_new(b,2,kvd,D,0,0); params_init_normal(L.wv[l],0.02f);
    nm(b,"wo",l);  L.wo[l]=P_new(b,2,D,D,0,0);   params_init_normal(L.wo[l],0.02f);
    nm(b,"fn",l);  L.fn[l]=P_new(b,1,D,0,0,0); params_init_ones(L.fn[l]);
    nm(b,"w1",l);  L.w1[l]=P_new(b,2,H,D,0,0); params_init_normal(L.w1[l],0.02f);
    nm(b,"w3",l);  L.w3[l]=P_new(b,2,H,D,0,0); params_init_normal(L.w3[l],0.02f);
    nm(b,"w2",l);  L.w2[l]=P_new(b,2,D,H,0,0); params_init_normal(L.w2[l],0.02f);
  }
  L.fno=P_new("fno",1,D,0,0,0); params_init_ones(L.fno);
  if(c->tie) L.outw=L.emb; else { L.outw=P_new("outw",2,V,D,0,0); params_init_normal(L.outw,0.02f); }
}
static void llama_cache_alloc(Cfg*c,int maxlen){
  int hd=c->dim/c->n_heads,kvd=c->n_kv_heads*hd;
  for(int l=0;l<c->n_layers;l++){ L.kv[l].k=(float*)calloc((size_t)maxlen*kvd,4); L.kv[l].v=(float*)calloc((size_t)maxlen*kvd,4);
    L.kv[l].cap=maxlen; L.kv[l].len=0; L.kv[l].nkv=c->n_kv_heads; L.kv[l].hd=hd; }
}
static void llama_cache_reset(Cfg*c){ for(int l=0;l<c->n_layers;l++) L.kv[l].len=0; }

static Tensor *llama_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads;
  Tensor *x=op_emb(L.emb,tok,B*T);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_rmsnorm(x,L.an[l]);
    Tensor *q=op_linear(h,L.wq[l]),*k=op_linear(h,L.wk[l]),*v=op_linear(h,L.wv[l]);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,v,B,T,c->n_heads,c->n_kv_heads,hd,uc?&L.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,L.wo[l]));
    Tensor *f=op_rmsnorm(x,L.fn[l]);
    Tensor *g=op_mul(op_act(op_linear(f,L.w1[l]),ACT_SILU),op_linear(f,L.w3[l]));
    x=op_add(x,op_linear(g,L.w2[l]));
  }
  x=op_rmsnorm(x,L.fno);
  return op_linear(x,L.outw);
}


/* ================================================================= *
 *  A1: sharedloop -- ONE transformer block iterated n_layers times.
 *  The only depth-specific parameters are two affine modulations (norm gain +
 *  bias) applied before attention and before the FFN.  Relative to Universal
 *  Transformer / ALBERT this drops the depth embedding entirely: depth is
 *  expressed purely as an affine reparameterisation of a shared operator.
 * ================================================================= */
typedef struct {
  Tensor *emb,*outw,*fno;
  Tensor *wq,*wk,*wv,*wo,*w1,*w2,*w3;      /* shared across depth */
  Tensor *an[MAXL],*ab[MAXL],*fn[MAXL],*fb[MAXL];  /* per-depth affine */
  KVCache kv[MAXL];
} Loop;
static Loop R;
static void loop_build(Cfg*c){
  int D=c->dim,H=c->hidden_dim,V=c->vocab,hd=D/c->n_heads,kvd=c->n_kv_heads*hd; char b[48];
  R.emb=P_new("emb",2,V,D,0,0); params_init_normal(R.emb,0.02f);
  R.wq=P_new("wq",2,D,D,0,0);   params_init_normal(R.wq,0.02f);
  R.wk=P_new("wk",2,kvd,D,0,0); params_init_normal(R.wk,0.02f);
  R.wv=P_new("wv",2,kvd,D,0,0); params_init_normal(R.wv,0.02f);
  R.wo=P_new("wo",2,D,D,0,0);   params_init_normal(R.wo,0.02f);
  R.w1=P_new("w1",2,H,D,0,0);   params_init_normal(R.w1,0.02f);
  R.w3=P_new("w3",2,H,D,0,0);   params_init_normal(R.w3,0.02f);
  R.w2=P_new("w2",2,D,H,0,0);   params_init_normal(R.w2,0.02f);
  for(int l=0;l<c->n_layers;l++){
    nm(b,"an",l); R.an[l]=P_new(b,1,D,0,0,0); params_init_ones(R.an[l]);
    nm(b,"ab",l); R.ab[l]=P_new(b,1,D,0,0,0); params_init_zeros(R.ab[l]);
    nm(b,"fn",l); R.fn[l]=P_new(b,1,D,0,0,0); params_init_ones(R.fn[l]);
    nm(b,"fb",l); R.fb[l]=P_new(b,1,D,0,0,0); params_init_zeros(R.fb[l]);
  }
  R.fno=P_new("fno",1,D,0,0,0); params_init_ones(R.fno);
  if(c->tie) R.outw=R.emb; else { R.outw=P_new("outw",2,V,D,0,0); params_init_normal(R.outw,0.02f); }
}
static void loop_cache_alloc(Cfg*c,int ml){
  int hd=c->dim/c->n_heads,kvd=c->n_kv_heads*hd;
  for(int l=0;l<c->n_layers;l++){ R.kv[l].k=(float*)calloc((size_t)ml*kvd,4); R.kv[l].v=(float*)calloc((size_t)ml*kvd,4);
    R.kv[l].cap=ml; R.kv[l].len=0; R.kv[l].nkv=c->n_kv_heads; R.kv[l].hd=hd; }
}
static void loop_cache_reset(Cfg*c){ for(int l=0;l<c->n_layers;l++) R.kv[l].len=0; }
static Tensor *loop_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads;
  Tensor *x=op_emb(R.emb,tok,B*T);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_addbias(op_rmsnorm(x,R.an[l]),R.ab[l]);
    Tensor *q=op_linear(h,R.wq),*k=op_linear(h,R.wk),*v=op_linear(h,R.wv);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,v,B,T,c->n_heads,c->n_kv_heads,hd,uc?&R.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,R.wo));
    Tensor *f=op_addbias(op_rmsnorm(x,R.fn[l]),R.fb[l]);
    Tensor *g=op_mul(op_act(op_linear(f,R.w1),ACT_SILU),op_linear(f,R.w3));
    x=op_add(x,op_linear(g,R.w2));
  }
  return op_linear(op_rmsnorm(x,R.fno),R.outw);
}

/* ================================================================= *
 *  A2: hashffn -- K FFN experts per layer, routed by a BIGRAM hash of
 *  (current token id, previous token id).  No router parameters, no load
 *  balancing loss, no learned gating: routing is a pure function of the input
 *  text.  Beyond published hash layers in that the route is context-dependent
 *  (bigram) rather than a fixed per-token-id assignment.
 * ================================================================= */
#define MAXEXP 8
typedef struct {
  Tensor *emb,*outw,*fno;
  Tensor *an[MAXL],*wq[MAXL],*wk[MAXL],*wv[MAXL],*wo[MAXL],*fn[MAXL];
  Tensor *w1[MAXL][MAXEXP],*w2[MAXL][MAXEXP],*w3[MAXL][MAXEXP];
  KVCache kv[MAXL];
  int K, last_tok;
} Hash;
static Hash Hh;
static void hash_build(Cfg*c){
  int D=c->dim,H=c->hidden_dim,V=c->vocab,hd=D/c->n_heads,kvd=c->n_kv_heads*hd; char b[48];
  Hh.K=cfg_geti(c,"experts",4); if(Hh.K>MAXEXP) Hh.K=MAXEXP; Hh.last_tok=256;
  Hh.emb=P_new("emb",2,V,D,0,0); params_init_normal(Hh.emb,0.02f);
  for(int l=0;l<c->n_layers;l++){
    nm(b,"an",l); Hh.an[l]=P_new(b,1,D,0,0,0); params_init_ones(Hh.an[l]);
    nm(b,"wq",l); Hh.wq[l]=P_new(b,2,D,D,0,0);   params_init_normal(Hh.wq[l],0.02f);
    nm(b,"wk",l); Hh.wk[l]=P_new(b,2,kvd,D,0,0); params_init_normal(Hh.wk[l],0.02f);
    nm(b,"wv",l); Hh.wv[l]=P_new(b,2,kvd,D,0,0); params_init_normal(Hh.wv[l],0.02f);
    nm(b,"wo",l); Hh.wo[l]=P_new(b,2,D,D,0,0);   params_init_normal(Hh.wo[l],0.02f);
    nm(b,"fn",l); Hh.fn[l]=P_new(b,1,D,0,0,0); params_init_ones(Hh.fn[l]);
    for(int e=0;e<Hh.K;e++){ char q[48];
      sprintf(q,"w1.%d.%d",l,e); Hh.w1[l][e]=P_new(q,2,H,D,0,0); params_init_normal(Hh.w1[l][e],0.02f);
      sprintf(q,"w3.%d.%d",l,e); Hh.w3[l][e]=P_new(q,2,H,D,0,0); params_init_normal(Hh.w3[l][e],0.02f);
      sprintf(q,"w2.%d.%d",l,e); Hh.w2[l][e]=P_new(q,2,D,H,0,0); params_init_normal(Hh.w2[l][e],0.02f); }
  }
  Hh.fno=P_new("fno",1,D,0,0,0); params_init_ones(Hh.fno);
  if(c->tie) Hh.outw=Hh.emb; else { Hh.outw=P_new("outw",2,V,D,0,0); params_init_normal(Hh.outw,0.02f); }
}
static void hash_cache_alloc(Cfg*c,int ml){
  int hd=c->dim/c->n_heads,kvd=c->n_kv_heads*hd;
  for(int l=0;l<c->n_layers;l++){ Hh.kv[l].k=(float*)calloc((size_t)ml*kvd,4); Hh.kv[l].v=(float*)calloc((size_t)ml*kvd,4);
    Hh.kv[l].cap=ml; Hh.kv[l].len=0; Hh.kv[l].nkv=c->n_kv_heads; Hh.kv[l].hd=hd; }
}
static void hash_cache_reset(Cfg*c){ for(int l=0;l<c->n_layers;l++) Hh.kv[l].len=0; Hh.last_tok=256; }
static inline int bigram_route(int cur,int prev,int K){
  unsigned h=(unsigned)cur*2654435761u ^ (unsigned)prev*40503u; h^=h>>13; return (int)(h%(unsigned)K);
}
static Tensor *hash_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads,M=B*T,K=Hh.K;
  /* route each row; prev token comes from the sequence, or from decode state */
  static int rt[1<<16]; static int idx[MAXEXP][1<<16]; int cnt[MAXEXP];
  for(int b=0;b<B;b++)for(int t=0;t<T;t++){ int m=b*T+t;
    int prev = (t>0)? tok[m-1] : (uc? Hh.last_tok : 256);
    rt[m]=bigram_route(tok[m],prev,K); }
  for(int e=0;e<K;e++) cnt[e]=0;
  for(int m=0;m<M;m++) idx[rt[m]][cnt[rt[m]]++]=m;
  if(uc) Hh.last_tok = tok[T-1];

  Tensor *x=op_emb(Hh.emb,tok,M);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_rmsnorm(x,Hh.an[l]);
    Tensor *q=op_linear(h,Hh.wq[l]),*k=op_linear(h,Hh.wk[l]),*v=op_linear(h,Hh.wv[l]);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,v,B,T,c->n_heads,c->n_kv_heads,hd,uc?&Hh.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,Hh.wo[l]));
    Tensor *f=op_rmsnorm(x,Hh.fn[l]);
    Tensor *acc=NULL;
    for(int e=0;e<K;e++){
      if(!cnt[e]) continue;
      Tensor *fe=op_rows(f,idx[e],cnt[e]);
      Tensor *ge=op_mul(op_act(op_linear(fe,Hh.w1[l][e]),ACT_SILU),op_linear(fe,Hh.w3[l][e]));
      Tensor *oe=op_scatter(op_linear(ge,Hh.w2[l][e]),idx[e],cnt[e],M);
      acc = acc? op_add(acc,oe) : oe;
    }
    x=op_add(x,acc);
  }
  return op_linear(op_rmsnorm(x,Hh.fno),Hh.outw);
}

/* ================================================================= *
 *  A3: ternffn -- FFN weights stored ternary (2 bits/weight, per-row scale,
 *  straight-through).  Attention and embeddings stay fp32.  Same parameter
 *  COUNT as the baseline, ~2.8x fewer weight bytes read per token.
 * ================================================================= */
static void tern_build(Cfg*c){
  llama_build(c);
  float bpe=cfg_get(c,"ffn_bpe",0.25f);
  for(int l=0;l<c->n_layers;l++){ L.w1[l]->bpe=bpe; L.w2[l]->bpe=bpe; L.w3[l]->bpe=bpe; }
}
static Tensor *tern_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads;
  Tensor *x=op_emb(L.emb,tok,B*T);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_rmsnorm(x,L.an[l]);
    Tensor *q=op_linear(h,L.wq[l]),*k=op_linear(h,L.wk[l]),*v=op_linear(h,L.wv[l]);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,v,B,T,c->n_heads,c->n_kv_heads,hd,uc?&L.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,L.wo[l]));
    Tensor *f=op_rmsnorm(x,L.fn[l]);
    Tensor *g=op_mul(op_act(op_linear(f,op_ternary(L.w1[l])),ACT_SILU),op_linear(f,op_ternary(L.w3[l])));
    x=op_add(x,op_linear(g,op_ternary(L.w2[l])));
  }
  return op_linear(op_rmsnorm(x,L.fno),L.outw);
}

/* ================================================================= *
 *  A4: novalue -- attention reuses K as V, deleting the value projection.
 *  Probes an assumption every variant so far shares: that the thing you
 *  average must be a different projection from the thing you match on.
 * ================================================================= */
static void nov_build(Cfg*c){
  int D=c->dim,H=c->hidden_dim,V=c->vocab,hd=D/c->n_heads,kvd=c->n_kv_heads*hd; char b[48];
  L.emb=P_new("emb",2,V,D,0,0); params_init_normal(L.emb,0.02f);
  for(int l=0;l<c->n_layers;l++){
    nm(b,"an",l); L.an[l]=P_new(b,1,D,0,0,0); params_init_ones(L.an[l]);
    nm(b,"wq",l); L.wq[l]=P_new(b,2,D,D,0,0);   params_init_normal(L.wq[l],0.02f);
    nm(b,"wk",l); L.wk[l]=P_new(b,2,kvd,D,0,0); params_init_normal(L.wk[l],0.02f);
    nm(b,"wo",l); L.wo[l]=P_new(b,2,D,D,0,0);   params_init_normal(L.wo[l],0.02f);
    nm(b,"fn",l); L.fn[l]=P_new(b,1,D,0,0,0); params_init_ones(L.fn[l]);
    nm(b,"w1",l); L.w1[l]=P_new(b,2,H,D,0,0); params_init_normal(L.w1[l],0.02f);
    nm(b,"w3",l); L.w3[l]=P_new(b,2,H,D,0,0); params_init_normal(L.w3[l],0.02f);
    nm(b,"w2",l); L.w2[l]=P_new(b,2,D,H,0,0); params_init_normal(L.w2[l],0.02f);
    L.wv[l]=NULL;
  }
  L.fno=P_new("fno",1,D,0,0,0); params_init_ones(L.fno);
  if(c->tie) L.outw=L.emb; else { L.outw=P_new("outw",2,V,D,0,0); params_init_normal(L.outw,0.02f); }
}
static Tensor *nov_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads;
  Tensor *x=op_emb(L.emb,tok,B*T);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_rmsnorm(x,L.an[l]);
    Tensor *q=op_linear(h,L.wq[l]),*k=op_linear(h,L.wk[l]);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,k,B,T,c->n_heads,c->n_kv_heads,hd,uc?&L.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,L.wo[l]));
    Tensor *f=op_rmsnorm(x,L.fn[l]);
    Tensor *g=op_mul(op_act(op_linear(f,L.w1[l]),ACT_SILU),op_linear(f,L.w3[l]));
    x=op_add(x,op_linear(g,L.w2[l]));
  }
  return op_linear(op_rmsnorm(x,L.fno),L.outw);
}


/* ================================================================= *
 *  A5: emaconv -- attention REPLACED by a two-timescale linear mixer:
 *  a short depthwise causal convolution (local) plus a per-channel learned
 *  exponential moving average (unbounded range, O(1) state).  Decode state is
 *  a fixed (W-1+1)*D floats per layer instead of a KV cache that grows with
 *  context.  Not a published SSM: no state expansion, no selectivity, no
 *  complex/diagonal parameterisation -- just two learned timescales per
 *  channel, gated multiplicatively.
 * ================================================================= */
typedef struct {
  Tensor *emb,*outw,*fno;
  Tensor *an[MAXL],*wu[MAXL],*wg[MAXL],*kern[MAXL],*alpha[MAXL],*wo[MAXL];
  Tensor *fn[MAXL],*w1[MAXL],*w2[MAXL],*w3[MAXL];
  float *hist[MAXL],*st[MAXL];
  int W;
} Ema;
static Ema E;
static void ema_build(Cfg*c){
  int D=c->dim,H=c->hidden_dim,V=c->vocab; char b[48];
  E.W=cfg_geti(c,"convw",4);
  E.emb=P_new("emb",2,V,D,0,0); params_init_normal(E.emb,0.02f);
  for(int l=0;l<c->n_layers;l++){
    nm(b,"an",l);  E.an[l]=P_new(b,1,D,0,0,0); params_init_ones(E.an[l]);
    nm(b,"wu",l);  E.wu[l]=P_new(b,2,D,D,0,0); params_init_normal(E.wu[l],0.02f);
    nm(b,"wg",l);  E.wg[l]=P_new(b,2,D,D,0,0); params_init_normal(E.wg[l],0.02f);
    nm(b,"kern",l);E.kern[l]=P_new(b,2,D,E.W,0,0);
      for(int d=0;d<D;d++) for(int j=0;j<E.W;j++) E.kern[l]->d[(size_t)d*E.W+j]=(j==0)?1.0f:rnd_normal()*0.1f;
    nm(b,"alpha",l);E.alpha[l]=P_new(b,1,D,0,0,0);
      for(int d=0;d<D;d++) E.alpha[l]->d[d]=1.0f+rnd_normal()*0.5f;   /* sigmoid ~ 0.73 */
    nm(b,"wo",l);  E.wo[l]=P_new(b,2,D,2*D,0,0); params_init_normal(E.wo[l],0.02f);
    nm(b,"fn",l);  E.fn[l]=P_new(b,1,D,0,0,0); params_init_ones(E.fn[l]);
    nm(b,"w1",l);  E.w1[l]=P_new(b,2,H,D,0,0); params_init_normal(E.w1[l],0.02f);
    nm(b,"w3",l);  E.w3[l]=P_new(b,2,H,D,0,0); params_init_normal(E.w3[l],0.02f);
    nm(b,"w2",l);  E.w2[l]=P_new(b,2,D,H,0,0); params_init_normal(E.w2[l],0.02f);
  }
  E.fno=P_new("fno",1,D,0,0,0); params_init_ones(E.fno);
  if(c->tie) E.outw=E.emb; else { E.outw=P_new("outw",2,V,D,0,0); params_init_normal(E.outw,0.02f); }
}
static void ema_cache_alloc(Cfg*c,int ml){
  (void)ml; int D=c->dim;
  for(int l=0;l<c->n_layers;l++){ E.hist[l]=(float*)calloc((size_t)(E.W>1?E.W-1:1)*D,4); E.st[l]=(float*)calloc(D,4); }
}
static void ema_cache_reset(Cfg*c){ int D=c->dim;
  for(int l=0;l<c->n_layers;l++){ memset(E.hist[l],0,(size_t)(E.W>1?E.W-1:1)*D*4); memset(E.st[l],0,(size_t)D*4); } }
static Tensor *ema_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  (void)pos0;
  Tensor *x=op_emb(E.emb,tok,B*T);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_rmsnorm(x,E.an[l]);
    Tensor *u=op_linear(h,E.wu[l]);
    Tensor *gt=op_act(op_linear(h,E.wg[l]),ACT_SILU);
    Tensor *cv=op_dwconv(u,E.kern[l],B,T,E.W,uc?E.hist[l]:NULL);
    Tensor *em=op_ema(u,E.alpha[l],B,T,uc?E.st[l]:NULL);
    Tensor *mix=op_mul(op_concat(cv,em),op_concat(gt,gt));
    x=op_add(x,op_linear(mix,E.wo[l]));
    Tensor *f=op_rmsnorm(x,E.fn[l]);
    Tensor *g=op_mul(op_act(op_linear(f,E.w1[l]),ACT_SILU),op_linear(f,E.w3[l]));
    x=op_add(x,op_linear(g,E.w2[l]));
  }
  return op_linear(op_rmsnorm(x,E.fno),E.outw);
}

/* ================================================================= *
 *  A6: ngrammem -- a hashed BIGRAM memory table injected into the residual
 *  stream at every depth with a per-depth learned scale.  The table has
 *  NSLOT rows and is addressed by hash(tok, prev_tok); exactly ONE row is read
 *  per token.  Parameters grow by NSLOT*D while unique bytes read grow by
 *  D*4 = 256 bytes.  A learned n-gram model welded to a transformer.
 * ================================================================= */
typedef struct {
  Tensor *mem, *msc[MAXL];
  int nslot, last_tok;
} NGram;
static NGram G;
static void ng_build(Cfg*c){
  llama_build(c);
  G.nslot=cfg_geti(c,"nslot",4096); G.last_tok=256;
  G.mem=P_new("mem",2,G.nslot,c->dim,0,0); params_init_normal(G.mem,0.02f);
  char b[48];
  for(int l=0;l<c->n_layers;l++){ nm(b,"msc",l); G.msc[l]=P_new(b,1,c->dim,0,0,0); params_init_zeros(G.msc[l]); }
}
static void ng_cache_reset(Cfg*c){ llama_cache_reset(c); G.last_tok=256; }
static Tensor *ng_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads,M=B*T;
  static int slot[1<<16];
  for(int b=0;b<B;b++)for(int t=0;t<T;t++){ int m=b*T+t;
    int prev=(t>0)?tok[m-1]:(uc?G.last_tok:256);
    unsigned h=(unsigned)tok[m]*2654435761u ^ (unsigned)prev*2246822519u; h^=h>>15;
    slot[m]=(int)(h%(unsigned)G.nslot); }
  if(uc) G.last_tok=tok[T-1];
  Tensor *mv=op_emb(G.mem,slot,M);          /* one row read per token */
  Tensor *x=op_emb(L.emb,tok,M);
  for(int l=0;l<c->n_layers;l++){
    x=op_add(x,op_mul(mv,op_addbias(op_scale(mv,0.f),G.msc[l])));  /* mv * msc_l */
    Tensor *h=op_rmsnorm(x,L.an[l]);
    Tensor *q=op_linear(h,L.wq[l]),*k=op_linear(h,L.wk[l]),*v=op_linear(h,L.wv[l]);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,v,B,T,c->n_heads,c->n_kv_heads,hd,uc?&L.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,L.wo[l]));
    Tensor *f=op_rmsnorm(x,L.fn[l]);
    Tensor *g=op_mul(op_act(op_linear(f,L.w1[l]),ACT_SILU),op_linear(f,L.w3[l]));
    x=op_add(x,op_linear(g,L.w2[l]));
  }
  return op_linear(op_rmsnorm(x,L.fno),L.outw);
}

/* ================================================================= *
 *  A7: deepgate -- per-token, per-layer FFN skip.  Each layer computes a
 *  scalar gate from the FFN input; rows with gate <= 1/2 do not execute the
 *  FFN at all, so w1/w2/w3 (73% of a layer's parameters) are never touched
 *  for those tokens.  The SAME rule runs in training, evaluation and decoding,
 *  so val_bpb and bytes/token describe one model, not two.
 *  Consequence worth stating up front: a closed row receives no gradient at
 *  its gate, so gates can shut but not reopen.  Whether that collapses is an
 *  empirical question this variant exists to answer.
 * ================================================================= */
typedef struct { Tensor *gw[MAXL],*gb[MAXL]; } Gate;
static Gate GA;
static void dg_build(Cfg*c){
  llama_build(c); char b[48];
  float b0=cfg_get(c,"gbias",2.0f);
  for(int l=0;l<c->n_layers;l++){
    nm(b,"gw",l); GA.gw[l]=P_new(b,2,1,c->dim,0,0); params_init_normal(GA.gw[l],0.02f);
    nm(b,"gb",l); GA.gb[l]=P_new(b,1,1,0,0,0); GA.gb[l]->d[0]=b0;   /* start ~88% open */
  }
}
static Tensor *dg_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads,M=B*T;
  static int idx[1<<16];
  Tensor *x=op_emb(L.emb,tok,M);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_rmsnorm(x,L.an[l]);
    Tensor *q=op_linear(h,L.wq[l]),*k=op_linear(h,L.wk[l]),*v=op_linear(h,L.wv[l]);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,v,B,T,c->n_heads,c->n_kv_heads,hd,uc?&L.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,L.wo[l]));

    Tensor *f=op_rmsnorm(x,L.fn[l]);
    Tensor *gate=op_act(op_addbias(op_linear(f,GA.gw[l]),GA.gb[l]),ACT_SIGM);  /* [M,1] */
    int n=0; for(int m=0;m<M;m++) if(gate->d[m]>0.5f) idx[n++]=m;
    if(n){
      Tensor *fr=op_rows(f,idx,n);
      Tensor *gr=op_rows(gate,idx,n);                 /* [n,1] soft value, carries grad */
      Tensor *g=op_mul(op_act(op_linear(fr,L.w1[l]),ACT_SILU),op_linear(fr,L.w3[l]));
      Tensor *o=op_linear(g,L.w2[l]);                 /* [n,D] */
      /* scale each surviving row by its own gate so the gate is trainable */
      Tensor *gb=op_matmul(gr,op_ones_row(D));        /* [n,1]x[1,D] -> [n,D] broadcast */
      x=op_add(x,op_scatter(op_mul(o,gb),idx,n,M));
    }
  }
  return op_linear(op_rmsnorm(x,L.fno),L.outw);
}

/* ================================================================= *
 *  A8: lowrankffn -- FFN up-projections factorised through a rank-r
 *  bottleneck shared between the two SwiGLU branches: w1 = A1 @ S, w3 = A3 @ S
 *  with S[r,D] shared.  Moves down-left on BOTH grid axes at once.
 * ================================================================= */
typedef struct { Tensor *S[MAXL],*A1[MAXL],*A3[MAXL]; int r; } LowR;
static LowR LR;
static void lr_build(Cfg*c){
  int D=c->dim,H=c->hidden_dim,V=c->vocab,hd=D/c->n_heads,kvd=c->n_kv_heads*hd; char b[48];
  LR.r=cfg_geti(c,"rank",24);
  L.emb=P_new("emb",2,V,D,0,0); params_init_normal(L.emb,0.02f);
  for(int l=0;l<c->n_layers;l++){
    nm(b,"an",l); L.an[l]=P_new(b,1,D,0,0,0); params_init_ones(L.an[l]);
    nm(b,"wq",l); L.wq[l]=P_new(b,2,D,D,0,0);   params_init_normal(L.wq[l],0.02f);
    nm(b,"wk",l); L.wk[l]=P_new(b,2,kvd,D,0,0); params_init_normal(L.wk[l],0.02f);
    nm(b,"wv",l); L.wv[l]=P_new(b,2,kvd,D,0,0); params_init_normal(L.wv[l],0.02f);
    nm(b,"wo",l); L.wo[l]=P_new(b,2,D,D,0,0);   params_init_normal(L.wo[l],0.02f);
    nm(b,"fn",l); L.fn[l]=P_new(b,1,D,0,0,0); params_init_ones(L.fn[l]);
    nm(b,"S", l); LR.S[l] =P_new(b,2,LR.r,D,0,0);  params_init_normal(LR.S[l],0.05f);
    nm(b,"A1",l); LR.A1[l]=P_new(b,2,H,LR.r,0,0);  params_init_normal(LR.A1[l],0.05f);
    nm(b,"A3",l); LR.A3[l]=P_new(b,2,H,LR.r,0,0);  params_init_normal(LR.A3[l],0.05f);
    nm(b,"w2",l); L.w2[l]=P_new(b,2,D,H,0,0); params_init_normal(L.w2[l],0.02f);
  }
  L.fno=P_new("fno",1,D,0,0,0); params_init_ones(L.fno);
  if(c->tie) L.outw=L.emb; else { L.outw=P_new("outw",2,V,D,0,0); params_init_normal(L.outw,0.02f); }
}
static Tensor *lr_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads;
  Tensor *x=op_emb(L.emb,tok,B*T);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_rmsnorm(x,L.an[l]);
    Tensor *q=op_linear(h,L.wq[l]),*k=op_linear(h,L.wk[l]),*v=op_linear(h,L.wv[l]);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,v,B,T,c->n_heads,c->n_kv_heads,hd,uc?&L.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,L.wo[l]));
    Tensor *f=op_rmsnorm(x,L.fn[l]);
    Tensor *s=op_linear(f,LR.S[l]);                      /* [M,r] shared bottleneck */
    Tensor *g=op_mul(op_act(op_linear(s,LR.A1[l]),ACT_SILU),op_linear(s,LR.A3[l]));
    x=op_add(x,op_linear(g,L.w2[l]));
  }
  return op_linear(op_rmsnorm(x,L.fno),L.outw);
}


/* ================================================================= *
 *  A9: multitok -- K tokens emitted per FULL network evaluation.
 *  The trunk is the baseline llama.  K light head-adapters (D x D each) feed a
 *  single shared unembedding, so head j predicts the token at offset j+1.
 *  At decode time all K are EMITTED -- there is no verification, no draft
 *  model, no speculative rollback.  Tokens 2..K are produced without having
 *  seen tokens 1..K-1, and we simply pay for that in quality.
 *  This is the only mechanism available that divides bytes-per-token by K
 *  without touching the network at all.
 *  val_bpb is scored under exactly this emission process (see ce_multi in
 *  main.c): the trunk runs only at positions t == 0 mod K and every token is
 *  scored by the head that would actually have produced it.
 * ================================================================= */
typedef struct { Tensor *ad[8]; int K; } Multi;
static Multi MT;
static void mt_build(Cfg*c){
  llama_build(c);
  MT.K=cfg_geti(c,"kout",2); if(MT.K>8) MT.K=8;
  char b[48];
  for(int j=1;j<MT.K;j++){ sprintf(b,"ad.%d",j);
    MT.ad[j]=P_new(b,2,c->dim,c->dim,0,0);
    /* near-identity init: head j starts as a copy of head 0 */
    for(int r=0;r<c->dim;r++) for(int q=0;q<c->dim;q++)
      MT.ad[j]->d[(size_t)r*c->dim+q]=(r==q?1.0f:0.0f)+rnd_normal()*0.02f;
  }
}
static Tensor *mt_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads;
  Tensor *x=op_emb(L.emb,tok,B*T);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_rmsnorm(x,L.an[l]);
    Tensor *q=op_linear(h,L.wq[l]),*k=op_linear(h,L.wk[l]),*v=op_linear(h,L.wv[l]);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,v,B,T,c->n_heads,c->n_kv_heads,hd,uc?&L.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,L.wo[l]));
    Tensor *f=op_rmsnorm(x,L.fn[l]);
    Tensor *g=op_mul(op_act(op_linear(f,L.w1[l]),ACT_SILU),op_linear(f,L.w3[l]));
    x=op_add(x,op_linear(g,L.w2[l]));
  }
  x=op_rmsnorm(x,L.fno);
  Tensor *out=op_linear(x,L.outw);                 /* head 0 */
  for(int j=1;j<MT.K;j++)
    out=op_concat(out,op_linear(op_linear(x,MT.ad[j]),L.outw));
  return out;                                       /* [M, K*V] */
}


/* ================================================================= *
 *  A10: quant -- one architecture, three independently quantisable regions
 *  (FFN / attention / embedding-and-unembedding).  Same parameter COUNT as the
 *  baseline in every configuration; only bytes-per-token moves.  This is the
 *  cheapest way to sweep the far-left columns of the grid, which no change of
 *  topology can reach.
 *  Knobs: qffn, qattn, qemb = quantiser levels (0 = keep fp32, 1 = ternary,
 *  7 = 4-bit, 127 = 8-bit).
 * ================================================================= */
static int QF,QA,QE;
static void q_build(Cfg*c){
  llama_build(c);
  QF=cfg_geti(c,"qffn",1); QA=cfg_geti(c,"qattn",0); QE=cfg_geti(c,"qemb",0);
  for(int l=0;l<c->n_layers;l++){
    if(QF){ float b=qrow_bpe(QF); L.w1[l]->bpe=b; L.w2[l]->bpe=b; L.w3[l]->bpe=b; }
    if(QA){ float b=qrow_bpe(QA); L.wq[l]->bpe=b; L.wk[l]->bpe=b; L.wv[l]->bpe=b; L.wo[l]->bpe=b; }
  }
  if(QE){ L.emb->bpe=qrow_bpe(QE); if(L.outw!=L.emb) L.outw->bpe=qrow_bpe(QE); }
}
#define QW(t,lv) ((lv)? op_qrow((t),(lv)) : (t))
static Tensor *q_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads;
  Tensor *embw = QE? op_qrow(L.emb,QE) : L.emb;
  Tensor *x=op_emb(embw,tok,B*T);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_rmsnorm(x,L.an[l]);
    Tensor *q=op_linear(h,QW(L.wq[l],QA)),*k=op_linear(h,QW(L.wk[l],QA)),*v=op_linear(h,QW(L.wv[l],QA));
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,v,B,T,c->n_heads,c->n_kv_heads,hd,uc?&L.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,QW(L.wo[l],QA)));
    Tensor *f=op_rmsnorm(x,L.fn[l]);
    Tensor *g=op_mul(op_act(op_linear(f,QW(L.w1[l],QF)),ACT_SILU),op_linear(f,QW(L.w3[l],QF)));
    x=op_add(x,op_linear(g,QW(L.w2[l],QF)));
  }
  Tensor *ow = (L.outw==L.emb)? embw : (QE? op_qrow(L.outw,QE) : L.outw);
  return op_linear(op_rmsnorm(x,L.fno),ow);
}

/* ================================================================= *
 *  A11: kvshare -- keys and values are computed ONCE at layer 0 and reused by
 *  every layer; only the queries are per-layer.  Deletes 4/5 of the K/V
 *  projections and 4/5 of the KV cache.  Probes whether depth needs to
 *  re-derive what to retrieve, or only how to ask.
 * ================================================================= */
static void kvs_build(Cfg*c){
  int D=c->dim,H=c->hidden_dim,V=c->vocab,hd=D/c->n_heads,kvd=c->n_kv_heads*hd; char b[48];
  L.emb=P_new("emb",2,V,D,0,0); params_init_normal(L.emb,0.02f);
  for(int l=0;l<c->n_layers;l++){
    nm(b,"an",l); L.an[l]=P_new(b,1,D,0,0,0); params_init_ones(L.an[l]);
    nm(b,"wq",l); L.wq[l]=P_new(b,2,D,D,0,0);   params_init_normal(L.wq[l],0.02f);
    if(l==0){ nm(b,"wk",l); L.wk[l]=P_new(b,2,kvd,D,0,0); params_init_normal(L.wk[l],0.02f);
              nm(b,"wv",l); L.wv[l]=P_new(b,2,kvd,D,0,0); params_init_normal(L.wv[l],0.02f); }
    else { L.wk[l]=NULL; L.wv[l]=NULL; }
    nm(b,"wo",l); L.wo[l]=P_new(b,2,D,D,0,0);   params_init_normal(L.wo[l],0.02f);
    nm(b,"fn",l); L.fn[l]=P_new(b,1,D,0,0,0); params_init_ones(L.fn[l]);
    nm(b,"w1",l); L.w1[l]=P_new(b,2,H,D,0,0); params_init_normal(L.w1[l],0.02f);
    nm(b,"w3",l); L.w3[l]=P_new(b,2,H,D,0,0); params_init_normal(L.w3[l],0.02f);
    nm(b,"w2",l); L.w2[l]=P_new(b,2,D,H,0,0); params_init_normal(L.w2[l],0.02f);
  }
  L.fno=P_new("fno",1,D,0,0,0); params_init_ones(L.fno);
  if(c->tie) L.outw=L.emb; else { L.outw=P_new("outw",2,V,D,0,0); params_init_normal(L.outw,0.02f); }
}
static void kvs_cache_alloc(Cfg*c,int ml){
  int hd=c->dim/c->n_heads,kvd=c->n_kv_heads*hd;
  L.kv[0].k=(float*)calloc((size_t)ml*kvd,4); L.kv[0].v=(float*)calloc((size_t)ml*kvd,4);
  L.kv[0].cap=ml; L.kv[0].len=0; L.kv[0].nkv=c->n_kv_heads; L.kv[0].hd=hd;
}
static void kvs_cache_reset(Cfg*c){ (void)c; L.kv[0].len=0; }
static Tensor *kvs_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads;
  Tensor *x=op_emb(L.emb,tok,B*T);
  Tensor *k0=NULL,*v0=NULL; int kvlen_at_entry = uc? L.kv[0].len : 0;
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_rmsnorm(x,L.an[l]);
    Tensor *q=op_linear(h,L.wq[l]);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    if(l==0){
      k0=op_rope(op_linear(h,L.wk[0]),B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
      v0=op_linear(h,L.wv[0]);
    }
    /* layers >0 re-attend over the SAME cache without appending to it */
    if(uc && l>0) L.kv[0].len = kvlen_at_entry;
    Tensor *a=op_attn(q,k0,v0,B,T,c->n_heads,c->n_kv_heads,hd,uc?&L.kv[0]:NULL,0.f);
    x=op_add(x,op_linear(a,L.wo[l]));
    Tensor *f=op_rmsnorm(x,L.fn[l]);
    Tensor *g=op_mul(op_act(op_linear(f,L.w1[l]),ACT_SILU),op_linear(f,L.w3[l]));
    x=op_add(x,op_linear(g,L.w2[l]));
  }
  return op_linear(op_rmsnorm(x,L.fno),L.outw);
}

/* ================================================================= *
 *  A12: loopexpert -- RECOMBINATION of A1 and A2.  One shared attention block
 *  iterated L times (tiny unique read), but the FFN at every iteration is
 *  chosen from K bigram-hash-routed experts (large parameter count).  The
 *  route depends only on the token pair, so the SAME expert is used at every
 *  depth: unique bytes per token stay at one expert's worth no matter how deep
 *  the loop goes.
 * ================================================================= */
typedef struct {
  Tensor *emb,*outw,*fno,*wq,*wk,*wv,*wo;
  Tensor *an[MAXL],*ab[MAXL],*fn[MAXL],*fb[MAXL];
  Tensor *w1[MAXEXP],*w2[MAXEXP],*w3[MAXEXP];
  KVCache kv[MAXL];
  int K,last_tok;
} LoopX;
static LoopX X;
static void lx_build(Cfg*c){
  int D=c->dim,H=c->hidden_dim,V=c->vocab,hd=D/c->n_heads,kvd=c->n_kv_heads*hd; char b[48];
  X.K=cfg_geti(c,"experts",4); if(X.K>MAXEXP) X.K=MAXEXP; X.last_tok=256;
  X.emb=P_new("emb",2,V,D,0,0); params_init_normal(X.emb,0.02f);
  X.wq=P_new("wq",2,D,D,0,0);   params_init_normal(X.wq,0.02f);
  X.wk=P_new("wk",2,kvd,D,0,0); params_init_normal(X.wk,0.02f);
  X.wv=P_new("wv",2,kvd,D,0,0); params_init_normal(X.wv,0.02f);
  X.wo=P_new("wo",2,D,D,0,0);   params_init_normal(X.wo,0.02f);
  for(int e=0;e<X.K;e++){ char q[48];
    sprintf(q,"w1.%d",e); X.w1[e]=P_new(q,2,H,D,0,0); params_init_normal(X.w1[e],0.02f);
    sprintf(q,"w3.%d",e); X.w3[e]=P_new(q,2,H,D,0,0); params_init_normal(X.w3[e],0.02f);
    sprintf(q,"w2.%d",e); X.w2[e]=P_new(q,2,D,H,0,0); params_init_normal(X.w2[e],0.02f); }
  for(int l=0;l<c->n_layers;l++){
    nm(b,"an",l); X.an[l]=P_new(b,1,D,0,0,0); params_init_ones(X.an[l]);
    nm(b,"ab",l); X.ab[l]=P_new(b,1,D,0,0,0); params_init_zeros(X.ab[l]);
    nm(b,"fn",l); X.fn[l]=P_new(b,1,D,0,0,0); params_init_ones(X.fn[l]);
    nm(b,"fb",l); X.fb[l]=P_new(b,1,D,0,0,0); params_init_zeros(X.fb[l]);
  }
  X.fno=P_new("fno",1,D,0,0,0); params_init_ones(X.fno);
  if(c->tie) X.outw=X.emb; else { X.outw=P_new("outw",2,V,D,0,0); params_init_normal(X.outw,0.02f); }
}
static void lx_cache_alloc(Cfg*c,int ml){
  int hd=c->dim/c->n_heads,kvd=c->n_kv_heads*hd;
  for(int l=0;l<c->n_layers;l++){ X.kv[l].k=(float*)calloc((size_t)ml*kvd,4); X.kv[l].v=(float*)calloc((size_t)ml*kvd,4);
    X.kv[l].cap=ml; X.kv[l].len=0; X.kv[l].nkv=c->n_kv_heads; X.kv[l].hd=hd; }
}
static void lx_cache_reset(Cfg*c){ for(int l=0;l<c->n_layers;l++) X.kv[l].len=0; X.last_tok=256; }
static Tensor *lx_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads,M=B*T,K=X.K;
  static int rt[1<<16]; static int idx[MAXEXP][1<<16]; int cnt[MAXEXP];
  for(int b=0;b<B;b++)for(int t=0;t<T;t++){ int m=b*T+t;
    int prev=(t>0)?tok[m-1]:(uc?X.last_tok:256); rt[m]=bigram_route(tok[m],prev,K); }
  for(int e=0;e<K;e++) cnt[e]=0;
  for(int m=0;m<M;m++) idx[rt[m]][cnt[rt[m]]++]=m;
  if(uc) X.last_tok=tok[T-1];
  Tensor *x=op_emb(X.emb,tok,M);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_addbias(op_rmsnorm(x,X.an[l]),X.ab[l]);
    Tensor *q=op_linear(h,X.wq),*k=op_linear(h,X.wk),*v=op_linear(h,X.wv);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,v,B,T,c->n_heads,c->n_kv_heads,hd,uc?&X.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,X.wo));
    Tensor *f=op_addbias(op_rmsnorm(x,X.fn[l]),X.fb[l]);
    Tensor *acc=NULL;
    for(int e=0;e<K;e++){ if(!cnt[e]) continue;
      Tensor *fe=op_rows(f,idx[e],cnt[e]);
      Tensor *g=op_mul(op_act(op_linear(fe,X.w1[e]),ACT_SILU),op_linear(fe,X.w3[e]));
      Tensor *oe=op_scatter(op_linear(g,X.w2[e]),idx[e],cnt[e],M);
      acc=acc?op_add(acc,oe):oe; }
    x=op_add(x,acc);
  }
  return op_linear(op_rmsnorm(x,X.fno),X.outw);
}


/* ================================================================= *
 *  A13: cascade -- two models, one emission process.
 *  A tiny bigram-memory model (its own 32-dim embedding, a hashed bigram table
 *  and its own unembedding -- ~33 KB read per byte) proposes every byte.  When
 *  its own max probability exceeds tau it emits and the transformer is not run
 *  at all.  When it hesitates, the transformer runs and emits instead.
 *
 *  The two networks share NO parameters, so their byte costs are disjoint and
 *  can be counted separately without double-billing.
 *
 *  The transformer must still see every byte or its KV cache develops holes, so
 *  when it is finally needed it processes the whole backlog in one batched
 *  catch-up pass -- reading its weights ONCE for the entire run of fast-emitted
 *  bytes.  Hence  bytes/token = fast_bytes + big_bytes / (mean run length).
 *  tau sweeps that trade-off continuously with a single trained model.
 *
 *  The gate depends only on the fast model's own output, which depends only on
 *  the context, so the mixture is a properly normalised conditional
 *  distribution and its bits-per-byte is directly comparable to every other row.
 *
 *  Attacks the assumption shared by every previous variant: that all bytes are
 *  worth the same amount of computation.
 * ================================================================= */
int g_casc_mode = 0;   /* 0 = both heads, 1 = fast only, 2 = full only */
typedef struct {
  Tensor *femb,*fmem,*fout,*fnorm;
  int nslot, fdim, last_tok;
} Casc;
static Casc C2;
static void casc_build(Cfg*c){
  llama_build(c);
  C2.nslot=cfg_geti(c,"nslot",4096);
  C2.fdim =cfg_geti(c,"fdim",32);
  C2.last_tok=256;
  C2.femb =P_new("femb", 2,c->vocab,C2.fdim,0,0); params_init_normal(C2.femb,0.05f);
  C2.fmem =P_new("fmem", 2,C2.nslot,C2.fdim,0,0); params_init_normal(C2.fmem,0.05f);
  C2.fout =P_new("fout", 2,c->vocab,C2.fdim,0,0); params_init_normal(C2.fout,0.05f);
  C2.fnorm=P_new("fnorm",1,C2.fdim,0,0,0);        params_init_ones(C2.fnorm);
}
static void casc_cache_reset(Cfg*c){ llama_cache_reset(c); C2.last_tok=256; }
static Tensor *casc_fast(Cfg*c,int*tok,int B,int T,int uc){
  int M=B*T; static int slot[1<<16];
  for(int b=0;b<B;b++)for(int t=0;t<T;t++){ int m=b*T+t;
    int prev=(t>0)?tok[m-1]:(uc?C2.last_tok:256);
    unsigned h=(unsigned)tok[m]*2654435761u ^ (unsigned)prev*2246822519u; h^=h>>15;
    slot[m]=(int)(h%(unsigned)C2.nslot); }
  Tensor *e=op_emb(C2.femb,tok,M);
  Tensor *g=op_emb(C2.fmem,slot,M);
  Tensor *hf=op_rmsnorm(op_add(e,g),C2.fnorm);
  return op_linear(hf,C2.fout);                   /* [M,V] */
}
static Tensor *casc_full(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  int D=c->dim,hd=D/c->n_heads;
  Tensor *x=op_emb(L.emb,tok,B*T);
  for(int l=0;l<c->n_layers;l++){
    Tensor *h=op_rmsnorm(x,L.an[l]);
    Tensor *q=op_linear(h,L.wq[l]),*k=op_linear(h,L.wk[l]),*v=op_linear(h,L.wv[l]);
    q=op_rope(q,B,T,c->n_heads,hd,pos0,c->rope_theta);
    k=op_rope(k,B,T,c->n_kv_heads,hd,pos0,c->rope_theta);
    Tensor *a=op_attn(q,k,v,B,T,c->n_heads,c->n_kv_heads,hd,uc?&L.kv[l]:NULL,0.f);
    x=op_add(x,op_linear(a,L.wo[l]));
    Tensor *f=op_rmsnorm(x,L.fn[l]);
    Tensor *gg=op_mul(op_act(op_linear(f,L.w1[l]),ACT_SILU),op_linear(f,L.w3[l]));
    x=op_add(x,op_linear(gg,L.w2[l]));
  }
  return op_linear(op_rmsnorm(x,L.fno),L.outw);
}
static Tensor *casc_fwd(Cfg*c,int*tok,int B,int T,int pos0,int uc){
  if(g_casc_mode==1){ Tensor*r=casc_fast(c,tok,B,T,uc); if(uc) C2.last_tok=tok[T-1]; return r; }
  if(g_casc_mode==2){ return casc_full(c,tok,B,T,pos0,uc); }
  Tensor *full=casc_full(c,tok,B,T,pos0,uc);
  Tensor *fast=casc_fast(c,tok,B,T,uc);
  if(uc) C2.last_tok=tok[T-1];
  return op_concat(full,fast);                    /* [M, 2V] */
}

/* ================================================================= *
 *  registry
 * ================================================================= */
Arch g_archs[] = {
  {"llama","baseline llama2: rmsnorm/GQA-rope/SwiGLU, tied emb",
   llama_build, llama_cache_alloc, llama_cache_reset, llama_fwd},
  {"sharedloop","one block iterated L times; only per-depth affine differs",
   loop_build, loop_cache_alloc, loop_cache_reset, loop_fwd},
  {"hashffn","K FFN experts routed by bigram hash of (tok,prev); no router params",
   hash_build, hash_cache_alloc, hash_cache_reset, hash_fwd},
  {"ternffn","ternary FFN weights (2 bit/wt, per-row scale, STE); fp32 attention",
   tern_build, llama_cache_alloc, llama_cache_reset, tern_fwd},
  {"novalue","attention reuses K as V: no value projection at all",
   nov_build, llama_cache_alloc, llama_cache_reset, nov_fwd},
  {"emaconv","attention replaced by depthwise causal conv + per-channel EMA (O(1) state)",
   ema_build, ema_cache_alloc, ema_cache_reset, ema_fwd},
  {"ngrammem","hashed bigram memory table injected at every depth; 1 row read/token",
   ng_build, llama_cache_alloc, ng_cache_reset, ng_fwd},
  {"deepgate","per-token hard FFN skip; data-dependent bytes/token",
   dg_build, llama_cache_alloc, llama_cache_reset, dg_fwd},
  {"lowrankffn","SwiGLU up-projections share a rank-r bottleneck",
   lr_build, llama_cache_alloc, llama_cache_reset, lr_fwd},
  {"multitok","emit K tokens per network evaluation, no verification",
   mt_build, llama_cache_alloc, llama_cache_reset, mt_fwd},
  {"quant","per-region row-quantised weights (knobs qffn/qattn/qemb = levels)",
   q_build, llama_cache_alloc, llama_cache_reset, q_fwd},
  {"kvshare","K and V computed once at layer 0 and reused by every layer",
   kvs_build, kvs_cache_alloc, kvs_cache_reset, kvs_fwd},
  {"loopexpert","shared looped attention block + bigram-routed FFN experts",
   lx_build, lx_cache_alloc, lx_cache_reset, lx_fwd},
  {"cascade","tiny bigram model emits confident bytes; transformer only on demand",
   casc_build, llama_cache_alloc, casc_cache_reset, casc_fwd},
};
int g_narchs = (int)(sizeof(g_archs)/sizeof(g_archs[0]));
Arch *arch_find(const char*n){ for(int i=0;i<g_narchs;i++) if(!strcmp(g_archs[i].name,n)) return &g_archs[i]; return NULL; }
