#include "nn.h"
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int g_train = 1;
long long g_wbytes = 0, g_sbytes = 0, g_flops = 0;
uint64_t g_rng = 88172645463325252ULL;

/* ============================ arena ============================ */
static char  *A_base = NULL;
static size_t A_cap = 0, A_off = 0, A_hi = 0;
void arena_init(size_t bytes){ A_base = (char*)malloc(bytes); if(!A_base){fprintf(stderr,"arena oom\n");exit(1);} A_cap=bytes; A_off=0; }
void arena_reset(void){ if(A_off>A_hi) A_hi=A_off; A_off = 0; }
size_t arena_used(void){ return A_hi>A_off?A_hi:A_off; }
void *arena_alloc(size_t bytes){
  bytes = (bytes + 63) & ~(size_t)63;
  if (A_off + bytes > A_cap){ fprintf(stderr,"arena exhausted (%zu + %zu > %zu)\n",A_off,bytes,A_cap); exit(1); }
  void *p = A_base + A_off; A_off += bytes; return p;
}

/* ============================ tensors ============================ */
Tensor *g_params[MAXPARAM]; int g_nparams = 0;

static Tensor *T_alloc_hdr(int ndim,int a,int b,int c,int dd,int arena){
  Tensor *t = arena ? (Tensor*)arena_alloc(sizeof(Tensor)) : (Tensor*)calloc(1,sizeof(Tensor));
  if(arena) memset(t,0,sizeof(Tensor));
  int sh[4]={a,b,c,dd}; t->ndim=ndim; t->n=1;
  for(int i=0;i<MAXDIM;i++){ t->shape[i]= i<ndim? sh[i]:1; if(i<ndim) t->n*=sh[i]; }
  t->bpe = 4.0f;
  return t;
}
Tensor *T_new(int ndim,int a,int b,int c,int dd){
  Tensor *t = T_alloc_hdr(ndim,a,b,c,dd,1);
  t->d = (float*)arena_alloc((size_t)t->n*sizeof(float));
  if (g_train){ t->g = (float*)arena_alloc((size_t)t->n*sizeof(float)); memset(t->g,0,(size_t)t->n*sizeof(float)); }
  return t;
}
Tensor *T_view(int R,int C,float*data){
  Tensor *t=T_alloc_hdr(2,R,C,0,0,1); t->d=data; t->g=NULL; return t;
}
Tensor *P_new(const char *name,int ndim,int a,int b,int c,int dd){
  Tensor *t = T_alloc_hdr(ndim,a,b,c,dd,0);
  t->d = (float*)calloc(t->n,sizeof(float));
  t->g = (float*)calloc(t->n,sizeof(float));
  t->is_param = 1; snprintf(t->name,sizeof(t->name),"%s",name);
  t->nrow = (ndim>=2)? t->shape[0] : 1; t->rowelem = t->n / t->nrow;
  t->touch = (unsigned char*)calloc(t->nrow,1); t->tany=0; t->tfull=0;
  t->touchw= (unsigned char*)calloc(t->nrow,1); t->wany=0; t->wfull=0;
  if (g_nparams>=MAXPARAM){fprintf(stderr,"too many params\n");exit(1);}
  g_params[g_nparams++] = t; return t;
}
long long wbytes_unique_step(void){
  long long s=0;
  for(int i=0;i<g_nparams;i++){ Tensor*t=g_params[i];
    if(!t->tany) continue;
    long long r=0; for(int k=0;k<t->nrow;k++) if(t->touch[k]) r++;
    s += (long long)(r*(double)t->rowelem*t->bpe);
    memset(t->touch,0,t->nrow); t->tany=0; t->tfull=0; }
  return s;
}
long long wbytes_window_flush(void){
  long long s=0;
  for(int i=0;i<g_nparams;i++){ Tensor*t=g_params[i];
    if(!t->wany) continue;
    long long r=0; for(int k=0;k<t->nrow;k++) if(t->touchw[k]) r++;
    s += (long long)(r*(double)t->rowelem*t->bpe);
    memset(t->touchw,0,t->nrow); t->wany=0; t->wfull=0; }
  return s;
}
long long params_count(void){ long long s=0; for(int i=0;i<g_nparams;i++) s+=g_params[i]->n; return s; }
double params_bytes(void){ double s=0; for(int i=0;i<g_nparams;i++) s+=(double)g_params[i]->n*g_params[i]->bpe; return s; }
float rnd_normal(void){ float u1=rnd_f()+1e-9f,u2=rnd_f(); return sqrtf(-2.0f*logf(u1))*cosf(6.28318530718f*u2); }
void params_init_normal(Tensor*t,float std){ for(int i=0;i<t->n;i++) t->d[i]=rnd_normal()*std; }
void params_init_zeros(Tensor*t){ memset(t->d,0,(size_t)t->n*sizeof(float)); }
void params_init_ones(Tensor*t){ for(int i=0;i<t->n;i++) t->d[i]=1.0f; }

/* ============================ tape ============================ */
enum { O_LINEAR,O_MATMUL,O_EMB,O_RMSNORM,O_ADD,O_MUL,O_BIAS,O_SCALE,O_ACT,O_SOFTMAX,
       O_SLICE,O_CONCAT,O_STE,O_ROPE,O_ATTN,O_CE,O_ROWS,O_SCATTER,O_DWCONV,O_EMA,O_PLACE };
typedef struct {
  int op; Tensor *a,*b,*c,*o;
  int i0,i1,i2,i3,i4,i5;
  float f0;
  void *aux;
} Node;
static Node  g_tape[MAXTAPE]; static int g_ntape=0;
void tape_reset(void){ g_ntape=0; }
static Node *push(int op){ if(g_ntape>=MAXTAPE){fprintf(stderr,"tape full\n");exit(1);} Node*n=&g_tape[g_ntape++]; memset(n,0,sizeof(Node)); n->op=op; return n; }

/* ============================ kernels ============================ */
/* out[M,O] = x[M,I] @ W[O,I]^T */
static void mm_nt(const float* restrict x,const float* restrict W,float* restrict o,int M,int I,int O){
  /* 4-way blocking over output cols: four independent FMA chains hide the
     reduction latency and the activation row is loaded once per 4 outputs. */
  #pragma omp parallel for schedule(static) if(M*(long)I*O > 200000)
  for(int m=0;m<M;m++){
    const float *xr = x + (size_t)m*I; float *orow = o + (size_t)m*O;
    int j=0;
    for(; j+4<=O; j+=4){
      const float *w0=W+(size_t)j*I,*w1=w0+I,*w2=w1+I,*w3=w2+I;
      float s0=0,s1=0,s2=0,s3=0;
      for(int k=0;k<I;k++){ float v=xr[k]; s0+=v*w0[k]; s1+=v*w1[k]; s2+=v*w2[k]; s3+=v*w3[k]; }
      orow[j]=s0; orow[j+1]=s1; orow[j+2]=s2; orow[j+3]=s3;
    }
    for(; j<O; j++){
      const float *w = W + (size_t)j*I; float s=0.f;
      for(int k=0;k<I;k++) s += xr[k]*w[k];
      orow[j]=s;
    }
  }
}
/* out[M,N] = a[M,K] @ b[K,N] */
static void mm_nn(const float* restrict a,const float* restrict b,float* restrict o,int M,int K,int N){
  #pragma omp parallel for schedule(static) if(M*(long)K*N > 200000)
  for(int m=0;m<M;m++){
    float *orow=o+(size_t)m*N; for(int j=0;j<N;j++) orow[j]=0.f;
    for(int k=0;k<K;k++){ float av=a[(size_t)m*K+k]; if(av==0.f) continue; const float*br=b+(size_t)k*N;
      for(int j=0;j<N;j++) orow[j]+=av*br[j]; }
  }
}

/* ============================ ops ============================ */
Tensor *op_linear(Tensor *x, Tensor *W){
  int M=x->shape[0], I=W->shape[1], O=W->shape[0];
  if (x->shape[1]!=I){fprintf(stderr,"linear shape %d vs %d\n",x->shape[1],I);exit(1);}
  Tensor *o=T_new(2,M,O,0,0);
  mm_nt(x->d,W->d,o->d,M,I,O);
  g_flops += (long long)M*I*O;
  count_read(W,(long long)I*O); touch_all(W);
  if(g_train){ Node*n=push(O_LINEAR); n->a=x;n->b=W;n->o=o;n->i0=M;n->i1=I;n->i2=O; }
  return o;
}
Tensor *op_matmul(Tensor *a, Tensor *b){
  int M=a->shape[0],K=a->shape[1],N=b->shape[1];
  if(b->shape[0]!=K){fprintf(stderr,"matmul shape\n");exit(1);}
  Tensor *o=T_new(2,M,N,0,0); mm_nn(a->d,b->d,o->d,M,K,N);
  g_flops += (long long)M*K*N; count_read(b,(long long)K*N); touch_all(b); count_read(a,0);
  if(g_train){ Node*n=push(O_MATMUL); n->a=a;n->b=b;n->o=o;n->i0=M;n->i1=K;n->i2=N; }
  return o;
}
Tensor *op_emb(Tensor *W,int *idx,int M){
  int D=W->shape[1]; Tensor *o=T_new(2,M,D,0,0);
  int *ix=(int*)arena_alloc(sizeof(int)*M); memcpy(ix,idx,sizeof(int)*M);
  for(int m=0;m<M;m++){ memcpy(o->d+(size_t)m*D, W->d+(size_t)ix[m]*D, D*sizeof(float)); touch_row(W,ix[m]); }
  count_read(W,(long long)M*D);
  if(g_train){ Node*n=push(O_EMB); n->b=W;n->o=o;n->i0=M;n->i1=D;n->aux=ix; }
  return o;
}
Tensor *op_gather(Tensor *W,int *idx,int M){ return op_emb(W,idx,M); }

Tensor *op_rmsnorm(Tensor *x,Tensor *w){
  int M=x->shape[0],D=x->shape[1]; Tensor *o=T_new(2,M,D,0,0);
  float *rs=(float*)arena_alloc(sizeof(float)*M);
  for(int m=0;m<M;m++){
    const float*xr=x->d+(size_t)m*D; float ss=0.f; for(int i=0;i<D;i++) ss+=xr[i]*xr[i];
    float r=1.0f/sqrtf(ss/D+1e-5f); rs[m]=r;
    float*orow=o->d+(size_t)m*D; for(int i=0;i<D;i++) orow[i]=xr[i]*r*w->d[i];
  }
  count_read(w,D); touch_all(w);
  if(g_train){ Node*n=push(O_RMSNORM); n->a=x;n->b=w;n->o=o;n->i0=M;n->i1=D;n->aux=rs; }
  return o;
}
Tensor *op_add(Tensor *a,Tensor *b){
  Tensor *o=T_new(2,a->shape[0],a->shape[1],0,0);
  for(int i=0;i<o->n;i++) o->d[i]=a->d[i]+b->d[i];
  if(g_train){ Node*n=push(O_ADD); n->a=a;n->b=b;n->o=o; }
  return o;
}
Tensor *op_mul(Tensor *a,Tensor *b){
  Tensor *o=T_new(2,a->shape[0],a->shape[1],0,0);
  for(int i=0;i<o->n;i++) o->d[i]=a->d[i]*b->d[i];
  if(g_train){ Node*n=push(O_MUL); n->a=a;n->b=b;n->o=o; }
  return o;
}
Tensor *op_addbias(Tensor *x,Tensor *b){
  int M=x->shape[0],D=x->shape[1]; Tensor*o=T_new(2,M,D,0,0);
  for(int m=0;m<M;m++) for(int i=0;i<D;i++) o->d[(size_t)m*D+i]=x->d[(size_t)m*D+i]+b->d[i];
  count_read(b,D); touch_all(b);
  if(g_train){ Node*n=push(O_BIAS); n->a=x;n->b=b;n->o=o;n->i0=M;n->i1=D; }
  return o;
}
Tensor *op_scale(Tensor *x,float s){
  Tensor*o=T_new(2,x->shape[0],x->shape[1],0,0);
  for(int i=0;i<o->n;i++) o->d[i]=x->d[i]*s;
  if(g_train){ Node*n=push(O_SCALE); n->a=x;n->o=o;n->f0=s; }
  return o;
}
static inline float act_f(float v,int k){
  switch(k){
    case ACT_SILU: return v/(1.0f+expf(-v));
    case ACT_GELU: return 0.5f*v*(1.0f+tanhf(0.79788456f*(v+0.044715f*v*v*v)));
    case ACT_RELU: return v>0?v:0;
    case ACT_SIGM: return 1.0f/(1.0f+expf(-v));
    case ACT_TANH: return tanhf(v);
    default:       return v*v;
  }
}
static inline float act_df(float v,int k){
  switch(k){
    case ACT_SILU:{ float s=1.0f/(1.0f+expf(-v)); return s*(1.0f+v*(1.0f-s)); }
    case ACT_GELU:{ float t=tanhf(0.79788456f*(v+0.044715f*v*v*v));
                    return 0.5f*(1.0f+t)+0.5f*v*(1.0f-t*t)*0.79788456f*(1.0f+3.0f*0.044715f*v*v); }
    case ACT_RELU: return v>0?1.0f:0.0f;
    case ACT_SIGM:{ float s=1.0f/(1.0f+expf(-v)); return s*(1.0f-s); }
    case ACT_TANH:{ float t=tanhf(v); return 1.0f-t*t; }
    default:       return 2.0f*v;
  }
}
Tensor *op_act(Tensor *x,int kind){
  Tensor*o=T_new(2,x->shape[0],x->shape[1],0,0);
  for(int i=0;i<o->n;i++) o->d[i]=act_f(x->d[i],kind);
  if(g_train){ Node*n=push(O_ACT); n->a=x;n->o=o;n->i0=kind; }
  return o;
}
Tensor *op_softmax(Tensor *x){
  int M=x->shape[0],D=x->shape[1]; Tensor*o=T_new(2,M,D,0,0);
  for(int m=0;m<M;m++){
    const float*xr=x->d+(size_t)m*D; float*orow=o->d+(size_t)m*D;
    float mx=xr[0]; for(int i=1;i<D;i++) if(xr[i]>mx) mx=xr[i];
    float s=0; for(int i=0;i<D;i++){ orow[i]=expf(xr[i]-mx); s+=orow[i]; }
    float inv=1.0f/s; for(int i=0;i<D;i++) orow[i]*=inv;
  }
  if(g_train){ Node*n=push(O_SOFTMAX); n->a=x;n->o=o;n->i0=M;n->i1=D; }
  return o;
}
Tensor *op_slice(Tensor *x,int off,int len){
  int M=x->shape[0],D=x->shape[1]; Tensor*o=T_new(2,M,len,0,0);
  for(int m=0;m<M;m++) memcpy(o->d+(size_t)m*len, x->d+(size_t)m*D+off, len*sizeof(float));
  if(g_train){ Node*n=push(O_SLICE); n->a=x;n->o=o;n->i0=M;n->i1=D;n->i2=off;n->i3=len; }
  return o;
}
Tensor *op_concat(Tensor *a,Tensor *b){
  int M=a->shape[0],Da=a->shape[1],Db=b->shape[1]; Tensor*o=T_new(2,M,Da+Db,0,0);
  for(int m=0;m<M;m++){ memcpy(o->d+(size_t)m*(Da+Db),a->d+(size_t)m*Da,Da*sizeof(float));
                        memcpy(o->d+(size_t)m*(Da+Db)+Da,b->d+(size_t)m*Db,Db*sizeof(float)); }
  if(g_train){ Node*n=push(O_CONCAT); n->a=a;n->b=b;n->o=o;n->i0=M;n->i1=Da;n->i2=Db; }
  return o;
}
void op_place(Tensor *dst,Tensor *src,int off){
  int M=src->shape[0],W=src->shape[1],DW=dst->shape[1];
  for(int m=0;m<M;m++) memcpy(dst->d+(size_t)m*DW+off, src->d+(size_t)m*W, W*sizeof(float));
  if(g_train){ Node*n=push(O_PLACE); n->a=src;n->o=dst;n->i0=M;n->i1=W;n->i2=off;n->i3=DW; }
}
Tensor *op_ste_quant(Tensor *x,int levels){
  Tensor*o=T_new(2,x->shape[0],x->shape[1],0,0);
  float mx=0; for(int i=0;i<x->n;i++){ float a=fabsf(x->d[i]); if(a>mx) mx=a; }
  if(mx<1e-8f) mx=1e-8f;
  float step=mx/levels;
  for(int i=0;i<o->n;i++){ float q=roundf(x->d[i]/step)*step; o->d[i]=q; }
  if(g_train){ Node*n=push(O_STE); n->a=x;n->o=o; }
  return o;
}
/* rope cos/sin table: [pos][hd/2] pairs, built once per (hd,theta) */
#define ROPE_MAXPOS 4096
static float *g_rope_cs=NULL; static int g_rope_hd=-1,g_rope_np=0; static float g_rope_th=-1;
static const float *rope_table(int hd,float theta,int need){
  if(g_rope_hd==hd && g_rope_th==theta && g_rope_np>=need) return g_rope_cs;
  int np = need>ROPE_MAXPOS?need:ROPE_MAXPOS;
  g_rope_cs=(float*)realloc(g_rope_cs,(size_t)np*hd*sizeof(float)); /* [pos][hd/2][2] */
  for(int p=0;p<np;p++) for(int i=0;i<hd;i+=2){
    float freq=1.0f/powf(theta,(float)i/(float)hd);
    g_rope_cs[(size_t)p*hd+i  ]=cosf(p*freq);
    g_rope_cs[(size_t)p*hd+i+1]=sinf(p*freq);
  }
  g_rope_hd=hd; g_rope_th=theta; g_rope_np=np; return g_rope_cs;
}
/* --- per-row symmetric weight quantisation with straight-through estimator ---
   levels==1 : {-s,0,s} with s = mean|w_row|   (absmax would zero most weights)
   levels>1  : 2*levels+1 uniform steps with s = max|w_row| / levels
   At inference the quantised copy is materialised once and then VIEWED, so the
   decode benchmark does not pay a re-quantisation cost the deployed model
   would not pay either.  I/O is billed to the source parameter at its bpe. */
float qrow_bpe(int levels){
  int states=2*levels+1, bits=1; while((1<<bits) < states) bits++;
  return bits/8.0f;
}
static void qrow_fill(const float*w,float*q,int R,int C,int levels){
  for(int r=0;r<R;r++){ const float*wr=w+(size_t)r*C; float*qr=q+(size_t)r*C; float sc;
    if(levels==1){ double s=0; for(int i=0;i<C;i++) s+=fabs(wr[i]); sc=(float)(s/C)+1e-9f; }
    else { float mx=0; for(int i=0;i<C;i++){ float a=fabsf(wr[i]); if(a>mx) mx=a; } sc=mx/levels+1e-9f; }
    for(int i=0;i<C;i++){ float v=wr[i]/sc; if(v>levels) v=levels; if(v<-levels) v=-levels;
      qr[i]=roundf(v)*sc; } }
}
Tensor *op_qrow(Tensor *W,int levels){
  int R=W->shape[0],C=W->shape[1];
  if(!g_train){
    if(!W->qvalid){
      if(!W->qbuf) W->qbuf=(float*)malloc((size_t)W->n*sizeof(float));
      qrow_fill(W->d,W->qbuf,R,C,levels); W->qvalid=1;
    }
    Tensor *o=T_view(R,C,W->qbuf); o->src_param=W; return o;
  }
  Tensor *o=T_new(2,R,C,0,0);
  qrow_fill(W->d,o->d,R,C,levels);
  o->src_param=W;
  Node*n=push(O_STE); n->a=W;n->o=o;
  return o;
}
Tensor *op_ternary(Tensor *W){ return op_qrow(W,1); }

/* --- activation row gather / scatter: the substrate for conditional compute --- */
Tensor *op_rows(Tensor *x,int *idx,int n){
  int D=x->shape[1]; Tensor*o=T_new(2,n,D,0,0);
  int *ix=(int*)arena_alloc(sizeof(int)*(n?n:1)); memcpy(ix,idx,sizeof(int)*n);
  for(int i=0;i<n;i++) memcpy(o->d+(size_t)i*D, x->d+(size_t)ix[i]*D, D*sizeof(float));
  if(g_train){ Node*nd=push(O_ROWS); nd->a=x;nd->o=o;nd->i0=n;nd->i1=D;nd->aux=ix; }
  return o;
}
Tensor *op_scatter(Tensor *src,int *idx,int n,int M){
  int D=src->shape[1]; Tensor*o=T_new(2,M,D,0,0);
  memset(o->d,0,(size_t)M*D*sizeof(float));
  int *ix=(int*)arena_alloc(sizeof(int)*(n?n:1)); memcpy(ix,idx,sizeof(int)*n);
  for(int i=0;i<n;i++) memcpy(o->d+(size_t)ix[i]*D, src->d+(size_t)i*D, D*sizeof(float));
  if(g_train){ Node*nd=push(O_SCATTER); nd->a=src;nd->o=o;nd->i0=n;nd->i1=D;nd->aux=ix; }
  return o;
}
Tensor *op_dwconv(Tensor *x,Tensor *kern,int B,int T,int W,float *hist){
  int M=x->shape[0],D=x->shape[1]; Tensor*o=T_new(2,M,D,0,0);
  count_read(kern,(long long)D*W); touch_all(kern);
  if(hist){ /* streaming: B=T=1, hist holds the previous W-1 inputs, newest first */
    for(int d=0;d<D;d++){ float s=kern->d[(size_t)d*W]*x->d[d];
      for(int j=1;j<W;j++) s+=kern->d[(size_t)d*W+j]*hist[(size_t)(j-1)*D+d];
      o->d[d]=s; }
    for(int j=W-2;j>=1;j--) memcpy(hist+(size_t)j*D,hist+(size_t)(j-1)*D,D*sizeof(float));
    if(W>1) memcpy(hist,x->d,D*sizeof(float));
    g_sbytes += (long long)(W-1)*D*4;
    return o;
  }
  for(int b=0;b<B;b++)for(int t=0;t<T;t++){ float*orow=o->d+(size_t)(b*T+t)*D;
    for(int d=0;d<D;d++){ float s=0;
      for(int j=0;j<W;j++){ int tt=t-j; if(tt<0) break; s+=kern->d[(size_t)d*W+j]*x->d[(size_t)(b*T+tt)*D+d]; }
      orow[d]=s; } }
  if(g_train){ Node*n=push(O_DWCONV); n->a=x;n->b=kern;n->o=o;n->i0=B;n->i1=T;n->i2=W;n->i3=D; }
  return o;
}
Tensor *op_ema(Tensor *x,Tensor *a,int B,int T,float *st){
  int M=x->shape[0],D=x->shape[1]; Tensor*o=T_new(2,M,D,0,0);
  count_read(a,D); touch_all(a);
  if(st){
    for(int d=0;d<D;d++){ float al=1.0f/(1.0f+expf(-a->d[d]));
      st[d]=al*st[d]+(1.0f-al)*x->d[d]; o->d[d]=st[d]; }
    g_sbytes += (long long)D*4;
    return o;
  }
  for(int b=0;b<B;b++){
    for(int d=0;d<D;d++){ float al=1.0f/(1.0f+expf(-a->d[d])); float s=0;
      for(int t=0;t<T;t++){ s=al*s+(1.0f-al)*x->d[(size_t)(b*T+t)*D+d]; o->d[(size_t)(b*T+t)*D+d]=s; } }
  }
  if(g_train){ Node*n=push(O_EMA); n->a=x;n->b=a;n->o=o;n->i0=B;n->i1=T;n->i2=D; }
  return o;
}
Tensor *op_rope(Tensor *x,int B,int T,int nh,int hd,int pos0,float theta){
  int M=x->shape[0],D=x->shape[1]; Tensor*o=T_new(2,M,D,0,0);
  const float *cs=rope_table(hd,theta,pos0+T+1);
  memcpy(o->d,x->d,(size_t)M*D*sizeof(float));
  for(int b=0;b<B;b++)for(int t=0;t<T;t++){
    const float *cst=cs+(size_t)(pos0+t)*hd; float*row=o->d+(size_t)(b*T+t)*D;
    for(int h=0;h<nh;h++){ float*q=row+h*hd;
      for(int i=0;i<hd;i+=2){
        float c=cst[i],s=cst[i+1];
        float a=q[i],bb=q[i+1]; q[i]=a*c-bb*s; q[i+1]=a*s+bb*c;
      }
    }
  }
  if(g_train){ Node*n=push(O_ROPE); n->a=x;n->o=o;n->i0=B;n->i1=T;n->i2=nh;n->i3=hd;n->i4=pos0;n->f0=theta; }
  return o;
}

/* fused causal multi-head attention with GQA and optional streaming cache */
typedef struct { float *att; } AttnAux;
Tensor *op_attn(Tensor *q,Tensor *k,Tensor *v,int B,int T,int nh,int nkv,int hd,
                KVCache *cache,float softcap){
  int M=q->shape[0], D=nh*hd, rep=nh/nkv;
  Tensor *o=T_new(2,M,D,0,0);
  float scale=1.0f/sqrtf((float)hd);
  if (cache){
    /* streaming: B==1, T==1 */
    int L=cache->len;
    memcpy(cache->k+(size_t)L*nkv*hd, k->d, (size_t)nkv*hd*sizeof(float));
    memcpy(cache->v+(size_t)L*nkv*hd, v->d, (size_t)nkv*hd*sizeof(float));
    cache->len = L+1; int S=cache->len;
    float *att=(float*)arena_alloc(sizeof(float)*S);
    for(int h=0;h<nh;h++){
      int kh=h/rep; const float *qh=q->d+h*hd; float mx=-1e30f;
      for(int s=0;s<S;s++){
        const float*kk=cache->k+(size_t)s*nkv*hd+kh*hd; float sc=0;
        for(int i=0;i<hd;i++) sc+=qh[i]*kk[i];
        sc*=scale; if(softcap>0) sc=softcap*tanhf(sc/softcap);
        att[s]=sc; if(sc>mx) mx=sc;
      }
      float ss=0; for(int s=0;s<S;s++){ att[s]=expf(att[s]-mx); ss+=att[s]; }
      float inv=1.0f/ss; float *oh=o->d+h*hd; for(int i=0;i<hd;i++) oh[i]=0;
      for(int s=0;s<S;s++){ float a=att[s]*inv; const float*vv=cache->v+(size_t)s*nkv*hd+kh*hd;
        for(int i=0;i<hd;i++) oh[i]+=a*vv[i]; }
    }
    g_sbytes += (long long)2*S*nkv*hd*4;
    g_flops  += (long long)2*S*nh*hd;
    return o;
  }
  /* full sequence */
  float *att = (float*)arena_alloc((size_t)B*nh*T*T*sizeof(float));
  #pragma omp parallel for collapse(2) schedule(static)
  for(int b=0;b<B;b++) for(int h=0;h<nh;h++){
    int kh=h/rep; float *A=att+(((size_t)b*nh+h)*T)*T;
    for(int t=0;t<T;t++){
      const float*qh=q->d+(size_t)(b*T+t)*D+h*hd; float mx=-1e30f; float*Ar=A+(size_t)t*T;
      for(int s=0;s<=t;s++){
        const float*kk=k->d+(size_t)(b*T+s)*(nkv*hd)+kh*hd; float sc=0;
        for(int i=0;i<hd;i++) sc+=qh[i]*kk[i];
        sc*=scale; if(softcap>0) sc=softcap*tanhf(sc/softcap);
        Ar[s]=sc; if(sc>mx) mx=sc;
      }
      float ss=0; for(int s=0;s<=t;s++){ Ar[s]=expf(Ar[s]-mx); ss+=Ar[s]; }
      float inv=1.0f/ss; for(int s=0;s<=t;s++) Ar[s]*=inv; for(int s=t+1;s<T;s++) Ar[s]=0;
      float*oh=o->d+(size_t)(b*T+t)*D+h*hd; for(int i=0;i<hd;i++) oh[i]=0;
      for(int s=0;s<=t;s++){ float a=Ar[s]; const float*vv=v->d+(size_t)(b*T+s)*(nkv*hd)+kh*hd;
        for(int i=0;i<hd;i++) oh[i]+=a*vv[i]; }
    }
  }
  g_flops += (long long)B*nh*T*T*hd;
  if(g_train){ Node*n=push(O_ATTN); n->a=q;n->b=k;n->o=o;
    n->i0=B;n->i1=T;n->i2=nh;n->i3=nkv;n->i4=hd;n->f0=softcap;
    AttnAux *ax=(AttnAux*)arena_alloc(sizeof(AttnAux)); ax->att=att; n->aux=ax;
    n->c=v;
  }
  return o;
}

double op_ce_loss(Tensor *logits,int *targets,int M){
  int V=logits->shape[1]; double loss=0;
  int *tg=(int*)arena_alloc(sizeof(int)*M); memcpy(tg,targets,sizeof(int)*M);
  float *probs = (float*)arena_alloc((size_t)M*V*sizeof(float));
  for(int m=0;m<M;m++){
    const float*x=logits->d+(size_t)m*V; float*p=probs+(size_t)m*V;
    float mx=x[0]; for(int i=1;i<V;i++) if(x[i]>mx) mx=x[i];
    double s=0; for(int i=0;i<V;i++){ double e=exp((double)x[i]-mx); p[i]=(float)e; s+=e; }
    double inv=1.0/s; for(int i=0;i<V;i++) p[i]=(float)(p[i]*inv);
    loss += -log((double)exp((double)x[tg[m]]-mx)*inv + 1e-300);
  }
  loss/=M;
  if(g_train){
    for(int m=0;m<M;m++){ float*p=probs+(size_t)m*V; float*gr=logits->g+(size_t)m*V;
      for(int i=0;i<V;i++) gr[i]+= (p[i] - (i==tg[m]?1.0f:0.0f))/M; }
  }
  return loss;
}

/* reusable scratch for backward accumulation */
static float *WS=NULL; static size_t WSN=0;
static float *wsbuf(size_t n){ if(n>WSN){ WS=(float*)realloc(WS,n*sizeof(float)); WSN=n; if(!WS){fprintf(stderr,"ws oom\n");exit(1);} } return WS; }

/* ---- profiling ---- */
double g_prof[24]; long long g_profn[24]; int g_prof_on=0;
static const char* g_opname[]={"LINEAR","MATMUL","EMB","RMSNORM","ADD","MUL","BIAS","SCALE","ACT","SOFTMAX","SLICE","CONCAT","STE","ROPE","ATTN","CE","ROWS","SCATTER","DWCONV","EMA","PLACE"};
void prof_dump(const char*tag){
  if(!g_prof_on) return; double tot=0; for(int i=0;i<16;i++) tot+=g_prof[i];
  fprintf(stderr,"[prof %s] total %.3fs\n",tag,tot);
  for(int i=0;i<21;i++) if(g_prof[i]>1e-4) fprintf(stderr,"   %-8s %7.3fs (%4.1f%%) n=%lld\n",g_opname[i],g_prof[i],100*g_prof[i]/tot,g_profn[i]);
}

/* ============================ backward ============================ */
void tape_backward(void){
  for(int ni=g_ntape-1; ni>=0; ni--){
    Node *n=&g_tape[ni];
    double _t0 = g_prof_on ? now_sec() : 0;
    switch(n->op){
      case O_LINEAR:{
        int M=n->i0,I=n->i1,O=n->i2; Tensor*x=n->a,*W=n->b,*o=n->o;
        /* dx = go @ W  (row-parallel, race-free) */
        #pragma omp parallel for schedule(static) if(M*(long)I*O>200000)
        for(int m=0;m<M;m++){ const float*go=o->g+(size_t)m*O; float*gx=x->g+(size_t)m*I;
          int j=0;
          for(; j+4<=O; j+=4){ float g0=go[j],g1=go[j+1],g2=go[j+2],g3=go[j+3];
            const float*w0=W->d+(size_t)j*I,*w1=w0+I,*w2=w1+I,*w3=w2+I;
            for(int kk=0;kk<I;kk++) gx[kk]+=g0*w0[kk]+g1*w1[kk]+g2*w2[kk]+g3*w3[kk]; }
          for(; j<O; j++){ float gv=go[j]; if(gv==0.f) continue; const float*w=W->d+(size_t)j*I;
            for(int kk=0;kk<I;kk++) gx[kk]+=gv*w[kk]; } }
        /* dW = go^T @ x, accumulated into per-thread private [O,I] tiles so each
           activation row is streamed exactly once instead of O times. */
        int NT=1;
#ifdef _OPENMP
        NT=omp_get_max_threads();
#endif
        if((long)M*I*O < 100000) NT=1;
        float *acc = wsbuf((size_t)NT*I*O);
        memset(acc,0,(size_t)NT*I*O*sizeof(float));
        #pragma omp parallel num_threads(NT)
        {
          int tid=0,nth=1;
#ifdef _OPENMP
          tid=omp_get_thread_num(); nth=omp_get_num_threads();
#endif
          float *A=acc+(size_t)tid*I*O;
          for(int m=tid;m<M;m+=nth){
            const float*go=o->g+(size_t)m*O; const float*xr=x->d+(size_t)m*I;
            for(int j=0;j<O;j++){ float gv=go[j]; if(gv==0.f) continue; float*a=A+(size_t)j*I;
              for(int kk=0;kk<I;kk++) a[kk]+=gv*xr[kk]; }
          }
        }
        for(int t=0;t<NT;t++){ const float*A=acc+(size_t)t*I*O;
          for(long q=0;q<(long)I*O;q++) W->g[q]+=A[q]; }
      } break;
      case O_MATMUL:{
        int M=n->i0,K=n->i1,N=n->i2; Tensor*a=n->a,*b=n->b,*o=n->o;
        for(int m=0;m<M;m++)for(int k=0;k<K;k++){ float s=0; const float*go=o->g+(size_t)m*N; const float*br=b->d+(size_t)k*N;
          for(int j=0;j<N;j++) s+=go[j]*br[j]; a->g[(size_t)m*K+k]+=s; }
        for(int k=0;k<K;k++)for(int j=0;j<N;j++){ float s=0;
          for(int m=0;m<M;m++) s+=a->d[(size_t)m*K+k]*o->g[(size_t)m*N+j]; b->g[(size_t)k*N+j]+=s; }
      } break;
      case O_EMB:{
        int M=n->i0,D=n->i1; int*ix=(int*)n->aux; Tensor*W=n->b,*o=n->o;
        for(int m=0;m<M;m++){ float*gw=W->g+(size_t)ix[m]*D; const float*go=o->g+(size_t)m*D;
          for(int i=0;i<D;i++) gw[i]+=go[i]; }
      } break;
      case O_RMSNORM:{
        int M=n->i0,D=n->i1; float*rs=(float*)n->aux; Tensor*x=n->a,*w=n->b,*o=n->o;
        for(int m=0;m<M;m++){
          const float*xr=x->d+(size_t)m*D; const float*go=o->g+(size_t)m*D; float r=rs[m];
          float dot=0; for(int i=0;i<D;i++){ dot += go[i]*w->d[i]*xr[i]; }
          for(int i=0;i<D;i++){
            w->g[i] += go[i]*xr[i]*r;
            x->g[(size_t)m*D+i] += go[i]*w->d[i]*r - xr[i]*r*r*r*dot/D;
          }
        }
      } break;
      case O_ADD:{ Tensor*o=n->o; for(int i=0;i<o->n;i++){ n->a->g[i]+=o->g[i]; n->b->g[i]+=o->g[i]; } } break;
      case O_MUL:{ Tensor*o=n->o; for(int i=0;i<o->n;i++){ n->a->g[i]+=o->g[i]*n->b->d[i]; n->b->g[i]+=o->g[i]*n->a->d[i]; } } break;
      case O_BIAS:{ int M=n->i0,D=n->i1; Tensor*o=n->o;
        for(int m=0;m<M;m++)for(int i=0;i<D;i++){ float gv=o->g[(size_t)m*D+i]; n->a->g[(size_t)m*D+i]+=gv; n->b->g[i]+=gv; } } break;
      case O_SCALE:{ Tensor*o=n->o; for(int i=0;i<o->n;i++) n->a->g[i]+=o->g[i]*n->f0; } break;
      case O_ACT:{ Tensor*o=n->o; int k=n->i0; for(int i=0;i<o->n;i++) n->a->g[i]+=o->g[i]*act_df(n->a->d[i],k); } break;
      case O_SOFTMAX:{ int M=n->i0,D=n->i1; Tensor*o=n->o;
        for(int m=0;m<M;m++){ const float*p=o->d+(size_t)m*D; const float*go=o->g+(size_t)m*D; float dot=0;
          for(int i=0;i<D;i++) dot+=go[i]*p[i];
          for(int i=0;i<D;i++) n->a->g[(size_t)m*D+i]+=p[i]*(go[i]-dot); } } break;
      case O_SLICE:{ int M=n->i0,D=n->i1,off=n->i2,len=n->i3; Tensor*o=n->o;
        for(int m=0;m<M;m++)for(int i=0;i<len;i++) n->a->g[(size_t)m*D+off+i]+=o->g[(size_t)m*len+i]; } break;
      case O_CONCAT:{ int M=n->i0,Da=n->i1,Db=n->i2; Tensor*o=n->o;
        for(int m=0;m<M;m++){ for(int i=0;i<Da;i++) n->a->g[(size_t)m*Da+i]+=o->g[(size_t)m*(Da+Db)+i];
                              for(int i=0;i<Db;i++) n->b->g[(size_t)m*Db+i]+=o->g[(size_t)m*(Da+Db)+Da+i]; } } break;
      case O_STE:{ Tensor*o=n->o; for(int i=0;i<o->n;i++) n->a->g[i]+=o->g[i]; } break;
      case O_DWCONV:{
        int B=n->i0,T=n->i1,W=n->i2,D=n->i3; Tensor*x=n->a,*k=n->b,*o=n->o;
        for(int b=0;b<B;b++)for(int t=0;t<T;t++){ const float*go=o->g+(size_t)(b*T+t)*D;
          for(int d=0;d<D;d++){ float gv=go[d];
            for(int j=0;j<W;j++){ int tt=t-j; if(tt<0) break;
              x->g[(size_t)(b*T+tt)*D+d] += k->d[(size_t)d*W+j]*gv;
              k->g[(size_t)d*W+j]        += x->d[(size_t)(b*T+tt)*D+d]*gv; } } }
      } break;
      case O_EMA:{
        int B=n->i0,T=n->i1,D=n->i2; Tensor*x=n->a,*a=n->b,*o=n->o;
        for(int b=0;b<B;b++)for(int d=0;d<D;d++){
          float al=1.0f/(1.0f+expf(-a->d[d])); float dal_da=al*(1.0f-al);
          float ds=0, gal=0;
          for(int t=T-1;t>=0;t--){
            ds += o->g[(size_t)(b*T+t)*D+d];
            float sprev = (t>0)? o->d[(size_t)(b*T+t-1)*D+d] : 0.0f;
            gal += ds*(sprev - x->d[(size_t)(b*T+t)*D+d]);
            x->g[(size_t)(b*T+t)*D+d] += (1.0f-al)*ds;
            ds *= al;
          }
          a->g[d] += gal*dal_da;
        }
      } break;
      case O_PLACE:{ int M=n->i0,W=n->i1,off=n->i2,DW=n->i3; Tensor*o=n->o;
        for(int m=0;m<M;m++){ const float*go=o->g+(size_t)m*DW+off; float*gs=n->a->g+(size_t)m*W;
          for(int k=0;k<W;k++) gs[k]+=go[k]; } } break;
      case O_ROWS:{ int nn=n->i0,D=n->i1; int*ix=(int*)n->aux; Tensor*o=n->o;
        for(int i=0;i<nn;i++){ float*gx=n->a->g+(size_t)ix[i]*D; const float*go=o->g+(size_t)i*D;
          for(int k=0;k<D;k++) gx[k]+=go[k]; } } break;
      case O_SCATTER:{ int nn=n->i0,D=n->i1; int*ix=(int*)n->aux; Tensor*o=n->o;
        for(int i=0;i<nn;i++){ float*gs=n->a->g+(size_t)i*D; const float*go=o->g+(size_t)ix[i]*D;
          for(int k=0;k<D;k++) gs[k]+=go[k]; } } break;
      case O_ROPE:{
        int B=n->i0,T=n->i1,nh=n->i2,hd=n->i3,pos0=n->i4; float theta=n->f0;
        Tensor*o=n->o; int D=o->shape[1];
        const float *cs=rope_table(hd,theta,pos0+T+1);
        for(int b=0;b<B;b++)for(int t=0;t<T;t++){ const float*cst=cs+(size_t)(pos0+t)*hd;
          const float*go=o->g+(size_t)(b*T+t)*D; float*gx=n->a->g+(size_t)(b*T+t)*D;
          for(int h=0;h<nh;h++){ const float*gg=go+h*hd; float*gq=gx+h*hd;
            for(int i=0;i<hd;i+=2){ float c=cst[i],s=cst[i+1];
              gq[i]  += gg[i]*c + gg[i+1]*s;
              gq[i+1]+= -gg[i]*s + gg[i+1]*c; } } }
      } break;
      case O_ATTN:{
        int B=n->i0,T=n->i1,nh=n->i2,nkv=n->i3,hd=n->i4; float softcap=n->f0;
        Tensor *q=n->a,*k=n->b,*v=n->c,*o=n->o;
        int D=nh*hd, rep=nh/nkv; float scale=1.0f/sqrtf((float)hd);
        float *att=((AttnAux*)n->aux)->att;
        /* parallel over batch only: heads sharing a kv-group would race on gk/gv */
        #pragma omp parallel for schedule(static)
        for(int b=0;b<B;b++){
        float *dA=(float*)malloc((size_t)T*sizeof(float));
        for(int h=0;h<nh;h++){
          int kh=h/rep; const float*A=att+(((size_t)b*nh+h)*T)*T;
          for(int t=0;t<T;t++){
            const float*Ar=A+(size_t)t*T; const float*go=o->g+(size_t)(b*T+t)*D+h*hd;
            for(int s=0;s<=t;s++){
              float da=0; const float*vv=v->d+(size_t)(b*T+s)*(nkv*hd)+kh*hd;
              float*gv=v->g+(size_t)(b*T+s)*(nkv*hd)+kh*hd;
              for(int i=0;i<hd;i++){ da+=go[i]*vv[i]; gv[i]+=Ar[s]*go[i]; }
              dA[s]=da;
            }
            float dot=0; for(int s=0;s<=t;s++) dot+=dA[s]*Ar[s];
            const float*qh=q->d+(size_t)(b*T+t)*D+h*hd; float*gq=q->g+(size_t)(b*T+t)*D+h*hd;
            for(int s=0;s<=t;s++){
              float dsc=Ar[s]*(dA[s]-dot);
              if(softcap>0){ /* undo tanh cap: sc_out = c*tanh(z/c) */
                const float*kk0=k->d+(size_t)(b*T+s)*(nkv*hd)+kh*hd; float z=0;
                for(int i=0;i<hd;i++) z+=qh[i]*kk0[i]; z*=scale;
                float th=tanhf(z/softcap); dsc *= (1.0f-th*th);
              }
              dsc*=scale;
              const float*kk=k->d+(size_t)(b*T+s)*(nkv*hd)+kh*hd;
              float*gk=k->g+(size_t)(b*T+s)*(nkv*hd)+kh*hd;
              for(int i=0;i<hd;i++){ gq[i]+=dsc*kk[i]; gk[i]+=dsc*qh[i]; }
            }
          }
        }
        free(dA);
        }
      } break;
      default: break;
    }
    if(g_prof_on){ g_prof[n->op]+=now_sec()-_t0; g_profn[n->op]++; }
  }
}

/* ============================ optimizer ============================ */
static AdamSlot g_slots[MAXPARAM];
/* Serialise the FULL optimiser state so a run can be resumed exactly.
   Params alone are not enough: Adam's moments AND the RNG (which drives batch
   order) both determine every subsequent step. Omitting either makes a resumed
   run diverge from an uninterrupted one, silently. */
void opt_state_save(FILE*f,int step){
  fwrite(&step,4,1,f); fwrite(&g_rng,8,1,f);
  for(int i=0;i<g_nparams;i++){
    fwrite(g_slots[i].m,4,g_params[i]->n,f);
    fwrite(g_slots[i].v,4,g_params[i]->n,f); }
}
int opt_state_load(FILE*f){
  int step=0; if(fread(&step,4,1,f)!=1) return -1;
  if(fread(&g_rng,8,1,f)!=1) return -1;
  for(int i=0;i<g_nparams;i++){
    if(fread(g_slots[i].m,4,g_params[i]->n,f)!=(size_t)g_params[i]->n) return -1;
    if(fread(g_slots[i].v,4,g_params[i]->n,f)!=(size_t)g_params[i]->n) return -1; }
  return step;
}
void opt_init(void){ for(int i=0;i<g_nparams;i++){ g_slots[i].m=(float*)calloc(g_params[i]->n,sizeof(float));
                                                    g_slots[i].v=(float*)calloc(g_params[i]->n,sizeof(float)); } }

/* ---------------- frozen mask ---------------- */
static unsigned char *g_mask[MAXPARAM];
static long long g_ktrain=0; static int g_maskon=0;
int frozen_active(void){ return g_maskon; }
long long frozen_trainable(void){ return g_ktrain; }
void frozen_build(float frac,uint32_t fseed,int fmode){
  if(frac>=1.0f){ g_maskon=0; return; }
  long long ntot=params_count();
  long long K=(long long)(frac*(double)ntot+0.5);
  if(K<1) K=1;
  for(int i=0;i<g_nparams;i++) g_mask[i]=(unsigned char*)calloc(g_params[i]->n,1);
  long long need=K, left=ntot;
  /* fmode 1 spends the budget on the 1-D gain vectors first.  "Small" is
     defined structurally (ndim==1), not by a hand-picked list of names. */
  if(fmode==1){
    for(int i=0;i<g_nparams && need>0;i++){
      if(g_params[i]->ndim!=1) continue;
      for(int j=0;j<g_params[i]->n && need>0;j++){ g_mask[i][j]=1; need--; left--; }
    }
  }
  /* uniform WITHOUT replacement over whatever is left: coordinate i is selected
     with probability (needed remaining)/(coordinates remaining).  Uses its OWN
     prng stream so the batch-order stream g_rng is untouched and a frozen run
     differs from a dense run only by the freezing itself. */
  uint64_t s=(uint64_t)fseed*6364136223846793005ULL+1442695040888963407ULL;
  for(int i=0;i<g_nparams;i++){
    for(int j=0;j<g_params[i]->n;j++){
      if(g_mask[i][j]) continue;
      s^=s>>12; s^=s<<25; s^=s>>27;
      double u=(double)((s*0x2545F4914F6CDD1DULL)>>11)/9007199254740992.0;
      if(left>0 && u*(double)left < (double)need){ g_mask[i][j]=1; need--; }
      left--;
    }
  }
  g_ktrain=K-need; g_maskon=1;
}
void frozen_gather(float *dst){
  long long k=0;
  for(int i=0;i<g_nparams;i++) for(int j=0;j<g_params[i]->n;j++)
    if(g_mask[i][j]) dst[k++]=g_params[i]->d[j];
}
void frozen_apply(const float *src){
  long long k=0;
  for(int i=0;i<g_nparams;i++) for(int j=0;j<g_params[i]->n;j++)
    if(g_mask[i][j]) g_params[i]->d[j]=src[k++];
  params_invalidate_q();
}
void params_invalidate_q(void){ for(int i=0;i<g_nparams;i++) g_params[i]->qvalid=0; }
void opt_zero_grad(void){ for(int i=0;i<g_nparams;i++) memset(g_params[i]->g,0,(size_t)g_params[i]->n*sizeof(float)); }
/* The dense path below is duplicated rather than guarded by a per-element test.
   That is deliberate and was forced by a measured failure: adding `if(mask[j])`
   inside the original loops kept them mathematically identical but changed how
   -O3 -ffast-math vectorised them, so the floating-point summation order moved
   and an archived checkpoint no longer reproduced bit-for-bit.  Keeping the
   unmasked code textually untouched restores exact reproducibility of every
   claim already recorded in model.lab. */
void opt_step(float lr,float b1,float b2,float eps,float wd,int t,float clip){
  double sq=0;
  if(!g_maskon){
    for(int i=0;i<g_nparams;i++){ Tensor*p=g_params[i]; for(int j=0;j<p->n;j++) sq+=(double)p->g[j]*p->g[j]; }
  } else {
    for(int i=0;i<g_nparams;i++){ Tensor*p=g_params[i]; const unsigned char*mk=g_mask[i];
      for(int j=0;j<p->n;j++) if(mk[j]) sq+=(double)p->g[j]*p->g[j]; }
  }
  float gn=(float)sqrt(sq), sc=1.0f; if(clip>0 && gn>clip) sc=clip/(gn+1e-6f);
  float c1=1.0f-powf(b1,(float)t), c2=1.0f-powf(b2,(float)t);
  if(!g_maskon){
    for(int i=0;i<g_nparams;i++){
      Tensor*p=g_params[i]; float*m=g_slots[i].m,*v=g_slots[i].v;
      for(int j=0;j<p->n;j++){
        float g=p->g[j]*sc;
        m[j]=b1*m[j]+(1-b1)*g; v[j]=b2*v[j]+(1-b2)*g*g;
        float mh=m[j]/c1, vh=v[j]/c2;
        p->d[j] -= lr*(mh/(sqrtf(vh)+eps) + wd*p->d[j]);
      }
      p->qvalid=0;
    }
    return;
  }
  for(int i=0;i<g_nparams;i++){
    Tensor*p=g_params[i]; float*m=g_slots[i].m,*v=g_slots[i].v;
    const unsigned char*mk=g_mask[i];
    for(int j=0;j<p->n;j++){
      if(!mk[j]) continue;          /* stays bit-exactly at its initial value */
      float g=p->g[j]*sc;
      m[j]=b1*m[j]+(1-b1)*g; v[j]=b2*v[j]+(1-b2)*g*g;
      float mh=m[j]/c1, vh=v[j]/c2;
      p->d[j] -= lr*(mh/(sqrtf(vh)+eps) + wd*p->d[j]);
    }
    p->qvalid=0;
  }
}

/* ============================ misc ============================ */
long peak_rss_kb(void){
  FILE*f=fopen("/proc/self/status","r"); if(!f) return -1; char line[256]; long v=-1;
  while(fgets(line,sizeof line,f)) if(!strncmp(line,"VmHWM:",6)){ sscanf(line+6,"%ld",&v); break; }
  fclose(f); return v;
}
double now_sec(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
