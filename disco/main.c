/* main.c -- training / evaluation driver.
 *
 * The evaluator is the point of this program.  Every reported number is measured:
 *   - val loss      : held-out batches, IDENTICAL offsets for every architecture
 *   - wbytes/token  : counted inside the ops during real cached autoregressive decode
 *   - peak RSS      : VmHWM of an inference-only process
 *   - tok/s         : wall clock, 1 thread, median of repeats
 *   - params        : counted from the registry
 * plus `consistency`: max |logit_train_path - logit_decode_path|.  If an architecture's
 * fast decode path silently disagrees with what was trained, this number exposes it.
 */
#include "model.h"
#include "mem.h"
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---------------- data ---------------- */
static uint16_t *DATA=NULL; static long NTOK=0, NTRAIN=0;
static void data_load(const char*p){
  FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"no %s\n",p);exit(1);}
  fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  NTOK=sz/2; DATA=(uint16_t*)malloc(sz); if(fread(DATA,1,sz,f)!=(size_t)sz){fprintf(stderr,"read\n");exit(1);}
  fclose(f); NTRAIN=(long)(NTOK*0.95);
}
/* infer mode reads a 2KB window only: the corpus must not pollute peak RSS,
   which is supposed to report the DEPLOYMENT footprint. */
static uint16_t PROMPTBUF[1024];
static void data_load_window(const char*p){
  FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"no %s\n",p);exit(1);}
  fseek(f,0,SEEK_END); long sz=ftell(f); NTOK=sz/2; NTRAIN=(long)(NTOK*0.95);
  fseek(f,(NTRAIN+1000)*2,SEEK_SET);
  if(fread(PROMPTBUF,2,1024,f)!=1024){fprintf(stderr,"read window\n");exit(1);}
  fclose(f);
}
/* deterministic val offsets shared by every architecture */
static void val_offsets(long *off,int nb,int T){
  uint64_t s=0xC0FFEEULL; long span=NTOK-NTRAIN-T-2;
  for(int i=0;i<nb;i++){ s^=s>>12;s^=s<<25;s^=s>>27; off[i]=NTRAIN+(long)((s*0x2545F4914F6CDD1DULL>>32)%span); }
}
static void fill_batch(long off,int B,int T,int*x,int*y,int stride){
  for(int b=0;b<B;b++){ long o=off+(long)b*stride;
    for(int t=0;t<T;t++){ x[b*T+t]=DATA[o+t]; y[b*T+t]=DATA[o+t+1]; } }
}

/* ---------------- checkpoint ---------------- */
static void ckpt_save(const char*p){
  FILE*f=fopen(p,"wb"); int magic=0xD15C0; fwrite(&magic,4,1,f); fwrite(&g_nparams,4,1,f);
  for(int i=0;i<g_nparams;i++){ fwrite(&g_params[i]->n,4,1,f); fwrite(g_params[i]->d,4,g_params[i]->n,f); }
  fclose(f);
}
static void ckpt_load(const char*p){
  FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"no ckpt %s\n",p);exit(1);}
  int magic,np; if(fread(&magic,4,1,f)!=1||fread(&np,4,1,f)!=1){exit(1);}
  if(magic!=0xD15C0||np!=g_nparams){fprintf(stderr,"ckpt mismatch %d vs %d\n",np,g_nparams);exit(1);}
  for(int i=0;i<g_nparams;i++){ int n; if(fread(&n,4,1,f)!=1||n!=g_params[i]->n){fprintf(stderr,"ckpt shape %d\n",i);exit(1);}
    if(fread(g_params[i]->d,4,n,f)!=(size_t)n){exit(1);} }
  fclose(f);
  params_invalidate_q();   /* weights changed outside opt_step */
}

/* --- K-token-per-forward loss.
   train=1 : every position, every head (dense signal).
   train=0 : only positions t %% K == 0, i.e. exactly the tokens the decode loop
             would actually emit -- so val_bpb describes the emission process. --- */
static double ce_multi(Tensor *lg,int*tok,int B,int T,int V,int K,int emission_only){
  int M=B*T; double tot=0; long cnt=0;
  int *sub=malloc(sizeof(int)*M); float *buf=NULL;
  Tensor tmp; memset(&tmp,0,sizeof tmp);
  for(int j=0;j<K;j++){
    /* head j at position t predicts token t+1+j */
    int n=0; static int rows[1<<16];
    for(int b=0;b<B;b++)for(int t=0;t<T;t++){
      if(t+1+j>=T) continue;
      if(emission_only && (t%K)!=0) continue;
      rows[n]=b*T+t; sub[n]=tok[b*T+t+1+j]; n++;
    }
    if(!n) continue;
    /* score the [n, V] slice belonging to head j */
    Tensor *sl=op_slice(lg,j*V,V);
    Tensor *rw=op_rows(sl,rows,n);
    tot += op_ce_loss(rw,sub,n)*n; cnt += n;
  }
  free(sub); (void)buf; (void)tmp;
  return cnt? tot/cnt : 0.0;
}

/* --- CONFIDENCE-GATED emission.  Same trunk and heads as fixed-K multitok, but
   head j is only used while its own max probability exceeds tau, so the number of
   tokens emitted per weight-read pass is data-dependent and tau sweeps a whole
   line of the grid with one trained model.  No extra parameters, no change to the
   training objective -- purely a decode rule, simulated here exactly. --- */
static double ce_adaptive(Tensor*lg,int*tok,int B,int T,int V,int K,float tau,double*tok_per_blk){
  double tot=0; long cnt=0, blocks=0; int W=K*V;
  for(int b=0;b<B;b++){
    int t=0;
    while(t<T-1){
      const float*row=lg->d+(size_t)(b*T+t)*W;
      int m=0;
      while(m<K && t+1+m<T){
        const float*h=row+(size_t)m*V;
        float mx=h[0]; for(int i=1;i<V;i++) if(h[i]>mx) mx=h[i];
        double s=0; for(int i=0;i<V;i++) s+=exp((double)h[i]-mx);
        tot += -((double)h[tok[b*T+t+1+m]] - mx - log(s));
        cnt++; m++;
        if(m>=K || t+1+m>=T) break;
        const float*hn=row+(size_t)m*V;
        float mx2=hn[0]; for(int i=1;i<V;i++) if(hn[i]>mx2) mx2=hn[i];
        double s2=0; for(int i=0;i<V;i++) s2+=exp((double)hn[i]-mx2);
        if(1.0/s2 < tau) break;              /* head m is not confident enough */
      }
      blocks++; t+=m;
    }
  }
  if(tok_per_blk) *tok_per_blk = blocks? (double)cnt/blocks : 1.0;
  return cnt? tot/cnt : 0;
}

/* --- CASCADE scoring.  logits are [M, 2V] = concat(transformer, fast).
   For each position the gate reads the FAST model's own max probability, which
   depends only on the context, so the mixture is a properly normalised
   conditional distribution and its bits-per-byte is directly comparable. --- */
static double ce_cascade(Tensor*lg,int*tok,int B,int T,int V,float tau,double*frac_fast){
  double tot=0; long cnt=0,nf=0; int W=2*V;
  for(int b=0;b<B;b++)for(int t=0;t<T-1;t++){
    const float*ff=lg->d+(size_t)(b*T+t)*W+V;
    float mx=ff[0]; for(int i=1;i<V;i++) if(ff[i]>mx) mx=ff[i];
    double s=0; for(int i=0;i<V;i++) s+=exp((double)ff[i]-mx);
    int y=tok[b*T+t+1];
    if(1.0/s > tau){ tot += -((double)ff[y]-mx-log(s)); nf++; }
    else{
      const float*fu=lg->d+(size_t)(b*T+t)*W;
      float m2=fu[0]; for(int i=1;i<V;i++) if(fu[i]>m2) m2=fu[i];
      double s2=0; for(int i=0;i<V;i++) s2+=exp((double)fu[i]-m2);
      tot += -((double)fu[y]-m2-log(s2));
    }
    cnt++;
  }
  if(frac_fast) *frac_fast = cnt? (double)nf/cnt : 0.0;
  return cnt? tot/cnt : 0.0;
}

/* ---------------- eval ---------------- */
static double g_tpb=1.0, g_ffast=0.0;
static int CASC=0; static float FLAM=0.5f;
static double eval_val(Arch*A,Cfg*c,int nb,int B,int T,int K,float tau){
  int sv=g_train; g_train=0; params_invalidate_q();
  long off[256]; val_offsets(off,nb,T*B+2);
  int *x=malloc(sizeof(int)*B*T),*y=malloc(sizeof(int)*B*T);
  double tot=0;
  for(int i=0;i<nb;i++){
    arena_reset(); tape_reset();
    fill_batch(off[i],B,T,x,y,T);
    Tensor*lg=A->fwd(c,x,B,T,0,0);
    if(CASC){ double ff; tot += ce_cascade(lg,x,B,T,c->vocab,tau,&ff); g_ffast=ff; }
    else if(K>1 && tau>0){ double tpb; tot += ce_adaptive(lg,x,B,T,c->vocab,K,tau,&tpb); g_tpb=tpb; }
    else if(K>1)     { tot += ce_multi(lg,x,B,T,c->vocab,K,1); g_tpb=K; }
    else             { tot += op_ce_loss(lg,y,B*T); g_tpb=1.0; }
  }
  free(x);free(y); g_train=sv; return tot/nb;
}

/* ---------------- generation ---------------- */
static int sample_tok(float*lg,int V,float temp,float topp){
  if(temp<=0){ int bi=0; for(int i=1;i<V;i++) if(lg[i]>lg[bi]) bi=i; return bi; }
  static float p[1<<14]; float mx=lg[0]; for(int i=1;i<V;i++) if(lg[i]>mx) mx=lg[i];
  double s=0; for(int i=0;i<V;i++){ p[i]=expf((lg[i]-mx)/temp); s+=p[i]; }
  for(int i=0;i<V;i++) p[i]/=(float)s;
  float r=rnd_f(), cum=0;
  for(int i=0;i<V;i++){ cum+=p[i]; if(r<cum) return i; }
  return V-1;
}

/* ---------------- gradcheck ---------------- */
static int gradcheck(Arch*A,Cfg*c){
  int B=2,T=6;
  A->build(c);
  int *x=malloc(sizeof(int)*B*T),*y=malloc(sizeof(int)*B*T);
  for(int i=0;i<B*T;i++){ x[i]=rnd_u32()%c->vocab; y[i]=rnd_u32()%c->vocab; }
  g_train=1; opt_zero_grad(); arena_reset(); tape_reset();
  Tensor*lg=A->fwd(c,x,B,T,0,0); op_ce_loss(lg,y,B*T); tape_backward();
  int nbad=0,ntest=0,nskip=0; double worst=0; char wname[256]="";
  for(int i=0;i<g_nparams;i++){
    Tensor*p=g_params[i];
    for(int r=0;r<3;r++){
      int j=rnd_u32()%p->n; double an=p->g[j]; float o=p->d[j]; float eps=1e-3f;
      int sv=g_train; g_train=0;
      p->d[j]=o+eps; params_invalidate_q(); arena_reset(); tape_reset(); double lp=op_ce_loss(A->fwd(c,x,B,T,0,0),y,B*T);
      p->d[j]=o-eps; params_invalidate_q(); arena_reset(); tape_reset(); double lm=op_ce_loss(A->fwd(c,x,B,T,0,0),y,B*T);
      p->d[j]=o; g_train=sv;
      double num=(lp-lm)/(2.0*eps);
      if(fabs(num)<1e-7 && fabs(an)<1e-7){ nskip++; continue; }
      double den=fabs(num)+fabs(an)+1e-9, rel=fabs(num-an)/den;
      ntest++; if(rel>worst){worst=rel;snprintf(wname,sizeof wname,"%s[%d] num=%.6f an=%.6f",p->name,j,num,an);}
      if(rel>2e-2) nbad++;
    }
  }
  printf("gradcheck: %d/%d bad (%d skipped as ~0), worst rel=%.6f  (%s)\n",nbad,ntest,nskip,worst,wname);

  /* directional-derivative test: O(1) magnitude, immune to the fp32 noise floor
     that swamps single-coordinate finite differences. */
  int fail=0;
  for(int trial=0;trial<3;trial++){
    double dot=0; long long tot=0;
    for(int i=0;i<g_nparams;i++) tot+=g_params[i]->n;
    float **dir=malloc(sizeof(float*)*g_nparams);
    for(int i=0;i<g_nparams;i++){ Tensor*p=g_params[i]; dir[i]=malloc(sizeof(float)*p->n);
      for(int j=0;j<p->n;j++){ dir[i][j]=rnd_normal(); dot+=(double)dir[i][j]*p->g[j]; } }
    float eps=1e-3f/sqrtf((float)tot)*30.0f;
    int sv=g_train; g_train=0;
    for(int i=0;i<g_nparams;i++){ Tensor*p=g_params[i]; for(int j=0;j<p->n;j++) p->d[j]+=eps*dir[i][j]; }
    params_invalidate_q(); arena_reset(); tape_reset(); double lp=op_ce_loss(A->fwd(c,x,B,T,0,0),y,B*T);
    for(int i=0;i<g_nparams;i++){ Tensor*p=g_params[i]; for(int j=0;j<p->n;j++) p->d[j]-=2*eps*dir[i][j]; }
    params_invalidate_q(); arena_reset(); tape_reset(); double lm=op_ce_loss(A->fwd(c,x,B,T,0,0),y,B*T);
    for(int i=0;i<g_nparams;i++){ Tensor*p=g_params[i]; for(int j=0;j<p->n;j++) p->d[j]+=eps*dir[i][j]; free(dir[i]); }
    free(dir); params_invalidate_q(); g_train=sv;
    double num=(lp-lm)/(2.0*eps);
    double rel=fabs(num-dot)/(fabs(num)+fabs(dot)+1e-12);
    printf("  directional %d: analytic=%.6f numeric=%.6f  rel=%.2e  %s\n",
           trial,dot,num,rel, rel<1e-2?"OK":"FAIL");
    if(rel>=1e-2) fail=1;
  }
  return fail;
}

/* ---------------- online memory test ----------------
 * A causal, single-token-at-a-time walk over one or more fixed windows,
 * identical in spirit to the `infer` decode loop but teacher-forced (it reads
 * the real next byte rather than sampling) so it produces a bpb number
 * directly comparable to eval_val's.  `cache` may be NULL (pure core, no
 * mixing).  `do_write` controls whether cache_write is called at each step.
 * Returns mean bpb over every position in every window; *out_writes receives
 * the write count if non-NULL.
 */
#define MAXCTX 4
static double stream_bpb(Arch*A,Cfg*c,long*offs,int nwin,int T,
                          Cache*cache,float lambda,double tau,int do_write,int gate_always,
                          long long*out_writes){
  int V=c->vocab; float*probs=malloc(sizeof(float)*V); float*mixed=malloc(sizeof(float)*V);
  double tot=0; long long cnt=0, writes=0;
  for(int w=0; w<nwin; w++){
    A->cache_reset(c);
    int ctxbuf[MAXCTX]; for(int i=0;i<MAXCTX;i++) ctxbuf[i]=256;
    for(int t=0;t<T;t++){
      int tok=(int)DATA[offs[w]+t];
      for(int i=MAXCTX-1;i>0;i--) ctxbuf[i]=ctxbuf[i-1]; ctxbuf[0]=tok;
      arena_reset(); tape_reset();
      Tensor*lg=A->fwd(c,&tok,1,1,t,1);
      float mx=lg->d[0]; for(int i=1;i<V;i++) if(lg->d[i]>mx) mx=lg->d[i];
      double s=0; for(int i=0;i<V;i++){ probs[i]=expf(lg->d[i]-mx); s+=probs[i]; }
      float inv=(float)(1.0/s); for(int i=0;i<V;i++) probs[i]*=inv;
      int target=(int)DATA[offs[w]+t+1];
      if(cache) cache_mix(cache,ctxbuf,probs,lambda,mixed); else memcpy(mixed,probs,sizeof(float)*V);
      double p=mixed[target]; if(p<1e-30) p=1e-30;
      tot += -log(p); cnt++;
      if(cache && do_write){
        double surprise = -log(probs[target]<1e-30?1e-30:probs[target]);
        writes += cache_write(cache,ctxbuf,target,surprise,tau,gate_always);
      }
    }
  }
  free(probs); free(mixed);
  if(out_writes) *out_writes=writes;
  return cnt? (tot/cnt)/0.6931471805599453 : 0.0;   /* nats -> bits */
}

/* ---------------- main ---------------- */
int main(int argc,char**argv){
  const char*mode = argc>1?argv[1]:"help";
  Cfg c={0}; c.dim=64;c.n_layers=5;c.n_heads=8;c.n_kv_heads=4;c.hidden_dim=176;
  c.vocab=257;c.seq_len=256;c.tie=1;c.rope_theta=10000.f;
  const char*archname="llama",*ckpt="runs/m.bin",*jsonp=NULL,*data="data/corpus.bin";
  int steps=2000,B=16,T=256,nvalb=24,gen=256,threads=4,seed=1337,warm=100,repeats=3,KOUT=1;
  float lr=1e-3f,wd=0.1f,minlr_frac=0.1f; size_t arena=1500ull<<20;
  const char*prompt=NULL;
  float mlambda=0.3f; double mtau=2.0; int mslots=16384, mctx=2;
  long mstart=18640100, mend=18762000;    /* the largest gap disjoint from every val_offsets window */

  for(int i=2;i<argc;i++){
    #define ARG(s) (!strcmp(argv[i],s) && i+1<argc)
    if(ARG("--arch")) archname=argv[++i];
    else if(ARG("--ckpt")) ckpt=argv[++i];
    else if(ARG("--json")) jsonp=argv[++i];
    else if(ARG("--data")) data=argv[++i];
    else if(ARG("--steps")) steps=atoi(argv[++i]);
    else if(ARG("--bs")) B=atoi(argv[++i]);
    else if(ARG("--seq")) T=atoi(argv[++i]);
    else if(ARG("--dim")) c.dim=atoi(argv[++i]);
    else if(ARG("--layers")) c.n_layers=atoi(argv[++i]);
    else if(ARG("--heads")) c.n_heads=atoi(argv[++i]);
    else if(ARG("--kvheads")) c.n_kv_heads=atoi(argv[++i]);
    else if(ARG("--hidden")) c.hidden_dim=atoi(argv[++i]);
    else if(ARG("--tie")) c.tie=atoi(argv[++i]);
    else if(ARG("--lr")) lr=atof(argv[++i]);
    else if(ARG("--wd")) wd=atof(argv[++i]);
    else if(ARG("--seed")) seed=atoi(argv[++i]);
    else if(ARG("--gen")) gen=atoi(argv[++i]);
    else if(ARG("--valb")) nvalb=atoi(argv[++i]);
    else if(ARG("--threads")) threads=atoi(argv[++i]);
    else if(ARG("--repeats")) repeats=atoi(argv[++i]);
    else if(ARG("--arena")) arena=(size_t)atoll(argv[++i])<<20;
    else if(ARG("--prompt")) prompt=argv[++i];
    else if(ARG("--mlambda")) mlambda=atof(argv[++i]);
    else if(ARG("--mtau")) mtau=atof(argv[++i]);
    else if(ARG("--mslots")) mslots=atoi(argv[++i]);
    else if(ARG("--mctx")) mctx=atoi(argv[++i]);
    else if(ARG("--mstart")) mstart=atol(argv[++i]);
    else if(ARG("--mend")) mend=atol(argv[++i]);
    else if(ARG("--knob")){ /* name=value */
      char *kv=argv[++i],*eq=strchr(kv,'='); if(eq){ *eq=0;
        snprintf(c.kname[c.nknob],24,"%s",kv); c.kval[c.nknob]=atof(eq+1); c.nknob++; *eq='='; } }
    else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
  }
#ifdef _OPENMP
  omp_set_num_threads(threads);
#endif
  g_rng = (uint64_t)seed*2654435761ULL + 12345ULL;
  Arch*A=arch_find(archname); if(!A){fprintf(stderr,"no arch '%s'\n",archname);return 1;}
  KOUT=cfg_geti(&c,"kout",1);
  float TAU=cfg_get(&c,"tau",0.0f);
  CASC=cfg_geti(&c,"casc",0); FLAM=cfg_get(&c,"flam",0.5f);
  c.seq_len = T>c.seq_len?T:c.seq_len;

  if(!strcmp(mode,"gradcheck")){
    Cfg g={0}; g.dim=16;g.n_layers=2;g.n_heads=2;g.n_kv_heads=1;g.hidden_dim=24;
    g.vocab=17;g.seq_len=8;g.tie=0;g.rope_theta=10000.f;
    for(int i=0;i<c.nknob;i++){ memcpy(g.kname[i],c.kname[i],24); g.kval[i]=c.kval[i]; } g.nknob=c.nknob;
    arena_init(64ull<<20);
    return gradcheck(A,&g)>0 ? 1 : 0;
  }
  if(!strcmp(mode,"list")){ for(int i=0;i<g_narchs;i++) printf("%-22s %s\n",g_archs[i].name,g_archs[i].desc); return 0; }

  if(!strcmp(mode,"infer")) data_load_window(data); else data_load(data);

  if(!strcmp(mode,"train")){
    arena_init(arena);
    A->build(&c); opt_init();
    long long np=params_count();
    fprintf(stderr,"[%s] params=%lld  B=%d T=%d steps=%d lr=%g\n",archname,np,B,T,steps,lr);
    int *x=malloc(sizeof(int)*B*T),*y=malloc(sizeof(int)*B*T);
    double t0=now_sec(); double best=1e9; float lastloss=0;
    double tf=0,tb=0,to=0,tl=0; g_prof_on = getenv("DISCO_PROF")!=NULL;
    FILE*lg = NULL; { char pb[512]; snprintf(pb,sizeof pb,"%s.trainlog",ckpt); lg=fopen(pb,"w"); }
    for(int s=1;s<=steps;s++){
      /* cosine schedule with warmup */
      float f = s<warm ? (float)s/warm
                       : minlr_frac + 0.5f*(1-minlr_frac)*(1+cosf(3.14159265f*(s-warm)/(float)(steps-warm+1)));
      float cur = lr*f;
      long o = NTRAIN>0 ? (long)(((uint64_t)rnd_u32()<<16 ^ rnd_u32()) % (NTRAIN - (long)B*T - 4)) : 0;
      arena_reset(); tape_reset(); opt_zero_grad();
      fill_batch(o,B,T,x,y,T);
      g_train=1;
      double _a=now_sec();
      Tensor*out=A->fwd(&c,x,B,T,0,0);
      double _b=now_sec(); tf+=_b-_a;
      double loss;
      if(CASC){   /* train both networks everywhere; the gate is applied only at eval */
        loss = op_ce_loss(op_slice(out,0,c.vocab),y,B*T)
             + FLAM*op_ce_loss(op_slice(out,c.vocab,c.vocab),y,B*T);
      } else loss = (KOUT>1)? ce_multi(out,x,B,T,c.vocab,KOUT,0) : op_ce_loss(out,y,B*T);
      double _c2=now_sec(); tl+=_c2-_b;
      tape_backward();
      double _d=now_sec(); tb+=_d-_c2;
      opt_step(cur,0.9f,0.95f,1e-8f,wd,s,1.0f);
      to+=now_sec()-_d;
      lastloss=(float)loss;
      if(s%50==0||s==1){ fprintf(stderr,"  step %5d  loss %.4f  lr %.2e  %.1fs\n",s,loss,cur,now_sec()-t0);
                         if(lg) fprintf(lg,"%d %.5f\n",s,loss); }
      if(loss!=loss){ fprintf(stderr,"NaN at step %d\n",s); break; }
    }
    (void)best;
    if(g_prof_on){ fprintf(stderr,"[phase] fwd %.2fs  celoss %.2fs  bwd %.2fs  opt %.2fs\n",tf,tl,tb,to); prof_dump("bwd"); }
    if(lg) fclose(lg);
    double ttrain=now_sec()-t0;
    double vl=eval_val(A,&c,nvalb,B,T,KOUT,TAU);
    ckpt_save(ckpt);
    fprintf(stderr,"train done %.1fs  final=%.4f  val=%.4f (bpb %.4f)\n",ttrain,lastloss,vl,vl/0.6931472);
    if(jsonp){ FILE*j=fopen(jsonp,"w");
      fprintf(j,"{\"arch\":\"%s\",\"params\":%lld,\"stored_bytes\":%.0f,\"train_loss\":%.5f,"
                "\"val_loss\":%.5f,\"val_bpb\":%.5f,\"train_sec\":%.1f,\"steps\":%d,\"bs\":%d,\"seq\":%d,"
                "\"lr\":%g,\"seed\":%d,\"tokens_seen\":%lld,\"tok_per_block\":%.3f,\"frac_fast\":%.4f}\n",
        archname,params_count(),params_bytes(),lastloss,vl,vl/0.6931472,ttrain,steps,B,T,lr,seed,
        (long long)steps*B*T,g_tpb,g_ffast);
      fclose(j); }
    return 0;
  }

  if(!strcmp(mode,"infer")){
    /* inference-only process: RSS here is the deployment footprint */
    arena_init(16ull<<20);
    A->build(&c); ckpt_load(ckpt);
    A->cache_alloc(&c,c.seq_len+gen+8);
    g_train=0;

    double maxdiff=0;

    /* ---- measured decode: bytes, speed ---- */
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    double best_tps=0; long long wb=0,sb=0,fl=0,wu=0,ww=0; double ffast=0;
    const int WINW=16;   /* window for the working-set measurement */
    char text[8192]; int tn=0,textlen=0;
    for(int rep=0;rep<repeats;rep++){
      A->cache_reset(&c);
      int tokn=256; /* BOS */
      /* warmup / prompt */
      g_wbytes=0; g_sbytes=0; g_flops=0; long long uniq=0, wwin=0; int nwin=0;
      double t0=0; int nmeas=0; tn=0; (void)nmeas;
      /* K-token emission, matching what eval_val scores.
         A block boundary at position p yields tokens p+1..p+K from heads 0..K-1.
         The trunk must still SEE every emitted token (otherwise the KV cache has
         gaps and decode diverges from the scored model), so the K tokens are then
         pushed through the trunk.  In a batched implementation that is ONE pass
         over K positions reading each weight once; here it is K single-position
         passes, so:
            - unique weight bytes are accumulated over the whole block and
              divided by K  -- this is exact, and is the grid axis;
            - MACs and wall-clock are NOT divided by K, because the arithmetic
              really is paid per token.  Multi-token emission buys I/O, not compute.
         wbytes_unique_step() is therefore called once per BLOCK: the per-row touch
         flags accumulate across the K sub-passes and a weight touched in several
         of them is counted once, which is exactly the batched cost. */
      if(CASC){
        /* Two-mode decode.  The fast net runs for EVERY byte (its bytes are
           counted every step).  The transformer runs only when the fast net is
           unsure, and then processes the whole backlog in one catch-up whose
           weight touches are counted ONCE -- the batched-pass cost. */
        static int hist[8192]; hist[0]=256; int tp=0;
        g_casc_mode=2; arena_reset(); tape_reset();
        Tensor *bigl=A->fwd(&c,&hist[0],1,1,0,1);
        wbytes_unique_step();
        long long nfast=0;
        for(int i=1;i<=gen;i++){
          if(i-1>=8 && t0==0){ t0=now_sec(); g_wbytes=0; g_sbytes=0; g_flops=0; uniq=0; nmeas=0; }
          g_casc_mode=1; arena_reset(); tape_reset();
          Tensor *fa=A->fwd(&c,&hist[i-1],1,1,i-1,1);
          long long ub=wbytes_unique_step();
          if(i%WINW==0){ long long w=wbytes_window_flush(); if(t0!=0){ wwin+=w; nwin++; } }
          float mx=fa->d[0]; for(int q=1;q<c.vocab;q++) if(fa->d[q]>mx) mx=fa->d[q];
          double sm=0; for(int q=0;q<c.vocab;q++) sm+=exp((double)fa->d[q]-mx);
          int nx;
          if(1.0/sm > TAU){ nx=sample_tok(fa->d,c.vocab,0.85f,1.0f); nfast++; }
          else{
            g_casc_mode=2;
            for(int p=tp+1;p<=i-1;p++){ arena_reset(); tape_reset();
              bigl=A->fwd(&c,&hist[p],1,1,p,1); }
            tp=i-1;
            ub += wbytes_unique_step();
            nx=sample_tok(bigl->d,c.vocab,0.85f,1.0f);
          }
          hist[i]=nx;
          if(rep==0 && tn<(int)sizeof(text)-2 && nx<256) text[tn++]=(char)nx;
          if(t0!=0){ nmeas++; uniq+=ub; }
        }
        g_casc_mode=0;
        if(nmeas==0) nmeas=1;
        double el=now_sec()-t0; double tps=nmeas/el;
        if(tps>best_tps){ best_tps=tps; wb=g_wbytes/nmeas; sb=g_sbytes/nmeas;
                          fl=g_flops/nmeas;  wu=uniq/nmeas; ww=nwin?wwin/nwin:0; ffast=(double)nfast/gen; }
        if(rep==0) textlen=tn;
        continue;
      }
      int pos=0, emitted=0;
      arena_reset(); tape_reset();
      Tensor *lgt=A->fwd(&c,&tokn,1,1,0,1);
      wbytes_unique_step();                      /* discard the priming block */
      int blk[8];
      while(emitted<gen){
        if(emitted>=8 && t0==0){ t0=now_sec(); g_wbytes=0; g_sbytes=0; g_flops=0; uniq=0; nmeas=0; }
        int nb=0;
        for(int j=0;j<KOUT && emitted<gen;j++){
          if(j>0 && TAU>0){                       /* head j must vouch for itself */
            const float*hn=lgt->d+(size_t)j*c.vocab;
            float mx=hn[0]; for(int i=1;i<c.vocab;i++) if(hn[i]>mx) mx=hn[i];
            double s=0; for(int i=0;i<c.vocab;i++) s+=exp((double)hn[i]-mx);
            if(1.0/s < TAU) break;
          }
          int nx=sample_tok(lgt->d+(size_t)j*c.vocab,c.vocab,0.85f,1.0f);
          blk[nb++]=nx;
          if(rep==0 && tn<(int)sizeof(text)-2 && nx<256) text[tn++]=(char)nx;
          emitted++; if(t0!=0) nmeas++;
        }
        for(int j=0;j<nb;j++){                   /* advance the trunk over the block */
          arena_reset(); tape_reset();
          lgt=A->fwd(&c,&blk[j],1,1,pos+1+j,1);
        }
        pos+=nb;
        long long u=wbytes_unique_step();        /* once per block, not per sub-pass */
        if(t0!=0) uniq+=u;
        if((pos/WINW)!=((pos-nb)/WINW)){ long long w=wbytes_window_flush(); if(t0!=0){ wwin+=w; nwin++; } }
        tokn=blk[nb-1];
      }
      if(nmeas==0) nmeas=1;
      double el=now_sec()-t0; double tps=nmeas/el;
      if(tps>best_tps){ best_tps=tps; wb=g_wbytes/nmeas; sb=g_sbytes/nmeas; fl=g_flops/nmeas; wu=uniq/nmeas; ww=nwin?wwin/nwin:0; }
      if(rep==0) textlen=tn;
    }
    text[textlen]=0;
    /* RSS is read HERE, before the consistency check runs: the deployment
       footprint is single-step decoding, and the check's full-sequence forward
       would add an architecture-dependent slab of arena pages to VmHWM. */
    long rss=peak_rss_kb();

    /* ---- consistency: cached decode must reproduce the full-seq forward ---- */
    {
      int TT=64; int *xs=malloc(sizeof(int)*TT);
      for(int i=0;i<TT;i++) xs[i]=(i==0)?256:(int)PROMPTBUF[i];
      g_casc_mode=0;
      arena_reset(); tape_reset();
      Tensor *full=A->fwd(&c,xs,1,TT,0,0);
      int LW=full->shape[1];
      float *ref=malloc(sizeof(float)*TT*LW);
      memcpy(ref,full->d,sizeof(float)*TT*LW);
      A->cache_reset(&c);
      for(int t=0;t<TT;t++){
        arena_reset(); tape_reset();
        Tensor *st=A->fwd(&c,&xs[t],1,1,t,1);
        for(int i=0;i<LW;i++){ double d=fabs(st->d[i]-ref[(size_t)t*LW+i]); if(d>maxdiff) maxdiff=d; }
      }
      free(ref); free(xs);
    }
    fprintf(stderr,"consistency=%.2e  uniqB/tok=%lld  ws16=%lld  trafficB/tok=%lld  sbytes/tok=%lld  tok/s=%.1f  rssKB=%ld\n",
            maxdiff,wu,ww,wb,sb,best_tps,rss);
    fprintf(stderr,"--- sample ---\n%.600s\n--------------\n",text);
    if(jsonp){ FILE*j=fopen(jsonp,"w");
      fprintf(j,"{\"arch\":\"%s\",\"params\":%lld,\"stored_bytes\":%.0f,\"wbytes_per_tok\":%lld,"
                "\"wtraffic_per_tok\":%lld,\"working_set_16tok\":%lld,\"sbytes_per_tok\":%lld,\"macs_per_tok\":%lld,"
                "\"tok_per_sec\":%.2f,\"peak_rss_kb\":%ld,\"consistency\":%.3e,\"gen\":%d,\"kout\":%d,"
                "\"frac_fast_decode\":%.4f}\n",
        archname,params_count(),params_bytes(),wu,wb,ww,sb,fl,best_tps,rss,maxdiff,gen,KOUT,ffast);
      fclose(j); }
    { char pb[512]; snprintf(pb,sizeof pb,"%s.sample.txt",ckpt); FILE*sf=fopen(pb,"w"); if(sf){fputs(text,sf);fclose(sf);} }
    (void)prompt;
    return 0;
  }

  if(!strcmp(mode,"rescore")){
    /* re-score an existing checkpoint under a different emission rule (tau).
       Training is unaffected by tau, so a whole line of the grid costs one run. */
    arena_init(arena); A->build(&c); ckpt_load(ckpt);
    double vl=eval_val(A,&c,nvalb,B,T,KOUT,TAU);
    fprintf(stderr,"rescore tau=%.3f val_bpb=%.5f tok_per_block=%.3f frac_fast=%.4f\n",
            TAU,vl/0.6931472,g_tpb,g_ffast);
    if(jsonp){ FILE*j=fopen(jsonp,"w");
      fprintf(j,"{\"arch\":\"%s\",\"params\":%lld,\"stored_bytes\":%.0f,\"val_loss\":%.5f,"
                "\"val_bpb\":%.5f,\"tau\":%.4f,\"tok_per_block\":%.3f,\"frac_fast\":%.4f,"
                "\"train_loss\":0,\"steps\":%d,\"bs\":%d,\"seq\":%d,\"lr\":%g,\"seed\":%d,"
                "\"train_sec\":0,\"tokens_seen\":0}\n",
        archname,params_count(),params_bytes(),vl,vl/0.6931472,TAU,g_tpb,g_ffast,steps,B,T,lr,seed);
      fclose(j); }
    return 0;
  }
  if(!strcmp(mode,"memtest")){
    /* Online, writable, surprise-gated key->count cache bolted onto a frozen,
     * already-trained core.  Not part of the autodiff graph (mem.c): it is
     * written directly at inference time, which is the entire point -- it
     * tests whether something can be written into a running system without
     * touching the backprop-trained weights, and whether that write helps
     * (forward transfer) without hurting (interference on the original
     * validation set) -- the concrete, buildable slice of the "separate
     * write path from read path" argument from the discussion in JOURNAL.md.
     */
    if(mctx>MAXCTX) mctx=MAXCTX;
    arena_init(16ull<<20); A->build(&c); ckpt_load(ckpt); A->cache_alloc(&c,T+8);

    long refoffs[256]; val_offsets(refoffs,nvalb,T*B+2);   /* same 24 windows used everywhere else */
    long adaptoff[1]; adaptoff[0]=mstart;
    int adaptT = (int)(mend-mstart-2); if(adaptT<64){fprintf(stderr,"adaptation gap too small\n");return 1;}

    fprintf(stderr,"[memtest] arch=%s lambda=%.2f tau=%.2f slots=%d ctx=%d gap=[%ld,%ld) len=%d\n",
            archname,mlambda,mtau,mslots,mctx,mstart,mend,adaptT);

    double before = stream_bpb(A,&c,refoffs,nvalb,T,NULL,0,0,0,0,NULL);

    Cache *cg = cache_new(mslots,c.vocab,mctx);          /* surprise-gated */
    long long wg=0;
    double pass1_g = stream_bpb(A,&c,adaptoff,1,adaptT,cg,mlambda,mtau,1,0,&wg);
    double pass2_g = stream_bpb(A,&c,adaptoff,1,adaptT,cg,mlambda,mtau,0,0,NULL);
    double after_g = stream_bpb(A,&c,refoffs,nvalb,T,cg,mlambda,mtau,0,0,NULL);

    Cache *cu = cache_new(mslots,c.vocab,mctx);          /* control: write every token */
    long long wu=0;
    double pass1_u = stream_bpb(A,&c,adaptoff,1,adaptT,cu,mlambda,mtau,1,1,&wu);
    double pass2_u = stream_bpb(A,&c,adaptoff,1,adaptT,cu,mlambda,mtau,0,1,NULL);
    double after_u = stream_bpb(A,&c,refoffs,nvalb,T,cu,mlambda,mtau,0,1,NULL);

    fprintf(stderr,"before(no cache)      bpb=%.4f  (%d tokens, %d windows)\n",before,T*nvalb,nvalb);
    fprintf(stderr,"--- gated (surprise>tau writes only, %lld/%d = %.1f%% of positions) ---\n",
            wg,adaptT,100.0*wg/adaptT);
    fprintf(stderr,"pass1 (adapt stream, first exposure)   bpb=%.4f\n",pass1_g);
    fprintf(stderr,"pass2 (same stream, second exposure)   bpb=%.4f   forward_transfer=%+.4f\n",pass2_g,pass1_g-pass2_g);
    fprintf(stderr,"after (original val, cache populated)  bpb=%.4f   interference=%+.4f\n",after_g,after_g-before);
    fprintf(stderr,"--- control (unconditional writes, %lld/%d = 100%% of positions) ---\n",wu,adaptT);
    fprintf(stderr,"pass1  bpb=%.4f\n",pass1_u);
    fprintf(stderr,"pass2  bpb=%.4f   forward_transfer=%+.4f\n",pass2_u,pass1_u-pass2_u);
    fprintf(stderr,"after  bpb=%.4f   interference=%+.4f\n",after_u,after_u-before);

    if(jsonp){ FILE*j=fopen(jsonp,"w");
      fprintf(j,"{\"arch\":\"%s\",\"params\":%lld,\"mlambda\":%g,\"mtau\":%g,\"mslots\":%d,\"mctx\":%d,"
                "\"adapt_len\":%d,\"before_bpb\":%.5f,"
                "\"gated_writes\":%lld,\"gated_pass1_bpb\":%.5f,\"gated_pass2_bpb\":%.5f,\"gated_after_bpb\":%.5f,"
                "\"gated_forward_transfer\":%.5f,\"gated_interference\":%.5f,"
                "\"ungated_writes\":%lld,\"ungated_pass1_bpb\":%.5f,\"ungated_pass2_bpb\":%.5f,\"ungated_after_bpb\":%.5f,"
                "\"ungated_forward_transfer\":%.5f,\"ungated_interference\":%.5f}\n",
        archname,params_count(),mlambda,mtau,mslots,mctx,adaptT,before,
        wg,pass1_g,pass2_g,after_g,pass1_g-pass2_g,after_g-before,
        wu,pass1_u,pass2_u,after_u,pass1_u-pass2_u,after_u-before);
      fclose(j); }
    cache_free(cg); cache_free(cu);
    return 0;
  }
  if(!strcmp(mode,"eval")){
    arena_init(arena); A->build(&c); ckpt_load(ckpt);
    double vl=eval_val(A,&c,nvalb,B,T,KOUT,TAU);
    printf("val_loss %.5f bpb %.5f\n",vl,vl/0.6931472); return 0;
  }
  fprintf(stderr,"usage: disco {train|infer|eval|memtest|gradcheck|list} [opts]\n"); return 1;
}
