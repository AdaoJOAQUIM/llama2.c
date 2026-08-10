#ifndef MODEL_H
#define MODEL_H
#include "nn.h"

#define NKNOB 24
typedef struct {
  int dim, n_layers, n_heads, n_kv_heads, hidden_dim, vocab, seq_len, tie;
  float rope_theta;
  /* generic named knobs so the search loop can sweep without recompiling */
  char  kname[NKNOB][24];
  float kval[NKNOB];
  int   nknob;
} Cfg;

float cfg_get(Cfg *c, const char *k, float dflt);
int   cfg_geti(Cfg *c, const char *k, int dflt);

typedef struct {
  const char *name;
  const char *desc;
  void    (*build)(Cfg*);
  void    (*cache_alloc)(Cfg*, int maxlen);
  void    (*cache_reset)(Cfg*);
  /* tokens: B*T ids. pos0: absolute position of first step. cache!=0 => streaming.
     returns logits [B*T, vocab] */
  Tensor* (*fwd)(Cfg*, int *tok, int B, int T, int pos0, int use_cache);
} Arch;

extern int g_casc_mode;
extern Arch  g_archs[];
extern int   g_narchs;
Arch *arch_find(const char *name);

#endif
