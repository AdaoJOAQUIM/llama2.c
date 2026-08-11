/* mem.h -- online, writable, surprise-gated key->count cache.
 *
 * This is deliberately NOT part of the autodiff graph in nn.c: it is not
 * trained by gradient descent, it is written directly at inference time.
 * That is the entire point -- it is the one piece of this repo that models
 * a store separate from the frozen, backprop-trained core, addressing the
 * question "can something be written into a running system without touching
 * the weights that were trained offline, and does that writing help without
 * hurting."
 */
#ifndef MEM_H
#define MEM_H
#include <stdint.h>

typedef struct {
  uint16_t *counts;   /* [slots][V], saturating */
  int slots, V, ctx;  /* ctx = number of trailing bytes hashed into the key */
  long long n_write, n_write_gated_out;
} Cache;

Cache *cache_new(int slots, int V, int ctx);
void   cache_free(Cache *c);
void   cache_reset(Cache *c);
static inline unsigned cache_hash(const int *ctxbuf, int ctx, int slots) {
  unsigned h = 2166136261u;
  for (int i = 0; i < ctx; i++) { h ^= (unsigned)ctxbuf[i]; h *= 16777619u; }
  return h % (unsigned)slots;
}
/* mix core logits (already softmax'd probs, length V) with the cache's
   empirical distribution at this context, weight lambda in [0,1]. writes
   result into out (length V, a valid probability distribution). */
void cache_mix(Cache *c, const int *ctxbuf, const float *core_probs, float lambda, float *out);
/* write: increment counts[slot][target] if gated (surprise > tau) or always
   if gate_always. returns 1 if a write happened. */
int cache_write(Cache *c, const int *ctxbuf, int target, double surprise_nats, double tau, int gate_always);

#endif
