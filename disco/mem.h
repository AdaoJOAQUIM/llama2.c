/* mem.h -- online, writable, surprise-gated, REVOCABLE key->count store.
 *
 * This is deliberately NOT part of the autodiff graph in nn.c: it is not
 * trained by gradient descent, it is written directly at inference time.
 * That is the entire point -- it models a store separate from the frozen,
 * backprop-trained core.
 *
 * The engineering claim this file exists to test is not "a cache helps".
 * It is the three properties that would make continual learning deployable:
 *
 *   LOCALISED    one write touches one (slot, target) counter, nothing else.
 *   ATTRIBUTABLE every write is logged with the episode that caused it.
 *   REVOCABLE    an episode can be un-written, and the store returns to a
 *                state BIT-IDENTICAL to before that episode ran.
 *
 * Revocability is checked, not asserted: memtest re-scores after revoking and
 * requires the bits-per-byte to return to the pre-write value exactly, in the
 * same spirit as the `consistency == 0.0` check on the decode path.
 */
#ifndef MEM_H
#define MEM_H
#include <stdint.h>

typedef struct { uint32_t slot; uint16_t target; uint32_t episode; } WriteRec;

typedef struct {
  uint32_t *counts;   /* [slots][V] -- 32-bit so the saturating halve below
                         never fires at experiment scale, which is what keeps
                         revocation exactly invertible */
  int slots, V, ctx;
  WriteRec *log; long long nlog, logcap;
  int halved;         /* set if a row was ever halved: revocation is then no
                         longer bit-exact and memtest must say so */
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

/* Mix the core's probabilities with the slot's empirical distribution.
 *
 * The interpolation weight is NOT constant.  It is
 *     lambda_eff = lambda_max * n / (n + kappa)
 * where n is the total count mass already written to this slot.  A slot
 * touched once by an unlucky hash collision therefore contributes almost
 * nothing, while a slot written a hundred times contributes nearly
 * lambda_max.  This is Jelinek-Mercer style count-dependent back-off, and it
 * is the specific fix the previous negative result pointed at: with a fixed
 * lambda, a single atypical write is trusted exactly as much as a
 * well-supported statistic, which is what made interference dominate.
 *
 * kappa == 0 reproduces the old fixed-lambda behaviour exactly, so the two
 * regimes are comparable within one binary.
 */
void cache_mix(Cache *c, const int *ctxbuf, const float *core_probs,
               float lambda_max, float kappa, float *out);

/* Write, gated on surprise unless gate_always.  Returns 1 if it wrote. */
int  cache_write(Cache *c, const int *ctxbuf, int target,
                 double surprise_nats, double tau, int gate_always, uint32_t episode);

/* Un-write every record belonging to `episode`.  Returns records reverted. */
long long cache_revoke(Cache *c, uint32_t episode);

/* Total count mass and number of occupied slots -- store-size accounting. */
void cache_stats(Cache *c, long long *total_mass, long long *occupied_slots);

#endif
