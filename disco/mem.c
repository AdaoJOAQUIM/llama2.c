#include "mem.h"
#include <stdlib.h>
#include <string.h>

Cache *cache_new(int slots, int V, int ctx) {
  Cache *c = (Cache*)calloc(1, sizeof(Cache));
  c->slots = slots; c->V = V; c->ctx = ctx;
  c->counts = (uint16_t*)calloc((size_t)slots * V, sizeof(uint16_t));
  return c;
}
void cache_free(Cache *c) { if (!c) return; free(c->counts); free(c); }
void cache_reset(Cache *c) { memset(c->counts, 0, (size_t)c->slots * c->V * sizeof(uint16_t)); c->n_write = c->n_write_gated_out = 0; }

void cache_mix(Cache *c, const int *ctxbuf, const float *core_probs, float lambda, float *out) {
  unsigned slot = cache_hash(ctxbuf, c->ctx, c->slots);
  const uint16_t *row = c->counts + (size_t)slot * c->V;
  long total = 0;
  for (int i = 0; i < c->V; i++) total += row[i];
  if (total == 0 || lambda <= 0.0f) {
    memcpy(out, core_probs, c->V * sizeof(float));
    return;
  }
  float inv = 1.0f / (float)total;
  for (int i = 0; i < c->V; i++) {
    float pcache = row[i] * inv;
    out[i] = (1.0f - lambda) * core_probs[i] + lambda * pcache;
  }
}

int cache_write(Cache *c, const int *ctxbuf, int target, double surprise_nats, double tau, int gate_always) {
  if (!gate_always && surprise_nats < tau) { c->n_write_gated_out++; return 0; }
  unsigned slot = cache_hash(ctxbuf, c->ctx, c->slots);
  uint16_t *row = c->counts + (size_t)slot * c->V;
  if (row[target] >= 60000) {                 /* saturating: halve the row before overflow */
    for (int i = 0; i < c->V; i++) row[i] >>= 1;
  }
  row[target]++;
  c->n_write++;
  return 1;
}
