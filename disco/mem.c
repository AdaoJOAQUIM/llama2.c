#include "mem.h"
#include <stdlib.h>
#include <string.h>

Cache *cache_new(int slots, int V, int ctx) {
  Cache *c = (Cache*)calloc(1, sizeof(Cache));
  c->slots = slots; c->V = V; c->ctx = ctx;
  c->counts = (uint32_t*)calloc((size_t)slots * V, sizeof(uint32_t));
  c->logcap = 1 << 16;
  c->log = (WriteRec*)malloc(sizeof(WriteRec) * c->logcap);
  return c;
}
void cache_free(Cache *c) { if (!c) return; free(c->counts); free(c->log); free(c); }
void cache_reset(Cache *c) {
  memset(c->counts, 0, (size_t)c->slots * c->V * sizeof(uint32_t));
  c->nlog = 0; c->halved = 0; c->n_write = c->n_write_gated_out = 0;
}

void cache_mix(Cache *c, const int *ctxbuf, const float *core_probs,
               float lambda_max, float kappa, float *out) {
  unsigned slot = cache_hash(ctxbuf, c->ctx, c->slots);
  const uint32_t *row = c->counts + (size_t)slot * c->V;
  double total = 0;
  for (int i = 0; i < c->V; i++) total += row[i];
  if (total <= 0 || lambda_max <= 0.0f) {
    memcpy(out, core_probs, c->V * sizeof(float));
    return;
  }
  /* count-dependent trust: one observation is not worth the same as a hundred */
  float lam = (float)(lambda_max * (total / (total + (double)kappa)));
  float inv = (float)(1.0 / total);
  for (int i = 0; i < c->V; i++)
    out[i] = (1.0f - lam) * core_probs[i] + lam * (row[i] * inv);
}

int cache_write(Cache *c, const int *ctxbuf, int target,
                double surprise_nats, double tau, int gate_always, uint32_t episode) {
  if (!gate_always && surprise_nats < tau) { c->n_write_gated_out++; return 0; }
  unsigned slot = cache_hash(ctxbuf, c->ctx, c->slots);
  uint32_t *row = c->counts + (size_t)slot * c->V;
  if (row[target] >= 0xF0000000u) {          /* safety valve; breaks bit-exact revocation */
    for (int i = 0; i < c->V; i++) row[i] >>= 1;
    c->halved = 1;
  }
  row[target]++;
  if (c->nlog >= c->logcap) {
    c->logcap *= 2;
    c->log = (WriteRec*)realloc(c->log, sizeof(WriteRec) * c->logcap);
  }
  c->log[c->nlog].slot = slot;
  c->log[c->nlog].target = (uint16_t)target;
  c->log[c->nlog].episode = episode;
  c->nlog++;
  c->n_write++;
  return 1;
}

long long cache_revoke(Cache *c, uint32_t episode) {
  long long n = 0, keep = 0;
  /* reverse order so the store retraces exactly the path it came by */
  for (long long i = c->nlog - 1; i >= 0; i--) {
    if (c->log[i].episode != episode) continue;
    uint32_t *row = c->counts + (size_t)c->log[i].slot * c->V;
    if (row[c->log[i].target] > 0) row[c->log[i].target]--;
    n++;
  }
  for (long long i = 0; i < c->nlog; i++)
    if (c->log[i].episode != episode) c->log[keep++] = c->log[i];
  c->nlog = keep;
  return n;
}

void cache_stats(Cache *c, long long *total_mass, long long *occupied_slots) {
  long long m = 0, occ = 0;
  for (int s = 0; s < c->slots; s++) {
    const uint32_t *row = c->counts + (size_t)s * c->V;
    long long rm = 0;
    for (int i = 0; i < c->V; i++) rm += row[i];
    if (rm) { occ++; m += rm; }
  }
  if (total_mass) *total_mass = m;
  if (occupied_slots) *occupied_slots = occ;
}
