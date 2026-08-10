# Autonomous architecture-discovery journal

Append-only. Entries are never rewritten after the fact. Predictions are recorded
*before* the run; verdicts after.

---

## 0. Substrate and harness

**Substrate.** `karpathy/llama2.c`, reimplemented as a self-contained C harness in
`disco/` so that a single forward function serves both training and cached decoding.
No new dependencies: pure C99 + OpenMP, `gcc -O3 -march=native`.

**Corpus.** TinyStories validation split (19.4 MB, 21,989 stories), byte-level
tokenised: vocab = 257 (256 byte values + BOS/document separator). 19,183,414 tokens.
Split 95/5 train/val by offset.

Byte-level rather than the 512-BPE tokenizer of `stories260K` on purpose: it removes
the tokenizer as a confound between architectures, and it makes the perplexity axis
directly interpretable as **bits per byte**, which is comparable across any change to
the model.

**Engine.** `nn.c` is a tape-based reverse-mode autodiff over fp32 tensors with ops:
linear, matmul, embedding, rmsnorm, add, mul, bias, scale, 6 activations, softmax,
slice, concat, STE-quantise, ternary-quantise, RoPE, fused causal GQA attention,
row-gather, row-scatter, cross-entropy.

Gradients are verified by a **directional-derivative test**: perturb every parameter
along a random Gaussian direction `d` and compare `(L(θ+εd) − L(θ−εd))/2ε` against
`⟨∇L, d⟩`. Single-coordinate finite differences were tried first and abandoned — with
an fp32 forward pass and a loss of order 1, per-coordinate gradients of order 1e-6 sit
below the numerical noise floor and produce meaningless "failures". The directional
test is O(1) in magnitude and passes at rel ≲ 1e-3 for every architecture registered.

---

## 1. The evaluator

The evaluator is the load-bearing component, so it is described in full.

Every architecture provides **one** `fwd()` used in two modes:

- `g_train=1`: full sequence, tape recorded, gradients allocated.
- `g_train=0, use_cache=1`: single step, KV/state cache, no tape.

This is the central anti-self-deception device. A separate "fast inference path"
could silently diverge from what was trained; here there is only one path, and the
harness additionally *measures* the divergence:

> **`consistency` = max |logit(full-sequence forward) − logit(cached step decode)|**
> over a 64-token prompt, reported in every run's JSON. For the baseline it is
> **exactly 0.0**. Any run with consistency > 1e-3 is flagged and refused.

### The metric vector (5 axes + diagnostics)

| axis | how it is obtained |
|---|---|
| `val_bpb` | mean NLL / ln2 over 24 held-out batches of 16×256 tokens. Offsets are drawn from a fixed seed (0xC0FFEE) so **every architecture is scored on the identical 98,304 tokens**. |
| `wbytes_per_tok` | **unique** weight bytes touched per emitted token. Each parameter tensor carries a per-row touched-flag array; ops set flags; the flags are summed and cleared after every decode step. |
| `peak_rss_kb` | `VmHWM` of an **inference-only** process. |
| `params` | counted from the parameter registry. |
| `tok_per_sec` | wall clock over 248 decoded tokens (8 discarded as warm-up), 1 thread, best of 3 repeats. |

Diagnostics also recorded: `wtraffic_per_tok` (total read events, re-reads included),
`sbytes_per_tok` (KV/state bytes), `macs_per_tok`, `stored_bytes`, `consistency`.

**Why unique bytes and not read traffic.** A weight read twice in one token costs one
memory fetch, not two. Measuring unique bytes is what makes a layer-recurrent model
legible: it has small unique bytes and large traffic, and the *gap* between the two
columns is the interesting quantity. Both are recorded.

**Measurement bugs found and fixed while building this** (each would have silently
biased results):

1. `peak_rss_kb` was 45.7 MB for every architecture because `infer` mode loaded the
   entire 38 MB corpus. Inference now reads a 2 KB window. Real footprint: 8.2 MB.
2. `touch_all()` short-circuited on an "any row touched" flag, so a **tied** embedding
   whose row had already been touched by the input lookup was never marked fully read
   — under-reporting unique bytes by 64 KB and flattering every weight-tied model.
   Fixed with a separate `tfull` flag.
3. `tok_per_sec` swung 9.4k–15.4k for the *same* checkpoint depending on concurrent
   load. Training now runs 4-way concurrent; inference benchmarks run strictly
   serialized.
4. Sample text was overwritten after benchmark repeat 0 and always came out empty.
5. `op_qrow` caches quantised weights whenever `g_train==0`, and the gradient check
   evaluates its numeric probes with `g_train==0` — so the probes never saw the
   perturbation. Symptom: 8-bit quantisation appeared to have a *worse* STE bias
   (rel 8.9e-2) than ternary (7.5e-3), which is backwards. After invalidating the
   cache on every out-of-band weight write (`ckpt_load`, `eval_val`, every gradcheck
   probe) the ordering is monotone and matches theory:

   | quantiser | states | measured STE bias (rel) |
   |---|---|---|
   | ternary, levels=1 | 3 | 2.7e-1 |
   | 4-bit, levels=7 | 15 | 1.7e-2 |
   | 8-bit, levels=127 | 255 | 6.3e-3 |

   The straight-through estimator is a deliberately biased gradient, so these
   numbers are a property of the method, not a defect. They are reported rather
   than suppressed. All eleven non-quantised architectures pass at rel < 1e-3.

**Invariant check, passing:** for a dense fp32 model, unique bytes per token must
equal `params × 4` exactly. Measured: 990,208 = 247,552 × 4. ✓
Read *traffic* is 990,464 — exactly 256 B higher, which is the tied embedding row
being read a second time as an embedding lookup. The evaluator can account for its
own numbers to the byte.

**Known limitations, stated rather than hidden:**

- At this scale the whole model (~1 MB) fits in L2/L3, so `tok_per_sec` does **not**
  respond to `wbytes_per_tok` the way it would at scale. The bytes axis is a
  structural measurement, not a predictor of speed *here*.
- `tok_per_sec` for quantised variants is measured with the quantised weights
  materialised once and then viewed (no per-token re-quantisation), but the arithmetic
  is still fp32 — a real 2-bit kernel would differ.
- The comparison is at **fixed token budget**, which favours architectures that learn
  fast. That is a property of the protocol, not a neutral fact.

---

## 2. Frozen protocol

Changing any of this invalidates cross-architecture comparison, so it is frozen:

```
steps 800   batch 16   seq 256   -> 3,276,800 training tokens
AdamW(0.9, 0.95) wd 0.1, grad-clip 1.0
warmup 100 steps, cosine decay to 10% of peak
val: 24 fixed batches = 98,304 held-out tokens
decode benchmark: 256 tokens, 1 thread, best of 3
screening seed 1337
```

Reference config (matches `stories260K` shape): dim 64, 5 layers, 8 heads,
4 KV heads, hidden 176, vocab 257 → **247,552 parameters**.

### Baseline learning-rate sweep (8 points, one seed)

| lr | 0.001 | 0.003 | 0.006 | 0.012 | 0.018 | 0.025 | 0.035 | 0.05 |
|---|---|---|---|---|---|---|---|---|
| val_bpb | 2.2152 | 1.8273 | 1.7812 | 1.7607 | **1.7496** | 1.7622 | 1.8112 | 1.8400 |

Smooth and unimodal, optimum at 0.018 with a flat basin from 0.012 to 0.025.

**`lr = 0.012` is frozen for every architecture**, because it was frozen before the
upper half of the sweep finished. The consequence is stated rather than hidden: the
baseline's own best configuration (1.7496 at lr 0.018) is **0.011 bpb better** than
the baseline at the protocol LR. Every variant is therefore compared against
**1.7496**, the best baseline observed at any learning rate. That handicaps the
variants, which is the correct direction of conservatism: a variant that wins under
this rule has really won, while a variant that loses may only be mis-tuned.

Tuning each variant's own LR would be fairer but costs a full sweep per variant
(~10 min x 8 per variant). Anything that comes close to the baseline gets an LR probe
before any claim is made about it.

---

## 3. MAP-Elites grid

Behaviour descriptor = (total parameters, unique weight bytes read per token).
Elite within a cell = lowest `val_bpb`.

Note the geometry: a dense fp32 model necessarily has `bytes = 4 × params`, so dense
models can only ever occupy one diagonal band. **The whole lower-left triangle is
reachable only by conditional computation (reading a subset of weights) or by
sub-fp32 storage.** The empty cells are therefore not arbitrary — they are a map of
the mechanisms this search has not yet used.

## Wave 0

### base_s2 — `llama`

**Hypothesis.** seed replicate for noise floor

**Prediction (recorded before the run).** n/a

```
bpb=1.7869  params=247552  uB/tok=990208  traffic/tok=990464  rss=8392KB  tok/s=15247  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```

```

Archive: new cell 4,6

<details><summary>sample</summary>

```

One day, 
Once upon a till, put it was very got of to go to rest and fun. But it was but the tarty, he loved special friends what she saw ren her mom. They can exd every became excited that sweats frew of the boy, they big friends house and friend became 
```

</details>

### base_s3 — `llama`

**Hypothesis.** seed replicate for noise floor

**Prediction (recorded before the run).** n/a

```
bpb=1.7632  params=247552  uB/tok=990208  traffic/tok=990464  rss=8388KB  tok/s=15126  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```

```

Archive: improves cell 4,6 (1.7869 -> 1.7632)

<details><summary>sample</summary>

```

Once upon a time there was a little dog. She liked to play. They found the box everyone again swing them. It who leant the felt in the easure to go mask. They are two stay fun and a spartrated and watched.

pectean a little maring was a little boud. She 
```

</details>

### base_s4 — `llama`

**Hypothesis.** seed replicate for noise floor

**Prediction (recorded before the run).** n/a

```
bpb=1.7505  params=247552  uB/tok=990208  traffic/tok=990464  rss=8396KB  tok/s=15662  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```

```

Archive: improves cell 4,6 (1.7632 -> 1.7505)

<details><summary>sample</summary>

```


Once upon a time, there was a big rock, went on the ando. But they loved to put it and never doing about the learled and the hat to thing the manoa. The ramman and down it was finished, doet on their hands aimserful blease, she saw a boat where soft to e
```

</details>

### base_s5 — `llama`

**Hypothesis.** seed replicate for noise floor

**Prediction (recorded before the run).** n/a

```
bpb=1.7767  params=247552  uB/tok=990208  traffic/tok=990464  rss=8376KB  tok/s=14737  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```

```

Archive: cell 4,6 already held by base_s4 at 1.7505

<details><summary>sample</summary>

```

Once there was a tuc. They like thes told her fairisived to play with her morais. They were ran indrever and ran a preter.

John was very playy wone a old bad. She was very eyes any pile. It is never and saw she could help her mom. They ran that she stre
```

</details>


---

## 4. Noise floor (wave 0)

Five seeds of the baseline under the frozen protocol:

| seed | 2 | 3 | 4 | 5 | 1337 |
|---|---|---|---|---|---|
| val_bpb | 1.7869 | 1.7632 | 1.7505 | 1.7767 | 1.7607 |

**mean 1.7676, sd 0.0143, range 0.0364, 2sd = 0.0285.**

> **Decision rule for the rest of the search: a variant has beaten the baseline only
> if it reaches val_bpb < 1.7390 (mean - 2sd). Anything between 1.739 and 1.796 is
> indistinguishable from the baseline and will be reported as such.**

This also settles the learning-rate worry from section 2 empirically: the gap between
lr 0.012 (1.7607) and the sweep optimum lr 0.018 (1.7496) is 0.011 bpb, which is
**under one standard deviation of seed noise**. The protocol LR is not costing the
baseline anything measurable, and the earlier decision to compare against 1.7496
was over-cautious rather than wrong.

Precision of the other axes, measured across the same five identical models:

| axis | spread across 5 identical runs | usable resolution |
|---|---|---|
| `params`, `wbytes_per_tok`, `macs_per_tok` | exactly 0 | exact |
| `peak_rss_kb` | 8376–8396 (±0.12%) | differences above ~50 KB are real |
| `tok_per_sec` | 14737–15662 (±3%) | differences below ~6% are noise |
| `val_bpb` | sd 0.0143 | differences below 0.029 are noise |

`params`, `wbytes_per_tok` and `macs_per_tok` are deterministic functions of the
architecture, so they carry no noise at all — which is precisely why they, and not
`val_bpb`, are the grid's behaviour axes.
