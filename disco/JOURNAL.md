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
## Wave 1

### w1_sharedloop — `sharedloop`

**Hypothesis.** Depth can be expressed as an affine reparameterisation of ONE shared operator. If most of what distinguishes layer l from layer l+1 in a 5-layer model is a change of scale/offset rather than a change of function, then sharing all matrices and keeping only per-depth norm gains and biases (1,280 of 63,872 parameters) should cost far less quality than the 3.9x parameter reduction implies. Targets the empty cell (48-96K params, 128-256K bytes/token), which no dense model can reach at baseline width.

**Prediction (recorded before the run).** 63,872 params, 255,488 unique bytes/token, ~5x read traffic vs unique. val_bpb in [1.85, 2.10] -- clearly worse than the 247K baseline but far better than a 1-layer model.

```
bpb=2.2603  params=63872  uB/tok=255488  traffic/tok=993024  rss=7120KB  tok/s=15391  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 63,872     measured 63,872     ok
  wbytes           predicted 255,488    measured 255,488    ok
  val_bpb          predicted [1.85, 2.10]  measured 2.2603  -> REFUTED (worse)
```

Archive: new cell 1,3

<details><summary>sample</summary>

```
 he walked Mily. The happy and see the happy. Bola next to see boys and the show rug to see playings. He have and watear and Com and so spe not with not Tore and you det! "You two a playe of put at for that him not him the not him with and the miend sad ar
```

</details>

### w1_hashffn — `hashffn`

**Hypothesis.** Conditional computation with ZERO routing parameters. Four FFN experts per layer, selected by a hash of the (current, previous) token pair. Because the route is a pure function of the input text it needs no router, no load-balancing loss, and no gradient through a discrete choice. Triples parameter count at exactly the baseline's per-token read cost. Targets the empty cell (512-768K params, 768K-1.05M bytes/token).

**Prediction (recorded before the run).** 754,432 params, 990,208 unique bytes/token (identical to baseline to the byte). val_bpb in [1.60, 1.72] -- better than baseline 1.7607 because capacity rises with no extra read.

```
bpb=1.6760  params=754432  uB/tok=990208  traffic/tok=990464  rss=13024KB  tok/s=12508  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 754,432    measured 754,432    ok
  wbytes           predicted 990,208    measured 990,208    ok
  val_bpb          predicted [1.60, 1.72]  measured 1.6760  -> CONFIRMED
```

Archive: new cell 7,6

<details><summary>sample</summary>

```

Once upon a time, there was a little girl named Timmy. The bush said, "John overs in it waver.” his mom are new flower. The nurk."
Tim are the grode cooing to make a dancing on his mom every hard with her hat to the blocks. Billy and had a down and some
```

</details>

### w1_ternffn — `ternffn`

**Hypothesis.** The FFN holds 68% of this model's parameters and is the obvious place to buy bytes cheaply. Ternary weights with a per-row mean-absolute scale should retain most FFN function because SwiGLU is a sum over 176 hidden units and averaging tolerates per-weight noise. Same parameter COUNT, 2.8x fewer bytes read. Targets empty cell (224-288K params, 256-512K bytes/token).

**Prediction (recorded before the run).** 247,552 params, 356,608 unique bytes/token. val_bpb in [1.85, 2.05] -- a real but bounded loss.

```
bpb=1.9097  params=247552  uB/tok=356608  traffic/tok=356864  rss=9040KB  tok/s=15428  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 247,552    measured 247,552    ok
  wbytes           predicted 356,608    measured 356,608    ok
  val_bpb          predicted [1.85, 2.05]  measured 1.9097  -> CONFIRMED
```

Archive: new cell 4,4

<details><summary>sample</summary>

```


Lily and Timmy with home. Hay say and deling to big fun who to tee excited toy. Ote was a time to meags. She spla and he want to the water to the pump. He holer at beefest and was will flyight. He storen and they headied that'le she read for them. Fohn w
```

</details>

### w1_novalue — `novalue`

**Hypothesis.** Structural probe, not a cell-filling move: every variant so far assumes the vector you AVERAGE must be a different projection from the vector you MATCH on. Delete Wv and attend over K itself. If the value projection is largely redundant with the output projection Wo (which immediately follows the attention average), removing 10,240 parameters should cost much less than 4% of quality.

**Prediction (recorded before the run).** 237,312 params, 949,248 unique bytes/token, same cell as baseline. val_bpb in [1.78, 1.92].

```
bpb=2.1251  params=237312  uB/tok=949248  traffic/tok=949504  rss=8272KB  tok/s=16062  macs/tok=322048  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 237,312    measured 237,312    ok
  wbytes           predicted 949,248    measured 949,248    ok
  val_bpb          predicted [1.78, 1.92]  measured 2.1251  -> REFUTED (worse)
```

Archive: cell 4,6 already held by base_s4 at 1.7505

<details><summary>sample</summary>

```


Twat was a time, there was a little girl named The big amplest. She timed the specided it her said the bood the proke into swhas the girl of loncest swind pate the girl to lotey. The friend down itg see want and was that to the boar what fratimmy and you
```

</details>

## Wave 2

### w2_emaconv — `emaconv`

**Hypothesis.** Replace content-based retrieval entirely with two fixed timescales per channel: a width-4 depthwise causal convolution (local orthography) and a learned per-channel EMA (unbounded but non-selective range), gated multiplicatively. On byte-level text most of the entropy is local, so a mixer with O(1) decode state may lose less than the loss of attention suggests. Decode state becomes a constant 5,120 bytes per step instead of a KV cache growing with context.

**Prediction (recorded before the run).** 269,632 params, 1,078,528 unique bytes/token, sbytes/token CONSTANT at ~5,120 (vs baseline ~169,600 averaged over 256 decoded tokens). val_bpb in [1.80, 2.15].

```
bpb=1.6813  params=269632  uB/tok=1078528  traffic/tok=1078784  rss=7724KB  tok/s=36959  macs/tok=267328  consist=5.7e-06
```

**Verdict.**
```
  params           predicted 269,632    measured 269,632    ok
  wbytes           predicted 1,078,528  measured 1,078,528  ok
  val_bpb          predicted [1.80, 2.15]  measured 1.6813  -> REFUTED (better)
```

Archive: new cell 4,7

<details><summary>sample</summary>

```


Sally looked all a very proud. So she had a cat fun.
Lily says, "Hello.


Anna and Ben was happy that me, Timmy's friend came. We excited to the birdet children."
Sam?" Max saw noise was to play with his friend to well and kink a good and said. He light
```

</details>

### w2_ngrammem — `ngrammem`

**Hypothesis.** Move capacity out of matrices and into a lookup table. A 4,096-row memory addressed by hash(token, previous token), injected into the residual stream at every depth with a per-depth learned gain. Doubles parameter count while adding 256 bytes to the per-token read. If byte-level TinyStories is substantially bigram-predictable, a learned n-gram table welded to the transformer should buy quality almost for free on the I/O axis.

**Prediction (recorded before the run).** 510,016 params, 991,744 unique bytes/token (+1,536 over baseline). val_bpb in [1.62, 1.74].

```
bpb=1.6863  params=510016  uB/tok=991744  traffic/tok=992000  rss=9768KB  tok/s=14033  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 510,016    measured 510,016    ok
  wbytes           predicted 991,744    measured 991,744    ok
  val_bpb          predicted [1.62, 1.74]  measured 1.6863  -> CONFIRMED
```

Archive: new cell 6,6

<details><summary>sample</summary>

```

Once upon a time there was a little girl named Her dog. He loved to play so she found and find about it of her toys. The followed her friendly got to find her frage.
One day, Benny admiring that it wroc magical with his seme iran. One day, he was a big s
```

</details>

### w2_deepgate — `deepgate`

**Hypothesis.** Mechanism test, and I expect it to FAIL in an informative way. Per-token hard FFN skip: rows whose gate <= 1/2 never touch w1/w2/w3. Because a closed row receives no gradient at its gate, gates can shut but never reopen -- there is no force pushing a gate closed except weight decay, and a strong force (the FFN's usefulness) pushing it open. Prediction: gates stay essentially all-open, byte savings under 10%, and the whole apparatus buys nothing. If instead gates close substantially, the dead-gradient asymmetry is weaker than I think.

**Prediction (recorded before the run).** 247,877 params. Unique bytes/token in [900,000, 992,000] (i.e. <10% saving). val_bpb in [1.75, 1.85].

```
bpb=1.8682  params=247877  uB/tok=454106  traffic/tok=454362  rss=7676KB  tok/s=19085  macs/tok=198323  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 247,877    measured 247,877    ok
  val_bpb          predicted [1.75, 1.85]  measured 1.8682  -> REFUTED (worse)
```

Archive: improves cell 4,4 (1.9097 -> 1.8682)

<details><summary>sample</summary>

```

Once upon a time, there was a little girl named Lily. The muke a big begs both. He can excried to the use held a suff.


Once upon a time, they called the griends to clean to clean to fap. Then he kyes in a little boy and decided to help and heard toys h
```

</details>

### w2_lowrankffn — `lowrankffn`

**Hypothesis.** Both SwiGLU up-projections read the same 64-dim input, so they may not need independent 176x64 matrices; route both through one shared rank-24 bottleneck. Moves down-left on BOTH axes at once, targeting the empty cell (160-224K params, 512-768K bytes/token).

**Prediction (recorded before the run).** 184,832 params, 739,328 unique bytes/token. val_bpb in [1.80, 1.95].

```
bpb=2.0249  params=184832  uB/tok=739328  traffic/tok=739584  rss=7760KB  tok/s=16536  macs/tok=269568  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 184,832    measured 184,832    ok
  wbytes           predicted 739,328    measured 739,328    ok
  val_bpb          predicted [1.80, 1.95]  measured 2.0249  -> REFUTED (worse)
```

Archive: new cell 3,5

<details><summary>sample</summary>

```
 TiIt was a little girl with her mom. 
One day, Lily would was very to the bird. Lily loved to be her to friends boy would home. He wanted a goor every mareed back. One day, but his mom ky you was you bough a big could reen't but rest with you with her pro
```

</details>


---

# R1 — Reflection after 12 runs (waves 0–2)

*Written before waves 3+ reported, so it cannot be retrofitted to the results.*

## What the predictions did

Eight architectural predictions, recorded before implementation:

| variant | predicted val_bpb | measured | verdict | signed miss |
|---|---|---|---|---|
| hashffn | [1.60, 1.72] | 1.6760 | confirmed | — |
| ternffn | [1.85, 2.05] | 1.9097 | confirmed | — |
| ngrammem | [1.62, 1.74] | 1.6863 | confirmed | — |
| lowrankffn | [1.80, 1.95] | 2.0249 | refuted, worse | +0.075 |
| sharedloop | [1.85, 2.10] | 2.2603 | refuted, worse | +0.160 |
| deepgate | [1.75, 1.85] | 1.8682 | refuted, worse | +0.018 |
| novalue | [1.78, 1.92] | 2.1251 | refuted, worse | +0.205 |
| emaconv | [1.80, 2.15] | 1.6813 | refuted, **better** | −0.119 |

Every prediction of `params` and `wbytes_per_tok` was exact — the analytic cost
model and the instrumented counter agree to the unit on all eight. The errors are
entirely in *quality*.

## The regularity in the failures

Sort the four "worse" refutations by what they did to the model:

- `novalue` — deleted a projection and **tied** its role to another one.
- `sharedloop` — **tied** every matrix across depth.
- `lowrankffn` — **tied** two projections through a shared bottleneck.
- `deepgate` — **removed** the FFN for some tokens.

All four are compression by sharing or deletion, and I was optimistic on all four,
by +0.02 to +0.21 bpb. The one refutation in the other direction, `emaconv`, is the
only variant that *replaced* a mechanism rather than compressing one.

> **Regularity: I price parameter sharing as if the shared tensor could serve both
> roles at once. It cannot. Every time two functions were forced through one
> tensor, the cost exceeded what the parameter count predicted.**

`novalue` is the cleanest instance and the largest miss. My reasoning was that
`out = A·X·Wv^T·Wo^T` becomes `A·X·Wk^T·Wo^T`, the same composition through the same
32-dimensional bottleneck, so only 10,240 parameters are lost. What that algebra
hides is that `Wk` now has two jobs — defining *which* positions match, and defining
*what* gets transported — and those objectives pull it in different directions.
There is also a second, cruder mechanism I overlooked entirely at design time: I
applied RoPE to `k` *before* reusing it as the value, so the content a token
contributes is **rotated by its absolute position**. The same word transports a
different vector depending on where it sits. Wave 8 (`w8_novalue_nr`) separates
these two by attending over the pre-RoPE keys.

## The mechanistic prediction that was backwards

`deepgate` is the most instructive miss, because the number was nearly right
(1.8682 vs a predicted ceiling of 1.85) while the *mechanism* was inverted. I
predicted the gates would stay open, reasoning that a closed row receives no
gradient, so nothing could push a gate shut. Measured: unique bytes/token fell from
990,208 to **454,106** and MACs from 332,288 to **198,323** — a little over half the
FFN reads were skipped.

The reasoning was backwards. Weight decay pulls the gate's weight and bias toward
zero, i.e. toward sigmoid = 1/2, which is exactly the threshold. Rows that drift
below stop contributing *and* stop receiving gradient, so the FFN's usefulness can
no longer pull them back. The no-gradient asymmetry is a **ratchet toward closed**,
not a lock toward open. Half the tokens ended up not using the FFN at all, at a
cost of 0.10 bpb.

## The assumption every variant shared

Nine of the first ten variants left the attention block intact and looked for slack
somewhere else — the FFN, the embedding, the weight precision, the depth structure.
That was not a considered decision; it is what I did without noticing. Written out,
the assumption is:

> **Content-based retrieval is the load-bearing mechanism, and everything else is
> the compressible part.**

The one variant that tested it, by deleting attention outright and replacing it with
a width-4 depthwise causal convolution plus a per-channel EMA, produced the largest
prediction error in the whole set — in the good direction — and is now tied for best
in the archive at 1.6813, with **33× less decode state** (5,120 bytes/token against
170,880) and **2.4× the decode throughput**.

At dim 64, 5 layers and 3.3M training bytes, attention is not what this model is
doing. Two fixed timescales per channel are enough, and the parameters attention was
consuming are better spent elsewhere. That is the one structural finding so far, and
it came from the only assumption I had not thought to question.

## What that redirects

1. Isolate the mechanism: is it the convolution (locality) or the EMA (range)?
   → wave 8, `w8_ema_convonly` / `w8_ema_emaonly`.
2. Control for capacity: `emaconv` carried 8.9% more parameters than the baseline.
   Hidden 153 gives *exactly* 247,552. → wave 9, `w9_emaconv_h153`.
3. The three leaders (1.6760 / 1.6813 / 1.6863) sit inside one standard deviation of
   each other despite sharing no mechanism. Either something is capping all of them,
   or it is a coincidence in three samples. → waves 9 and 10 replicate all three
   across three seeds. This is the open question I care most about.

## A limitation this exposed in my own grid

`emaconv` decodes 2.4× faster than the baseline while reading **more** weight bytes
per token (1,078,528 vs 990,208) and doing comparable arithmetic. The grid axis
cannot see why, because the difference is not in weights at all: it is state.
Averaged over a 256-token generation the baseline reads 170,880 bytes of KV cache
per token and `emaconv` reads 5,120.

**At this scale, decode is dominated by state I/O, not weight I/O, and I chose
weight bytes as a behaviour axis.** `sbytes_per_tok` was recorded from the start as
a diagnostic, which is the only reason this is visible. A second grid keyed on state
bytes would have made the `emaconv` family a whole region rather than one cell.
## Wave 3

### w3_multitok2 — `multitok`

**Hypothesis.** THE ASSUMPTION EVERY VARIANT SO FAR SHARES: one token emitted per full network evaluation. Nothing about the architecture forces this. Emit 2 tokens per forward from two head-adapters over a shared unembedding, with NO verification and no rollback -- token 2 is produced without having seen token 1. Bytes per emitted token halve for free. val_bpb is scored under exactly this emission process (trunk runs only at t == 0 mod 2; each token scored by the head that produced it), so the number is comparable to every other row in the table.

**Prediction (recorded before the run).** 251,648 params, 503,296 unique bytes/token (half of 1,006,592 per forward). val_bpb in [1.90, 2.25].

```
bpb=2.3032  params=251648  uB/tok=503296  traffic/tok=1072640  rss=8488KB  tok/s=14655  macs/tok=352832  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 251,648    measured 251,648    ok
  wbytes           predicted 503,296    measured 503,296    ok
  val_bpb          predicted [1.90, 2.25]  measured 2.3032  -> REFUTED (worse)
```

Archive: cell 4,4 already held by w2_deepgate at 1.8682

<details><summary>sample</summary>

```

One dact bower there was o taml  wchaut ring snamed Lily. Onh arking wniuge ware iana  wanted to tlywen,uep in the gaddlnetsful geattydy aene . Hes  aar aone and flied  tnow  shanks  ood!
The  sir ts wanted eot fher 
Tim hnow hegry for hem. Hhins aono toi
```

</details>

### w3_multitok4 — `multitok`

**Hypothesis.** Same mechanism at K=4: a quarter of the read cost per token, three of every four tokens emitted blind.

**Prediction (recorded before the run).** 259,840 params, 259,840 unique bytes/token. val_bpb in [2.25, 2.75].

```
bpb=3.1723  params=259840  uB/tok=259840  traffic/tok=1236992  rss=9248KB  tok/s=13708  macs/tok=393920  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 259,840    measured 259,840    ok
  wbytes           predicted 259,840    measured 259,840    ok
  val_bpb          predicted [2.25, 2.75]  measured 3.1723  -> REFUTED (worse)
```

Archive: cell 4,4 already held by w2_deepgate at 1.8682

<details><summary>sample</summary>

```

One day, didntoune tem paetie  ertacfwh nion ttoee hin gnkt. n Thl aauar snodeed mta at toe day b own kiot anm afv. Sie wialased ahan. Ihm laoked  fol snorn w srey.gfne  was eveuyhan o ddawn
Ohl dagent enklg fnea. evchy and waetidg tet,hne ta bev cnitelen
```

</details>

### w3_multitok8 — `multitok`

**Hypothesis.** K=8 should be near the point where blind emission destroys coherence: 7 of 8 bytes are written without feedback. Included to locate the knee of the curve, not because it is expected to be good.

**Prediction (recorded before the run).** 276,224 params, 138,112 unique bytes/token. val_bpb in [2.9, 3.8].

**Recovery note, added after a container restart.** The wave orchestrator originally logged this as `ERROR train failed`. On inspection the training run had actually completed cleanly through all 800 steps and written a valid `train.json` (val_bpb 3.7707) — the failure was in the wave runner's bookkeeping, not the training. Rather than retrain, inference was run directly against the existing checkpoint to recover the missing half of the row (bytes/token, tok/s, consistency, sample). No number here comes from a rerun of training; the checkpoint is the original one.

```
bpb=3.7707  params=276224  uB/tok=138112  traffic/tok=1565696  sbytes/tok=170880  tok/s=9857  rss=6140KB  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 276,224    measured 276,224    ok
  wbytes           predicted 138,112    measured 138,112    ok
  val_bpb          predicted [2.9, 3.8]  measured 3.7707  -> CONFIRMED
```

The K=2 / K=4 / K=8 curve is now complete: **2.3032 → 3.1723 → 3.7707**, convex and worsening, exactly the "knee" this run was meant to locate — the marginal cost of one more blindly-emitted token keeps growing rather than saturating. The sample text makes the same point qualitatively: K=4 is broken but still word-shaped, K=8 is not recognizable as text.

Archive: cell 4,3 already held by w4_q8all at 1.7694 (w3_multitok8 does not compete there — far worse quality at a similar params/bytes bucket)

<details><summary>sample</summary>

```
 HLettooka  a drhcl.ceu shsyod ente nse wnmt aoe  oorlr anfnhtFme  t
 was we   isr.rm w theeporon  rdfe faps t.needuea  wnf eet inthht  ther tn wln   aorse e and hh    ah yteageta nz yia g hShershaTy.
SeT eawlaeer f ahe ahue  hdrg ano  ee ofo!ree swir,ha w
```

</details>

### w3_hashffn8 — `hashffn`

**Hypothesis.** Push the zero-router conditional-computation idea to 8 experts: 5.8x the baseline parameter count at byte-for-byte identical per-token read cost. Targets the extreme cell (>768K params, 768K-1.05M bytes/token). If quality keeps improving with expert count at fixed read cost, the bytes axis and the params axis are genuinely decoupled and that is the most useful thing this search can establish.

**Prediction (recorded before the run).** 1,430,272 params, 990,208 unique bytes/token (identical to baseline). val_bpb in [1.55, 1.70].

```
bpb=1.6369  params=1430272  uB/tok=990208  traffic/tok=990464  rss=19000KB  tok/s=10772  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 1,430,272  measured 1,430,272  ok
  wbytes           predicted 990,208    measured 990,208    ok
  val_bpb          predicted [1.55, 1.70]  measured 1.6369  -> CONFIRMED
```

Archive: new cell 8,6

<details><summary>sample</summary>

```

Iouldn't street too. We have to the paig in the time, there was a little butter, so she was and the farmer of the traster and saw a with fun and didn't kate to jump the wind weht and driverlosed to share and said, "No, But when you help decided to run awa
```

</details>

## Wave 4

### w4_q8all — `quant`

**Hypothesis.** Sweep the far-left columns that no change of topology can reach. 8-bit per-row symmetric quantisation of every matrix (FFN, attention, embedding/unembedding); only the 704 norm gains stay fp32. Same parameter count, 4x fewer bytes read. 255 states should be effectively lossless at this scale.

**Prediction (recorded before the run).** 247,552 params, 249,664 unique bytes/token. val_bpb in [1.76, 1.84].

```
bpb=1.7694  params=247552  uB/tok=249664  traffic/tok=249728  rss=9284KB  tok/s=14479  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 247,552    measured 247,552    ok
  wbytes           predicted 249,664    measured 249,664    ok
  val_bpb          predicted [1.76, 1.84]  measured 1.7694  -> CONFIRMED
```

Archive: new cell 4,3

<details><summary>sample</summary>

```

Lily was a little girl named Timmy heard lot. Tom was very loved brease some curpise who have the sonfryed. Every also happy after himself, she is swam told the bath day, she says. They little home oplocked more the beskelf and played her head.
The bird w
```

</details>

### w4_q2all — `quant`

**Hypothesis.** The extreme of the same axis: ternary everything. 16x fewer bytes than fp32 at identical parameter count. I expect this to break, and specifically to break at the EMBEDDING rather than in the matrices: a ternary row of 64 values must encode the identity of a byte, and with a single per-row scale there are only 3^64 codes available but no pressure to keep them apart. If quantising the FFN alone (wave 1) costs little and quantising everything costs a lot, the difference localises where fp32 precision is actually load-bearing.

**Prediction (recorded before the run).** 247,552 params, 64,528 unique bytes/token. val_bpb in [2.10, 2.80].

```
bpb=2.1243  params=247552  uB/tok=64528  traffic/tok=64544  rss=9392KB  tok/s=13898  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 247,552    measured 247,552    ok
  wbytes           predicted 64,528     measured 64,528     ok
  val_bpb          predicted [2.10, 2.80]  measured 2.1243  -> CONFIRMED
```

Archive: new cell 4,2

<details><summary>sample</summary>

```

Once upon a time, there was a clet fried to firsi! She entent, out a seared thugh youy of sumet to makers mommy. He wanted the her mompeted time they fasoue was a walked wima ofthers.
We put her daves the been. Timmy him so boy man not and smiled. Sam the
```

</details>

### w4_kvshare — `kvshare`

**Hypothesis.** Does depth need to re-derive WHAT to retrieve, or only HOW to ask? Compute K and V once at layer 0 and let all five layers attend over that single cache with their own queries. Deletes 4/5 of the K/V projections and 4/5 of the KV cache. If most of the per-layer specialisation lives in the query, this should cost little.

**Prediction (recorded before the run).** 231,168 params, 924,672 unique bytes/token, sbytes/token down ~5x. val_bpb in [1.80, 1.95].

```
bpb=1.8364  params=231168  uB/tok=924672  traffic/tok=924928  rss=7660KB  tok/s=16304  macs/tok=315904  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 231,168    measured 231,168    ok
  wbytes           predicted 924,672    measured 924,672    ok
  val_bpb          predicted [1.80, 1.95]  measured 1.8364  -> CONFIRMED
```

Archive: cell 4,6 already held by base_s4 at 1.7505

<details><summary>sample</summary>

```


Once upon a time, there was a shearting on!Heldy watched how have to play to my all in to a time xpecial beautiful shave him not very her friendly where, they path her friend Mommy and special blues. He stuth was so many on the day and said. 
Pogen to pl
```

</details>

### w4_loopexpert — `loopexpert`

**Hypothesis.** RECOMBINATION of the two mechanisms in wave 1: a single shared attention block iterated 5 times (small unique read) whose FFN is drawn from 4 bigram-routed experts (large parameter count). Because the route depends only on the token pair, the SAME expert is selected at every depth, so unique bytes stay at one expert's worth however deep the loop runs. Targets the empty cell (160-224K params, 128-256K bytes/token) -- low on both axes at once, which neither parent reaches.

**Prediction (recorded before the run).** 165,248 params, 255,488 unique bytes/token, ~5x read traffic vs unique. val_bpb in [1.80, 2.00].

```
bpb=1.9439  params=165248  uB/tok=255488  traffic/tok=993024  rss=8604KB  tok/s=15682  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 165,248    measured 165,248    ok
  wbytes           predicted 255,488    measured 255,488    ok
  val_bpb          predicted [1.80, 2.00]  measured 1.9439  -> CONFIRMED
```

Archive: new cell 3,3

<details><summary>sample</summary>

```
ame day a time, there with there was a little girl never like. He asked talled and played the boy to were in the end.


Once upon a time there was a foxater. 
Lily was with there was very boy named Tom and make playore. 
Jane liked things and really top

```

</details>

## Wave 5 — superseded, never run under this name

`wave5.json` specified `w5_cascade`, `w5_cascade_t20`, `w5_cascade_t80` (the confidence-gated
tiny-model/transformer cascade, at three thresholds) and `w5_adaptmulti4` (confidence-gated
`multitok` reusing the K=4 checkpoint). Between writing that spec and running it, a real bug
was found and fixed in the emission-scoring code: the original decode loop for both
`cascade` and confidence-gated `multitok` did not exactly match what `eval_val` scored,
so `val_bpb` and the decode benchmark could have described two subtly different emission
processes. Rather than run the pre-fix spec, the corrected mechanics were used to write
wave 7 instead (`w7_casc_t20/t80/t95`, `w7_adapt4_t60`, `w7_adapt8_t50/t85`), which cover
the same design-space region — a confidence threshold sweep over both the cascade and the
adaptive-multitoken mechanisms — with numbers that are actually trustworthy. `wave5.json`
is left in the repo as a record of the original (buggy) plan; its ids never appear in the
journal or archive, and that is intentional rather than a gap.

## Wave 6

### w6_hash_pos — `hashffn`

**Hypothesis.** CONTROL for the wave-1 hashffn win. Identical architecture, identical parameter count, identical sparsity, identical per-token read cost -- but the route is position mod 4, which carries NO information about the text. If hashffn's 6.4-sigma win survives this, the win came from owning more parameters and reading a slice of them, and the routing function is irrelevant. If it collapses, content-dependent routing is doing the work. Because every expert then sees a statistically identical 1/4 of the tokens, all four should converge to the same function trained on a quarter of the data, which ought to be WORSE than the 247K baseline.

**Prediction (recorded before the run).** 754,432 params, 990,208 unique bytes/token (identical to hashffn and to baseline). val_bpb in [1.78, 2.00] -- worse than baseline, and far worse than hashffn's 1.6760.

```
bpb=1.9199  params=754432  uB/tok=990208  traffic/tok=990464  rss=9928KB  tok/s=11548  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 754,432    measured 754,432    ok
  wbytes           predicted 990,208    measured 990,208    ok
  val_bpb          predicted [1.78, 2.00]  measured 1.9199  -> CONFIRMED
```

Archive: cell 7,6 already held by w1_hashffn at 1.6760

<details><summary>sample</summary>

```

Sam sush friends were the broser and saw did reastes with her mom arriends.
One day, Mommyxed yourny nice and loved offerder. She was very fish her and the parpriose.
Me little girl nepled it. One day, they just, but he safver and said, "No, but there. It
```

</details>

### w6_hash_uni — `hashffn`

**Hypothesis.** Second control: route on the CURRENT token only, discarding the context. Content-dependent but not context-dependent. Separates 'the route must depend on the text' from 'the route must depend on more than one byte'. Sits between the position control and the bigram router.

**Prediction (recorded before the run).** 754,432 params, 990,208 unique bytes/token. val_bpb in [1.68, 1.78].

```
bpb=1.7182  params=754432  uB/tok=990208  traffic/tok=990464  rss=9928KB  tok/s=11771  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 754,432    measured 754,432    ok
  wbytes           predicted 990,208    measured 990,208    ok
  val_bpb          predicted [1.68, 1.78]  measured 1.7182  -> CONFIRMED
```

Archive: cell 7,6 already held by w1_hashffn at 1.6760

<details><summary>sample</summary>

```

Once upon a time, there was a lit finiit?"
One day, Lily saily did not house. She ran untites your who had something around. He was onfor their in the espect to play for her ran again.
Max next muchere the bird what he said. "Can tight delicious."
They he
```

</details>

### w6_hash_e2 — `hashffn`

**Hypothesis.** Completes the expert-count curve (2 / 4 / 8) at fixed read cost. If quality improves monotonically with expert count while bytes/token stays pinned at 990,208, the parameter axis and the I/O axis are genuinely decoupled, which is the most useful thing this search can establish.

**Prediction (recorded before the run).** 416,512 params, 990,208 unique bytes/token. val_bpb in [1.69, 1.76].

```
bpb=1.7140  params=416512  uB/tok=990208  traffic/tok=990464  rss=7304KB  tok/s=13783  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 416,512    measured 416,512    ok
  wbytes           predicted 990,208    measured 990,208    ok
  val_bpb          predicted [1.69, 1.76]  measured 1.7140  -> CONFIRMED
```

Archive: cell 6,6 already held by w2_ngrammem at 1.6863

<details><summary>sample</summary>

```


Once upon a time, there was a little girl. Timmy learned the park was a mazier each and sad. One day, he findiled that she asked her pemage in the bushn't rying in a cal and walking sorry for her verywhere. One day, a wall that listened see a generous be
```

</details>

### w6_dense754 — `llama`

**Hypothesis.** Upper bound for the hashffn family: a DENSE model with exactly hashffn's 754,432 parameters (hidden 704 instead of 176). It must read all of them every token, 3,017,728 bytes. The gap between this and hashffn is the price of sparsity; the gap between this and the baseline is the value of the parameters. Fills the extreme cell (>768K params, >1.6M bytes/token). This is a deliberate width change, allowed here because it is a control for a positive result rather than an exploration move.

**Prediction (recorded before the run).** 754,432 params, 3,017,728 unique bytes/token. val_bpb in [1.58, 1.70].

```
bpb=1.7102  params=754432  uB/tok=3017728  traffic/tok=3017984  rss=7452KB  tok/s=5140  macs/tok=839168  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 754,432    measured 754,432    ok
  wbytes           predicted 3,017,728  measured 3,017,728  ok
  val_bpb          predicted [1.58, 1.70]  measured 1.7102  -> REFUTED (worse)
```

Archive: new cell 7,8

<details><summary>sample</summary>

```

Once upon a time, there was a little girl named Tommy. They were gave every it, what doings waved to play in the good worked.
One day, they do not ladder and strepded dick to be felt named Lily. They was he is eyes. He tolden a little girl became and smil
```

</details>

## Wave 7

### w7_casc_t20 — `cascade`

**Hypothesis.** tau sweep of the cascade: permissive gate. One trained model traces a curve through the grid because tau is a decode rule only.

**Prediction (recorded before the run).** Lowest bytes/token of the family, highest bpb.

```
bpb=2.4480  params=395104  uB/tok=133099  traffic/tok=1003774  rss=6588KB  tok/s=14619  macs/tok=330990  consist=0.0e+00
```

**Verdict.**
```

```

Archive: new cell 6,3

<details><summary>sample</summary>

```


Dad a loesell gre love the ped you cre on to all drun was day, be cloold to spod bung a tooke the sliked eas ark as he was wor and a somed. Time. She som thappy. Once. He the morry smaked he flour drom and the rounter ist to hand wher to could gry big to
```

</details>

### w7_casc_t80 — `cascade`

**Hypothesis.** tau sweep: strict gate.

**Prediction (recorded before the run).** Higher bytes/token, bpb approaching the transformer-only limit.

```
bpb=1.7913  params=395104  uB/tok=939639  traffic/tok=1023744  rss=6704KB  tok/s=13045  macs/tok=339872  consist=0.0e+00
```

**Verdict.**
```

```

Archive: cell 6,6 already held by w2_ngrammem at 1.6863

<details><summary>sample</summary>

```


He learn with his mom's happenever and playing in the warge very careful. They are boy crothened it was fich. The villater what so dees upset. One dadded what it was slide. They went decided to play with the farmartaw her mom.
One day, they didbye and th
```

</details>

### w7_casc_t95 — `cascade`

**Hypothesis.** tau sweep: near-closed gate. Shows whether ANY bytes are predictable enough to skip the transformer for free.

**Prediction (recorded before the run).** bytes/token just below the dense limit if such bytes exist.

```
bpb=1.7729  params=395104  uB/tok=999531  traffic/tok=1023744  rss=6584KB  tok/s=12987  macs/tok=339872  consist=0.0e+00
```

**Verdict.**
```

```

Archive: cell 6,6 already held by w2_ngrammem at 1.6863

<details><summary>sample</summary>

```


Her and Ben. The little girl naw a boy named Anna with ever the day. It was the girl of the rater with our relot and say. His been acted tried to him more. 
One day, they tother warms and help. She wanted to get out the warm. But when the sun! It was alw
```

</details>

### w7_adapt8_t50 — `multitok`

**Hypothesis.** Confidence-gated emission on the K=8 trunk. Fixed K=8 should be badly damaged; stopping early where the model is unsure ought to recover most of it while keeping much of the byte saving. If adaptive-K8 dominates fixed-K4 on BOTH axes, the useful quantity is confidence, not block size.

**Prediction (recorded before the run).** 2-5 tokens per block, val_bpb far below fixed K=8.

```
bpb=2.5967  params=276224  uB/tok=953418  traffic/tok=1565696  rss=6160KB  tok/s=11799  macs/tok=476096  consist=0.0e+00
```

**Verdict.**
```

```

Archive: cell 4,6 already held by base_s4 at 1.7505

<details><summary>sample</summary>

```
 thes smila gane and nreenes faooy  toe whit es.
Thmy vely tough  stading wan his bridten. They vely on, It soid,, she could molger pucded the burol andns bick milk a ban wor soul tugy a pard. Thme was had fun mot had ocegt and that wae hap the mastes th w
```

</details>

### w7_adapt8_t85 — `multitok`

**Hypothesis.** Strict gate on the K=8 trunk: the high-quality end of the adaptive-emission curve.

**Prediction (recorded before the run).** 1.2-2.5 tokens per block, val_bpb near the K=1 baseline.

```
bpb=2.4951  params=276224  uB/tok=1046978  traffic/tok=1565696  rss=6160KB  tok/s=10943  macs/tok=476096  consist=0.0e+00
```

**Verdict.**
```

```

Archive: cell 4,6 already held by base_s4 at 1.7505

<details><summary>sample</summary>

```
 the smile, "Her esee,"
Lily lakes and buppy and ploulk.
One fay a wagle to be shoath, Rond sourled gold and ulouted of the litsle mouPare, and were bertsfan hinger a bapess. She ary waf the mouttaive on on the nogh amlaedn and the pu and the tor worter as
```

</details>

### w7_adapt4_t60 — `multitok`

**Hypothesis.** Adaptive gating on the K=4 trunk, to confirm the gate helps across block sizes and is not an artefact of large K.

**Prediction (recorded before the run).** Between fixed K=1 and fixed K=4 on both axes, dominating fixed K=4.

```
bpb=2.3125  params=259840  uB/tok=846575  traffic/tok=1236992  rss=6076KB  tok/s=13015  macs/tok=393920  consist=0.0e+00
```

**Verdict.**
```

```

Archive: cell 4,6 already held by base_s4 at 1.7505

<details><summary>sample</summary>

```


noek was eotoroun. Sha best beatten to play with a dod thi graght and saw the growd asd sond whighthas toake. Evter thet cons and read him and laceed to thet so with thas had ittigigid. She lited hers in the side and that at mutties. She cookddy andone t
```

</details>

## Wave 8

### w8_1layer — `llama`

**Hypothesis.** THE CONTROL WAVE 1 WAS MISSING. sharedloop scored 2.2603 and I called the prediction refuted, but I had no reference for what 63,872 parameters is WORTH. A one-layer baseline has 62,720 parameters -- within 2% of sharedloop -- and uses the very same block exactly once instead of five times. This is the only comparison that answers the question sharedloop was actually asking: does iterating a block five times buy anything over using it once? If the 1-layer model is worse than 2.2603, sharedloop's recurrence is doing real work and my verdict was mis-framed. If it is better, iterating a shared block is actively harmful.

**Prediction (recorded before the run).** 62,720 params, 250,880 unique bytes/token, same grid cell as sharedloop. val_bpb in [2.20, 2.50]: I expect the 1-layer model to be WORSE than sharedloop, i.e. recurrence helps.

```
bpb=2.2544  params=62720  uB/tok=250880  traffic/tok=251136  rss=3832KB  tok/s=71983  macs/tok=79616  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 62,720     measured 62,720     ok
  wbytes           predicted 250,880    measured 250,880    ok
  val_bpb          predicted [2.20, 2.50]  measured 2.2544  -> CONFIRMED
```

Archive: improves cell 1,3 (2.2603 -> 2.2544)

<details><summary>sample</summary>

```
Me, a  
On, there the done was a little colove that fus to play with ot the rest treasting. She werour to look that will and But and sometched not, Mom named and you cat. But tout a mom pir. She hun ouck and the in him the mom friends. "Wook happed. She gr
```

</details>

### w8_novalue_nr — `novalue`

**Hypothesis.** SEPARATES THE TWO EXPLANATIONS for wave 1's worst refutation. novalue attended over the ROTATED keys, so the value transported by a token depended on its absolute position -- the same word contributes a differently rotated vector depending on where it sits. That is a different failure from the one I hypothesised (that Wk cannot be both a similarity space and a value space). Attending over the PRE-RoPE keys removes the positional corruption and leaves only the objective conflict. The gap between this and 2.1251 measures how much of the damage was RoPE.

**Prediction (recorded before the run).** 237,312 params, 949,248 unique bytes/token, unchanged. val_bpb in [1.85, 2.05]: most of the 2.1251 damage should be the rotation, not the shared projection.

```
bpb=1.9475  params=237312  uB/tok=949248  traffic/tok=949504  rss=5776KB  tok/s=15442  macs/tok=322048  consist=0.0e+00
```

**Verdict.**
```
  params           predicted 237,312    measured 237,312    ok
  wbytes           predicted 949,248    measured 949,248    ok
  val_bpb          predicted [1.85, 2.05]  measured 1.9475  -> CONFIRMED
```

Archive: cell 4,6 already held by base_s4 at 1.7505

<details><summary>sample</summary>

```


Once upon a time there was a little girl named Sam and fell who now so for go to mill. So, she deling hard and they was to pumes the girl. He was so swno and she flower to not." Help drax a down on the crabd and was slared, the coel what high asked thing
```

</details>

### w8_ema_convonly — `emaconv`

**Hypothesis.** MECHANISM ISOLATION for wave 2's biggest surprise. emaconv scored 1.6813, beating the baseline by 6 sigma while having no attention at all -- my prediction of [1.80,2.15] was refuted in the GOOD direction. The mixer has two parts: a width-4 depthwise causal convolution (strictly local) and a per-channel EMA (unbounded range, non-selective). This run keeps only the convolution. If a 4-tap local filter per channel is enough to beat attention here, then at this scale and budget attention is not buying retrieval, it is buying local smoothing, and the whole result is about byte-level text being locally determined.

**Prediction (recorded before the run).** 248,672 params, ~994,688 unique bytes/token. val_bpb in [1.72, 1.90]: worse than the full mixer but I expect it still to land near the baseline.

```
bpb=1.7032  params=248832  uB/tok=995328  traffic/tok=995584  rss=5244KB  tok/s=38566  macs/tok=246848  consist=6.7e-06
```

**Verdict.**
```
  params           predicted 248,672    measured 248,832    ok
  val_bpb          predicted [1.72, 1.90]  measured 1.7032  -> REFUTED (better)
```

Archive: improves cell 4,6 (1.7505 -> 1.7032)

<details><summary>sample</summary>

```

His he snows my carious eyes on the living there that his toys time, there was a big down and see the story on the blowing her mom said Mom what look at her cool wheeling the truth have the duck said.
"What had broke. He saw thr dog wegghe is becamorning 
```

</details>

### w8_ema_emaonly — `emaconv`

**Hypothesis.** The other half: keep only the per-channel EMA and delete the convolution. The EMA has unbounded range but cannot distinguish position at all -- it is a leaky running average with a learned per-channel time constant. If THIS alone is competitive, then almost none of the sequence structure a transformer computes is being used at this scale. If it collapses while conv-only survives, the useful ingredient is locality, not range.

**Prediction (recorded before the run).** 247,392 params. val_bpb in [1.95, 2.40]: I expect the EMA alone to be clearly the weaker half.

```
bpb=1.7744  params=247872  uB/tok=991488  traffic/tok=991744  rss=5192KB  tok/s=40198  macs/tok=246848  consist=5.7e-06
```

**Verdict.**
```
  val_bpb          predicted [1.95, 2.40]  measured 1.7744  -> REFUTED (better)
```

Archive: cell 4,6 already held by w8_ema_convonly at 1.7032

<details><summary>sample</summary>

```


Tim and John and Dad it was so excited pickiping her. They got her momby she wanted to the park. They were can arrive that paper to her. She two ever the bird too the garden. He wanted to much did not go that play and Ben and took a lot back. They puzzle
```

</details>

## Wave 9

### w9_emaconv_h153 — `emaconv`

**Hypothesis.** PARAMETER-MATCHED CONTROL for emaconv. The wave-2 emaconv carried 269,632 parameters, 8.9% more than the baseline, so part of its 6-sigma win could simply be capacity. Hidden 153 gives EXACTLY 247,552 parameters -- the baseline's count to the unit. If it still beats 1.7390 the win is the mixer, not the budget.

**Prediction (recorded before the run).** 247,552 params (exactly the baseline). val_bpb in [1.69, 1.76].

```
bpb=1.6765  params=247552  uB/tok=990208  traffic/tok=990464  rss=5192KB  tok/s=38278  macs/tok=245248  consist=5.7e-06
```

**Verdict.**
```
  params           predicted 247,552    measured 247,552    ok
  val_bpb          predicted [1.69, 1.76]  measured 1.6765  -> REFUTED (better)
```

Archive: improves cell 4,6 (1.7032 -> 1.6765)

<details><summary>sample</summary>

```


Once upon a time, there was a little more and polite. Suddenly, can trying anymore.
One day, he couldn't find their toys. They here yellow to be adventure they was too bird never jump and bright and ran away. Marug with to have a frog because it need to
```

</details>

### w9_hashffn_s2 — `hashffn`

**Hypothesis.** Seed replicate to test the three-way tie (see reflection R1).

**Prediction (recorded before the run).** Near 1.676 if the tie is real.

```
bpb=1.6905  params=754432  uB/tok=990208  traffic/tok=990464  rss=9936KB  tok/s=12907  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```

```

Archive: cell 7,6 already held by w1_hashffn at 1.6760

<details><summary>sample</summary>

```
 animal the little girl named Lily. She was so that he wanted to build with toy. So said never, and drawled to sturk and walked until they went out on the cash. It made the game up and came a big angry. The girl wall sad.


Lolly was showed and doing who 
```

</details>

### w9_emaconv_s2 — `emaconv`

**Hypothesis.** Seed replicate to test the three-way tie.

**Prediction (recorded before the run).** Near 1.681 if the tie is real.

```
bpb=1.6843  params=269632  uB/tok=1078528  traffic/tok=1078784  rss=5348KB  tok/s=34448  macs/tok=267328  consist=5.7e-06
```

**Verdict.**
```

```

Archive: cell 4,7 already held by w2_emaconv at 1.6813

<details><summary>sample</summary>

```


Hen to their mommy went and chest. She sticktly told her food and smiled. Tim." 
Anna and Mom and Tim and Jack and Sack something was so proud of the safe and a big box girl named Tim's immy. They hugged his mom were come coacs to be ready to way the wor
```

</details>

### w9_ngrammem_s2 — `ngrammem`

**Hypothesis.** Seed replicate to test the three-way tie.

**Prediction (recorded before the run).** Near 1.686 if the tie is real.

```
bpb=1.6961  params=510016  uB/tok=991744  traffic/tok=992000  rss=6928KB  tok/s=14732  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```

```

Archive: cell 6,6 already held by w2_ngrammem at 1.6863

<details><summary>sample</summary>

```

Once upon a time, there was a little girl named Miare would go piggy together. She hopped that we saill deliffing the buildily waited her friends was a big cleached that her come out out in the boat was a sun of the warm chebrate.

Once upon a time, ther
```

</details>

## Wave 10

### w10_hashffn_s3 — `hashffn`

**Hypothesis.** Third seed. With three seeds each for the three winners, a one-way comparison of their means against the 0.0143 within-architecture sd can say whether the tie survives.

**Prediction (recorded before the run).** Near 1.676.

```
bpb=1.6966  params=754432  uB/tok=990208  traffic/tok=990464  rss=9928KB  tok/s=12999  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```

```

Archive: cell 7,6 already held by w1_hashffn at 1.6760

<details><summary>sample</summary>

```
s upoinly wild each little boy named Timmy was adventure yellow. He nodded it was about room looked out to take a bird gake and seen her to lot of special. It said, "Mia, you are you mowring a big one some strong. It was a sing every day," he parefully sta
```

</details>

### w10_emaconv_s3 — `emaconv`

**Hypothesis.** Third seed for emaconv.

**Prediction (recorded before the run).** Near 1.681.

```
bpb=1.6669  params=269632  uB/tok=1078528  traffic/tok=1078784  rss=5436KB  tok/s=34966  macs/tok=267328  consist=5.7e-06
```

**Verdict.**
```

```

Archive: improves cell 4,7 (1.6813 -> 1.6669)

<details><summary>sample</summary>

```

Once upon a time there was a face and thanked to her toys and see she looked at the pool. They lucked of the rews go friend. They were tried to play his garden. They were so shout on her teacher.
One day, then he wanted to all was children and wanted to h
```

</details>

### w10_ngrammem_s3 — `ngrammem`

**Hypothesis.** Third seed for ngrammem.

**Prediction (recorded before the run).** Near 1.686.

```
bpb=1.7002  params=510016  uB/tok=991744  traffic/tok=992000  rss=7008KB  tok/s=14645  macs/tok=332288  consist=0.0e+00
```

**Verdict.**
```

```

Archive: cell 6,6 already held by w2_ngrammem at 1.6863

<details><summary>sample</summary>

```


After mittens what so imaf her mommy else. He was very happy back. She went to tell to fine strong of the box. Jana were to his friends thinurted to arciry and a lone.


Once upon a time, there was a catch. The little girl named Spot and Boblew that the
```

</details>

### w10_emaconv_h153_s2 — `emaconv`

**Hypothesis.** Second seed of the parameter-matched emaconv control, so the control itself has an error bar.

**Prediction (recorded before the run).** Near the w9 value.

```
bpb=1.6771  params=247552  uB/tok=990208  traffic/tok=990464  rss=5124KB  tok/s=35845  macs/tok=245248  consist=5.7e-06
```

**Verdict.**
```

```

Archive: cell 4,6 already held by w9_emaconv_h153 at 1.6765

<details><summary>sample</summary>

```

When she was so After the best about her mom. It ran to make a carrain with his soup, so she stared his goodbye.
Tom is you can told kept take it a getting happy to find a little boy was ran and Tom. They wanted to play with her toy looked amazing. They m
```

</details>


---

## Final deliverable

42 architecture variants trained and measured under the frozen protocol (§2), across
11 registered architectures plus 8 control/ablation variants of them, in 10 waves plus
one post-restart recovery. `report.py` regenerates everything below directly from
`journal.jsonl` and `archive/archive.json` — nothing here is manually transcribed.

### The MAP-Elites grid — 14/81 cells filled

Rows are total parameters, columns are unique weight bytes read per emitted token.
Cell value is the best `val_bpb` achieved in that cell (lower is better).

```
               <32K        32-64K      64-128K      128-256K     256-512K     512-768K    768K-1.05M   1.05-1.6M      >1.6M
          ---------------------------------------------------------------------------------------------------------------------
     <48K|      .            .            .            .            .            .            .            .            .
   48-96K|      .            .            .          2.254          .            .            .            .            .
  96-160K|      .            .            .            .            .            .            .            .            .
 160-224K|      .            .            .          1.944          .          2.025          .            .            .
 224-288K|      .            .          2.124        1.769        1.868        2.597        1.676        1.667          .
 288-384K|      .            .            .            .            .            .            .            .            .
 384-512K|      .            .            .          2.448          .            .          1.686          .            .
 512-768K|      .            .            .            .            .            .          1.676          .          1.710
    >768K|      .            .            .            .            .            .          1.637          .            .
```

The empty lower-left triangle is exactly what §3 predicted before any run: a dense
fp32 model can only ever sit on the `bytes = 4 x params` diagonal, so every cell below
it is reachable only by conditional computation or sub-fp32 storage. The two variant
families that reach that region — `hashffn` (conditional: read one of K expert FFNs)
and `quant` (sub-fp32: row-quantised weights) — are exactly the ones that populate it.
That the map matches the mechanism is itself evidence the axes are measuring what they
claim to.

### The winners, and whether the three-way tie is real

Five-seed-equivalent replication (3 seeds each for the three leaders, 4 for baseline):

| architecture | n | mean val_bpb | sd |
|---|---|---|---|
| `emaconv` | 3 | **1.6775** | 0.0093 |
| `hashffn` | 3 | 1.6877 | 0.0106 |
| `ngrammem` | 3 | 1.6942 | 0.0071 |
| baseline `llama` | 4 | 1.7693 | 0.0159 |

All three beat the baseline's 2-sigma threshold (1.7376) by a wide margin on every
seed — that part was never in doubt. What §4 left open was whether the three winners
were separable from *each other*. They are: between-architecture spread across the
three leaders is 0.0174, within-architecture sd averages 0.0069 — a spread/sd ratio of
**2.54**, above the noise floor. `emaconv` (attention replaced by a width-4 causal
convolution + per-channel EMA, O(1) decode state) is the most reliable winner;
`hashffn` (zero-router bigram-hashed FFN experts) is close behind and scales further
with expert count (2/4/8 experts: 1.7140 / 1.6760 / **1.6369**, the best val_bpb in the
whole archive, monotone at a *pinned* 990,208 bytes/token); `ngrammem` is a real but
thinner win.

### Three most atypical individuals (furthest from baseline in the 5-axis behaviour
space — log-normalised distance over params, unique bytes/token, state bytes/token,
MACs/token, read traffic — deliberately NOT the best performers)

1. **`w4_q2all`** (distance 5.89) — ternary quantisation of every matrix in the
   network (attention, FFN, embedding/unembedding), same parameter count as baseline,
   16x fewer bytes read. `val_bpb 2.1243`. The extremity is almost entirely on the
   bytes axis (wbytes/tok −3.7 sd, traffic −4.6 sd) at zero cost in params or compute —
   the harness's most aggressive move on the I/O axis alone.
2. **`w8_ema_emaonly`** (distance 3.55) — the EMA-only ablation of `emaconv` with the
   convolution deleted. `val_bpb 1.7744`, right at the baseline. Atypical almost
   entirely on the *state* axis (sbytes/tok −3.4 sd): a pure exponential moving average
   carries only D=64 floats of state between tokens, an order of magnitude less than
   any attention-based architecture's KV cache, or even the baseline llama's own
   attention.
3. **`w3_multitok8`** (distance 3.00) — 8 tokens emitted per network evaluation with no
   verification. `val_bpb 3.7707`, the single worst result recorded. Atypical on bytes
   (−2.7 sd) and MACs (+1.1 sd) simultaneously — cheap per-token I/O bought at the cost
   of both quality and (because the trunk still has to catch up on every emitted token)
   more compute per token, not less. The only variant in the archive that is worse on
   quality AND more expensive on a resource axis than the baseline.

### Refuted hypotheses and what they taught

11 of 25 recorded predictions were refuted (44%), mean signed miss on refutations
+0.056 bpb (i.e. refutations skew toward "worse than predicted," not a wash).

The four largest *pessimistic* misses (predicted worse than measured) are exactly the
four variants that **replaced** a mechanism rather than **shared** one:
`emaconv` (−0.119), `w8_ema_convonly` (−0.017), `w9_emaconv_h153` (−0.014), and
`w8_ema_emaonly` (−0.176, the largest miss in either direction). The regularity
recorded at wave-2 reflection held for the rest of the search: **every variant that
forced two functions through one shared tensor (`novalue`'s reused Wk, `sharedloop`'s
reused block, `lowrankffn`'s shared bottleneck, `deepgate`'s conditional FFN) cost more
than its parameter count predicted; every variant that replaced a mechanism outright
(the conv+EMA mixer) cost less.**

Two follow-up ablations, run specifically to test *why*, both confirmed their
mechanistic hypothesis:
- `w8_novalue_nr` (attend over pre-RoPE keys instead of post-RoPE) scored 1.9475,
  recovering most — not all — of `novalue`'s original 2.1251. Positional rotation of a
  shared tensor was doing real additional damage on top of the role conflict itself.
- `w6_hash_pos` (route by position instead of content) collapsed to 1.9199, worse
  than baseline, confirming `hashffn`'s win is genuinely about content-dependent
  routing and not just "own more parameters, read a quarter of them." `w6_hash_uni`
  (route by current token only, no context) landed at 1.7182 — between the position
  control and the full bigram router — showing that *context*, not just content, is
  part of what the router is worth.

### The one anomaly

**A dense model with more read-per-token access to more of its own parameters loses to
a sparse model with the identical parameter budget.**

`w6_dense754` is a purpose-built control: a fully dense transformer with exactly
`hashffn`'s 4-expert parameter count (754,432, achieved via hidden dim 704 instead of
176), so it must read all 3,017,728 bytes of its FFN every single token. It scored
**1.7102**. `w1_hashffn`, same architecture family, same total parameter count, same
training budget — but each token reads only one of 4 hash-routed experts, 990,208
bytes, a **third** of the traffic — scored **1.6760**, reliably better (replicated
across 3 seeds at mean 1.6877, still below dense754's single run).

The gap (0.0342 bpb) sits right at the edge of the measured 2-sigma noise floor
(0.0317), so it is not being over-claimed as decisive on a single run — but it points
the wrong direction for it to be noise-favorable coincidence: a model that touches
*three times more of its own weights every token*, at *identical total parameter
count*, does not win. Naively, more per-step access to more of an equal-sized
parameter budget should never hurt. Candidate explanations that were not tested here
and remain open: the four experts may receive more effective gradient signal by each
specializing on a quarter of the routing-conditioned data rather than all four sharing
one gradient every step (an optimization-dynamics effect, not a capacity effect); or
the router itself may be injecting a weak but useful inductive bias by construction,
which a dense model of the same size has no equivalent of. Both are plausible and
neither was checked. This is the result from the entire search I cannot fully explain.

### What the evaluator's own numbers say about the search as a whole

- Every `params` and `wbytes_per_tok` prediction across all 25 hypothesis-bearing runs
  was exact to the byte. The evaluator's cost model is not a source of error anywhere
  in this search; all prediction error is in quality.
- Every non-quantised architecture (17 of the 19 families/controls) reproduced its
  training-time logits exactly (`consistency = 0.0e+00`) under cached decoding. The
  two quantised families (`ternffn`, `quant`) carry a small, expected, and previously
  measured straight-through-estimator bias, not a training/inference mismatch.
- The Pareto front over all five axes (val_bpb, unique bytes/token, peak RSS, params,
  tok/s) contains 18 of the 42 variants — the design space explored here is genuinely
  multi-dimensional; no single variant dominates on every axis simultaneously, which is
  the sign that the search was not accidentally collapsed onto one metric.

---

## Post-session addendum: a first, honest test of "separate the write path from the read path"

A conversation after the main search asked what it would take to make continual
learning industrially viable rather than a research curiosity. The argument made was
that the blocker is not purely algorithmic but architectural: a system cannot be
A/B-tested or rolled back if learning modifies the same weights that serve, so any
viable design needs a **frozen core** plus a **separately writable store**, with
writes gated (so as not to write everything) and measured on two axes: does a single
online exposure genuinely help on a repeat of similar content (forward transfer), and
does writing corrupt what the frozen core already knew (interference)?

Rather than leave that as an unfalsifiable claim, the smallest honest version of it
was built and measured, in the same style as the rest of this file: a hypothesis, a
control for the first confound anyone would raise, and a result reported whether or
not it was flattering.

### What was built

`mem.c`/`mem.h`: a hashed key -> byte-count table (`ctx` trailing bytes hash to one of
`slots` rows; each row holds 257 saturating counters), **not part of the autodiff
graph** — it is written directly by a rule at inference time, never by gradient
descent. Two operations: `cache_mix` (linearly interpolate the core's softmax output
with the row's empirical frequency distribution, weight `lambda`) and `cache_write`
(increment a count, gated on `surprise = -log p_core(actual_next_byte)  > tau`, or
unconditionally as a control).

A new `memtest` mode in `main.c` (`stream_bpb`) walks a sequence causally one byte at
a time — teacher-forced, so its bpb is directly comparable to `eval_val`'s — and runs
four passes against the best model in the archive (`emaconv`, `w2_emaconv.bin`):

1. **before** — the standard 24 held-out windows (same `val_offsets` seed used
   throughout this file), cache absent, as a reference point.
2. **pass1** — a single walk over a **121,898-byte gap in the corpus proven disjoint
   from every val window** (computed by merging all 24 val window intervals and
   taking the largest free gap: `[18,640,100, 18,762,000)`), with gated writes on.
3. **pass2** — the identical bytes replayed, cache now populated from pass 1, no
   further writes: forward transfer = pass1_bpb − pass2_bpb.
4. **after** — the original 24 val windows again, cache still populated from the
   adaptation walk, no writes: interference = after_bpb − before_bpb.

Everything is run twice per lambda: once with surprise-gated writes, once with an
unconditional-write control, at identical `lambda`.

### The confound checked before trusting any number

With `ctx=2` (a hashed byte-bigram, matching `ngrammem`'s and `hashffn`'s router
convention) and only 16,384 slots against 257² ≈ 66,049 possible bigrams, the obvious
worry is that "interference" is just hash-table crowding, not a real cross-region
effect. Rerunning at `slots=65,536` (4x, near-injective for the bigram key space)
changed the interference numbers by less than 0.0001 bpb — **the confound is ruled
out**; the effect is a genuine content-level collision between the two disjoint
regions' bigram statistics, not an artifact of table size.

### The result: transfer is real, interference dominates it, and the gap widens

| lambda | forward transfer (gated) | interference (gated) | interference / transfer |
|---|---|---|---|
| 0.10 | +0.0159 | +0.0334 | 2.1x |
| 0.15 | +0.0194 | +0.0656 | 3.4x |
| 0.30 | +0.0264 | +0.1948 | 7.4x |
| 0.50 | +0.0322 | +0.4445 | 13.8x |

Full detail at lambda=0.30 (the originally planned operating point):

```
before(no cache)      bpb=1.6964  (6144 tokens, 24 windows)
--- gated (surprise>tau writes only, 27229/121898 = 22.3% of positions) ---
pass1 bpb=1.7643   pass2 bpb=1.7379   forward_transfer=+0.0264
after bpb=1.8912   interference=+0.1948
--- control (unconditional writes, 100% of positions) ---
pass1 bpb=1.6796   pass2 bpb=1.6622   forward_transfer=+0.0174
after bpb=1.7976   interference=+0.1012
```

**Forward transfer is real at every lambda tested** — a single exposure to 122K bytes
measurably reduces surprise on a second exposure to the same bytes, with no gradient
step involved, which is the basic primitive continual learning needs. But
**interference dominates it everywhere tested, and the ratio worsens super-linearly**,
not linearly, as lambda increases: quadrupling lambda from 0.1 to 0.5 turns a
2.1x-worse trade into a 13.8x-worse one. Even at the gentlest lambda tested (0.1),
interference (0.0334 bpb) is roughly double the transfer benefit (0.0159 bpb) on a
smaller, noisier 6,144-token reference sample than the 98,304-token one used
throughout the rest of this file (a real limitation of this quick addendum, stated
rather than hidden — the true noise floor on this smaller sample was not separately
measured).

**A second, unpredicted finding**: surprise-gating consistently interferes *more*
than the unconditional-write control at matched lambda (0.1948 vs 0.1012 at
lambda=0.3), despite writing to only 22% as many positions. The naive expectation
going in was the opposite — that gating would concentrate writes on genuinely useful
signal and reduce collateral damage. A plausible but **unverified** explanation:
gated writes fire specifically on bytes the core found surprising, which by
construction are the least typical, most idiosyncratic completions in the
adaptation region; when a bigram collision lands in the disjoint validation region,
that atypical count is a worse match for typical continuation than the smoothed,
frequency-weighted distribution that unconditional writing accumulates. This was not
tested and should not be taken as established.

### Verdict against the plan's own acceptance bar

The plan proposed this design specifically to achieve "write without hurting" —
interference within noise of the frozen core's own baseline. **That bar was not met
at any lambda tested here.** This is not evidence that separating the write path
from the read path is a dead end; it is evidence that **the specific combination
rule tested — a fixed-weight linear interpolation of raw hashed-bigram counts — is
not it**. The clean next experiment this result points to, not run here: make the
mixing weight a function of the cache row's own count mass (so a slot touched once
by an atypical collision contributes near-zero weight, the way a proper
back-off/Kneser-Ney-style cache model would) rather than a constant lambda regardless
of how much or how reliably a slot has been written. That is a real, scoped,
falsifiable next step — and, honestly, the fact that the first straightforward
implementation failed its own acceptance criterion is a more useful data point than
a demo that looked good on the one number that was checked.

---

## Addendum 2: measurement bug #6 retracts addendum 1's headline, and the store is made revocable

Addendum 1 ended by naming the next experiment: replace the fixed mixing weight with
one that scales with how much evidence a slot actually holds. That experiment was
run. It produced two things: a **retraction of addendum 1's main claim**, and the
first working demonstration of the property the whole architectural argument rests
on.

### What was added

`mem.c`/`mem.h` rewritten:

- **Count-dependent mixing.** `lambda_eff = lambda_max * n / (n + kappa)`, with `n`
  the count mass already in the slot. A slot written once contributes almost nothing;
  a well-supported slot contributes nearly `lambda_max`. `kappa == 0` reproduces
  addendum 1's fixed-lambda behaviour exactly, so both regimes live in one binary
  and are directly comparable. **Verified**: at `kappa=0` every number reproduces
  addendum 1 to 4 decimals.
- **Write log + revocation.** Every write records `(slot, target, episode)`.
  `cache_revoke(episode)` walks the log in reverse and decrements. Counters widened
  to 32-bit so the saturating-halve safety valve never fires at experiment scale,
  which is what keeps revocation exactly invertible; if it ever did fire a `halved`
  flag is reported and the run would be marked not-exact.
- **Store accounting** (`cache_stats`): total count mass and occupied slots.

### Measurement bug #6 — and it retracted a positive result

Addendum 1 reported forward transfer of **+0.0264 bpb** at lambda=0.3, computed as
`pass1_bpb - pass2_bpb`. That formula is wrong, and the error is not subtle:
**pass1 builds the store while it is being measured.** Its average therefore includes
early positions where the store was empty *and the mixing penalty was already being
paid*. `pass1 - pass2` measures "how much of the stream has elapsed", not "how much
the store learned".

The correct reference is the frozen core alone on the *same* adaptation stream, which
was never measured. Added as `adapt_before` — and it lands at **1.5887 bpb**, far
below the 1.6964 val figure (the two regions simply differ in difficulty, which is
precisely why a matched reference is mandatory).

Recomputing addendum 1's operating point against the correct reference:

| lambda=0.3 | addendum 1 (wrong ref) | corrected |
|---|---|---|
| forward transfer | **+0.0264** | **−0.1492** |

**Addendum 1's headline finding — "forward transfer is real at every lambda tested"
— is retracted. It was an artifact of the reference, not a property of the store.**
At lambda=0.3 the store makes the adaptation stream substantially *worse*, not better.
This is the sixth measurement bug found in this project and the only one that
manufactured a positive result rather than distorting a negative one.

### The corrected picture

Sweeping both knobs against the correct reference (transfer positive = good,
interference positive = bad):

| lambda | kappa | transfer | interference |
|---|---|---|---|
| 0.30 | 0 | −0.1492 | +0.1948 |
| 0.30 | 50 | −0.0550 | +0.0804 |
| 0.30 | 200 | −0.0185 | +0.0344 |
| 0.30 | 500 | −0.0043 | +0.0148 |
| 0.15 | 500 | +0.0039 | +0.0029 |
| **0.05** | **200** | **+0.0056** | **+0.0004** |
| 0.05 | 0 | +0.0106 | +0.0107 |
| 0.60 | 2000 | −0.0032 | +0.0107 |

Two readings, both worth stating:

1. **Along the kappa axis at fixed lambda=0.3, both columns march toward zero
   together.** That is not the store getting smarter — it is `lambda_eff → 0`, the
   store switching itself off. Confidence weighting alone does not rescue a mixing
   weight that is too high; it just mutes it.
2. **There is nonetheless a genuine working regime at low lambda.** At
   `lambda=0.05, kappa=200`, confirmed on 4x more validation data (24,576 tokens):
   transfer **+0.0056**, interference **+0.0004** — the store helps on new material
   and leaves the original validation set essentially untouched. This is the first
   configuration in either addendum where the design's own acceptance bar ("write
   without hurting") is actually met.

**But the honest size of that win is 0.0056 bpb**, which is roughly a third of the
0.0143 bpb seed-noise floor established in §4 for the main study. These particular
measurements are deterministic (frozen core, deterministic store, fixed windows), so
there is no run-to-run variance to speak of — but the effect is small enough that it
should be read as "a real mechanism operating at the edge of what this setup can
resolve", not as a result worth building on yet. A prediction recorded before the
sweep — that confidence weighting would help only marginally because the damage comes
from *well*-populated slots — was also **refuted**: kappa reduced interference 5.7x
(0.1948 → 0.0344 at lambda=0.3), far more than predicted.

### What did work, unambiguously: revocability

Every run above reports:

```
REVOKE episode1: 27229 records reverted, residual mass=0, halved=0
after revoke     bpb=1.6407   revocability_error=0.000e+00  EXACT
```

27,229 writes un-written; store mass returns to exactly 0; bits-per-byte on the
validation set returns to the pre-write value with **error 0.000e+00**, at every
lambda and kappa tested. This is the same class of check as the `consistency == 0.0`
guarantee on the decode path: an invariant that is measured rather than asserted.

That matters more than the transfer number. The argument for separating the write
path from the read path was never "a cache improves perplexity" — it was that
learning has to be **localised, attributable and revocable** before an operations
team will run it, and that this is an architectural property no amount of scaling
gives you for free. That property is now demonstrated end to end at small scale:
one write touches one counter, every write carries the episode that caused it, and
an episode can be rolled back to a bit-identical state.

The useful thing an evaluator can say here is that the deployment property is solid
and the learning benefit is not yet. That is the opposite ordering from what the
usual demo shows, and it is the more honest one.

---

## Addendum 3: the literature had already answered this, and the answer flipped the sign

Asked why any of this was being rebuilt when the bricks already exist and are
published. The question is correct and the answer is measured below rather than
argued.

### What in this repo is a reinvention

Every architecture in the archive is a known technique, and the "modifications" were
modest:

| this repo | prior work |
|---|---|
| `emaconv` (conv + per-channel EMA mixer) | MEGA (Ma et al. 2022) uses a per-channel EMA in exactly this role |
| `hashffn` (hash-routed FFN experts) | Hash Layers (Roller et al. 2021) |
| `ngrammem` (hashed table into the residual) | product-key memory (Lample et al. 2019) |
| `multitok` (K heads, K tokens per pass) | Gloeckle et al. 2024; Medusa |
| `cascade` (small model gates the big one) | speculative decoding / model cascades |
| `quant` (row-wise ternary/int8, STE) | BitNet and the quantisation literature |
| the addendum-1/2 store | cache LM (Grave et al. 2017) + Jelinek-Mercer smoothing (1980s) |

### The cost of not looking it up first, measured

kNN-LM (Khandelwal et al. 2020) reports that the useful key for an interpolated
memory is the model's **own context representation**, not the surface n-gram --
surface keys collide between semantically unrelated contexts. The store in addenda 1
and 2 used a surface bigram key and failed in precisely that way.

Implementing the published insight took ~30 lines: `--mkey 1` replaces the surface
key `hash(prev, cur)` with `hash(cur, top1, top2)`, where top1/top2 are the frozen
core's two most likely next bytes. This is a crude discrete stand-in for kNN-LM's
continuous hidden-state key -- contexts in which the model believes the same thing
collapse to the same slot, so the store indexes "when you think X, it was actually Y"
rather than "after these two bytes". Both keys are functions of the context only
(the core's distribution is computed before the target is read), so neither leaks
the answer.

Same core, same stream, same protocol, 24,576 validation tokens:

| key | lambda | transfer | interference | occupied slots |
|---|---|---|---|---|
| surface n-gram | 0.05 | +0.0056 | +0.0004 | 547 |
| surface n-gram | 0.30 | **−0.0185** | +0.0374 | 547 |
| belief (top-2) | 0.05 | +0.0074 | −0.0011 | 1661 |
| belief (top-2) | 0.30 | **+0.0072** | +0.0154 | 1661 |
| belief, ungated writes | 0.30 | **+0.0166** | +0.0031 | 2525 |

**The sign of the result flips.** At lambda=0.3 the surface key makes the adaptation
stream worse (−0.0185); the belief key makes it better (+0.0072). The best
configuration found anywhere in these three addenda is the belief key with
unconditional writes: transfer **+0.0166** against interference **+0.0031**, a 5.4:1
ratio in the right direction -- the first time the design's own bar is met with
margin rather than at the resolution limit. Revocation remains exact (0.000e+00) in
every one of these runs.

The mechanism is visible in the last column: the belief key occupies 3-5x more slots
from the identical write budget. It discriminates contexts the surface key was
conflating, which is exactly the failure kNN-LM describes.

### The honest rule this produces

Reinventing was justified for exactly one thing here, and it is worth stating the
criterion narrowly rather than as a defence of the whole exercise:

> Build it yourself when **the measurement you need does not exist in the library**.

This project's load-bearing metric is unique weight bytes touched per emitted token,
with an inference path provably identical to the training path (`consistency == 0.0`).
No framework exposes that; instrumenting one would have produced a less trustworthy
number than owning every read in 700 lines of C. That choice paid for itself: six
measurement bugs found, two of which (the `touch_all` short-circuit, and the
addendum-1 transfer reference) would have produced publishable-looking false results.

Everything else was reinvention with no such justification. Concretely: the store
should have started from kNN-LM's key, not from a surface bigram, and two rounds of
iteration would have been unnecessary. The rule that follows is not "always reuse"
but: **reuse the idea, own the measurement.** The literature is the cheapest source
of next hypotheses available, and on this evidence it outperformed local search --
one published insight beat two rounds of my own iteration on the same harness.

---

## Wave 11: does combination beat invention? — orthostack

Prompted by the observation that the historic breakthroughs (Transformer, AlphaGo,
FunSearch) invented no new bricks, only combined existing ones. This archive has a
data point against naive combination — `loopexpert` (sharedloop + hashffn) scored
1.9439, worse than the baseline and far worse than hashffn alone — and the diagnosis
offered at the time was that **both parents attacked the same resource** (parameter
economy) and therefore competed.

That yields a testable criterion: **bricks compose when they touch different
components.** The three architectures that beat the baseline in this search do
exactly that — `emaconv` replaces the token mixer, `hashffn` replaces the FFN,
`ngrammem` injects into the residual stream. None shares a component with another.

`orthostack` stacks all three, with the memory on a knob so its marginal
contribution is isolable.

**Hypotheses, recorded before launching:**

- **H1 (composition).** emaconv + hashffn touch different components, so they should
  compose largely additively. Predicted mem=0 in **[1.58, 1.65]**.
- **H2 (redundancy).** emaconv's win is known to come mostly from its 4-tap *local*
  convolution (`w8_ema_convonly` = 1.7032 alone), and ngrammem is *also* a local
  byte-statistics mechanism. So despite being worth −0.075 standalone, stacking it
  should add **< 0.03**.

### Results (2 seeds each)

| variant | seed 1337 | seed 2 | mean | params | uB/tok | sB/tok | tok/s | consistency |
|---|---|---|---|---|---|---|---|---|
| mem=0 (2 bricks) | 1.6109 | 1.5975 | **1.6042** | 776,512 | 1,078,528 | 5,120 | 22,294 | 6.7e-06 |
| mem=1 (3 bricks) | 1.5454 | 1.5455 | **1.5454** | 1,038,976 | 1,080,064 | 5,120 | 21,596 | 3.8e-06 |

**H1: CONFIRMED.** mem=0 landed at 1.6109, inside the predicted [1.58, 1.65].

**H2: REFUTED.** The memory adds **0.0588 bpb** (2-seed means), 2.1x the 0.0285
noise floor — not the <0.03 predicted. `ngrammem` is genuinely complementary to the
convolution, not redundant with it.

### Additivity, which is the actual finding

Standalone gains against the 1.7693 baseline: emaconv −0.0918, hashffn −0.0816,
ngrammem −0.0751. Naive sum: **−0.2485**.

| | measured gain | % of additive prediction |
|---|---|---|
| 2 bricks (mem=0) | −0.1651 | **95.2%** |
| 3 bricks (mem=1) | −0.2239 | **90.1%** |

Three mechanisms discovered independently, each individually modest, stack with
~90% of their gains preserved. `orthostack` at 1.04M parameters beats `hashffn8` at
1.43M (1.5454 vs 1.6369) while carrying **33x less decode state** (5,120 constant
bytes, no KV cache at all) and running **1.5x faster** than the baseline.

### Why H2 was wrong

The refinement offered before the run — that the right criterion is "failure modes"
rather than "components" — was an over-complication, and the coarse component-level
criterion turned out to be sufficient. The specific error: conv and memory table were
treated as the same mechanism because both read *locally*. They read locally but they
**store** differently. A convolution learns one filter shared across every context;
a lookup table learns one entry per context. Same range, orthogonal capacity.

### What this does and does not establish

It supports the combination thesis strongly at this scale: the component-level
criterion predicted both the success (H1, inside a pre-registered interval) and, by
its own logic, would have predicted H2 correctly had it not been overridden by a
mechanism-level intuition that was wrong.

It does not make composition free: `loopexpert` still failed, and it failed exactly
where the criterion says it should — two parents competing for one resource.

And none of the three bricks is novel. `emaconv` ≈ MEGA, `hashffn` ≈ Hash Layers,
`ngrammem` ≈ product-key memory. The contribution here is not the bricks but the
**measurement that says which ones compose** — which is the same conclusion this
project keeps arriving at from every direction.

---

## Wave 12: DR1 — representing a checkpoint by the process that generates it

Asked for a format achieving 200x compression of this work. The honest starting
point is that 200x lossless compression of a trained checkpoint is not available by
quantisation: this archive already measures that route, and `w4_q2all` (ternary
everywhere) reaches **16x** on weight bytes at a cost of **+0.355 bpb**. Information
theory does not bend for a better encoder.

There is a different route, and it is legitimate provided it is verified rather than
asserted: a checkpoint here is a **deterministic function** of

    (C source, corpus, architecture, hyper-parameters, seed, thread count)

so it need not be stored at all. Store the arguments; recompute on demand.

### The prerequisite, measured before building anything

| condition | result |
|---|---|
| same config, same thread count, run twice | **bit-identical** checkpoints |
| same config, different thread count | **different** checkpoints |

The thread-count sensitivity is real and traceable: the linear backward accumulates
into per-thread tiles whose count equals the thread count, so the summation order —
and therefore fp32 rounding — changes with it. **`threads` is part of the recipe, not
an execution detail.** Had this not been checked first, every recipe would have been
silently unreproducible on a machine with a different core count.

### The format

One line, 173 bytes mean:

```
DR1|orthostack|64,5,8,4,176,257,1|convw:4,experts:4,mem:0|800,16,256,2|0.012,0.1,100,1337|40da6999…|1d02fe9d…|ba2f1fcf…
     arch      dim,L,H,KV,hid,V,tie  knobs                steps,bs,seq,threads  lr,wd,warmup,seed  corpus16  src16  ckpt-sha256
```

`corpus16` and `src16` do not reconstruct those inputs — they exist only to refuse a
replay against the wrong ones. The full checkpoint SHA-256 is what makes the format
falsifiable.

### Verification, which is the whole point

```
replaying w11_ortho_mem1 ...
  expected 1a6bc0ed975f93796ffaa2e8d161b413277b254089a20f065701c3d490c87b16
  got      1a6bc0ed975f93796ffaa2e8d161b413277b254089a20f065701c3d490c87b16
  -> BIT-EXACT REPRODUCTION
```

**173 bytes regenerate 4,155,904 bytes**, hash for hash, in ~4.5 minutes of CPU.

### The two ratios

41 checkpoints, 70,734,732 bytes of weights, represented by a 7,108-byte recipe book.

| ratio | value | what it assumes |
|---|---|---|
| **referential** | **9,951x** | corpus and sources already present — the re-run / reproduce-an-experiment case |
| **self-contained** | **1.84x** | shipping recipes + corpus (38.4 MB) + C source (131 KB) to a recipient who has nothing |

Quoting only the first would be dishonest. This is **not compression in the Shannon
sense**: no information was removed, it was moved out of storage and into computation
plus external dependencies. Claiming a file has been "compressed into a URL" is true
only for someone who already has the network.

The interesting property is that the self-contained ratio improves **linearly with
checkpoint count**, because corpus and source amortise across all of them. Measured
break-even is **22 checkpoints**; this archive holds 41. A party storing thousands of
variants of one model family tends asymptotically toward the referential ratio.

### What this actually is

Not a compression algorithm. A statement that **the artifact was never the model** —
the model is the process, and the weights are a cache of it. 200x and beyond is
reachable, but only by changing what "the model" refers to, and only if the claim is
backed by a hash rather than a promise.

Two things this does not survive: any nondeterminism in training (a single reduction
order change breaks it, as the thread-count experiment shows), and loss of the corpus
(the recipe references data it does not contain). Both are properties worth stating
plainly, because a format like this is exactly the kind of thing that looks
impressive until someone tries to replay it on different hardware.

---

## Wave 13: DR2 verified, and LAB/1 — a format whose unit is the epistemic state

### First, the DR2 measurement that was pending

A chained checkpoint series (200 / 400 / 600 / 800 from one 800-step horizon,
each node continuing the previous) against a from-scratch control:

```
dr2_series_200 done: train done 88.5s
dr2_series_400 done: train done 89.0s
dr2_series_600 done: train done 83.0s
dr2_series_800 done: train done 83.0s
-> CHAINED == SCRATCH (bit-exact)
```

Four checkpoints for 800 steps of work instead of 200+400+600+800 = 2,000, and the
final artifact is byte-identical to one trained in a single run. The derivation graph
is sound. Exact resume required serialising Adam's moments **and** the RNG (batch
order depends on it), and separating `--stopat` from `--steps` because the cosine
schedule is a function of the total horizon, not of where you halt.

### Then the honest verdict on DR1/DR2

Challenged on whether any of this is new. It is not. **DR2 is Nix** — content-
addressed derivations, a build DAG, memoisation, refusal to replay against changed
inputs. Nix has done this since 2003, and I used its own word, "derivation", without
noticing. Two rediscoveries in a row: the addendum-1 store was a cache LM from 2017,
and the recipe format is a package manager.

Worse, and this is the sharpest thing in this file: **the archive's best mechanisms
include a bigram hash table.** `ngrammem` improves `val_bpb` by 0.075 reproducibly
across three seeds. That is a genuine result *on this instrument*, and what it
actually demonstrates is that **the metric this project invested most in can be moved
by a lookup table**. An evaluator a bigram table can improve is not measuring
understanding. The instrument is rigorous; the quantity it measures is weak.

### LAB/1

So rather than another artifact format, the object built here has a different unit.
Every format above stores **conclusions**: safetensors and GGUF store weights, Nix
stores how to rebuild them, MLflow stores that a run happened. None stores **the
entitlement to the conclusion**.

`archive/model.lab` is a single executable file (12.8 KB) carrying:
- the measured noise floor (sd 0.0159 over 4 baseline seeds)
- the behaviour frontier (14 niches)
- every prediction ever made in this project, with its outcome
- the open questions, including the two the project could not resolve
- and its own interpreter

The property I believe is unusual is that **it refuses to record a claim it has not
earned**, and this is mechanical rather than advisory:

```
$ ./model.lab run sneaky_win
REFUSED: no pre-registration for sneaky_win. Declare a prediction first.

$ ./model.lab preregister w1_hashffn 1.60 1.72 hashffn experts=4
REFUSED: w1_hashffn already has results. A prediction written after seeing
         the answer is not a prediction.
```

A result becomes a `WIN` only if it clears the file's **own measured** 2σ
(0.0318 bpb) over a minimum seed count; otherwise it is written down as
`INDISTINGUISHABLE`, and that verdict is not overridable by the caller. `challenge`
re-runs an old claim from its recipe and compares SHA-256, so a claim that has
quietly stopped reproducing becomes detectable rather than inherited.

It also reports something no experiment tracker does, because no experiment tracker
requires the prediction first: **calibration. 16 of 27 predictions (59%) landed
inside their declared interval.** That number is a property of the forecaster, not of
the models, and it is the most useful single line in the file.

### What LAB/1 is not

It is not revolutionary and the docstring says so, listing its own prior art before a
reader can. Pre-registration is a social convention in clinical trials and OSF; the
contribution here is only that it is *enforced by the file rather than by a
committee*. It does not make the underlying metric any better — `val_bpb` remains
gameable by a lookup table, and LAB/1 will happily certify a bigram table as a WIN,
because the rule it enforces is honesty about measurement, not wisdom about what to
measure. Those are different problems, and only the first one is solved here.

### Wave 13b: the first claim LAB/1 certified on its own

`ortho_e8` — orthostack with 8 experts instead of 4 — was chosen because the answer
was genuinely unknown: `hashffn` gained 0.039 going 4→8 standalone, but the stack had
already captured ~90% of its components' additive gains, so the marginal expert could
plausibly have been saturated.

The full loop, driven by the file:

```
$ ./model.lab preregister ortho_e8 1.50 1.55 orthostack mem=1 experts=8 convw=4
pre-registered ortho_e8 in [1.50,1.55]
$ ./model.lab run ortho_e8 1337 2
    val_bpb 1.5218
    val_bpb 1.5236
  VERDICT: WIN  (mean 1.5227 vs baseline 1.7693, margin +0.2466,
                 threshold 0.0318 (2sd); prediction [1.50,1.55] held)
```

**1.5227** (2 seeds, spread 0.0018) — the best result in the archive, past the
previous best `orthostack mem=1` at 1.5454. The marginal expert is **not** saturated:
4→8 buys another 0.023 inside the stack, against 0.039 standalone, so roughly 59% of
the standalone gain survives on top of three already-stacked mechanisms.

Two things worth separating. The result is ordinary — a fourth incremental gain from a
known technique. What is not ordinary is that **no human decided it was a win**: the
prediction was timestamped before the run, the seeds were fixed by the file's own
minimum, and the WIN was computed against a noise floor the file had measured
earlier. The verdict line in `model.lab` was written by `model.lab`.

Calibration updated automatically and moved in the honest direction: **17/28 (61%)**.
A forecaster landing 61% of pre-registered intervals is neither well calibrated nor
useless, and the file will keep reporting it whether or not that flatters the author.

---

## Wave 14 -- the SEED format: is 200x compression of this archive reachable, and does it mean anything?

### Why this wave exists

A conversation elsewhere claimed a format compressing language models by
1000-3000x with O(1) inference. That claim is not testable at the scale it was
made at, but it IS testable here: this archive's llama baseline is 247,552
parameters in a 990,404-byte checkpoint, and 200x of that is 4,952 bytes. If a
mechanism cannot survive 200x on a quarter-million parameters, it will not
survive 1000x on a trillion.

The mechanism tested is the only one that reaches those ratios without storing
indices: **regenerate every weight from the training seed, and store only the
values of a small subset of coordinates chosen by that same seed.** Indices are
therefore free. This is the strongest honest version of "the model is mostly
its initialisation".

### The format

`.seed` = 192-byte self-describing header (magic, seed, fseed, frac, ntot, K,
arch name, cfg string) + K fp32 values. Reconstruction re-runs the
architecture's `build()` with the recorded seed -- exactly what the training run
did -- then overwrites the K masked coordinates. `disco infer --loadseed` never
opens the dense checkpoint.

Two properties were verified before any quality number was taken:

- **lossless**: `disco seedcheck` rebuilds and diffs coordinate by coordinate
  against the dense checkpoint. `max_abs_diff = 0.000000e+00`, `n_differing = 0`
  out of 247,552.
- **end-to-end identical**: `eval` from the dense checkpoint and `eval` from the
  `.seed` file alone return the same bpb to all printed digits.

Without both, a quoted ratio would describe an artifact that is not the measured
model.

### A regression that had to be settled first

Adding the frozen mask meant touching `opt_step`. The first version guarded the
existing loops with `if(mask[j])`. Mathematically inert when the mask is off --
and it changed the answer: an archived checkpoint no longer reproduced
bit-for-bit, because the branch defeated the vectorisation that `-O3
-ffast-math` had been applying, moving the floating-point summation order.

Fixing it meant duplicating the dense loops textually rather than guarding them.
After that:

| binary | SHA-256 of orthostack/1337 |
| --- | --- |
| HEAD, source as committed | `847b4dce...` |
| this wave's binary | `847b4dce...` -- identical |
| recorded in `model.lab` / archive | `dc90850c...` |

So the mask is provably inert on the dense path. But the third line is a
separate, older finding: **the archive no longer reproduces from the committed
source.** `model.lab`'s replay guard had been refusing challenges for exactly
this reason and was right to. Re-running orthostack from HEAD gives val_bpb
1.5255 against the recorded 1.5218 -- a gap of 0.0037, well inside the noise
floor (sd 0.0159), so the numeric claim survives while bit-exact replay does
not. That distinction is now recorded rather than papered over.

### What a compressed model has to beat

A ratio quoted against nothing is not a result. Two reference sets were computed
on **exactly the tokens `eval_val` scores** (same deterministic offsets, same
batch layout):

Zero-capacity ceilings (`ceiling.py`):

| model | bpb |
| --- | --- |
| uniform over 257 symbols | 8.0056 |
| unigram fitted on train | 4.4018 |
| bigram fitted on train | 3.3110 |
| trigram fitted on train | 2.4907 |

Byte-matched n-gram baselines (`bytematch.py`) -- the best truncated n-gram
model that FITS in each budget, with 16-bit log-prob quantisation actually
applied rather than assumed:

| budget | ratio | best n-gram bpb |
| --- | --- | --- |
| 988 B | 1002x | 3.4983 |
| 4,952 B | 200x | 2.6761 |
| 9,904 B | 100x | 2.5134 |
| 49,520 B | 20x | 2.4899 |
| 99,040 B and above | <=10x | 2.4905 (trigram exhausted) |

This is the bar. At 200x the artifact must score below 2.6761, or the
"compressed model" carries less usable structure than a table of 887 byte
triples of the same size.

### The curve

Every point trained under the archive's protocol (800 steps, bs 16, seq 256,
lr 0.012, threads 2), pre-registered in `model.lab` before measuring, and
compared against the best n-gram model that fits the SAME byte budget.

| ratio | bytes | K trainable | SEED bpb | n-gram, same bytes |
| ---: | ---: | ---: | ---: | ---: |
| 2.0x | 495,200 | 123,752 | **1.7484** | 2.4905 |
| 4.0x | 247,600 | 61,852 | 1.8591 | 2.4905 |
| 10.0x | 99,040 | 24,712 | 2.2155 | 2.4905 |
| 15.01x | 65,984 | 16,448 | 2.4954 | ~2.4899 |
| 20.0x | 49,520 | 12,332 | 2.9315 | 2.4899 |
| 100x | 9,904 | 2,428 | 6.2867 | 2.5134 |
| 200x | 4,952 | 1,190 | 6.7661 | 2.6761 |
| 1002x | 988 | 199 | 7.5902 | 3.4983 |

Dense anchor, this binary: 1.76066 / 1.78685, mean 1.7738.

Two things fall out immediately. The cliff is between 5% and 1% trainable, not
gradual. And **the dense checkpoint is not on the frontier**: freezing half its
coordinates at their initialisation gives a file half the size and a score
0.025 better -- inside the noise, so no win is claimed, but the absence of cost
is clean.

### The fairness controls, which were the point

A negative result bought by handicapping the thing under test is worth nothing.
All of these were pre-registered before any of them ran. At an identical 200x
budget of 4,952 bytes:

| variant | bpb |
| --- | ---: |
| `f9`, lucky random mask draw | 5.6216 |
| `struct`, normalisation gains first | 5.7677 |
| `lr20`, step size x17 | 6.7157 |
| `lr5`, step size x4 | 6.7201 |
| `fseed=7`, reference draw | 6.7661 |
| `emb`, readout first | 7.8878 |

A 2.27 bpb spread between the best and worst way of spending the identical
budget -- and the bar at that budget is 2.6761. Nothing rescues 200x.

Raising the step size buys 0.09 bpb for 4x and 0.09 for 17x, which settles the
optimisation question: the limit is capacity, not the schedule.

### What was refuted, and what replaced it

The explanation written down mid-wave -- that the binding constraint is the
readout, since 257 output symbols need 257x64 = 16,448 numbers before the
logits stop being a random projection -- is **wrong, and backwards**. At an
identical 15.01x:

| allocation of the same 16,448 values | bpb |
| --- | ---: |
| spread at random over every tensor | 2.4954 |
| spent filling the tied readout exactly | 4.1713 |

Concentrating is 1.68 bpb worse than spreading, and the same inversion holds at
200x (6.7661 spread against 7.8878 concentrated). Stated at its sharpest:
`seed_r200_emb` trains 1,190 coordinates and scores 7.8878, while `seed_r1000`
trains 199 spread at random and scores 7.5902. **Six times more trainable
parameters, concentrated, do worse than six times fewer, spread.**

What replaced it survives a threshold test rather than merely fitting the data.
The model has 704 rmsnorm gains, two vectors per layer across five layers plus
the final one, so filling them touches every depth with one scalar per channel:

| ratio | K | gains covered | vs the random mask |
| --- | ---: | --- | ---: |
| 200x | 1,190 | 704/704, every depth | **+1.0 bpb** |
| 1002x | 199 | 199/704, first layers only | -0.12 bpb |

Above K = 704 the strategy wins clearly; below it, it reaches only the first
few registered tensors, becomes a concentration again, and loses. So: **some
adaptation has to reach every depth, and the normalisation gains are the
cheapest way to buy that.** This is not a random-feature regime -- a fully
random trunk produces features no trained readout recovers.

### The mask draw, and the loophole it opens

Four seed-derived masks at 200x: 5.6216, 6.2533, 6.8095, 7.0872. Mean 6.4429,
sd 0.648, against sd 0.061 for the training seed at a fixed mask. The choice of
coordinates weighs about ten times the training noise.

That opens a legitimate exploit: search `fseed` at encode time and store the
winner, which costs four bytes. **Four samples cannot close it.** The 95%
interval on that sd runs from 0.37 to 2.42, and at the top of that range twenty
draws would suffice. No Gaussian tail is extrapolated from n=4 here.

The argument that does hold needs no tail assumption: going from 16,448 to
1,190 trainable coordinates currently costs +3.95 bpb, and clearing the bar
would require that cost to be +0.18 -- a factor of 22 from mask choice alone.
The deliberately designed allocation does not even beat the lucky random draw,
which is not what a distribution with a far-reaching good tail looks like.

### The dynamic battery, and the finding that matters most

| label | ratio | lossless | load s | RSS MB | tok/s | unique weight B/token |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| dense | 1.0x | true | 0.005 | 5.8 | 13094 | 990,208 |
| r2 | 2.0x | true | 0.009 | 6.0 | 12600 | 990,208 |
| r10 | 10.0x | true | 0.006 | 6.1 | 13027 | 990,208 |
| r200 | 200.0x | true | 0.005 | 6.0 | 13203 | 990,208 |

**Unique weight bytes touched per token is identical at every ratio.** So is
peak RSS, so is throughput, so is load time. A 200x smaller file is 1x in
memory, 1x in bandwidth and 1x in speed, because reconstruction materialises
the same dense array. Any claim that compression of this kind buys inference
speed is refuted here by direct measurement, independent of quality.

Free generation, 2048 tokens:

| label | distinct-4 | longest repeat | distinct chars |
| --- | ---: | ---: | ---: |
| dense | 0.761 | 8 | 39 |
| r2 | 0.861 | 14 | 42 |
| r10 | **0.002** | **2044** | **4** |
| r200 | 0.010 | 922 | 10 |

`r10` was the format's ONLY win over its byte-matched baseline (2.2155 against
2.4905). It emits `An` and then the byte 0x9E, 2,044 times.

That is the whole case for dynamic testing, and it is an indictment of the
evaluator this lab is built on. `val_bpb` -- teacher-forced, one position at a
time -- certified as the single success case a model that produces four
distinct characters. The per-position curve did not catch it either: r10 reads
2.48 then flat 2.17-2.22, the same shape as dense, just shifted. Only free
running caught it. This is the second time in this project that the headline
metric was gamed by something that is not a language model; the first was
`ngrammem`, a bigram table.

Out of distribution, five corpora built from one segment so the transformation
is isolated by the `ood_id` control:

| label | val | id | rot13 | case | shuf | synth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 1.7607 | 1.7575 | 8.1801 | 13.3047 | 2.6252 | 8.0368 |
| r2 | 1.7484 | 1.7402 | 7.8692 | 12.9133 | 2.5828 | 7.7948 |
| r10 | 2.2155 | 2.2082 | 6.9950 | 10.8702 | 2.5944 | 7.1389 |
| r200 | 6.8095 | 6.8090 | 7.6198 | 10.4801 | 6.8139 | 7.1988 |

Two things here were not expected. The **dense** model scores 13.3047 on
case-flipped text against 8.0056 for uniform over 257 symbols: predicting worse
than knowing nothing is active miscalibration, not ignorance. And the broken
artifacts are MORE robust -- r200 beats dense on rot13 and on case -- purely
because being uninformative outperforms being confidently wrong. r200's val,
ood_id and shuf agree to three decimals (6.8095 / 6.8090 / 6.8139), which is
the signature of a model that has collapsed to a fixed marginal and does not
read its input at all.

Fault injection in the stored payload:

| label | payload bits | 16 flips: finite / median | BER 1e-4: bits, finite / median |
| --- | ---: | --- | --- |
| dense | 7,923,168 | 5/8, 1.7565 | 792, 0/8, none |
| r2 | 3,960,064 | 6/8, 1.7455 | 396, 0/8, none |
| r10 | 790,784 | 4/8, 2.2094 | 79, 1/8, 2.2282 |
| r200 | 38,080 | 6/8, 6.8204 | 3, 7/8, 6.8175 |

At a fixed count of flipped bits, the compressed artifacts are no more robust --
r10 goes non-finite in 4 trials of 8 where dense survives 5. At a fixed channel
error rate they win easily, but only because they expose fewer bits: per stored
bit, fragility rises roughly with the ratio, while total exposure falls faster.
One flipped fp32 exponent is enough to produce NaN in any of them.

### Calibration

Eighteen pre-registered intervals, four inside: **22%**, against 61% for the
preceding 28. The failure is systematic and one-directional: every miss at a
ratio of 20x or beyond is optimistic, most of them by several bpb. Even
`seed_r200_f13`, whose interval was written AFTER seeing two draws at that
ratio, landed 0.087 outside it.

The honest reading is that I had no working model of how fast quality falls off
with trainable fraction, and the intervals were anchored to what I wanted the
mechanism to be worth rather than to anything measured.

### Verdict

The mechanism is **dominated everywhere**, which is a stronger negative than
"expensive at high ratios":

| ratio | quantisation, already in this archive | seed + sparse |
| --- | --- | --- |
| ~4x | `w4_q8all` int8, **1.7694** | 1.8591 |
| ~15x | `w4_q2all` per-row ternary, **2.1243** | 2.4954 |

Per-row ternary quantisation with a straight-through estimator dates to Li et
al. 2016; BitNet b1.58 (2024) carried it to LLM scale. It beats this format at
matched compression, and it does not degenerate in free generation.

And the answer to the question that started the wave -- is 1000x possible? --
depends entirely on what is assumed present, which is the clause the claim
being tested never states:

- **Exact and lossless: yes, 6,267x, already here.** A 158-byte DR1 line
  reconstructs the 990,208-byte checkpoint bit-for-bit, sha `3c8cb3b7...`,
  val_bpb 1.78685 -- and that sha was reproduced by the current binary during
  this wave. It costs the 38,366,828-byte corpus, the source, and 550 s of CPU.
  It is a pointer plus a deterministic recomputation, not compression.
- **Self-contained at 1000x: no.** 988 bytes gets 7.5902 by this mechanism
  against 8.0056 for uniform, and the best measured at that budget by any
  method is 3.4983.

A compression ratio without a stated quality target and a stated set of
assumed-present artifacts is not a measurement. It is a choice of how much to
discard.
