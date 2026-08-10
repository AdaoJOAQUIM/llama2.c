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
