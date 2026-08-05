# Does an inverse-dynamics encoder stay blind to a noisy TV?

ICM (Pathak et al. 2017) trains a state encoder φ with an inverse dynamics loss
only: given φ(s), φ(s′), predict the action taken between them. The paper argues
this makes φ discard environment noise, since noise carries no information about
which action was taken — and that this is what protects a curiosity agent from
the noisy-TV trap.

**The claim tested here:** that argument holds only while the noise is
action-*independent*. Give the agent a button that *resamples* the noise, and the
noise becomes the thing that identifies the button press, so the inverse loss can
no longer discard it.

This repository is an encoder-only test of that claim. No policy, no reward, no
forward model, no RL loop. See [Scope](#scope-what-this-does-not-show).

---

## Setup

8-cell corridor, position `p ∈ [0,7]`, walls at both ends. The TV is K=32
independent random bits. Observation is `one_hot(p, 8) ++ bits` → 40 dims, of
which 3 bits are signal and 32 are noise. Actions: `left`, `right`, `press`.

Two conditions, differing in one branch:

| | `press` does |
|---|---|
| **FROZEN** | nothing — TV bits fixed for the episode |
| **SLOT** | resamples all 32 bits |

`press` and move-into-wall both leave `p` unchanged, so position alone cannot
separate them. In SLOT, the resampled TV is the only tell.

Data is 100k transitions under a uniform random policy, in episodes of 100 steps,
with `p` and the TV resampled at each episode start. Per-episode resampling is
required: with one TV draw for the whole dataset the probes have a constant
target and read 100% trivially.

φ is `40 → 64 → EMB_DIM`; the inverse head is `2·EMB_DIM → 64 → 3`. Trained
jointly on cross-entropy against the true action, Adam, lr 3e-3, 15 epochs.

### The ceilings are derived, not measured

In FROZEN, `press` leaves the observation byte-identical, so left-at-p=0 and
press-at-p=0 produce indistinguishable pairs. **No model can beat 0.50** on those
transitions. In SLOT, `press` flips ~16 bits and a resample reproduces the old
pattern with probability 2⁻³², so the ceiling is **~1.00**.

That yields a falsifiable prediction for FROZEN's overall accuracy. The position
chain is a lazy random walk with clamping, whose stationary distribution is
uniform:

- P(at a wall) = 2/8 = 0.25
- given a wall, P(action is one of the two stationary ones) = 2/3
- unresolvable fraction = 1/6, half of which you get by guessing → lose 1/12

Predicted **0.9167**. Measured: 0.9133 / 0.9097 / 0.9118 / 0.9131 across all four
embedding widths. FROZEN sits exactly where the information budget says it must,
which is the main evidence that the environment and evaluation are wired
correctly.

---

## The four tests

### 1. Inverse accuracy on wall-stationary states — `inv_wall`

*Can the trained inverse model tell `press` from move-into-wall?*

Filter held-out transitions to: at a wall, position unchanged. Position is useless
here by construction, so this is the only place the TV can contribute. FROZEN's
ceiling is 0.50, SLOT's is 1.00.

This test does not involve a probe. It asks what the trained system achieves, and
because the FROZEN ceiling is a hard information limit, SLOT exceeding 0.50 is
proof that φ carries TV information. Everything else in this repo is about *what
form* that information takes.

### 2. Linear content probe — `lin`

*Can you read the TV bits back out of φ(s) with a linear decoder?*

Freeze φ. Generate fresh states. For each of the 32 bits, fit logistic regression
from φ(s) to that bit on 10k states, score on 5k held out, average across bits.
Chance is 0.50.

### 3. Nonlinear content probe — `mlp`

*Can you read the bits out with a nonlinear decoder?*

Identical protocol, `MLPClassifier((64,64))` instead of logistic regression, over
8 of the 32 bits (they are iid, so the mean is sound; it is a noisier estimate
than `lin`).

This exists because `lin` alone cannot distinguish **"the information is gone"**
from **"the information is present but tangled."** That distinction matters here
more than usual: EMB_DIM=2 is *not* an information bottleneck — two float32s have
mantissa enough to hold all 35 bits of state — so nothing forces the noise out on
capacity grounds. Whether φ actually drops it has to be measured.

### 4. Change detector — `change`

*Can you read "the TV just resampled" out of the **pair** (φ(s), φ(s′))?*

An MLP on the concatenated pair, predicting whether the transition was a press.
SLOT only; in FROZEN the target is constant.

This is a different question from tests 2 and 3, and the contrast between them is
the main result. Note φ never sees a transition — it is applied to each state
separately and the inverse head does the comparison — so φ cannot *store*
"changed." All it can do is map different bit patterns to different points, and
let a downstream module compare them.

### The reference column — `lin_init`

A linear probe on φ **before training**. This is not a control on the effect; it
is a **validity check on the instrument**. It measures how much a random
projection of this width already leaks, which bounds how much headroom the probe
has to detect anything training did.

---

## Why sweep EMB_DIM

`lin_init` is the reason, and it is the single most important methodological point
here.

```
EMB_DIM        2        4        8       32
lin_init    0.5770   0.6307   0.6655   0.9178
```

At EMB_DIM=32, an untrained random projection already leaks **92%** of the TV
bits. The probe is pinned near ceiling before training starts, so it cannot tell
you whether training preserved or destroyed anything — there is no dynamic range
left. Any content claim measured at that width is uninterpretable.

At EMB_DIM=2, `lin_init` is 0.577, close enough to chance that the probe has room
to move. So the sweep is not a hyperparameter search; it maps **where the
measurement is valid**. The narrow rows carry the content claims; the wide rows
exist to show why you cannot make those claims at width 32.

The reason width controls the probe and not the encoder: a linear decoder reads at
most ~EMB_DIM independent directions out of φ(s), regardless of how much is
encoded in there.

**Minimum useful sweep: {2, 32}** — one width where the probe works, one showing
saturation. {2, 8, 32} additionally shows the trend is monotone. The four-width
version is not load-bearing.

---

## Results

SEED=0, `python sweep.py`:

```
 emb cond     inv_all  inv_wall   ceil  lin_init     lin     mlp  change
------------------------------------------------------------------------
   2 FROZEN    0.9133    0.5113  ~0.50    0.5770  0.4952  0.5485     nan
   2 SLOT      0.9890    0.9360  ~1.00    0.5771  0.5092  0.5622  0.9878

   4 FROZEN    0.9097    0.4907  ~0.50    0.6307  0.5673  0.5878     nan
   4 SLOT      0.9987    0.9928  ~1.00    0.6274  0.5869  0.6220  0.9990

   8 FROZEN    0.9118    0.5028  ~0.50    0.6655  0.6274  0.5814     nan
   8 SLOT      0.9980    0.9879  ~1.00    0.6948  0.6504  0.6350  0.9974

  32 FROZEN    0.9131    0.5101  ~0.50    0.9178  0.8855  0.8261     nan
  32 SLOT      0.9980    0.9879  ~1.00    0.9526  0.9110  0.9099  0.9980
```

Seeds 0/1/2 at the two widths where the probe is valid, **paired by seed**
(SLOT − FROZEN within a seed, since seeds share init and data structure):

```
                emb=2                        emb=4
          gap per seed        mean     gap per seed        mean
inv_wall  +.425 +.487 +.448   +0.453   +.502 +.499 +.470   +0.490
lin       +.014 +.054 -.005   +0.021   +.020 +.005 +.005   +0.010
mlp       +.014 +.012 +.009   +0.011   +.034 -.021 +.033   +0.015

change (SLOT only)   0.9916 ± 0.0039           0.9982 ± 0.0007
```

---

## Conclusions

**1. A resampling action does force φ to carry the TV.** SLOT reaches 0.94–0.99 on
wall-stationary states at every width including EMB_DIM=2, against a FROZEN
ceiling of 0.50 that is a hard information limit rather than a baseline. FROZEN
sits at that ceiling everywhere. The paired gap is +0.45. This part is
unambiguous and does not depend on any probe.

**2. What φ carries is a change detector, not the noise content.** At EMB_DIM=2,
one frozen SLOT encoder:

| question | reads |
|---|---|
| did the TV resample? (`change`) | **0.9916 ± 0.0039** |
| what were the bits? (`mlp`) | **0.5489 ± 0.0147** |

Detection ≈ 0.99, content ≈ 0.55, same weights, same probe budget. That is a
hash: a function whose output reliably moves when its input changes, without
preserving the input. The MLP column is what makes this a claim about the
representation rather than about linear decodability — a nonlinear decoder does no
better than logistic regression.

**3. The content leak is real but small.** SLOT's content probes do sit above
FROZEN's: +0.011 on `mlp` at EMB_DIM=2, positive in all three seeds. So φ is not
perfectly clean — some bit information survives. But it is ~+0.01 against
detection's +0.45. Both facts are needed to state the result precisely: φ retains
*some* content, and *far* more detection than content.

**So "φ must encode the noise content" is too strong. "φ must encode a change
detector for the noise" is what the data supports.**

**4. What this implies for ICM, as a deduction rather than a measurement.** If φ
is TV-sensitive, every module operating in φ-space inherits that sensitivity — a
forward model predicting φ(s′) from (φ(s), press) cannot predict the hash of
freshly-drawn bits, so prediction error stays irreducibly high at the TV. That is
a valid necessary-condition argument, and it is the limit of what encoder-only
work can establish. The *magnitude* of the resulting intrinsic reward is not
measured here and does not follow from these numbers. See below.

---

## Scope: what this does *not* show

Stated plainly, because the gap between what is measured and what is claimed is
where this kind of result usually breaks.

**The failure mode itself is not demonstrated.** ICM's noisy-TV pathology is a
statement about *intrinsic reward* and *behavior*: the agent gets stuck at the TV
because forward-prediction error stays high. There is no forward model here, no
reward, no policy. This establishes the **representational precondition** for that
failure — φ cannot be TV-blind — not the failure. Closing that gap is one
experiment: add a forward model, measure prediction error on press vs move in both
conditions.

**And the magnitude is genuinely open.** A 2-dim hash contributes some variance to
forward-prediction error, but whether that variance is large *relative to the
error on ordinary position prediction* is unmeasured. "Intrinsic reward stays
high" could be true and negligible. This is the most important open question.

**Scope of the claim about ICM.** The classic noisy TV changes on its own,
independent of the agent's actions. Against *that*, Pathak's argument works, and
FROZEN confirms it: the probe sits at chance and the encoder gains nothing. The
result here applies specifically to **action-triggered** noise — the "TV with a
channel button" variant. That is a real case, and one the original invariance
argument is usually stated broadly enough to cover, but it is narrower than "ICM
does not solve the noisy TV."

**The environment is a toy.** One-hot corridor, 40-dim observations, MLP encoder.
Pathak's setting is pixels and convnets. Whether a conv encoder trained on real
observations forms a comparable hash of TV static is untested.

**The data comes from a random policy.** Real ICM has a feedback loop — the agent
seeks the TV, which changes the data distribution, which changes φ. That loop is
absent here, and the noisy-TV pathology is fundamentally a closed-loop phenomenon.

**Probe negatives are decoder-relative.** "Not recoverable by logistic regression
or a 64×64 MLP from 10k samples" is strong evidence but is not
information-theoretic absence.

**n = 3 seeds**, one architecture, one hyperparameter setting.

### The incremental contribution, stated conservatively

- *Known:* inverse-dynamics features discard action-independent noise. FROZEN
  reproduces this.
- *Argued here, with an information-theoretic floor:* when the noise is
  action-triggered, the inverse loss **cannot** discard it. The 0.50 ceiling makes
  this a proof, not an empirical trend.
- *Novel, and the interesting part:* what φ retains is not the noise but a
  low-dimensional signature of it — detection without content. The naive
  expectation is that φ encodes the TV; it does not, and the distinction is
  measurable.

---

## Files

| file | contents |
|---|---|
| `corridor.py` | environment — state, actions, observation encoding |
| `rollout.py` | episodes → `(obs, act, next_obs, tv)` tensors; wall-stationary mask |
| `icm.py` | encoder, inverse head, training loop, evaluation |
| `probe.py` | `content()` (tests 2 and 3), `change()` (test 4) |
| `sweep.py` | runs the grid, prints the table |

Imports run strictly downward: `corridor → rollout → icm → sweep`, with `probe`
off to the side.

```
python sweep.py             show each epoch as it trains
python sweep.py -q          table only
python sweep.py --seed 1    another draw
```

Several minutes, dominated by the MLP probe columns.

Watching it train is worth it once. SLOT's loss sits flat near 0.11 for several
epochs — that is the model solving everything *except* the wall-stationary states,
i.e. sitting at FROZEN's ceiling — then collapses when it finds the TV. FROZEN
never makes that jump.

---

## Notes on the setup

- **`lin` is below `lin_init` in both conditions at every width.** Training makes
  the TV *less* linearly readable than a random projection does, in SLOT too. So
  `lin_init` is a scale reference, not a control.
- Per-episode TV resampling (not per-dataset) is required for the probes to be
  non-degenerate.
- lr 3e-3 / 15 epochs, not 1e-3 / 5. At the slower setting SLOT is nowhere near
  converged at 5 epochs — it climbs from ~0.50 and only reaches ~0.97 near epoch
  25 — which would understate the effect for optimizer reasons rather than
  representational ones. FROZEN is flat at 0.50 under every setting tried.
- Weight decay deliberately omitted. It would create the "drop" pressure that
  makes FROZEN's probe fall, but it is an extra knob and would confound what the
  inverse loss does on its own.
- EMB_DIM=2 is the noisiest cell; `inv_wall` moves several points across seeds
  where wider settings barely move.

## Next

1. **Add the forward model.** Measure intrinsic reward on press vs move in both
   conditions, and its size relative to baseline prediction error. This converts
   the central claim from a deduction to a measurement and is the highest-value
   next step by a wide margin.
2. **Probe position as a control.** If φ at EMB_DIM=2 reads ~1.0 on position while
   reading 0.55 on the TV, that is a much cleaner statement than 0.55 alone — it
   shows the probe works on this encoder and fails specifically on the noise.
3. **Scale K.** 32 bits into a 2-dim hash is a soft test; find where the collision
   rate starts to bite.
4. **Pixels.** Whether any of this survives a conv encoder on real observations.
