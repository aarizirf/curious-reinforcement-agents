## Does inverse dynamics actually look away?

Curiosity-driven agents fail in a specific, famous way. Put a television showing
static in the corner of a maze, give the agent an intrinsic reward proportional to
how badly it predicts the next observation, and the agent stops exploring the maze.
The static is unpredictable forever, so the reward never decays. The agent has found
a slot machine that always pays.

The standard answer is to stop rewarding prediction error in pixel space and start
rewarding it in a *learned feature space* — one built to contain only what the agent
can control. ICM builds that space with an inverse dynamics loss: encode two
consecutive states, and from the pair of embeddings predict the action that connected
them. The argument for why this works is clean enough to state in a sentence. Noise
carries no information about which action was taken, so a representation optimized to
recover the action has no reason to keep the noise.

We wanted to know how far that argument actually reaches. It has a load-bearing
assumption sitting right in the middle of it — that the noise is independent of what
the agent does. Real environments are not always so accommodating. A television with
a channel button is still a television, and pressing it is still an action. So we
built the smallest environment we could that isolates that one change.

### One branch, two worlds

An eight-cell corridor with walls at both ends. The agent's position is three bits of
signal. Bolted alongside it is a "TV": thirty-two independent random bits, which the
agent observes but which say nothing about where it is. Three actions — left, right,
press. The entire experiment lives in what `press` does:

```python
if a == LEFT:
    self.pos = max(0, self.pos - 1)
elif a == RIGHT:
    self.pos = min(N_POS - 1, self.pos + 1)
else:
    if self.slot:                                    # SLOT: press resamples
        self.bits = self.rng.integers(2, size=K_BITS)
```

In **FROZEN**, `press` does nothing and the TV is ordinary action-independent noise.
In **SLOT**, `press` reshuffles every bit. Nothing else differs — same architecture,
same data volume, same optimizer.

The corridor's walls are doing quiet but essential work here. Moving into a wall and
pressing the button both leave position unchanged, so at a wall these two actions are
indistinguishable from the position bits alone. That is the one place in the
environment where the TV could possibly be useful.

![Observation pairs at the left wall. In FROZEN, both left and press leave the
observation byte-identical. In SLOT, press changes the TV bits.](figures/ambiguity.svg)

We like this setup because it hands us something rare in representation learning: a
number we can derive before running anything. In FROZEN, a press and a move-into-wall
produce byte-identical observation pairs, so no model of any size can exceed **0.50**
on those transitions. In SLOT, `press` flips about half the bits, and a resample
reproduces the previous pattern with probability 2⁻³², so the ceiling is **~1.00**.
The gap between those two numbers is not an empirical effect we hope to observe. It
is a property of the dataset, and any model that crosses 0.50 in SLOT has provably
used the TV.

That prediction extends to the overall accuracy, which turns out to be a good check
that we wired everything correctly. The unresolvable transitions are one sixth of the
data — a quarter of states sit at a wall, two of three actions are stationary there —
and we get half of them by guessing, so FROZEN should land at 1 − 1/12 ≈ 0.9167. It
lands at 0.913, 0.910, 0.912, 0.913 across four embedding widths.

### The encoder cannot look away, but it also barely looks

SLOT reaches 0.96 on the wall-stationary states against FROZEN's 0.51. So the answer
to the headline question is yes: when an action resamples the noise, the inverse loss
can no longer discard it, and this holds all the way down to a two-dimensional
embedding.

The more interesting question is *what* it kept. We freeze the encoder and ask three
separate things of it. Two are content questions — can a decoder recover the actual
TV bits from φ(s)? — asked once with logistic regression and once with an MLP, because
a linear probe alone cannot distinguish "the information is gone" from "the
information is tangled." The third is a detection question, and it has to be asked of
the *pair*:

```python
z      = phi(obs)                                   # content: what did the TV say?
z_pair = torch.cat([phi(obs), phi(next_obs)], 1)    # detection: did the TV change?
```

That split matters more than it looks. φ never sees a transition — it is applied to
each state independently, and the inverse head does the comparing. So φ cannot store
"the TV changed." All it can do is map different bit patterns to different points and
let something downstream notice they moved.

![At EMB_DIM=2, SLOT reaches 0.96 telling press from wall-move and 0.99 detecting a
TV resample, but only 0.55 reading the TV bits.](figures/detection-vs-content.svg)

Detection reads 0.99. Content reads 0.55, and the MLP does no better than the linear
probe. Same weights, same probe budget, same held-out states.

We read this as a hash. The encoder has learned a function whose output reliably moves
when the TV changes without preserving what the TV said — which is, on reflection,
exactly and only what the inverse loss asked for. The loss never requested the bits.
It requested the ability to tell three actions apart, and a scrambled two-dimensional
signature is sufficient for that.

It is worth being precise about the residual. SLOT's content probes do sit slightly
above FROZEN's — about +0.011 on the MLP probe, positive in all three seeds — so the
representation is not perfectly clean. But the leak is roughly +0.01 against
detection's +0.45. Both numbers belong in the claim: φ retains *some* content, and
vastly more detection than content.

### What we think this does and doesn't establish

The honest scope is narrower than "ICM doesn't solve the noisy TV," and we want to
state it that way. Against a television that flickers on its own, Pathak's argument
holds, and our FROZEN condition reproduces it — the probe sits at chance and the
encoder gains nothing from the noise. What breaks is the *action-triggered* case.

There is also a gap between what we measured and what we would like to conclude. The
noisy-TV pathology is a claim about intrinsic reward and about behavior. There is no
forward model here, no reward, no policy, no closed loop in which the agent seeks out
the TV and reshapes its own data distribution. What we have established is a
representational precondition — the feature space cannot be TV-blind, and anything
built on top of it inherits that. The deduction from there to "intrinsic reward stays
high" is sound in direction but silent on magnitude, and magnitude is the whole
question. A two-dimensional hash contributes *some* variance to forward-prediction
error. Whether it contributes enough to outweigh the error on ordinary position
prediction is not something these numbers can tell us. It could turn out to be true
and negligible.

That is the next experiment, and it is a small one: add the forward model, measure
prediction error on press versus move in both conditions, and compare it against the
baseline error the agent would see anywhere else. We would also like a positive
control on the probe — reading *position* out of the same two-dimensional embedding,
which should come back near 1.0 and would show the probe is failing specifically on
the noise rather than failing generally.

Further out, the questions we actually care about are whether the hash survives scale
(thirty-two bits into two dimensions is a soft test, and somewhere there is a
collision rate that bites), and whether any of this reproduces in a convolutional
encoder on pixels, which is where the original argument was made and where a
"signature of TV static" is a much less obvious object.

The broader thing we take from this round: representations trained on a proxy
objective satisfy the proxy in the cheapest available way, and the cheapest way is
often not the one the intuition behind the objective assumed. Inverse dynamics does
not learn *what the environment is doing*. It learns just enough to name the action.
Most of the time those coincide. The noisy TV with a button is a case where they come
apart, and it is worth knowing which of the two your feature space is actually giving
you.

---

*Code: [`corridor.py`](corridor.py), [`rollout.py`](rollout.py),
[`icm.py`](icm.py), [`probe.py`](probe.py), [`sweep.py`](sweep.py). Full
results and caveats in the [README](README.md). Everything runs on CPU in a few
minutes.*
