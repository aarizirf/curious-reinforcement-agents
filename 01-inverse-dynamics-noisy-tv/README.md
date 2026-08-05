### Notes

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
