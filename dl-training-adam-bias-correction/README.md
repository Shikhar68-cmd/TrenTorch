# Adam: Bias Correction

Intermediate | optimization

### The problem, from first principles

`SGD + Momentum` maintained one running average of gradients (velocity), initialized at zero. That zero initialization has a real, measurable cost right at the START of training: on the very first step, the running average is mostly still "zero, blended with a tiny sliver of real gradient," a systematic UNDERESTIMATE of the true gradient's actual scale. `SGD + Momentum` never corrects for this (it doesn't need to, in practice the effect is small and the optimizer isn't ALSO trying to divide by this quantity), but Adam, the next real optimizer this track builds toward, tracks two running averages, and specifically USES one of them (the squared-gradient average) as a DIVISOR in its update rule, which makes an early, zero-biased underestimate a genuinely serious problem: dividing by a falsely-small number early in training would produce a wildly oversized, unstable step.

This question builds Adam's fix in isolation, before the full update rule needs it: a precise correction formula that removes exactly the zero-initialization bias, self-adjusting so it matters a lot on step 1 and fades away naturally as training proceeds.

### From theory to code

Theory tracks two running averages, `m` (mean of the gradient) and `v` (mean of the SQUARED gradient), each updated with an exponential-moving-average formula, and derives an exact correction factor, `1 / (1 - beta^t)`, that removes the systematic zero-initialization bias at step `t`.

Implement `update_moments(m, v, grad, beta1=0.9, beta2=0.999)` first, then `bias_correct(moment, beta, t)`.

### Constraints

- `m` and `v` both start at `0.0` before the first call.
- `t` is 1-indexed: `t=1` is the very first update.
- `bias_correct` works identically for both `m` (with `beta1`) and `v` (with `beta2`), it's the same formula, only which moment and which beta differ.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`update_moments` is two exponential-moving-average updates, one for `grad`, one for `grad**2`, using `beta1` and `beta2` respectively.

</details>

<details>
<summary>Hint 2</summary>

`bias_correct(moment, beta, t)` is `moment / (1 - beta**t)`, a single division.

</details>
