# Cross-Entropy

Intermediate | deep-learning | loss-functions | autograd

### The problem, from first principles

`04-softmax` turns raw class scores into a probability distribution. For multi-class classification, the standard loss then asks a single question of that distribution: how much probability did the model assign to the _actual_ correct class? Cross-entropy is exactly that question turned into a differentiable loss — and computing it well means never actually materializing the softmax probabilities and then taking their log separately, since that route reintroduces the same overflow risk `04-softmax`'s row-max shift was built to avoid.

### From theory to code

Implement `cross_entropy_forward(logits, target, reduction="mean")` and `cross_entropy_backward(logits, target, reduction="mean", grad_output=1.0)`, mirroring `torch.nn.functional.cross_entropy`: `logits` is `(n, num_classes)` raw scores, `target` is `(n,)` integer class indices (not one-hot).

### Constraints

- `logits`: shape `(n, num_classes)`. `target`: shape `(n,)`, integer class indices in `[0, num_classes)`.
- `reduction="mean"`: returns a scalar, the average per-sample loss.
- `reduction="sum"`: returns a scalar, the total loss.
- `reduction="none"`: returns shape `(n,)`, the per-sample loss.
- `cross_entropy_backward` takes the same `logits`/`target`/`reduction`, plus `grad_output`, and returns a `(n, num_classes)` gradient regardless of `reduction`.
- Compute in log-space directly (log-softmax), never `log(softmax(logits))` as two separate steps.
- No Python loops over samples; no mutating `logits` or `target`.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`log_softmax(logits)` computed directly (shift by the row max, subtract `log(sum(exp(shifted)))`) avoids ever calling `exp` on a large raw logit and then immediately undoing it with `log` — compute it in one pass, not as `softmax` followed by `log`.

</details>

<details>
<summary>Hint 2</summary>

`log_probs[np.arange(n), target]` picks out, per row, the log-probability the model assigned to that row's true class — negate and reduce.

</details>
