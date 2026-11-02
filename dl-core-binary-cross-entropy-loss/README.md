# Binary Cross-Entropy

Intermediate | deep-learning | loss-functions | autograd

### The problem, from first principles

`02-cross-entropy` handles the general multi-class case via a full softmax over every class. Binary classification is the two-class special case of that same idea, common enough (and simple enough) that it gets its own dedicated, explicit loss rather than routing through a full softmax over two classes.

### From theory to code

Implement `bce_loss_forward(probs, target, reduction="mean")` and `bce_loss_backward(probs, target, reduction="mean", grad_output=1.0)`, mirroring `torch.nn.functional.binary_cross_entropy`: `probs` is already-sigmoided values in `(0, 1)`, `target` is `0`/`1` labels.

### Constraints

- `probs`: any NumPy array shape, values in `(0, 1)` (already passed through a sigmoid). `target`: matching shape, `0`/`1` labels.
- `reduction="mean"`/`"sum"`/`"none"`: same three modes as `01-mse` and `02-cross-entropy`.
- Must stay finite even when `probs` is exactly `0` or `1` (guard `log` with clipping).
- `bce_loss_backward` returns the gradient with respect to `probs` (not logits) — a different, messier expression than the fused sigmoid+BCE gradient.
- No Python loops; no mutating `probs` or `target`.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

Only one of the two log terms is ever "active" per example, depending on whether `target_i` is `0` or `1` — but you don't need an `if`, the `(1 - target_i)` and `target_i` factors already zero out whichever term doesn't apply.

</details>

<details>
<summary>Hint 2</summary>

Clip `probs` into `[eps, 1 - eps]` before any `log`, in both the forward AND backward pass — a confident, well-trained sigmoid can genuinely output exactly `0.0` or `1.0` in floating point.

</details>
