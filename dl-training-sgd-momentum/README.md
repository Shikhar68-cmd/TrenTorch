# SGD + Momentum

Intermediate | optimization

### The problem, from first principles

Plain `SGD` steps directly opposite whatever the CURRENT gradient happens to be, every single step, with zero memory of any earlier step. On a loss surface that curves much more steeply in one direction than another (an "ill-conditioned" surface, `Positive-definite matrices, and why they matter for optimization`'s own vocabulary applies here too), this makes SGD zigzag: it overcorrects across the steep direction on every step while crawling painfully slowly along the shallow direction, wasting most of its movement on oscillation rather than genuine progress toward the minimum.

Momentum fixes this with an idea borrowed directly from physics: instead of responding only to the instantaneous gradient, accumulate a "velocity" that builds up over consecutive steps pointing in a consistent direction, and gets partially cancelled out by steps that keep flip-flopping. A ball rolling downhill doesn't instantly reverse direction the moment the slope changes sign, it has momentum, and that's exactly the behavior this optimizer borrows.

### From theory to code

Theory maintains one extra piece of state per parameter, its velocity, blends the current gradient into that velocity (weighted by a `momentum` coefficient), and steps opposite the BLENDED velocity instead of the raw gradient.

Implement `sgd_momentum_step(params, grads, velocities, lr, momentum=0.9)` against that reasoning. The signature and docstring are already in the editor.

### Constraints

- `velocities` is a separate list, parallel to `params`/`grads`, that must be threaded through from one call to the next (a training loop calls this once per step, carrying the returned velocities into the next call).
- Return `(new_params, new_velocities)`, both as new lists.
- `momentum` defaults to `0.9`, matching a common real-world default.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

Update the velocity FIRST: `new_velocity = momentum * velocity + grad`.

</details>

<details>
<summary>Hint 2</summary>

Then step using the NEW velocity, not the raw gradient: `new_param = param - lr * new_velocity`, exactly `SGD`'s own formula with `velocity` standing in for `grad`.

</details>
