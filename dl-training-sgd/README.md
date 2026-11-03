# SGD

Beginner | optimization

### The problem, from first principles

`04-gd-step` (Classical ML) already implemented gradient descent's update rule, but specifically for one weight matrix and one bias vector, `linear_regression`'s own two parameters. A real neural network has dozens, sometimes billions, of parameters spread across many layers, and every single one of them needs the exact same update rule applied, independently, every training step. This question generalizes `04-gd-step`'s specific update into the general form every optimizer in this track (and every optimizer in `torch.optim`) actually takes: operate on a LIST of parameters and their gradients, applying one uniform rule to all of them at once.

### From theory to code

Theory applies the identical "step opposite the gradient" rule from `04-gd-step` to every `(parameter, gradient)` pair in two lists, independently.

Implement `sgd_step(params, grads, lr)` against that reasoning. The signature and docstring are already in the editor.

### Constraints

- `params` and `grads` are parallel lists (same length, matched by index).
- Return a NEW list, don't mutate the input arrays in place.
- Every parameter uses the SAME learning rate `lr`.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

This is `04-gd-step`'s exact update formula, `param - lr * grad`, applied inside a loop (or list comprehension) over every parameter.

</details>

<details>
<summary>Hint 2</summary>

`[p - lr * g for p, g in zip(params, grads)]` is the entire function, in one line.

</details>
