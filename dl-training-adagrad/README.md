# Adagrad

Intermediate | deep-learning | optimization | training

### The problem, from first principles

A single learning rate does not always work equally well for every parameter.

Some parameters may receive large gradients over and over again, while others may receive small or infrequent gradients. Applying the same step size to every parameter can make frequently updated parameters move too aggressively and make sparse parameters learn too slowly.

Adagrad addresses this by keeping a running sum of squared gradients for every parameter. Parameters that have accumulated larger squared gradients receive smaller future updates.

The update used in this exercise is:

$$
\text{accum}_{\text{new}}
=
\text{accum}
+
\text{grad}^2
$$

$$
\text{param}_{\text{new}}
=
\text{param}
-
\text{lr}
\frac{\text{grad}}
{\sqrt{\text{accum}_{\text{new}}} + \text{eps}}
$$

The accumulator starts at zeros.

### From theory to code

Implement:

```python
adagrad_step(param, grad, accum, lr, eps)
```

The function returns:

```python
(new_param, new_accum)
```

The signature is provided in the editor.

The accumulator stores the element-wise sum of squared gradients seen so far. It is updated before calculating the parameter update.

### Constraints

- `param`, `grad`, and `accum` must have the same shape. A mismatch raises `ValueError`.
- `lr` must be greater than `0`. Otherwise, raise `ValueError`.
- `eps` must be greater than `0`. Otherwise, raise `ValueError`.
- Inputs must not be modified.
- Return new arrays.
- The accumulator starts at zeros.
- The squared gradient and accumulator are element-wise operations.

### Hints

<details>
<summary>Hint 1</summary>

Compute the new accumulator before computing the parameter update.

</details>

<details>
<summary>Hint 2</summary>

The denominator uses the square root of the new accumulator plus `eps`.

</details>

<details>
<summary>Hint 3</summary>

Write the two update lines exactly as given, then translate each operation directly into NumPy.

</details>
