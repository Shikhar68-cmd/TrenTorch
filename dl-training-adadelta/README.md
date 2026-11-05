# Adadelta

Intermediate | deep-learning | optimization | training

### The problem, from first principles

Adagrad adapts the learning rate for each parameter by accumulating squared gradients. This works well when parameters have different update frequencies, but its accumulator only grows over time.

As the accumulated squared gradients become larger, the effective learning rate keeps getting smaller. Eventually, updates can become extremely small even when further learning would still be useful.

Adadelta addresses this by replacing the ever-growing sum with exponentially decaying averages. It keeps track of two quantities:

1. A moving average of squared gradients.
2. A moving average of squared parameter updates.

The update used in this exercise is:

$$
\text{square\_avg}_{\text{new}}
=
\rho \cdot \text{square\_avg}
+
(1-\rho)\cdot\text{grad}^2
$$

$$
\text{delta}
=
\frac{
\sqrt{\text{acc\_delta}+\text{eps}}
}{
\sqrt{\text{square\_avg}_{\text{new}}+\text{eps}}
}
\cdot
\text{grad}
$$

$$
\text{param}_{\text{new}}
=
\text{param}
-
\text{lr}\cdot\text{delta}
$$

$$
\text{acc\_delta}_{\text{new}}
=
\rho\cdot\text{acc\_delta}
+
(1-\rho)\cdot\text{delta}^2
$$

Both `square_avg` and `acc_delta` start at zeros.

### From theory to code

Implement:

```python
adadelta_step(
    param,
    grad,
    square_avg,
    acc_delta,
    lr,
    rho,
    eps,
)
```

The function returns:

```python
(new_param, new_square_avg, new_acc_delta)
```

The signature is provided in the editor.

The function should first update the moving average of squared gradients. That value determines the scale of the current update.

After computing the update `delta`, use it to update both the parameter and the moving average of squared updates.

### Constraints

- `param`, `grad`, `square_avg`, and `acc_delta` must have the same shape. A mismatch raises `ValueError`.
- `lr` must be greater than `0`. Otherwise, raise `ValueError`.
- `rho` must be in `[0, 1)`. Otherwise, raise `ValueError`.
- `eps` must be greater than `0`. Otherwise, raise `ValueError`.
- Inputs must not be modified.
- Return new arrays.
- `square_avg` starts at zeros.
- `acc_delta` starts at zeros.

### Hints

<details>
<summary>Hint 1</summary>

Compute `square_avg_new` first. It is an exponentially weighted average of the squared gradient.

</details>

<details>
<summary>Hint 2</summary>

Compute `delta` using both the previous `acc_delta` and the newly computed `square_avg_new`.

</details>

<details>
<summary>Hint 3</summary>

After computing `delta`, use it to update both `param` and `acc_delta`.

</details>
