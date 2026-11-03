# Backward for multiplication

Beginner | neural-networks | autograd

### The problem, from first principles

`Backward for addition` was almost suspiciously simple, the local derivative didn't even depend on the input VALUES, just the operation. Multiplication breaks that pattern immediately: `z = a * b`'s sensitivity to `a` depends directly on what `b` happens to be (double `b` and a nudge to `a` moves `z` twice as much), and vice versa. This is the first operation in this track whose backward rule genuinely needs to remember something about the FORWARD pass, the input values themselves, not just the operation's identity.

### From theory to code

Theory applies the chain rule to `z = a * b`: `dz/da = b` and `dz/db = a`, so the gradient flowing back to each input is the upstream gradient scaled by the OTHER input's value.

Implement `mul_backward(grad_output, a, b)` against that reasoning. The signature and docstring are already in the editor.

### Constraints

- Unlike `add_backward`, this function needs `a` and `b`'s actual values, not just `grad_output`.
- Return `(dL/da, dL/db)`, in that order.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`d(a*b)/da = b` (treating `b` as a constant while differentiating with respect to `a`), and symmetrically `d(a*b)/db = a`.

</details>

<details>
<summary>Hint 2</summary>

`grad_a = grad_output * b`, `grad_b = grad_output * a`, each input's gradient is scaled by the OTHER input, not its own.

</details>
