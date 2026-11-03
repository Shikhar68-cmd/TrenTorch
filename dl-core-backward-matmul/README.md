# Backward for matmul

Intermediate | neural-networks | autograd

### The problem, from first principles

`Backward for multiplication` handled scalars, where "sensitivity to `a` depends on `b`" is a single number times a single number. `linear`, this curriculum's very first question, is built on matrix multiplication instead, `input @ weight.T`, and every gradient computed in `linear_regression` and `classification`'s training loops secretly depended on getting THIS backward rule right, even though those tracks derived it by hand for their specific case rather than deriving the fully general matmul backward rule this question builds directly.

The genuinely new difficulty here, beyond `Backward for multiplication`'s scalar case: the SHAPES have to work out. `grad_output` has the shape of the OUTPUT, `C`, but the gradients you need to return must match the shapes of the INPUTS, `A` and `B`, which are generally different shapes entirely. Getting the transpose placement right is what makes those shapes line up correctly.

### From theory to code

Theory derives `dL/dA` and `dL/dB` for `C = A @ B` by generalizing the scalar chain rule to matrices, landing on a formula built entirely from matrix multiplications and transposes, the exact operations `03-matrix-multiplication` and `Transpose, and its role in reshaping without copying data` (both Math & Statistics) already cover individually.

Implement `matmul_backward(grad_output, a, b)` against that reasoning. The signature and docstring are already in the editor.

### Constraints

- `a` is `(m, k)`, `b` is `(k, n)`, `grad_output` is `(m, n)` (matching `C`'s shape).
- `grad_a` must come out shaped `(m, k)` (matching `a`), `grad_b` shaped `(k, n)` (matching `b`).
- Use only `@` and `.T`, no explicit element-by-element loop.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`grad_a`'s shape must match `a`'s shape, `(m, k)`. Only one arrangement of `grad_output` `(m, n)` and `b.T` `(n, k)` multiplies to that shape.

</details>

<details>
<summary>Hint 2</summary>

`grad_a = grad_output @ b.T`, `grad_b = a.T @ grad_output`, check the shapes yourself: `(m,n)@(n,k) = (m,k)` and `(k,m)@(m,n) = (k,n)`, both match.

</details>
