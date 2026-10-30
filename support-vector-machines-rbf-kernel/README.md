# RBF kernel

Intermediate | classic-ml

### The problem, from first principles

`Linear kernel` measures similarity with a dot product, which only ever draws straight boundaries. The radial basis function (RBF) kernel, also called the Gaussian kernel, measures similarity by distance instead: two points that are close score near `1`, and points far apart score near `0`. A classifier built on it can carve out curved, island-shaped regions that no hyperplane could.

Given `X` with shape `(n, d)` and `Y` with shape `(m, d)`, return the `(n, m)` matrix of RBF similarities with a width parameter `gamma`.

### Constraints

- Return an array of shape `(n, m)` where entry `[i, j]` equals `exp(-gamma * ||X[i] - Y[j]||^2)`.
- `gamma` defaults to `1.0`.
- Every entry must be finite, including the diagonal of `rbf_kernel(X, X)`, which must be exactly `1` up to floating point error.
- Do not modify `X` or `Y`.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

Squared distance expands as `||x||^2 + ||y||^2 - 2 x.y`. That expansion gives the whole matrix from three pieces, with no double loop.

</details>

<details>
<summary>Hint 2</summary>

Floating point rounding can make a squared distance slightly negative for identical points, which would push `exp` above `1`. Clip the squared distances at zero before exponentiating.

</details>
