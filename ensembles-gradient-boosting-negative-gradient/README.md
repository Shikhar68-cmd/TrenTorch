# Gradient Boosting: fit one tree to the negative gradient of the loss

Intermediate | classical-ml | ensembles | gradient-boosting

### The problem, from first principles

Bagging trains independent trees. Boosting trains the next tree specifically to correct the current ensemble's remaining errors. For squared error, that correction target has a simple exact form: the signed residual.

### From theory to code

Implement `negative_gradient` and `fit_tree_to_negative_gradient`, using the existing regression-tree builder to fit the current residuals rather than original targets.

### Constraints

- `targets` and `predictions` have matching shapes.
- The negative gradient is `targets - predictions`, not the opposite sign.
- A perfect prediction produces a zero gradient.
- Fit `build_regression_tree(input, residuals, max_depth)`.
- Do not rebuild regression-tree logic here.

### Hints

Open one at a time. Each gives away a little more than the last.

<details><summary>Hint 1</summary>

For a squared-error underprediction, the correction must be positive.

</details>

<details><summary>Hint 2</summary>

Compute the residual through `negative_gradient`, then pass it as the tree's target array.

</details>
