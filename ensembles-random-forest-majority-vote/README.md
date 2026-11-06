# Random Forest: majority vote aggregation

Beginner | classical-ml | ensembles | random-forest

### The problem, from first principles

One tree can make a brittle decision from its particular training data. A forest lets independently built trees vote, so patterns they agree on survive while individual tree mistakes can be outvoted. This question combines already-built classification trees; it does not build or traverse them itself.

### From theory to code

Implement `random_forest_predict(trees, input)` by collecting each tree's predictions and taking a separate majority vote for every input row.

### Constraints

- `trees` is a list of compatible fitted tree dictionaries and `input` has shape `(n_samples, n_features)`.
- Return integer labels with shape `(n_samples,)`.
- Reuse `predict_tree` once per tree; do not reimplement traversal.
- Vote independently down each prediction column.
- Break ties by the lower class label.

### Hints

Open one at a time. Each gives away a little more than the last.

<details><summary>Hint 1</summary>

Stacking one prediction vector per tree produces rows of tree opinions and columns of opinions about one sample.

</details>

<details><summary>Hint 2</summary>

For each column, `np.unique(..., return_counts=True)` supplies sorted labels and vote totals. `np.argmax` chooses the first maximum.

</details>
