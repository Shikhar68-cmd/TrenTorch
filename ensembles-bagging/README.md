# Stretch: bagging concept

Beginner | classical-ml | ensembles | random-forest | stretch

### The problem, from first principles

Majority voting needs trees that are not exact copies. Bagging creates that diversity by training the same learner on many same-sized random samples drawn with replacement. A row may appear repeatedly or not at all, but its label must always travel with it.

### From theory to code

Implement `bootstrap_sample` and `train_random_forest`. Draw paired bootstrap rows, then train exactly `n_trees` with the existing `build_tree` function.

### Constraints

- Bootstrap input and label arrays preserve their original shapes.
- Draw exactly `input.shape[0]` indices from `[0, n_samples)` with replacement.
- Use the same index array for `input` and `labels`.
- Build one `np.random.default_rng(seed)` before the tree loop.
- Return exactly `n_trees` independently bootstrapped `build_tree` results.
- Equal fixed seeds reproduce a call; a reused generator must advance between trees.

### Hints

Open one at a time. Each gives away a little more than the last.

<details><summary>Hint 1</summary>

Replacement means an index-drawing operation may repeat a value; it does not need a permutation.

</details>

<details><summary>Hint 2</summary>

Pass one advancing generator into each bootstrap call rather than re-creating it inside the loop.

</details>
