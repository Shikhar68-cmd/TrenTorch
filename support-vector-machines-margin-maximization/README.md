# Margin maximization intuition

Intermediate | classic-ml

### The problem, from first principles

`Decision Boundary / Thresholding` drew one line separating two classes, but for most datasets, MANY different lines would separate the two classes equally well, on the training data. Which one should you actually prefer? Support Vector Machines answer this with a specific, principled criterion: prefer the line that sits as FAR as possible from the nearest points of either class, the widest possible "street" between the two classes, rather than a line that happens to squeeze uncomfortably close to some points even while technically separating them correctly.

This question builds the exact quantity SVMs maximize: the margin, the distance from the decision boundary to its single closest point. Understanding this concretely is what makes `Linear SVM via gradient descent on hinge loss` (the next question) make sense as "training an SVM" rather than just "yet another linear classifier with a different loss function."

### From theory to code

Theory defines two related but distinct notions of "margin": the functional margin (a raw, unnormalized number that depends on how large `weight` happens to be) and the geometric margin (the SAME idea, rescaled into an actual, scale-invariant distance). The dataset's overall margin is the SMALLEST geometric margin across every point, whichever point sits closest to the boundary.

Implement `functional_margin(weight, bias, X, y)` first, then `geometric_margin(weight, bias, X, y)` on top of it, then `dataset_margin(weight, bias, X, y)`.

### Constraints

- `y` is `{-1, +1}` (`Hinge loss`'s own convention).
- `geometric_margin` must be invariant to scaling `weight` and `bias` by the same positive constant (Theory explains why this matters).
- `dataset_margin` returns a single scalar: the minimum geometric margin across all points.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`functional_margin` is exactly the same expression `hinge_loss` computes before the `1 -` and `max(0, ...)`: `target * (X @ weight + bias)`.

</details>

<details>
<summary>Hint 2</summary>

Dividing `functional_margin` by `np.linalg.norm(weight)` is what makes it scale-invariant, `dataset_margin` is then just `np.min` over `geometric_margin`'s result.

</details>
