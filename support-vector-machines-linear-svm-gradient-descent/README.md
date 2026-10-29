# Linear SVM via gradient descent on hinge loss

Advanced | classic-ml

### The problem, from first principles

`Margin maximization intuition` defined exactly what an SVM wants: the decision boundary with the largest possible margin to its nearest points. Real SVM solvers usually reach that boundary via a different route entirely (quadratic programming on the problem's "dual" formulation, the classic textbook approach), but there's a much simpler path that reuses everything this curriculum has already built: treat "maximize the margin, subject to correct classification" as an ordinary loss-minimization problem, plug it into the same gradient descent machinery `Full Training Loop` already uses, and let optimization find the answer directly.

`Hinge loss` supplies exactly the right loss for this: minimizing it directly pushes toward correct, confidently-margined predictions, and adding an L2 penalty on the weights (this question's contribution) is precisely what connects "small weights" back to "large margin," `geometric_margin`'s own formula divides by `||weight||`, so keeping `||weight||` small, for a given functional margin, is mathematically the same thing as making the geometric margin large.

### From theory to code

Theory combines mean hinge loss with an L2 penalty into a single scalar objective, derives its (sub)gradient (hinge loss's kink at the margin boundary makes the gradient piecewise, exactly zero for comfortably-correct points, a specific nonzero contribution for margin-violating points), and trains by plain gradient descent, reusing `Full Training Loop`'s own loop structure.

Implement `svm_objective(weight, bias, X, y, lambda_reg)` first, then `svm_gradient(weight, bias, X, y, lambda_reg)`, then `train_linear_svm(input, target, lr=0.01, epochs=1000, lambda_reg=0.01)`.

### Constraints

- `target` is `{-1, +1}` (`Hinge loss`'s own convention throughout this track).
- `svm_gradient` must match `svm_objective`'s true gradient (verify with a finite-difference check before trusting it).
- `train_linear_svm` uses plain gradient descent, the same loop shape `Full Training Loop` and `Full Linear Regression Training Loop` both use.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`svm_objective` is one line, calling the already-imported `hinge_loss` and adding `lambda_reg * np.dot(weight, weight)`.

</details>

<details>
<summary>Hint 2</summary>

`svm_gradient`'s "which points violate the margin" mask is `(target * scores) < 1`; only those points contribute to the hinge-loss part of the gradient, everything else contributes zero.

</details>
