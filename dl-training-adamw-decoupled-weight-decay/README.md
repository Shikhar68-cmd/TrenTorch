# AdamW: decoupled weight decay

Intermediate | optimization

### The problem, from first principles

`Ridge Regression (L2)` (Classical ML) added its penalty directly to the LOSS, `mse_loss + alpha * sum(weight^2)`. Differentiating that combined loss, the penalty shows up as an extra term ADDED INTO the gradient: `grad + 2*alpha*weight`. For plain SGD, adding that extra term to the gradient before the update is perfectly fine, mathematically identical to a direct parameter shrinkage. But `Adam: full update rule` doesn't just use the raw gradient, it feeds it through `m` and `v`, the adaptive moment estimates, BEFORE using it. Sneak a weight-decay term into the gradient before it enters that adaptive machinery, and it gets treated like any other gradient signal, divided by `sqrt(v_hat)` along with everything else, which means the ACTUAL amount of shrinkage a parameter receives ends up depending, unpredictably, on that parameter's own gradient history, not a clean, uniform "shrink every weight by this fixed fraction" the way L2 regularization is supposed to behave.

AdamW's fix has a name that says exactly what it does: DECOUPLE weight decay from the gradient-based update entirely. Apply it as its own, separate, always-the-same-size shrinkage step, directly to the parameter, with the adaptive Adam update layered on top, independently.

### From theory to code

Theory computes Adam's usual moment-based update (identical to `Adam: full update rule`), but applies weight decay as a SEPARATE term, `lr * weight_decay * param`, subtracted from the parameter directly, never mixed into the gradient or the moment estimates at all.

Implement `adamw_step(params, grads, m_list, v_list, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)` against that reasoning.

### Constraints

- Weight decay applies to `param` DIRECTLY (`param - lr * weight_decay * param`), never added into `grad` before the moment updates.
- Everything else matches `Adam: full update rule`'s own structure exactly (reuse `update_moments` and `bias_correct`).
- `weight_decay` defaults to `0.01`, matching a common real-world default.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

Compute the moment-based update exactly like `Adam: full update rule` does, unchanged.

</details>

<details>
<summary>Hint 2</summary>

Apply weight decay as a SEPARATE subtraction from `param` first (`param - lr * weight_decay * param`), THEN subtract the Adam update from that decayed value.

</details>
