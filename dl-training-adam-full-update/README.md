# Adam: Update Rule

Intermediate | optimization

### The problem, from first principles

`Adam: bias-corrected moment estimates` built the two pieces, `m` (a bias-corrected running average of the gradient, exactly `SGD + Momentum`'s velocity) and `v` (a bias-corrected running average of the SQUARED gradient, a new idea: tracking gradient MAGNITUDE separately from direction). This question assembles them into Adam's actual parameter update, and the way they combine is genuinely clever: `m_hat` decides WHICH WAY to step (momentum's own job), while `sqrt(v_hat)` decides HOW FAR, dividing the step down for any parameter whose gradients have consistently been large, and allowing a relatively bigger step for a parameter whose gradients have consistently been small.

This gives every parameter, effectively, its OWN adaptive learning rate, computed automatically from its own gradient history, rather than the single, uniform `lr` every parameter in `SGD`/`SGD + Momentum` shares.

### From theory to code

Theory updates and bias-corrects both moments (reusing `Adam: bias-corrected moment estimates`'s own functions directly), then combines them into one update: step opposite the corrected mean gradient, scaled by the learning rate and divided by the corrected root-mean-square gradient magnitude (plus a tiny `eps` for numerical safety).

Implement `adam_step(params, grads, m_list, v_list, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)` against that reasoning. The signature and docstring are already in the editor.

### Constraints

- `t` starts at `1` (1-indexed), matching `bias_correct`'s own convention.
- Both `m_list` and `v_list` must be threaded through to the caller's next step (same pattern `SGD + Momentum`'s `velocities` used).
- `lr` defaults to `0.001`, matching real Adam's own much-smaller-than-SGD default (the adaptive per-parameter scaling means a smaller base `lr` is typically appropriate).

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

For each `(param, grad, m, v)`, call `update_moments` then `bias_correct` on both results, exactly the two functions `Adam: bias-corrected moment estimates` already built.

</details>

<details>
<summary>Hint 2</summary>

The final update is `param - lr * m_hat / (np.sqrt(v_hat) + eps)`, one line, reusing the corrected moments you just computed.

</details>
