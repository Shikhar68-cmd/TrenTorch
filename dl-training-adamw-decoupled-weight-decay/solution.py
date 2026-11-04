import numpy as np

def adamw_step(
    params,
    grads,
    m_list,
    v_list,
    t,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.01
):
    m_new = []
    v_new = []
    new_params = []

    for param, grad, m, v in zip(params, grads, m_list, v_list):

        m_updated, v_updated = update_moments(
            m, v, grad, beta1, beta2
        )

        m_hat = bias_correct(m_updated, beta1, t)
        v_hat = bias_correct(v_updated, beta2, t)

        decayed_param = param - lr * weight_decay * param

        param_new = decayed_param - lr * m_hat / (
            np.sqrt(v_hat) + eps
        )

        new_params.append(param_new)
        m_new.append(m_updated)
        v_new.append(v_updated)

    return new_params, m_new, v_new
