
import numpy as np

from _load import load_solution

hinge_loss = load_solution("support-vector-machines-hinge-loss").hinge_loss


def svm_objective(weight: np.ndarray, bias: float, X: np.ndarray, y: np.ndarray, lambda_reg: float) -> float:
    """
    The soft-margin SVM's training objective: mean hinge loss plus an
    L2 penalty on the weights (encouraging a SMALL ||weight||, which,
    per Margin maximization intuition's own geometric_margin formula,
    is exactly what makes the margin LARGE for a given functional
    margin, connecting this objective directly back to that question's
    margin-maximization goal).

        objective = hinge_loss(X @ weight + bias, y) + lambda_reg * ||weight||^2
    """
    l2_norm = np.linalg.norm(weight)
    l2_norm_sq = l2_norm**2

    loss = hinge_loss(X @ weight + bias , y) + lambda_reg*l2_norm_sq

    return float(loss)


def svm_gradient(
    weight: np.ndarray, bias: float, X: np.ndarray, y: np.ndarray, lambda_reg: float
) -> tuple[np.ndarray, float]:
    """
    The (sub)gradient of svm_objective. Hinge loss's kink at the
    margin boundary means its derivative is only PIECEWISE defined:
    zero contribution from any point with margin >= 1 (comfortably
    correct, hinge_loss's own "stop caring" property), and a
    contribution of `-y_i * x_i` from every point that violates the
    margin (margin < 1).

        grad_weight = -mean(y_i * x_i for violating points) + 2 * lambda_reg * weight
        grad_bias   = -mean(y_i for violating points)
    """
    score = X @ weight + bias
    margin = y*score

    # Select samples whose margin is less than 1
    violating = margin < 1

    # Gradient of hinge loss + L2 regularization
    grad_wt = (
        -(y[violating, None] * X[violating]).sum(axis=0) / len(y)
        + 2 * lambda_reg * weight
        )

    grad_bias = -np.sum(y[violating]) / len(y)

    return grad_wt ,grad_bias

def train_linear_svm(
    input: np.ndarray, target: np.ndarray, lr: float = 0.01, epochs: int = 1000, lambda_reg: float = 0.01
) -> tuple[np.ndarray, float]:
    """
    Plain gradient descent on svm_objective, using svm_gradient at
    every step, the exact same training-loop shape Full Training Loop
    and Full Linear Regression Training Loop both already use.
    """
    n_samples, n_features = input.shape

    weight = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        grad_weight, grad_bias = svm_gradient(
            weight, bias, input, target, lambda_reg
        )

        weight = weight - lr * grad_weight
        bias = bias - lr * grad_bias

    return weight, bias
