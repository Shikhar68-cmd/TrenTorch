import numpy as np


def variance(targets: np.ndarray) -> float:
    """Population variance of targets. Empty input has variance 0.0 by convention."""
    # TODO: np.var handles this in one call -- just guard the empty case.

    if targets.size == 0:
        return 0.0
    return float(np.var(targets))


def variance_reduction(
    parent_targets: np.ndarray,
    left_targets: np.ndarray,
    right_targets: np.ndarray,
) -> float:
    """Same shape as information_gain, with variance in place of Gini impurity."""
    # TODO: parent variance minus the size-weighted average of the two
    # children's variances.
    n = parent_targets.size

    weighted_child_var = (left_targets.size/n)*variance(left_targets) + (right_targets.size / n)*variance(right_targets)

    return variance(parent_targets) - weighted_child_var


def find_best_regression_split(
    input: np.ndarray, targets: np.ndarray
) -> tuple[int, float, float] | None:

    n_features = input.shape[1]

    best_reduction = 0.0
    best_feature = None
    best_threshold = None

    for feature in range(n_features):

        values = np.unique(input[:, feature])

        if values.size < 2:
            continue

        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:

            left_mask = input[:, feature] <= threshold

            reduction = variance_reduction(
                targets,
                targets[left_mask],
                targets[~left_mask]
            )

            if reduction > best_reduction:
                best_reduction = reduction
                best_feature = feature
                best_threshold = threshold

    if best_feature is None:
        return None

    return best_feature, float(best_threshold), float(best_reduction)


def build_regression_tree(input: np.ndarray, targets: np.ndarray, max_depth: int) -> dict:
    """
    Same recursive structure as build_tree, for regression:
      leaf node:  {"leaf": True, "prediction": <mean of this node's targets>}
      split node: {"leaf": False, "feature": i, "threshold": t, "left": ..., "right": ...}

    Stop at a leaf when max_depth reaches 0, fewer than 2 samples
    remain, every target in this node is already identical (variance
    0), or find_best_regression_split finds nothing.
    """
    # TODO: Base case -> leaf with prediction = mean(targets).
    # Otherwise: find_best_regression_split(), then recurse on each
    # side with max_depth - 1.
    if max_depth == 0 or targets.size < 2 or variance(targets) == 0.0:
        return {"leaf" : True , "prediction" : float(np.mean(targets))}

    split = find_best_regression_split(input , targets)

    if split is None:
        return {"leaf": True , "prediction": float(np.mean(targets))}

    feature , threshold , _ = split
    left_mask = input[: , feature] <= threshold

    return {
        "leaf" : False,
        "feature" : feature,
        "threshold": threshold,
        "left" : build_regression_tree(input[left_mask] , targets[left_mask] , max_depth - 1),
        "right" : build_regression_tree(input[~left_mask] , targets[~left_mask] , max_depth - 1)
    }
    

def predict_regression_tree(tree: dict, input: np.ndarray) -> np.ndarray:
    predictions = np.empty(input.shape[0], dtype=float)
    for i in range(input.shape[0]):
        node = tree
        while not node["leaf"]:
            if input[i, node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        predictions[i] = node["prediction"]
    return predictions
