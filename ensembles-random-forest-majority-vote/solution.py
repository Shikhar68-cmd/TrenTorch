import numpy as np

from _load import load_solution

predict_tree = load_solution("decision-trees-best-split-minimal-tree").predict_tree

def random_forest_predict(trees: list[dict], input: np.ndarray) -> np.ndarray:
    """
    trees: a list of already-built trees (03-best-split-minimal-tree's
    build_tree output), each trained independently.
    input: shape (n_samples, n_features)

    Returns:
        shape (n_samples,), the majority-vote class across every tree's
        prediction for each sample. Ties broken by the lower class label.
    """
    # TODO: Run predict_tree() for every tree in trees, collect the
    # predictions into shape (n_trees, n_samples), then for each sample
    # (each column) take the class with the most votes across trees.
    if len(trees) == 0:
        raise ValueError("Random Forest must contain at least one tree")
    
    votes = np.array([predict_tree(tree , input) for tree in trees])

    predictions = np.empty(input.shape[0] , dtype = int)

    for i in range(input.shape[0]):
        values , counts = np.unique(votes[: , i] , return_counts = True)
        predictions[i] = values[np.argmax(counts)]

    return predictions
