import numpy as np


def weighted_bce_loss(p: np.ndarray, y: np.ndarray, class_weights: dict) -> float:
    """class_weights: e.g. {0: 1.0, 1: 5.0}"""
    # TODO: Same as bce_loss, but scale each sample's loss term by its
    # class weight before averaging. Clip p away from 0/1 first.
    p = np.clip(p , 1e-12 , 1 - 1e-12)
    weights = np.where( y == 1 , class_weights[1] , class_weights[0])
    losses = weights * (y* np.log(p) + (1-y)* np.log(1-p))

    return float(-np.mean(losses))
