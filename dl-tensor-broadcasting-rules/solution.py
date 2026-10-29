def broadcast_shapes(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...] | None:
    """
    Mirrors NumPy/PyTorch's broadcasting rule for determining the
    result shape of an elementwise operation between two arrays.

    Returns:
        the broadcast result shape, or None if the two shapes are
        incompatible.
    """
    # TODO: Align the two shapes from the RIGHT by padding the shorter
    # one with 1s on the LEFT. Then, for each pair of aligned
    # dimensions: they're compatible if they're equal, or either one
    # is 1 (the size-1 dimension "stretches" to match). The result
    # dimension is max(dim_a, dim_b). If any pair is incompatible
    # (neither equal nor either 1), return None.

    len_diff = len(shape_a) - len(shape_b)
    if len_diff > 0:
        shape_b = (1,) * len_diff + tuple(shape_b)
    elif len_diff < 0:
        shape_a = (1,) * (-len_diff) + tuple(shape_a)

    result = []
    for dim_a, dim_b in zip(shape_a, shape_b):
        if dim_a == dim_b or dim_a == 1 or dim_b == 1:
            result.append(max(dim_a, dim_b))
        else:
            return None

    return tuple(result)
