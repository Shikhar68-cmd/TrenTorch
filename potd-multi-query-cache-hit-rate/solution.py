def cache_hit_rate(keys: list[int]) -> tuple[int, int, float]:
    """
    keys: a sequence of request keys in arrival order.

    A request is a hit iff its key has appeared earlier in the sequence
    (the first occurrence of any key is always a miss).

    Return (hits, misses, rate) where rate = hits / len(keys).
    """
    # TODO: track which keys have been seen so far, in order.
    seen = set()
    miss = 0
    hits = 0
    for key in keys:
        if key not in seen:
            seen.add(key)
            miss = miss + 1
        else :
            hits = hits + 1

    return hits , miss , hits / len(keys)
