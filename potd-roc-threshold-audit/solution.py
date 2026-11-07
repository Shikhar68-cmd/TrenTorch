
def roc_threshold_audit(
    probs: list[float], labels: list[int], tau: float
) -> tuple[int, int, int, int, float, float]:

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for prob, label in zip(probs, labels):
        if prob >= tau:
            if label == 1:
                tp += 1
            else:
                fp += 1

        else:
            if label == 1:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return (tp, fp, tn, fn, precision, recall)
