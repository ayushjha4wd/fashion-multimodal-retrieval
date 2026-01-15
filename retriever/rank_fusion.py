def fuse(scores, weights):
    total = 0.0
    for k in scores:
        total += weights[k] * scores[k]
    return total
