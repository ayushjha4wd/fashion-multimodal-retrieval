def recall_at_k(results, k):
    return sum(results[:k]) / len(results)
