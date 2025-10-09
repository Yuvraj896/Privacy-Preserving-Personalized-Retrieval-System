import numpy as np

def precision_at_k(retrieved, relevant , k):
    retrieved_k = retrieved[:k]
    num_relevant = sum([1 for doc in retrieved_k if doc in relevant])
    return num_relevant / k


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    num_relevant = sum([1 for doc in retrieved_k if doc in relevant])
    total_relevant = len(relevant)
    return num_relevant / total_relevant if total_relevant > 0 else 0.0

def dcg_at_k(retrieved, relevant, k):
    """
    Discounted Cumulative Gain
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = 1 if doc_id in relevant else 0
        dcg += (2**rel - 1) / (np.log2(i + 2))  # log2(rank+1)
    return dcg

def ndcg_at_k(retrieved, relevant, k):
    """
    Normalized DCG
    """
    ideal_retrieved = relevant[:k]  # ideal ranking
    ideal_dcg = dcg_at_k(ideal_retrieved, relevant, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal_dcg