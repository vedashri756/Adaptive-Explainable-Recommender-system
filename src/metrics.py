import numpy as np

def precision_at_k(recommended_ids, relevant_ids, k):
    if k == 0:
        return 0.0
    recommended_ids = recommended_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for item in recommended_ids if item in relevant_set)
    return hits / k


def diversity_score(recommended_items):
    """
    Proxy diversity: proportion of unique items
    """
    if len(recommended_items) == 0:
        return 0.0
    return len(set(recommended_items)) / len(recommended_items)


def explainability_score(weights):
    """
    Higher when system relies more on popularity-based model
    """
    return weights.get("popularity", 0.0)
