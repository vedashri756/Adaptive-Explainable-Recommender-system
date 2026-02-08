def explain_item(title, weights, cf_available):
    pop_w = weights["popularity"]
    cf_w = weights["user_cf"]

    if not cf_available:
        return {
            "reason": "Cold-start fallback",
            "detail": "Limited user history → system relied on globally popular items.",
            "dominant": "Popularity"
        }

    if cf_w > 0.6:
        return {
            "reason": "Strong collaborative signal",
            "detail": "Users with similar taste rated this highly.",
            "dominant": "User-based CF"
        }

    if pop_w > 0.6:
        return {
            "reason": "Global popularity",
            "detail": "Consistently high ratings across many users.",
            "dominant": "Popularity"
        }

    return {
        "reason": "Balanced recommendation",
        "detail": "Combination of personal similarity and overall popularity.",
        "dominant": "Hybrid"
    }
