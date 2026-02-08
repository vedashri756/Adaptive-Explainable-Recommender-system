def compute_user_activity(data, user_id):
    return data[data["userId"] == user_id].shape[0]


def compute_weights(data, user_id, explainability_bias=0.5):
    """
    explainability_bias ∈ [0,1]
    0   → favor accuracy/personalization
    1   → favor explainability/stability
    """

    activity = compute_user_activity(data, user_id)

    # Cold-start or low activity users
    if activity < 10:
        w_pop = 0.8
        w_cf = 0.2

    # Active users
    else:
        w_pop = 0.3
        w_cf = 0.7

    # Shift weights based on explainability preference
    w_pop = w_pop + explainability_bias * 0.2
    w_cf = w_cf - explainability_bias * 0.2

    # Normalize
    total = w_pop + w_cf
    w_pop /= total
    w_cf /= total

    return w_pop, w_cf


import pandas as pd

def adaptive_recommend(
    data,
    user_id,
    pop_recs,
    cf_recs,
    top_k=5,
    explainability_bias=0.5
):
    w_pop, w_cf = compute_weights(data, user_id, explainability_bias)

    pop_recs = pop_recs.copy()
    pop_recs["score"] = pop_recs["mean_rating"]
    pop_recs = pop_recs[["title", "score"]]

    if cf_recs.empty:
        cf_recs = pd.DataFrame(columns=["title", "score"])
    else:
        cf_recs = cf_recs[["title", "score"]]


    combined = pd.concat([
        pop_recs.assign(weight=w_pop),
        cf_recs.assign(weight=w_cf)
    ])

    combined["final_score"] = combined["score"] * combined["weight"]

    final = (
        combined.groupby("title")["final_score"]
        .sum()
        .sort_values(ascending=False)
        .head(top_k)
        .reset_index()
    )

    return final, {"popularity": w_pop, "user_cf": w_cf}
