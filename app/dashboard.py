import streamlit as st
import sys
import os

# Make src importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explanations import explain_item
from src.metrics import precision_at_k, diversity_score, explainability_score
from src.data_loader import load_data
from src.baseline_models import popularity_recommender, user_based_cf
from src.adaptive_engine import adaptive_recommend

st.set_page_config(page_title="Adaptive Recommender", layout="wide")

st.title("Adaptive Explainable Recommender System")

# Load data
data = load_data()

# Sidebar controls
st.sidebar.header("Controls")

user_id = st.sidebar.selectbox(
    "Select User",
    sorted(data["userId"].unique())
)

explainability_bias = st.sidebar.slider(
    "Explainability ↔ Personalization",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)

# Generate recommendations
pop_recs = popularity_recommender(data, top_k=10)
cf_recs = user_based_cf(data, user_id=user_id, top_k=10)

final_recs, weights = adaptive_recommend(
    data,
    user_id,
    pop_recs,
    cf_recs,
    explainability_bias=explainability_bias
)

# Ground truth: items user has already interacted with
user_history = data[data["userId"] == user_id]["movieId"].tolist()

# Recommended movie IDs
recommended_titles = final_recs["title"].tolist()
recommended_ids = (
    data[data["title"].isin(recommended_titles)]["movieId"]
    .drop_duplicates()
    .tolist()
)

precision = precision_at_k(
    recommended_ids,
    user_history,
    k=len(recommended_ids)
)

diversity = diversity_score(recommended_titles)
explain_score = explainability_score(weights)


if cf_recs.empty:
    st.warning(
        "User-based recommendations are unavailable due to limited user history. "
        "The system is relying more on popularity-based recommendations."
    )

st.subheader("System Metrics")

col_m1, col_m2, col_m3 = st.columns(3)

col_m1.metric("Precision@K", f"{precision:.2f}")
col_m2.metric("Diversity", f"{diversity:.2f}")
col_m3.metric("Explainability", f"{explain_score:.2f}")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Final Recommendations")
    st.table(final_recs)

with col2:
    st.subheader("Model Contribution")
    st.write(weights)
    
import pandas as pd

weight_df = pd.DataFrame({
    "Model": ["Popularity", "User-based CF"],
    "Weight": [weights["popularity"], weights["user_cf"]]
})

st.subheader("Model Trade-off")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,6))
ax.bar(weight_df["Model"], weight_df["Weight"])
ax.set_ylabel("Weight")
ax.set_ylim(0, 1)

st.pyplot(fig)

    
st.markdown("---")

st.info(
    "This system dynamically balances personalization and explainability. "
    "Each recommendation shows which signal dominated the decision."
)

st.subheader("Why these recommendations?")

cf_available = not cf_recs.empty

for _, row in final_recs.iterrows():
    exp = explain_item(row["title"], weights, cf_available)

    with st.container():
        st.markdown(
            f"""
            ### 🎬 {row['title']}
            **Reason:** {exp['reason']}  
            *{exp['detail']}*
            """
        )

        # Contribution bars
        st.progress(min(weights["user_cf"], 1.0))
        st.caption(f"👥 User Similarity Contribution: {weights['user_cf']:.2f}")

        st.progress(min(weights["popularity"], 1.0))
        st.caption(f"🌍 Popularity Contribution: {weights['popularity']:.2f}")

        st.markdown("---")



