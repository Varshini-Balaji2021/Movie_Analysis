"""Streamlit dashboard for the movie success investigation.

Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from src import analysis as an
from src.config import MIN_MEANINGFUL_BUDGET
from src.data_loader import explode_column, get_financial_movies, load_movies
from src.investigation import japanese_animation_evidence
from src.modeling import train_and_compare

st.set_page_config(page_title="Movie Success Investigation", page_icon="🎬", layout="wide")


@st.cache_data
def load_data():
    movies = load_movies()
    return movies, get_financial_movies(movies)


@st.cache_data
def run_models(fin: pd.DataFrame):
    comparison, _, importance = train_and_compare(fin)
    return comparison, importance


try:
    movies, fin_all = load_data()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

# ---------------- Sidebar filters ----------------
st.sidebar.header("Filters")
y_min, y_max = int(fin_all["release_year"].min()), int(fin_all["release_year"].max())
year_range = st.sidebar.slider("Release years", y_min, y_max, (y_min, y_max))

all_genres = sorted(
    explode_column(fin_all, "genres", "genre")["genre"].dropna().unique()
)
chosen_genres = st.sidebar.multiselect("Genres (any of)", all_genres)

fin = fin_all[fin_all["release_year"].between(*year_range)]
if chosen_genres:
    pattern = "|".join(chosen_genres)
    fin = fin[fin["genres"].fillna("").str.contains(pattern, regex=True)]

st.title("🎬 What Makes a Movie Successful?")
st.caption(
    "Revenue-to-budget multiple = revenue ÷ budget. It is not true ROI "
    "(marketing and distribution costs are unavailable). Results show associations, not causes."
)

if fin.empty:
    st.warning("No movies match these filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Movies", f"{len(fin):,}")
c2.metric("Median budget", f"${fin['budget'].median() / 1e6:.1f}M")
c3.metric("Median revenue", f"${fin['revenue'].median() / 1e6:.1f}M")
c4.metric("Median multiple", f"{fin['revenue_budget_multiple'].median():.2f}x")

tabs = st.tabs([
    "Budget & Revenue", "Genres", "Release timing",
    "Language & Companies", "Audience & Runtime", "Prediction model", "Animation study",
])

# ---------------- Budget & revenue ----------------
with tabs[0]:
    st.subheader("Does budget predict revenue?")
    corr = an.revenue_correlations(fin).drop("revenue")
    st.write("Correlation with revenue (filtered movies):")
    st.bar_chart(corr)

    plot_df = fin[["budget", "revenue"]].assign(
        log_budget=np.log10(fin["budget"]), log_revenue=np.log10(fin["revenue"])
    )
    st.write("Budget vs revenue (log10 scale)")
    st.scatter_chart(plot_df, x="log_budget", y="log_revenue")

    st.write("Does a higher budget mean higher efficiency?")
    st.dataframe(an.budget_group_summary(fin).round(2))

    left, right = st.columns(2)
    left.write("Top 10 by revenue")
    left.dataframe(an.top_by(fin, "revenue"), hide_index=True)
    right.write(f"Top 10 by multiple (budget ≥ ${MIN_MEANINGFUL_BUDGET:,})")
    right.dataframe(
        an.top_by(fin, "revenue_budget_multiple", min_budget=MIN_MEANINGFUL_BUDGET),
        hide_index=True,
    )

# ---------------- Genres ----------------
with tabs[1]:
    st.subheader("Which genres perform best financially?")
    genres = an.genre_performance(fin)
    metric = st.radio("Rank genres by", ["median_revenue", "median_multiple"], horizontal=True)
    st.bar_chart(genres[metric].sort_values(ascending=False))
    st.dataframe(genres.round(2))

    st.write("Top genre combinations (at least 20 movies)")
    combos = an.genre_combination_performance(fin)
    if combos.empty:
        st.info("Not enough movies per combination for the current filters.")
    else:
        st.bar_chart(combos["median_revenue"].head(10))

# ---------------- Release timing ----------------
with tabs[2]:
    st.subheader("Does release timing matter?")
    months = an.month_analysis(fin).set_index("release_month_name")
    month_metric = st.radio("Metric", ["median_revenue", "median_multiple", "movies"],
                            horizontal=True, key="month_metric")
    st.bar_chart(months[month_metric])

    years = an.year_analysis(fin).set_index("release_year")
    st.write("Trend by release year")
    year_metric = st.radio("Metric", ["median_revenue", "median_multiple", "movies"],
                           horizontal=True, key="year_metric")
    st.line_chart(years[year_metric])

# ---------------- Language & companies ----------------
with tabs[3]:
    min_n = st.slider("Minimum movies per group", 5, 50, 20)
    left, right = st.columns(2)
    langs = an.language_performance(fin, min_count=min_n)
    left.write("Original language")
    if langs.empty:
        left.info("No languages meet the minimum.")
    else:
        left.bar_chart(langs["median_revenue"].head(10))
        left.dataframe(langs.round(2))
    comps = an.company_performance(fin, min_count=min_n)
    right.write("Production companies")
    if comps.empty:
        right.info("No companies meet the minimum.")
    else:
        right.bar_chart(comps["median_revenue"].head(10))
        right.dataframe(comps.round(2))

# ---------------- Audience & runtime ----------------
with tabs[4]:
    left, right = st.columns(2)
    left.subheader("Audience rating")
    ratings = an.quartile_summary(
        fin, "vote_average", ["Low Rated", "Medium-Low", "Medium-High", "High Rated"])
    left.bar_chart(ratings["median_revenue"])
    left.dataframe(ratings.round(2))

    right.subheader("Runtime")
    runtime = an.quartile_summary(
        fin, "runtime", ["Shortest", "Short-Medium", "Medium-Long", "Longest"])
    right.bar_chart(runtime["median_revenue"])
    right.dataframe(runtime.round(2))

    st.write("Correlation matrix")
    st.dataframe(an.synthesis_correlation(fin).round(2).style.background_gradient(
        cmap="coolwarm", vmin=-1, vmax=1))

# ---------------- Model ----------------
with tabs[5]:
    st.subheader("Pre-release revenue prediction")
    st.write(
        "Trained on all movies with recorded budget and revenue (filters above are ignored). "
        "Uses budget, runtime, release year, genre combination and language."
    )
    if st.button("Train models"):
        with st.spinner("Training Linear Regression, Random Forest and Gradient Boosting..."):
            comparison, importance = run_models(fin_all)
        st.dataframe(comparison.round(4), hide_index=True)
        st.write("Top features (Gradient Boosting)")
        st.bar_chart(importance.head(15).set_index("Feature")["Importance"])
        st.caption("Single 80/20 train-test split. Errors are large, so treat this as an "
                   "estimation tool, not a reliable forecast for individual movies.")

# ---------------- Animation study ----------------
with tabs[6]:
    st.subheader("Japanese-language Animation vs Non-Animation")
    evidence = japanese_animation_evidence(fin)
    if evidence.empty:
        st.info("No Japanese-language movies in the current filters.")
    else:
        st.dataframe(evidence.round(2), hide_index=True)
        st.caption("Small samples with overlapping distributions: this does not prove "
                   "Animation is inherently more efficient.")
