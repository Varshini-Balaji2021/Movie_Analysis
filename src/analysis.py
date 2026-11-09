"""Pure analysis functions: each returns a DataFrame/Series, no plotting."""
import pandas as pd

from .config import BUDGET_LABELS, MIN_GROUP_SIZE
from .data_loader import explode_column

SUMMARY_AGGS = dict(
    movies=("movie_id", "count"),
    median_budget=("budget", "median"),
    median_revenue=("revenue", "median"),
    median_multiple=("revenue_budget_multiple", "median"),
    median_rating=("vote_average", "median"),
    median_popularity=("popularity", "median"),
)


def yearly_overview(movies: pd.DataFrame) -> pd.DataFrame:
    return (
        movies.groupby("release_year")
        .agg(
            movies=("movie_id", "count"),
            avg_rating=("vote_average", "mean"),
            avg_popularity=("popularity", "mean"),
            avg_budget=("budget", "mean"),
            avg_revenue=("revenue", "mean"),
        )
        .reset_index()
    )


def top_by(fin: pd.DataFrame, column: str, n: int = 10, min_budget: float = 0) -> pd.DataFrame:
    cols = ["title", "release_year", "budget", "revenue", "revenue_budget_multiple"]
    return fin[fin["budget"] >= min_budget].nlargest(n, column)[cols]


def revenue_correlations(fin: pd.DataFrame) -> pd.Series:
    cols = ["budget", "revenue", "runtime", "vote_average", "vote_count", "popularity"]
    return fin[cols].corr()["revenue"].sort_values(ascending=False)


def synthesis_correlation(fin: pd.DataFrame) -> pd.DataFrame:
    cols = ["budget", "revenue", "runtime", "vote_average", "vote_count",
            "popularity", "revenue_budget_multiple"]
    return fin[cols].corr()


def quartile_summary(fin: pd.DataFrame, column: str, labels) -> pd.DataFrame:
    """Median metrics for each quartile of `column`."""
    groups = pd.qcut(fin[column], q=4, labels=labels, duplicates="drop")
    return fin.groupby(groups, observed=True).agg(
        movies=("movie_id", "count"),
        median_value=(column, "median"),
        median_budget=("budget", "median"),
        median_revenue=("revenue", "median"),
        median_multiple=("revenue_budget_multiple", "median"),
        median_popularity=("popularity", "median"),
    )


def group_performance(df: pd.DataFrame, by, min_count: int = 1, sort_by="median_revenue") -> pd.DataFrame:
    perf = df.groupby(by).agg(**SUMMARY_AGGS)
    return perf[perf["movies"] >= min_count].sort_values(sort_by, ascending=False)


def genre_performance(fin: pd.DataFrame) -> pd.DataFrame:
    exploded = explode_column(fin, "genres", "genre")
    return group_performance(exploded, "genre")


def company_performance(fin: pd.DataFrame, min_count: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    exploded = explode_column(fin, "production_companies", "company")
    return group_performance(exploded, "company", min_count=min_count)


def language_performance(fin: pd.DataFrame, min_count: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    return group_performance(fin, "original_language", min_count=min_count)


def genre_combination_performance(fin: pd.DataFrame, min_count: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    counts = fin["genres"].value_counts()
    keep = counts[counts >= min_count].index
    return group_performance(fin[fin["genres"].isin(keep)], "genres")


def year_analysis(fin: pd.DataFrame) -> pd.DataFrame:
    return group_performance(fin, "release_year", sort_by="release_year").sort_index().reset_index()


def month_analysis(fin: pd.DataFrame) -> pd.DataFrame:
    perf = fin.groupby(["release_month", "release_month_name"]).agg(**SUMMARY_AGGS)
    return perf.reset_index().sort_values("release_month")


def budget_group_summary(fin: pd.DataFrame) -> pd.DataFrame:
    return quartile_summary(fin, "budget", BUDGET_LABELS)


def genre_by_budget_group(fin: pd.DataFrame, min_count: int = 15) -> pd.DataFrame:
    """Median revenue and multiple per (budget quartile, genre)."""
    data = fin.copy()
    data["budget_group"] = pd.qcut(data["budget"], q=4, labels=BUDGET_LABELS, duplicates="drop")
    data["genre"] = data["genres"].fillna("Unknown").str.split("|")
    data = data.explode("genre")
    summary = data.groupby(["budget_group", "genre"], observed=True).agg(
        movie_count=("title", "count"),
        median_revenue=("revenue", "median"),
        median_multiple=("revenue_budget_multiple", "median"),
    ).reset_index()
    summary = summary[summary["movie_count"] >= min_count].copy()
    summary["median_revenue_millions"] = summary["median_revenue"] / 1e6
    return summary


def multiple_percentiles(fin: pd.DataFrame) -> pd.Series:
    s = fin["revenue_budget_multiple"]
    return pd.Series({
        "median": s.median(),
        "p75": s.quantile(0.75),
        "p90": s.quantile(0.90),
        "p95": s.quantile(0.95),
        "p99": s.quantile(0.99),
    })
