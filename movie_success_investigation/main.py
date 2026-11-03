"""Run the full 'What Makes a Movie Successful?' analysis.

Usage:  python main.py
Needs:  data/movies_raw.csv
Output: outputs/figures/*.png and outputs/tables/*.csv
"""
import pandas as pd

from src import analysis as an
from src import plots
from src.config import BUDGET_LABELS, MIN_MEANINGFUL_BUDGET, TABLE_DIR
from src.data_loader import get_financial_movies, load_movies
from src.investigation import japanese_animation_evidence
from src.modeling import train_and_compare

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save_table(df, name):
    df.to_csv(TABLE_DIR / f"{name}.csv")


def main():
    # ---- 1. Load & prepare -------------------------------------------------
    section("1. DATA OVERVIEW")
    movies = load_movies()
    fin = get_financial_movies(movies)
    print("Date range:", movies["release_year"].min(), "to", movies["release_year"].max())
    print(f"Financially observable movies: {len(fin)} "
          f"({len(fin) / len(movies):.1%} of {len(movies)})")

    yearly = an.yearly_overview(movies)
    plots.line(yearly["release_year"], yearly["movies"], "Number of Movies by Release Year",
               "Release Year", "Number of Movies", "movies_per_year")

    # ---- 2. Budget, revenue, multiple --------------------------------------
    section("2. BUDGET, REVENUE & REVENUE-TO-BUDGET MULTIPLE")
    print(fin[["budget", "revenue", "profit", "revenue_budget_multiple"]].describe())
    print("\nTop 10 by revenue:\n", an.top_by(fin, "revenue"))
    print("\nTop 10 by multiple (budget >= $100k):\n",
          an.top_by(fin, "revenue_budget_multiple", min_budget=MIN_MEANINGFUL_BUDGET))
    print("\nCorrelation with revenue:\n", an.revenue_correlations(fin))

    plots.budget_vs_revenue(fin)
    meaningful = fin[fin["budget"] >= MIN_MEANINGFUL_BUDGET]
    plots.multiple_distributions(fin, meaningful)

    budget_summary = an.budget_group_summary(fin)
    print("\nBy budget quartile:\n", budget_summary)
    save_table(budget_summary, "budget_quartiles")
    print("\nMultiple percentiles:\n", an.multiple_percentiles(fin))

    # ---- 3. Genres ---------------------------------------------------------
    section("3. GENRES")
    genres = an.genre_performance(fin)
    print(genres)
    save_table(genres, "genre_performance")
    combos = an.genre_combination_performance(fin)
    save_table(combos, "genre_combinations")
    top_combos = combos.head(10).sort_values("median_revenue")
    plots.bar(top_combos.index, top_combos["median_revenue"],
              "Top Genre Combinations by Median Revenue", "Median Revenue",
              "Genre Combination", "genre_combinations", money=True, horizontal=True)
    genre_budget = an.genre_by_budget_group(fin)
    save_table(genre_budget, "genre_by_budget_group")
    plots.genre_multiple_by_budget(genre_budget)

    # ---- 4. Audience rating / popularity -----------------------------------
    section("4. AUDIENCE RATING & POPULARITY")
    ratings = an.quartile_summary(
        fin, "vote_average", ["Low Rated", "Medium-Low", "Medium-High", "High Rated"])
    print(ratings)
    save_table(ratings, "rating_quartiles")
    plots.bar(ratings.index, ratings["median_revenue"], "Median Revenue by Audience Rating",
              "Rating Group", "Median Revenue (USD)", "revenue_by_rating", money=True)
    plots.bar(ratings.index, ratings["median_multiple"], "Revenue-to-Budget Multiple by Rating",
              "Rating Group", "Median Multiple", "multiple_by_rating")

    # ---- 5. Runtime --------------------------------------------------------
    section("5. RUNTIME")
    runtime = an.quartile_summary(
        fin, "runtime", ["Shortest", "Short-Medium", "Medium-Long", "Longest"])
    print(runtime)
    save_table(runtime, "runtime_quartiles")
    plots.runtime_vs_revenue(fin)
    plots.bar(runtime.index, runtime["median_revenue"], "Median Revenue by Runtime Group",
              "Runtime Group", "Median Revenue (USD)", "revenue_by_runtime", money=True)
    plots.bar(runtime.index, runtime["median_multiple"], "Multiple by Runtime Group",
              "Runtime Group", "Median Multiple", "multiple_by_runtime")

    # ---- 6. Release timing -------------------------------------------------
    section("6. RELEASE TIMING")
    years = an.year_analysis(fin)
    months = an.month_analysis(fin)
    print(years.tail(20))
    print(months)
    save_table(years, "by_year")
    save_table(months, "by_month")
    plots.line(years["release_year"], years["median_revenue"], "Median Revenue by Release Year",
               "Release Year", "Median Revenue (USD)", "revenue_by_year", money=True)
    plots.line(years["release_year"], years["median_multiple"],
               "Median Multiple by Release Year", "Release Year", "Median Multiple",
               "multiple_by_year")
    plots.bar(months["release_month_name"], months["median_revenue"],
              "Median Revenue by Release Month", "Month", "Median Revenue (USD)",
              "revenue_by_month", money=True)
    plots.bar(months["release_month_name"], months["median_multiple"],
              "Multiple by Release Month", "Month", "Median Multiple", "multiple_by_month")

    # ---- 7. Companies & languages -----------------------------------------
    section("7. PRODUCTION COMPANIES & LANGUAGES")
    companies = an.company_performance(fin)
    languages = an.language_performance(fin)
    print(companies.head(10))
    print(languages)
    save_table(companies, "company_performance")
    save_table(languages, "language_performance")
    top_c = companies.nlargest(10, "median_revenue").sort_values("median_revenue")
    plots.bar(top_c.index, top_c["median_revenue"], "Top Companies by Median Revenue",
              "Median Revenue (USD)", "Company", "companies_by_revenue",
              money=True, horizontal=True)
    top_l = languages.nlargest(10, "median_revenue").sort_values("median_revenue")
    plots.bar(top_l.index, top_l["median_revenue"], "Top Languages by Median Revenue",
              "Median Revenue (USD)", "Language", "languages_by_revenue",
              money=True, horizontal=True)
    if len(languages) > 1:
        corr = languages["median_budget"].corr(languages["median_revenue"])
        print(f"\nBudget vs revenue correlation across languages: {corr:.4f}")
        plots.language_budget_revenue(languages)

    # ---- 8. Correlation synthesis -----------------------------------------
    section("8. CORRELATION SYNTHESIS")
    corr = an.synthesis_correlation(fin)
    print(corr)
    plots.correlation_heatmap(corr)

    # ---- 9. Machine learning ----------------------------------------------
    section("9. MACHINE LEARNING (pre-release revenue prediction)")
    comparison, _, importance = train_and_compare(fin)
    print(comparison.round(4))
    print("\nTop features:\n", importance.head(20).round(4))
    save_table(comparison, "model_comparison")
    save_table(importance, "feature_importance")
    plots.feature_importance(importance)

    # ---- 10. Curiosity investigation --------------------------------------
    section("10. JAPANESE-LANGUAGE ANIMATION vs NON-ANIMATION")
    evidence = japanese_animation_evidence(fin)
    print(evidence.round(2))
    save_table(evidence, "japanese_animation_evidence")
    plots.japanese_multiples(evidence)

    print("\nDone. Figures -> outputs/figures | Tables -> outputs/tables")


if __name__ == "__main__":
    main()
