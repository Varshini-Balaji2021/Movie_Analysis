"""All charts. Every function saves a PNG to outputs/figures and closes the figure."""
import matplotlib

matplotlib.use("Agg")  # save to files; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from .config import FIG_DIR, MAX_MULTIPLE_CLEAN

MILLIONS = FuncFormatter(lambda v, _: f"${v / 1e6:.0f}M")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


def line(x, y, title, xlabel, ylabel, name, money=False):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, marker="o")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if money:
        ax.yaxis.set_major_formatter(MILLIONS)
    ax.tick_params(axis="x", rotation=45)
    _save(fig, name)


def bar(labels, values, title, xlabel, ylabel, name, money=False, horizontal=False):
    fig, ax = plt.subplots(figsize=(11, 6) if horizontal else (10, 5))
    labels = [str(l) for l in labels]
    if horizontal:
        ax.barh(labels, values)
        if money:
            ax.xaxis.set_major_formatter(MILLIONS)
    else:
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=30)
        if money:
            ax.yaxis.set_major_formatter(MILLIONS)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    _save(fig, name)


def budget_vs_revenue(fin):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(fin["budget"], fin["revenue"], alpha=0.35)
    ax.set(title="Production Budget vs Revenue", xlabel="Budget", ylabel="Revenue")
    _save(fig, "budget_vs_revenue")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(fin["budget"], fin["revenue"], alpha=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title="Budget vs Revenue (log scale)", xlabel="Budget (log)", ylabel="Revenue (log)")
    _save(fig, "budget_vs_revenue_log")


def multiple_distributions(fin, meaningful):
    clean = fin[fin["revenue_budget_multiple"] <= MAX_MULTIPLE_CLEAN]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(clean["revenue_budget_multiple"], bins=50)
    ax.set(title=f"Revenue-to-Budget Multiple (capped at {MAX_MULTIPLE_CLEAN}x)",
           xlabel="Revenue / Budget", ylabel="Number of Movies")
    _save(fig, "multiple_histogram")

    values = meaningful["revenue_budget_multiple"].dropna()
    values = values[values > 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.log10(values), bins=40, edgecolor="black")
    ax.set(title="Revenue-to-Budget Multiples (log10)",
           xlabel="Log10 of Revenue / Budget", ylabel="Number of Movies")
    _save(fig, "multiple_histogram_log")


def runtime_vs_revenue(fin):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(fin["runtime"], fin["revenue"], alpha=0.3)
    ax.set(title="Runtime vs Revenue", xlabel="Runtime (minutes)", ylabel="Revenue")
    _save(fig, "runtime_vs_revenue")


def correlation_heatmap(corr):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, center=0, ax=ax)
    ax.set_title("Correlation Between Movie Performance Variables")
    _save(fig, "correlation_heatmap")


def language_budget_revenue(lang):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(lang["median_budget"], lang["median_revenue"], s=100, alpha=0.8)
    for code, row in lang.iterrows():
        ax.annotate(code, (row["median_budget"], row["median_revenue"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set(title="Median Budget vs Median Revenue by Language",
           xlabel="Median Budget", ylabel="Median Revenue")
    ax.grid(True, linestyle="--", alpha=0.5)
    _save(fig, "language_budget_vs_revenue")


def genre_multiple_by_budget(genre_budget):
    pivot = genre_budget[genre_budget["budget_group"].isin(["Lower budget", "Higher budget"])].pivot(
        index="genre", columns="budget_group", values="median_multiple"
    ).dropna()
    if pivot.empty:
        return
    pivot["difference"] = pivot["Higher budget"] - pivot["Lower budget"]
    top = pivot.reindex(pivot["difference"].abs().nlargest(12).index).sort_values("difference")
    ax = top[["Lower budget", "Higher budget"]].plot(kind="barh", figsize=(10, 7))
    ax.set(title="How Genre Multiples Differ by Budget Group",
           xlabel="Median Revenue-to-Budget Multiple", ylabel="Genre")
    _save(ax.figure, "genre_multiple_by_budget")


def feature_importance(fi, n=15):
    top = fi.head(n).sort_values("Importance")
    bar(top["Feature"], top["Importance"], f"Top {n} Features for Revenue Prediction",
        "Feature Importance", "Feature", "feature_importance", horizontal=True)


def japanese_multiples(evidence):
    data = evidence.set_index("Movie Group")
    groups = [g for g in ["Animation", "Non-Animation"] if g in data.index]
    if not groups:
        return
    med = data.loc[groups, "Median Multiple"].values
    p25 = data.loc[groups, "25th Percentile"].values
    p75 = data.loc[groups, "75th Percentile"].values
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(med, groups, xerr=np.array([med - p25, p75 - med]), fmt="o", capsize=6)
    for g, m in zip(groups, med):
        ax.annotate(f"{m:.2f}x median", (m, g), xytext=(8, 10), textcoords="offset points")
    ax.set(title="Japanese-Language Movies: Revenue-to-Budget Multiples",
           xlabel="Revenue-to-Budget Multiple", ylabel="Movie Group")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, "japanese_multiples")
