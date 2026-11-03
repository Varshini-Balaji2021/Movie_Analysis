"""Curiosity investigation: Japanese-language Animation vs Non-Animation multiples."""
import pandas as pd


def _summarize(group: pd.DataFrame) -> dict:
    multiples = group["revenue_budget_multiple"].dropna()
    trimmed = multiples.sort_values(ascending=False).iloc[5:]
    return {
        "Movies": len(group),
        "Median Budget ($M)": group["budget"].median() / 1e6,
        "Median Revenue ($M)": group["revenue"].median() / 1e6,
        "25th Percentile": multiples.quantile(0.25),
        "Median Multiple": multiples.median(),
        "75th Percentile": multiples.quantile(0.75),
        "Median Without Top 5": trimmed.median(),
    }


def japanese_animation_evidence(fin: pd.DataFrame) -> pd.DataFrame:
    jp = fin[fin["original_language"] == "ja"].copy()
    jp["Movie Group"] = (
        jp["genres"].fillna("").str.contains("Animation", regex=False)
        .map({True: "Animation", False: "Non-Animation"})
    )
    rows = [{"Movie Group": name, **_summarize(g)} for name, g in jp.groupby("Movie Group")]
    return pd.DataFrame(rows)
