"""Loading and cleaning the raw movie dataset."""
import pandas as pd

from .config import DATA_PATH


def load_movies(path=DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and add date-derived columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Copy movies_raw.csv into the 'data' folder and run again."
        )
    movies = pd.read_csv(path)
    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    movies["release_year"] = movies["release_date"].dt.year
    return movies


def get_financial_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Keep movies with a recorded positive budget AND revenue; add metrics.

    revenue_budget_multiple = revenue / budget (not true ROI: other costs
    such as marketing and distribution are unavailable).
    """
    fin = movies[(movies["budget"] > 0) & (movies["revenue"] > 0)].copy()
    fin["profit"] = fin["revenue"] - fin["budget"]
    fin["revenue_budget_multiple"] = fin["revenue"] / fin["budget"]
    fin["release_month"] = fin["release_date"].dt.month
    fin["release_month_name"] = fin["release_date"].dt.month_name()
    return fin


def explode_column(df: pd.DataFrame, column: str, new_name: str, sep: str = "|") -> pd.DataFrame:
    """Split a '|'-separated column into one row per value."""
    out = df[df[column].notna()].copy()
    out[new_name] = out[column].str.split(sep)
    out = out.explode(new_name)
    out[new_name] = out[new_name].str.strip()
    return out
