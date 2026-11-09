"""Pre-release revenue prediction: Linear Regression vs Random Forest vs Gradient Boosting."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE

FEATURES = ["budget", "runtime", "release_year", "genres", "original_language"]


def build_features(fin: pd.DataFrame):
    """One-hot encode genre combination and language; revenue is the target."""
    data = fin[FEATURES + ["revenue"]].copy()
    for col in ("genres", "original_language"):
        data[col] = data[col].fillna("Unknown").replace("", "Unknown")
    X = pd.get_dummies(data[FEATURES], columns=["genres", "original_language"], dtype=int)
    return X, data["revenue"]


def get_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            min_samples_leaf=5, random_state=RANDOM_STATE),
    }


def train_and_compare(fin: pd.DataFrame):
    """Returns (comparison_table, fitted_models, feature_importance_df)."""
    X, y = build_features(fin)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": r2_score(y_test, pred),
        })
        fitted[name] = model

    comp = pd.DataFrame(rows)
    comp["MAE Rank"] = comp["MAE"].rank(ascending=True)
    comp["RMSE Rank"] = comp["RMSE"].rank(ascending=True)
    comp["R2 Rank"] = comp["R2"].rank(ascending=False)
    comp["Average Rank"] = comp[["MAE Rank", "RMSE Rank", "R2 Rank"]].mean(axis=1)
    comp = comp.sort_values("Average Rank").reset_index(drop=True)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": fitted["Gradient Boosting"].feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return comp, fitted, importance
