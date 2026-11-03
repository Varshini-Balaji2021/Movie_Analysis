# What Makes a Movie Successful?

A data-driven exploration of movie economics (budget, revenue, genre, timing, language,
production companies) plus a pre-release revenue-prediction model.

## Setup (VS Code)

1. Unzip and open the folder in VS Code (**File → Open Folder**).
2. Put your dataset at `data/movies_raw.csv`.
3. Open a terminal (**Terminal → New Terminal**) and run:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Or press **F5** and choose "Run full analysis".

## Outputs
- `outputs/figures/` – all charts as PNG
- `outputs/tables/` – summary tables as CSV
- Printed results appear in the terminal.

## Structure
```
main.py              runs everything
src/config.py        paths and constants
src/data_loader.py   loading, cleaning, financial subset
src/analysis.py      summary tables (no plotting)
src/plots.py         all charts
src/modeling.py      Linear Regression / Random Forest / Gradient Boosting
src/investigation.py Japanese Animation vs Non-Animation study
```

## Notes
- `revenue_budget_multiple` = revenue / budget. It is **not** true ROI (marketing and
  distribution costs are unavailable).
- Only movies with positive budget and revenue are used for financial analysis.
- Findings are associations, not causal claims.
