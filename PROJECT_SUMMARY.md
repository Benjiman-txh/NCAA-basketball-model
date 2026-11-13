# March Machine Learning Mania 2026 — Kaggle Competition

**Result: Silver Medal — 39th out of 3,462 teams (top ~1.1%)** · Kaggle, Feb–Apr 2026 · $50,000 prize competition, 12,252 entrants

## Overview
Forecast the outcome of every possible matchup in the 2026 NCAA Division I men's and women's basketball tournaments, submitting a win probability for all ~130,000 potential team pairings. Submissions were scored on the Brier score (mean squared error of the probabilities) against actual tournament results.

## My Approach
I built an end-to-end pipeline in Python (Polars for data engineering, SciPy, XGBoost, scikit-learn) that combined a custom-engineered team rating system with a gradient-boosted prediction model.

### 1. Custom "Silver ELO" rating engine
Rather than use a vanilla ELO rating, I implemented a margin-of-victory–aware rating system (inspired by Nate Silver's methodology) with several extensions that I validated against a vanilla baseline through backtesting on 40 seasons of games (1985–2025):

- **Margin-of-victory weighting** — rating updates scale with the score margin via a diminishing-returns transform, so blowouts count more than narrow wins without runaway inflation.
- **Per-team home-court advantage via a Kalman filter** — instead of a single league-wide home bonus, each team's home advantage is estimated as a latent state that updates game-by-game and re-inflates its uncertainty between seasons.
- **Travel-distance penalty** — I geocoded all 509 venue cities (Nominatim + hardcoded fallbacks for international/tricky venues), mapped every team to a home city, and computed per-game great-circle travel distance to penalise long-distance road teams.
- **Conference-based mean reversion** — between seasons, ratings partially revert toward their conference mean to handle roster turnover.
- **Adaptive K-factor** — larger rating updates early in a team's season, decaying as sample size grows; tournament games are up-weighted.

Ratings were snapshotted daily for every team, enabling a point-in-time (leak-free) `join_asof` when building training features.

### 2. Gradient-boosted margin model
I trained an XGBoost model to predict game point margin (not just win/loss), then calibrated margins into probabilities:

- **Features:** Silver ELO ratings, external computer rankings (Pomeroy, Moore, Sagarin) via as-of joins, box-score season averages (OT-adjusted), strength-of-schedule–adjusted win rates, recent-form (last-14-days) win ratio, away-win rate, tournament seeds, and Laplace-smoothed head-to-head records.
- **Robust objective:** a custom Cauchy loss to reduce the influence of blowout outliers on the margin regression.
- **Calibration:** XGBoost leaf indices → TF-IDF → logistic regression to turn predicted margins into well-calibrated win probabilities.
- **Validation:** leave-one-season-out cross-validation on tournament games only, reported with MAE, Brier score, and accuracy — used to compare my Silver ELO against both a vanilla ELO and the reference Nate Silver ELO.

## Key Engineering Decisions
- **Point-in-time correctness:** all features were joined as-of the game date to guarantee no future information leaked into training — a common and costly mistake in sports modelling.
- **Reproducible, cached data pipeline:** geocoding, home-city mapping, and distance computation are cached to CSV so the expensive steps run once.
- **Backtesting harness:** a dedicated rolling backtest (predict-then-update from 1995 onward) let me measure the marginal value of each rating feature against a baseline before adding it to the model.

## Tech Stack
Python · Polars · XGBoost · scikit-learn · SciPy · NumPy · GeoPy · Jupyter

## Outcome
The combination of a hand-engineered, domain-aware rating system with a calibrated gradient-boosted model placed **39th of 3,462 teams (silver medal, top 1.1%)**.
