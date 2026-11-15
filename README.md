# Data Science Project: Part A (California House Prices) and Part B (Learning Game)

- **Part A:** machine-learning project, described below. Run `python src/run_project.py`.
- **Part B:** the game *Data Detective Casebook*. Open `game/index.html` in a browser. Details in [`game/README.md`](game/README.md); playtest evidence in [`game/PLAYTEST.md`](game/PLAYTEST.md).

# Part A: California House Price Prediction

**Team:** T. Kishor (ec24b1130), S. Haran Karthikeyan (ec24b1126) | **Course:** IDSE

## 1. Problem and data

**Question.** Given census information about a neighbourhood (income, housing stock,
population, location), how accurately can we predict its median house value, and which
factors drive it most?

**Measurable goals**
1. Beat a naive "predict the median" baseline by at least 50% in RMSE.
2. Reach R² ≥ 0.80 on an untouched test set.
3. Explain *where* and *why* the model fails, not just report a score.

**Dataset.** California Housing (1990 US Census, Pace & Barry 1997), loaded with
`sklearn.datasets.fetch_california_housing`. 20,640 block groups, 8 numeric features.
Chosen because (a) it is a real, mid-sized dataset large enough for cross-validation and
tuning, (b) it has genuine data issues (capped target, extreme outliers, spatial
dependence), and (c) it is downloaded automatically, so the project runs with no manual steps.
Target `MedHouseVal` is in units of $100,000.

## 2. How to run

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_project.py                              # ~1-3 min
```
Outputs: `reports/figures/*.png`, `reports/results.md`, `reports/results.json`,
`reports/worst_predictions.csv`. All randomness is controlled by `SEED = 42`.

## 3. Data preparation and EDA

| Check | Finding (see `reports/results.md` → Data audit) | Decision and reasoning |
|---|---|---|
| Missing values | 0 | Median imputer kept **inside the pipeline** so it is learned on training folds only (safe if new data has gaps). |
| Duplicates | 0 | `drop_duplicates()` is still applied; duplicates would inflate scores if they land in both train and test. |
| Data types | all numeric | No encoding needed. |
| Outliers | `AveRooms` (max 141.9), `AveBedrms` (max 34.1), `Population` (max 35,682) and `AveOccup` (max 1,243) have extreme values | Winsorised at 0.5%/99.5%. Trees are robust to them but Ridge is not; these are almost certainly data-entry or institutional-housing artefacts. |
| Target cap | 992 rows (4.81%) sit exactly at $500k | Kept, because deleting them would bias the test set, but their errors are analysed separately (section 5). |

**Feature engineering:** `RoomsPerPerson`, `BedroomRatio`, `LogPopulation`, distance to
SF / LA / San Diego / San Jose and `DistNearestCity`. Raw latitude/longitude are hard for
linear models, so distance-to-city gives them a usable coastal-metro signal.

**EDA insights** (figures in `reports/figures/`):
- `01_target_distribution.png`: House prices are skewed to the right, with most areas below $300k. There is a sharp spike at $500k because 992 areas (4.81%) are capped at that value, so the model cannot see true prices above it.
- `02_outliers_before_after.png`: Some values are extreme, for example average occupancy up to 1,243 people per house. Winsorising removed these extremes, so the boxplots became readable and linear models are not distorted.
- `03_price_map.png`: Expensive areas lie along the coast, especially around San Francisco and Los Angeles. Inland areas are cheaper.
- `04_correlations.png`: Median income has the strongest positive correlation with price (0.688). Distance to the nearest city is negatively correlated (-0.437), so houses closer to big cities cost more. `RoomsPerPerson` (0.341), an engineered feature, is also useful.
- `05_income_vs_value.png`: Price rises with income, but the spread gets wider at high incomes, and a flat line of points at $500k is visible because of the cap.

## 4. Modelling

- **Split:** 70% train (14,448) / 15% validation (3,096) / 15% test (3,096), `random_state=42`. The test set is used
  once, at the very end, after the model is chosen.
- **No leakage:** feature engineering, imputation and scaling are `sklearn.Pipeline` steps,
  so they are fitted only on training folds. Winsorising caps use feature quantiles only.
- **Models**

  | Role | Model | Why |
  |---|---|---|
  | Baseline 1 | Predict training median | The floor any model must beat. |
  | Baseline 2 | Ridge regression | Simple, interpretable linear benchmark. |
  | Approach A | Random Forest | Captures non-linearity and interactions, little tuning needed. |
  | Approach B | Gradient Boosting (`HistGradientBoostingRegressor`) | Usually the strongest on tabular data; tuned with `RandomizedSearchCV` (20 draws × 5-fold CV on the training set). |

- **Best hyper-parameters found:** `min_samples_leaf=20, max_leaf_nodes=63, max_iter=400, learning_rate=0.05, l2_regularization=0.1`.
- **Selection:** best validation RMSE. The chosen model is refit on train + validation and
  scored once on test.

## 5. Evaluation and interpretation

**Metrics.** RMSE (penalises large misses, in $100k), MAE (typical error, easy to explain)
and R² (share of variance explained). Chosen because the target is continuous and
large errors on expensive areas matter.

**Results:**

| Model | Val RMSE | Val R² | 5-fold CV RMSE |
|---|---|---|---|
| Median baseline | 1.174 | -0.058 | 1.190 |
| Ridge | 0.651 | 0.674 | 0.643 |
| Random Forest | 0.483 | 0.821 | 0.481 |
| Gradient Boosting (default) | 0.462 | 0.836 | 0.465 |
| Gradient Boosting (tuned) | 0.445 | 0.848 | 0.455 |

Final test result (Gradient Boosting, tuned): RMSE 0.414, MAE 0.274, R² 0.871.
Goal 1: **MET** (test RMSE fell from 1.182 for the median baseline to 0.414, a 65% reduction).
Goal 2: **MET** (R² 0.871 is above 0.80).

**Error by price band (test set):**

| Band | n | MAE | Mean residual (actual − predicted) |
|---|---|---|---|
| < $150k | 1137 | 0.194 | -0.131 |
| $150-300k | 1386 | 0.254 | -0.028 |
| $300-450k | 385 | 0.444 | 0.154 |
| $450k+ (capped) | 188 | 0.560 | 0.523 |

**Discussion**
- *Model comparison:* Ridge was clearly the weakest model (R² 0.674). Random Forest and Gradient Boosting reduced the error a lot (RMSE from 0.651 to about 0.46), which shows the relationship between the features and price is non-linear. Tuning gave a smaller gain (validation RMSE 0.462 to 0.445), but it is larger than the CV standard deviation of about 0.012, so it is a real improvement.
- *Error analysis* (`07`-`09` and the table above): The error is lowest for cheap houses (MAE 0.194) and highest for expensive ones (MAE 0.560 for the capped group). The model over-predicts cheap areas (mean residual -0.131) and under-predicts expensive areas, which is the common "pull towards the average" effect. For the $450k+ group it under-predicts by about $52k on average, because the true values are capped and these areas are rare (188 of 3,096). The error map (`09_error_map.png`) shows where the larger errors are located geographically.
- *Feature importance* (`10_permutation_importance.png`): Latitude (0.618), Longitude (0.553) and MedInc (0.406) are the most important features, so location and income drive prices. This shows association, not causation.
- *Worst predictions:* see `reports/worst_predictions.csv`. The largest errors are in the expensive end of the market, which is consistent with the price-band table above.

## 6. Limitations, bias and next steps

- **Old data (1990):** prices, demographics and neighbourhoods have changed; the model should not be used to value houses today.
- **Capped target:** values above $500k are recorded as $500k, so the model cannot learn how expensive the top end really is.
- **Spatial dependence:** nearby block groups look alike, so a random split can be optimistic.
  We checked this: random CV RMSE 0.439 vs spatial-held-out CV RMSE 0.714. The error is much higher when whole regions are hidden, which means the random split is optimistic: the model partly learns from nearby similar neighbourhoods. It would be less accurate in a region it has never seen.
- **Aggregated units:** each row is a block-group average, so we predict neighbourhoods, not individual homes (ecological fallacy).
- **Bias / fairness:** location and income proxy for demographics, so using such a model for lending or
  appraisal could reinforce historical redlining patterns.
- **Next steps:** add school and crime data, use a time-aware dataset, try quantile regression for
  prediction intervals, handle the censored target with a censored-regression approach.

## 7. Repository layout

```
├── README.md            report (this file)
├── requirements.txt
├── src/run_project.py   full pipeline: audit -> EDA -> models -> tuning -> test -> error analysis
├── reports/             generated figures, results.md, results.json, worst_predictions.csv
└── game/                Part B: index.html (the game), README.md (how to play, objectives), PLAYTEST.md
```
