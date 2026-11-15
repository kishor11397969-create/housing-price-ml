"""End-to-end pipeline: California house prices.

Run:  python src/run_project.py
Outputs: reports/figures/*.png, reports/results.json, reports/results.md
Everything random uses SEED, so results are reproducible.
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
FIG = os.path.join("reports", "figures")
os.makedirs(FIG, exist_ok=True)
RESULTS = {}


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
def load_data():
    """1990 US census block groups for California (20,640 rows).
    Target MedHouseVal is in units of $100,000."""
    return fetch_california_housing(as_frame=True).frame


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=130)
    plt.close()


# ----------------------------------------------------------------------------
# 2. DATA PREPARATION + EDA
# ----------------------------------------------------------------------------
def audit(df):
    """Data-quality checks. Each check is logged so the README can cite it."""
    info = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "dtypes_all_numeric": bool(all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes)),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_capped_rows": int((df["MedHouseVal"] >= 5.0).sum()),
        "target_capped_pct": round(float((df["MedHouseVal"] >= 5.0).mean() * 100), 2),
        "house_age_capped_rows": int((df["HouseAge"] >= 52).sum()),
    }
    outliers = {}
    for c in ["AveRooms", "AveBedrms", "Population", "AveOccup"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        hi = q3 + 3 * (q3 - q1)
        outliers[c] = {"max": float(df[c].max()), "n_above_3IQR": int((df[c] > hi).sum())}
    info["extreme_outliers"] = outliers
    RESULTS["data_audit"] = info
    return info


def clean(df):
    """Reasoned cleaning decisions (see README section 3)."""
    df = df.drop_duplicates().copy()
    # Winsorise extreme skewed ratios at the 0.5 / 99.5 percentile.
    # Caps are computed from features only, so no target information leaks.
    for c in ["AveRooms", "AveBedrms", "Population", "AveOccup"]:
        lo, hi = df[c].quantile([0.005, 0.995])
        df[c] = df[c].clip(lo, hi)
    return df


# Reference points (lat, lon) for distance-to-city features
CITIES = {"SF": (37.77, -122.42), "LA": (34.05, -118.24), "SD": (32.72, -117.16), "SJ": (37.34, -121.89)}


def add_features(X):
    X = X.copy()
    X["RoomsPerPerson"] = X["AveRooms"] / X["AveOccup"]
    X["BedroomRatio"] = X["AveBedrms"] / X["AveRooms"]
    X["LogPopulation"] = np.log1p(X["Population"])
    dists = []
    for name, (la, lo) in CITIES.items():
        X[f"Dist_{name}"] = np.sqrt((X["Latitude"] - la) ** 2 + (X["Longitude"] - lo) ** 2)
        dists.append(f"Dist_{name}")
    X["DistNearestCity"] = X[dists].min(axis=1)
    return X


def eda(df_raw, df):
    # 1. target distribution -> shows the $500k cap
    plt.figure(figsize=(6, 4))
    plt.hist(df_raw["MedHouseVal"] * 100, bins=60, color="#4C72B0")
    plt.xlabel("Median house value ($ thousands)"); plt.ylabel("Block groups")
    plt.title("Target is right-skewed and capped at $500k")
    savefig("01_target_distribution.png")

    # 2. raw vs winsorised for the worst outlier features
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].boxplot([df_raw["AveRooms"], df_raw["AveOccup"]], tick_labels=["AveRooms", "AveOccup"])
    ax[0].set_yscale("log"); ax[0].set_title("Raw (log scale)")
    ax[1].boxplot([df["AveRooms"], df["AveOccup"]], tick_labels=["AveRooms", "AveOccup"])
    ax[1].set_title("After winsorising (0.5-99.5%)")
    savefig("02_outliers_before_after.png")

    # 3. map of prices
    plt.figure(figsize=(6.5, 5.5))
    sc = plt.scatter(df_raw["Longitude"], df_raw["Latitude"], c=df_raw["MedHouseVal"] * 100,
                     s=3, cmap="viridis")
    plt.colorbar(sc, label="Median value ($k)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title("Prices cluster along the coast")
    savefig("03_price_map.png")

    # 4. correlations (with engineered features)
    feats = add_features(df.drop(columns="MedHouseVal"))
    corr = pd.concat([feats, df["MedHouseVal"]], axis=1).corr()["MedHouseVal"].drop("MedHouseVal").sort_values()
    plt.figure(figsize=(6, 6))
    plt.barh(corr.index, corr.values, color=np.where(corr.values > 0, "#55A868", "#C44E52"))
    plt.xlabel("Pearson correlation with target"); plt.title("Feature correlation with house value")
    savefig("04_correlations.png")
    RESULTS["top_correlations"] = {k: round(float(v), 3) for k, v in
                                   corr.sort_values(key=abs, ascending=False).head(6).items()}

    # 5. income vs value
    plt.figure(figsize=(6, 4))
    plt.scatter(df["MedInc"], df["MedHouseVal"] * 100, s=2, alpha=0.3)
    plt.xlabel("Median income (x $10k)"); plt.ylabel("Median house value ($k)")
    plt.title("Income is the strongest single predictor")
    savefig("05_income_vs_value.png")


# ----------------------------------------------------------------------------
# 3. MODELLING
# ----------------------------------------------------------------------------
def make_pipeline(model, scale=False):
    """Feature engineering + imputation (+ scaling) live INSIDE the pipeline, so
    during cross-validation every statistic is learned from the training fold only."""
    steps = [("fe", FunctionTransformer(add_features, validate=False)),
             ("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def metrics(y_true, y_pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred))}


def main():
    df_raw = load_data()
    audit(df_raw)
    df = clean(df_raw)
    eda(df_raw, df)

    X, y = df.drop(columns="MedHouseVal"), df["MedHouseVal"]
    # 70 / 15 / 15 split. The test set is touched exactly once, at the end.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)
    RESULTS["split_sizes"] = {"train": len(X_tr), "val": len(X_val), "test": len(X_te)}

    cv = KFold(5, shuffle=True, random_state=SEED)
    candidates = {
        "Baseline: predict median": make_pipeline(DummyRegressor(strategy="median")),
        "Baseline: Ridge regression": make_pipeline(Ridge(alpha=1.0), scale=True),
        "Random Forest": make_pipeline(RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                            n_jobs=-1, random_state=SEED)),
        "Gradient Boosting (default)": make_pipeline(HistGradientBoostingRegressor(random_state=SEED)),
    }

    # --- compare on the validation set + 5-fold CV on the training set
    table = {}
    for name, pipe in candidates.items():
        cv_rmse = -cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
        pipe.fit(X_tr, y_tr)
        m = metrics(y_val, pipe.predict(X_val))
        m["CV_RMSE_mean"], m["CV_RMSE_std"] = float(cv_rmse.mean()), float(cv_rmse.std())
        table[name] = m
        print(f"{name:30s} val RMSE={m['RMSE']:.3f}  R2={m['R2']:.3f}  CV RMSE={cv_rmse.mean():.3f}+/-{cv_rmse.std():.3f}")

    # --- hyper-parameter tuning of gradient boosting (CV on train only)
    search = RandomizedSearchCV(
        make_pipeline(HistGradientBoostingRegressor(random_state=SEED)),
        param_distributions={
            "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "model__max_leaf_nodes": [15, 31, 63, 127],
            "model__min_samples_leaf": [10, 20, 40],
            "model__l2_regularization": [0.0, 0.1, 1.0],
            "model__max_iter": [200, 400, 600],
        },
        n_iter=20, cv=cv, scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=-1)
    search.fit(X_tr, y_tr)
    tuned = search.best_estimator_
    m = metrics(y_val, tuned.predict(X_val))
    m["CV_RMSE_mean"], m["CV_RMSE_std"] = float(-search.best_score_), float("nan")
    table["Gradient Boosting (tuned)"] = m
    RESULTS["best_params"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    RESULTS["validation_table"] = table

    names = list(table); rmse = [table[n]["RMSE"] for n in names]
    plt.figure(figsize=(8, 4))
    plt.barh(names[::-1], rmse[::-1], color="#4C72B0")
    plt.xlabel("Validation RMSE ($100k) - lower is better"); plt.title("Model comparison")
    savefig("06_model_comparison.png")

    # --- final: refit best on train+val, evaluate ONCE on the test set
    pool = {"Random Forest": candidates["Random Forest"],
            "Gradient Boosting (default)": candidates["Gradient Boosting (default)"],
            "Gradient Boosting (tuned)": tuned}
    best_name = min(pool, key=lambda n: table[n]["RMSE"])
    final = clone(pool[best_name]).fit(pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val]))
    pred = final.predict(X_te)
    RESULTS["final_model"] = best_name
    RESULTS["test_metrics"] = metrics(y_te, pred)
    RESULTS["test_baseline_median_RMSE"] = float(
        np.sqrt(mean_squared_error(y_te, np.full(len(y_te), y_tr.median()))))
    print("FINAL", best_name, RESULTS["test_metrics"])

    # --- spatial robustness check: neighbouring block groups are similar, so a random
    # split can be optimistic. Re-score with whole geographic clusters held out.
    groups = KMeans(n_clusters=20, random_state=SEED, n_init=10).fit_predict(X[["Latitude", "Longitude"]])
    sp = -cross_val_score(clone(pool[best_name]), X, y, groups=groups, cv=GroupKFold(5),
                          scoring="neg_root_mean_squared_error")
    rd = -cross_val_score(clone(pool[best_name]), X, y, cv=cv, scoring="neg_root_mean_squared_error")
    RESULTS["random_cv_RMSE"] = float(rd.mean())
    RESULTS["spatial_cv_RMSE"] = float(sp.mean())

    # ------------------------------------------------------------------------
    # 4. ERROR ANALYSIS
    # ------------------------------------------------------------------------
    res = y_te - pred
    plt.figure(figsize=(5.5, 5))
    plt.scatter(y_te * 100, pred * 100, s=3, alpha=0.4)
    plt.plot([0, 520], [0, 520], "r--", lw=1)
    plt.xlabel("Actual ($k)"); plt.ylabel("Predicted ($k)"); plt.title("Predicted vs actual (test set)")
    savefig("07_pred_vs_actual.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(pred * 100, res * 100, s=3, alpha=0.4); plt.axhline(0, color="r", lw=1)
    plt.xlabel("Predicted ($k)"); plt.ylabel("Residual ($k)"); plt.title("Residuals vs prediction")
    savefig("08_residuals.png")

    bins = pd.cut(y_te, [0, 1.5, 3.0, 4.5, 5.01], labels=["<$150k", "$150-300k", "$300-450k", "$450k+ (capped)"])
    by_bin = (pd.DataFrame({"abs_err": res.abs(), "res": res, "bin": bins})
              .groupby("bin", observed=True)
              .agg(n=("abs_err", "size"), MAE=("abs_err", "mean"), mean_residual=("res", "mean")))
    RESULTS["error_by_price_band"] = by_bin.round(3).reset_index().astype({"bin": str}).to_dict("records")

    plt.figure(figsize=(6.5, 5.5))
    sc = plt.scatter(X_te["Longitude"], X_te["Latitude"], c=res.abs() * 100, s=5, cmap="magma_r", vmax=100)
    plt.colorbar(sc, label="Absolute error ($k)"); plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title("Where is the model wrong?")
    savefig("09_error_map.png")

    (X_te.assign(actual=y_te, pred=pred, abs_err=res.abs())
         .sort_values("abs_err", ascending=False).head(10).round(3)
         .to_csv(os.path.join("reports", "worst_predictions.csv")))

    pi = permutation_importance(final, X_te, y_te, n_repeats=5, random_state=SEED, n_jobs=-1,
                                scoring="neg_root_mean_squared_error")
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values()
    plt.figure(figsize=(6, 4)); plt.barh(imp.index, imp.values, color="#55A868")
    plt.xlabel("Increase in RMSE when feature is shuffled"); plt.title("Permutation importance")
    savefig("10_permutation_importance.png")
    RESULTS["top_permutation_importance"] = {k: round(float(v), 3) for k, v in
                                             imp.sort_values(ascending=False).head(5).items()}

    # ------------------------------------------------------------------------
    # 5. SAVE
    # ------------------------------------------------------------------------
    with open(os.path.join("reports", "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    write_markdown(table)


def write_markdown(table):
    t = RESULTS["test_metrics"]
    lines = ["# Auto-generated results (re-created every run)\n",
             f"Split sizes: {RESULTS['split_sizes']}\n",
             "## Validation comparison\n",
             "| Model | Val RMSE | Val MAE | Val R2 | 5-fold CV RMSE |", "|---|---|---|---|---|"]
    for n, m in table.items():
        lines.append(f"| {n} | {m['RMSE']:.3f} | {m['MAE']:.3f} | {m['R2']:.3f} | {m['CV_RMSE_mean']:.3f} |")
    lines += ["", f"## Final test result ({RESULTS['final_model']})",
              f"RMSE {t['RMSE']:.3f} | MAE {t['MAE']:.3f} | R2 {t['R2']:.3f}  (RMSE/MAE units: $100k)", "",
              f"Median-baseline test RMSE: {RESULTS['test_baseline_median_RMSE']:.3f}", "",
              f"Random 5-fold CV RMSE: {RESULTS['random_cv_RMSE']:.3f}  vs  "
              f"spatial (held-out regions) CV RMSE: {RESULTS['spatial_cv_RMSE']:.3f}", "",
              "## Error by price band", "| Band | n | MAE | mean residual |", "|---|---|---|---|"]
    for r in RESULTS["error_by_price_band"]:
        lines.append(f"| {r['bin']} | {r['n']} | {r['MAE']} | {r['mean_residual']} |")
    lines += ["", f"Best hyper-parameters: {RESULTS['best_params']}",
              f"Top correlations: {RESULTS['top_correlations']}",
              f"Top permutation importances: {RESULTS['top_permutation_importance']}",
              f"Data audit: {RESULTS['data_audit']}"]
    with open(os.path.join("reports", "results.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
