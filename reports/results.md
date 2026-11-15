# Auto-generated results (re-created every run)

Split sizes: {'train': 14448, 'val': 3096, 'test': 3096}

## Validation comparison

| Model | Val RMSE | Val MAE | Val R2 | 5-fold CV RMSE |
|---|---|---|---|---|
| Baseline: predict median | 1.174 | 0.877 | -0.058 | 1.190 |
| Baseline: Ridge regression | 0.651 | 0.469 | 0.674 | 0.643 |
| Random Forest | 0.483 | 0.307 | 0.821 | 0.481 |
| Gradient Boosting (default) | 0.462 | 0.301 | 0.836 | 0.465 |
| Gradient Boosting (tuned) | 0.445 | 0.282 | 0.848 | 0.455 |

## Final test result (Gradient Boosting (tuned))
RMSE 0.414 | MAE 0.274 | R2 0.871  (RMSE/MAE units: $100k)

Median-baseline test RMSE: 1.182

Random 5-fold CV RMSE: 0.439  vs  spatial (held-out regions) CV RMSE: 0.714

## Error by price band
| Band | n | MAE | mean residual |
|---|---|---|---|
| <$150k | 1137 | 0.194 | -0.131 |
| $150-300k | 1386 | 0.254 | -0.028 |
| $300-450k | 385 | 0.444 | 0.154 |
| $450k+ (capped) | 188 | 0.56 | 0.523 |

Best hyper-parameters: {'model__min_samples_leaf': 20, 'model__max_leaf_nodes': 63, 'model__max_iter': 400, 'model__learning_rate': 0.05, 'model__l2_regularization': 0.1}
Top correlations: {'MedInc': 0.688, 'DistNearestCity': -0.437, 'RoomsPerPerson': 0.341, 'AveOccup': -0.269, 'BedroomRatio': -0.267, 'AveRooms': 0.25}
Top permutation importances: {'Latitude': 0.618, 'Longitude': 0.553, 'MedInc': 0.406, 'AveOccup': 0.194, 'AveRooms': 0.149}
Data audit: {'rows': 20640, 'columns': 9, 'dtypes_all_numeric': True, 'missing_total': 0, 'duplicate_rows': 0, 'target_capped_rows': 992, 'target_capped_pct': 4.81, 'house_age_capped_rows': 1273, 'extreme_outliers': {'AveRooms': {'max': 141.9090909090909, 'n_above_3IQR': 180}, 'AveBedrms': {'max': 34.06666666666667, 'n_above_3IQR': 692}, 'Population': {'max': 35682.0, 'n_above_3IQR': 421}, 'AveOccup': {'max': 1243.3333333333333, 'n_above_3IQR': 132}}}