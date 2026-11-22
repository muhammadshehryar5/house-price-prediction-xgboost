# House Price Prediction with XGBoost and Stacking

**Kaggle leaderboard RMSLE: 0.12094** | 5-fold CV RMSE: 0.10936 (log scale)

- Competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Notebook: [house-price-xg-boost on Kaggle](https://www.kaggle.com/code/sheharyaaarr/house-price-xg-boost)

Predicting the sale price of homes in Ames, Iowa from 79 features, using a full ML pipeline: data cleaning, feature engineering, leak-free encoding, tuned XGBoost, stacking and SHAP explanations.

---

## Results

| Stage | CV RMSE (log scale) | Change |
|---|---|---|
| XGBoost, no feature engineering (default) | 0.13901 | baseline |
| + feature engineering & encoding | 0.13033 | -0.00868 |
| + learning rate 0.05 & early stopping | 0.11468 | -0.01565 |
| + Optuna tuning (100 trials) | 0.11356 | -0.00113 |
| + stacking (XGBoost + Ridge + Gradient Boosting) | **0.10936** | -0.00420 |

Total improvement over the baseline: 0.02965 (21.3%).

The leaderboard score (0.12094) is higher than the CV score by about 0.012. This gap is typical: the hidden test set differs a little from the training folds, and tuning on the same folds used for scoring is slightly optimistic. My first submission (a Lasso/XGBoost blend) scored 0.12496.

![Model comparison](images/model_comparison.png)

---

## Why RMSLE and the log transform

The competition scores `RMSE(log(predicted), log(actual))`, called RMSLE. Errors are judged in percentage terms: a $20K miss on a $100K house costs more than the same miss on a $500K house.

`SalePrice` is right-skewed (skewness 1.88). After `log1p` the skewness is 0.12, so the target is close to normal. All models are trained on `log1p(SalePrice)`, so plain RMSE in log space equals the competition metric, and predictions are converted back with `expm1`.

![SalePrice distribution](images/saleprice_distribution.png)

---

## Pipeline

1. **Outliers:** removed 2 houses with `GrLivArea > 4000` that sold for under $300K (Ids 524 and 1299).

   ![Outliers](images/grlivarea_outliers.png)

2. **Missing values:** 19 columns in train and 33 in test. Most NAs mean "feature absent" (no pool, no garage, no alley), so they are filled with `"None"` or `0`. Genuine gaps use train-only statistics (median `LotFrontage` by neighborhood, mode for a few categoricals). Result: zero nulls.

   ![Missing values](images/missing_values.png)

3. **Feature engineering, drops and transforms** (summary below).
4. **Encoding:** ordinal, k-fold target and one-hot.
5. **Modeling:** baseline, early stopping, Optuna, stacking.
6. **Explanation:** SHAP and residual analysis.

### Feature engineering summary

| What | Details |
|---|---|
| **Created (15)** | `TotalSF`, `TotalBaths`, `AgeAtSale`, `AgeRemod`, `TotalPorchSF`, `OverallScore` (quality x condition), `QualAreaScore` (quality x living area), plus 8 binary flags: `HasPool`, `Has2ndFloor`, `HasGarage`, `HasBasement`, `HasFireplace`, `HasPorch`, `IsRemodeled`, `IsNew` |
| **Dropped** | `Utilities` and `Street` (one value in about 99.6-99.9% of rows) and `Id` |
| **Ordinal encoded (20)** | Quality and condition scales (Po to Ex), basement exposure and finish, garage finish, functional, land slope, lot shape, paved drive, central air and fence, mapped to integers in natural order (`None` = 0) |
| **Target encoded (3)** | `Neighborhood`, `Exterior1st`, `Exterior2nd`: smoothed target mean using 5-fold out-of-fold encoding on train (no leakage), full-train encoder for test |
| **One-hot encoded** | Remaining nominal columns, plus `MSSubClass` and `MoSold` (numeric codes that are really categories). Near-constant dummies dropped |
| **log1p-transformed (19)** | Continuous features with skewness > 0.75, measured on train only |
| **Final feature count** | 171 |

Out-of-fold target encoding matters: letting each row see its own price makes `Neighborhood` look more predictive than it really is (correlation 0.742 vs 0.721 out-of-fold).

![Correlation heatmap](images/correlation_heatmap.png)

### Impact of each pipeline stage

Feature engineering and encoding alone lowered the error by about 6%. The largest single gain came from lowering XGBoost's learning rate and stopping early: the default settings overfit 1,458 training rows.

---

## Hyperparameter tuning (Optuna)

100 trials of TPE search, each scored by 5-fold CV with early stopping. Best CV RMSE 0.11346 (0.11356 re-checked with a fixed number of trees).

| Parameter | Best value |
|---|---|
| `max_depth` | 3 |
| `min_child_weight` | 3 |
| `subsample` | 0.636 |
| `colsample_bytree` | 0.423 |
| `reg_alpha` | 0.00495 |
| `reg_lambda` | 0.00925 |
| `gamma` | 0.0066 |
| `learning_rate` / `n_estimators` | 0.05 / 330 |

Shallow trees (depth 3) suit this small dataset.

![Optuna history](images/optuna_history.png)

---

## Stacking architecture

```
Training data (1,458 rows, 171 features)
        |
        |--> XGBoost (Optuna-tuned) ------|
        |--> Ridge (RobustScaler) --------|--> 3-column out-of-fold (OOF)
        |--> Gradient Boosting (Huber) ---|    prediction matrix (5-fold CV)
                                                    |
                                                    v
                                  Ridge meta-learner (non-negative weights)
                                                    |
                                                    v
                          log1p(price) prediction --> expm1 --> price in $
```

Each base model produces out-of-fold predictions, so the meta-learner only trains on predictions made for rows the base models never saw. Stacking improved CV RMSE from 0.11356 (best single model) to 0.10936.

---

## SHAP: what drives house prices

SHAP values from the tuned XGBoost model (`TreeExplainer`). Effects are in log-price units and converted to percentages for readability.

![SHAP summary](images/shap_summary.png)

**Quantified insights** (from the notebook's SHAP cell):

1. **Quality x size is the strongest driver.** `QualAreaScore` (overall quality x living area) moves the predicted price by about 9.5% on average, followed by `TotalSF` (about 6.0%), `Neighborhood_TE` (about 3.8%) and `OverallQual` (about 3.2%).
2. **Location is worth about 9% of the price.** Comparing the top and bottom quartile of neighborhood encoding adds +9.2% (about $14,900 at the median price of $163,000), all else equal.
3. **One quality point matters a lot.** `OverallQual` 9 vs 7 adds +7.0% (about $11,400) from that feature alone, and the model credits quality again through `QualAreaScore` and `OverallScore`, so the full effect is larger.

*Note: credit is shared between correlated features (`QualAreaScore`, `TotalSF` and `OverallQual` all contain size or quality), so a single feature's effect understates the total effect of size and quality.*

| Most expensive house | Least expensive house |
|---|---|
| ![Expensive](images/shap_waterfall_expensive.png) | ![Cheap](images/shap_waterfall_cheap.png) |

- **Most expensive house** (actual $755,000, predicted about $713,000): `QualAreaScore` alone adds about +42% to the price, followed by `TotalSF` and `OverallQual`.
- **Least expensive house** (actual $34,900, predicted about $39,000): the model is pulled down by `QualAreaScore` and `TotalSF`, by commercial zoning (`MSZoning_C`, about -21%) and by an abnormal sale (about -7%).

![Dependence plots](images/shap_dependence.png)

---

## Residual analysis

Residuals use out-of-fold stacking predictions, so every house is predicted by models that never saw it.

![Actual vs predicted](images/actual_vs_predicted.png)
![Residuals](images/residuals.png)

---

## Limitations

- Only 1,458 training rows, so a few unusual sales (partial, abnormal or very cheap houses) dominate the remaining error.
- Hyperparameters and stacking weights were tuned on the same folds used for scoring, so CV is slightly optimistic.
- The two outliers were removed by inspection; other unusual houses were kept.
- Neighborhood effects are learned from limited data per neighborhood (smoothed target encoding helps but does not remove this).

## What would improve the score

- More diverse base models (LightGBM, CatBoost, kernel ridge or SVR) and repeated-seed or repeated-CV averaging
- Tuning Ridge and Gradient Boosting as carefully as XGBoost
- A neural network on tabular data as an extra stacking input
- Feature selection and targeted interactions (for example neighborhood x quality)

---

## Reproducing

1. Open the [Kaggle notebook](https://www.kaggle.com/code/sheharyaaarr/house-price-xg-boost) and attach the competition data, or download the data from the competition page (see `NOTE.md`).
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook top to bottom. Every cell is safe to re-run, uses `RANDOM_STATE = 42`, and saves checkpoints so a disconnected session can resume. The Optuna search stores its trials in a SQLite file and resumes where it stopped.

## Repository contents

| File / folder | Contents |
|---|---|
| `README.md` | This document |
| `house_price_xgboost.ipynb` | Notebook downloaded from Kaggle |
| `images/` | Charts used in this README |
| `requirements.txt` | Libraries with versions |
| `NOTE.md` | Data availability note |

## Tech stack

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost, LightGBM (available, not used in the final blend), Optuna, SHAP, category_encoders.
