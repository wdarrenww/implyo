# PiP-Impute Tutorials

This guide provides detailed examples and explanations for the main features of the PiP-Impute library.

## 1. Analyzing Missing Data

Before imputing, it's crucial to understand the extent and patterns of missing data.

### Summarizing Missingness

The `missing_value_summary` function provides a quick overview of the number and percentage of missing values in each column.

```python
import pandas as pd
import numpy as np
import pip_impute as pipi

data = {'A': [1, np.nan, 3, 4, 5], 'B': [np.nan, 'x', 'y', 'z', np.nan], 'C': [1,2,3,4,5]}
df = pd.DataFrame(data)

summary = pipi.missing_value_summary(df)
print(summary)
#      column_name  missing_count  missing_percentage
# 0             B              2                40.0
# 1             A              1                20.0
# 2             C              0                 0.0
```

### Visualizing Missingness

Visualizations can reveal patterns that numbers alone cannot.

* `plot_missingness_heatmap`: Shows a heatmap of missing data locations.
* `plot_missingness_bar`: A bar chart of missing value counts per column.

```python
# These functions will open plot windows
# pipi.plot_missingness_heatmap(df)
# pipi.plot_missingness_bar(df)
```

### Analyzing Patterns (MCAR, MAR, MNAR)

The library provides tools for a preliminary investigation into whether data is Missing Completely at Random (MCAR) or Missing at Random (MAR).

```python
# Check if the missingness in column 'B' is related to observed values in 'A' or 'C'
mcar_test_results = pipi.preliminary_mcar_test(df, missing_col='B')
print(mcar_test_results)

# Get a high-level suggestion for the missingness pattern of all columns
pattern_suggestions = pipi.suggest_missingness_pattern(df)
print(pattern_suggestions)
```

---

## 2. Simple Imputation Strategies

These imputers are fast and easy to use, suitable for simple cases or as a baseline. All follow the `fit_transform` API.

```python
df_simple_impute = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [10, np.nan, 10, 20]})

# Mean Imputation (only affects numeric columns)
mean_imputer = pipi.MeanImputer()
df_mean = mean_imputer.fit_transform(df_simple_impute.copy())
print("Mean Imputed:\n", df_mean)
# Mean of A = (1+2+4)/3 = 2.333. Mean of B = (10+10+20)/3 = 13.333

# Mode Imputation (works on numeric and categorical)
mode_imputer = pipi.ModeImputer()
df_mode = mode_imputer.fit_transform(df_simple_impute.copy())
print("\nMode Imputed:\n", df_mode)
# Mode of B is 10.

# Random Sample Imputation
random_imputer = pipi.RandomSampleImputer(random_state=42)
df_random = random_imputer.fit_transform(df_simple_impute.copy())
print("\nRandom Sample Imputed:\n", df_random)
```

---

## 3. Model-Based Imputation

These imputers use machine learning models to predict missing values based on other features, often providing more accurate results.

### K-Nearest Neighbors (KNN) Imputer

`KNNImputer` fills missing values using the average (or distance-weighted average) of the `n_neighbors` most similar rows. It is best suited for numeric data.

```python
df_knn_impute = pd.DataFrame({
    'height': [165, 180, np.nan, 172, 185],
    'weight': [60, 85, 70, np.nan, 90],
    'age':    [25, 40, 35, 30, 45] # Complete column to help find neighbors
})

# Use 3 neighbors, weighting closer neighbors more heavily
knn_imputer = pipi.KNNImputer(n_neighbors=3, weights='distance')
df_knn_imputed = knn_imputer.fit_transform(df_knn_impute.copy())

print("Original Data for KNN:\n", df_knn_impute)
print("\nKNN Imputed Data:\n", df_knn_imputed)
```

### IterativeImputer (MICE)

`IterativeImputer` is a powerful multivariate technique that models each feature with missing values as a function of all other features. It iteratively fits models (e.g., Linear Regression for numeric, Logistic Regression for categorical) and updates imputations until they converge.

```python
df_mice_impute = pd.DataFrame({
    'age': [29, 45, 32, np.nan, 22],
    'income': [70000, 95000, np.nan, 120000, 50000],
    'education': ['Bachelors', 'Masters', 'Masters', 'PhD', np.nan]
})

# For a single, deterministic imputation
mice_imputer_det = pipi.IterativeImputer(
    max_iter=5,
    sample_posterior=False, # Deterministic prediction
    random_state=42
)

df_mice_imputed = mice_imputer_det.fit_transform(df_mice_impute.copy())
print("Deterministically Imputed with MICE:\n", df_mice_imputed)
```

---

## 4. Multiple Imputation and Uncertainty Quantification

Single imputation does not account for the uncertainty about what the missing values might have been. **Multiple Imputation (MI)** addresses this by creating several plausible versions of the completed dataset.

### Generating Multiple Datasets

To perform MI, we use `IterativeImputer` with `sample_posterior=True`. This makes the imputation process stochastic:
* For numeric features, it adds random noise based on the model's prediction errors.
* For categorical features, it samples a category based on the model's predicted probabilities.

```python
mice_imputer_stochastic = pipi.IterativeImputer(
    max_iter=5,
    sample_posterior=True, # Enable stochastic imputation
    random_state=42
)

# Fit the imputer once
mice_imputer_stochastic.fit(df_mice_impute.copy())

# Generate 5 different imputed datasets
imputed_datasets = mice_imputer_stochastic.impute_multiple(df_mice_impute.copy(), n_imputations=5)
print(f"Generated {len(imputed_datasets)} datasets.")

# Observe the different imputed values for 'income'
print("\nImputed 'income' values across datasets:")
for i, df in enumerate(imputed_datasets):
    print(f"  Dataset {i+1}: {df.loc[2, 'income']:.2f}")
```

### Pooling Results with Rubin's Rules

After performing an analysis on each of the `m` imputed datasets, the results must be combined (pooled) into a single estimate. PiP-Impute provides tools for this based on Rubin's Rules.

Suppose we want to find the mean `income` and its confidence interval.

```python
# 1. Calculate the mean income in each of the 5 datasets
all_means = [df['income'].mean() for df in imputed_datasets]
print("\nMean 'income' from each dataset:", [round(m) for m in all_means])

# 2. Use pool_means to get the final point estimate
pooled_mean_income = pipi.pool_means(imputed_datasets, target_vars='income')
print(f"\nFinal Pooled Mean Income: {pooled_mean_income:.2f}")

# 3. Use pool_variances_rubin to get the total variance, accounting for uncertainty
total_variance_income = pipi.pool_variances_rubin(imputed_datasets, target_vars='income')
print(f"Total Pooled Variance: {total_variance_income:.2f}")

# 4. Calculate standard error and a 95% confidence interval
std_error = np.sqrt(total_variance_income)
confidence_interval = (
    pooled_mean_income - 1.96 * std_error,
    pooled_mean_income + 1.96 * std_error
)
print(f"Standard Error of Pooled Mean: {std_error:.2f}")
print(f"95% Confidence Interval for Mean Income: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")
```
This final confidence interval is wider than one from a single imputation, correctly reflecting the uncertainty introduced by the missing data.