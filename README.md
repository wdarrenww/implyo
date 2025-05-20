# Implyo: A Python Library for Advanced Data Imputation

Implyo is a Python library designed to provide a comprehensive suite of tools for understanding, analyzing, and imputing missing data in pandas DataFrames. It aims to offer both simple and advanced imputation techniques, along with robust missing data analysis and visualization capabilities.

**Current Version: 0.1.0-alpha**

## Features

* **Core Imputation API:**
    * Base imputer class (`BaseImputer`) following scikit-learn conventions (`fit`, `transform`, `fit_transform`).
    * Seamless integration with pandas DataFrames.
* **Simple Imputers:**
    * `MeanImputer`: Imputes with column mean (for numeric data).
    * `MedianImputer`: Imputes with column median (for numeric data).
    * `ModeImputer`: Imputes with column mode (for numeric or categorical data).
    * `ConstantImputer`: Imputes with a user-defined constant value.
    * `RandomSampleImputer`: Imputes by randomly sampling from observed column values.
* **Missing Data Analysis:**
    * Identification of missing values.
    * Summary statistics (count and percentage of missing values per column).
    * Preliminary checks for missingness patterns (MCAR, MAR).
* **Missing Data Visualization:**
    * Missingness heatmaps (`msno.matrix`).
    * Missingness bar plots (`msno.bar`).
    * Summary bar plot of missing percentages.

## Installation

### Prerequisites

* Python (>= 3.8 recommended)
* pip

### Dependencies

Implyo relies on the following libraries:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* scipy
* missingno

These will be installed automatically if you install Implyo via pip (once packaged).

### Local Installation (from source)

1.  Clone the repository (or download the source code):
    ```bash
    git clone https://github.com/wdarrenww/implyo.git
    cd implyo
    ```
2.  Install the package in editable mode (recommended for development):
    ```bash
    pip install -e .
    ```
    Alternatively, for a standard install:
    ```bash
    pip install .
    ```

## Getting Started

Here's a quick example of how to use Implyo:

```python
import pandas as pd
import numpy as np
import pip_impute as pipi

# 1. Create a sample DataFrame with missing values
data = {
    'age': [25, 30, np.nan, 45, 33, np.nan, 50],
    'income': [50000, 60000, 75000, np.nan, 55000, 80000, 90000],
    'gender': ['Male', 'Female', 'Female', np.nan, 'Male', 'Female', 'Male'],
    'city': ['A', 'B', np.nan, 'A', 'C', np.nan, 'B']
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# 2. Analyze missing data
print("\nMissing Value Summary:")
summary = pipi.missing_value_summary(df)
print(summary)

# Visualize missingness (plots will open in new windows)
# pipi.plot_missingness_heatmap(df)
# pipi.plot_missingness_bar(df)

# 3. Impute missing values
# Example: Mean imputation for numeric columns
mean_imputer = pipi.MeanImputer()
df_mean_imputed = mean_imputer.fit_transform(df.copy()) # Use .copy() to avoid modifying original df
print("\nDataFrame after Mean Imputation:")
print(df_mean_imputed)
print("Fitted means:", mean_imputer.statistics_)

# Example: Mode imputation for 'gender' and 'city'
mode_imputer = pipi.ModeImputer(columns=['gender', 'city'])
# Make a fresh copy for this imputation step if df was already modified
df_mode_imputed = mode_imputer.fit_transform(df.copy())
print("\nDataFrame after Mode Imputation (gender, city):")
print(df_mode_imputed[['gender', 'city']])
print("Fitted modes:", mode_imputer.statistics_)

# Example: Constant imputation for 'income'
constant_imputer = pipi.ConstantImputer(fill_value=-1, columns=['income'])
df_constant_imputed = constant_imputer.fit_transform(df.copy())
print("\nDataFrame after Constant Imputation (income filled with -1):")
print(df_constant_imputed[['income']])

# Example: Random Sample Imputation for 'age'
random_imputer = pipi.RandomSampleImputer(columns=['age'], random_state=42)
df_random_imputed = random_imputer.fit_transform(df.copy())
print("\nDataFrame after Random Sample Imputation (age):")
print(df_random_imputed[['age']])

```

## License
This project is licensed under the MIT License - see the LICENSE file for details 

