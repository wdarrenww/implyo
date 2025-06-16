import pandas as pd
import numpy as np
import warnings
from implyo.imputers import IterativeImputer
from implyo.imputers import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.impute import KNNImputer as SKKNNImputer

# Test data
df_all_nan_cat = pd.DataFrame({
    'cat_all_nan': [np.nan, np.nan, np.nan],
    'num': [1,2,3]
})

print("DataFrame:")
print(df_all_nan_cat)
print("\nDataFrame info:")
print(df_all_nan_cat.info())
print("\nDataFrame dtypes:")
print(df_all_nan_cat.dtypes)

# Check if cat_all_nan is considered categorical
print(f"\nIs cat_all_nan numeric? {pd.api.types.is_numeric_dtype(df_all_nan_cat['cat_all_nan'])}")
print(f"Is cat_all_nan boolean? {pd.api.types.is_bool_dtype(df_all_nan_cat['cat_all_nan'])}")

# Test the imputer
imputer = IterativeImputer(random_state=42)

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    imputer.fit(df_all_nan_cat.copy())
    
    print(f"\nNumber of warnings captured: {len(w)}")
    for i, warning in enumerate(w):
        print(f"Warning {i}: {warning.message}")

print(f"\nFeature info: {imputer.feature_info_}")
print(f"Label encoders: {imputer.label_encoders_}")
print(f"Estimators: {imputer.estimators_}")

# Test data for KNN imputer - no missing values in numeric columns
df_no_numeric_na = pd.DataFrame({'A':[1,2], 'B':['x','y']})

print("DataFrame:")
print(df_no_numeric_na)
print("\nDataFrame info:")
print(df_no_numeric_na.info())
print("\nDataFrame dtypes:")
print(df_no_numeric_na.dtypes)

# Check which columns are numeric
print(f"\nIs A numeric? {pd.api.types.is_numeric_dtype(df_no_numeric_na['A'])}")
print(f"Is B numeric? {pd.api.types.is_numeric_dtype(df_no_numeric_na['B'])}")

# Check for missing values
print(f"\nMissing values in A: {df_no_numeric_na['A'].isnull().sum()}")
print(f"Missing values in B: {df_no_numeric_na['B'].isnull().sum()}")

# Test the imputer
imputer = KNNImputer()

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    imputer.fit(df_no_numeric_na)
    
    print(f"\nNumber of warnings captured: {len(w)}")
    for i, warning in enumerate(w):
        print(f"Warning {i}: {warning.message}")

print(f"\nFeature names in: {imputer.feature_names_in_}")
print(f"Fit X columns numeric: {imputer._fit_X_columns_numeric}")
print(f"Numeric columns to impute names: {imputer._numeric_columns_to_impute_names}")
print(f"Is fitted: {imputer._is_fitted}")

# Test with truly no numeric columns
df_no_numeric = pd.DataFrame({'B':['x','y'], 'C':['a','b']})
print(f"\n\nTesting with no numeric columns:")
print(df_no_numeric)
print(f"Is B numeric? {pd.api.types.is_numeric_dtype(df_no_numeric['B'])}")
print(f"Is C numeric? {pd.api.types.is_numeric_dtype(df_no_numeric['C'])}")

imputer2 = KNNImputer()
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    imputer2.fit(df_no_numeric)
    
    print(f"\nNumber of warnings captured: {len(w2)}")
    for i, warning in enumerate(w2):
        print(f"Warning {i}: {warning.message}")

# Test how nan_euclidean_distances behaves with no shared features
print("Testing nan_euclidean_distances behavior:")

# Case 1: Shared features
X1 = np.array([[1., 100.], [2., 100.]])
Y1 = np.array([[4., 100.]])
dist1 = nan_euclidean_distances(Y1, X1)
print(f"Distance from [4,100] to [1,100]: {dist1[0,0]}")
print(f"Distance from [4,100] to [2,100]: {dist1[0,1]}")

# Case 2: No shared features (one has NaN)
X2 = np.array([[1., 100.], [np.nan, 100.]])
Y2 = np.array([[4., 100.]])
dist2 = nan_euclidean_distances(Y2, X2)
print(f"Distance from [4,100] to [1,100]: {dist2[0,0]}")
print(f"Distance from [4,100] to [nan,100]: {dist2[0,1]}")

# Case 3: Only one shared feature
X3 = np.array([[1., 100.], [np.nan, 100.]])
Y3 = np.array([[4., 100.]])
dist3 = nan_euclidean_distances(Y3, X3)
print(f"Distance from [4,100] to [1,100]: {dist3[0,0]}")
print(f"Distance from [4,100] to [nan,100]: {dist3[0,1]}")

# Case 4: Only feat3 shared (feat1 is NaN in reference)
X4 = np.array([[np.nan, 100.], [5., 100.]])
Y4 = np.array([[4., 100.]])
dist4 = nan_euclidean_distances(Y4, X4)
print(f"Distance from [4,100] to [nan,100]: {dist4[0,0]}")
print(f"Distance from [4,100] to [5,100]: {dist4[0,1]}")

data = {'feat1': [1., 2., np.nan, 4., 5.],
        'feat2': [10., 20., 30., np.nan, 50.],
        'feat3': [100., 100., 100., 100., 100.]}
df = pd.DataFrame(data)

print("\nscikit-learn KNNImputer results:")
sk_imputer_3 = SKKNNImputer(n_neighbors=3)
sk_transformed_3 = sk_imputer_3.fit_transform(df.copy())
print(f"Imputed value for feat1[2]: {sk_transformed_3[2,0]}")
print(f"Imputed value for feat2[3]: {sk_transformed_3[3,1]}") 