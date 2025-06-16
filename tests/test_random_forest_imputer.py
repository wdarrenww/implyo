"""Tests for the RandomForestImputer class."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from implyo import RandomForestImputer


@pytest.fixture
def sample_data():
    """Create a sample dataset with mixed types and missing values."""
    np.random.seed(42)
    n_samples = 100
    
    # Create numeric features
    numeric1 = np.random.normal(0, 1, n_samples)
    numeric2 = np.random.normal(5, 2, n_samples)
    
    # Create categorical features
    categories = ['A', 'B', 'C', 'D']
    categorical1 = np.random.choice(categories, n_samples)
    categorical2 = np.random.choice(['X', 'Y', 'Z'], n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'numeric1': numeric1,
        'numeric2': numeric2,
        'categorical1': categorical1,
        'categorical2': categorical2
    })
    
    # Add missing values
    mask = np.random.random(df.shape) < 0.2
    df[mask] = np.nan
    
    return df


def test_basic_imputation(sample_data):
    """Test basic imputation functionality."""
    imputer = RandomForestImputer(
        n_estimators=10,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    
    # Fit and transform
    X_imputed = imputer.fit_transform(sample_data)
    
    # Check that all missing values are imputed
    assert not X_imputed.isna().any().any()
    
    # Check that non-numeric columns are preserved
    assert all(col in X_imputed.columns for col in sample_data.columns)
    
    # Check that categorical columns are properly encoded
    assert all(X_imputed[col].dtype == 'object' for col in ['categorical1', 'categorical2'])
    assert all(X_imputed[col].isin(sample_data[col].dropna().unique()) for col in ['categorical1', 'categorical2'])


def test_edge_cases():
    """Test various edge cases."""
    # Empty DataFrame
    with pytest.raises(ValueError):
        imputer = RandomForestImputer()
        imputer.fit_transform(pd.DataFrame())
        
    # DataFrame with no numeric columns
    df = pd.DataFrame({
        'cat1': ['A', 'B', 'C'],
        'cat2': ['X', 'Y', 'Z']
    })
    imputer = RandomForestImputer(categorical_features=['cat1', 'cat2'])
    X_imputed = imputer.fit_transform(df)
    assert not X_imputed.isna().any().any()
    
    # DataFrame with all NaN values
    df = pd.DataFrame({
        'num1': [np.nan, np.nan, np.nan],
        'num2': [np.nan, np.nan, np.nan]
    })
    imputer = RandomForestImputer()
    with pytest.raises(ValueError):
        imputer.fit_transform(df)
        
    # DataFrame with no missing values
    df = pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [4, 5, 6]
    })
    imputer = RandomForestImputer()
    X_imputed = imputer.fit_transform(df)
    assert X_imputed.equals(df)


def test_algorithm_selection(sample_data):
    """Test different algorithm choices."""
    # Test Random Forest
    imputer_rf = RandomForestImputer(
        n_estimators=10,
        tree_type='rf',
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed_rf = imputer_rf.fit_transform(sample_data)
    
    # Test Extra Trees
    imputer_et = RandomForestImputer(
        n_estimators=10,
        tree_type='et',
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed_et = imputer_et.fit_transform(sample_data)
    
    # Check that both algorithms produce valid results
    assert not X_imputed_rf.isna().any().any()
    assert not X_imputed_et.isna().any().any()
    
    # Check that results are different (due to different algorithms)
    assert not X_imputed_rf.equals(X_imputed_et)


def test_initial_strategy(sample_data):
    """Test different initial imputation strategies."""
    strategies = ['mean', 'median', 'most_frequent', 'constant']
    
    for strategy in strategies:
        imputer = RandomForestImputer(
            n_estimators=10,
            initial_strategy=strategy,
            random_state=42,
            categorical_features=['categorical1', 'categorical2']
        )
        X_imputed = imputer.fit_transform(sample_data)
        assert not X_imputed.isna().any().any()


def test_column_selection(sample_data):
    """Test imputation with specific column selection."""
    # Select only numeric columns
    imputer = RandomForestImputer(
        n_estimators=10,
        columns=['numeric1', 'numeric2'],
        random_state=42
    )
    X_imputed = imputer.fit_transform(sample_data)
    
    # Check that only selected columns are imputed
    assert not X_imputed[['numeric1', 'numeric2']].isna().any().any()
    assert X_imputed[['categorical1', 'categorical2']].equals(
        sample_data[['categorical1', 'categorical2']]
    )
    
    # Test with non-existent column
    with pytest.warns(UserWarning):
        imputer = RandomForestImputer(columns=['nonexistent'])
        imputer.fit_transform(sample_data)


def test_large_dataset():
    """Test performance and correctness on a larger dataset."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create large dataset
    data = np.random.normal(0, 1, (n_samples, n_features))
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    
    # Add missing values
    mask = np.random.random(df.shape) < 0.1
    df[mask] = np.nan
    
    # Fit and transform
    imputer = RandomForestImputer(
        n_estimators=10,
        n_jobs=-1,
        random_state=42
    )
    X_imputed = imputer.fit_transform(df)
    
    # Check results
    assert not X_imputed.isna().any().any()
    assert X_imputed.shape == df.shape


def test_transform_without_fit(sample_data):
    """Test that transform raises error if called without fit."""
    imputer = RandomForestImputer()
    with pytest.raises(NotFittedError):
        imputer.transform(sample_data)


def test_invalid_parameters():
    """Test handling of invalid parameters."""
    # Invalid n_estimators
    with pytest.raises(ValueError):
        RandomForestImputer(n_estimators=0)
        
    # Invalid max_iter
    with pytest.raises(ValueError):
        RandomForestImputer(max_iter=0)
        
    # Invalid tol
    with pytest.raises(ValueError):
        RandomForestImputer(tol=0)
        
    # Invalid min_samples
    with pytest.raises(ValueError):
        RandomForestImputer(min_samples=1)
        
    # Invalid uncertainty_quantile
    with pytest.raises(ValueError):
        RandomForestImputer(uncertainty_quantile=1.5)
        
    # Invalid tree_type
    with pytest.raises(ValueError):
        RandomForestImputer(tree_type='invalid')


def test_missing_value_patterns(sample_data):
    """Test different patterns of missing values."""
    # Create dataset with all values missing in some columns
    df = sample_data.copy()
    df['numeric1'] = np.nan
    df['categorical1'] = np.nan
    
    imputer = RandomForestImputer(
        n_estimators=10,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed = imputer.fit_transform(df)
    
    assert not X_imputed.isna().any().any()
    
    # Create dataset with all values missing in some rows
    df = sample_data.copy()
    df.iloc[0] = np.nan
    
    X_imputed = imputer.fit_transform(df)
    assert not X_imputed.isna().any().any()


def test_consistency(sample_data):
    """Test that imputation is consistent across multiple calls."""
    imputer = RandomForestImputer(
        n_estimators=10,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    
    # First call
    X_imputed1 = imputer.fit_transform(sample_data)
    
    # Second call
    X_imputed2 = imputer.fit_transform(sample_data)
    
    # Check consistency
    assert X_imputed1.equals(X_imputed2)


def test_mixed_dtypes(sample_data):
    """Test handling of mixed data types."""
    # Add a boolean column
    sample_data['boolean'] = np.random.choice([True, False], size=len(sample_data))
    sample_data.loc[np.random.random(len(sample_data)) < 0.2, 'boolean'] = np.nan
    
    # Add a datetime column
    sample_data['datetime'] = pd.date_range('2020-01-01', periods=len(sample_data))
    sample_data.loc[np.random.random(len(sample_data)) < 0.2, 'datetime'] = np.nan
    
    imputer = RandomForestImputer(
        n_estimators=10,
        random_state=42,
        categorical_features=['categorical1', 'categorical2', 'boolean']
    )
    X_imputed = imputer.fit_transform(sample_data)
    
    assert not X_imputed.isna().any().any()
    assert X_imputed['boolean'].dtype == 'bool'
    assert X_imputed['datetime'].dtype == 'datetime64[ns]'


def test_high_dimensionality():
    """Test behavior with high-dimensional data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 100
    
    # Create high-dimensional dataset
    data = np.random.normal(0, 1, (n_samples, n_features))
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    
    # Add missing values
    mask = np.random.random(df.shape) < 0.1
    df[mask] = np.nan
    
    imputer = RandomForestImputer(
        n_estimators=10,
        max_features='sqrt',
        random_state=42
    )
    X_imputed = imputer.fit_transform(df)
    
    assert not X_imputed.isna().any().any()
    assert X_imputed.shape == df.shape


def test_sparse_missing(sample_data):
    """Test with sparse missing value patterns."""
    # Create sparse missing pattern
    df = sample_data.copy()
    df.iloc[::2, ::2] = np.nan  # Every other row and column
    
    imputer = RandomForestImputer(
        n_estimators=10,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed = imputer.fit_transform(df)
    
    assert not X_imputed.isna().any().any()


def test_uncertainty_quantification(sample_data):
    """Test uncertainty quantification features."""
    imputer = RandomForestImputer(
        n_estimators=100,
        uncertainty_quantile=0.95,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed = imputer.fit_transform(sample_data)
    
    # Check that uncertainty intervals are computed
    assert imputer.uncertainty_intervals_ is not None
    assert all(col in imputer.uncertainty_intervals_ for col in ['numeric1', 'numeric2'])
    
    # Check interval shapes
    for col in ['numeric1', 'numeric2']:
        intervals = imputer.uncertainty_intervals_[col]
        assert intervals.shape == (len(sample_data), 2)
        assert np.all(intervals[:, 0] <= intervals[:, 1])


def test_feature_importances(sample_data):
    """Test feature importance computation."""
    imputer = RandomForestImputer(
        n_estimators=10,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed = imputer.fit_transform(sample_data)
    
    # Check that feature importances are computed
    assert imputer.feature_importances_ is not None
    assert all(col in imputer.feature_importances_ for col in ['numeric1', 'numeric2'])
    
    # Check importance shapes
    for col in ['numeric1', 'numeric2']:
        importances = imputer.feature_importances_[col]
        assert len(importances) == len(sample_data.columns) - 1  # Excluding target column


def test_convergence_history(sample_data):
    """Test convergence history tracking."""
    imputer = RandomForestImputer(
        n_estimators=10,
        max_iter=5,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed = imputer.fit_transform(sample_data)
    
    # Check convergence history
    assert imputer.convergence_history_ is not None
    assert all(metric in imputer.convergence_history_ for metric in ['rmse', 'mae', 'max_diff'])
    assert len(imputer.convergence_history_['rmse']) <= 5  # max_iter
    
    # Check that metrics are non-negative
    for metric in imputer.convergence_history_.values():
        assert all(x >= 0 for x in metric)


def test_parallel_processing(sample_data):
    """Test parallel processing capabilities."""
    # Test single process
    imputer_single = RandomForestImputer(
        n_estimators=10,
        n_jobs=1,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed_single = imputer_single.fit_transform(sample_data)
    
    # Test multiple processes
    imputer_multi = RandomForestImputer(
        n_estimators=10,
        n_jobs=-1,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed_multi = imputer_multi.fit_transform(sample_data)
    
    # Check that results are identical
    assert X_imputed_single.equals(X_imputed_multi)


def test_warm_start(sample_data):
    """Test warm start functionality."""
    # First fit with fewer trees
    imputer = RandomForestImputer(
        n_estimators=10,
        warm_start=True,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed1 = imputer.fit_transform(sample_data)
    
    # Second fit with more trees
    imputer.n_estimators = 20
    X_imputed2 = imputer.fit_transform(sample_data)
    
    # Check that results are different (due to more trees)
    assert not X_imputed1.equals(X_imputed2)


def test_oob_score(sample_data):
    """Test out-of-bag score computation."""
    imputer = RandomForestImputer(
        n_estimators=100,
        oob_score=True,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed = imputer.fit_transform(sample_data)
    
    # Check that OOB scores are computed for numeric columns
    for col in ['numeric1', 'numeric2']:
        if col in imputer.estimators_:
            assert hasattr(imputer.estimators_[col], 'oob_score_')


def test_class_weight(sample_data):
    """Test class weight handling for categorical variables."""
    imputer = RandomForestImputer(
        n_estimators=10,
        class_weight='balanced',
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed = imputer.fit_transform(sample_data)
    
    # Check that categorical columns are properly imputed
    assert not X_imputed.isna().any().any()
    assert all(X_imputed[col].isin(sample_data[col].dropna().unique()) 
              for col in ['categorical1', 'categorical2'])


def test_ccp_alpha(sample_data):
    """Test cost complexity pruning."""
    imputer = RandomForestImputer(
        n_estimators=10,
        ccp_alpha=0.1,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed = imputer.fit_transform(sample_data)
    
    assert not X_imputed.isna().any().any()


def test_max_samples(sample_data):
    """Test max_samples parameter."""
    imputer = RandomForestImputer(
        n_estimators=10,
        max_samples=0.5,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed = imputer.fit_transform(sample_data)
    
    assert not X_imputed.isna().any().any()


def test_add_indicator(sample_data):
    """Test missing value indicator addition."""
    imputer = RandomForestImputer(
        n_estimators=10,
        add_indicator=True,
        random_state=42,
        categorical_features=['categorical1', 'categorical2']
    )
    X_imputed = imputer.fit_transform(sample_data)
    
    # Check that indicator columns are added
    indicator_cols = [f"{col}_missing" for col in sample_data.columns]
    assert all(col in X_imputed.columns for col in indicator_cols)
    
    # Check that indicators are binary
    for col in indicator_cols:
        assert X_imputed[col].isin([0, 1]).all()


def test_copy_parameter(sample_data):
    """Test copy parameter behavior."""
    # Test with copy=True
    imputer = RandomForestImputer(copy=True)
    X_imputed = imputer.fit_transform(sample_data)
    assert not X_imputed.equals(sample_data)
    
    # Test with copy=False
    imputer = RandomForestImputer(copy=False)
    X_imputed = imputer.fit_transform(sample_data)
    assert X_imputed.equals(sample_data)


def test_get_feature_names_out(sample_data):
    """Test get_feature_names_out method."""
    imputer = RandomForestImputer(add_indicator=True)
    imputer.fit(sample_data)
    
    # Test with default input_features
    feature_names = imputer.get_feature_names_out()
    assert len(feature_names) == len(sample_data.columns) * 2  # Original + indicator columns
    
    # Test with custom input_features
    custom_features = ['custom1', 'custom2']
    feature_names = imputer.get_feature_names_out(custom_features)
    assert len(feature_names) == len(custom_features) * 2
    
    # Test without fitting
    imputer = RandomForestImputer()
    with pytest.raises(NotFittedError):
        imputer.get_feature_names_out() 