"""Tests for gradient boosting based imputers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from implyo import XGBoostImputer, LightGBMImputer


@pytest.fixture
def mixed_data():
    """Create a mixed-type dataset with missing values."""
    np.random.seed(42)
    n_samples = 100
    
    # Create numeric features
    numeric_data = pd.DataFrame(
        np.random.randn(n_samples, 3),
        columns=['numeric1', 'numeric2', 'numeric3']
    )
    
    # Create categorical features
    categorical_data = pd.DataFrame({
        'categorical1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'binary': np.random.choice([0, 1], n_samples)
    })
    
    # Combine features
    data = pd.concat([numeric_data, categorical_data], axis=1)
    
    # Introduce missing values
    for col in data.columns:
        mask = np.random.random(n_samples) < 0.2
        data.loc[mask, col] = np.nan
        
    return data


@pytest.fixture
def regression_data():
    """Create a regression dataset with missing values."""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        noise=0.1,
        random_state=42
    )
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    data['target'] = y
    
    # Introduce missing values
    for col in data.columns:
        mask = np.random.random(100) < 0.2
        data.loc[mask, col] = np.nan
        
    return data


@pytest.fixture
def classification_data():
    """Create a classification dataset with missing values."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_classes=3,
        random_state=42
    )
    data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    data['target'] = y
    
    # Introduce missing values
    for col in data.columns:
        mask = np.random.random(100) < 0.2
        data.loc[mask, col] = np.nan
        
    return data


class TestXGBoostImputer:
    """Test suite for XGBoostImputer."""
    
    def test_basic_imputation(self, mixed_data):
        """Test basic imputation functionality."""
        imputer = XGBoostImputer(
            n_estimators=50,
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        
        # Fit and transform
        X_imputed = imputer.fit_transform(mixed_data)
        
        # Check no missing values
        assert not X_imputed.isna().any().any()
        
        # Check data types preserved
        assert X_imputed['categorical1'].dtype == 'object'
        assert X_imputed['binary'].dtype in ['int64', 'int32']
        
        # Check feature importances
        assert len(imputer.feature_importances_) > 0
        
    def test_edge_cases(self):
        """Test various edge cases."""
        # Empty DataFrame
        imputer = XGBoostImputer()
        with pytest.raises(ValueError):
            imputer.fit_transform(pd.DataFrame())
            
        # DataFrame with no numeric columns
        data = pd.DataFrame({'cat': ['A', 'B', 'C']})
        with pytest.raises(ValueError):
            imputer.fit_transform(data)
            
        # DataFrame with all NaN
        data = pd.DataFrame({'col': [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError):
            imputer.fit_transform(data)
            
    def test_algorithm_selection(self, mixed_data):
        """Test different tree methods and boosters."""
        # Test different tree methods
        for tree_method in ['auto', 'exact', 'approx', 'hist']:
            imputer = XGBoostImputer(
                tree_method=tree_method,
                categorical_features=['categorical1', 'categorical2', 'binary'],
                random_state=42
            )
            X_imputed = imputer.fit_transform(mixed_data)
            assert not X_imputed.isna().any().any()
            
        # Test different boosters
        for booster in ['gbtree', 'gblinear', 'dart']:
            imputer = XGBoostImputer(
                booster=booster,
                categorical_features=['categorical1', 'categorical2', 'binary'],
                random_state=42
            )
            X_imputed = imputer.fit_transform(mixed_data)
            assert not X_imputed.isna().any().any()
            
    def test_initial_strategy(self, mixed_data):
        """Test different initial imputation strategies."""
        strategies = ['mean', 'median', 'most_frequent', 'constant']
        for strategy in strategies:
            imputer = XGBoostImputer(
                initial_strategy=strategy,
                categorical_features=['categorical1', 'categorical2', 'binary'],
                random_state=42
            )
            X_imputed = imputer.fit_transform(mixed_data)
            assert not X_imputed.isna().any().any()
            
    def test_column_selection(self, mixed_data):
        """Test imputation with specific column selection."""
        # Test with specific columns
        columns = ['numeric1', 'categorical1']
        imputer = XGBoostImputer(
            columns=columns,
            categorical_features=['categorical1'],
            random_state=42
        )
        X_imputed = imputer.fit_transform(mixed_data)
        assert not X_imputed[columns].isna().any().any()
        
        # Test with non-existent column
        with pytest.warns(UserWarning):
            imputer = XGBoostImputer(columns=['non_existent'])
            imputer.fit_transform(mixed_data)
            
    def test_large_dataset(self):
        """Test performance and correctness on larger dataset."""
        # Create larger dataset
        n_samples = 1000
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        
        # Introduce missing values
        for col in data.columns:
            mask = np.random.random(n_samples) < 0.2
            data.loc[mask, col] = np.nan
            
        imputer = XGBoostImputer(n_estimators=50, random_state=42)
        X_imputed = imputer.fit_transform(data)
        assert not X_imputed.isna().any().any()
        
    def test_transform_without_fit(self, mixed_data):
        """Test that transform without fit raises error."""
        imputer = XGBoostImputer()
        with pytest.raises(ValueError):
            imputer.transform(mixed_data)
            
    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        with pytest.raises(ValueError):
            XGBoostImputer(n_estimators=0)
        with pytest.raises(ValueError):
            XGBoostImputer(max_iter=0)
        with pytest.raises(ValueError):
            XGBoostImputer(tol=0)
        with pytest.raises(ValueError):
            XGBoostImputer(min_samples=1)
        with pytest.raises(ValueError):
            XGBoostImputer(uncertainty_quantile=2)
            
    def test_missing_value_patterns(self, mixed_data):
        """Test different patterns of missing values."""
        # Test with column of all NaN
        data = mixed_data.copy()
        data['all_nan'] = np.nan
        imputer = XGBoostImputer(
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        with pytest.raises(ValueError):
            imputer.fit_transform(data)
            
        # Test with row of all NaN
        data = mixed_data.copy()
        data.iloc[0, :] = np.nan
        imputer = XGBoostImputer(
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        X_imputed = imputer.fit_transform(data)
        assert not X_imputed.isna().any().any()
        
    def test_consistency(self, mixed_data):
        """Test that imputation is consistent across multiple calls."""
        imputer = XGBoostImputer(
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        X_imputed1 = imputer.fit_transform(mixed_data)
        X_imputed2 = imputer.transform(mixed_data)
        pd.testing.assert_frame_equal(X_imputed1, X_imputed2)
        
    def test_mixed_dtypes(self):
        """Test handling of mixed data types."""
        data = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5],
            'categorical': ['A', 'B', 'C', np.nan, 'E'],
            'boolean': [True, False, True, np.nan, False],
            'datetime': pd.date_range('2020-01-01', periods=5)
        })
        data.loc[2, 'datetime'] = np.nan
        
        imputer = XGBoostImputer(
            categorical_features=['categorical', 'boolean'],
            random_state=42
        )
        X_imputed = imputer.fit_transform(data)
        assert not X_imputed.isna().any().any()
        
    def test_high_dimensionality(self):
        """Test behavior with high-dimensional data."""
        n_samples = 100
        n_features = 100
        X = np.random.randn(n_samples, n_features)
        data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        
        # Introduce missing values
        for col in data.columns:
            mask = np.random.random(n_samples) < 0.2
            data.loc[mask, col] = np.nan
            
        imputer = XGBoostImputer(
            n_estimators=50,
            max_depth=3,  # Limit depth for high-dimensional data
            random_state=42
        )
        X_imputed = imputer.fit_transform(data)
        assert not X_imputed.isna().any().any()
        
    def test_sparse_missing(self, mixed_data):
        """Test with sparse missing value patterns."""
        # Create sparse missing pattern
        data = mixed_data.copy()
        for i in range(len(data)):
            if i % 3 == 0:  # Every third row has missing values
                data.iloc[i, :] = np.nan
                
        imputer = XGBoostImputer(
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        X_imputed = imputer.fit_transform(data)
        assert not X_imputed.isna().any().any()
        
    def test_uncertainty_quantification(self, mixed_data):
        """Test uncertainty quantification."""
        imputer = XGBoostImputer(
            uncertainty_quantile=0.95,
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        X_imputed = imputer.fit_transform(mixed_data)
        
        # Check uncertainty intervals
        assert imputer.uncertainty_intervals_ is not None
        for col in imputer.uncertainty_intervals_:
            lower, upper = imputer.uncertainty_intervals_[col]
            assert np.all(lower <= upper)
            
    def test_feature_importances(self, mixed_data):
        """Test feature importance computation."""
        imputer = XGBoostImputer(
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        imputer.fit_transform(mixed_data)
        
        # Check feature importances
        assert len(imputer.feature_importances_) > 0
        for col, importance in imputer.feature_importances_.items():
            assert np.all(importance >= 0)
            assert np.all(importance <= 1)
            
    def test_convergence_history(self, mixed_data):
        """Test convergence history tracking."""
        imputer = XGBoostImputer(
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        imputer.fit_transform(mixed_data)
        
        # Check convergence history
        assert len(imputer.convergence_history_) > 0
        for metric in imputer.convergence_history_.values():
            assert len(metric) > 0
            assert np.all(np.array(metric) >= 0)
            
    def test_parallel_processing(self, mixed_data):
        """Test parallel processing capabilities."""
        # Test with single process
        imputer = XGBoostImputer(
            n_jobs=1,
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        X_imputed1 = imputer.fit_transform(mixed_data)
        
        # Test with multiple processes
        imputer = XGBoostImputer(
            n_jobs=-1,
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        X_imputed2 = imputer.fit_transform(mixed_data)
        
        # Results should be similar (not exactly equal due to parallel processing)
        pd.testing.assert_frame_equal(X_imputed1, X_imputed2, check_exact=False, rtol=1e-3)
        
    def test_warm_start(self, mixed_data):
        """Test warm start functionality."""
        # First fit with fewer trees
        imputer = XGBoostImputer(
            n_estimators=50,
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        imputer.fit_transform(mixed_data)
        
        # Then increase number of trees
        imputer.n_estimators = 100
        X_imputed = imputer.fit_transform(mixed_data)
        assert not X_imputed.isna().any().any()
        
    def test_out_of_bag_score(self, regression_data):
        """Test out-of-bag score computation."""
        imputer = XGBoostImputer(
            subsample=0.8,  # Enable OOB
            random_state=42
        )
        imputer.fit_transform(regression_data)
        
        # Check that OOB scores are computed for numeric columns
        for col in regression_data.select_dtypes(include=[np.number]).columns:
            if col in imputer.estimators_:
                assert hasattr(imputer.estimators_[col], 'oob_score_')
                
    def test_class_weight_handling(self, classification_data):
        """Test handling of class weights for categorical variables."""
        imputer = XGBoostImputer(
            categorical_features=['target'],
            class_weight='balanced',
            random_state=42
        )
        X_imputed = imputer.fit_transform(classification_data)
        assert not X_imputed.isna().any().any()
        
    def test_missing_value_indicator(self, mixed_data):
        """Test missing value indicator functionality."""
        imputer = XGBoostImputer(
            add_indicator=True,
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        X_imputed = imputer.fit_transform(mixed_data)
        
        # Check that indicators are added and are binary
        indicator_cols = [col for col in X_imputed.columns if col.endswith('_missing')]
        assert len(indicator_cols) > 0
        for col in indicator_cols:
            assert X_imputed[col].isin([0, 1]).all()
            
    def test_copy_parameter(self, mixed_data):
        """Test copy parameter behavior."""
        # Test with copy=True
        imputer = XGBoostImputer(copy=True, random_state=42)
        X_imputed = imputer.fit_transform(mixed_data)
        assert not X_imputed.isna().any().any()
        assert not mixed_data.equals(X_imputed)  # Should be different objects
        
        # Test with copy=False
        imputer = XGBoostImputer(copy=False, random_state=42)
        X_imputed = imputer.fit_transform(mixed_data)
        assert not X_imputed.isna().any().any()
        assert mixed_data.equals(X_imputed)  # Should be same object
        
    def test_get_feature_names_out(self, mixed_data):
        """Test get_feature_names_out method."""
        imputer = XGBoostImputer(
            add_indicator=True,
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        imputer.fit(mixed_data)
        
        # Get feature names
        feature_names = imputer.get_feature_names_out()
        assert len(feature_names) == len(mixed_data.columns) + len(
            [col for col in mixed_data.columns if mixed_data[col].isna().any()]
        )


class TestLightGBMImputer:
    """Test suite for LightGBMImputer."""
    
    def test_basic_imputation(self, mixed_data):
        """Test basic imputation functionality."""
        imputer = LightGBMImputer(
            n_estimators=50,
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        
        # Fit and transform
        X_imputed = imputer.fit_transform(mixed_data)
        
        # Check no missing values
        assert not X_imputed.isna().any().any()
        
        # Check data types preserved
        assert X_imputed['categorical1'].dtype == 'object'
        assert X_imputed['binary'].dtype in ['int64', 'int32']
        
        # Check feature importances
        assert len(imputer.feature_importances_) > 0
        
    def test_edge_cases(self):
        """Test various edge cases."""
        # Empty DataFrame
        imputer = LightGBMImputer()
        with pytest.raises(ValueError):
            imputer.fit_transform(pd.DataFrame())
            
        # DataFrame with no numeric columns
        data = pd.DataFrame({'cat': ['A', 'B', 'C']})
        with pytest.raises(ValueError):
            imputer.fit_transform(data)
            
        # DataFrame with all NaN
        data = pd.DataFrame({'col': [np.nan, np.nan, np.nan]})
        with pytest.raises(ValueError):
            imputer.fit_transform(data)
            
    def test_boosting_type(self, mixed_data):
        """Test different boosting types."""
        for boosting_type in ['gbdt', 'rf', 'dart', 'goss']:
            imputer = LightGBMImputer(
                boosting_type=boosting_type,
                categorical_features=['categorical1', 'categorical2', 'binary'],
                random_state=42
            )
            X_imputed = imputer.fit_transform(mixed_data)
            assert not X_imputed.isna().any().any()
            
    def test_uncertainty_quantification(self, mixed_data):
        """Test uncertainty quantification."""
        imputer = LightGBMImputer(
            uncertainty_quantile=0.95,
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        X_imputed = imputer.fit_transform(mixed_data)
        
        # Check uncertainty intervals
        assert imputer.uncertainty_intervals_ is not None
        for col in imputer.uncertainty_intervals_:
            lower, upper = imputer.uncertainty_intervals_[col]
            assert np.all(lower <= upper)
            
    def test_parallel_processing(self, mixed_data):
        """Test parallel processing capabilities."""
        # Test with single process
        imputer = LightGBMImputer(
            n_jobs=1,
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        X_imputed1 = imputer.fit_transform(mixed_data)
        
        # Test with multiple processes
        imputer = LightGBMImputer(
            n_jobs=-1,
            categorical_features=['categorical1', 'categorical2', 'binary'],
            random_state=42
        )
        X_imputed2 = imputer.fit_transform(mixed_data)
        
        # Results should be similar (not exactly equal due to parallel processing)
        pd.testing.assert_frame_equal(X_imputed1, X_imputed2, check_exact=False, rtol=1e-3) 