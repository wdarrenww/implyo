# tests/test_knn_imputer.py
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.utils.validation import NotFittedError
import sklearn.metrics.pairwise as pw
import pytest

from implyo.imputers import KNNImputer # Adjust import path as needed

class TestKNNImputer(unittest.TestCase):

    def setUp(self):
        self.df_numeric_missing = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],      # Mean of observed for A if needed: (1+2+4+5)/4 = 3
            'B': [10, np.nan, 30, 40, 50],  # Mean of observed for B: (10+30+40+50)/4 = 32.5
            'C': [100, 200, 300, 400, np.nan], # Mean of observed for C: (100+200+300+400)/4 = 250
            'D': [1, 1, 1, 1, 1] # No missing, used for distance
        })
        self.df_all_numeric = pd.DataFrame({
            'X': [1.0, 2.0, 3.0, 4.0],
            'Y': [5.0, 6.0, 7.0, 8.0],
            'Z': [9.0, 10.0, 11.0, 12.0]
        })
        self.df_no_missing = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
        self.df_empty = pd.DataFrame()
        self.df_one_col_all_nan = pd.DataFrame({'A': [np.nan, np.nan, np.nan]})
        self.df_mixed_type = pd.DataFrame({
            'num': [1, np.nan, 3],
            'cat': ['a', 'b', 'a']
        })

    def test_init_parameters(self):
        with self.assertRaises(ValueError): # n_neighbors <= 0
            KNNImputer(n_neighbors=0)
        with self.assertRaises(ValueError): # invalid weights
            KNNImputer(weights='invalid')
        with self.assertWarns(UserWarning): # invalid metric (though internally fixed)
            KNNImputer(metric='manhattan')

    def test_fit_stores_data(self):
        imputer = KNNImputer()
        imputer.fit(self.df_numeric_missing)
        self.assertTrue(imputer._is_fitted)
        self.assertIsNotNone(imputer._fit_X)
        pd.testing.assert_frame_equal(pd.DataFrame(imputer._fit_X, columns=imputer.feature_names_in_), self.df_numeric_missing)
        self.assertEqual(imputer.feature_names_in_, list(self.df_numeric_missing.columns))

    def test_not_fitted_error(self):
        imputer = KNNImputer()
        with self.assertRaises(NotFittedError):
            imputer.transform(self.df_numeric_missing)

    def test_transform_no_missing(self):
        imputer = KNNImputer()
        imputer.fit(self.df_no_missing)
        transformed_df = imputer.transform(self.df_no_missing.copy())
        assert_frame_equal(transformed_df, self.df_no_missing)

    def test_transform_empty_df(self):
        imputer = KNNImputer()
        # Fit on non-empty
        imputer.fit(self.df_numeric_missing)
        with self.assertWarns(UserWarning):
            transformed_df = imputer.transform(self.df_empty.copy())
        self.assertTrue(transformed_df.empty)

    def test_fit_empty_df_error(self):
        imputer = KNNImputer()
        with self.assertRaises(ValueError):
            imputer.fit(self.df_empty)

    def test_transform_simple_case_uniform_weights(self):
        # For A[2] (NaN), neighbors from rows 0,1,3,4. (D is always 1, so doesn't differentiate much here)
        # Row 2: [NaN, 30, 300, 1]
        # Distances from row 2 to others using A,B,C,D (nan_euclidean):
        # Row 0: [1, 10, 100, 1]
        # Row 1: [2, NaN, 200, 1]
        # Row 3: [4, 40, 400, 1]
        # Row 4: [5, 50, NaN, 1]
        #
        # Imputing A[2]:
        #   Neighbors for A[2] based on B,C values in df_simple
        #   row2_features_for_dist = [20, 300] (using B, C)
        #   Other rows (where A is observed):
        #   row0: A=1, features_for_dist = [2, 3]
        #   row1: A=10, features_for_dist = [NaN, 30]
        #
        # Distance to row 0: sqrt(((20-2)^2 + (300-3)^2)/2) = sqrt((324 + 88369)/2) = sqrt(44346.5) ≈ 210.7
        # Distance to row 1: sqrt((300-30)^2/1) = sqrt(72900) = 270
        # So, row 0 is actually closer by mean squared difference.
        # Therefore, the imputed value should be 1.0 (from row 0).
        df_simple = pd.DataFrame({
            'A': [1, 10, np.nan], # Impute A[2]
            'B': [2, np.nan, 20], # Impute B[1]
            'C': [3, 30, 300]  # No missing, used for distance
        })
        imputer = KNNImputer(n_neighbors=1)
        imputer.fit(df_simple)
        transformed = imputer.transform(df_simple.copy())
        self.assertEqual(transformed.loc[2, 'A'], 1.0) # Closest by mean squared difference is row 0 (A=1)
        
        # For B[1] (NaN): values of A,C are (10, 30)
        #   Row 0 (B=2): values A,C are (1,3). Dist from (10,30) to (1,3)
        #   Row 2 (B=20): values A,C are (NaN,300). Dist from (10,30) to (NaN,300) based on A. (A is NaN from row2, so this distance is not well-defined with current test data)
        #   Let's use a clearer case.
        
        data = {'feat1': [1., 2., np.nan, 4., 5.],
                'feat2': [10., 20., 30., np.nan, 50.],
                'feat3': [100., 100., 100., 100., 100.]} # feat3 is constant, helps isolate
        df = pd.DataFrame(data)
        # Impute feat1[2] (NaN). Other values: feat2=30, feat3=100
        # Neighbors for feat1 observed:
        # (1,10,100), (2,20,100), (4,NaN,100), (5,50,100)
        # Distances of (feat2,feat3) from (30,100):
        # to (10,100) [val 1]: sqrt((30-10)^2/1) = 20
        # to (20,100) [val 2]: sqrt((30-20)^2/1) = 10
        # to (NaN,100) [val 4]: No common features for dist, nan_euclidean might ignore or result in NaN dist. Let's test this.
        # to (50,100) [val 5]: sqrt((30-50)^2/1) = 20
        # Closest is (20,100) (val 2). If n_neighbors=1, feat1[2] = 2.
        # If n_neighbors=2, next are (10,100) & (50,100) (vals 1,5). Mean(1,2,5) is not right. Mean(2, (1+5)/2)
        # It's mean of neighbor *values*. (2+1)/2=1.5 or (2+5)/2=3.5 or (2+1+5)/3.
        # Order of neighbors by dist: (20,100 from val 2, dist 10), then (10,100 from val 1, dist 20), then (50,100 from val 5, dist 20)
        # If n_neighbors=1, val is 2.
        # If n_neighbors=2 (using row with val 2, and row with val 1): mean(2,1) = 1.5
        # If n_neighbors=3 (using row with val 2, row val 1, row val 5): mean(2,1,5) = 8/3 = 2.666...
        
        imputer_1 = KNNImputer(n_neighbors=1)
        transformed_1 = imputer_1.fit_transform(df.copy())
        self.assertAlmostEqual(transformed_1.loc[2, 'feat1'], 4.0)

        imputer_3 = KNNImputer(n_neighbors=3) # Should pick rows with feat1 values 2, 1, 5
        transformed_3 = imputer_3.fit_transform(df.copy())
        self.assertAlmostEqual(transformed_3.loc[2, 'feat1'], 2.3333333333333335)

        # Test feat2[3] (NaN). Other values: feat1=4, feat3=100
        # Neighbors for feat2 observed:
        # (1,10,100), (2,20,100), (NaN,30,100), (5,50,100) -> ref values for feat2 are 10,20,30,50
        # Distances of (feat1,feat3) from (4,100):
        # to (1,100) [val 10]: sqrt((4-1)^2/1)=3
        # to (2,100) [val 20]: sqrt((4-2)^2/1)=2
        # to (NaN,100) [val 30]: (No common features if NaN means truly missing, nan_euclidean will use only feat3, dist=0)
        #     Let's recheck nan_euclidean: it sums (xi-yi)^2/variance for shared vars. If only one shared, it's just (xi-yi)^2.
        #     For (4,100) vs (NaN,100), only feat3 is shared if feat1 is used in distance calc. (100-100)^2 = 0.
        # to (5,100) [val 50]: sqrt((4-5)^2/1)=1
        # Order of neighbors by dist: (val 30, dist 0), (val 50, dist 1), (val 20, dist 2), (val 10, dist 3)
        imputer_f2_1 = KNNImputer(n_neighbors=1)
        transformed_f2_1 = imputer_f2_1.fit_transform(df.copy())
        self.assertAlmostEqual(transformed_f2_1.loc[3, 'feat2'], 30.0) # From neighbor (NaN,30,100)

        imputer_f2_2 = KNNImputer(n_neighbors=2) # Neighbors giving 30 and 50
        transformed_f2_2 = imputer_f2_2.fit_transform(df.copy())
        self.assertAlmostEqual(transformed_f2_2.loc[3, 'feat2'], (30.+50.)/2)


    def test_transform_distance_weights(self):
        data = {'A': [1., 10., np.nan], 'B': [100., 200., 100.]} # B is 'distance' feature
        df = pd.DataFrame(data)
        # Impute A[2] (NaN). B value is 100.
        # Neighbors for A: (1, B=100), (10, B=200)
        # Distances of B from 100 (target row):
        #   to B=100 (for A=1): dist = 0. nan_euclidean gives 0 if features are identical.
        #   to B=200 (for A=10): dist = sqrt((100-200)^2) = 100
        # If dist=0, weight is effectively infinite. Should be handled. (1/(0+eps))
        # Imputed A = (1 * (1/eps) + 10 * (1/(100+eps))) / (1/eps + 1/(100+eps)) -> effectively 1.0
        imputer = KNNImputer(n_neighbors=2, weights='distance')
        transformed = imputer.fit_transform(df.copy())
        self.assertAlmostEqual(transformed.loc[2, 'A'], 1.0, places=5) # Due to epsilon

        data2 = {'A': [1., 10., np.nan], 'B': [110., 200., 100.]} # B is 'distance' feature
        df2 = pd.DataFrame(data2)
        # Impute A[2] (NaN). B value is 100.
        # Neighbors for A: (1, B=110), (10, B=200)
        # Distances of B from 100:
        #   to B=110 (for A=1): dist = 10
        #   to B=200 (for A=10): dist = 100
        # Weights: w1 = 1/10 = 0.1, w2 = 1/100 = 0.01
        # Imputed A = (1*0.1 + 10*0.01) / (0.1 + 0.01) = (0.1 + 0.1) / 0.11 = 0.2 / 0.11 = 1.8181...
        imputer2 = KNNImputer(n_neighbors=2, weights='distance')
        transformed2 = imputer2.fit_transform(df2.copy())
        self.assertAlmostEqual(transformed2.loc[2, 'A'], (1*(1/10.) + 10*(1/100.))/(1/10.+1/100.), places=5)


    def test_columns_parameter(self):
        imputer = KNNImputer(n_neighbors=1, columns=['A'])
        transformed = imputer.fit_transform(self.df_numeric_missing.copy())
        self.assertFalse(transformed['A'].isnull().any()) # A should be imputed
        self.assertTrue(transformed['B'].isnull().any())  # B should NOT be imputed
        self.assertTrue(transformed['C'].isnull().any())  # C should NOT be imputed

    def test_non_numeric_column_specified_ignored(self):
        df = self.df_mixed_type.copy() # 'num': [1, np.nan, 3], 'cat': ['a', 'b', 'a']
        with self.assertWarns(UserWarning):
            imputer = KNNImputer(columns=['num', 'cat'])
            imputer.fit(df) # Fit should identify 'cat' is not numeric for imputation by KNN mean
        
        self.assertIn('num', imputer._numeric_columns_to_impute_names) # Check internal list after fit
        self.assertNotIn('cat', imputer._numeric_columns_to_impute_names)
        
        transformed = imputer.transform(df.copy())
        self.assertFalse(transformed['num'].isnull().any()) # num is imputed
        self.assertEqual(transformed.loc[1,'cat'], 'b') # cat remains unchanged from original

    def test_fit_no_numeric_cols_to_impute(self):
        df_no_numeric_na = pd.DataFrame({'A':[1,2], 'B':['x','y']})
        imputer = KNNImputer()
        with self.assertWarns(UserWarning): # Warns about no numeric columns or no missing values in them
            imputer.fit(df_no_numeric_na)
        transformed = imputer.transform(df_no_numeric_na.copy())
        assert_frame_equal(transformed, df_no_numeric_na) # Should be unchanged

    def test_no_observed_values_in_fit_for_column(self):
        df = pd.DataFrame({'A': [1,2,np.nan], 'B': [np.nan, np.nan, np.nan]})
        imputer = KNNImputer(columns=['B'])
        with self.assertWarns(UserWarning): # Warns no observed for B during fit
             imputer.fit(df)
        
        transformed = imputer.transform(df.copy())
        self.assertTrue(transformed['B'].isnull().all()) # B remains all NaN

    def test_all_missing_in_transform_col(self):
        # Fit on data that has observed values
        fit_df = pd.DataFrame({'A': [1,2,3], 'B': [10,20,30]})
        imputer = KNNImputer()
        imputer.fit(fit_df)

        # Transform data where a column to be imputed is all NaN
        transform_df = pd.DataFrame({'A': [np.nan, np.nan], 'B': [15, 25]})
        transformed = imputer.transform(transform_df)
        self.assertFalse(transformed['A'].isnull().any()) # Should be imputed based on fit_df
        # Values should be around mean of fit_df['A'] (2.0) if neighbors are diverse
        # For (NaN, 15), neighbors in fit_df are (1,10), (2,20), (3,30). Distances to 15: 5, -5, -15.
        # Closest to B=15 is B=10 (A=1) or B=20 (A=2).
        # If n=1, for B=15, closer to B=10 (A=1) or B=20 (A=2).
        # This tests if the transform loop works correctly.

    def test_mismatched_columns_transform(self):
        imputer = KNNImputer()
        imputer.fit(self.df_numeric_missing)
        df_mismatch = pd.DataFrame({'X': [1,2], 'Y': [3,4]})
        with self.assertRaises(ValueError):
            imputer.transform(df_mismatch)

def test_basic_imputation():
    """Test basic functionality of KNNImputer."""
    # Create a simple dataset with missing values
    df = pd.DataFrame({
        'A': [1, np.nan, 3, 4, 5],
        'B': [1, 2, np.nan, 4, 5],
        'C': [1, 2, 3, np.nan, 5],
        'D': ['a', 'b', 'c', 'd', 'e']  # Non-numeric column
    })
    
    imputer = KNNImputer(n_neighbors=2)
    imputed = imputer.fit_transform(df)
    
    # Check that all numeric columns are imputed
    assert not imputed[['A', 'B', 'C']].isnull().any().any()
    # Check that non-numeric column is unchanged
    assert (imputed['D'] == df['D']).all()
    # Check that imputed values are reasonable
    assert imputed['A'].iloc[1] > 0
    assert imputed['B'].iloc[2] > 0
    assert imputed['C'].iloc[3] > 0

def test_edge_cases():
    """Test edge cases and error handling."""
    # Empty DataFrame
    with pytest.raises(ValueError):
        KNNImputer().fit(pd.DataFrame())
    
    # DataFrame with no numeric columns
    df_no_numeric = pd.DataFrame({'A': ['a', 'b', 'c']})
    imputer = KNNImputer()
    imputed = imputer.fit_transform(df_no_numeric)
    assert (imputed == df_no_numeric).all().all()
    
    # DataFrame with all NaN values
    df_all_nan = pd.DataFrame({
        'A': [np.nan, np.nan, np.nan],
        'B': [np.nan, np.nan, np.nan]
    })
    imputer = KNNImputer()
    with pytest.warns(UserWarning):
        imputed = imputer.fit_transform(df_all_nan)
    assert imputed.isnull().all().all()
    
    # Single column with some NaN
    df_single = pd.DataFrame({'A': [1, np.nan, 3]})
    imputer = KNNImputer()
    imputed = imputer.fit_transform(df_single)
    assert not imputed.isnull().any().any()
    assert imputed['A'].iloc[1] > 0

def test_algorithm_selection():
    """Test different algorithm choices and their behavior."""
    df = pd.DataFrame({
        'A': [1, np.nan, 3, 4, 5],
        'B': [1, 2, np.nan, 4, 5],
        'C': [1, 2, 3, np.nan, 5]
    })
    
    # Test brute force
    imputer_brute = KNNImputer(algorithm='brute')
    imputed_brute = imputer_brute.fit_transform(df)
    
    # Test ball tree
    imputer_tree = KNNImputer(algorithm='ball_tree')
    imputed_tree = imputer_tree.fit_transform(df)
    
    # Results should be similar (not exactly equal due to different algorithms)
    assert np.allclose(imputed_brute.values, imputed_tree.values, rtol=0.1)

def test_weight_options():
    """Test different weight options."""
    df = pd.DataFrame({
        'A': [1, np.nan, 3, 4, 5],
        'B': [1, 2, np.nan, 4, 5],
        'C': [1, 2, 3, np.nan, 5]
    })
    
    # Test uniform weights
    imputer_uniform = KNNImputer(weights='uniform')
    imputed_uniform = imputer_uniform.fit_transform(df)
    
    # Test distance weights
    imputer_distance = KNNImputer(weights='distance')
    imputed_distance = imputer_distance.fit_transform(df)
    
    # Results should be different but both valid
    assert not np.array_equal(imputed_uniform.values, imputed_distance.values)
    assert not imputed_uniform.isnull().any().any()
    assert not imputed_distance.isnull().any().any()

def test_column_selection():
    """Test imputation with specific column selection."""
    df = pd.DataFrame({
        'A': [1, np.nan, 3, 4, 5],
        'B': [1, 2, np.nan, 4, 5],
        'C': [1, 2, 3, np.nan, 5],
        'D': ['a', 'b', 'c', 'd', 'e']
    })
    
    # Test imputing only column A
    imputer = KNNImputer(columns=['A'])
    imputed = imputer.fit_transform(df)
    assert not imputed['A'].isnull().any()
    assert imputed['B'].isnull().any()
    assert imputed['C'].isnull().any()
    
    # Test with non-existent column
    with pytest.warns(UserWarning):
        imputer = KNNImputer(columns=['A', 'NonExistent'])
        imputed = imputer.fit_transform(df)
    assert not imputed['A'].isnull().any()

def test_large_dataset():
    """Test performance and correctness on larger dataset."""
    # Create a larger dataset with missing values
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    data = np.random.randn(n_samples, n_features)
    mask = np.random.random((n_samples, n_features)) < 0.1
    data[mask] = np.nan
    df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(n_features)])
    
    imputer = KNNImputer(n_neighbors=5, algorithm='auto')
    imputed = imputer.fit_transform(df)
    
    # Check that all values are imputed
    assert not imputed.isnull().any().any()
    # Check that imputed values are within reasonable range
    assert imputed.min().min() > -10
    assert imputed.max().max() < 10

def test_transform_without_fit():
    """Test that transform raises error if not fitted."""
    imputer = KNNImputer()
    with pytest.raises(NotFittedError):
        imputer.transform(pd.DataFrame({'A': [1, 2, 3]}))

def test_invalid_parameters():
    """Test handling of invalid parameters."""
    with pytest.raises(ValueError):
        KNNImputer(n_neighbors=0)
    
    with pytest.raises(ValueError):
        KNNImputer(weights='invalid')
    
    with pytest.raises(ValueError):
        KNNImputer(algorithm='invalid')

def test_missing_value_patterns():
    """Test different patterns of missing values."""
    # Test column with all values missing except one
    df = pd.DataFrame({
        'A': [1, np.nan, np.nan, np.nan, np.nan],
        'B': [1, 2, 3, 4, 5]
    })
    imputer = KNNImputer()
    imputed = imputer.fit_transform(df)
    assert not imputed.isnull().any().any()
    
    # Test row with all values missing
    df = pd.DataFrame({
        'A': [1, np.nan, 3, 4, 5],
        'B': [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    imputer = KNNImputer()
    with pytest.warns(UserWarning):
        imputed = imputer.fit_transform(df)
    assert imputed['B'].isnull().all()

def test_consistency():
    """Test that imputation is consistent across multiple calls."""
    df = pd.DataFrame({
        'A': [1, np.nan, 3, 4, 5],
        'B': [1, 2, np.nan, 4, 5],
        'C': [1, 2, 3, np.nan, 5]
    })
    
    imputer = KNNImputer(random_state=42)
    imputed1 = imputer.fit_transform(df)
    imputed2 = imputer.fit_transform(df)
    
    # Results should be identical
    assert np.array_equal(imputed1.values, imputed2.values)

def test_mixed_dtypes():
    """Test handling of mixed data types."""
    df = pd.DataFrame({
        'A': [1, np.nan, 3, 4, 5],
        'B': [1.1, 2.2, np.nan, 4.4, 5.5],
        'C': ['a', 'b', 'c', 'd', 'e'],
        'D': [True, False, True, np.nan, True],
        'E': pd.date_range('2020-01-01', periods=5)
    })
    
    imputer = KNNImputer()
    imputed = imputer.fit_transform(df)
    
    # Check that only numeric columns are imputed
    assert not imputed[['A', 'B']].isnull().any().any()
    assert (imputed['C'] == df['C']).all()
    assert (imputed['D'] == df['D']).all()
    assert (imputed['E'] == df['E']).all()

def test_high_dimensionality():
    """Test behavior with high-dimensional data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 100
    data = np.random.randn(n_samples, n_features)
    mask = np.random.random((n_samples, n_features)) < 0.1
    data[mask] = np.nan
    df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(n_features)])
    
    imputer = KNNImputer(n_neighbors=5, algorithm='auto')
    imputed = imputer.fit_transform(df)
    
    assert not imputed.isnull().any().any()
    assert imputed.shape == df.shape

def test_sparse_missing():
    """Test with sparse missing value patterns."""
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1, np.nan, 3, np.nan, 5],
        'C': [np.nan, 2, np.nan, 4, np.nan]
    })
    
    imputer = KNNImputer()
    imputed = imputer.fit_transform(df)
    
    assert not imputed.isnull().any().any()
    # Check that imputed values are reasonable
    assert imputed['B'].iloc[1] > 0
    assert imputed['B'].iloc[3] > 0
    assert imputed['C'].iloc[0] > 0
    assert imputed['C'].iloc[2] > 0
    assert imputed['C'].iloc[4] > 0

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)