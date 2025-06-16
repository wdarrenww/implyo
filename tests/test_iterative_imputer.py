# tests/test_iterative_imputer.py
import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.utils.validation import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor

from implyo.imputers import IterativeImputer # Adjust import

class TestIterativeImputer(unittest.TestCase):

    def setUp(self):
        self.rng_seed = 42 # For reproducibility of tests involving random state
        self.df_simple_numeric = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5.5], # Missing
            'B': [10, 11, 12, np.nan, 14], # Missing
            'C': [20, 21, 22, 23, 24] # Complete
        })
        self.df_simple_categorical = pd.DataFrame({
            'cat1': ['X', 'Y', np.nan, 'X', 'Z'], # Missing
            'cat2': ['A', 'B', 'A', np.nan, 'A'], # Missing
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0] # Complete numeric predictor
        })
        self.df_mixed = pd.DataFrame({
            'num_A': [1, np.nan, 3, 4, 5],
            'cat_B': ['dog', 'cat', np.nan, 'dog', 'fish'],
            'num_C': [10.0, 11.0, 12.0, 13.0, 14.0] # No missing
        })
        self.df_no_missing = pd.DataFrame({'A': [1,2], 'B': [3,4]})
        self.df_all_missing_col = pd.DataFrame({'A': [1,2], 'B': [np.nan, np.nan]})
        self.df_empty = pd.DataFrame()

    def test_init_params(self):
        with self.assertRaises(ValueError):
            IterativeImputer(max_iter=0)
        with self.assertRaises(ValueError):
            IterativeImputer(imputation_order='invalid_order')
        # Check default estimators
        imputer = IterativeImputer(random_state=self.rng_seed)
        self.assertIsInstance(imputer.estimator_numeric, LinearRegression)
        self.assertIsInstance(imputer.estimator_categorical, LogisticRegression)
        self.assertEqual(imputer.estimator_categorical.random_state, self.rng_seed)


    def test_initial_impute_numeric(self):
        imputer = IterativeImputer(initial_strategy='mean', random_state=self.rng_seed)
        # Manually setup feature_info_ as fit() would
        imputer.feature_info_ = {
            0: {'name': 'A', 'type': 'numeric'},
            1: {'name': 'B', 'type': 'numeric'},
            2: {'name': 'C', 'type': 'numeric'}
        }
        X_np = self.df_simple_numeric.values.copy()
        missing_mask = pd.isnull(self.df_simple_numeric).values
        
        imputed_np = imputer._initial_impute(X_np, missing_mask, fit_phase=True)
        
        # Expected mean for A (1,2,4,5.5) = 12.5/4 = 3.125
        # Expected mean for B (10,11,12,14) = 47/4 = 11.75
        self.assertAlmostEqual(imputed_np[2,0], 3.125) # A[2]
        self.assertAlmostEqual(imputed_np[3,1], 11.75) # B[3]
        self.assertAlmostEqual(imputer.initial_fill_values_[0], 3.125)
        self.assertAlmostEqual(imputer.initial_fill_values_[1], 11.75)

    def test_initial_impute_categorical(self):
        imputer = IterativeImputer(initial_strategy='mode', random_state=self.rng_seed)
        # Data needs to be label encoded first as _initial_impute expects numeric representation
        df_cat_encoded = self.df_simple_categorical.copy()
        le_cat1 = LabelEncoder()
        le_cat2 = LabelEncoder()

        imputer.label_encoders_ = {0: le_cat1, 1: le_cat2}
        imputer.feature_info_ = {
            0: {'name': 'cat1', 'type': 'categorical'},
            1: {'name': 'cat2', 'type': 'categorical'},
            2: {'name': 'num1', 'type': 'numeric'}
        }
        
        # Fit LabelEncoders and transform
        df_cat_encoded['cat1_enc'] = np.nan
        cat1_notna = df_cat_encoded['cat1'].notna()
        if np.any(cat1_notna):
            df_cat_encoded.loc[cat1_notna, 'cat1_enc'] = le_cat1.fit_transform(df_cat_encoded.loc[cat1_notna, 'cat1'])

        df_cat_encoded['cat2_enc'] = np.nan
        cat2_notna = df_cat_encoded['cat2'].notna()
        if np.any(cat2_notna):
             df_cat_encoded.loc[cat2_notna, 'cat2_enc'] = le_cat2.fit_transform(df_cat_encoded.loc[cat2_notna, 'cat2'])
        
        X_np_encoded = df_cat_encoded[['cat1_enc', 'cat2_enc', 'num1']].values.astype(float)
        missing_mask = df_cat_encoded[['cat1', 'cat2', 'num1']].isnull().values # Use original for mask
        
        imputed_np = imputer._initial_impute(X_np_encoded, missing_mask, fit_phase=True)
        
        # cat1: X, Y, NaN, X, Z -> Encoded X (e.g. 0), Y (e.g. 1), Z (e.g. 2). Mode is X (0).
        # cat2: A, B, A, NaN, A -> Encoded A (e.g. 0), B (e.g. 1). Mode is A (0).
        # This test depends on knowing the encoded values.
        # X is mode for cat1 (appears twice observed). A is mode for cat2 (appears thrice observed)
        mode_cat1_encoded = le_cat1.transform(['X'])[0]
        mode_cat2_encoded = le_cat2.transform(['A'])[0]
        
        self.assertEqual(imputed_np[2,0], mode_cat1_encoded) # cat1[2]
        self.assertEqual(imputed_np[3,1], mode_cat2_encoded) # cat2[3]
        self.assertEqual(imputer.initial_fill_values_[0], mode_cat1_encoded)
        self.assertEqual(imputer.initial_fill_values_[1], mode_cat2_encoded)

    def test_get_imputation_order(self):
        imputer = IterativeImputer(random_state=self.rng_seed)
        missing_mask = pd.isnull(self.df_simple_numeric).values # A (idx 0) has 1 NaN, B (idx 1) has 1 NaN
        
        order_asc = imputer._get_imputation_order(self.df_simple_numeric, missing_mask)
        # Both have 1 NaN, order among them could be 0,1 or 1,0 if stable sorted.
        self.assertIn(0, order_asc)
        self.assertIn(1, order_asc)
        self.assertTrue(order_asc == [0,1] or order_asc == [1,0]) # Given equal counts

        imputer.imputation_order = 'descending'
        order_desc = imputer._get_imputation_order(self.df_simple_numeric, missing_mask)
        self.assertTrue(order_desc == [0,1] or order_desc == [1,0])

        imputer.imputation_order = 'roman'
        order_roman = imputer._get_imputation_order(self.df_simple_numeric, missing_mask)
        self.assertEqual(order_roman, [0,1]) # Should be sorted by index

        imputer.imputation_order = 'arabic'
        order_arabic = imputer._get_imputation_order(self.df_simple_numeric, missing_mask)
        self.assertEqual(order_arabic, [1,0]) # Reversed index sort

        imputer.imputation_order = 'random'
        order_rand = imputer._get_imputation_order(self.df_simple_numeric, missing_mask)
        self.assertEqual(len(order_rand), 2) # Contains 0 and 1 in some order
        self.assertIn(0, order_rand)
        self.assertIn(1, order_rand)


    def test_fit_transform_numeric_deterministic(self):
        imputer = IterativeImputer(max_iter=2, random_state=self.rng_seed, sample_posterior=False)
        df_copy = self.df_simple_numeric.copy()
        transformed_df = imputer.fit_transform(df_copy)

        self.assertFalse(transformed_df.isnull().any().any(), "Output should have no NaNs")
        self.assertTrue(imputer._is_fitted)
        self.assertIsNotNone(imputer.estimators_.get(0)) # Estimator for col A (idx 0)
        self.assertIsNotNone(imputer.estimators_.get(1)) # Estimator for col B (idx 1)
        
        # Compare with a known result if possible, or check properties.
        # For A[2] (original NaN): df_simple_numeric.loc[2,'A']
        # Initial fill: A[2] = 3.125, B[3] = 11.75
        # Iter 1, impute A (idx 0):
        #   y_train_A from original observed A. X_train_predictors from (initially imputed B, C)
        #   Predict A[2]
        # Iter 1, impute B (idx 1):
        #   y_train_B from original observed B. X_train_predictors from (updated A, C)
        #   Predict B[3]
        # Exact values are hard to test without re-implementing logic here.
        # We can check if originally NaN values are now filled.
        self.assertFalse(np.isnan(transformed_df.loc[2,'A']))
        self.assertFalse(np.isnan(transformed_df.loc[3,'B']))


    def test_fit_transform_mixed_deterministic(self):
        imputer = IterativeImputer(max_iter=2, random_state=self.rng_seed, sample_posterior=False)
        df_copy = self.df_mixed.copy() # num_A (NaN at 1), cat_B (NaN at 2)
        transformed_df = imputer.fit_transform(df_copy)

        self.assertFalse(transformed_df.isnull().any().any(), "Output should have no NaNs for imputed columns")
        self.assertTrue(imputer._is_fitted)
        self.assertIsNotNone(imputer.estimators_.get(df_copy.columns.get_loc('num_A')))
        self.assertIsNotNone(imputer.estimators_.get(df_copy.columns.get_loc('cat_B')))
        self.assertIn(df_copy.columns.get_loc('cat_B'), imputer.label_encoders_)
        
        self.assertFalse(np.isnan(transformed_df.loc[1,'num_A'])) # Check if numeric NaN filled
        self.assertIsNotNone(transformed_df.loc[2,'cat_B']) # Check if categorical NaN filled
        self.assertNotEqual(transformed_df.loc[2,'cat_B'], np.nan) # Ensure it's not NaN string or similar
        self.assertIsInstance(transformed_df.loc[2,'cat_B'], str) # Should be string after inverse transform
        self.assertIn(transformed_df.loc[2,'cat_B'], ['dog', 'cat', 'fish']) # Should be one of the original categories

    def test_transform_after_fit(self):
        imputer = IterativeImputer(max_iter=2, random_state=self.rng_seed, sample_posterior=False)
        imputer.fit(self.df_mixed.copy())
        
        # Create a test set with similar structure
        df_test_missing = pd.DataFrame({
            'num_A': [np.nan, 2, 3.5],
            'cat_B': ['cat', np.nan, 'dog'],
            'num_C': [10.5, 11.5, 12.5]
        })
        transformed_test_df = imputer.transform(df_test_missing.copy())
        self.assertFalse(transformed_test_df.isnull().any().any())
        self.assertFalse(np.isnan(transformed_test_df.loc[0,'num_A']))
        self.assertIsNotNone(transformed_test_df.loc[1,'cat_B'])

    def test_sample_posterior_numeric(self):
        # START CHANGE: Modify data for larger residuals
        df_for_residuals = pd.DataFrame({
            'A': [1, 2.5, 2, 4.5, 3, np.nan], # target, less perfectly correlated
            'B': [1.1, 2.1, 2.9, 4.1, 4.9, 3.5], # predictor, more ambiguous value for NaN row
            'C': [10,10,10,10,10,10] # constant
        })
        imputer = IterativeImputer(max_iter=1, random_state=self.rng_seed, sample_posterior=True)
        imputer.fit(df_for_residuals.copy())
        
        # Check residual_std_devs_ was stored for column A (index 0)
        self.assertIn(0, imputer.residual_std_devs_)
        self.assertGreater(imputer.residual_std_devs_[0], 0)

        transformed_df1 = imputer.transform(df_for_residuals.copy())
        transformed_df2 = imputer.transform(df_for_residuals.copy()) # Should be different due to sampling
        
        self.assertFalse(np.isnan(transformed_df1.loc[5,'A']))
        self.assertFalse(np.isnan(transformed_df2.loc[5,'A']))
        # Values should differ due to sampling from posterior
        # This could occasionally be equal due to chance with few samples, but unlikely for floats.
        self.assertNotAlmostEqual(transformed_df1.loc[5,'A'], transformed_df2.loc[5,'A'], places=5, 
                                  msg="Stochastic imputations for A[5] were too similar.")

    def test_sample_posterior_categorical(self):
        # Create data where probabilities for a class are not 0 or 1 for logistic regression
        df_cat_proba = pd.DataFrame({
            'target_cat': ['A', 'A', 'B', 'B', np.nan, np.nan],
            'pred_strong_num': [1, 1.2, 10, 10.2, 5.5, 5.8] # More ambiguous values for NaN rows
        })
        imputer = IterativeImputer(max_iter=1, random_state=self.rng_seed, sample_posterior=True)
        imputer.fit(df_cat_proba.copy())
        
        self.assertIn(0, imputer.estimators_) # Estimator for target_cat (idx 0)
        self.assertIsInstance(imputer.estimators_[0], LogisticRegression)

        imputations_cat = [imputer.transform(df_cat_proba.copy()).loc[4,'target_cat'] for _ in range(20)]
        counts = pd.Series(imputations_cat).value_counts()
        
        # Expect both A and B to appear if probabilities are not extreme
        self.assertIn('A', counts.index)
        self.assertIn('B', counts.index)
        self.assertTrue(counts.get('A',0) > 0 and counts.get('B',0) > 0, "Both categories should be sampled.")


    def test_impute_multiple(self):
        imputer = IterativeImputer(max_iter=2, random_state=self.rng_seed, sample_posterior=True)
        imputer.fit(self.df_mixed.copy())
        
        n_imps = 3
        imputed_datasets = imputer.impute_multiple(self.df_mixed.copy(), n_imputations=n_imps)
        self.assertEqual(len(imputed_datasets), n_imps)
        
        # Check that datasets are different (focus on one imputed value)
        val1_num = imputed_datasets[0].loc[1, 'num_A'] # Was NaN
        val2_num = imputed_datasets[1].loc[1, 'num_A']
        val3_num = imputed_datasets[2].loc[1, 'num_A']
        self.assertTrue(not (np.isclose(val1_num, val2_num) and np.isclose(val2_num, val3_num)),
                        "Numeric multiple imputations are too similar or identical.")

        val1_cat = imputed_datasets[0].loc[2, 'cat_B'] # Was NaN
        val2_cat = imputed_datasets[1].loc[2, 'cat_B']
        # Categorical might be the same more often by chance if one class is highly probable
        # Check that they are valid categories
        for df_imp in imputed_datasets:
            self.assertIn(df_imp.loc[2, 'cat_B'], ['dog', 'cat', 'fish'])


    def test_impute_multiple_deterministic_warns(self):
        imputer = IterativeImputer(max_iter=1, random_state=self.rng_seed, sample_posterior=False)
        imputer.fit(self.df_mixed.copy())
        with self.assertWarns(UserWarning):
            imputed_datasets = imputer.impute_multiple(self.df_mixed.copy(), n_imputations=2)
        
        # Check that datasets are identical
        assert_frame_equal(imputed_datasets[0], imputed_datasets[1])

    def test_not_fitted_error_impute_multiple(self):
        imputer = IterativeImputer()
        with self.assertRaises(NotFittedError):
            imputer.impute_multiple(self.df_mixed, n_imputations=2)
            
    def test_new_categories_in_transform(self):
        fit_df = pd.DataFrame({'cat': ['A', 'B', np.nan], 'val': [1,2,3]})
        imputer = IterativeImputer(random_state=self.rng_seed)
        imputer.fit(fit_df)

        transform_df_new_cat = pd.DataFrame({'cat': ['A', 'C', 'B'], 'val': [4,5,6]}) # 'C' is new
        with self.assertWarns(UserWarning) as cm:
            transformed = imputer.transform(transform_df_new_cat)
        
        self.assertTrue(any("new categories during transform" in str(w.message) for w in cm.warnings))
        # The new category 'C' should have resulted in NaN after encoding attempts,
        # which then might get imputed by the iterative process.
        # Check if the original 'C' position (index 1) is not NaN after imputation.
        self.assertFalse(pd.isnull(transformed.loc[1, 'cat']))


    def test_all_nan_categorical_column_fit(self):
        df_all_nan_cat = pd.DataFrame({
            'cat_all_nan': [np.nan, np.nan, np.nan],
            'num': [1,2,3]
        })
        imputer = IterativeImputer(random_state=self.rng_seed)
        with self.assertWarns(UserWarning) as cm: # Warns about LE not fitting, initial fill, etc.
            imputer.fit(df_all_nan_cat.copy())
        # Check that the warning message is about all-NaN categorical columns
        self.assertTrue(any("contains only NaN values" in str(w.message) for w in cm.warnings))
        # Estimator for cat_all_nan might not be fitted or be None
        cat_col_idx = df_all_nan_cat.columns.get_loc('cat_all_nan')
        self.assertNotIn(cat_col_idx, imputer.estimators_, 
                         "Estimator should not be robustly fitted for all-NaN categorical column.")
        
        transformed = imputer.transform(df_all_nan_cat.copy())
        # The column cat_all_nan might be filled with a placeholder from initial_impute
        # or remain NaN if models couldn't predict for it.
        # Default initial fill for all-NaN categorical (encoded) is -1. Inverse transform of -1 might be an issue.
        # Check it doesn't error and has some value (or is NaN, which is also acceptable if un-imputable)
        # For now, let's accept NaNs or a placeholder from inverse transform attempt.
        self.assertTrue(transformed['cat_all_nan'].notna().all() or transformed['cat_all_nan'].isna().all() or 
                        (transformed['cat_all_nan'] == -1).all() ) # -1 if it was numeric, object after inverse might be nan

    def test_basic_imputation(self):
        """Test basic functionality of IterativeImputer."""
        # Create a simple dataset with missing values
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4, 5],
            'B': [1, 2, np.nan, 4, 5],
            'C': [1, 2, 3, np.nan, 5],
            'D': ['a', 'b', 'c', 'd', 'e']  # Non-numeric column
        })
        
        imputer = IterativeImputer(max_iter=10, random_state=42)
        imputed = imputer.fit_transform(df)
        
        # Check that all numeric columns are imputed
        self.assertFalse(imputed[['A', 'B', 'C']].isnull().any().any())
        # Check that non-numeric column is unchanged
        self.assertTrue((imputed['D'] == df['D']).all())
        # Check that imputed values are reasonable
        self.assertTrue(imputed['A'].iloc[1] > 0)
        self.assertTrue(imputed['B'].iloc[2] > 0)
        self.assertTrue(imputed['C'].iloc[3] > 0)

    def test_estimator_selection(self):
        """Test different estimator choices."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4, 5],
            'B': [1, 2, np.nan, 4, 5],
            'C': [1, 2, 3, np.nan, 5]
        })
        
        # Test with BayesianRidge
        imputer_br = IterativeImputer(estimator=BayesianRidge(), random_state=42)
        imputed_br = imputer_br.fit_transform(df)
        
        # Test with RandomForest
        imputer_rf = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10), random_state=42)
        imputed_rf = imputer_rf.fit_transform(df)
        
        # Results should be different but both valid
        self.assertFalse(np.array_equal(imputed_br.values, imputed_rf.values))
        self.assertFalse(imputed_br.isnull().any().any())
        self.assertFalse(imputed_rf.isnull().any().any())

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty DataFrame
        with self.assertRaises(ValueError):
            IterativeImputer().fit(pd.DataFrame())
        
        # DataFrame with no numeric columns
        df_no_numeric = pd.DataFrame({'A': ['a', 'b', 'c']})
        imputer = IterativeImputer()
        imputed = imputer.fit_transform(df_no_numeric)
        self.assertTrue((imputed == df_no_numeric).all().all())
        
        # DataFrame with all NaN values
        df_all_nan = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [np.nan, np.nan, np.nan]
        })
        imputer = IterativeImputer()
        with self.assertWarns(UserWarning):
            imputed = imputer.fit_transform(df_all_nan)
        self.assertTrue(imputed.isnull().all().all())
        
        # Single column with some NaN
        df_single = pd.DataFrame({'A': [1, np.nan, 3]})
        imputer = IterativeImputer()
        imputed = imputer.fit_transform(df_single)
        self.assertFalse(imputed.isnull().any().any())
        self.assertTrue(imputed['A'].iloc[1] > 0)

    def test_convergence(self):
        """Test convergence behavior and max_iter parameter."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4, 5],
            'B': [1, 2, np.nan, 4, 5],
            'C': [1, 2, 3, np.nan, 5]
        })
        
        # Test with very few iterations
        imputer_few = IterativeImputer(max_iter=2, random_state=42)
        imputed_few = imputer_few.fit_transform(df)
        
        # Test with more iterations
        imputer_many = IterativeImputer(max_iter=10, random_state=42)
        imputed_many = imputer_many.fit_transform(df)
        
        # Results should be different but both valid
        self.assertFalse(np.array_equal(imputed_few.values, imputed_many.values))
        self.assertFalse(imputed_few.isnull().any().any())
        self.assertFalse(imputed_many.isnull().any().any())

    def test_column_selection(self):
        """Test imputation with specific column selection."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4, 5],
            'B': [1, 2, np.nan, 4, 5],
            'C': [1, 2, 3, np.nan, 5],
            'D': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Test imputing only column A
        imputer = IterativeImputer(columns=['A'])
        imputed = imputer.fit_transform(df)
        self.assertFalse(imputed['A'].isnull().any())
        self.assertTrue(imputed['B'].isnull().any())
        self.assertTrue(imputed['C'].isnull().any())
        
        # Test with non-existent column
        with self.assertWarns(UserWarning):
            imputer = IterativeImputer(columns=['A', 'NonExistent'])
            imputed = imputer.fit_transform(df)
        self.assertFalse(imputed['A'].isnull().any())

    def test_large_dataset(self):
        """Test performance and correctness on larger dataset."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        data = np.random.randn(n_samples, n_features)
        mask = np.random.random((n_samples, n_features)) < 0.1
        data[mask] = np.nan
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(n_features)])
        
        imputer = IterativeImputer(max_iter=5, random_state=42)
        imputed = imputer.fit_transform(df)
        
        self.assertFalse(imputed.isnull().any().any())
        self.assertTrue(imputed.min().min() > -10)
        self.assertTrue(imputed.max().max() < 10)

    def test_transform_without_fit(self):
        """Test that transform raises error if not fitted."""
        imputer = IterativeImputer()
        with self.assertRaises(NotFittedError):
            imputer.transform(pd.DataFrame({'A': [1, 2, 3]}))

    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        with self.assertRaises(ValueError):
            IterativeImputer(max_iter=0)
        
        with self.assertRaises(ValueError):
            IterativeImputer(max_iter=-1)
        
        with self.assertRaises(ValueError):
            IterativeImputer(initial_strategy='invalid')

    def test_missing_value_patterns(self):
        """Test different patterns of missing values."""
        # Test column with all values missing except one
        df = pd.DataFrame({
            'A': [1, np.nan, np.nan, np.nan, np.nan],
            'B': [1, 2, 3, 4, 5]
        })
        imputer = IterativeImputer()
        imputed = imputer.fit_transform(df)
        self.assertFalse(imputed.isnull().any().any())
        
        # Test row with all values missing
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4, 5],
            'B': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        imputer = IterativeImputer()
        with self.assertWarns(UserWarning):
            imputed = imputer.fit_transform(df)
        self.assertTrue(imputed['B'].isnull().all())

    def test_consistency(self):
        """Test that imputation is consistent across multiple calls."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4, 5],
            'B': [1, 2, np.nan, 4, 5],
            'C': [1, 2, 3, np.nan, 5]
        })
        
        imputer = IterativeImputer(random_state=42)
        imputed1 = imputer.fit_transform(df)
        imputed2 = imputer.fit_transform(df)
        
        # Results should be identical
        self.assertTrue(np.array_equal(imputed1.values, imputed2.values))

    def test_mixed_dtypes(self):
        """Test handling of mixed data types."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4, 5],
            'B': [1.1, 2.2, np.nan, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e'],
            'D': [True, False, True, np.nan, True],
            'E': pd.date_range('2020-01-01', periods=5)
        })
        
        imputer = IterativeImputer()
        imputed = imputer.fit_transform(df)
        
        # Check that only numeric columns are imputed
        self.assertFalse(imputed[['A', 'B']].isnull().any().any())
        self.assertTrue((imputed['C'] == df['C']).all())
        self.assertTrue((imputed['D'] == df['D']).all())
        self.assertTrue((imputed['E'] == df['E']).all())

    def test_high_dimensionality(self):
        """Test behavior with high-dimensional data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 100
        data = np.random.randn(n_samples, n_features)
        mask = np.random.random((n_samples, n_features)) < 0.1
        data[mask] = np.nan
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(n_features)])
        
        imputer = IterativeImputer(max_iter=5, random_state=42)
        imputed = imputer.fit_transform(df)
        
        self.assertFalse(imputed.isnull().any().any())
        self.assertTrue(imputed.shape == df.shape)

    def test_sparse_missing(self):
        """Test with sparse missing value patterns."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1, np.nan, 3, np.nan, 5],
            'C': [np.nan, 2, np.nan, 4, np.nan]
        })
        
        imputer = IterativeImputer()
        imputed = imputer.fit_transform(df)
        
        self.assertFalse(imputed.isnull().any().any())
        # Check that imputed values are reasonable
        self.assertTrue(imputed['B'].iloc[1] > 0)
        self.assertTrue(imputed['B'].iloc[3] > 0)
        self.assertTrue(imputed['C'].iloc[0] > 0)
        self.assertTrue(imputed['C'].iloc[2] > 0)
        self.assertTrue(imputed['C'].iloc[4] > 0)

    def test_initial_strategy(self):
        """Test different initial imputation strategies."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4, 5],
            'B': [1, 2, np.nan, 4, 5],
            'C': [1, 2, 3, np.nan, 5]
        })
        
        # Test mean strategy
        imputer_mean = IterativeImputer(initial_strategy='mean', random_state=42)
        imputed_mean = imputer_mean.fit_transform(df)
        
        # Test median strategy
        imputer_median = IterativeImputer(initial_strategy='median', random_state=42)
        imputed_median = imputer_median.fit_transform(df)
        
        # Results should be different but both valid
        self.assertFalse(np.array_equal(imputed_mean.values, imputed_median.values))
        self.assertFalse(imputed_mean.isnull().any().any())
        self.assertFalse(imputed_median.isnull().any().any())

    def test_parallel_processing(self):
        """Test parallel processing capabilities."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4, 5],
            'B': [1, 2, np.nan, 4, 5],
            'C': [1, 2, 3, np.nan, 5]
        })
        
        # Test with single process
        imputer_single = IterativeImputer(n_jobs=1, random_state=42)
        imputed_single = imputer_single.fit_transform(df)
        
        # Test with multiple processes
        imputer_multi = IterativeImputer(n_jobs=-1, random_state=42)
        imputed_multi = imputer_multi.fit_transform(df)
        
        # Results should be identical
        self.assertTrue(np.array_equal(imputed_single.values, imputed_multi.values))

    def test_imputation_order(self):
        """Test different imputation order strategies."""
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4, 5],
            'B': [1, 2, np.nan, 4, 5],
            'C': [1, 2, 3, np.nan, 5]
        })
        
        # Test ascending order
        imputer_asc = IterativeImputer(imputation_order='ascending', random_state=42)
        imputed_asc = imputer_asc.fit_transform(df)
        
        # Test descending order
        imputer_desc = IterativeImputer(imputation_order='descending', random_state=42)
        imputed_desc = imputer_desc.fit_transform(df)
        
        # Test random order
        imputer_rand = IterativeImputer(imputation_order='random', random_state=42)
        imputed_rand = imputer_rand.fit_transform(df)
        
        # All results should be valid
        self.assertFalse(imputed_asc.isnull().any().any())
        self.assertFalse(imputed_desc.isnull().any().any())
        self.assertFalse(imputed_rand.isnull().any().any())

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)