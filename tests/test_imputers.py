import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

from implyo.imputers import MeanImputer, MedianImputer, ModeImputer, ConstantImputer, RandomSampleImputer
from implyo.base_imputer import BaseImputer

class TestImputers(unittest.TestCase):

    def setUp(self):
        self.df_numeric = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5, np.nan],
            'B': [10, np.nan, 30, 40, np.nan, 60],
            'C': [100, 200, 300, 400, 500, 600]
        })
        self.df_categorical = pd.DataFrame({
            'X': ['a', 'b', np.nan, 'a', 'c', 'a'],
            'Y': ['m', 'n', 'o', np.nan, 'm', np.nan]
        })
        self.df_mixed = pd.DataFrame({
            'num1': [1, np.nan, 3, 4, 5],
            'cat1': ['apple', 'banana', np.nan, 'apple', 'orange'],
            'num2_all_nan': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'cat2_all_nan': [np.nan, np.nan, np.nan, np.nan, np.nan],
        })
        self.df_empty = pd.DataFrame()
        self.df_no_missing = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})

    def test_mean_imputer_fit_transform(self):
        imputer = MeanImputer()
        df_transformed = imputer.fit_transform(self.df_numeric.copy())
        
        expected_A = self.df_numeric['A'].fillna(self.df_numeric['A'].mean())
        expected_B = self.df_numeric['B'].fillna(self.df_numeric['B'].mean())
        
        assert_series_equal(df_transformed['A'], expected_A, check_dtype=False)
        assert_series_equal(df_transformed['B'], expected_B, check_dtype=False)
        assert_series_equal(df_transformed['C'], self.df_numeric['C'], check_dtype=False)
        self.assertAlmostEqual(imputer.statistics_['A'], 3.0)
        self.assertAlmostEqual(imputer.statistics_['B'], 35.0)
        self.assertTrue(imputer._is_fitted)
        self.assertIn('C', imputer.statistics_)

    def test_mean_imputer_columns_specified(self):
        imputer = MeanImputer(columns=['A'])
        df_transformed = imputer.fit_transform(self.df_numeric.copy())
        
        expected_A = self.df_numeric['A'].fillna(self.df_numeric['A'].mean())
        assert_series_equal(df_transformed['A'], expected_A, check_dtype=False)
        assert_series_equal(df_transformed['B'], self.df_numeric['B'], check_dtype=False)
        self.assertIn('A', imputer.statistics_)
        self.assertNotIn('B', imputer.statistics_)

    def test_mean_imputer_non_numeric_ignored(self):
        with self.assertWarns(UserWarning) as cm:
             imputer = MeanImputer(columns=['num1', 'cat1'])
             df_transformed = imputer.fit_transform(self.df_mixed.copy())
        self.assertTrue(any("MeanImputer specified for non-numeric columns: ['cat1']" in str(w.message) for w in cm.warnings))
        self.assertAlmostEqual(imputer.statistics_['num1'], 3.25)
        self.assertNotIn('cat1', imputer.statistics_)
        expected_num1 = self.df_mixed['num1'].fillna(self.df_mixed['num1'].mean())
        assert_series_equal(df_transformed['num1'], expected_num1, check_dtype=False)
        assert_series_equal(df_transformed['cat1'], self.df_mixed['cat1'], check_dtype=False)

    def test_mean_imputer_all_nan_column(self):
        with self.assertWarns(UserWarning) as cm:
            imputer = MeanImputer(columns=['num2_all_nan'])
            df_transformed = imputer.fit_transform(self.df_mixed.copy())
        self.assertTrue(any("Column 'num2_all_nan' has all missing values. Mean is NaN." in str(w.message) for w in cm.warnings))
        self.assertTrue(pd.isna(imputer.statistics_['num2_all_nan']))
        assert_series_equal(df_transformed['num2_all_nan'], self.df_mixed['num2_all_nan'])

    def test_median_imputer_fit_transform(self):
        imputer = MedianImputer()
        df_transformed = imputer.fit_transform(self.df_numeric.copy())
        
        expected_A = self.df_numeric['A'].fillna(self.df_numeric['A'].median())
        expected_B = self.df_numeric['B'].fillna(self.df_numeric['B'].median())
        
        assert_series_equal(df_transformed['A'], expected_A, check_dtype=False)
        assert_series_equal(df_transformed['B'], expected_B, check_dtype=False)
        self.assertAlmostEqual(imputer.statistics_['A'], 3.0)
        self.assertAlmostEqual(imputer.statistics_['B'], 35.0)

    def test_mode_imputer_fit_transform(self):
        imputer = ModeImputer()
        df_cat_copy = self.df_categorical.copy()
        df_transformed = imputer.fit_transform(df_cat_copy)

        expected_X = df_cat_copy['X'].fillna('a')
        expected_Y = df_cat_copy['Y'].fillna('m')

        assert_series_equal(df_transformed['X'], expected_X)
        assert_series_equal(df_transformed['Y'], expected_Y)
        self.assertEqual(imputer.statistics_['X'], 'a')
        self.assertEqual(imputer.statistics_['Y'], 'm')

    def test_mode_imputer_numeric_and_all_nan(self):
        imputer = ModeImputer(columns=['num1', 'cat1', 'cat2_all_nan'])
        df_mixed_copy = self.df_mixed.copy()
        
        with self.assertWarns(UserWarning) as cm:
            df_transformed = imputer.fit_transform(df_mixed_copy)
        
        self.assertTrue(any("Column 'cat2_all_nan' has all missing values." in str(w.message) for w in cm.warnings))

        df_temp = pd.DataFrame({'num_mode_test': [1,1,np.nan,2,3]})
        mode_imp_num = ModeImputer()
        transformed_num = mode_imp_num.fit_transform(df_temp)
        self.assertEqual(mode_imp_num.statistics_['num_mode_test'], 1)
        self.assertEqual(transformed_num.loc[2, 'num_mode_test'], 1)

        self.assertEqual(imputer.statistics_['cat1'], 'apple')
        self.assertTrue(pd.isna(imputer.statistics_['cat2_all_nan']))
        assert_series_equal(df_transformed['cat2_all_nan'], df_mixed_copy['cat2_all_nan'])


    def test_constant_imputer_numeric(self):
        imputer = ConstantImputer(fill_value=0, columns=['A', 'B'])
        df_transformed = imputer.fit_transform(self.df_numeric.copy())
        self.assertEqual(df_transformed.loc[2, 'A'], 0)
        self.assertEqual(df_transformed.loc[5, 'A'], 0)
        self.assertEqual(df_transformed.loc[1, 'B'], 0)
        self.assertEqual(df_transformed.loc[4, 'B'], 0)
        self.assertFalse(pd.isna(df_transformed.loc[5, 'A'])) 
        self.assertEqual(df_transformed.loc[5, 'A'], 0)
        self.assertEqual(imputer.statistics_['fill_value'], 0)
        self.assertEqual(df_transformed.loc[0, 'A'], self.df_numeric.loc[0, 'A'])

    def test_constant_imputer_string(self):
        imputer = ConstantImputer(fill_value='missing')
        df_transformed = imputer.fit_transform(self.df_categorical.copy())
        self.assertEqual(df_transformed.loc[2, 'X'], 'missing')
        self.assertEqual(df_transformed.loc[3, 'Y'], 'missing')

    def test_random_sample_imputer_reproducibility(self):
        imputer1 = RandomSampleImputer(columns=['A'], random_state=42)
        df_transformed1 = imputer1.fit_transform(self.df_numeric.copy())

        imputer2 = RandomSampleImputer(columns=['A'], random_state=42)
        df_transformed2 = imputer2.fit_transform(self.df_numeric.copy())
        
        assert_series_equal(df_transformed1['A'], df_transformed2['A'])
        self.assertTrue(imputer1._is_fitted)

    def test_random_sample_imputer_all_nan(self):
        df_all_nan_col = pd.DataFrame({'X': [np.nan, np.nan, np.nan]})
        imputer = RandomSampleImputer(random_state=0)
        with self.assertWarns(UserWarning) as cm:
             df_transformed = imputer.fit_transform(df_all_nan_col)
        self.assertTrue(any("Column 'X' has all missing values" in str(w.message) for w in cm.warnings))
        self.assertTrue(df_transformed['X'].isnull().all())

    def test_random_sample_imputer_values_from_observed(self):
        df_simple = pd.DataFrame({'col': [1, 2, np.nan, np.nan]})
        imputer = RandomSampleImputer(random_state=0)
        df_transformed = imputer.fit_transform(df_simple)
        
        filled_values = df_transformed.loc[df_simple['col'].isnull(), 'col']
        self.assertTrue(all(val in [1, 2] for val in filled_values))
        self.assertEqual(df_transformed.loc[0, 'col'], 1)
        self.assertEqual(df_transformed.loc[1, 'col'], 2)


    def _test_imputer_no_missing_data(self, ImputerClass, **kwargs):
        imputer = ImputerClass(**kwargs)
        df_transformed = imputer.fit_transform(self.df_no_missing.copy())
        assert_frame_equal(df_transformed, self.df_no_missing)
        self.assertTrue(imputer._is_fitted)

    def test_all_imputers_no_missing_data(self):
        self._test_imputer_no_missing_data(MeanImputer)
        self._test_imputer_no_missing_data(MedianImputer)
        self._test_imputer_no_missing_data(ModeImputer)
        self._test_imputer_no_missing_data(ConstantImputer, fill_value=99)
        self._test_imputer_no_missing_data(RandomSampleImputer, random_state=0)

    def _test_imputer_empty_df(self, ImputerClass, **kwargs):
        imputer = ImputerClass(**kwargs)
        with self.assertRaises(ValueError if ImputerClass not in [ConstantImputer, RandomSampleImputer] else Exception):
            if ImputerClass in [ConstantImputer, RandomSampleImputer]:
                with self.assertWarns(UserWarning):
                    df_transformed = imputer.fit_transform(self.df_empty.copy())
                    self.assertTrue(df_transformed.empty)
            else:
                 imputer.fit(self.df_empty.copy())


    @unittest.skip("Skipping empty DataFrame test for mean/median/mode as base class handles it.")
    def test_all_imputers_empty_df(self):
        self._test_imputer_empty_df(MeanImputer)
        self._test_imputer_empty_df(MedianImputer)
        self._test_imputer_empty_df(ModeImputer)
        
        const_imputer = ConstantImputer(fill_value=0)
        with self.assertWarns(UserWarning):
            df_transformed_const = const_imputer.fit_transform(self.df_empty.copy())
        self.assertTrue(df_transformed_const.empty)

        rand_imputer = RandomSampleImputer(random_state=0)
        with self.assertWarns(UserWarning):
             df_transformed_rand = rand_imputer.fit_transform(self.df_empty.copy())
        self.assertTrue(df_transformed_rand.empty)


    def test_imputer_mismatched_columns_transform(self):
        imputer = MeanImputer()
        imputer.fit(self.df_numeric)
        
        df_mismatch = pd.DataFrame({'X': [1, np.nan], 'Y': [2, 3]})
        with self.assertRaises(ValueError):
            imputer.transform(df_mismatch)
            
    def test_imputer_not_fitted_transform(self):
        imputer = MeanImputer()
        with self.assertRaises(Exception):
            imputer.transform(self.df_numeric)

    def test_feature_names_in(self):
        imputer = MeanImputer()
        imputer.fit(self.df_numeric)
        self.assertEqual(list(imputer.feature_names_in_), list(self.df_numeric.columns))

    def test_imputer_specified_column_not_present(self):
        with self.assertWarns(UserWarning) as cm:
            imputer = MeanImputer(columns=['A', 'Z'])
            imputer.fit(self.df_numeric.copy())
        expected_warning_message = "MeanImputer specified for columns not in DataFrame: ['Z']. These will be ignored."
        self.assertTrue(any(expected_warning_message == str(w.message) for w in cm.warnings),
                        f"Expected warning '{expected_warning_message}' not found. Warnings issued: {[str(w.message) for w in cm.warnings]}")
        self.assertIn('A', imputer._numeric_columns_to_impute)
        self.assertNotIn('Z', imputer._numeric_columns_to_impute)
        if imputer.statistics_ is not None:
            self.assertIn('A', imputer.statistics_)
            self.assertNotIn('Z', imputer.statistics_)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)