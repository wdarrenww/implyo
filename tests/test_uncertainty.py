# tests/test_uncertainty.py
import unittest
import pandas as pd
import numpy as np

from implyo.uncertainty import pool_means, pool_variances_rubin # Adjust import

class TestUncertaintyTooling(unittest.TestCase):

    def setUp(self):
        self.df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [10.0, 11.0, 12.0], 'C': ['x','y','z']}) # Mean A=2, Var A=1. Mean B=11, Var B=1
        self.df2 = pd.DataFrame({'A': [2, 3, 4], 'B': [11.0, 12.0, 13.0], 'C': ['x','y','z']}) # Mean A=3, Var A=1. Mean B=12, Var B=1
        self.df3 = pd.DataFrame({'A': [3, 4, 5], 'B': [12.0, 13.0, 14.0], 'C': ['x','y','z']}) # Mean A=4, Var A=1. Mean B=13, Var B=1
        self.imputed_datasets_AB = [self.df1, self.df2, self.df3]
        
        # For m=2 case
        self.imputed_datasets_m2 = [
            pd.DataFrame({'val': [1, 2, 3, 4, 5]}), # mean 3, var 2.5
            pd.DataFrame({'val': [2, 3, 4, 5, 6]})  # mean 4, var 2.5
        ]

    # --- Tests for pool_means ---
    def test_pool_means_single_var_str(self):
        pooled_A = pool_means(self.imputed_datasets_AB, target_vars='A')
        self.assertAlmostEqual(pooled_A, (2+3+4)/3) # Expected: 3.0

    def test_pool_means_single_var_list(self):
        pooled_A_series = pool_means(self.imputed_datasets_AB, target_vars=['A'])
        self.assertIsInstance(pooled_A_series, pd.Series)
        self.assertAlmostEqual(pooled_A_series['A'], 3.0)

    def test_pool_means_multiple_vars(self):
        pooled_AB = pool_means(self.imputed_datasets_AB, target_vars=['A', 'B'])
        self.assertIsInstance(pooled_AB, pd.Series)
        self.assertAlmostEqual(pooled_AB['A'], 3.0)
        self.assertAlmostEqual(pooled_AB['B'], (11+12+13)/3) # Expected: 12.0

    def test_pool_means_none_vars(self): # All numeric
        pooled_all = pool_means(self.imputed_datasets_AB)
        self.assertIsInstance(pooled_all, pd.Series)
        self.assertAlmostEqual(pooled_all['A'], 3.0)
        self.assertAlmostEqual(pooled_all['B'], 12.0)
        self.assertNotIn('C', pooled_all.index)

    def test_pool_means_errors(self):
        with self.assertRaises(ValueError): # Empty list
            pool_means([])
        with self.assertRaises(TypeError): # Not list of DFs
            pool_means([self.df1, "not_a_df"])
        df_mismatch_cols = pd.DataFrame({'X': [1,2]})
        with self.assertRaises(ValueError): # Mismatched columns
            pool_means([self.df1, df_mismatch_cols])
        with self.assertRaises(ValueError): # Var not found
            pool_means(self.imputed_datasets_AB, target_vars='Z')
        with self.assertRaises(ValueError): # Non-numeric var
            pool_means(self.imputed_datasets_AB, target_vars='C')
        with self.assertRaises(ValueError): # No numeric columns if target_vars=None and only non-numeric exist
            pool_means([pd.DataFrame({'C':['x','y']})])


    # --- Tests for pool_variances_rubin ---
    def test_pool_variances_single_var_str(self):
        # For A: Q_hat_i = [2, 3, 4]. Q_bar = 3.
        # U_i = [1, 1, 1]. U_bar = 1.
        # B = ((2-3)^2 + (3-3)^2 + (4-3)^2) / (3-1) = (1+0+1)/2 = 1.
        # T = U_bar + (1 + 1/m) * B = 1 + (1 + 1/3) * 1 = 1 + (4/3) * 1 = 1 + 1.333... = 2.333...
        pooled_var_A = pool_variances_rubin(self.imputed_datasets_AB, target_vars='A')
        self.assertAlmostEqual(pooled_var_A, 1 + (4/3)*1)

    def test_pool_variances_multiple_vars_with_provided_means(self):
        pooled_means_val = pool_means(self.imputed_datasets_AB, target_vars=['A', 'B'])
        pooled_vars = pool_variances_rubin(self.imputed_datasets_AB, target_vars=['A', 'B'], pooled_means_input=pooled_means_val)
        self.assertIsInstance(pooled_vars, pd.Series)
        self.assertAlmostEqual(pooled_vars['A'], 1 + (4/3)*1)
        # For B: Q_hat_i = [11, 12, 13]. Q_bar = 12.
        # U_i = [1, 1, 1]. U_bar = 1.
        # B = ((11-12)^2 + (12-12)^2 + (13-12)^2) / (3-1) = (1+0+1)/2 = 1.
        # T = 1 + (4/3)*1 = 2.333...
        self.assertAlmostEqual(pooled_vars['B'], 1 + (4/3)*1)

    def test_pool_variances_none_vars(self):
        pooled_vars_all = pool_variances_rubin(self.imputed_datasets_AB) # Calculates means internally
        self.assertIsInstance(pooled_vars_all, pd.Series)
        self.assertAlmostEqual(pooled_vars_all['A'], 1 + (4/3)*1)
        self.assertAlmostEqual(pooled_vars_all['B'], 1 + (4/3)*1)
        self.assertNotIn('C', pooled_vars_all.index)

    def test_pool_variances_m_equals_2(self):
        # For 'val': Q_hat_i = [3, 4]. Q_bar = 3.5
        # U_i = [2.5, 2.5]. U_bar = 2.5
        # B = ((3-3.5)^2 + (4-3.5)^2) / (2-1) = ((-0.5)^2 + (0.5)^2) / 1 = (0.25 + 0.25) / 1 = 0.5
        # T = U_bar + (1 + 1/m) * B = 2.5 + (1 + 1/2) * 0.5 = 2.5 + 1.5 * 0.5 = 2.5 + 0.75 = 3.25
        pooled_var_val = pool_variances_rubin(self.imputed_datasets_m2, target_vars='val')
        self.assertAlmostEqual(pooled_var_val, 3.25)

    def test_pool_variances_errors(self):
        with self.assertRaises(ValueError): # m < 2
            pool_variances_rubin([self.df1])
        with self.assertRaises(TypeError): # Pooled means input wrong type
             pool_variances_rubin(self.imputed_datasets_AB, target_vars='A', pooled_means_input=[1,2,3])
        with self.assertRaises(ValueError): # Pooled means Series does not contain target_var
             pool_variances_rubin(self.imputed_datasets_AB, target_vars='A', pooled_means_input=pd.Series({'Z':0}))


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)