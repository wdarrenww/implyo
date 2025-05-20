import unittest
import pandas as pd
import numpy as np
from scipy import stats

from implyo.analysis import pattern_analyzer, missing_handler

class TestPatternAnalyzer(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.RandomState(42)
        self.n_samples = 100
        self.df_mcar = pd.DataFrame({
            'A': self.rng.randn(self.n_samples),
            'B': self.rng.choice(['x', 'y', 'z'], self.n_samples),
            'C_missing_mcar': self.rng.randn(self.n_samples)
        })
        idx_mcar = self.rng.choice(self.df_mcar.index, size=20, replace=False)
        self.df_mcar.loc[idx_mcar, 'C_missing_mcar'] = np.nan

        self.df_mar = pd.DataFrame({
            'observed_num': self.rng.randint(0, 100, self.n_samples),
            'observed_cat': self.rng.choice(['cat1', 'cat2', 'cat3'], self.n_samples),
            'D_missing_mar_num': self.rng.randn(self.n_samples),
            'E_missing_mar_cat': self.rng.randn(self.n_samples)
        })
        idx_mar_num = self.df_mar[self.df_mar['observed_num'] > 50].sample(frac=0.5, random_state=self.rng).index
        self.df_mar.loc[idx_mar_num, 'D_missing_mar_num'] = np.nan
        
        idx_mar_cat = self.df_mar[self.df_mar['observed_cat'] == 'cat1'].sample(frac=0.6, random_state=self.rng).index
        self.df_mar.loc[idx_mar_cat, 'E_missing_mar_cat'] = np.nan
        
        self.df_all_nan_col = pd.DataFrame({'X': [1,2,3], 'Y': [np.nan, np.nan, np.nan]})
        self.df_no_missing = pd.DataFrame({'X': [1,2,3], 'Y': [4,5,6]})


    def test_preliminary_mcar_test_mcar_scenario(self):
        results = pattern_analyzer.preliminary_mcar_test(self.df_mcar, 'C_missing_mcar', significance_level=0.01)
        for col, res in results.items():
            if 'p_value' in res:
                self.assertFalse(res['reject_null'], f"MCAR test for C_missing_mcar related to {col} should not reject null (p={res['p_value']})")

    def test_preliminary_mcar_test_mar_num_scenario(self):
        results = pattern_analyzer.preliminary_mcar_test(self.df_mar, 'D_missing_mar_num', significance_level=0.05)
        self.assertTrue(results['observed_num']['reject_null'], 
                        f"MAR test for D_missing_mar_num related to observed_num should reject null (p={results['observed_num'].get('p_value')})")
        if 'observed_cat' in results and 'reject_null' in results['observed_cat']:
            self.assertFalse(results['observed_cat']['reject_null'], 
                         f"MAR test for D_missing_mar_num related to observed_cat should not reject null (p={results['observed_cat'].get('p_value')})")

    def test_preliminary_mcar_test_mar_cat_scenario(self):
        results = pattern_analyzer.preliminary_mcar_test(self.df_mar, 'E_missing_mar_cat', significance_level=0.05)
        if 'observed_num' in results and 'reject_null' in results['observed_num']:
            self.assertFalse(results['observed_num']['reject_null'],
                        f"MAR test for E_missing_mar_cat related to observed_num should not reject null (p={results['observed_num'].get('p_value')})")
        self.assertTrue(results['observed_cat']['reject_null'],
                        f"MAR test for E_missing_mar_cat related to observed_cat should reject null (p={results['observed_cat'].get('p_value')})")

    def test_preliminary_mcar_test_no_missing(self):
        with self.assertWarns(UserWarning):
            results = pattern_analyzer.preliminary_mcar_test(self.df_no_missing, 'X')
        self.assertEqual(results, {})

    def test_preliminary_mcar_test_all_missing(self):
        with self.assertWarns(UserWarning):
            results = pattern_analyzer.preliminary_mcar_test(self.df_all_nan_col, 'Y')
        self.assertEqual(results, {})

    def test_preliminary_mcar_test_single_aux_column_skipped(self):
        df_aux_all_nan = pd.DataFrame({
            'target': [1, np.nan, 3, np.nan],
            'aux_all_nan': [np.nan, np.nan, np.nan, np.nan],
            'aux_valid': [10,20,30,40]
        })
        results = pattern_analyzer.preliminary_mcar_test(df_aux_all_nan, 'target')
        self.assertEqual(results['aux_all_nan']['test'], 'skipped')
        self.assertIn('p_value', results['aux_valid'])


    def test_suggest_missingness_pattern_mcar(self):
        suggestions = pattern_analyzer.suggest_missingness_pattern(self.df_mcar, significance_level=0.01)
        c_suggestion = suggestions[suggestions['column_name'] == 'C_missing_mcar'].iloc[0]
        self.assertIn("Likely MCAR", c_suggestion['suggested_pattern'])

    def test_suggest_missingness_pattern_mar(self):
        suggestions = pattern_analyzer.suggest_missingness_pattern(self.df_mar, significance_level=0.05)
        
        d_suggestion = suggestions[suggestions['column_name'] == 'D_missing_mar_num'].iloc[0]
        self.assertIn("Likely MAR", d_suggestion['suggested_pattern'])
        self.assertIn("observed_num", d_suggestion['evidence'])
        
        e_suggestion = suggestions[suggestions['column_name'] == 'E_missing_mar_cat'].iloc[0]
        self.assertIn("Likely MAR", e_suggestion['suggested_pattern'])
        self.assertIn("observed_cat", e_suggestion['evidence'])
        
    def test_suggest_missingness_pattern_no_missing(self):
        suggestions = pattern_analyzer.suggest_missingness_pattern(self.df_no_missing)
        self.assertTrue(suggestions.empty)

    def test_suggest_missingness_pattern_all_missing_col(self):
        suggestions = pattern_analyzer.suggest_missingness_pattern(self.df_all_nan_col)
        y_suggestion = suggestions[suggestions['column_name'] == 'Y'].iloc[0]
        self.assertIn("Undetermined", y_suggestion['suggested_pattern'])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)