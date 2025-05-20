import unittest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_index_equal

from implyo.analysis import missing_handler

class TestMissingHandler(unittest.TestCase):

    def setUp(self):
        self.df_mixed_missing = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': ['x', np.nan, 'y', 'z'],
            'C': [pd.Timestamp('2023-01-01'), pd.NaT, pd.Timestamp('2023-01-03'), pd.Timestamp('2023-01-04')]
        })
        self.df_no_missing = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        self.df_all_missing_col = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [np.nan, np.nan, np.nan]
        })
        self.df_empty = pd.DataFrame()

    def test_identify_missing_values(self):
        expected = pd.DataFrame({
            'A': [False, False, True, False],
            'B': [False, True, False, False],
            'C': [False, True, False, False]
        })
        result = missing_handler.identify_missing_values(self.df_mixed_missing)
        assert_frame_equal(result, expected)

        result_no_missing = missing_handler.identify_missing_values(self.df_no_missing)
        expected_no_missing = pd.DataFrame({
            'A': [False, False, False],
            'B': [False, False, False]
        })
        assert_frame_equal(result_no_missing, expected_no_missing)

        with self.assertRaises(TypeError):
            missing_handler.identify_missing_values("not a dataframe")

    def test_missing_value_summary(self):
        summary = missing_handler.missing_value_summary(self.df_mixed_missing)
        self.assertEqual(summary.loc[summary['column_name'] == 'A', 'missing_count'].iloc[0], 1)
        self.assertEqual(summary.loc[summary['column_name'] == 'B', 'missing_count'].iloc[0], 1)
        self.assertEqual(summary.loc[summary['column_name'] == 'C', 'missing_count'].iloc[0], 1)
        self.assertAlmostEqual(summary.loc[summary['column_name'] == 'A', 'missing_percentage'].iloc[0], 25.0)

        summary_no_missing = missing_handler.missing_value_summary(self.df_no_missing)
        self.assertTrue(summary_no_missing.empty or (summary_no_missing['missing_count'] == 0).all())


        summary_all_missing_col = missing_handler.missing_value_summary(self.df_all_missing_col)
        self.assertEqual(summary_all_missing_col.loc[summary_all_missing_col['column_name'] == 'B', 'missing_count'].iloc[0], 3)
        self.assertAlmostEqual(summary_all_missing_col.loc[summary_all_missing_col['column_name'] == 'B', 'missing_percentage'].iloc[0], 100.0)

        summary_empty = missing_handler.missing_value_summary(self.df_empty)
        self.assertTrue(summary_empty.empty)
        
        with self.assertRaises(TypeError):
            missing_handler.missing_value_summary([1,2,3])

    def test_get_rows_with_missing_values(self):
        result = missing_handler.get_rows_with_missing_values(self.df_mixed_missing)
        self.assertEqual(len(result), 2) 
        
        self.assertTrue(pd.isna(result.loc[1, 'B']) or pd.isna(result.loc[1, 'C']))
        self.assertTrue(pd.isna(result.loc[2, 'A']))


        result_no_missing = missing_handler.get_rows_with_missing_values(self.df_no_missing)
        self.assertTrue(result_no_missing.empty)
        
        with self.assertRaises(TypeError):
            missing_handler.get_rows_with_missing_values(None)


    def test_get_columns_with_missing_values(self):
        result = missing_handler.get_columns_with_missing_values(self.df_mixed_missing)
        expected = pd.Index(['A', 'B', 'C'])
        assert_index_equal(result, expected, check_order=False)

        result_no_missing = missing_handler.get_columns_with_missing_values(self.df_no_missing)
        self.assertTrue(result_no_missing.empty)

        result_all_missing_col = missing_handler.get_columns_with_missing_values(self.df_all_missing_col)
        expected_all_missing_col = pd.Index(['B'])
        assert_index_equal(result_all_missing_col, expected_all_missing_col)

        with self.assertRaises(TypeError):
            missing_handler.get_columns_with_missing_values(123)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)