import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch

from implyo.viz import missing_plots

class TestMissingPlots(unittest.TestCase):

    def setUp(self):
        self.df_missing = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [np.nan, 'x', 'y'],
            'C': [pd.NaT, pd.Timestamp('20230101'), pd.Timestamp('20230102')]
        })
        self.df_no_missing = pd.DataFrame({'X': [1, 2], 'Y': ['a', 'b']})
        self.df_empty = pd.DataFrame()
    
    @patch('matplotlib.pyplot.show')
    def test_plot_missingness_heatmap(self, mock_show):
        try:
            ax = missing_plots.plot_missingness_heatmap(self.df_missing.copy())
            self.assertIsNotNone(ax)
            mock_show.assert_called_once()
            
            plt.close('all')
            mock_show.reset_mock()

            ax_no_missing = missing_plots.plot_missingness_heatmap(self.df_no_missing.copy())
            self.assertIsNotNone(ax_no_missing)
            mock_show.assert_called_once()
            plt.close('all')
            mock_show.reset_mock()

            ax_empty = missing_plots.plot_missingness_heatmap(self.df_empty.copy())
            self.assertIsNotNone(ax_empty)
            mock_show.assert_called_once()
            plt.close('all')

        except Exception as e:
            self.fail(f"plot_missingness_heatmap raised an exception: {e}")


    @patch('matplotlib.pyplot.show')
    def test_plot_missingness_bar(self, mock_show):
        try:
            ax = missing_plots.plot_missingness_bar(self.df_missing.copy())
            self.assertIsNotNone(ax)
            mock_show.assert_called_once()
            plt.close('all')
            mock_show.reset_mock()

            ax_no_missing = missing_plots.plot_missingness_bar(self.df_no_missing.copy())
            self.assertIsNotNone(ax_no_missing)
            mock_show.assert_called_once()
            plt.close('all')
            mock_show.reset_mock()
            
            ax_empty = missing_plots.plot_missingness_bar(self.df_empty.copy())
            self.assertIsNotNone(ax_empty)
            mock_show.assert_called_once()
            plt.close('all')

        except Exception as e:
            self.fail(f"plot_missingness_bar raised an exception: {e}")

    @patch('matplotlib.pyplot.show')
    def test_plot_missingness_summary_bar(self, mock_show):
        try:
            ax = missing_plots.plot_missingness_summary_bar(self.df_missing.copy())
            self.assertIsNotNone(ax)
            mock_show.assert_called_once()
            plt.close('all')
            mock_show.reset_mock()

            with unittest.mock.patch('builtins.print') as mocked_print:
                 ax_no_missing = missing_plots.plot_missingness_summary_bar(self.df_no_missing.copy())
                 self.assertIsNone(ax_no_missing)
                 mocked_print.assert_any_call("No missing values to plot in summary bar chart.")
            self.assertFalse(mock_show.called)
            plt.close('all')
            mock_show.reset_mock()
            
            with unittest.mock.patch('builtins.print') as mocked_print:
                ax_empty = missing_plots.plot_missingness_summary_bar(self.df_empty.copy())
                self.assertIsNone(ax_empty)
                mocked_print.assert_any_call("No missing values to plot in summary bar chart.")
            self.assertFalse(mock_show.called)
            plt.close('all')

            ax_top_n = missing_plots.plot_missingness_summary_bar(self.df_missing.copy(), top_n=1)
            self.assertIsNotNone(ax_top_n)
            self.assertEqual(len(ax_top_n.get_yticklabels()), 1)
            mock_show.assert_called_once()
            plt.close('all')


        except Exception as e:
            self.fail(f"plot_missingness_summary_bar raised an exception: {e}")
            plt.close('all')

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)