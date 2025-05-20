# pip_impute/__init__.py
from .base_imputer import BaseImputer
from .analysis import missing_handler, pattern_analyzer # Expose modules
from .viz import missing_plots # Expose module

# Expose specific functions directly from analysis and viz
from .analysis.missing_handler import (
    identify_missing_values,
    missing_value_summary,
    get_rows_with_missing_values,
    get_columns_with_missing_values
)
from .analysis.pattern_analyzer import (
    preliminary_mcar_test,
    suggest_missingness_pattern
)
from .viz.missing_plots import (
    plot_missingness_heatmap,
    plot_missingness_bar,
    plot_missingness_summary_bar
)

# START CHANGED SECTION: Add imputers
from .imputers import (
    MeanImputer,
    MedianImputer,
    ModeImputer,
    ConstantImputer,
    RandomSampleImputer
)
# END CHANGED SECTION

__version__ = "0.1.0-alpha"

# START CHANGED SECTION: Update __all__
__all__ = [
    'BaseImputer',
    # Analysis
    'missing_handler', 
    'pattern_analyzer',
    'identify_missing_values',
    'missing_value_summary',
    'get_rows_with_missing_values',
    'get_columns_with_missing_values',
    'preliminary_mcar_test',
    'suggest_missingness_pattern',
    # Visualization
    'missing_plots',
    'plot_missingness_heatmap',
    'plot_missingness_bar',
    'plot_missingness_summary_bar',
    # Imputers
    'MeanImputer',
    'MedianImputer',
    'ModeImputer',
    'ConstantImputer',
    'RandomSampleImputer',
    '__version__'
]
# END CHANGED SECTION