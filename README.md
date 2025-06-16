# Implyo: Advanced Missing Value Imputation Library

Implyo is a powerful Python library for handling missing values in mixed-type data, with a focus on performance, accuracy, and uncertainty quantification. It provides a collection of advanced imputation algorithms that can handle both numeric and categorical variables efficiently.

## Features

### Core Imputation Algorithms

- **KNN Imputer**: Fast and efficient k-nearest neighbors imputation with support for mixed data types
- **MICE (Iterative Imputer)**: Multiple Imputation by Chained Equations with various estimator options
- **Random Forest Imputer**: Tree-based imputation with uncertainty quantification
- **XGBoost Imputer**: Gradient boosting based imputation with advanced features
- **LightGBM Imputer**: Light gradient boosting based imputation with high performance

### Key Features

- **Mixed Data Type Support**: Handle both numeric and categorical variables seamlessly
- **Uncertainty Quantification**: Get prediction intervals for imputed values
- **Parallel Processing**: Efficient handling of large datasets
- **Early Stopping**: Automatic convergence detection
- **Feature Importance**: Track which features are most important for imputation
- **Missing Value Indicators**: Optional indicators for missing value patterns
- **Comprehensive Testing**: Extensive test coverage for all imputers
- **Benchmarking Tools**: Compare performance across different imputers

## Installation

```bash
pip install implyo
```

For development installation:

```bash
git clone https://github.com/yourusername/implyo.git
cd implyo
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
import numpy as np
from implyo import XGBoostImputer, LightGBMImputer, KNNImputer

# Create a sample dataset with missing values
data = pd.DataFrame({
    'numeric1': [1, 2, np.nan, 4, 5],
    'numeric2': [1.1, np.nan, 3.3, 4.4, 5.5],
    'categorical': ['a', 'b', 'c', np.nan, 'e']
})

# Initialize and fit the imputer
imputer = XGBoostImputer(
    n_estimators=100,
    categorical_features=['categorical'],
    uncertainty_quantile=0.95,  # Get prediction intervals
    random_state=42
)

# Fit and transform the data
X_imputed = imputer.fit_transform(data)

# Get uncertainty intervals
intervals = imputer.uncertainty_intervals_

# Get feature importances
importances = imputer.feature_importances_
```

## Advanced Usage

### Uncertainty Quantification

All tree-based imputers (Random Forest, XGBoost, LightGBM) support uncertainty quantification:

```python
from implyo import RandomForestImputer

imputer = RandomForestImputer(
    uncertainty_quantile=0.95,  # 95% prediction intervals
    n_estimators=100,
    random_state=42
)

X_imputed = imputer.fit_transform(data)
intervals = imputer.uncertainty_intervals_

# Access intervals for a specific column
lower, upper = intervals['numeric1']
```

### Parallel Processing

All imputers support parallel processing for faster computation:

```python
imputer = XGBoostImputer(
    n_jobs=-1,  # Use all available cores
    n_estimators=100,
    random_state=42
)
```

### Feature Importance

Tree-based imputers provide feature importance information:

```python
imputer = LightGBMImputer(
    n_estimators=100,
    random_state=42
)
imputer.fit_transform(data)

# Get feature importances for each imputed variable
importances = imputer.feature_importances_
```

### Missing Value Indicators

Add binary indicators for missing value patterns:

```python
imputer = KNNImputer(
    add_indicator=True,  # Add missing value indicators
    n_neighbors=5
)
X_imputed = imputer.fit_transform(data)
```

## Benchmarking

The package includes comprehensive benchmarking tools to compare different imputers:

```python
from implyo.benchmarks import run_benchmark

# Run benchmarks with different configurations
results = run_benchmark(
    n_samples=1000,
    n_numeric_features=5,
    n_categorical_features=3,
    missing_ratio=0.2,
    n_repeats=3
)
print(results)
```

## Performance

Implyo's imputers are optimized for performance:

- **KNN Imputer**: Faster than scikit-learn's implementation
- **XGBoost Imputer**: Efficient handling of large datasets
- **LightGBM Imputer**: High performance with low memory usage
- **Random Forest Imputer**: Balanced performance and accuracy
- **MICE**: Flexible and robust for complex missing patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Implyo in your research, please cite:

```bibtex
@software{implyo2024,
  author = {Darren Wei},
  title = {Implyo: Advanced Missing Value Imputation Library},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/implyo}
}
```

## Roadmap

- [ ] Add more advanced imputation algorithms
- [ ] Support for time series data
- [ ] Integration with deep learning models
- [ ] Web-based visualization tools
- [ ] Distributed computing support
- [ ] GPU acceleration for large datasets