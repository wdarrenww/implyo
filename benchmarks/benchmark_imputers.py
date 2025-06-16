"""Benchmark script for comparing different imputation methods."""

import time
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from implyo import (
    KNNImputer,
    IterativeImputer,
    RandomForestImputer,
    XGBoostImputer,
    LightGBMImputer,
)


def create_mixed_dataset(
    n_samples: int = 1000,
    n_numeric_features: int = 5,
    n_categorical_features: int = 3,
    n_classes: int = 3,
    missing_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a mixed-type dataset with missing values.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples.
    n_numeric_features : int, default=5
        Number of numeric features.
    n_categorical_features : int, default=3
        Number of categorical features.
    n_classes : int, default=3
        Number of classes for categorical features.
    missing_ratio : float, default=0.2
        Ratio of missing values to introduce.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Original and missing data.
    """
    np.random.seed(random_state)
    
    # Create numeric features
    X_num, y = make_regression(
        n_samples=n_samples,
        n_features=n_numeric_features,
        n_informative=n_numeric_features // 2,
        noise=0.1,
        random_state=random_state
    )
    numeric_data = pd.DataFrame(
        X_num,
        columns=[f'numeric_{i}' for i in range(n_numeric_features)]
    )
    
    # Create categorical features
    categorical_data = pd.DataFrame()
    for i in range(n_categorical_features):
        X_cat, _ = make_classification(
            n_samples=n_samples,
            n_features=1,
            n_classes=n_classes,
            random_state=random_state + i
        )
        categorical_data[f'categorical_{i}'] = LabelEncoder().fit_transform(X_cat.ravel())
        
    # Combine features
    data = pd.concat([numeric_data, categorical_data], axis=1)
    data['target'] = y
    
    # Create copy with missing values
    data_missing = data.copy()
    for col in data.columns:
        mask = np.random.random(n_samples) < missing_ratio
        data_missing.loc[mask, col] = np.nan
        
    return data, data_missing


def evaluate_imputation(
    imputer: Union[KNNImputer, IterativeImputer, RandomForestImputer, XGBoostImputer, LightGBMImputer],
    data: pd.DataFrame,
    data_missing: pd.DataFrame,
    categorical_features: List[str]
) -> Dict[str, float]:
    """Evaluate imputation performance.
    
    Parameters
    ----------
    imputer : Union[KNNImputer, IterativeImputer, RandomForestImputer, XGBoostImputer, LightGBMImputer]
        Imputer to evaluate.
    data : pd.DataFrame
        Original data without missing values.
    data_missing : pd.DataFrame
        Data with missing values.
    categorical_features : List[str]
        List of categorical feature names.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of evaluation metrics.
    """
    # Record time
    start_time = time.time()
    
    # Fit and transform
    imputer.fit(data_missing)
    data_imputed = imputer.transform(data_missing)
    
    # Calculate time
    imputation_time = time.time() - start_time
    
    # Calculate metrics
    metrics = {'imputation_time': imputation_time}
    
    # Numeric features
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'target':  # Skip target variable
            mask = data_missing[col].isna()
            if mask.any():
                mse = mean_squared_error(
                    data.loc[mask, col],
                    data_imputed.loc[mask, col]
                )
                metrics[f'mse_{col}'] = mse
                
    # Categorical features
    for col in categorical_features:
        mask = data_missing[col].isna()
        if mask.any():
            accuracy = accuracy_score(
                data.loc[mask, col],
                data_imputed.loc[mask, col]
            )
            metrics[f'accuracy_{col}'] = accuracy
            
    # Overall metrics
    metrics['mean_mse'] = np.mean([v for k, v in metrics.items() if k.startswith('mse_')])
    metrics['mean_accuracy'] = np.mean([v for k, v in metrics.items() if k.startswith('accuracy_')])
    
    return metrics


def run_benchmark(
    n_samples: int = 1000,
    n_numeric_features: int = 5,
    n_categorical_features: int = 3,
    missing_ratio: float = 0.2,
    n_repeats: int = 3,
    random_state: int = 42
) -> pd.DataFrame:
    """Run benchmark comparison of different imputers.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples.
    n_numeric_features : int, default=5
        Number of numeric features.
    n_categorical_features : int, default=3
        Number of categorical features.
    missing_ratio : float, default=0.2
        Ratio of missing values to introduce.
    n_repeats : int, default=3
        Number of times to repeat the benchmark.
    random_state : int, default=42
        Random state for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with benchmark results.
    """
    # Create dataset
    data, data_missing = create_mixed_dataset(
        n_samples=n_samples,
        n_numeric_features=n_numeric_features,
        n_categorical_features=n_categorical_features,
        missing_ratio=missing_ratio,
        random_state=random_state
    )
    
    # Get categorical features
    categorical_features = [f'categorical_{i}' for i in range(n_categorical_features)]
    
    # Define imputers to benchmark
    imputers = {
        'KNN': KNNImputer(n_neighbors=5, categorical_features=categorical_features),
        'MICE': IterativeImputer(categorical_features=categorical_features),
        'RandomForest': RandomForestImputer(
            n_estimators=100,
            categorical_features=categorical_features
        ),
        'XGBoost': XGBoostImputer(
            n_estimators=100,
            categorical_features=categorical_features
        ),
        'LightGBM': LightGBMImputer(
            n_estimators=100,
            categorical_features=categorical_features
        )
    }
    
    # Run benchmarks
    results = []
    for _ in range(n_repeats):
        for name, imputer in imputers.items():
            metrics = evaluate_imputation(
                imputer,
                data,
                data_missing,
                categorical_features
            )
            metrics['imputer'] = name
            results.append(metrics)
            
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Aggregate results
    summary = results_df.groupby('imputer').agg({
        'imputation_time': ['mean', 'std'],
        'mean_mse': ['mean', 'std'],
        'mean_accuracy': ['mean', 'std']
    }).round(4)
    
    return summary


if __name__ == '__main__':
    # Run benchmarks with different configurations
    print("Running benchmarks...")
    
    # Small dataset
    print("\nSmall dataset (1000 samples):")
    results_small = run_benchmark(n_samples=1000)
    print(results_small)
    
    # Medium dataset
    print("\nMedium dataset (10000 samples):")
    results_medium = run_benchmark(n_samples=10000)
    print(results_medium)
    
    # Large dataset
    print("\nLarge dataset (50000 samples):")
    results_large = run_benchmark(n_samples=50000)
    print(results_large)
    
    # High-dimensional dataset
    print("\nHigh-dimensional dataset (20 features):")
    results_high_dim = run_benchmark(
        n_numeric_features=10,
        n_categorical_features=10
    )
    print(results_high_dim)
    
    # High missing ratio
    print("\nHigh missing ratio (40%):")
    results_high_missing = run_benchmark(missing_ratio=0.4)
    print(results_high_missing) 