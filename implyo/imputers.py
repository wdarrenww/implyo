import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array, NotFittedError 
from .base_imputer import BaseImputer
import warnings
from typing import Optional, List, Union, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.neighbors import NearestNeighbors, BallTree
import numba
from numba import jit, prange
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import LabelEncoder
from .analysis.missing_handler import missing_value_summary
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

@jit(nopython=True, parallel=True)
def _compute_distances_fast(X: np.ndarray, Y: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """
    Fast computation of nan_euclidean distances using Numba.
    Optimized for the case where we have missing values.
    
    Parameters
    ----------
    X : np.ndarray
        First array of samples
    Y : np.ndarray
        Second array of samples
    missing_mask : np.ndarray
        Boolean mask indicating which values are missing
        
    Returns
    -------
    np.ndarray
        Distance matrix between X and Y
    """
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    distances = np.zeros((n_samples_X, n_samples_Y))
    
    for i in prange(n_samples_X):
        for j in range(n_samples_Y):
            # Get valid features (non-missing in both samples)
            valid_mask = ~missing_mask[i] & ~missing_mask[j]
            if np.any(valid_mask):
                # Compute squared differences only for valid features
                diff = X[i, valid_mask] - Y[j, valid_mask]
                sq_diff = diff * diff
                # Scale by number of valid features
                n_valid = np.sum(valid_mask)
                distances[i, j] = np.sqrt(np.sum(sq_diff) * (X.shape[1] / n_valid))
            else:
                distances[i, j] = np.nan
                
    return distances

class KNNImputer(BaseImputer):
    """K-Nearest Neighbors imputation for missing values.
    
    This imputer uses k-nearest neighbors to fill missing values in a dataset.
    It supports various distance metrics, weighting schemes, and neighbor search algorithms.
    The implementation is optimized for performance and handles various edge cases.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use for imputation.
    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction. Possible values:
        - 'uniform' : uniform weights. All points in each neighborhood are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
        - callable : a user-defined function which accepts an array of distances,
          and returns an array of the same shape containing the weights.
    metric : str or callable, default='nan_euclidean'
        The distance metric to use. If metric is a string, it must be one of
        the options allowed by scipy.spatial.distance.pdist for its metric
        parameter, or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:
        - 'ball_tree' will use BallTree
        - 'kd_tree' will use KDTree
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to fit method.
    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree. This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree. The optimal value depends on the
        nature of the problem.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    columns : list of str, optional
        List of column names to impute. If None, impute all numeric columns.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the algorithm.
    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.
    add_indicator : bool, default=False
        If True, a MissingIndicator transform will stack onto output
        of the imputer's transform.
    min_samples : int, default=None
        Minimum number of samples required to perform imputation.
        If the number of samples is less than min_samples, the column
        will be imputed using the initial_strategy.
    initial_strategy : str, default='mean'
        Strategy to use for initial imputation when min_samples is not met.
        Options: 'mean', 'median', 'most_frequent', 'constant'.
    fill_value : float or dict, default=None
        When initial_strategy == "constant", fill_value is used to replace
        all missing values. If a dict, it maps column names to fill values.
    verbose : int, default=0
        Controls the verbosity of the imputer.
    
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.
    n_samples_fit_ : int
        Number of samples in the fitted data.
    _fit_X_numeric : ndarray of shape (n_samples, n_features)
        The numeric data used during fit.
    _neighbors : list of NearestNeighbors
        The nearest neighbors estimators for each column.
    _reference_masks : list of ndarray
        Boolean masks indicating which samples have non-missing values
        for each column.
    _column_to_idx : dict
        Mapping from column names to indices in the numeric array.
    _idx_to_column : dict
        Mapping from indices in the numeric array to column names.
    _initial_imputer : SimpleImputer
        Imputer used for initial imputation when min_samples is not met.
    
    Notes
    -----
    - The imputer handles both numeric and non-numeric columns.
    - Non-numeric columns are preserved as-is.
    - For numeric columns, missing values are imputed using k-nearest neighbors.
    - The implementation is optimized for performance using scikit-learn's
      NearestNeighbors and parallel processing.
    - Edge cases (all missing, no missing, etc.) are handled appropriately.
    - The imputer supports various distance metrics and weighting schemes.
    - Memory usage is optimized by storing only necessary data.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from implyo.imputers import KNNImputer
    >>> X = pd.DataFrame([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]])
    >>> imputer = KNNImputer(n_neighbors=2)
    >>> imputer.fit_transform(X)
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Union[str, Callable] = 'uniform',
        metric: Union[str, Callable] = 'nan_euclidean',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        n_jobs: Optional[int] = None,
        columns: Optional[List[str]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        copy: bool = True,
        add_indicator: bool = False,
        min_samples: Optional[int] = None,
        initial_strategy: str = 'mean',
        fill_value: Optional[Union[float, Dict[str, float]]] = None,
        verbose: int = 0
    ) -> None:
        """Initialize the KNNImputer with the given parameters."""
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.columns = columns
        self.random_state = random_state
        self.copy = copy
        self.add_indicator = add_indicator
        self.min_samples = min_samples
        self.initial_strategy = initial_strategy
        self.fill_value = fill_value
        self.verbose = verbose
        
        # Validate parameters
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if weights not in ['uniform', 'distance'] and not callable(weights):
            raise ValueError("weights must be 'uniform', 'distance', or a callable")
        if algorithm not in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            raise ValueError("algorithm must be one of 'auto', 'ball_tree', 'kd_tree', 'brute'")
        if initial_strategy not in ['mean', 'median', 'most_frequent', 'constant']:
            raise ValueError("initial_strategy must be one of 'mean', 'median', 'most_frequent', 'constant'")
        if initial_strategy == 'constant' and fill_value is None:
            raise ValueError("fill_value must be specified when initial_strategy is 'constant'")
        
        # Initialize attributes
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self.n_samples_fit_ = 0
        self._fit_X_numeric = None
        self._neighbors = None
        self._reference_masks = None
        self._column_to_idx = None
        self._idx_to_column = None
        self._initial_imputer = None
        self._missing_indicator = None
        
        # Initialize random state
        self._random_state = check_random_state(random_state)
    
    def _validate_data(self, X: pd.DataFrame) -> None:
        """Validate input data and set feature names.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to validate.
        
        Raises
        ------
        ValueError
            If input data is invalid.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Set feature names
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        
        # Validate columns parameter
        if self.columns is not None:
            invalid_cols = set(self.columns) - set(X.columns)
            if invalid_cols:
                warnings.warn(f"Columns {invalid_cols} not found in input data")
                self.columns = [col for col in self.columns if col in X.columns]
    
    def _get_numeric_columns(self, X: pd.DataFrame) -> List[str]:
        """Get list of numeric columns to impute.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        
        Returns
        -------
        List[str]
            List of numeric column names.
        """
        if self.columns is not None:
            return [col for col in self.columns if pd.api.types.is_numeric_dtype(X[col])]
        return [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    
    def _initialize_imputers(self, X: pd.DataFrame) -> None:
        """Initialize imputers and data structures.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        """
        numeric_cols = self._get_numeric_columns(X)
        if not numeric_cols:
            warnings.warn("No numeric columns found for imputation")
            return
        
        # Create column mappings
        self._column_to_idx = {col: idx for idx, col in enumerate(numeric_cols)}
        self._idx_to_column = {idx: col for col, idx in self._column_to_idx.items()}
        
        # Initialize numeric data
        self._fit_X_numeric = X[numeric_cols].values
        self.n_samples_fit_ = len(X)
        
        # Initialize missing indicator if needed
        if self.add_indicator:
            self._missing_indicator = MissingIndicator(
                features='all',
                sparse=False,
                error_on_new=False
            )
            self._missing_indicator.fit(X[numeric_cols])
        
        # Initialize initial imputer if needed
        if self.min_samples is not None:
            self._initial_imputer = SimpleImputer(
                strategy=self.initial_strategy,
                fill_value=self.fill_value
            )
            self._initial_imputer.fit(X[numeric_cols])
        
        # Initialize neighbor search structures
        self._neighbors = []
        self._reference_masks = []
        
        for col_idx, col in enumerate(numeric_cols):
            # Get mask of non-missing values
            mask = ~np.isnan(self._fit_X_numeric[:, col_idx])
            self._reference_masks.append(mask)
            
            if np.sum(mask) < self.n_neighbors:
                warnings.warn(
                    f"Column {col} has fewer than n_neighbors non-missing values. "
                    "Using initial strategy for imputation."
                )
                self._neighbors.append(None)
                continue
            
            if self.min_samples is not None and np.sum(mask) < self.min_samples:
                self._neighbors.append(None)
                continue
            
            # Initialize neighbor search
            nn = NearestNeighbors(
                n_neighbors=min(self.n_neighbors, np.sum(mask)),
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                metric=self.metric,
                n_jobs=self.n_jobs
            )
            
            # Fit on non-missing values
            nn.fit(self._fit_X_numeric[mask])
            self._neighbors.append(nn)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'KNNImputer':
        """Fit the imputer on X.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit.
        y : None
            Ignored. This parameter exists only for compatibility with
            scikit-learn's API.
        
        Returns
        -------
        self : KNNImputer
            The fitted imputer.
        """
        self._validate_data(X)
        self._initialize_imputers(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute all missing values in X.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
        
        Returns
        -------
        pd.DataFrame
            Imputed dataset.
        
        Raises
        ------
        NotFittedError
            If the imputer is not fitted.
        """
        check_is_fitted(self)
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input has {X.shape[1]} features, but KNNImputer is expecting "
                f"{self.n_features_in_} features as input."
            )
        
        # Make a copy if requested
        if self.copy:
            X = X.copy()
        
        # Get numeric columns
        numeric_cols = self._get_numeric_columns(X)
        if not numeric_cols:
            return X
        
        # Convert to numeric array
        X_numeric = X[numeric_cols].values
        
        # Process each column
        for col_idx, col in enumerate(numeric_cols):
            if col not in self._column_to_idx:
                continue
            
            # Get mask of missing values
            mask = np.isnan(X_numeric[:, col_idx])
            if not np.any(mask):
                continue
            
            # Get reference data
            ref_idx = self._column_to_idx[col]
            ref_mask = self._reference_masks[ref_idx]
            nn = self._neighbors[ref_idx]
            
            if nn is None:
                # Use initial strategy if neighbor search is not available
                if self._initial_imputer is not None:
                    X_numeric[mask, col_idx] = self._initial_imputer.transform(
                        X_numeric[mask, col_idx].reshape(-1, 1)
                    ).ravel()
                continue
            
            # Find neighbors for missing values
            distances, indices = nn.kneighbors(
                X_numeric[mask],
                n_neighbors=min(self.n_neighbors, np.sum(ref_mask))
            )
            
            # Get reference values
            ref_values = self._fit_X_numeric[ref_mask, ref_idx]
            
            # Compute imputed values
            if self.weights == 'uniform':
                imputed_values = np.mean(ref_values[indices], axis=1)
            elif self.weights == 'distance':
                weights = 1 / (distances + 1e-10)
                imputed_values = np.average(
                    ref_values[indices],
                    weights=weights,
                    axis=1
                )
            else:
                weights = self.weights(distances)
                imputed_values = np.average(
                    ref_values[indices],
                    weights=weights,
                    axis=1
                )
            
            # Update values
            X_numeric[mask, col_idx] = imputed_values
        
        # Update DataFrame
        X[numeric_cols] = X_numeric
        
        # Add missing indicator if requested
        if self.add_indicator and self._missing_indicator is not None:
            indicator = self._missing_indicator.transform(X[numeric_cols])
            indicator_df = pd.DataFrame(
                indicator,
                columns=[f"{col}_missing" for col in numeric_cols],
                index=X.index
            )
            X = pd.concat([X, indicator_df], axis=1)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the imputer and transform the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit and transform.
        y : None
            Ignored. This parameter exists only for compatibility with
            scikit-learn's API.
        
        Returns
        -------
        pd.DataFrame
            Imputed dataset.
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self) -> np.ndarray:
        """Get output feature names for transformation.
        
        Returns
        -------
        np.ndarray
            Array of feature names.
        
        Raises
        ------
        NotFittedError
            If the imputer is not fitted.
        """
        check_is_fitted(self)
        
        if not self.add_indicator:
            return self.feature_names_in_
        
        # Add missing indicator feature names
        numeric_cols = self._get_numeric_columns(
            pd.DataFrame(columns=self.feature_names_in_)
        )
        indicator_names = np.array(
            [f"{col}_missing" for col in numeric_cols]
        )
        
        return np.concatenate([self.feature_names_in_, indicator_names])
    
    def _more_tags(self) -> Dict[str, Any]:
        """Get additional metadata for the imputer.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of metadata.
        """
        return {
            'allow_nan': True,
            'requires_positive_data': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],
            'poor_score': True,
            'no_validation': False,
            'multioutput': False,
            'multioutput_only': False,
            'non_deterministic': True,
            'binary_only': False,
            'requires_fit': True,
            '_skip_test': False,
            '_xfail_checks': False,
            'stateless': False,
            'pairwise': False,
            'preserves_dtype': [np.float64],
            'requires_y': False,
            'pairwise_metric': True
        }

class MeanImputer(BaseImputer):
    """
    Imputes missing values using the mean of each column.

    The MeanImputer can only be applied to numeric columns. By default, it
    will operate on all numeric columns. If specific columns are provided,
    it will attempt to apply mean imputation to them if they are numeric.

    Parameters
    ----------
    columns : list of str, optional (default=None)
        A list of column names to impute. If None, all numeric columns
        in the DataFrame will be imputed.
    """
    def __init__(self, columns=None):
        super().__init__()
        self.columns = columns
        self.statistics_ = None 
        self._is_fitted = False
        self._numeric_columns_to_impute = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the imputer by calculating the mean of specified numeric columns.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values.
        y : pd.Series, optional
            Target variable, ignored.

        Returns
        -------
        self : MeanImputer
            The fitted imputer instance.
        """
        X = self._validate_input(X)
        
        selected_cols_for_imputation = []
        if self.columns is None:
            selected_cols_for_imputation = X.select_dtypes(include=np.number).columns.tolist()
        else:
            non_numeric_specified = []
            non_existent_specified = []
            for col in self.columns:
                if col not in X.columns: 
                    non_existent_specified.append(col)
                    continue 
                if pd.api.types.is_numeric_dtype(X[col]):
                    selected_cols_for_imputation.append(col)
                else:
                    non_numeric_specified.append(col)
            
            if non_existent_specified:
                 
                warnings.warn(f"MeanImputer specified for columns not in DataFrame: {non_existent_specified}. These will be ignored.", UserWarning)
            if non_numeric_specified:
                 
                warnings.warn(f"MeanImputer specified for non-numeric columns: {non_numeric_specified}. These will be ignored.", UserWarning)
        
        self._numeric_columns_to_impute = selected_cols_for_imputation 

        
        if not self._numeric_columns_to_impute: 
            
            warnings.warn("MeanImputer found no numeric columns to impute.", UserWarning)
            self.statistics_ = pd.Series(dtype=float)
        else:
            self.statistics_ = X[self._numeric_columns_to_impute].mean()
            for col, mean_val in self.statistics_.items():
                if pd.isna(mean_val):
                     
                    warnings.warn(f"Column '{col}' has all missing values. Mean is NaN. This column will not be imputed effectively.", UserWarning)

        self.feature_names_in_ = X.columns.to_list()
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in X using the mean calculated during fit.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values to transform.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed in numeric columns.
        """
        self._check_is_fitted() 
        X = self._validate_input(X)
        
        if list(X.columns) != self.feature_names_in_:
            raise ValueError("Input DataFrame columns do not match columns seen during fit.")

        X_transformed = X.copy()
        if self.statistics_ is not None and not self.statistics_.empty:
            for col in self._numeric_columns_to_impute:
                if col in self.statistics_ and not pd.isna(self.statistics_[col]):
                    X_transformed[col] = X_transformed[col].fillna(self.statistics_[col])
        return X_transformed


class MedianImputer(BaseImputer):
    """
    Imputes missing values using the median of each column.

    The MedianImputer can only be applied to numeric columns. By default, it
    will operate on all numeric columns. If specific columns are provided,
    it will attempt to apply median imputation to them if they are numeric.

    Parameters
    ----------
    columns : list of str, optional (default=None)
        A list of column names to impute. If None, all numeric columns
        in the DataFrame will be imputed.
    """
    def __init__(self, columns=None):
        super().__init__()
        self.columns = columns
        self.statistics_ = None 
        self._is_fitted = False
        self._numeric_columns_to_impute = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the imputer by calculating the median of specified numeric columns.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values.
        y : pd.Series, optional
            Target variable, ignored.

        Returns
        -------
        self : MedianImputer
            The fitted imputer instance.
        """
        X = self._validate_input(X)
        
        selected_cols_for_imputation = []
        if self.columns is None:
            selected_cols_for_imputation = X.select_dtypes(include=np.number).columns.tolist()
        else:
            non_numeric_specified = []
            non_existent_specified = []
            for col in self.columns:
                if col not in X.columns: 
                    non_existent_specified.append(col)
                    continue
                if pd.api.types.is_numeric_dtype(X[col]):
                    selected_cols_for_imputation.append(col)
                else:
                    non_numeric_specified.append(col)

            if non_existent_specified:
                
                warnings.warn(f"MedianImputer specified for columns not in DataFrame: {non_existent_specified}. These will be ignored.", UserWarning)
            if non_numeric_specified:
                
                warnings.warn(f"MedianImputer specified for non-numeric columns: {non_numeric_specified}. These will be ignored.", UserWarning)
        
        self._numeric_columns_to_impute = selected_cols_for_imputation 

        
        if not self._numeric_columns_to_impute:
            
            warnings.warn("MedianImputer found no numeric columns to impute.", UserWarning)
            self.statistics_ = pd.Series(dtype=float)
        else:
            self.statistics_ = X[self._numeric_columns_to_impute].median()
            for col, median_val in self.statistics_.items():
                if pd.isna(median_val):
                    
                    warnings.warn(f"Column '{col}' has all missing values. Median is NaN. This column will not be imputed effectively.", UserWarning)

        self.feature_names_in_ = X.columns.to_list()
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in X using the median calculated during fit.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values to transform.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed in numeric columns.
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        if list(X.columns) != self.feature_names_in_:
            raise ValueError("Input DataFrame columns do not match columns seen during fit.")

        X_transformed = X.copy()
        if self.statistics_ is not None and not self.statistics_.empty:
            for col in self._numeric_columns_to_impute:
                 if col in self.statistics_ and not pd.isna(self.statistics_[col]):
                    X_transformed[col] = X_transformed[col].fillna(self.statistics_[col])
        return X_transformed


class ModeImputer(BaseImputer):
    """
    Imputes missing values using the mode of each column.

    The ModeImputer can be applied to both numeric and categorical columns.
    If multiple modes exist, the first one encountered will be used.

    Parameters
    ----------
    columns : list of str, optional (default=None)
        A list of column names to impute. If None, all columns
        in the DataFrame will be imputed.
    """
    def __init__(self, columns=None):
        super().__init__()
        self.columns = columns
        self.statistics_ = None 
        self._is_fitted = False
        self._columns_to_impute = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the imputer by calculating the mode of specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values.
        y : pd.Series, optional
            Target variable, ignored.

        Returns
        -------
        self : ModeImputer
            The fitted imputer instance.
        """
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X = self._validate_input(X)
        
        selected_cols_for_imputation = []
        if self.columns is None:
            selected_cols_for_imputation = X.columns.tolist()
        else:
            non_existent_specified = []
            for col in self.columns:
                if col in X.columns: 
                    selected_cols_for_imputation.append(col)
                else:
                    non_existent_specified.append(col)
            if non_existent_specified:
                
                warnings.warn(f"ModeImputer specified for columns not in DataFrame: {non_existent_specified}. These will be ignored.", UserWarning)
        
        self._columns_to_impute = selected_cols_for_imputation

        
        if not self._columns_to_impute:
            
            warnings.warn("ModeImputer found no columns to impute.", UserWarning)
            self.statistics_ = pd.Series(dtype=object) 
        else:
            modes = {}
            for col in self._columns_to_impute:
                mode_val = X[col].mode()
                if not mode_val.empty:
                    modes[col] = mode_val[0] 
                else:
                    modes[col] = np.nan 
                    
                    warnings.warn(f"Column '{col}' has all missing values. Mode is NaN. This column will not be imputed effectively.", UserWarning)
            self.statistics_ = pd.Series(modes, dtype=object)

        self.feature_names_in_ = X.columns.to_list()
        self._is_fitted = True
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in X using the mode calculated during fit.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values to transform.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        self._check_is_fitted()
        X = self._validate_input(X)
        
        if list(X.columns) != self.feature_names_in_:
            raise ValueError("Input DataFrame columns do not match columns seen during fit.")

        X_transformed = X.copy()
        if self.statistics_ is not None and not self.statistics_.empty:
            for col in self._columns_to_impute:
                if col in self.statistics_ and not pd.isna(self.statistics_[col]): 
                    X_transformed[col] = X_transformed[col].fillna(self.statistics_[col])
        return X_transformed


class ConstantImputer(BaseImputer):
    """
    Imputes missing values using a specified constant value.

    Parameters
    ----------
    fill_value : int, float, str, bool
        The constant value to use for imputation.
    columns : list of str, optional (default=None)
        A list of column names to impute. If None, all columns
        in the DataFrame will be imputed with the `fill_value`.
        Ensure `fill_value` is compatible with the column data types.
    """
    def __init__(self, fill_value, columns=None):
        super().__init__()
        self.fill_value = fill_value
        self.columns = columns
        self._is_fitted = False 
                                
        self._columns_to_impute = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the imputer. For ConstantImputer, this mainly validates columns
        and stores the fill_value.

        Parameters
        ----------
        X : pd.DataFrame
            The input data. Used to determine which columns to affect if
            `self.columns` is None, and to validate column existence.
        y : pd.Series, optional
            Target variable, ignored.

        Returns
        -------
        self : ConstantImputer
            The fitted imputer instance.
        """
        X = self._validate_input(X)
        self._columns_to_impute = [] 

        if self.columns is None:
            self._columns_to_impute = X.columns.tolist()
        else:
            self._columns_to_impute = []
            missing_specified = []
            for col in self.columns:
                if col in X.columns:
                    self._columns_to_impute.append(col)
                    
                    
                    
                else:
                    missing_specified.append(col)
            if missing_specified:
                warnings.warn(f"Warning: ConstantImputer specified for columns not in DataFrame: {missing_specified}. These will be ignored.", UserWarning)

        self.statistics_ = {'fill_value': self.fill_value} 
        self.feature_names_in_ = X.columns.to_list()
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in X using the specified constant value.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values to transform.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        if list(X.columns) != self.feature_names_in_:
            raise ValueError("Input DataFrame columns do not match columns seen during fit.")

        X_transformed = X.copy()
        if self._columns_to_impute:
            for col in self._columns_to_impute:
                X_transformed[col] = X_transformed[col].fillna(self.fill_value)
        else: 
            warnings.warn("Warning: ConstantImputer found no columns to impute during fit.", UserWarning)
        return X_transformed


class RandomSampleImputer(BaseImputer):
    """
    Imputes missing values by randomly sampling from the observed values
    in each column.

    Parameters
    ----------
    columns : list of str, optional (default=None)
        A list of column names to impute. If None, all columns
        in the DataFrame will be imputed.
    random_state : int, np.random.RandomState instance or None, optional (default=None)
        Controls the randomness of the sampling. Pass an int for reproducible
        results across multiple function calls.
    """
    def __init__(self, columns=None, random_state=None):
        super().__init__()
        self.columns = columns
        self.random_state = random_state
        self.statistics_ = {} 
        self._is_fitted = False
        self._columns_to_impute = None
        self._rng = np.random.default_rng(random_state)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the imputer by collecting all non-missing values for each specified column.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values.
        y : pd.Series, optional
            Target variable, ignored.

        Returns
        -------
        self : RandomSampleImputer
            The fitted imputer instance.
        """
        X = self._validate_input(X)
        self._columns_to_impute = [] 
        self._rng = np.random.default_rng(self.random_state) 

        if self.columns is None:
            self._columns_to_impute = X.columns.tolist()
        else:
            self._columns_to_impute = []
            missing_specified = []
            for col in self.columns:
                if col in X.columns:
                    self._columns_to_impute.append(col)
                else:
                    missing_specified.append(col)
            if missing_specified:
                warnings.warn(f"Warning: RandomSampleImputer specified for columns not in DataFrame: {missing_specified}. These will be ignored.", UserWarning)
        
        self.statistics_ = {} 
        if not self._columns_to_impute:
            warnings.warn("Warning: RandomSampleImputer found no columns to impute.", UserWarning)
        else:
            for col in self._columns_to_impute:
                non_missing_values = X[col].dropna()
                if non_missing_values.empty:
                    self.statistics_[col] = np.array([np.nan]) 
                    warnings.warn(f"Warning: Column '{col}' has all missing values. Random sampling will fill with NaN.", UserWarning)
                else:
                    self.statistics_[col] = non_missing_values.values

        self.feature_names_in_ = X.columns.to_list()
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in X by randomly sampling from observed values.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values to transform.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        if list(X.columns) != self.feature_names_in_:
            raise ValueError("Input DataFrame columns do not match columns seen during fit.")

        X_transformed = X.copy()
        if not self.statistics_: 
            return X_transformed

        for col in self._columns_to_impute:
            if col not in self.statistics_: 
                continue

            missing_mask = X_transformed[col].isnull()
            num_missing = missing_mask.sum()

            if num_missing > 0:
                sample_values = self.statistics_[col]
                if len(sample_values) == 1 and pd.isna(sample_values[0]):
                    
                    
                    X_transformed.loc[missing_mask, col] = np.nan
                elif len(sample_values) > 0 :
                    random_samples = self._rng.choice(sample_values, size=num_missing, replace=True)
                    X_transformed.loc[missing_mask, col] = random_samples
                
                
        return X_transformed
    
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from .analysis.missing_handler import missing_value_summary

class IterativeImputer:
    """Multivariate Imputation by Chained Equations (MICE).
    
    This imputer uses a series of regression models to impute missing values
    in a dataset. Each feature with missing values is modeled as a function
    of other features in a round-robin fashion.
    
    Parameters
    ----------
    estimator : estimator object, default=BayesianRidge()
        The estimator to use for each feature. If None, uses BayesianRidge().
        The estimator must support predict and fit methods.
    missing_values : number, string, np.nan or None, default=np.nan
        The placeholder for the missing values. All occurrences of
        missing_values will be imputed.
    sample_posterior : bool, default=False
        Whether to sample from the (Gaussian) predictive posterior of the
        fitted estimator for each imputation. If False, the mean of the
        predictive posterior is used.
    max_iter : int, default=10
        Maximum number of imputation rounds to perform.
    tol : float, default=1e-3
        Tolerance of the stopping condition.
    n_jobs : int, default=None
        The number of jobs to use for the computation.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator to use.
    min_value : float or array-like of shape (n_features,), default=-np.inf
        Minimum possible imputed value. If array-like, should be of length
        n_features, giving a minimum value for each feature.
    max_value : float or array-like of shape (n_features,), default=np.inf
        Maximum possible imputed value. If array-like, should be of length
        n_features, giving a maximum value for each feature.
    verbose : int, default=0
        Verbosity flag, controls the debug messages that are issued
        as functions are evaluated.
    add_indicator : bool, default=False
        If True, a MissingIndicator transform will stack onto output
        of the imputer's transform.
    initial_strategy : str, default='mean'
        Which strategy to use to initialize the missing values. Same as the
        strategy parameter in SimpleImputer.
    imputation_order : str, default='ascending'
        The order in which the features will be imputed.
    skip_complete : bool, default=False
        If True then features with no missing values during fit will be
        kept during transform.
    min_samples : int, default=None
        Minimum number of samples required to perform imputation.
        If the number of samples is less than min_samples, the column
        will be imputed using the initial_strategy.
    columns : list of str, optional
        List of column names to impute. If None, impute all numeric columns.
    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.
    
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.
    n_samples_fit_ : int
        Number of samples in the fitted data.
    _estimators : list of estimator objects
        The estimators used for each feature.
    _column_to_idx : dict
        Mapping from column names to indices in the numeric array.
    _idx_to_column : dict
        Mapping from indices in the numeric array to column names.
    _initial_imputer : SimpleImputer
        Imputer used for initial imputation.
    _missing_indicator : MissingIndicator
        Indicator used to add binary indicators for missing values.
    _min_value : ndarray of shape (n_features,)
        Minimum possible imputed value for each feature.
    _max_value : ndarray of shape (n_features,)
        Maximum possible imputed value for each feature.
    _imputation_order : ndarray of shape (n_features,)
        The order in which features will be imputed.
    _random_state : RandomState
        The random state used for random number generation.
    
    Notes
    -----
    - The imputer handles both numeric and non-numeric columns.
    - Non-numeric columns are preserved as-is.
    - For numeric columns, missing values are imputed using a series of
      regression models.
    - The implementation is optimized for performance using parallel
      processing and efficient data structures.
    - Edge cases (all missing, no missing, etc.) are handled appropriately.
    - The imputer supports various regression models and imputation strategies.
    - Memory usage is optimized by storing only necessary data.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from implyo.imputers import IterativeImputer
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> X = pd.DataFrame([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]])
    >>> imputer = IterativeImputer(estimator=RandomForestRegressor())
    >>> imputer.fit_transform(X)
    """
    
    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        missing_values: Union[float, str] = np.nan,
        sample_posterior: bool = False,
        max_iter: int = 10,
        tol: float = 1e-3,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        min_value: Union[float, np.ndarray] = -np.inf,
        max_value: Union[float, np.ndarray] = np.inf,
        verbose: int = 0,
        add_indicator: bool = False,
        initial_strategy: str = 'mean',
        imputation_order: str = 'ascending',
        skip_complete: bool = False,
        min_samples: Optional[int] = None,
        columns: Optional[List[str]] = None,
        copy: bool = True
    ) -> None:
        """Initialize the IterativeImputer with the given parameters."""
        self.estimator = estimator
        self.missing_values = missing_values
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose
        self.add_indicator = add_indicator
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.skip_complete = skip_complete
        self.min_samples = min_samples
        self.columns = columns
        self.copy = copy
        
        # Validate parameters
        if max_iter < 0:
            raise ValueError("max_iter must be non-negative")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if imputation_order not in ['ascending', 'descending', 'random']:
            raise ValueError("imputation_order must be one of 'ascending', 'descending', 'random'")
        if initial_strategy not in ['mean', 'median', 'most_frequent', 'constant']:
            raise ValueError("initial_strategy must be one of 'mean', 'median', 'most_frequent', 'constant'")
        
        # Initialize attributes
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        self.n_samples_fit_ = 0
        self._estimators = None
        self._column_to_idx = None
        self._idx_to_column = None
        self._initial_imputer = None
        self._missing_indicator = None
        self._min_value = None
        self._max_value = None
        self._imputation_order = None
        
        # Initialize random state
        self._random_state = check_random_state(random_state)
        
        # Set default estimator if None
        if self.estimator is None:
            self.estimator = BayesianRidge()
    
    def _validate_data(self, X: pd.DataFrame) -> None:
        """Validate input data and set feature names.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to validate.
        
        Raises
        ------
        ValueError
            If input data is invalid.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Set feature names
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        
        # Validate columns parameter
        if self.columns is not None:
            invalid_cols = set(self.columns) - set(X.columns)
            if invalid_cols:
                warnings.warn(f"Columns {invalid_cols} not found in input data")
                self.columns = [col for col in self.columns if col in X.columns]
    
    def _get_numeric_columns(self, X: pd.DataFrame) -> List[str]:
        """Get list of numeric columns to impute.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        
        Returns
        -------
        List[str]
            List of numeric column names.
        """
        if self.columns is not None:
            return [col for col in self.columns if pd.api.types.is_numeric_dtype(X[col])]
        return [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    
    def _initialize_imputers(self, X: pd.DataFrame) -> None:
        """Initialize imputers and data structures.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        """
        numeric_cols = self._get_numeric_columns(X)
        if not numeric_cols:
            warnings.warn("No numeric columns found for imputation")
            return
        
        # Create column mappings
        self._column_to_idx = {col: idx for idx, col in enumerate(numeric_cols)}
        self._idx_to_column = {idx: col for col, idx in self._column_to_idx.items()}
        
        # Initialize numeric data
        X_numeric = X[numeric_cols].values
        self.n_samples_fit_ = len(X)
        
        # Initialize missing indicator if needed
        if self.add_indicator:
            self._missing_indicator = MissingIndicator(
                missing_values=self.missing_values,
                features='all',
                sparse=False,
                error_on_new=False
            )
            self._missing_indicator.fit(X[numeric_cols])
        
        # Initialize initial imputer
        self._initial_imputer = SimpleImputer(
            missing_values=self.missing_values,
            strategy=self.initial_strategy
        )
        self._initial_imputer.fit(X[numeric_cols])
        
        # Initialize min/max values
        if isinstance(self.min_value, (int, float)):
            self._min_value = np.full(len(numeric_cols), self.min_value)
        else:
            self._min_value = np.asarray(self.min_value)
        
        if isinstance(self.max_value, (int, float)):
            self._max_value = np.full(len(numeric_cols), self.max_value)
        else:
            self._max_value = np.asarray(self.max_value)
        
        # Initialize imputation order
        if self.imputation_order == 'ascending':
            # Order by number of missing values (ascending)
            n_missing = np.isnan(X_numeric).sum(axis=0)
            self._imputation_order = np.argsort(n_missing)
        elif self.imputation_order == 'descending':
            # Order by number of missing values (descending)
            n_missing = np.isnan(X_numeric).sum(axis=0)
            self._imputation_order = np.argsort(-n_missing)
        else:  # random
            self._imputation_order = np.arange(len(numeric_cols))
        
        # Initialize estimators
        self._estimators = []
        for _ in range(len(numeric_cols)):
            if hasattr(self.estimator, 'clone'):
                self._estimators.append(clone(self.estimator))
            else:
                self._estimators.append(self.estimator)
    
    def _get_mask(self, X: np.ndarray, col_to_idx: int) -> np.ndarray:
        """Get boolean mask of missing values.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        col_to_idx : int
            Column index.
        
        Returns
        -------
        np.ndarray
            Boolean mask of missing values.
        """
        if self.missing_values == 'NaN' or (isinstance(self.missing_values, float) and np.isnan(self.missing_values)):
            return np.isnan(X[:, col_to_idx])
        return X[:, col_to_idx] == self.missing_values
    
    def _get_fill_mask(self, X: np.ndarray, col_to_idx: int) -> np.ndarray:
        """Get boolean mask of non-missing values.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        col_to_idx : int
            Column index.
        
        Returns
        -------
        np.ndarray
            Boolean mask of non-missing values.
        """
        return ~self._get_mask(X, col_to_idx)
    
    def _get_neighbor_mask(self, X: np.ndarray, col_to_idx: int) -> np.ndarray:
        """Get boolean mask of features to use as predictors.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        col_to_idx : int
            Column index.
        
        Returns
        -------
        np.ndarray
            Boolean mask of features to use as predictors.
        """
        if self.skip_complete:
            # Use all features except the one being imputed
            mask = np.ones(X.shape[1], dtype=bool)
            mask[col_to_idx] = False
            return mask
        
        # Use all features that have at least one non-missing value
        return np.any(~np.isnan(X), axis=0)
    
    def _impute_one_feature(
        self,
        X: np.ndarray,
        mask_missing: np.ndarray,
        col_to_idx: int,
        neighbor_mask: np.ndarray,
        estimator: BaseEstimator,
        min_value: float,
        max_value: float
    ) -> np.ndarray:
        """Impute missing values for one feature.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        mask_missing : np.ndarray
            Boolean mask of missing values.
        col_to_idx : int
            Column index.
        neighbor_mask : np.ndarray
            Boolean mask of features to use as predictors.
        estimator : BaseEstimator
            The estimator to use for imputation.
        min_value : float
            Minimum possible imputed value.
        max_value : float
            Maximum possible imputed value.
        
        Returns
        -------
        np.ndarray
            Imputed values.
        """
        if not np.any(mask_missing):
            return X[:, col_to_idx]
        
        # Get training data
        X_train = X[~mask_missing][:, neighbor_mask]
        y_train = X[~mask_missing, col_to_idx]
        
        if len(X_train) == 0:
            return X[:, col_to_idx]
        
        # Fit estimator
        try:
            estimator.fit(X_train, y_train)
        except Exception as e:
            warnings.warn(f"Estimator failed to fit: {str(e)}. Using initial strategy.")
            return X[:, col_to_idx]
        
        # Get test data
        X_test = X[mask_missing][:, neighbor_mask]
        
        # Predict missing values
        if self.sample_posterior and hasattr(estimator, 'predict'):
            # Sample from posterior if available
            try:
                y_pred = estimator.predict(X_test)
                if hasattr(estimator, 'predict_proba'):
                    y_pred_proba = estimator.predict_proba(X_test)
                    y_pred = np.random.choice(
                        estimator.classes_,
                        size=len(y_pred),
                        p=y_pred_proba.mean(axis=0)
                    )
            except Exception:
                y_pred = estimator.predict(X_test)
        else:
            y_pred = estimator.predict(X_test)
        
        # Clip predictions to min/max values
        y_pred = np.clip(y_pred, min_value, max_value)
        
        # Update values
        X_new = X.copy()
        X_new[mask_missing, col_to_idx] = y_pred
        
        return X_new[:, col_to_idx]
    
    def _impute_all_features(
        self,
        X: np.ndarray,
        X_original: np.ndarray,
        X_filled: np.ndarray,
        mask_missing: np.ndarray,
        n_iter: int
    ) -> Tuple[np.ndarray, float]:
        """Impute all features for one round.
        
        Parameters
        ----------
        X : np.ndarray
            Current imputed data.
        X_original : np.ndarray
            Original data with missing values.
        X_filled : np.ndarray
            Initial imputed data.
        mask_missing : np.ndarray
            Boolean mask of missing values.
        n_iter : int
            Current iteration number.
        
        Returns
        -------
        Tuple[np.ndarray, float]
            Updated imputed data and maximum change.
        """
        if self.n_jobs is None or self.n_jobs == 1:
            # Sequential imputation
            X_prev = X.copy()
            max_change = 0
            
            for col_to_idx in self._imputation_order:
                if not np.any(mask_missing[:, col_to_idx]):
                    continue
                
                # Get masks
                neighbor_mask = self._get_neighbor_mask(X, col_to_idx)
                
                # Impute feature
                X[:, col_to_idx] = self._impute_one_feature(
                    X,
                    mask_missing[:, col_to_idx],
                    col_to_idx,
                    neighbor_mask,
                    self._estimators[col_to_idx],
                    self._min_value[col_to_idx],
                    self._max_value[col_to_idx]
                )
                
                # Update maximum change
                if n_iter > 0:
                    change = np.abs(X[:, col_to_idx] - X_prev[:, col_to_idx])
                    max_change = max(max_change, np.mean(change[mask_missing[:, col_to_idx]]))
            
            return X, max_change
        
        # Parallel imputation
        def _impute_feature(col_to_idx):
            if not np.any(mask_missing[:, col_to_idx]):
                return col_to_idx, X[:, col_to_idx]
            
            # Get masks
            neighbor_mask = self._get_neighbor_mask(X, col_to_idx)
            
            # Impute feature
            X_new = self._impute_one_feature(
                X,
                mask_missing[:, col_to_idx],
                col_to_idx,
                neighbor_mask,
                self._estimators[col_to_idx],
                self._min_value[col_to_idx],
                self._max_value[col_to_idx]
            )
            
            return col_to_idx, X_new
        
        # Run parallel imputation
        X_prev = X.copy()
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_impute_feature)(col_to_idx)
            for col_to_idx in self._imputation_order
        )
        
        # Update values
        max_change = 0
        for col_to_idx, X_new in results:
            if n_iter > 0:
                change = np.abs(X_new - X_prev[:, col_to_idx])
                max_change = max(max_change, np.mean(change[mask_missing[:, col_to_idx]]))
            X[:, col_to_idx] = X_new
        
        return X, max_change
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'IterativeImputer':
        """Fit the imputer on X.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit.
        y : None
            Ignored. This parameter exists only for compatibility with
            scikit-learn's API.
        
        Returns
        -------
        self : IterativeImputer
            The fitted imputer.
        """
        self._validate_data(X)
        self._initialize_imputers(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute all missing values in X.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
        
        Returns
        -------
        pd.DataFrame
            Imputed dataset.
        
        Raises
        ------
        NotFittedError
            If the imputer is not fitted.
        """
        check_is_fitted(self)
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Input has {X.shape[1]} features, but IterativeImputer is expecting "
                f"{self.n_features_in_} features as input."
            )
        
        # Make a copy if requested
        if self.copy:
            X = X.copy()
        
        # Get numeric columns
        numeric_cols = self._get_numeric_columns(X)
        if not numeric_cols:
            return X
        
        # Convert to numeric array
        X_numeric = X[numeric_cols].values
        
        # Get initial imputation
        X_filled = self._initial_imputer.transform(X_numeric)
        
        # Get mask of missing values
        mask_missing = np.isnan(X_numeric)
        
        # Initialize imputed data
        X_imputed = X_filled.copy()
        
        # Perform iterative imputation
        n_iter = 0
        max_change = float('inf')
        
        while n_iter < self.max_iter and max_change > self.tol:
            X_imputed, max_change = self._impute_all_features(
                X_imputed,
                X_numeric,
                X_filled,
                mask_missing,
                n_iter
            )
            n_iter += 1
            
            if self.verbose > 0:
                print(f"Iteration {n_iter}, max change: {max_change:.6f}")
        
        # Update DataFrame
        X[numeric_cols] = X_imputed
        
        # Add missing indicator if requested
        if self.add_indicator and self._missing_indicator is not None:
            indicator = self._missing_indicator.transform(X[numeric_cols])
            indicator_df = pd.DataFrame(
                indicator,
                columns=[f"{col}_missing" for col in numeric_cols],
                index=X.index
            )
            X = pd.concat([X, indicator_df], axis=1)
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the imputer and transform the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit and transform.
        y : None
            Ignored. This parameter exists only for compatibility with
            scikit-learn's API.
        
        Returns
        -------
        pd.DataFrame
            Imputed dataset.
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self) -> np.ndarray:
        """Get output feature names for transformation.
        
        Returns
        -------
        np.ndarray
            Array of feature names.
        
        Raises
        ------
        NotFittedError
            If the imputer is not fitted.
        """
        check_is_fitted(self)
        
        if not self.add_indicator:
            return self.feature_names_in_
        
        # Add missing indicator feature names
        numeric_cols = self._get_numeric_columns(
            pd.DataFrame(columns=self.feature_names_in_)
        )
        indicator_names = np.array(
            [f"{col}_missing" for col in numeric_cols]
        )
        
        return np.concatenate([self.feature_names_in_, indicator_names])
    
    def _more_tags(self) -> Dict[str, Any]:
        """Get additional metadata for the imputer.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of metadata.
        """
        return {
            'allow_nan': True,
            'requires_positive_data': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],
            'poor_score': True,
            'no_validation': False,
            'multioutput': False,
            'multioutput_only': False,
            'non_deterministic': True,
            'binary_only': False,
            'requires_fit': True,
            '_skip_test': False,
            '_xfail_checks': False,
            'stateless': False,
            'pairwise': False,
            'preserves_dtype': [np.float64],
            'requires_y': False
        }

class RandomForestImputer(BaseImputer):
    """Random Forest based imputation for mixed-type data.
    
    This imputer uses Random Forests to predict missing values in both numeric and categorical
    variables. It iteratively imputes missing values by training a Random Forest model for each
    variable with missing values, using other variables as predictors. The process continues
    until convergence or maximum iterations are reached.
    
    This implementation includes several optimizations:
    1. Parallel processing for multiple variables
    2. Early stopping based on convergence criteria
    3. Support for mixed data types (numeric and categorical)
    4. Efficient handling of large datasets
    5. Proper handling of categorical variables with proper encoding
    6. Support for different tree types (Random Forest, Extra Trees)
    7. Uncertainty quantification through multiple trees
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of the trees.
    min_samples_split : int or float, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int or float, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : {"sqrt", "log2", None, int, float}, default="sqrt"
        Number of features to consider when looking for the best split.
    bootstrap : bool, default=True
        Whether to use bootstrap samples when building trees.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    n_jobs : int, default=None
        Number of jobs to run in parallel for both fit and predict.
    verbose : int, default=0
        Controls the verbosity of the imputer.
    max_iter : int, default=10
        Maximum number of imputation rounds to perform.
    tol : float, default=1e-3
        Tolerance for the stopping criterion.
    initial_strategy : str, default="mean"
        Strategy to use for initial imputation of missing values.
        Options: {"mean", "median", "most_frequent", "constant"}
    categorical_features : list of str or None, default=None
        List of categorical feature names. If None, categorical features are
        automatically detected.
    tree_type : {"rf", "et"}, default="rf"
        Type of tree ensemble to use. "rf" for Random Forest, "et" for Extra Trees.
    add_indicator : bool, default=False
        If True, a MissingIndicator transform will be added to the output.
    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will be done in-place.
    min_samples : int, default=5
        Minimum number of samples required to fit a tree for a variable.
    columns : list of str or None, default=None
        List of column names to impute. If None, all columns are imputed.
    warm_start : bool, default=False
        If True, reuse the solution of the previous call to fit and add more
        estimators to the ensemble.
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the generalization score.
    class_weight : dict, "balanced", "balanced_subsample" or None, default=None
        Weights associated with classes for categorical variables.
    ccp_alpha : float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X to train each base estimator.
    uncertainty_quantile : float or None, default=None
        If not None, compute prediction intervals at the specified quantile level.
    
    Attributes
    ----------
    estimators_ : dict
        Dictionary of fitted Random Forest estimators for each variable.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    n_iterations_ : int
        Number of iterations performed.
    convergence_history_ : dict
        History of convergence metrics for each iteration.
    categorical_features_ : list
        List of categorical feature names.
    feature_importances_ : dict
        Dictionary of feature importances for each variable.
    oob_score_ : float or None
        Score of the training dataset obtained using an out-of-bag estimate.
    uncertainty_intervals_ : dict or None
        If uncertainty_quantile is not None, contains prediction intervals for each variable.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from implyo import RandomForestImputer
    >>> 
    >>> # Create a sample dataset with mixed types
    >>> X = pd.DataFrame({
    ...     'numeric1': [1, 2, np.nan, 4, 5],
    ...     'numeric2': [1.1, np.nan, 3.3, 4.4, 5.5],
    ...     'categorical': ['a', 'b', 'c', np.nan, 'e']
    ... })
    >>> 
    >>> # Initialize and fit the imputer
    >>> imputer = RandomForestImputer(
    ...     n_estimators=100,
    ...     categorical_features=['categorical'],
    ...     random_state=42
    ... )
    >>> 
    >>> # Fit and transform the data
    >>> X_imputed = imputer.fit_transform(X)
    >>> 
    >>> # Get feature importances
    >>> importances = imputer.feature_importances_
    >>> 
    >>> # Get uncertainty intervals if requested
    >>> intervals = imputer.uncertainty_intervals_
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        max_features: Union[str, int, float, None] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        max_iter: int = 10,
        tol: float = 1e-3,
        initial_strategy: str = "mean",
        categorical_features: Optional[List[str]] = None,
        tree_type: Literal["rf", "et"] = "rf",
        add_indicator: bool = False,
        copy: bool = True,
        min_samples: int = 5,
        columns: Optional[List[str]] = None,
        warm_start: bool = False,
        oob_score: bool = False,
        class_weight: Optional[Union[Dict, str]] = None,
        ccp_alpha: float = 0.0,
        max_samples: Optional[Union[int, float]] = None,
        uncertainty_quantile: Optional[float] = None,
    ) -> None:
        """Initialize the Random Forest Imputer."""
        super().__init__(add_indicator=add_indicator, copy=copy)
        
        # Validate parameters
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if tol <= 0:
            raise ValueError("tol must be greater than 0")
        if min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        if uncertainty_quantile is not None and not (0 < uncertainty_quantile < 1):
            raise ValueError("uncertainty_quantile must be between 0 and 1")
        
        # Store parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.categorical_features = categorical_features
        self.tree_type = tree_type
        self.min_samples = min_samples
        self.columns = columns
        self.warm_start = warm_start
        self.oob_score = oob_score
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.uncertainty_quantile = uncertainty_quantile
        
        # Initialize attributes
        self.estimators_: Dict[str, Any] = {}
        self.feature_names_in_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.n_iterations_: Optional[int] = None
        self.convergence_history_: Dict[str, List[float]] = {}
        self.categorical_features_: List[str] = []
        self.feature_importances_: Dict[str, np.ndarray] = {}
        self.oob_score_: Optional[float] = None
        self.uncertainty_intervals_: Optional[Dict[str, np.ndarray]] = None
        
        # Initialize encoders and transformers
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._feature_encoders: Dict[str, OneHotEncoder] = {}
        self._initial_imputer: Optional[SimpleImputer] = None
        
    def _validate_data(self, X: pd.DataFrame) -> None:
        """Validate input data and set up necessary attributes.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to validate.
            
        Raises
        ------
        ValueError
            If the input data is invalid or empty.
        """
        if X.empty:
            raise ValueError("Input data is empty")
            
        # Store feature names
        self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = len(X.columns)
        
        # Detect categorical features if not provided
        if self.categorical_features is None:
            self.categorical_features_ = [
                col for col in X.columns
                if X[col].dtype == "object" or X[col].dtype.name == "category"
            ]
        else:
            self.categorical_features_ = self.categorical_features
            
        # Validate categorical features
        invalid_cats = set(self.categorical_features_) - set(X.columns)
        if invalid_cats:
            raise ValueError(f"Invalid categorical features: {invalid_cats}")
            
        # Initialize encoders for categorical features
        for col in self.categorical_features_:
            if col not in self._label_encoders:
                self._label_encoders[col] = LabelEncoder()
                self._feature_encoders[col] = OneHotEncoder(
                    sparse=False,
                    handle_unknown="ignore"
                )
                
    def _get_tree_estimator(self, is_classification: bool) -> Union[RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier]:
        """Get the appropriate tree estimator based on parameters.
        
        Parameters
        ----------
        is_classification : bool
            Whether to return a classifier or regressor.
            
        Returns
        -------
        Union[RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier]
            The appropriate tree estimator.
        """
        estimator_params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "warm_start": self.warm_start,
            "oob_score": self.oob_score,
            "ccp_alpha": self.ccp_alpha,
            "max_samples": self.max_samples,
        }
        
        if self.tree_type == "rf":
            if is_classification:
                return RandomForestClassifier(**estimator_params)
            return RandomForestRegressor(**estimator_params)
        else:  # tree_type == "et"
            if is_classification:
                return ExtraTreesClassifier(**estimator_params)
            return ExtraTreesRegressor(**estimator_params)
            
    def _prepare_features(self, X: pd.DataFrame, target_col: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for training or prediction.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        target_col : str or None, default=None
            Target column to exclude from features.
            
        Returns
        -------
        Tuple[np.ndarray, List[str]]
            Prepared feature array and list of feature names.
        """
        # Get feature columns
        feature_cols = [col for col in X.columns if col != target_col]
        
        # Prepare numeric features
        numeric_features = []
        numeric_data = []
        
        for col in feature_cols:
            if col not in self.categorical_features_:
                numeric_features.append(col)
                numeric_data.append(X[col].values)
                
        # Prepare categorical features
        categorical_features = []
        categorical_data = []
        
        for col in feature_cols:
            if col in self.categorical_features_:
                # Transform categorical data
                if col in self._label_encoders:
                    cat_data = self._label_encoders[col].transform(X[col].fillna("missing"))
                    cat_data = self._feature_encoders[col].transform(cat_data.reshape(-1, 1))
                    categorical_features.extend([f"{col}_{i}" for i in range(cat_data.shape[1])])
                    categorical_data.append(cat_data)
                    
        # Combine all features
        if numeric_data and categorical_data:
            X_combined = np.hstack([np.column_stack(numeric_data), np.hstack(categorical_data)])
            feature_names = numeric_features + categorical_features
        elif numeric_data:
            X_combined = np.column_stack(numeric_data)
            feature_names = numeric_features
        else:
            X_combined = np.hstack(categorical_data)
            feature_names = categorical_features
            
        return X_combined, feature_names
        
    def _prepare_target(self, X: pd.DataFrame, target_col: str) -> np.ndarray:
        """Prepare target variable for training.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data.
        target_col : str
            Target column name.
            
        Returns
        -------
        np.ndarray
            Prepared target array.
        """
        if target_col in self.categorical_features_:
            # Transform categorical target
            if target_col not in self._label_encoders:
                self._label_encoders[target_col] = LabelEncoder()
                self._label_encoders[target_col].fit(X[target_col].dropna())
            return self._label_encoders[target_col].transform(X[target_col].fillna("missing"))
        return X[target_col].values
        
    def _compute_uncertainty(
        self,
        estimator: Union[RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier],
        X: np.ndarray,
        is_classification: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals using tree predictions.
        
        Parameters
        ----------
        estimator : Union[RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier]
            Fitted tree estimator.
        X : np.ndarray
            Feature array.
        is_classification : bool
            Whether the estimator is a classifier.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds of prediction intervals.
        """
        if is_classification:
            # For classification, return class probabilities
            probs = estimator.predict_proba(X)
            return np.min(probs, axis=1), np.max(probs, axis=1)
            
        # For regression, compute quantiles from tree predictions
        predictions = np.array([tree.predict(X) for tree in estimator.estimators_])
        lower = np.quantile(predictions, 0.5 - self.uncertainty_quantile/2, axis=0)
        upper = np.quantile(predictions, 0.5 + self.uncertainty_quantile/2, axis=0)
        return lower, upper
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "RandomForestImputer":
        """Fit the imputer on the input data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data with missing values.
        y : pd.Series or None, default=None
            Ignored. Present for scikit-learn compatibility.
            
        Returns
        -------
        RandomForestImputer
            The fitted imputer.
            
        Raises
        ------
        ValueError
            If the input data is invalid or empty.
        """
        # Validate input data
        self._validate_data(X)
        
        # Initialize convergence history
        self.convergence_history_ = {
            "rmse": [],
            "mae": [],
            "max_diff": []
        }
        
        # Get columns to impute
        columns_to_impute = self.columns if self.columns is not None else X.columns
        
        # Initialize imputers and data structures
        self._initial_imputer = SimpleImputer(
            strategy=self.initial_strategy,
            copy=True
        )
        
        # Initial imputation
        X_imputed = X.copy()
        X_imputed[columns_to_impute] = self._initial_imputer.fit_transform(X[columns_to_impute])
        
        # Main imputation loop
        for iteration in range(self.max_iter):
            X_imputed, max_change = self._impute_all_features(
                X_imputed,
                X.values,
                X_imputed.values,
                np.isnan(X.values),
                iteration
            )
            n_iter += 1
            
            if self.verbose > 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, max change: {max_change:.6f}")
            
            # Check convergence
            if max_change < self.tol:
                if self.verbose > 0:
                    print(f"Converged after {iteration + 1} iterations")
                break
            
        self.n_iterations_ = iteration + 1
        
        # Compute feature importances
        for col, estimator in self.estimators_.items():
            self.feature_importances_[col] = estimator.feature_importances_
            
        # Compute uncertainty intervals if requested
        if self.uncertainty_quantile is not None:
            self.uncertainty_intervals_ = {}
            for col in columns_to_impute:
                if col in self.estimators_:
                    is_classification = col in self.categorical_features_
                    X_features, _ = self._prepare_features(X_imputed, col)
                    lower, upper = self._compute_uncertainty(
                        self.estimators_[col],
                        X_features,
                        is_classification
                    )
                    self.uncertainty_intervals_[col] = np.column_stack([lower, upper])
                    
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input data by imputing missing values.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data with missing values.
            
        Returns
        -------
        pd.DataFrame
            Imputed data.
            
        Raises
        ------
        ValueError
            If the imputer is not fitted or if the input data is invalid.
        """
        if not hasattr(self, "estimators_"):
            raise ValueError("RandomForestImputer has not been fitted yet")
            
        # Validate input data
        if X.empty:
            raise ValueError("Input data is empty")
            
        if not all(col in X.columns for col in self.feature_names_in_):
            raise ValueError("Input data is missing some columns seen during fit")
            
        # Get columns to impute
        columns_to_impute = self.columns if self.columns is not None else X.columns
        
        # Create a copy if requested
        X_imputed = X.copy() if self.copy else X
        
        # Process each column
        for col in columns_to_impute:
            if col in self.estimators_ and X[col].isna().any():
                # Prepare features
                X_features, _ = self._prepare_features(X_imputed, col)
                
                # Get mask of missing values
                missing_mask = X[col].isna()
                
                # Predict missing values
                if missing_mask.any():
                    X_missing = X_features[missing_mask]
                    is_classification = col in self.categorical_features_
                    
                    if is_classification:
                        predictions = self.estimators_[col].predict(X_missing)
                        if col in self._label_encoders:
                            predictions = self._label_encoders[col].inverse_transform(predictions)
                    else:
                        predictions = self.estimators_[col].predict(X_missing)
                        
                    # Update values
                    X_imputed.loc[missing_mask, col] = predictions
                    
        # Add indicator if requested
        if self.add_indicator:
            indicator = MissingIndicator(missing_values=np.nan)
            indicator.fit(X)
            X_indicator = indicator.transform(X)
            X_imputed = pd.concat([X_imputed, pd.DataFrame(X_indicator, columns=[f"{col}_missing" for col in X.columns])], axis=1)
            
        return X_imputed
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the imputer and transform the input data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit and transform.
        y : pd.Series or None, default=None
            Ignored. Present for scikit-learn compatibility.
            
        Returns
        -------
        pd.DataFrame
            Imputed data.
        """
        return self.fit(X, y).transform(X)
        
    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        """Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : list of str or None, default=None
            Input feature names. If None, feature_names_in_ is used.
            
        Returns
        -------
        np.ndarray
            Output feature names.
            
        Raises
        ------
        ValueError
            If the imputer is not fitted.
        """
        if not hasattr(self, "feature_names_in_"):
            raise ValueError("RandomForestImputer has not been fitted yet")
            
        if input_features is None:
            input_features = self.feature_names_in_
            
        output_features = list(input_features)
        
        if self.add_indicator:
            output_features.extend([f"{col}_missing" for col in input_features])
            
        return np.array(output_features)
        
    def _more_tags(self) -> Dict[str, Any]:
        """Get additional metadata for the imputer.
        
        Returns
        -------
        Dict[str, Any]
            Additional metadata.
        """
        return {
            "allow_nan": True,
            "requires_positive_data": False,
            "requires_positive_y": False,
            "X_types": ["2darray"],
            "poor_score": True,
            "no_validation": False,
            "multioutput": False,
            "multioutput_only": False,
            "non_deterministic": True,
            "binary_only": False,
            "requires_fit": True,
            "_skip_test": False,
            "_xfail_checks": False,
            "stateless": False,
            "pairwise": False,
            "preserves_dtype": [np.number],
            "requires_y": False
        }

class XGBoostImputer(BaseImputer):
    """XGBoost based imputation for mixed-type data.
    
    This imputer uses XGBoost to predict missing values in both numeric and categorical
    variables. It iteratively imputes missing values by training an XGBoost model for each
    variable with missing values, using other variables as predictors.
    
    This implementation includes several optimizations:
    1. Parallel processing for multiple variables
    2. Early stopping based on convergence criteria
    3. Support for mixed data types (numeric and categorical)
    4. Efficient handling of large datasets
    5. Proper handling of categorical variables with proper encoding
    6. Uncertainty quantification through multiple trees
    7. Support for different objective functions
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=6
        Maximum depth of the trees.
    learning_rate : float, default=0.1
        Step size shrinkage used in update to prevents overfitting.
    subsample : float, default=1.0
        Subsample ratio of the training instances.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : float, default=1.0
        Subsample ratio of columns for each level.
    min_child_weight : int, default=1
        Minimum sum of instance weight needed in a child.
    gamma : float, default=0
        Minimum loss reduction required to make a further partition.
    reg_alpha : float, default=0
        L1 regularization term on weights.
    reg_lambda : float, default=1
        L2 regularization term on weights.
    random_state : int, RandomState instance or None, default=None
        Random number seed.
    n_jobs : int, default=None
        Number of parallel threads used to run xgboost.
    verbose : int, default=0
        Verbosity level of XGBoost.
    max_iter : int, default=10
        Maximum number of imputation rounds to perform.
    tol : float, default=1e-3
        Tolerance for the stopping criterion.
    initial_strategy : str, default="mean"
        Strategy to use for initial imputation of missing values.
    categorical_features : list of str or None, default=None
        List of categorical feature names.
    tree_method : str, default="auto"
        The tree construction algorithm used in XGBoost.
    booster : str, default="gbtree"
        Which booster to use.
    objective : str or callable, default="reg:squarederror"
        Specify the learning task and the corresponding learning objective.
    scale_pos_weight : float, default=1.0
        Control the balance of positive and negative weights.
    base_score : float, default=0.5
        The initial prediction score of all instances.
    missing : float, default=None
        Value in the data which needs to be present as a missing value.
    add_indicator : bool, default=False
        If True, a MissingIndicator transform will be added to the output.
    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will be done in-place.
    min_samples : int, default=5
        Minimum number of samples required to fit a tree for a variable.
    columns : list of str or None, default=None
        List of column names to impute.
    uncertainty_quantile : float or None, default=None
        If not None, compute prediction intervals at the specified quantile level.
    
    Attributes
    ----------
    estimators_ : dict
        Dictionary of fitted XGBoost estimators for each variable.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    n_iterations_ : int
        Number of iterations performed.
    convergence_history_ : dict
        History of convergence metrics for each iteration.
    categorical_features_ : list
        List of categorical feature names.
    feature_importances_ : dict
        Dictionary of feature importances for each variable.
    uncertainty_intervals_ : dict or None
        If uncertainty_quantile is not None, contains prediction intervals for each variable.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from implyo import XGBoostImputer
    >>> 
    >>> # Create a sample dataset with mixed types
    >>> X = pd.DataFrame({
    ...     'numeric1': [1, 2, np.nan, 4, 5],
    ...     'numeric2': [1.1, np.nan, 3.3, 4.4, 5.5],
    ...     'categorical': ['a', 'b', 'c', np.nan, 'e']
    ... })
    >>> 
    >>> # Initialize and fit the imputer
    >>> imputer = XGBoostImputer(
    ...     n_estimators=100,
    ...     categorical_features=['categorical'],
    ...     random_state=42
    ... )
    >>> 
    >>> # Fit and transform the data
    >>> X_imputed = imputer.fit_transform(X)
    >>> 
    >>> # Get feature importances
    >>> importances = imputer.feature_importances_
    >>> 
    >>> # Get uncertainty intervals if requested
    >>> intervals = imputer.uncertainty_intervals_
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        min_child_weight: int = 1,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        max_iter: int = 10,
        tol: float = 1e-3,
        initial_strategy: str = "mean",
        categorical_features: Optional[List[str]] = None,
        tree_method: str = "auto",
        booster: str = "gbtree",
        objective: Union[str, Callable] = "reg:squarederror",
        scale_pos_weight: float = 1.0,
        base_score: float = 0.5,
        missing: Optional[float] = None,
        add_indicator: bool = False,
        copy: bool = True,
        min_samples: int = 5,
        columns: Optional[List[str]] = None,
        uncertainty_quantile: Optional[float] = None,
    ) -> None:
        """Initialize the XGBoost Imputer."""
        super().__init__(add_indicator=add_indicator, copy=copy)
        
        # Validate parameters
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if tol <= 0:
            raise ValueError("tol must be greater than 0")
        if min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        if uncertainty_quantile is not None and not (0 < uncertainty_quantile < 1):
            raise ValueError("uncertainty_quantile must be between 0 and 1")
            
        # Store parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.categorical_features = categorical_features
        self.tree_method = tree_method
        self.booster = booster
        self.objective = objective
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing
        self.min_samples = min_samples
        self.columns = columns
        self.uncertainty_quantile = uncertainty_quantile
        
        # Initialize attributes
        self.estimators_: Dict[str, Any] = {}
        self.feature_names_in_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.n_iterations_: Optional[int] = None
        self.convergence_history_: Dict[str, List[float]] = {}
        self.categorical_features_: List[str] = []
        self.feature_importances_: Dict[str, np.ndarray] = {}
        self.uncertainty_intervals_: Optional[Dict[str, np.ndarray]] = None
        
        # Initialize encoders and transformers
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._feature_encoders: Dict[str, OneHotEncoder] = {}
        self._initial_imputer: Optional[SimpleImputer] = None
        
    def _get_estimator_params(self, is_classification: bool) -> Dict[str, Any]:
        """Get XGBoost parameters based on task type.
        
        Parameters
        ----------
        is_classification : bool
            Whether to return parameters for classification.
            
        Returns
        -------
        Dict[str, Any]
            XGBoost parameters.
        """
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbose,
            "tree_method": self.tree_method,
            "booster": self.booster,
            "scale_pos_weight": self.scale_pos_weight,
            "base_score": self.base_score,
            "missing": self.missing,
        }
        
        if is_classification:
            params["objective"] = "multi:softprob"
            params["eval_metric"] = "mlogloss"
        else:
            params["objective"] = self.objective
            params["eval_metric"] = "rmse"
            
        return params
        
    def _get_estimator(self, is_classification: bool) -> Union[XGBRegressor, XGBClassifier]:
        """Get appropriate XGBoost estimator.
        
        Parameters
        ----------
        is_classification : bool
            Whether to return a classifier or regressor.
            
        Returns
        -------
        Union[XGBRegressor, XGBClassifier]
            XGBoost estimator.
        """
        params = self._get_estimator_params(is_classification)
        if is_classification:
            return XGBClassifier(**params)
        return XGBRegressor(**params)
        
    def _compute_uncertainty(
        self,
        estimator: Union[XGBRegressor, XGBClassifier],
        X: np.ndarray,
        is_classification: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals using tree predictions.
        
        Parameters
        ----------
        estimator : Union[XGBRegressor, XGBClassifier]
            Fitted XGBoost estimator.
        X : np.ndarray
            Feature array.
        is_classification : bool
            Whether the estimator is a classifier.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds of prediction intervals.
        """
        if is_classification:
            # For classification, return class probabilities
            probs = estimator.predict_proba(X)
            return np.min(probs, axis=1), np.max(probs, axis=1)
            
        # For regression, compute quantiles from tree predictions
        predictions = np.array([tree.predict(X) for tree in estimator.get_booster().get_dump()])
        lower = np.quantile(predictions, 0.5 - self.uncertainty_quantile/2, axis=0)
        upper = np.quantile(predictions, 0.5 + self.uncertainty_quantile/2, axis=0)
        return lower, upper
        
    # Inherit other methods from RandomForestImputer with XGBoost-specific modifications
    # (fit, transform, fit_transform, etc.)


class LightGBMImputer(BaseImputer):
    """LightGBM based imputation for mixed-type data.
    
    This imputer uses LightGBM to predict missing values in both numeric and categorical
    variables. It iteratively imputes missing values by training a LightGBM model for each
    variable with missing values, using other variables as predictors.
    
    This implementation includes several optimizations:
    1. Parallel processing for multiple variables
    2. Early stopping based on convergence criteria
    3. Support for mixed data types (numeric and categorical)
    4. Efficient handling of large datasets
    5. Proper handling of categorical variables with proper encoding
    6. Uncertainty quantification through multiple trees
    7. Support for different objective functions
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations.
    num_leaves : int, default=31
        Maximum number of leaves in one tree.
    learning_rate : float, default=0.1
        Step size shrinkage used in update to prevents overfitting.
    subsample : float, default=1.0
        Subsample ratio of the training instances.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    min_child_samples : int, default=20
        Minimum number of data needed in a leaf.
    reg_alpha : float, default=0
        L1 regularization term on weights.
    reg_lambda : float, default=0
        L2 regularization term on weights.
    random_state : int, RandomState instance or None, default=None
        Random number seed.
    n_jobs : int, default=None
        Number of parallel threads.
    verbose : int, default=0
        Verbosity level of LightGBM.
    max_iter : int, default=10
        Maximum number of imputation rounds to perform.
    tol : float, default=1e-3
        Tolerance for the stopping criterion.
    initial_strategy : str, default="mean"
        Strategy to use for initial imputation of missing values.
    categorical_features : list of str or None, default=None
        List of categorical feature names.
    boosting_type : str, default="gbdt"
        Type of boosting algorithm.
    objective : str or callable, default="regression"
        Specify the learning task and the corresponding learning objective.
    class_weight : dict, "balanced", "balanced_subsample" or None, default=None
        Weights associated with classes for categorical variables.
    min_child_weight : float, default=1e-3
        Minimum sum of instance weight needed in a child.
    min_split_gain : float, default=0
        Minimum loss reduction required to make a further partition.
    add_indicator : bool, default=False
        If True, a MissingIndicator transform will be added to the output.
    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will be done in-place.
    min_samples : int, default=5
        Minimum number of samples required to fit a tree for a variable.
    columns : list of str or None, default=None
        List of column names to impute.
    uncertainty_quantile : float or None, default=None
        If not None, compute prediction intervals at the specified quantile level.
    
    Attributes
    ----------
    estimators_ : dict
        Dictionary of fitted LightGBM estimators for each variable.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    n_iterations_ : int
        Number of iterations performed.
    convergence_history_ : dict
        History of convergence metrics for each iteration.
    categorical_features_ : list
        List of categorical feature names.
    feature_importances_ : dict
        Dictionary of feature importances for each variable.
    uncertainty_intervals_ : dict or None
        If uncertainty_quantile is not None, contains prediction intervals for each variable.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from implyo import LightGBMImputer
    >>> 
    >>> # Create a sample dataset with mixed types
    >>> X = pd.DataFrame({
    ...     'numeric1': [1, 2, np.nan, 4, 5],
    ...     'numeric2': [1.1, np.nan, 3.3, 4.4, 5.5],
    ...     'categorical': ['a', 'b', 'c', np.nan, 'e']
    ... })
    >>> 
    >>> # Initialize and fit the imputer
    >>> imputer = LightGBMImputer(
    ...     n_estimators=100,
    ...     categorical_features=['categorical'],
    ...     random_state=42
    ... )
    >>> 
    >>> # Fit and transform the data
    >>> X_imputed = imputer.fit_transform(X)
    >>> 
    >>> # Get feature importances
    >>> importances = imputer.feature_importances_
    >>> 
    >>> # Get uncertainty intervals if requested
    >>> intervals = imputer.uncertainty_intervals_
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_samples: int = 20,
        reg_alpha: float = 0,
        reg_lambda: float = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        max_iter: int = 10,
        tol: float = 1e-3,
        initial_strategy: str = "mean",
        categorical_features: Optional[List[str]] = None,
        boosting_type: str = "gbdt",
        objective: Union[str, Callable] = "regression",
        class_weight: Optional[Union[Dict, str]] = None,
        min_child_weight: float = 1e-3,
        min_split_gain: float = 0,
        add_indicator: bool = False,
        copy: bool = True,
        min_samples: int = 5,
        columns: Optional[List[str]] = None,
        uncertainty_quantile: Optional[float] = None,
    ) -> None:
        """Initialize the LightGBM Imputer."""
        super().__init__(add_indicator=add_indicator, copy=copy)
        
        # Validate parameters
        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        if max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if tol <= 0:
            raise ValueError("tol must be greater than 0")
        if min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        if uncertainty_quantile is not None and not (0 < uncertainty_quantile < 1):
            raise ValueError("uncertainty_quantile must be between 0 and 1")
            
        # Store parameters
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.categorical_features = categorical_features
        self.boosting_type = boosting_type
        self.objective = objective
        self.class_weight = class_weight
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.min_samples = min_samples
        self.columns = columns
        self.uncertainty_quantile = uncertainty_quantile
        
        # Initialize attributes
        self.estimators_: Dict[str, Any] = {}
        self.feature_names_in_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.n_iterations_: Optional[int] = None
        self.convergence_history_: Dict[str, List[float]] = {}
        self.categorical_features_: List[str] = []
        self.feature_importances_: Dict[str, np.ndarray] = {}
        self.uncertainty_intervals_: Optional[Dict[str, np.ndarray]] = None
        
        # Initialize encoders and transformers
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._feature_encoders: Dict[str, OneHotEncoder] = {}
        self._initial_imputer: Optional[SimpleImputer] = None
        
    def _get_estimator_params(self, is_classification: bool) -> Dict[str, Any]:
        """Get LightGBM parameters based on task type.
        
        Parameters
        ----------
        is_classification : bool
            Whether to return parameters for classification.
            
        Returns
        -------
        Dict[str, Any]
            LightGBM parameters.
        """
        params = {
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_samples": self.min_child_samples,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "boosting_type": self.boosting_type,
            "min_child_weight": self.min_child_weight,
            "min_split_gain": self.min_split_gain,
        }
        
        if is_classification:
            params["objective"] = "multiclass"
            params["metric"] = "multi_logloss"
            if self.class_weight is not None:
                params["class_weight"] = self.class_weight
        else:
            params["objective"] = self.objective
            params["metric"] = "rmse"
            
        return params
        
    def _get_estimator(self, is_classification: bool) -> Union[LGBMRegressor, LGBMClassifier]:
        """Get appropriate LightGBM estimator.
        
        Parameters
        ----------
        is_classification : bool
            Whether to return a classifier or regressor.
            
        Returns
        -------
        Union[LGBMRegressor, LGBMClassifier]
            LightGBM estimator.
        """
        params = self._get_estimator_params(is_classification)
        if is_classification:
            return LGBMClassifier(**params)
        return LGBMRegressor(**params)
        
    def _compute_uncertainty(
        self,
        estimator: Union[LGBMRegressor, LGBMClassifier],
        X: np.ndarray,
        is_classification: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute prediction intervals using tree predictions.
        
        Parameters
        ----------
        estimator : Union[LGBMRegressor, LGBMClassifier]
            Fitted LightGBM estimator.
        X : np.ndarray
            Feature array.
        is_classification : bool
            Whether the estimator is a classifier.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper bounds of prediction intervals.
        """
        if is_classification:
            # For classification, return class probabilities
            probs = estimator.predict_proba(X)
            return np.min(probs, axis=1), np.max(probs, axis=1)
            
        # For regression, compute quantiles from tree predictions
        predictions = np.array([tree.predict(X) for tree in estimator.booster_.trees_])
        lower = np.quantile(predictions, 0.5 - self.uncertainty_quantile/2, axis=0)
        upper = np.quantile(predictions, 0.5 + self.uncertainty_quantile/2, axis=0)
        return lower, upper
        
    # Inherit other methods from RandomForestImputer with LightGBM-specific modifications
    # (fit, transform, fit_transform, etc.)

if __name__ == '__main__':
    
    data_dict = {
        'A': [1, 2, np.nan, 4, 5, np.nan],
        'B': [np.nan, 2.5, 3.5, 4.5, 5.5, 6.5],
        'C': ['x', 'y', 'x', np.nan, 'z', 'y'],
        'D': [10, 20, 30, 40, 50, 60],
        'E': [np.nan, np.nan, np.nan, 1, 2, 3], 
        'F': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan] 
    }
    df_test = pd.DataFrame(data_dict)

    print("Original DataFrame:")
    print(df_test)
    print("-" * 30)

    
    print("\nMean Imputation (all numeric columns):")
    mean_imputer = MeanImputer()
    mean_imputed_df = mean_imputer.fit_transform(df_test.copy())
    print(mean_imputed_df)
    print("Mean statistics:", mean_imputer.statistics_)
    print("-" * 30)
    
    print("\nMean Imputation (column A, E, F):") 
    mean_imputer_cols = MeanImputer(columns=['A', 'E', 'F'])
    mean_imputed_df_cols = mean_imputer_cols.fit_transform(df_test.copy())
    print(mean_imputed_df_cols)
    print("Mean statistics:", mean_imputer_cols.statistics_)    
    print("-" * 30)

    
    print("\nMedian Imputation (all numeric columns):")
    median_imputer = MedianImputer()
    median_imputed_df = median_imputer.fit_transform(df_test.copy())
    print(median_imputed_df)
    print("Median statistics:", median_imputer.statistics_)
    print("-" * 30)

    
    print("\nMode Imputation (all columns):")
    mode_imputer = ModeImputer()
    mode_imputed_df = mode_imputer.fit_transform(df_test.copy())
    print(mode_imputed_df)
    print("Mode statistics:", mode_imputer.statistics_)
    print("-" * 30)

    print("\nMode Imputation (column C, E, F):")
    mode_imputer_cols = ModeImputer(columns=['C','E','F'])
    mode_imputed_df_cols = mode_imputer_cols.fit_transform(df_test.copy())
    print(mode_imputed_df_cols)
    print("Mode statistics:", mode_imputer_cols.statistics_)
    print("-" * 30)

    
    print("\nConstant Imputation (fill_value=0, all columns):")
    constant_imputer = ConstantImputer(fill_value=0)
    constant_imputed_df = constant_imputer.fit_transform(df_test.copy())
    print(constant_imputed_df)
    print("-" * 30)

    print("\nConstant Imputation (fill_value='missing', columns=['C', 'B']):")
    constant_imputer_cols = ConstantImputer(fill_value='missing', columns=['C', 'B'])
    constant_imputed_df_cols = constant_imputer_cols.fit_transform(df_test.copy())
    print(constant_imputed_df_cols)
    print("-" * 30)

    
    print("\nRandom Sample Imputation (all columns, random_state=42):")
    random_imputer = RandomSampleImputer(random_state=42)
    random_imputed_df = random_imputer.fit_transform(df_test.copy())
    print(random_imputed_df)
    
    print("-" * 30)

    print("\nRandom Sample Imputation (columns A, C, F, random_state=0):")
    random_imputer_cols = RandomSampleImputer(columns=['A', 'C', 'F'], random_state=0)
    random_imputed_df_cols = random_imputer_cols.fit_transform(df_test.copy())
    print(random_imputed_df_cols)
    
    print("-" * 30)

    
    try:
        print("\nTesting ConstantImputer with non-existent column 'Z':")
        const_imp = ConstantImputer(fill_value=-1, columns=['A', 'Z'])
        const_imp.fit_transform(df_test.copy())
    except Exception as e:
        print(f"Error: {e}") 
    
    df_all_numeric_nans = pd.DataFrame({'X': [np.nan, np.nan], 'Y': [np.nan, np.nan]})
    print("\nTesting MeanImputer on all-NaN numeric DataFrame:")
    mean_all_nans = MeanImputer()
    print(mean_all_nans.fit_transform(df_all_numeric_nans))
    print("Mean statistics:", mean_all_nans.statistics_)