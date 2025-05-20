
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array 
from .base_imputer import BaseImputer
import warnings


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