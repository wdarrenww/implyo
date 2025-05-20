from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class BaseImputer(BaseEstimator, TransformerMixin, ABC):
    """
    Base class for all imputers in PiP-Impute.

    This class provides the basic structure and common methods for imputers,
    ensuring compatibility with scikit-learn pipelines.

    All imputers should inherit from this class and implement the `fit`
    and `transform` methods.
    """

    def __init__(self):
        """
        Initializes the BaseImputer.
        Subclasses should call super().__init__() and then initialize
        their specific parameters.
        """
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the imputer on the provided data.

        This method should learn the imputation strategy from the input data X.
        For example, for a mean imputer, it would calculate the mean of each column.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values, where columns are features and
            rows are samples.
        y : pd.Series, optional
            The target variable, ignored in most unsupervised imputers.
            (default is None)

        Returns
        -------
        self : object
            The fitted imputer instance.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("Input DataFrame X cannot be empty.")
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in X using the learned imputation strategy.

        This method should apply the imputation strategy learned during the `fit`
        phase to the input data X.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values to be transformed.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("Input DataFrame X cannot be empty.")

        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            pass

        return X.copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None, **fit_params) -> pd.DataFrame:
        """
        Fit the imputer on X and then transform X.

        Equivalent to calling fit(X, y, **fit_params).transform(X), but
        potentially more efficient.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with missing values.
        y : pd.Series, optional
            The target variable. (default is None)
        **fit_params : dict
            Additional parameters to pass to the `fit` method.

        Returns
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        return self.fit(X, y, **fit_params).transform(X)

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the input DataFrame.
        Can be extended by subclasses for more specific validation.
        """
        if not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except Exception as e:
                raise TypeError(f"Input X could not be converted to a pandas DataFrame. Error: {e}")
        return X

    def _check_is_fitted(self):
        """
        Performs checks to ensure the imputer has been fitted.
        Relies on scikit-learn's check_is_fitted by checking for attributes
        that end with an underscore. Subclasses must set such attributes in `fit`.
        Example: self.statistics_ = ...
        """
        check_is_fitted(self)