import pandas as pd
import numpy as np
from typing import List, Union, Optional

def pool_means(imputed_datasets: List[pd.DataFrame], 
               target_vars: Optional[Union[str, List[str]]] = None) -> Union[float, pd.Series]:
    """
    Pools the means from multiple imputed datasets.

    Calculates the mean of a target variable (or variables) across
    multiple imputed datasets and then averages these means.

    Parameters
    ----------
    imputed_datasets : List[pd.DataFrame]
        A list of m pandas DataFrames, where each DataFrame is an imputed dataset.
        All DataFrames should have the same columns.
    target_vars : str or List[str], optional
        The name of the column (or a list of column names) for which to pool means.
        If None, attempts to pool means for all numeric columns.

    Returns
    -------
    float or pd.Series
        The pooled mean if a single target_vars is provided.
        A pandas Series of pooled means if multiple target_vars are processed,
        indexed by variable names.

    Raises
    ------
    ValueError
        If imputed_datasets is empty, or DataFrames have different columns,
        or target_vars are not found or are non-numeric.
    """
    if not imputed_datasets:
        raise ValueError("imputed_datasets list cannot be empty.")
    
    m = len(imputed_datasets)
    if m == 0:
        raise ValueError("imputed_datasets list cannot be empty.")

    # Validate input DataFrames
    first_df_cols = imputed_datasets[0].columns
    for i, df in enumerate(imputed_datasets):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Element {i} in imputed_datasets is not a pandas DataFrame.")
        if not df.columns.equals(first_df_cols):
            raise ValueError(f"DataFrame {i} has different columns than the first DataFrame.")

    df_template = imputed_datasets[0] # Use for column checks

    if target_vars is None:
        # Default to all numeric columns
        vars_to_pool = df_template.select_dtypes(include=np.number).columns.tolist()
        if not vars_to_pool:
            raise ValueError("No numeric columns found to pool means.")
    elif isinstance(target_vars, str):
        vars_to_pool = [target_vars]
    elif isinstance(target_vars, list):
        vars_to_pool = target_vars
    else:
        raise TypeError("target_vars must be a string, list of strings, or None.")

    # Validate target_vars
    for var_name in vars_to_pool:
        if var_name not in df_template.columns:
            raise ValueError(f"Target variable '{var_name}' not found in DataFrames.")
        if not pd.api.types.is_numeric_dtype(df_template[var_name]):
            raise ValueError(f"Target variable '{var_name}' is not numeric and cannot be averaged.")

    all_means = []
    for df in imputed_datasets:
        all_means.append(df[vars_to_pool].mean()) # df.mean() gives a Series

    # `all_means` is a list of Series. Concatenate them into a DataFrame.
    means_df = pd.concat(all_means, axis=1) # Each column is a dataset's means, rows are variables
    pooled_q = means_df.mean(axis=1) # Average across datasets (axis=1)

    if isinstance(target_vars, str) and len(vars_to_pool) == 1:
        return pooled_q.iloc[0]
    return pooled_q


def pool_variances_rubin(imputed_datasets: List[pd.DataFrame],
                         target_vars: Optional[Union[str, List[str]]] = None,
                         pooled_means_input: Optional[Union[float, pd.Series]] = None
                        ) -> Union[float, pd.Series]:
    """
    Pools variances from multiple imputed datasets using Rubin's Rules.

    This function calculates the total variance for an estimated mean,
    combining within-imputation and between-imputation variances.

    Parameters
    ----------
    imputed_datasets : List[pd.DataFrame]
        A list of m pandas DataFrames, each an imputed dataset.
    target_vars : str or List[str], optional
        The column name(s) for which to pool variances. If None, pools for
        all numeric columns.
    pooled_means_input : float or pd.Series, optional
        Pre-calculated pooled mean(s) for the target_vars. If None, they
        will be calculated internally. Should match the structure of target_vars.

    Returns
    -------
    float or pd.Series
        The total pooled variance (T) if a single target_vars is provided.
        A pandas Series of pooled variances if multiple target_vars are processed.

    Raises
    ------
    ValueError
        If inputs are invalid, or m < 2 (Rubin's rules require at least 2 imputations).
    """
    if not imputed_datasets:
        raise ValueError("imputed_datasets list cannot be empty.")
    m = len(imputed_datasets)
    if m < 2:
        raise ValueError("Rubin's Rules require at least m=2 imputed datasets.")

    # Basic validation (more in pool_means if called)
    df_template = imputed_datasets[0]
    if target_vars is None:
        vars_to_pool = df_template.select_dtypes(include=np.number).columns.tolist()
        if not vars_to_pool:
            raise ValueError("No numeric columns found to pool variances.")
    elif isinstance(target_vars, str):
        vars_to_pool = [target_vars]
    elif isinstance(target_vars, list):
        vars_to_pool = target_vars
    else:
        raise TypeError("target_vars must be a string, list of strings, or None.")

    for var_name in vars_to_pool:
        if var_name not in df_template.columns:
            raise ValueError(f"Target variable '{var_name}' not found.")
        if not pd.api.types.is_numeric_dtype(df_template[var_name]):
            raise ValueError(f"Target variable '{var_name}' is not numeric.")

    # Calculate pooled means if not provided
    if pooled_means_input is None:
        _pooled_means = pool_means(imputed_datasets, vars_to_pool)
    elif isinstance(target_vars, str) and isinstance(pooled_means_input, (float, int)):
        _pooled_means = pd.Series([pooled_means_input], index=[target_vars])
    elif isinstance(pooled_means_input, pd.Series):
        _pooled_means = pooled_means_input
        if not set(vars_to_pool).issubset(set(_pooled_means.index)):
            raise ValueError("pooled_means_input does not contain all target_vars.")
    else:
        raise TypeError("pooled_means_input is not of expected type (float or pd.Series).")


    within_variances_list = [] # List of U_i (variances from each dataset)
    means_list = []            # List of Q_hat_i (means from each dataset)

    for df in imputed_datasets:
        within_variances_list.append(df[vars_to_pool].var(ddof=1)) # Sample variance
        means_list.append(df[vars_to_pool].mean())
    
    # Average within-imputation variance (U_bar)
    # within_variances_list contains Series, concatenate them
    within_variances_df = pd.concat(within_variances_list, axis=1)
    U_bar = within_variances_df.mean(axis=1) # Series indexed by var_name

    # Between-imputation variance (B)
    # means_list contains Series, concatenate them
    means_df = pd.concat(means_list, axis=1) # Each column is Q_hat_i for a dataset
    
    # (_pooled_means should be a Series here)
    # B = (1/(m-1)) * sum((Q_hat_i - Q_bar_pooled)^2)
    # This needs to be calculated per variable
    B_values = {}
    for var_name in vars_to_pool:
        q_hat_i_for_var = means_df.loc[var_name] # Series of means for var_name from each dataset
        q_bar_pooled_for_var = _pooled_means.loc[var_name]
        B_values[var_name] = np.sum((q_hat_i_for_var - q_bar_pooled_for_var)**2) / (m - 1)
    
    B = pd.Series(B_values) # Series indexed by var_name

    # Total variance T = U_bar + (1 + 1/m) * B
    T = U_bar + (1 + 1/m) * B

    if isinstance(target_vars, str) and len(vars_to_pool) == 1:
        return T.iloc[0]
    return T

if __name__ == '__main__':
    # Example for uncertainty tooling
    # Create 3 dummy imputed datasets
    data1 = {'A': [1, 2, 3, 4, 5], 'B': [10, 12, 10, 14, 15], 'C': ['x','y','x','y','x']}
    data2 = {'A': [1.5, 2.5, 3, 4, 5.5], 'B': [11, 12, 11, 13, 14], 'C': ['x','y','x','y','x']}
    data3 = {'A': [1, 2.2, 3.3, 4.4, 5], 'B': [10, 11, 12, 13, 16], 'C': ['x','y','x','y','x']}
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    
    imputed_dfs = [df1, df2, df3]

    print("--- Pool Means ---")
    # Pool mean for a single variable 'A'
    pooled_mean_A = pool_means(imputed_dfs, target_vars='A')
    print(f"Pooled mean for A: {pooled_mean_A}") # (3+3.3+3.18)/3 = 3.16

    # Pool means for multiple variables 'A' and 'B'
    pooled_means_AB = pool_means(imputed_dfs, target_vars=['A', 'B'])
    print(f"Pooled means for A and B:\n{pooled_means_AB}")

    # Pool means for all numeric variables
    pooled_means_all = pool_means(imputed_dfs)
    print(f"Pooled means for all numeric vars:\n{pooled_means_all}")
    
    print("\n--- Pool Variances (Rubin's Rules) ---")
    # Pool variance for 'A'
    pooled_var_A = pool_variances_rubin(imputed_dfs, target_vars='A')
    print(f"Pooled variance for A: {pooled_var_A}")

    # Pool variances for 'A' and 'B'
    pooled_vars_AB = pool_variances_rubin(imputed_dfs, target_vars=['A', 'B'])
    print(f"Pooled variances for A and B:\n{pooled_vars_AB}")
    
    # Pool variances for all numeric variables, providing pooled means
    pooled_vars_all = pool_variances_rubin(imputed_dfs, pooled_means_input=pooled_means_all)
    print(f"Pooled variances for all numeric vars (with provided means):\n{pooled_vars_all}")

    # Example:
    # For A:
    # Q_hat_i: df1.A.mean()=3, df2.A.mean()=3.3, df3.A.mean()=3.18
    # Q_bar_pooled_A = (3 + 3.3 + 3.18) / 3 = 9.48 / 3 = 3.16
    # U_i for A: df1.A.var()=2.5, df2.A.var()=2.325, df3.A.var()=2.357
    # U_bar_A = (2.5 + 2.325 + 2.357) / 3 = 7.182 / 3 = 2.394
    # B_A = ( (3-3.16)^2 + (3.3-3.16)^2 + (3.18-3.16)^2 ) / (3-1)
    #     = ( (-0.16)^2 + (0.14)^2 + (0.02)^2 ) / 2
    #     = ( 0.0256 + 0.0196 + 0.0004 ) / 2
    #     = 0.0456 / 2 = 0.0228
    # T_A = U_bar_A + (1 + 1/3)*B_A = 2.394 + (4/3)*0.0228 = 2.394 + 0.0304 = 2.4244
    # Manual check:
    # Means: [3.0, 3.3, 3.18] -> pooled = 3.16
    # Variances: [2.5, 2.325, 2.3569999999999993] -> U_bar = 2.3939999999999997
    # B = np.var([3.0,3.3,3.18], ddof=1) = 0.0228
    # T = 2.3939999999999997 + (1 + 1/3) * 0.0228 = 2.3939999999999997 + 0.0304 = 2.4243999999999997
    # Looks consistent.