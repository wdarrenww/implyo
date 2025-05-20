import pandas as pd
import numpy as np

def identify_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        A boolean DataFrame of the same shape as `df`, where True indicates
        a missing value (NaN, None, NaT).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    return df.isnull()

def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provides a summary of missing values for each column in the DataFrame.

    The summary includes the count of missing values and the percentage
    of missing values for each column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to analyze.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'column_name': The name of the column.
        - 'missing_count': The number of missing values in that column.
        - 'missing_percentage': The percentage of missing values in that column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        return pd.DataFrame(columns=['column_name', 'missing_count', 'missing_percentage'])

    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    summary_df = pd.DataFrame({
        'column_name': df.columns,
        'missing_count': missing_counts,
        'missing_percentage': missing_percentages
    })
    
    summary_df = summary_df.sort_values(by='missing_percentage', ascending=False)
    summary_df.reset_index(drop=True, inplace=True)
    
    return summary_df

def get_rows_with_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns all rows from the DataFrame that contain at least one missing value.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.DataFrame
        A subset of the input DataFrame containing only rows with missing values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    return df[df.isnull().any(axis=1)]

def get_columns_with_missing_values(df: pd.DataFrame) -> pd.Index:
    """
    Returns the names of columns that contain at least one missing value.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.

    Returns
    -------
    pd.Index
        An index object containing the names of columns with missing values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    return df.columns[df.isnull().any()]

if __name__ == '__main__':
    data = {
        'col1': [1, 2, np.nan, 4, 5],
        'col2': ['A', np.nan, 'C', 'D', np.nan],
        'col3': [1.1, 2.2, 3.3, 4.4, 5.5],
        'col4': [np.nan, np.nan, np.nan, np.nan, np.nan]
    }
    example_df = pd.DataFrame(data)

    print("DataFrame with Missing Values:")
    print(example_df)
    print("\n--- Identifying Missing Values ---")
    print(identify_missing_values(example_df))

    print("\n--- Missing Value Summary ---")
    summary = missing_value_summary(example_df)
    print(summary)

    print("\n--- Rows with Missing Values ---")
    print(get_rows_with_missing_values(example_df))

    print("\n--- Columns with Missing Values ---")
    print(get_columns_with_missing_values(example_df))