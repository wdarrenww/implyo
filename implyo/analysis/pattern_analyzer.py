import pandas as pd
import numpy as np
from scipy import stats
import warnings
from .missing_handler import missing_value_summary 
import warnings
def preliminary_mcar_test(df: pd.DataFrame, missing_col: str, significance_level: float = 0.05) -> dict:
    """
    Performs a preliminary MCAR (Missing Completely At Random) test for a specific column.
    This test checks if the missingness in `missing_col` is correlated with the
    observed values in other columns.
    For numerical columns: uses t-tests to compare means of groups (missing vs. observed in `missing_col`).
    For categorical columns: uses Chi-squared tests for independence.
    Disclaimer: This is a preliminary test. True MCAR is hard to prove.
    This test only checks for relationships with *observed* data.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    missing_col : str
        The name of the column with missing values to be analyzed.
    significance_level : float, optional
        The significance level for the statistical tests (default is 0.05).
    Returns
    -------
    dict
        A dictionary where keys are other column names and values are dictionaries
        containing the test performed, p-value, and whether the null hypothesis
        (missingness is not related to this column's values) is rejected.
        Example:
        {
            'other_col1': {'test': 't-test', 'p_value': 0.03, 'reject_null': True,
                           'interpretation': 'Missingness in missing_col might be related to other_col1.'},
            'other_col2': {'test': 'chi2-test', 'p_value': 0.50, 'reject_null': False,
                           'interpretation': 'No significant evidence that missingness in missing_col is related to other_col2.'}
        }
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame.")
    if missing_col not in df.columns:
        raise ValueError(f"Column '{missing_col}' not found in DataFrame.")
    if df[missing_col].notnull().all():
        warnings.warn(f"Column '{missing_col}' has no missing values. Test cannot be performed.", UserWarning)
        return {}
    if df[missing_col].isnull().all():
        warnings.warn(f"Column '{missing_col}' has all missing values. Test cannot be meaningfully performed.", UserWarning)
        return {}
    results = {}
    missing_indicator = df[missing_col].isnull()
    for col in df.columns:
        if col == missing_col:
            continue
        if df[col].isnull().all():
            results[col] = {'test': 'skipped', 'reason': 'Column is entirely null.'}
            continue
        temp_df = df[[col]].copy()
        temp_df['missing_indicator'] = missing_indicator
        temp_df.dropna(subset=[col], inplace=True) 
        if temp_df['missing_indicator'].nunique() < 2: 
            results[col] = {'test': 'skipped', 'reason': f'Not enough groups to compare for {col} after handling its own NaNs or uniform missingness.'}
            continue
        group_observed = temp_df[temp_df['missing_indicator'] == False][col]
        group_missing = temp_df[temp_df['missing_indicator'] == True][col]
        if len(group_observed) < 2 or len(group_missing) < 2: 
             results[col] = {'test': 'skipped', 'reason': f'Not enough samples in one of the groups for {col} to perform test.'}
             continue
        test_result = {}
        if pd.api.types.is_numeric_dtype(df[col]):
            stat, p_value = stats.ttest_ind(group_observed, group_missing, nan_policy='omit', equal_var=False)
            test_result['test'] = 't-test (Welch)'
        elif isinstance(df[col].dtype, pd.CategoricalDtype) or df[col].dtype == 'object':
            contingency_table = pd.crosstab(temp_df[col], temp_df['missing_indicator'])
            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                results[col] = {'test': 'skipped', 'reason': f'Contingency table for {col} is too small for Chi-squared test.'}
                continue
            try:
                chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                test_result['test'] = 'chi2-test'
            except ValueError as e: 
                results[col] = {'test': 'chi2-test', 'p_value': np.nan, 'error': str(e),
                                'interpretation': 'Error during Chi-squared test, likely due to sparse data.'}
                continue
        else:
            results[col] = {'test': 'skipped', 'reason': f'Unsupported data type for column {col}: {df[col].dtype}'}
            continue
        test_result['p_value'] = p_value
        test_result['reject_null'] = p_value < significance_level
        if test_result['reject_null']:
            test_result['interpretation'] = (
                f"Significant: Missingness in '{missing_col}' MIGHT BE RELATED to values in '{col}'. "
                "This could suggest MAR (if '{col}' is observed) or deviate from MCAR."
            )
        else:
            test_result['interpretation'] = (
                f"Not significant: No strong evidence that missingness in '{missing_col}' is related to values in '{col}'. "
                "This is consistent with MCAR with respect to '{col}'."
            )
        results[col] = test_result
    return results
def suggest_missingness_pattern(df: pd.DataFrame, significance_level: float = 0.05) -> pd.DataFrame:
    """
    Suggests a missingness pattern (MCAR, MAR, or hints of MNAR) for each column with missing data.
    This function iterates through columns with missing data and uses `preliminary_mcar_test`.
    - If no significant relationships are found with other observed variables, suggests MCAR.
    - If significant relationships are found, suggests MAR.
    - MNAR is difficult to test statistically without domain knowledge or assumptions.
      This function will primarily distinguish between MCAR and MAR based on tests.
      It will note that MNAR is always a possibility.
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    significance_level : float, optional
        The significance level for the internal MCAR tests (default is 0.05).
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'column_name': The name of the column with missing values.
        - 'missing_percentage': Percentage of missing values.
        - 'suggested_pattern': Suggested pattern (e.g., "Likely MCAR", "Likely MAR", "Potentially MNAR").
        - 'evidence': Brief explanation or list of related columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    missing_summary_df = missing_value_summary(df)
    cols_with_missing = missing_summary_df[missing_summary_df['missing_count'] > 0]['column_name']
    results = []
    if not cols_with_missing.any():
        return pd.DataFrame(columns=['column_name', 'missing_percentage', 'suggested_pattern', 'evidence'])
    for col_name in cols_with_missing:
        mcar_results = preliminary_mcar_test(df, col_name, significance_level)
        related_vars = [var for var, res in mcar_results.items() 
                        if 'reject_null' in res and res['reject_null']]
        missing_pct = missing_summary_df[missing_summary_df['column_name'] == col_name]['missing_percentage'].iloc[0]
        suggestion = ""
        evidence_str = ""
        if not mcar_results: 
             suggestion = "Undetermined (e.g., column is all NaN or no comparable columns)"
             evidence_str = "No tests could be performed."
        elif not related_vars:
            suggestion = "Likely MCAR (relative to other observed variables)"
            evidence_str = "No significant relationship found with other observed variables."
        else:
            suggestion = "Likely MAR (missingness depends on other observed variables)"
            evidence_str = f"Missingness appears related to: {', '.join(related_vars)}."
        evidence_str += " MNAR is always a possibility and often requires domain knowledge to assess."
        results.append({
            'column_name': col_name,
            'missing_percentage': missing_pct,
            'suggested_pattern': suggestion,
            'evidence': evidence_str
        })
    return pd.DataFrame(results)
if __name__ == '__main__':
    n_samples = 200
    rng = np.random.RandomState(42)
    data = {
        'age': rng.randint(20, 60, n_samples),
        'income': rng.normal(50000, 15000, n_samples),
        'education': rng.choice(['HighSchool', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'satisfaction': rng.randint(1, 10, n_samples),
        'illness_reported': rng.choice([0,1], n_samples, p=[0.8, 0.2])
    }
    example_df = pd.DataFrame(data)
    idx_mcar = rng.choice(example_df.index, size=int(n_samples * 0.1), replace=False)
    example_df.loc[idx_mcar, 'satisfaction'] = np.nan
    idx_mar_income = example_df[example_df['age'] < 30].sample(frac=0.3, random_state=rng).index
    example_df.loc[idx_mar_income, 'income'] = np.nan
    temp_satisfaction = example_df['satisfaction'].copy() 
    idx_mar_edu = example_df[(temp_satisfaction < 4) & (temp_satisfaction.notna())].sample(frac=0.4, random_state=rng).index
    example_df.loc[idx_mar_edu, 'education'] = np.nan
    idx_mnar_proxy = example_df[example_df['income'] > 70000].sample(frac=0.5, random_state=rng).index
    example_df.loc[idx_mnar_proxy, 'illness_reported'] = np.nan
    print("DataFrame with introduced missing values:")
    print(missing_value_summary(example_df))
    print("\n--- Preliminary MCAR Test for 'income' ---")
    mcar_test_income = preliminary_mcar_test(example_df, 'income')
    for k, v in mcar_test_income.items():
        print(f"  {k}: p-value={v.get('p_value', 'N/A')}, interpretation: {v.get('interpretation', v.get('reason'))}")
    print("\n--- Preliminary MCAR Test for 'satisfaction' ---")
    mcar_test_satisfaction = preliminary_mcar_test(example_df, 'satisfaction')
    for k, v in mcar_test_satisfaction.items():
      print(f"  {k}: p-value={v.get('p_value', 'N/A')}, interpretation: {v.get('interpretation', v.get('reason'))}")
    print("\n--- Suggest Missingness Pattern for all columns ---")
    pattern_suggestions = suggest_missingness_pattern(example_df)
    for _, row in pattern_suggestions.iterrows():
        print(f"Column: {row['column_name']}")
        print(f"  Missing %: {row['missing_percentage']:.2f}%")
        print(f"  Suggested Pattern: {row['suggested_pattern']}")
        print(f"  Evidence: {row['evidence']}\n")