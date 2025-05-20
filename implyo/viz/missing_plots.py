import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from ..analysis.missing_handler import missing_value_summary

def plot_missingness_heatmap(df: pd.DataFrame, **kwargs):
    """
    Generates a heatmap of missing values using the missingno library.
    White lines indicate missing data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    **kwargs : dict
        Additional keyword arguments to pass to `msno.matrix()`.
        E.g., `figsize=(10,5)`, `fontsize=12`, `sparkline=True`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    if df.empty:
        print("Input DataFrame is empty. Cannot generate missingness heatmap.")
        fig, ax = plt.subplots(**kwargs.get('figsize', {}))
        ax.text(0.5, 0.5, "Empty DataFrame", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Missing Data Heatmap (Matrix) - Empty DataFrame', fontsize=16)
        if kwargs.get('_show_plot', True):
            plt.show()
        return ax

    
    print("Displaying missingness heatmap (white lines indicate missing data).")
    ax = msno.matrix(df, **kwargs)
    plt.title('Missing Data Heatmap (Matrix)', fontsize=16)
    if kwargs.get('_show_plot', True): plt.show()
    return ax


def plot_missingness_bar(df: pd.DataFrame, **kwargs):
    """
    Generates a bar plot of missing value counts per column using missingno.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    **kwargs : dict
        Additional keyword arguments to pass to `msno.bar()`.
        E.g., `figsize=(10,5)`, `fontsize=12`, `color='dodgerblue'`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        print("Input DataFrame is empty. Cannot generate missingness bar plot.")
        fig, ax = plt.subplots(**kwargs.get('figsize', {}))
        ax.text(0.5, 0.5, "Empty DataFrame", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Missing Data Bar Plot - Empty DataFrame', fontsize=16)
        if kwargs.get('_show_plot', True): plt.show()
        return ax

    print("Displaying missingness bar plot.")
    ax = msno.bar(df, **kwargs)
    plt.title('Missing Data Bar Plot', fontsize=16)
    if kwargs.get('_show_plot', True): plt.show()
    return ax

def plot_missingness_summary_bar(df: pd.DataFrame, top_n: int = None, **kwargs):
    """
    Generates a bar plot of missing value percentages per column using seaborn.
    Uses the summary from missing_handler.missing_value_summary.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    top_n : int, optional
        Display only the top N columns with the most missing data.
        If None, displays all columns with missing data.
    **kwargs : dict
        Additional keyword arguments to pass to `seaborn.barplot()`.
        E.g., `figsize=(12,6)`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    summary = missing_value_summary(df)
    summary_to_plot = summary[summary['missing_count'] > 0]

    if summary_to_plot.empty:
        print("No missing values to plot in summary bar chart.")
        return None
        
    if top_n:
        summary_to_plot = summary_to_plot.head(top_n)

    fig_size = kwargs.pop('figsize', (12, max(6, len(summary_to_plot) * 0.5)))
    
    plt.figure(figsize=fig_size)
    barplot = sns.barplot(x='missing_percentage', y='column_name', data=summary_to_plot, orient='h', **kwargs)
    
    plt.title(f'Percentage of Missing Values per Column{" (Top " + str(top_n) + ")" if top_n else ""}', fontsize=16)
    plt.xlabel('Percentage Missing (%)', fontsize=12)
    plt.ylabel('Column Name', fontsize=12)
    plt.tight_layout()
    
    for index, value in enumerate(summary_to_plot['missing_percentage']):
        if value > 0:
             barplot.text(value + 0.5, index, f'{value:.2f}%', color='black', ha="left", va="center", fontsize=10)
    
    plt.show()
    return barplot.figure.gca()


if __name__ == '__main__':
    data = {
        'col1': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
        'col2': ['A', np.nan, 'C', 'D', np.nan, 'F', 'G', 'H', 'I', 'J'],
        'col3': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10],
        'col4': [np.nan, np.nan, np.nan, np.nan, np.nan, 1, 2, 3, 4, 5],
        'col5': list(range(10)),
        'col6': [np.nan] * 5 + list(range(5,10))
    }
    example_df = pd.DataFrame(data)

    print("DataFrame with Missing Values:")
    print(example_df)
    
    print("\n--- Plotting Missingness Heatmap (using missingno) ---")
    plot_missingness_heatmap(example_df, figsize=(10,6), fontsize=10)
    
    print("\n--- Plotting Missingness Bar Plot (using missingno) ---")
    plot_missingness_bar(example_df, figsize=(10,6), fontsize=10, color="tomato")

    print("\n--- Plotting Missingness Summary Bar Plot (using seaborn) ---")
    plot_missingness_summary_bar(example_df)
    
    print("\n--- Plotting Missingness Summary Bar Plot (Top 3) ---")
    plot_missingness_summary_bar(example_df, top_n=3, palette="viridis")

    no_missing_df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
    print("\n--- Plotting with no missing data ---")
    plot_missingness_heatmap(no_missing_df)
    plot_missingness_bar(no_missing_df)
    plot_missingness_summary_bar(no_missing_df)