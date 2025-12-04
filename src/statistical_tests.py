import pandas as pd
import numpy as np
import itertools
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def kruskal_wallis_test(df, group_col, value_col):
    """
    Perform Kruskal-Wallis H-test (non-parametric ANOVA).
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    group_col : str
        Column name for grouping variable
    value_col : str
        Column name for continuous variable to test
    
    Returns:
    --------
    dict: Dictionary with test results
    """
    # Extract groups
    groups = df[group_col].dropna().unique()
    group_data = []
    group_names = []
    
    for group in groups:
        data = df[df[group_col] == group][value_col].dropna()
        if len(data) > 0:
            group_data.append(data.values)
            group_names.append(group)
    
    # Perform Kruskal-Wallis test
    stat, p = kruskal(*group_data)
    
    results = {
        'test': 'Kruskal-Wallis',
        'group_col': group_col,
        'value_col': value_col,
        'statistic': stat,
        'p_value': p,
        'n_groups': len(group_data),
        'groups': group_names,
        'interpretation': 'Significant difference between groups' if p < 0.05 else 'No significant difference'
    }
    
    # Print summary
    print("=" * 60)
    print("KRUSKAL-WALLIS TEST RESULTS")
    print("=" * 60)
    print(f"Groups: {group_col}")
    print(f"Variable: {value_col}")
    print(f"H-statistic: {stat:.4f}")
    print(f"P-value: {p:.6f}")
    print(f"Number of groups: {len(group_data)}")
    print(f"\nConclusion: {results['interpretation']} (α=0.05)")
    print("=" * 60)
    
    return results


def mannwhitneyu_pairwise(df, group_col, value_col, 
                          correction_method='holm', 
                          alternative='two-sided',
                          verbose=True):
    """
    Perform pairwise Mann-Whitney U tests with multiple comparison correction.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    group_col : str
        Column name for grouping variable
    value_col : str
        Column name for continuous variable to test
    correction_method : str
        Multiple comparison correction method (default: 'holm')
    alternative : str
        Alternative hypothesis: 'two-sided', 'less', or 'greater'
    verbose : bool
        Whether to print results
    
    Returns:
    --------
    DataFrame: Results of pairwise comparisons
    """
    # Get unique groups
    races = df[group_col].dropna().unique()
    
    # Generate all pairwise combinations
    pairs = []
    p_values = []
    u_stats = []
    
    for r1, r2 in itertools.combinations(races, 2):
        group1 = df[df[group_col] == r1][value_col].dropna()
        group2 = df[df[group_col] == r2][value_col].dropna()
        
        # Only perform test if both groups have data
        if len(group1) > 0 and len(group2) > 0:
            stat, p = mannwhitneyu(group1, group2, alternative=alternative)
            pairs.append((r1, r2))
            p_values.append(p)
            u_stats.append(stat)
    
    # Apply multiple comparison correction
    if len(p_values) > 0:
        significance, p_corrected, _, _ = multipletests(p_values, method=correction_method)
    else:
        significance, p_corrected = [], []
    
    # Create results dataframe
    results = pd.DataFrame({
        'Group 1': [p[0] for p in pairs],
        'Group 2': [p[1] for p in pairs],
        'U-statistic': u_stats,
        'Raw p-value': p_values,
        f'{correction_method.capitalize()}-adjusted p-value': p_corrected,
        'α<0.05': significance
    })
    
    # Print summary if verbose
    if verbose:
        print("=" * 60)
        print("MANN-WHITNEY U TEST (PAIRWISE)")
        print("=" * 60)
        print(f"Groups: {group_col}")
        print(f"Variable: {value_col}")
        print(f"Correction method: {correction_method}")
        print(f"Alternative hypothesis: {alternative}")
        print(f"Number of comparisons: {len(pairs)}")
        print(f"Significant comparisons: {sum(significance)}")
        print("=" * 60)
        
        if len(results) > 0:
            print("\nSignificant pairwise comparisons (α<0.05):")
            sig_results = results[results['α<0.05']]
            if len(sig_results) > 0:
                print(sig_results[['Group 1', 'Group 2', 'U-statistic', f'{correction_method.capitalize()}-adjusted p-value']].to_string(index=False))
            else:
                print("None")
    
    return results


def run_full_analysis(df, group_col, value_col, correction_method='holm'):
    """
    Run both Kruskal-Wallis and pairwise Mann-Whitney U tests.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    group_col : str
        Column name for grouping variable
    value_col : str
        Column name for continuous variable to test
    correction_method : str
        Multiple comparison correction method
    
    Returns:
    --------
    tuple: (kw_results, pairwise_results)
    """
    print("FULL NON-PARAMETRIC ANALYSIS")
    print("=" * 60)
    
    # Run Kruskal-Wallis
    kw_results = kruskal_wallis_test(df, group_col, value_col)
    
    print("\n" + "=" * 60 + "\n")
    
    # Run pairwise comparisons only if Kruskal-Wallis is significant
    if kw_results['p_value'] < 0.05:
        print("Kruskal-Wallis test is significant (p < 0.05).")
        print("Proceeding with pairwise comparisons...\n")
        pairwise_results = mannwhitneyu_pairwise(df, group_col, value_col, 
                                                correction_method=correction_method)
    else:
        print("Kruskal-Wallis test is not significant (p ≥ 0.05).")
        print("No need for pairwise comparisons.")
        pairwise_results = pd.DataFrame()
    
    return kw_results, pairwise_results


# Quick test if run as script
if __name__ == "__main__":
    print("This is the statistical_tests module.")
    print("Import it in your notebook to use:")
    print("\nfrom statistical_tests import kruskal_wallis_test, mannwhitneyu_pairwise, run_full_analysis")