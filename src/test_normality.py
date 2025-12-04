#!/usr/bin/env python3
"""
Normality Testing Module
Contains functions for statistical normality tests.
"""

import numpy as np
import pandas as pd
from scipy.stats import normaltest, shapiro, anderson
import warnings


def test_normality(data, column_name=None, method='dagostino', alpha=0.05, verbose=True):
    """
    Test if a distribution is normal using statistical tests.
    
    Parameters:
    -----------
    data : array-like, Series, or DataFrame
        The data to test for normality
    column_name : str, optional
        If data is a DataFrame, specify which column to test
    method : str, default='dagostino'
        Test method: 'dagostino', 'shapiro', or 'anderson'
    alpha : float, default=0.05
        Significance level for the test
    verbose : bool, default=True
        Whether to print results
    
    Returns:
    --------
    dict : Dictionary containing test results
    """
    
    # Extract data based on input type
    if isinstance(data, pd.DataFrame):
        if column_name is None:
            raise ValueError("column_name must be specified when data is a DataFrame")
        data_array = data[column_name].dropna().values
    elif isinstance(data, pd.Series):
        data_array = data.dropna().values
    else:
        data_array = np.array(data)
        data_array = data_array[~np.isnan(data_array)]
    
    # Check if we have enough data
    n_samples = len(data_array)
    if n_samples < 8:
        warnings.warn(f"Sample size ({n_samples}) is very small for normality tests")
    
    results = {
        'sample_size': n_samples,
        'method': method,
        'alpha': alpha,
        'is_normal': None,
        'test_statistic': None,
        'p_value': None
    }
    
    # Perform the selected test
    if method.lower() == 'dagostino':
        try:
            stat, p = normaltest(data_array)
            results['test_statistic'] = stat
            results['p_value'] = p
            results['is_normal'] = p > alpha
        except Exception as e:
            print(f"Error in D'Agostino test: {e}")
            return results
            
    elif method.lower() == 'shapiro':
        try:
            stat, p = shapiro(data_array)
            results['test_statistic'] = stat
            results['p_value'] = p
            results['is_normal'] = p > alpha
        except Exception as e:
            print(f"Error in Shapiro-Wilk test: {e}")
            return results
            
    elif method.lower() == 'anderson':
        try:
            result = anderson(data_array)
            results['test_statistic'] = result.statistic
            # Anderson-Darling returns critical values, not p-value
            # We'll compare the statistic to critical value at alpha
            critical_value = result.critical_values[2]  # index 2 corresponds to 5%
            results['is_normal'] = result.statistic < critical_value
            # For consistency, we'll store the critical comparison
            results['critical_value'] = critical_value
        except Exception as e:
            print(f"Error in Anderson-Darling test: {e}")
            return results
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'dagostino', 'shapiro', or 'anderson'")
    
    # Print results if verbose
    if verbose:
        _print_results(results, column_name or 'data', method)
    
    return results


def _print_results(results, column_name, method):
    """Helper function to print formatted results"""
    print("=" * 50)
    print(f"NORMALITY TEST RESULTS")
    print("=" * 50)
    print(f"Column: {column_name}")
    print(f"Sample size: {results['sample_size']}")
    print(f"Test method: {method.upper()}")
    print(f"Significance level (alpha): {results['alpha']}")
    
    if method.lower() == 'anderson':
        print(f"\nAnderson-Darling Test Statistic: {results['test_statistic']:.4f}")
        print(f"Critical Value at 5%: {results.get('critical_value', 'N/A'):.4f}")
    else:
        print(f"\nTest Statistic: {results['test_statistic']:.4f}")
        print(f"P-value: {results['p_value']:.4f}")
    
    print("\nCONCLUSION:")
    if results['is_normal']:
        print("→ Cannot reject normality hypothesis.")
        print(f"  The data appears to follow a normal distribution (p > {results['alpha']}).")
    else:
        print("→ Distribution is NOT normal.")
        if method.lower() != 'anderson':
            print(f"  Significant deviation from normality (p < {results['alpha']}).")
        else:
            print(f"  Test statistic exceeds critical value at {results['alpha']*100}% level.")
    print("=" * 50)


def batch_normality_tests(df, columns, method='dagostino', alpha=0.05):
    """
    Test multiple columns for normality.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    columns : list
        List of column names to test
    method : str, default='dagostino'
        Test method to use
    alpha : float, default=0.05
        Significance level
    
    Returns:
    --------
    DataFrame : Results summary
    """
    results_list = []
    
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe")
            continue
            
        result = test_normality(df, col, method=method, alpha=alpha, verbose=False)
        result['column'] = col
        results_list.append(result)
    
    # Create summary dataframe
    summary_df = pd.DataFrame(results_list)
    
    # Reorder columns for readability
    col_order = ['column', 'sample_size', 'method', 'alpha', 'test_statistic']
    if 'p_value' in summary_df.columns:
        col_order.append('p_value')
    if 'critical_value' in summary_df.columns:
        col_order.append('critical_value')
    col_order.append('is_normal')
    
    summary_df = summary_df[col_order]
    
    # Print summary
    print("=" * 60)
    print("BATCH NORMALITY TESTS SUMMARY")
    print("=" * 60)
    print(f"Method: {method.upper()}")
    print(f"Significance level: {alpha}")
    print(f"Number of columns tested: {len(summary_df)}")
    print(f"Normal distributions: {summary_df['is_normal'].sum()}")
    print(f"Non-normal distributions: {(~summary_df['is_normal']).sum()}")
    print("=" * 60)
    
    return summary_df


# Example usage when run as script
if __name__ == "__main__":
    # Demo with sample data
    print("Demonstrating normality tests with sample data...")
    
    # Create some sample data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 100)
    non_normal_data = np.random.exponential(1, 100)
    
    # Create a sample dataframe
    sample_df = pd.DataFrame({
        'normal_col': normal_data,
        'non_normal_col': non_normal_data,
        'mixed_col': np.concatenate([normal_data[:50], non_normal_data[:50]])
    })
    
    # Test single column
    print("\n1. Testing single column:")
    test_normality(sample_df, 'normal_col', method='dagostino')
    
    # Test batch of columns
    print("\n2. Testing multiple columns:")
    batch_results = batch_normality_tests(sample_df, 
                                         ['normal_col', 'non_normal_col', 'mixed_col'],
                                         method='dagostino')
    print("\nDetailed results:")
    print(batch_results)