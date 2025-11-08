import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy import stats

def plot_urgency_by_age(df: DataFrame) -> None:
    """
    Create a scatter plot of urgency scores vs age with a regression line.
    
    Args:
        df (DataFrame): DataFrame containing 'age' and 'parsed_urgency_score' columns
    """
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with regression line
    sns.regplot(data=df, x='age', y='parsed_urgency_score',
                scatter_kws={'alpha':0.5},
                line_kws={'color': 'red'})
    
    # Calculate correlation coefficient and p-value
    # correlation = stats.pearsonr(df['age'], df['parsed_urgency_score'])
    # r_value = correlation[0]
    # p_value = correlation[1]
    
    # Add title with correlation information
    # plt.title(f'Urgency Score vs Age\nCorrelation: {r_value:.3f} (p-value: {p_value:.3f})')
    plt.xlabel('Age')
    plt.ylabel('Urgency Score')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_urgency_distribution_by_age(df: DataFrame) -> None:
    """
    Create a KDE plot showing the distribution of urgency scores for different age groups.
    
    Args:
        df (DataFrame): DataFrame containing 'age' and 'parsed_urgency_score' columns
    """
    for age in df['age'].unique():
        subset = df[df['age'] == age]
        sns.kdeplot(subset['parsed_urgency_score'], label=age, fill=True)
    plt.title('Distribution of Urgency Score by Age')
    plt.xlabel('Urgency Score')
    plt.ylabel('Density')
    plt.legend(title='Age')
    plt.show()