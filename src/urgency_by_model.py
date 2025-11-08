import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame

def plot_urgency_by_model(df: DataFrame) -> None:
    """
    Create a box plot showing the distribution of urgency scores across different models.
    
    Args:
        df (DataFrame): DataFrame containing 'model_display_name' and 'parsed_urgency_score' columns
    """
    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='model_display_name', y='parsed_urgency_score')

    # Customize the plot
    plt.title('Distribution of Urgency Scores by Model')
    plt.xlabel('Model')
    plt.ylabel('Urgency Score')
    plt.xticks(rotation=45)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show the plot
    plt.show()