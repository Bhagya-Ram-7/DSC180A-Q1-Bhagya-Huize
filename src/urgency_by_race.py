import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame

def plot_urgency_by_race(df: DataFrame) -> None:
    """
    Plot overlapping histograms of urgency scores by race/ethnicity
    with mean lines for each group.
    
    Args:
        df (DataFrame): must contain 'race_ethnicity' and 'parsed_urgency_score'
    """

    # Keep only valid rows
    tmp = df[['race_ethnicity', 'parsed_urgency_score']].dropna()

    if tmp.empty:
        print("No valid data to plot.")
        return

    races = sorted(tmp['race_ethnicity'].unique())
    if len(races) == 0:
        print("No race_ethnicity values found.")
        return

    # Make sure we have enough colors
    colors = sns.color_palette("Set2", n_colors=len(races))
    color_dict = dict(zip(races, colors))

    plt.figure(figsize=(12, 6))

    # Plot histograms + mean lines
    for race in races:
        race_data = tmp[tmp['race_ethnicity'] == race]['parsed_urgency_score']
        if race_data.empty:
            continue

        # Histogram
        plt.hist(
            race_data,
            bins=20,
            alpha=0.4,
            density=True,
            color=color_dict[race]
        )

        # Mean line
        mean_val = race_data.mean()
        plt.axvline(
            x=mean_val,
            linestyle='--',
            linewidth=1.5,
            alpha=0.9,
            color=color_dict[race]
        )

    plt.title("Distribution of Urgency Scores by Race/Ethnicity")
    plt.xlabel("Urgency Score")
    plt.ylabel("Density")

    # Custom legend: one color box + mean line per race
    legend_handles = []
    legend_labels = []

    for race in races:
        c = color_dict[race]
        # histogram patch
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4))
        legend_labels.append(race)
        # mean line
        legend_handles.append(plt.Line2D([0], [0], color=c, linestyle='--', linewidth=1.5))
        legend_labels.append(f"{race} mean")

    plt.legend(
        legend_handles,
        legend_labels,
        title="Race/Ethnicity",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

def plot_urgency_hist_2(df: DataFrame) -> None:
    for race in df['race_ethnicity'].unique():
        subset = df[df['race_ethnicity'] == race]
        sns.histplot(subset['parsed_urgency_score'], label=race, bins=18, stat='density')

    plt.title('Distribution of Urgency Score by race')
    plt.xlabel('Urgency Score')
    plt.ylabel('Density')
    plt.legend(title='Race', loc='upper left')
    plt.show()