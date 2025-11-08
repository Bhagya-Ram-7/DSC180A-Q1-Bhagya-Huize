import os
import json
import pandas as pd


def load_triage_data(csv_path1="data/triage_data.csv", csv_path2="data/data.csv"):
    """
    Load and preprocess triage data from two CSV files:
    - Reads both CSV files
    - Keeps selected columns from both dataframes
    - Combines the dataframes
    - Converts urgency score to numeric
    """

    # Resolve full paths
    csv_full_path1 = os.path.join(os.getcwd(), csv_path1)
    csv_full_path2 = os.path.join(os.getcwd(), csv_path2)

    # Read both datasets
    df1 = pd.read_csv(csv_full_path1)
    df2 = pd.read_csv(csv_full_path2)

    # Keep selected columns for both dataframes
    columns_to_use = ["model_display_name", "age", "gender", "race_ethnicity", "occupation", "parsed_urgency_score", "parsed_triage_level"]
    df1 = df1[columns_to_use]
    df2 = df2[columns_to_use]

    # Combine the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Convert urgency_score to numeric
    if "parsed_urgency_score" in combined_df.columns:
        combined_df["parsed_urgency_score"] = pd.to_numeric(combined_df["parsed_urgency_score"], errors="coerce")

    return combined_df
