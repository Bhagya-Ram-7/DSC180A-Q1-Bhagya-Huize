import os
import json
import pandas as pd
import re

def load_triage_data(csv_path1="data/data_hmao.csv", csv_path2="data/data_bram.csv"):
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

    def clean_json_string(json_str):
        """
        Clean JSON strings and extract only the JSON part
        """
        if pd.isna(json_str):
            return None
        
        json_str = str(json_str)
        
        # Remove markdown code blocks
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        
        # Extract only the JSON object (everything between first { and last })
        # This handles cases where there's extra text like "..."
        match = re.search(r'(\{.*\})', json_str, re.DOTALL)
        if match:
            json_str = match.group(1)
        
        json_str = json_str.strip()
        
        if not json_str:
            return None
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # If still failing, try to fix common issues
            try:
                # Remove trailing commas or extra text
                json_str = re.sub(r',\s*\}', '}', json_str)  # Remove trailing comma
                json_str = re.sub(r',\s*$', '', json_str)    # Remove trailing comma
                return json.loads(json_str)
            except:
                print(f"Failed to parse: {json_str[:100]}...")
                return None
    

    df1['extracted_clean'] = df1['extracted'].apply(clean_json_string)

    # Normalize the JSON data into separate columns
    extracted_df = pd.json_normalize(df1['extracted_clean'])

    # Rename columns to match your desired names
    extracted_df = extracted_df.rename(columns={
        'urgency_score': 'parsed_urgency_score',
        'triage_level': 'parsed_triage_level'
    })

    # Add the extracted columns back to the original dataframe
    df1 = pd.concat([df1, extracted_df], axis=1)

    # Drop the temporary cleaned column if you don't need it
    df1 = df1.drop(columns=['extracted_clean'])

    # Keep selected columns for both dataframes
    columns_to_use = [
        "model_display_name",
        "age",
        "gender",
        "language",
        "occupation",
        "race_ethnicity",
        "safety_plan",
        "trauma",
        "parsed_triage_level",
        "parsed_urgency_score",
    ]
    df1 = df1[columns_to_use]
    df2 = df2[columns_to_use]

    # Combine the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Convert urgency_score to numeric
    if "parsed_urgency_score" in combined_df.columns:
        combined_df["parsed_urgency_score"] = pd.to_numeric(combined_df["parsed_urgency_score"], errors="coerce")

    return combined_df.dropna()
