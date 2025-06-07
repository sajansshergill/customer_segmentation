import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df['job_description'] = df['job_description'].fillna('')
    df['sector'] = df['sector'].fillna('Unknown')
    df['job_type'] = df['job_type'].fillna('Unknown')
    df['description_length'] = df['job_description'].apply(lambda x: len(str(x).split()))

    # Salary cleaning
    def extract_salary_range(val):
        if pd.isnull(val): return np.nan
        val = str(val).replace("$", "").replace(",", "")
        parts = val.split("-")
        try:
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2
            return float(parts[0])
        except: return np.nan

    df['median_salary'] = df['salary'].apply(extract_salary_range)
    df['state'] = df['location'].str.extract(r',\s*(\w{2})')[0]
    state_counts = df['state'].value_counts().to_dict()
    df['state_popularity'] = df['state'].map(state_counts).fillna(0)

    return df