import pandas as pd

df = pd.read_csv('./path-to-dataset')

test = df[~df['name'].isna() & ~df['name'].isin([''])]

def convert_to_seconds(x):
    return pd.Timedelta(x).total_seconds()

df['workflow_frequency'] = df['workflow_frequency'].apply(convert_to_seconds)

df.to_csv('./path-to-dataset', index=False)