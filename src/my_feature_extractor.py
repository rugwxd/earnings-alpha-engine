import pandas as pd
from pathlib import Path

#Load the filtered transcripts
df = pd.read_csv("data/processed/filtered_transcripts.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(3))