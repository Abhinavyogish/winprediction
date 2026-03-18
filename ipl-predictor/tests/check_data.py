import pandas as pd

df = pd.read_parquet("data/processed/deliveries.parquet")

print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns)