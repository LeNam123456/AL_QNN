import pandas as pd

df6 = pd.read_csv("outputs/stage1a/stage1a_bcw_6q_runs.csv")
print("=== 6Q COLUMNS ===")
print(df6.columns.tolist())
print(df6.head(3).to_string())
