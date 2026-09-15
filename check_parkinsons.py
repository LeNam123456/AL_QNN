import numpy as np
from sklearn.datasets import fetch_openml

try:
    print("Fetching 'parkinsons' from OpenML...")
    data = fetch_openml(name='parkinsons', version=1, as_frame=True, parser='auto')
    df = data.frame
    print("Shape:", df.shape)
    print("Target column:", data.target_names)
    target = data.target
    print("Target distribution:\n", target.value_counts())
    print("Columns:", df.columns.tolist()[:5], "...")
except Exception as e:
    print("Error:", e)
