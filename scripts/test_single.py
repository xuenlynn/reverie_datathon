# scripts/test_single.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import pairwise_distances

base = Path('models')
model = joblib.load(base / 'final_model.sav')
scaler = None
try:
    scaler = joblib.load(base / 'scaler.sav')
except Exception:
    pass

df = pd.read_csv('data/processed/sample_one_row.csv')
feature_cols = ['Income','Total_Spend','Wine_Ratio','Meat_Ratio',
                'Web_Purchase_Ratio','Spend_per_Purchase','Age',
                'Total_Children','Customer_Tenure','Recency']
X = df[feature_cols].values
X_scaled = scaler.transform(X) if scaler is not None else X

# discrete label
label = model.predict(X_scaled)
print('Predicted cluster label:', label)

# distances to each centroid
dists = pairwise_distances(X_scaled, model.cluster_centers_)
print('Distances to centroids:', dists[0])

# soft-ish scores (approximate probs)
scores = np.exp(-dists)
probs = scores / scores.sum(axis=1, keepdims=True)
print('Approx. cluster probabilities:', probs[0])

# optional: show center values (unscaled if scaler available)
if scaler is not None:
    centers_unscaled = scaler.inverse_transform(model.cluster_centers_)
    print('Cluster centers (unscaled):')
    print(centers_unscaled)
else:
    print('Cluster centers (scaled):')
    print(model.cluster_centers_)