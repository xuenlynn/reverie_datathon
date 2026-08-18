import joblib
from pathlib import Path

# The .sav files in this folder are joblib-serialized Python objects
# (saved from training with `joblib.dump`). Use joblib.load to read them.
base = Path(__file__).parent
model_path = base / 'final_model.sav'
scaler_path = base / 'scaler.sav'

def load_joblib(path: Path):
	try:
		obj = joblib.load(path)
		print(f"Loaded {path.name}:", type(obj))
		return obj
	except Exception as e:
		print(f"Failed to load {path}: {e}")
		raise

if __name__ == '__main__':
	model = load_joblib(model_path)
	try:
		scaler = load_joblib(scaler_path)
	except Exception:
		scaler = None
	# Example: inspect model attributes (sklearn estimators)
	if hasattr(model, 'predict'):
		print('Model supports predict()')
	else:
		print('Model object loaded; not a scikit-learn estimator necessarily')

	# --- Example: single-sample prediction ---
	# Replace the values below with the real numeric feature values
	# in the same order used during training. The sample must be 2D
	# (one row -> list inside a list) for scikit-learn.
	# Example uses placeholder numeric values; update them accordingly.
	sample = [160803.0,1717,0.032033,0.944671,0.000000,39.022727,32,0,694,21]
	X_sample = [sample]
	try:
		# Determine expected number of features (prefer scaler, fall back to model)
		expected = None
		if scaler is not None and hasattr(scaler, 'mean_'):
			expected = scaler.mean_.shape[0]
		elif hasattr(model, 'n_features_in_'):
			expected = model.n_features_in_
		# Basic validation
		if expected is not None and len(sample) != expected:
			raise ValueError(f"Sample has {len(sample)} features but expected {expected}")
		X = scaler.transform(X_sample) if scaler is not None else X_sample
		pred = model.predict(X)
		print('Prediction for sample:', pred)
	except Exception as e:
		print('Could not run sample prediction (check feature types/order):', e)

	# --- Example: predict on a CSV and save results ---
	# Uncomment and adjust column selection before running.
	# import pandas as pd
	# df = pd.read_csv('data/processed/test.csv')         # adjust path/columns
	# features = df.values                                # or df[['f1','f2',...]]
	# X_all = scaler.transform(features) if scaler is not None else features
	# df['cluster'] = model.predict(X_all)
	# df.to_csv('data/processed/test_with_clusters.csv', index=False)
