import joblib
from pathlib import Path
from typing import Optional

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


def build_placeholder_sample(model, scaler) -> list:
    expected = None
    if scaler is not None and hasattr(scaler, 'mean_'):
        expected = scaler.mean_.shape[0]
    elif hasattr(model, 'n_features_in_'):
        expected = model.n_features_in_
    if expected is None:
        expected = 10
    return [0.0] * expected


if __name__ == '__main__':
    model = load_joblib(model_path)
    try:
        scaler = load_joblib(scaler_path)
    except Exception:
        scaler = None

    # Build a properly formatted sample (replace zeros with real values)
    sample = build_placeholder_sample(model, scaler)
    # e.g. sample = [66426.0, 1377, 0.757444, 0.070443, 0.15, 68.85, 53, 1, 270, 14]
    X_sample = [sample]

    try:
        expected = None
        if scaler is not None and hasattr(scaler, 'mean_'):
            expected = scaler.mean_.shape[0]
        elif hasattr(model, 'n_features_in_'):
            expected = model.n_features_in_
        if expected is not None and len(sample) != expected:
            raise ValueError(f"Sample has {len(sample)} features but expected {expected}")

        X = scaler.transform(X_sample) if scaler is not None else X_sample
        pred = model.predict(X)
        print('Prediction for sample:', pred)
    except Exception as e:
        print('Could not run sample prediction (check feature types/order):', e)

    # CSV example (commented):
    # import pandas as pd
    # df = pd.read_csv('data/processed/test.csv')
    # features = df.values
    # X_all = scaler.transform(features) if scaler is not None else features
    # df['cluster'] = model.predict(X_all)
    # df.to_csv('data/processed/test_with_clusters.csv', index=False)
