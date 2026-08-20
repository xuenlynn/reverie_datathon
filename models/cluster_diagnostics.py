import json
from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

try:
    from scipy.stats import kruskal
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


BASE = Path(__file__).parent
MODEL_PATH = BASE / 'final_model.sav'
SCALER_PATH = BASE / 'scaler.sav'
DATA_DIR = BASE.parent / 'data' / 'processed'
REPORTS = BASE.parent / 'reports'
REPORTS.mkdir(exist_ok=True)

FEATURE_COLS = [
    'Income', 'Total_Spend', 'Wine_Ratio', 'Meat_Ratio',
    'Web_Purchase_Ratio', 'Spend_per_Purchase', 'Age',
    'Total_Children', 'Customer_Tenure', 'Recency'
]

ACCEPT_COLS = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
DEMOGRAPHIC_COLS = ['Education', 'Marital_Status', 'Total_Children', 'Customer_Tenure']


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = None
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        print('No scaler found or failed to load scaler; proceeding without scaler')
    return model, scaler


def load_splits():
    train = pd.read_csv(DATA_DIR / 'train.csv')
    val = pd.read_csv(DATA_DIR / 'val.csv')
    test = pd.read_csv(DATA_DIR / 'test.csv')
    return train, val, test


def transform_features(df, scaler):
    X = df[FEATURE_COLS].values
    if scaler is not None:
        X = scaler.transform(df[FEATURE_COLS])
    return X


def cluster_checks(model, scaler, train, val, test):
    X_train = transform_features(train, scaler)
    X_val = transform_features(val, scaler)
    X_test = transform_features(test, scaler)

    # labels
    train_labels = getattr(model, 'labels_', model.predict(X_train))
    val_labels = model.predict(X_val)
    test_labels = model.predict(X_test)

    k_config = getattr(model, 'n_clusters', None)
    unique_labels = np.unique(train_labels)

    diagnostics = {}
    diagnostics['configured_k'] = int(k_config) if k_config is not None else None
    diagnostics['observed_k'] = int(len(unique_labels))
    diagnostics['train_label_counts'] = dict(zip(map(int, unique_labels), np.bincount(train_labels)))

    if diagnostics['observed_k'] != diagnostics['configured_k']:
        diagnostics['warning'] = 'Observed fewer unique clusters than configured k'

    # per-split silhouette and proportions
    splits = [('train', X_train, train_labels), ('val', X_val, val_labels), ('test', X_test, test_labels)]
    diagnostics['splits'] = {}
    for name, X, labels in splits:
        info = {}
        uniq = np.unique(labels)
        info['n_clusters_observed'] = int(len(uniq))
        info['proportions'] = pd.Series(labels).value_counts(normalize=True).sort_index().to_dict()
        try:
            if len(uniq) > 1:
                info['silhouette'] = float(silhouette_score(X, labels))
            else:
                info['silhouette'] = None
        except Exception as e:
            info['silhouette'] = None
            info['silhouette_error'] = str(e)
        diagnostics['splits'][name] = info

    return diagnostics, (train_labels, val_labels, test_labels)


def profiling_and_tests(model, scaler, train, val, test, labels_tuple):
    train_labels, val_labels, test_labels = labels_tuple
    # combine raw dataframes and assign clusters using model on feature columns
    dfs = []
    for df, lbls in [(train, train_labels), (val, val_labels), (test, test_labels)]:
        df_copy = df.copy()
        df_copy['Cluster'] = lbls
        dfs.append(df_copy)
    all_data = pd.concat(dfs, ignore_index=True)

    profile = {}
    # acceptance rates
    profile['acceptance_rates'] = all_data.groupby('Cluster')[ACCEPT_COLS].mean().round(4).to_dict()

    # demographic breakdowns: categorical distributions and numeric summaries
    import pandas.api.types as ptypes
    demo = {}
    for col in DEMOGRAPHIC_COLS:
        if col in all_data.columns:
            if ptypes.is_numeric_dtype(all_data[col]):
                demo[col] = all_data.groupby('Cluster')[col].agg(['count', 'mean', 'median']).round(3).to_dict()
            else:
                demo[col] = all_data.groupby('Cluster')[col].value_counts(normalize=True).unstack(fill_value=0).to_dict()
    profile['demographics'] = demo

    # statistical tests on numeric features (Kruskal-Wallis non-parametric)
    tests = {}
    numeric_cols = FEATURE_COLS + ['Spend_per_Purchase']
    groups = [g for _, g in all_data.groupby('Cluster')]
    for col in numeric_cols:
        col_groups = [g[col].dropna().values for g in groups]
        # require at least 2 groups with data and each group length >=1
        try:
            if SCIPY_AVAILABLE:
                stat, p = kruskal(*col_groups)
                tests[col] = {'test': 'kruskal', 'pvalue': float(p)}
            else:
                tests[col] = {'test': 'kruskal', 'pvalue': None, 'note': 'scipy not available'}
        except Exception as e:
            tests[col] = {'error': str(e)}
    profile['stat_tests'] = tests

    return profile, all_data


def extract_outliers(all_data, target_cluster=3):
    rows = all_data[all_data['Cluster'] == target_cluster]
    out_path = REPORTS / f'cluster_{target_cluster}_rows.csv'
    if not rows.empty:
        rows.to_csv(out_path, index=False)
    return out_path, len(rows)


def pca_plot(model, scaler, train, val, test, all_data):
    X_train = transform_features(train, scaler)
    X_all = np.vstack([transform_features(df, scaler) for df in [train, val, test]])
    labels_all = all_data['Cluster'].values

    pca = PCA(n_components=2, random_state=42)
    X_train_2 = pca.fit_transform(X_train)
    X_all_2 = pca.transform(X_all)
    centers_2 = pca.transform(model.cluster_centers_)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_all_2[:, 0], y=X_all_2[:, 1], hue=labels_all, palette='tab10', s=15, alpha=0.6, legend='full')
    plt.scatter(centers_2[:, 0], centers_2[:, 1], c='black', s=120, marker='X', label='centroids')
    plt.title('PCA projection of clusters')
    plt.legend(title='Cluster')
    out_file = REPORTS / 'pca_clusters.png'
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    return out_file


def save_diagnostics(diagnostics, profile, out_path=REPORTS / 'cluster_diagnostics.json'):
    def make_serializable(o):
        if isinstance(o, dict):
            return {make_serializable(k): make_serializable(v) for k, v in o.items()}
        if isinstance(o, list):
            return [make_serializable(x) for x in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return o

    out = {'diagnostics': make_serializable(diagnostics), 'profile': make_serializable(profile)}
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    return out_path


def main():
    model, scaler = load_artifacts()
    train, val, test = load_splits()
    diagnostics, labels = cluster_checks(model, scaler, train, val, test)
    profile, all_data = profiling_and_tests(model, scaler, train, val, test, labels)
    out_path, n_rows = extract_outliers(all_data, target_cluster=3)
    profile['cluster_3_count'] = n_rows
    pca_file = pca_plot(model, scaler, train, val, test, all_data)
    diagnostics_path = save_diagnostics(diagnostics, profile)

    print('Diagnostics written to:', diagnostics_path)
    print('PCA plot saved to:', pca_file)
    print('Cluster 3 rows exported to:', out_path)


if __name__ == '__main__':
    main()
