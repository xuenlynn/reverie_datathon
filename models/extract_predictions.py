import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "processed" / "test.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "test_with_clusters.csv"
DEFAULT_SUMMARY_OUTPUT = REPO_ROOT / "data" / "processed" / "cluster_summary.csv"
DEFAULT_MODEL = REPO_ROOT / "models" / "final_model.sav"
DEFAULT_SCALER = REPO_ROOT / "models" / "scaler.sav"

FALLBACK_FEATURE_COLS = [
    "Income",
    "Total_Spend",
    "Wine_Ratio",
    "Meat_Ratio",
    "Web_Purchase_Ratio",
    "Spend_per_Purchase",
    "Age",
    "Total_Children",
    "Customer_Tenure",
    "Recency",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load the saved KMeans model and scaler, predict customer clusters, "
            "and export per-row predictions plus cluster summaries."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path, help="CSV to score.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help="Destination for input rows with cluster predictions.",
    )
    parser.add_argument(
        "--summary-output",
        default=DEFAULT_SUMMARY_OUTPUT,
        type=Path,
        help="Destination for cluster-level summary statistics.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        type=Path,
        help="Path to saved KMeans joblib model.",
    )
    parser.add_argument(
        "--scaler",
        default=DEFAULT_SCALER,
        type=Path,
        help="Path to saved StandardScaler joblib artifact.",
    )
    return parser.parse_args()


def load_artifact(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    artifact = joblib.load(path)
    print(f"Loaded {label}: {path} ({type(artifact).__name__})")
    return artifact


def get_feature_cols(scaler) -> list[str]:
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return FALLBACK_FEATURE_COLS


def validate_inputs(df: pd.DataFrame, feature_cols: list[str], model, scaler) -> None:
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing required feature columns: {missing_cols}")

    missing_values = df[feature_cols].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise ValueError(
            "Input CSV has missing values in feature columns: "
            f"{missing_values.to_dict()}"
        )

    expected_features = getattr(scaler, "n_features_in_", None)
    if expected_features is None:
        expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and len(feature_cols) != expected_features:
        raise ValueError(
            f"Feature column count is {len(feature_cols)}, but model expects "
            f"{expected_features} features."
        )

    if not hasattr(model, "predict"):
        raise TypeError("Loaded model does not support predict().")
    if not hasattr(model, "transform"):
        raise TypeError("Loaded model does not support transform() for distances.")
    if not hasattr(scaler, "transform"):
        raise TypeError("Loaded scaler does not support transform().")


def add_predictions(df: pd.DataFrame, feature_cols: list[str], model, scaler) -> pd.DataFrame:
    X = df[feature_cols]
    X_scaled = scaler.transform(X)

    clusters = model.predict(X_scaled)
    distances = model.transform(X_scaled)

    result = df.copy()
    result["predicted_cluster"] = clusters

    for cluster_id in range(distances.shape[1]):
        result[f"distance_to_cluster_{cluster_id}"] = distances[:, cluster_id]

    result["distance_to_assigned_cluster"] = distances[np.arange(len(clusters)), clusters]
    return result


def build_cluster_summary(result: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    summary = (
        result.groupby("predicted_cluster")
        .agg(
            row_count=("predicted_cluster", "size"),
            avg_distance_to_assigned_cluster=(
                "distance_to_assigned_cluster",
                "mean",
            ),
            **{f"avg_{col}": (col, "mean") for col in feature_cols},
        )
        .reset_index()
        .sort_values("predicted_cluster")
    )
    return summary


def main() -> None:
    args = parse_args()

    model = load_artifact(args.model, "model")
    scaler = load_artifact(args.scaler, "scaler")
    feature_cols = get_feature_cols(scaler)

    print(f"Using feature columns: {feature_cols}")

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    df = pd.read_csv(args.input)

    validate_inputs(df, feature_cols, model, scaler)
    result = add_predictions(df, feature_cols, model, scaler)
    summary = build_cluster_summary(result, feature_cols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)

    cluster_labels = sorted(result["predicted_cluster"].unique().tolist())
    print(f"Scored rows: {len(result)}")
    print(f"Predicted clusters: {cluster_labels}")
    print(f"Wrote predictions: {args.output}")
    print(f"Wrote cluster summary: {args.summary_output}")


if __name__ == "__main__":
    main()
