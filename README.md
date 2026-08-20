# Customer Personality Analysis — Segmentation for Targeted Marketing

We use unsupervised clustering (K-Means) on the Customer Personality Analysis dataset to identify distinct customer personas, each mapped to a marketing recommendation.

## Dataset

[Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) — Kaggle, by imakash3011.



## Problem

**Customer Segmentation for Targeted Marketing.** We considered and rejected
several other framings of this dataset — campaign response prediction, CLV /
retention risk, deal sensitivity, and channel preference — and settled on
segmentation because it directly supports actionable marketing personas rather
than a single predictive score.

## Repo structure

```
reverie_datathon/
├── eda/
│   └── 01_eda.ipynb          # exploration only, no modeling
├── data/
│   ├── raw/                  # untouched original CSV
│   ├── processed/            # train.csv, val.csv, test.csv (generated)
│   └── split_data.py         # cleaning, feature engineering, and splitting pipeline
├── models/                   # final_model.sav (KMeans model + scaler, via joblib)
├── training/
│   └── 01_training.ipynb     # scaling, k selection, final fit, evaluation
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=reverie_datathon
```

Download `marketing_campaign.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
and place it at `data/raw/marketing_campaign.csv`.

## How to reproduce

1. **Generate the processed splits:**
   ```bash
   python data/split_data.py
   ```
   This cleans the raw data, engineers features, and writes leakage-safe
   `train.csv` / `val.csv` / `test.csv` to `data/processed/`.

2. **Run the EDA notebook** (`eda/01_eda.ipynb`) — optional, exploration only.

3. **Run the training notebook** (`training/01_training.ipynb`, kernel:
   `reverie_datathon`) — loads the processed splits, scales features, selects
   `k` via elbow + silhouette, fits the final model, evaluates cluster
   stability across splits, profiles each cluster, visualizes clusters via
   PCA, and saves the model + scaler to `models/final_model.sav`.

## Usage / Running predictions

After the model is trained, you can use it to predict customer cluster assignments:

**Option 1: Test on a single sample row**
```bash
python scripts/test_single.py
```
This loads the pre-trained model and scaler, runs a prediction on the sample
in `data/processed/sample_one_row.csv`, and outputs the cluster label,
distances to centroids, and approximate cluster probabilities.

**Option 2: Run the prediction demo**
```bash
python models/predict_demo.py
```
This demonstrates programmatic usage — loads the model and scaler, builds a
placeholder sample, and runs a prediction. Includes commented examples for
batch prediction on CSV data.

**Option 3: Load the model programmatically**
```bash
python models/loader.py
```
This utility loads the model and scaler from `models/final_model.sav` and
`models/scaler.sav`, validates feature count, and shows how to run predictions
in Python code.

**Option 4: Inspect the model file**
```bash
python models/probe.py
```
Diagnose the model file format (joblib or other).

## Methodology highlights

- **Leakage-safe pipeline:** all data-dependent statistics — `Income` median
  imputation, the income outlier filter, the customer-tenure reference date,
  and the fitted `StandardScaler` — are computed on the train split only and
  applied to val/test, never the reverse.
- **Feature set for clustering (10 features):** `Income`, `Total_Spend`,
  `Wine_Ratio`, `Meat_Ratio`, `Web_Purchase_Ratio`, `Spend_per_Purchase`,
  `Age`, `Total_Children`, `Customer_Tenure`, `Recency`. Raw `Mnt*` spend
  columns and campaign-response columns (`Response`, `AcceptedCmp1`–`5`) are
  excluded from the clustering input and reserved for post-hoc cluster
  profiling instead.
- **Model selection:** `k=5`, chosen by reviewing the elbow plot (visibly
  flattens at 5) and silhouette score (local peak of ~0.154 at k=5, higher
  than k=4 or k=6). `k=2` had the highest silhouette overall but was judged
  too coarse to be actionable for marketing segmentation.
- **Train/val/test holdout:** not strictly required for unsupervised
  learning, but used deliberately here to check that cluster assignments and
  silhouette scores are stable across splits rather than an artifact of the
  training data.

## Known limitations

- Feature engineering logic (spend/purchase ratios, age, tenure, etc.)
  currently lives only in `data/split_data.py`; there is no separate shared
  `preprocess.py` module. This is a single-pipeline project, so duplication
  risk is low, but any future changes to feature logic should be made in
  `split_data.py` only, since `training/01_training.ipynb` reads the
  already-engineered columns from the processed CSVs rather than
  recomputing them.
