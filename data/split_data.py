import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# parsing and cleaning WHOLE DATASET (train, val & test)
def load_and_clean(path):
    df = pd.read_csv(path, sep='\t')

    df = df.drop_duplicates()

    # --- parse dates ---
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')

    # --- clean categorical noise ---
    df['Marital_Status'] = df['Marital_Status'].replace({
        'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single'
    })
    df['Education_Grouped'] = df['Education'].replace({
        'Basic': 'Undergraduate', '2n Cycle': 'Undergraduate',
        'Graduation': 'Graduate', 'Master': 'Postgraduate', 'PhD': 'Postgraduate'
    })

    # --- outlier removal (hardcoded rule, not a statistic) ---
    df['Age'] = 2014 - df['Year_Birth']
    df = df[df['Age'] <= 100]

    # --- row-wise feature engineering (safe pre-split: no cross-row statistics) ---
    df['Total_Children'] = df['Kidhome'] + df['Teenhome']

    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                'MntSweetProducts', 'MntGoldProds']
    df['Total_Spend'] = df[mnt_cols].sum(axis=1)
    df['Wine_Ratio'] = (df['MntWines'] / df['Total_Spend'].replace(0, np.nan)).fillna(0)
    df['Meat_Ratio'] = (df['MntMeatProducts'] / df['Total_Spend'].replace(0, np.nan)).fillna(0)

    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases']
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
    df['Web_Purchase_Ratio'] = (df['NumWebPurchases'] / df['Total_Purchases'].replace(0, np.nan)).fillna(0)
    df['Spend_per_Purchase'] = (df['Total_Spend'] / df['Total_Purchases'].replace(0, np.nan)).fillna(0)

    # --- drop unhelpful columns ---
    df = df.drop(columns=['Z_CostContact', 'Z_Revenue'])

    # --- encode fixed/known categories (safe pre-split: not data-driven) ---
    df = pd.get_dummies(df, columns=['Education_Grouped', 'Marital_Status'], drop_first=True, dtype=int)

    return df


def split_data(df, test_size=0.3, random_state=42):
    # unsupervised: no target, no stratify — but keep a holdout for cluster stability checks
    # train 0.7 dataset, temp gets split into val and test later on 0.15 each (0.5 of 0.3)
    train, temp = train_test_split(df, test_size=test_size, random_state=random_state)
    val, test = train_test_split(temp, test_size=0.5, random_state=random_state)
    return train, val, test


def fit_and_apply_preprocessing(train, val, test):
    """Learn statistics from train ONLY, apply to all three splits."""
    income_median = train['Income'].median()
    reference_date = train['Dt_Customer'].max()   # <-- moved here, train-only

    splits = {}
    for name, split in [('train', train), ('val', val), ('test', test)]:
        # use splits to apply the below preprocessing steps to all 3 datasets
        split = split.copy()
        split['Income'] = split['Income'].fillna(income_median)
        split['Customer_Tenure'] = (reference_date - split['Dt_Customer']).dt.days
        splits[name] = split

    return splits['train'], splits['val'], splits['test']


def save_splits(train, val, test, out_dir='data/processed'):
    train.to_csv(f'{out_dir}/train.csv', index=False)
    val.to_csv(f'{out_dir}/val.csv', index=False)
    test.to_csv(f'{out_dir}/test.csv', index=False)
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")


if __name__ == '__main__':
    df = load_and_clean('data/raw/marketing_campaign.csv')
    train, val, test = split_data(df)
    train, val, test = fit_and_apply_preprocessing(train, val, test)
    save_splits(train, val, test)