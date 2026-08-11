import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_clean(path):
    df = pd.read_csv(path, sep='\t')

    # --- parse dates ---
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')

    # --- missing values ---
    df['Income'] = df['Income'].fillna(df['Income'].median())

    # --- clean categorical noise ---
    df['Marital_Status'] = df['Marital_Status'].replace({
        'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single'
    })
    df['Education_Grouped'] = df['Education'].replace({
        'Basic': 'Undergraduate', '2n Cycle': 'Undergraduate',
        'Graduation': 'Graduate', 'Master': 'Postgraduate', 'PhD': 'Postgraduate'
    })

    # --- outlier removal (BEFORE feature engineering that depends on these cols) ---
    df['Age'] = 2014 - df['Year_Birth']
    df = df[df['Age'] <= 100]

    Q1, Q3 = df['Income'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df['Income'] >= Q1 - 1.5 * IQR) & (df['Income'] <= Q3 + 1.5 * IQR)]

    # --- feature engineering (now safe — runs on outlier-free data) ---
    reference_date = df['Dt_Customer'].max()
    df['Customer_Tenure'] = (reference_date - df['Dt_Customer']).dt.days
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

    # --- encode (pick ONE education column, not both) ---
    df = pd.get_dummies(df, columns=['Education_Grouped', 'Marital_Status'], drop_first=True, dtype=int)

    return df


def split_and_save(df):
    # unsupervised: no target, no stratify — but keep a holdout for cluster stability checks
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")


if __name__ == '__main__':
    df = load_and_clean('data/raw/marketing_campaign.csv')
    split_and_save(df)