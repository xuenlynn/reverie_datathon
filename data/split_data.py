import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean(path):
    df = pd.read_csv(path, sep='\t')
    # apply the cleaning steps you found in EDA
    df = df[df['Year_Birth'] > 1930]
    df['Income'] = df['Income'].fillna(df['Income'].median())
    df = df.dropna(subset=['Income'])  # or however you handle remaining NAs
    return df

def split_and_save(df, target_col=None):
    if target_col:
        # supervised path (e.g. campaign response)
        train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target_col])
        val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp[target_col])
    else:
        # unsupervised path (e.g. segmentation) — no stratify needed, no real "test" concept,
        # but keep a holdout to validate cluster stability
        train, temp = train_test_split(df, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

if __name__ == '__main__':
    df = load_and_clean('data/raw/marketing_campaign.csv')
    split_and_save(df, target_col='Response')  # or target_col=None if clustering