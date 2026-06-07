import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, login

warnings.filterwarnings('ignore')

# ── Authenticate ──────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get('HF_TOKEN')
login(token=HF_TOKEN)

DATASET_PATH = 'hf://datasets/Suhani2128/tourism-dataset/tourism.csv'

# ── Load dataset directly from the Hugging Face data space ───────────────
print('Loading dataset from Hugging Face...')
df = pd.read_csv(DATASET_PATH)
print(f'Raw shape: {df.shape}')
print(df.head(3))

# ── Data Cleaning: remove unnecessary columns ─────────────────────────────
cols_to_drop = ['Unnamed: 0', 'CustomerID']   # ID columns — not predictive
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
print(f'After dropping ID columns: {df.shape}')

# Remove duplicate rows
before = len(df)
df.drop_duplicates(inplace=True)
print(f'Duplicates removed: {before - len(df)}')

# Impute missing values
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = [c for c in df.select_dtypes(include=['float64','int64']).columns
            if c != 'ProdTaken']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
print(f'Missing values remaining: {df.isnull().sum().sum()}')

# ── Label-encode all categorical columns ──────────────────────────────────
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
print(f'Categorical columns encoded: {cat_cols}')

# ── Stratified 80-20 train / test split ───────────────────────────────────
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
train_df = pd.concat([X_train.reset_index(drop=True),
                      y_train.reset_index(drop=True)], axis=1)
test_df  = pd.concat([X_test.reset_index(drop=True),
                      y_test.reset_index(drop=True)],  axis=1)
print(f'Train: {train_df.shape}  |  Test: {test_df.shape}')

# ── Save train and test sets locally ──────────────────────────────────────
os.makedirs('tourism_project/data', exist_ok=True)
TRAIN_PATH = 'tourism_project/data/train.csv'
TEST_PATH  = 'tourism_project/data/test.csv'
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH,  index=False)
print(f'Saved locally: {TRAIN_PATH}  |  {TEST_PATH}')

# ── Upload train and test datasets back to the Hugging Face data space ────
api = HfApi()
repo_id = 'Suhani2128/tourism-dataset'
for fname, fpath in [('train.csv', TRAIN_PATH), ('test.csv', TEST_PATH)]:
    api.upload_file(
        path_or_fileobj = fpath,
        path_in_repo    = fname,
        repo_id         = repo_id,
        repo_type       = 'dataset',
        token           = HF_TOKEN
    )
    print(f'Uploaded {fname} → HF dataset space')
print('Data preparation complete.')
