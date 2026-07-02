"""
Shopper Spectrum - Step 1: Data Cleaning
------------------------------------------
Loads the raw online retail transaction data and cleans it according to the
project requirements:
    - Remove missing values (CustomerID, Description)
    - Remove duplicate records
    - Remove cancelled orders (InvoiceNo starting with 'C')
    - Remove invalid quantities and prices (<= 0)
    - Strip extra / leading / trailing whitespace from text columns

Run:
    python src/data_cleaning.py
"""

import os
import pandas as pd

RAW_PATH = os.path.join("data", "online_retail.csv")
CLEAN_PATH = os.path.join("data", "cleaned_data.csv")


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV. The UCI Online Retail file uses latin-1 encoding."""
    df = pd.read_csv(path, encoding="ISO-8859-1")
    return df


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Remove extra spaces (leading, trailing, and repeated internal spaces)
    from every text/object column, including StockCode/InvoiceNo which are
    sometimes read as strings when the file contains mixed types."""
    text_cols = df.select_dtypes(include="object").columns.tolist()
    # Also include string-typed columns created by pandas' newer dtypes
    text_cols += [c for c in df.columns if str(df[c].dtype) == "string"]
    text_cols = list(dict.fromkeys(text_cols))

    for col in text_cols:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()  # leading/trailing spaces
            .str.replace(r"\s+", " ", regex=True)  # collapse internal double spaces
        )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Strip extra spaces from text columns first, so downstream filters
    #    (e.g. checking InvoiceNo prefixes) work on clean values.
    df = strip_whitespace(df)

    # Standardise column dtypes
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)

    before = len(df)

    # 2. Remove missing values -> CustomerID is essential for RFM/segmentation,
    #    Description is essential for the recommendation module.
    df = df.dropna(subset=["CustomerID", "Description"])

    # 3. Remove duplicate records
    df = df.drop_duplicates()

    # 4. Remove cancelled orders (InvoiceNo starting with 'C')
    df = df[~df["InvoiceNo"].str.startswith("C")]

    # 5. Remove invalid quantities and prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Feature engineering used across later steps
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    after = len(df)
    print(f"Rows before cleaning: {before:,}")
    print(f"Rows after cleaning:  {after:,}")
    print(f"Rows removed:         {before - after:,}")

    return df.reset_index(drop=True)


def main():
    print("Loading raw data...")
    df_raw = load_data()
    print(f"Raw shape: {df_raw.shape}")

    print("\nCleaning data...")
    df_clean = clean_data(df_raw)
    print(f"Clean shape: {df_clean.shape}")

    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset to: {CLEAN_PATH}")


if __name__ == "__main__":
    main()
