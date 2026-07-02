"""
Shopper Spectrum - Step 4: Product Recommendation Engine
-------------------------------------------------------------
Builds an item-based collaborative filtering recommender:
    1. Build a Customer x Product matrix (quantity purchased).
    2. Compute item-item cosine similarity.
    3. Given a product name, return the top-N most similar products.

Artifacts saved:
    models/product_similarity.pkl -> DataFrame of item-item cosine similarity
    models/product_lookup.pkl     -> {StockCode: Description} and reverse map

Run:
    python src/recommendation.py
"""

import os
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

CLEAN_PATH = os.path.join("data", "cleaned_data.csv")
MODEL_DIR = "models"


def build_item_similarity(df: pd.DataFrame):
    # Use StockCode as the canonical product id (Description can vary in
    # spelling for the same code); keep a StockCode -> Description lookup
    # for display purposes, using the most frequent description per code.
    lookup = (
        df.groupby("StockCode")["Description"]
        .agg(lambda x: x.value_counts().idxmax())
        .to_dict()
    )

    pivot = pd.pivot_table(
        df, index="CustomerID", columns="StockCode", values="Quantity", aggfunc="sum", fill_value=0
    )

    print(f"Customer-Product matrix shape: {pivot.shape}")

    similarity = cosine_similarity(pivot.T)
    similarity_df = pd.DataFrame(similarity, index=pivot.columns, columns=pivot.columns)

    return similarity_df, lookup


def recommend_similar_products(product_name: str, similarity_dict: dict, lookup: dict, n=5):
    """Given a product name (case-insensitive, partial match allowed),
    return the top-n most similar products by StockCode/Description."""
    desc_to_code = {v.lower(): k for k, v in lookup.items()}

    match_code = None
    product_name_lower = product_name.strip().lower()

    if product_name_lower in desc_to_code:
        match_code = desc_to_code[product_name_lower]
    else:
        # partial match fallback
        candidates = [desc for desc in desc_to_code if product_name_lower in desc]
        if candidates:
            match_code = desc_to_code[candidates[0]]

    if match_code is None or match_code not in similarity_dict:
        return None, []

    # Retrieve precomputed recommendations
    recs = similarity_dict[match_code][:n]
    results = [(code, lookup.get(code, code), round(score, 3)) for code, score in recs]

    return lookup.get(match_code, match_code), results


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded cleaned data: {df.shape}")

    similarity_df, lookup = build_item_similarity(df)

    # Convert the similarity DataFrame into a dictionary of top 50 recommendations
    # to significantly reduce the size of the pickle artifact.
    print("Converting similarity DataFrame to top-50 lookup dictionary...")
    similarity_dict = {}
    for col in similarity_df.columns:
        sims = similarity_df[col].drop(index=col).sort_values(ascending=False)
        top_recs = sims.head(50)
        similarity_dict[col] = list(zip(top_recs.index, top_recs.values))

    with open(os.path.join(MODEL_DIR, "product_similarity.pkl"), "wb") as f:
        pickle.dump(similarity_dict, f)
    with open(os.path.join(MODEL_DIR, "product_lookup.pkl"), "wb") as f:
        pickle.dump(lookup, f)

    print("Saved model artifacts to:", MODEL_DIR)

    # quick sanity check
    test_product = "WHITE HANGING HEART T-LIGHT HOLDER"
    matched, recs = recommend_similar_products(test_product, similarity_dict, lookup, n=5)
    print(f"\nSample recommendation for '{test_product}':")
    print(f"Matched product: {matched}")
    for code, name, score in recs:
        print(f"  {name} (StockCode {code}) - similarity {score}")


if __name__ == "__main__":
    main()
