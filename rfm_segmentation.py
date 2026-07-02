"""
Shopper Spectrum - Step 3: Customer Segmentation (RFM + Clustering)
-----------------------------------------------------------------------
Builds Recency, Frequency, Monetary (RFM) features per customer, scales
them, fits a KMeans model, and labels each cluster with a business-friendly
segment name (High Value / Regular / Occasional / At-Risk).

Artifacts saved:
    data/rfm_segments.csv     -> per-customer RFM values + segment
    models/scaler.pkl         -> fitted StandardScaler
    models/kmeans_model.pkl   -> fitted KMeans model
    models/segment_map.pkl    -> {cluster_id: segment_name}
    outputs/figures/elbow_plot.png
    outputs/figures/rfm_clusters.png

Run:
    python src/rfm_segmentation.py
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CLEAN_PATH = os.path.join("data", "cleaned_data.csv")
RFM_PATH = os.path.join("data", "rfm_segments.csv")
FIG_DIR = os.path.join("outputs", "figures")
MODEL_DIR = "models"

N_CLUSTERS = 4
RANDOM_STATE = 42


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary per customer."""
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    return rfm


def find_best_k(X_scaled, k_range=range(2, 9)):
    """Elbow method + silhouette score to help pick k. Saves the elbow plot."""
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), silhouettes, marker="o", color="orange")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Score")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "elbow_plot.png"), dpi=120)
    plt.close(fig)
    print(f"Saved: {os.path.join(FIG_DIR, 'elbow_plot.png')}")

    for k, sil in zip(k_range, silhouettes):
        print(f"  k={k}: silhouette={sil:.3f}")


def label_segments(rfm: pd.DataFrame, cluster_col="Cluster") -> dict:
    """
    Map numeric cluster ids to business-friendly names using each cluster's
    mean RFM profile:
        - Low recency + high frequency + high monetary  -> High-Value
        - Low recency + moderate frequency/monetary       -> Regular
        - High recency + low frequency/monetary           -> At-Risk
        - everything else                                  -> Occasional
    """
    profile = rfm.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean()

    # Rank clusters: lower recency is better, higher frequency/monetary is better
    r_rank = profile["Recency"].rank(ascending=True)      # 1 = most recent (best)
    f_rank = profile["Frequency"].rank(ascending=False)    # 1 = most frequent (best)
    m_rank = profile["Monetary"].rank(ascending=False)     # 1 = highest spend (best)

    score = r_rank + f_rank + m_rank  # lower total = better customer
    ordered_clusters = score.sort_values().index.tolist()

    names = ["High-Value", "Regular", "Occasional", "At-Risk"]
    # if N_CLUSTERS != 4, pad/truncate gracefully
    while len(names) < len(ordered_clusters):
        names.append(f"Segment-{len(names)+1}")
    names = names[: len(ordered_clusters)]

    segment_map = {cluster: name for cluster, name in zip(ordered_clusters, names)}
    print("\nCluster -> Segment mapping (based on RFM profile):")
    print(profile)
    print(segment_map)
    return segment_map


def plot_clusters(rfm: pd.DataFrame):
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        rfm["Recency"], rfm["Monetary"], c=rfm["Cluster"], cmap="viridis", alpha=0.6, s=20
    )
    ax.set_xlabel("Recency (days since last purchase)")
    ax.set_ylabel("Monetary (total spend)")
    ax.set_title("Customer Segments (Recency vs Monetary)")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rfm_clusters.png"), dpi=120)
    plt.close(fig)
    print(f"Saved: {os.path.join(FIG_DIR, 'rfm_clusters.png')}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(CLEAN_PATH, parse_dates=["InvoiceDate"])
    print(f"Loaded cleaned data: {df.shape}")

    rfm = build_rfm(df)
    print(f"\nRFM table: {rfm.shape}")
    print(rfm.describe())

    # Log-transform Monetary/Frequency to reduce skew before scaling (common
    # practice for RFM clustering since these are heavily right-skewed).
    rfm_features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    rfm_log = rfm_features.copy()
    rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])
    rfm_log["Monetary"] = np.log1p(rfm_log["Monetary"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_log)

    print("\nEvaluating k (elbow + silhouette)...")
    find_best_k(X_scaled)

    print(f"\nFitting final KMeans with k={N_CLUSTERS}...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(X_scaled)

    segment_map = label_segments(rfm)
    rfm["Segment"] = rfm["Cluster"].map(segment_map)

    plot_clusters(rfm)

    rfm.to_csv(RFM_PATH, index=False)
    print(f"\nSaved RFM + segments to: {RFM_PATH}")

    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(MODEL_DIR, "segment_map.pkl"), "wb") as f:
        pickle.dump(segment_map, f)

    print("Saved model artifacts to:", MODEL_DIR)
    print("\nSegment counts:")
    print(rfm["Segment"].value_counts())


if __name__ == "__main__":
    main()
