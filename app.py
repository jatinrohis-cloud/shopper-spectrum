"""
Shopper Spectrum - Streamlit Application
------------------------------------------
Two features:
    1. Product Recommendation: enter a product name, get 5 similar products.
    2. Customer Segmentation: enter Recency, Frequency, Monetary values,
       get the predicted customer segment.

Run:
    streamlit run app.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# Bootstrap logic to run Streamlit app programmatically if executed with python directly.
if __name__ == "__main__":
    from streamlit.runtime import get_instance
    try:
        get_instance()
    except RuntimeError:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛍️", layout="wide")

# Custom CSS for Premium Design Aesthetics
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Global font override */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Card design with glassmorphism */
    .metric-card, .prod-card, .segment-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 15px;
    }
    
    .prod-card:hover, .segment-card:hover {
        transform: translateY(-5px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 12px 40px 0 rgba(99, 102, 241, 0.15);
    }
    
    /* Beautiful gradients for headers */
    .gradient-text {
        background: linear-gradient(to right, #818cf8, #c084fc, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.02);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.15) !important;
        color: #e0e7ff !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Custom Buttons styling */
    div.stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px 0 rgba(168, 85, 247, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_recommendation_artifacts():
    with open(MODELS_DIR / "product_similarity.pkl", "rb") as f:
        similarity_dict = pickle.load(f)
    with open(MODELS_DIR / "product_lookup.pkl", "rb") as f:
        lookup = pickle.load(f)
    return similarity_dict, lookup


@st.cache_resource
def load_segmentation_artifacts():
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(MODELS_DIR / "segment_map.pkl", "rb") as f:
        segment_map = pickle.load(f)
    return scaler, kmeans, segment_map


@st.cache_data
def load_cleaned_data():
    df = pd.read_csv(DATA_DIR / "cleaned_data.csv")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@st.cache_data
def load_rfm_segments():
    df = pd.read_csv(DATA_DIR / "rfm_segments.csv")
    return df


def recommend_similar_products(product_name, similarity_dict, lookup, n=5):
    desc_to_code = {v.lower(): k for k, v in lookup.items()}
    product_name_lower = product_name.strip().lower()

    match_code = desc_to_code.get(product_name_lower)
    if match_code is None:
        candidates = [desc for desc in desc_to_code if product_name_lower in desc]
        if candidates:
            match_code = desc_to_code[candidates[0]]

    if match_code is None or match_code not in similarity_dict:
        return None, []

    recs = similarity_dict[match_code][:n]
    results = [(lookup.get(code, code), round(score, 3)) for code, score in recs]
    return lookup.get(match_code, match_code), results


def predict_segment(recency, frequency, monetary, scaler, kmeans, segment_map):
    features = pd.DataFrame(
        {"Recency": [recency], "Frequency": [np.log1p(frequency)], "Monetary": [np.log1p(monetary)]}
    )
    scaled = scaler.transform(features)
    cluster = kmeans.predict(scaled)[0]
    return segment_map.get(cluster, f"Cluster {cluster}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown('<h1 class="gradient-text" style="font-size: 3em; margin-bottom: 0;">🛍️ Shopper Spectrum</h1>', unsafe_allow_html=True)
st.caption("Customer Segmentation and Product Recommendation in E-Commerce")

tab0, tab1, tab2, tab3 = st.tabs([
    "📋 Project Overview", 
    "📊 E-Commerce Insights", 
    "👥 Customer Segmentation", 
    "🎯 Product Recommendation"
])

with tab0:
    st.subheader("📋 Project Overview & Business Case")
    
    col_overview1, col_overview2 = st.columns([3, 2])
    with col_overview1:
        st.markdown("### 📣 Problem Statement")
        st.write(
            "The global e-commerce industry generates vast amounts of transaction data daily, "
            "offering valuable insights into customer purchasing behaviors. Analyzing this data is "
            "essential for identifying meaningful customer segments and recommending relevant products "
            "to enhance customer experience and drive business growth."
        )
        st.write(
            "This project aims to examine transaction data from an online retail business to uncover "
            "patterns in customer purchase behavior, segment customers based on Recency, Frequency, "
            "and Monetary (RFM) analysis, and develop a product recommendation system using collaborative "
            "filtering techniques."
        )

        st.markdown("### 🎯 Business Objectives")
        st.markdown(
            "- **Understand customer buying behavior** and help the company make better business decisions.\n"
            "- **Find and reward valuable customers** (High-Value segment).\n"
            "- **Personalized product recommendations** to improve customer shopping experience.\n"
            "- **Reduce customer churn** by identifying and re-engaging at-risk shoppers."
        )

    with col_overview2:
        st.markdown("### 📌 Real-time Business Use Cases")
        st.info(
            "1. **Targeted Marketing**: Deliver tailormade campaigns to specific customer segments.\n"
            "2. **Personalized Upselling**: Suggest products co-occurring frequently with past purchases.\n"
            "3. **Customer Retention**: Identify at-risk clients and target them with win-back campaigns.\n"
            "4. **Stock & Inventory Optimization**: Track product popularity to maintain ideal stock levels."
        )
        
    st.markdown("---")
    
    col_flow1, col_flow2 = st.columns([1, 1])
    with col_flow1:
        st.markdown("### 📊 Dataset Details")
        st.write("The dataset records actual transactions of a UK-based e-commerce retail store.")
        
        # Display dataset columns table
        cols_dict = {
            "Column Name": ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
            "Meaning & Description": [
                "Transaction bill number (prefixed with 'C' if cancelled)",
                "Unique product/item code identifier",
                "Product name / item description",
                "Number of units purchased per transaction",
                "Date and time when the transaction was generated",
                "Price of a single unit of the product",
                "Unique customer identifier reference",
                "Country where the customer is based"
            ]
        }
        st.table(pd.DataFrame(cols_dict))

    with col_flow2:
        st.markdown("### ⚙️ Project Flow")
        flow_steps = [
            "1. **Load & Explore**: Read transaction records and examine structures.",
            "2. **Data Cleaning**: Remove duplicates, missing CustomerID, cancelled bills, and negative quantities/prices.",
            "3. **Exploratory Data Analysis**: Visualize sales trends, top countries, top products, and purchase patterns.",
            "4. **RFM Feature Engineering**: Calculate Recency, Log-transformed Frequency, and Log-transformed Monetary.",
            "5. **Unsupervised Clustering**: Standardize scores and run KMeans to cluster into 4 business segments.",
            "6. **Collaborative Recommendation**: Compute item-cosine similarity matrix to recommend top-5 products."
        ]
        for step in flow_steps:
            st.markdown(step)

with tab1:
    st.subheader("📊 E-Commerce Insights & Exploratory Data Analysis")
    st.write("Explore business metrics, sales trends, and customer patterns interactively.")

    df_clean = load_cleaned_data()
    
    # Filter Section
    with st.expander("🔍 Filter Dashboard Data", expanded=True):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            countries = sorted(df_clean["Country"].unique())
            selected_countries = st.multiselect(
                "Select Countries (Leave empty for All)",
                options=countries,
                default=[]
            )
        with col_f2:
            min_date = df_clean["InvoiceDate"].min().date()
            max_date = df_clean["InvoiceDate"].max().date()
            start_date, end_date = st.slider(
                "Select Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )

    # Filter Data
    filtered_df = df_clean[
        (df_clean["InvoiceDate"].dt.date >= start_date) & 
        (df_clean["InvoiceDate"].dt.date <= end_date)
    ]
    if selected_countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]

    # Metrics Calculations
    if len(filtered_df) > 0:
        total_revenue = filtered_df["TotalPrice"].sum()
        total_orders = filtered_df["InvoiceNo"].nunique()
        active_customers = filtered_df["CustomerID"].nunique()
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    else:
        total_revenue = 0
        total_orders = 0
        active_customers = 0
        avg_order_value = 0

    # Display KPI Metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Revenue", f"£{total_revenue:,.2f}")
    with col_m2:
        st.metric("Total Orders", f"{total_orders:,}")
    with col_m3:
        st.metric("Active Customers", f"{active_customers:,}")
    with col_m4:
        st.metric("Avg. Order Value", f"£{avg_order_value:,.2f}")

    st.markdown("---")

    # Visualizations Grid
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # 1. Monthly Revenue Trend
        st.markdown("### 📈 Monthly Revenue Trend")
        if not filtered_df.empty:
            monthly_trend = filtered_df.copy()
            monthly_trend["Month"] = monthly_trend["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
            monthly_data = monthly_trend.groupby("Month")["TotalPrice"].sum().reset_index()
            monthly_data = monthly_data.rename(columns={"TotalPrice": "Revenue"})
            st.area_chart(monthly_data.set_index("Month"), y="Revenue", color="#FF4B4B")
        else:
            st.info("No data available.")

        # 2. Top-Selling Products
        st.markdown("### 🛍️ Top 10 Best-Selling Products")
        if not filtered_df.empty:
            top_products = filtered_df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10).reset_index()
            st.bar_chart(top_products.set_index("Description"), y="Quantity", color="#262730")
        else:
            st.info("No data available.")

    with col_chart2:
        # 3. Country-wise Revenue
        st.markdown("### 🌍 Top 10 Countries by Revenue")
        if not filtered_df.empty:
            top_countries = filtered_df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10).reset_index()
            top_countries = top_countries.rename(columns={"TotalPrice": "Revenue"})
            st.bar_chart(top_countries.set_index("Country"), y="Revenue", color="#00C49F")
        else:
            st.info("No data available.")

        # 4. Top Customers by Spend
        st.markdown("### 👥 Top 10 Customers by Spend")
        if not filtered_df.empty:
            top_cust = filtered_df.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False).head(10).reset_index()
            top_cust = top_cust.rename(columns={"TotalPrice": "Total Spend"})
            top_cust["CustomerID"] = "Customer " + top_cust["CustomerID"].astype(str)
            st.bar_chart(top_cust.set_index("CustomerID"), y="Total Spend", color="#FFBB28")
        else:
            st.info("No data available.")

    st.markdown("---")

    # Customer Behavior
    st.markdown("### 👥 Customer Purchase Behavior")
    if not filtered_df.empty:
        orders_per_cust = filtered_df.groupby("CustomerID")["InvoiceNo"].nunique().reset_index()
        
        col_beh1, col_beh2 = st.columns([2, 1])
        with col_beh1:
            st.markdown("**Distribution of Orders per Customer**")
            # Capped distribution for neat plotting
            orders_capped = orders_per_cust["InvoiceNo"].clip(upper=20)
            counts = orders_capped.value_counts().sort_index().reset_index()
            counts.columns = ["Number of Orders", "Number of Customers"]
            counts["Number of Orders"] = counts["Number of Orders"].astype(str)
            counts.loc[counts["Number of Orders"] == "20", "Number of Orders"] = "20+"
            st.bar_chart(counts.set_index("Number of Orders"), y="Number of Customers", color="#8884D8")
            
        with col_beh2:
            st.markdown("**Order Statistics**")
            stats = orders_per_cust["InvoiceNo"].describe()
            stats_df = pd.DataFrame({
                "Metric": ["Average Orders/Customer", "Median Orders/Customer", "Min Orders", "Max Orders", "Standard Deviation"],
                "Value": [f"{stats['mean']:.2f}", f"{stats['50%']:.0f}", f"{stats['min']:.0f}", f"{stats['max']:.0f}", f"{stats['std']:.2f}"]
            })
            st.dataframe(stats_df.set_index("Metric"), width="stretch")
    else:
        st.info("No data available.")

with tab2:
    st.subheader("👥 Customer Segmentation & RFM Analysis")
    st.write("Group customers based on Recency, Frequency, and Monetary value using KMeans Clustering.")
    
    sub1, sub2 = st.tabs(["🔮 Predict Customer Segment", "📊 RFM Clustering & Elbow Method"])
    
    with sub1:
        st.markdown("### Predict a Customer's Segment")
        st.write("Enter a customer's RFM values to see which segment they fall into.")

        scaler, kmeans, segment_map = load_segmentation_artifacts()

        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, key="rec_input")
        with col2:
            frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5, key="freq_input")
        with col3:
            monetary = st.number_input("Monetary (total amount spent)", min_value=0.0, value=500.0, step=10.0, key="mon_input")

        if st.button("Predict Segment", type="primary", key="pred_btn"):
            segment = predict_segment(recency, frequency, monetary, scaler, kmeans, segment_map)

            segment_colors = {
                "High-Value": "🟢",
                "Regular": "🔵",
                "Occasional": "🟡",
                "At-Risk": "🔴",
            }
            icon = segment_colors.get(segment, "⚪")

            segment_descriptions = {
                "High-Value": "Frequent, high-spending, recently active customers. Reward and retain them.",
                "Regular": "Steady customers with moderate frequency and spend. Nurture with upsells.",
                "Occasional": "Infrequent buyers with low spend. Engage with targeted promotions.",
                "At-Risk": "Haven't purchased in a long time. Win them back with re-engagement campaigns.",
            }

            segment_themes = {
                "High-Value": {"color": "#10b981", "bg": "rgba(16, 185, 129, 0.1)", "border": "#10b981"},
                "Regular": {"color": "#3b82f6", "bg": "rgba(59, 130, 246, 0.1)", "border": "#3b82f6"},
                "Occasional": {"color": "#eab308", "bg": "rgba(234, 179, 8, 0.1)", "border": "#eab308"},
                "At-Risk": {"color": "#ef4444", "bg": "rgba(239, 68, 68, 0.1)", "border": "#ef4444"},
            }
            theme = segment_themes.get(segment, {"color": "#94a3b8", "bg": "rgba(148, 163, 184, 0.1)", "border": "#94a3b8"})
            desc = segment_descriptions.get(segment, "")
            
            st.markdown(f"""
            <div class="segment-card" style="background: {theme['bg']}; border: 1px solid {theme['border']}; border-radius: 12px; padding: 20px; margin-top: 15px;">
                <h3 style="color: {theme['color']}; margin: 0 0 10px 0; font-size: 1.5em; font-weight: 600; display: flex; align-items: center; gap: 8px;">
                    {icon} Predicted Segment: {segment}
                </h3>
                <p style="margin: 0; font-size: 1.1em; color: #f1f5f9; line-height: 1.5;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    with sub2:
        st.markdown("### 🧠 Clustering Methodology & RFM Analysis")
        st.write(
            "Customers are segmented using the **K-Means Clustering** algorithm applied to scaled "
            "Recency, Frequency, and Monetary (RFM) metrics. Frequency and Monetary values were "
            "log-transformed to normalize right-skewed distributions before fitting the model."
        )
        
        rfm_df = load_rfm_segments()
        
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("#### 📐 K-Means Elbow Method")
            st.write(
                "The optimal number of clusters ($k=4$) was selected using the **Elbow Method** and **Silhouette Coefficient**. "
                "As shown below, a clear bend/elbow is visible at $k=4$, which also yields a strong average silhouette score."
            )
            st.image(FIGURES_DIR / "elbow_plot.png", caption="Elbow Curve and Silhouette Score Analysis")
            
            # Display Silhouette Scores Table
            st.markdown("**Silhouette Scores by Cluster Count ($k$):**")
            sil_data = pd.DataFrame({
                "Number of Clusters (k)": [2, 3, 4, 5, 6],
                "Silhouette Score": [0.4062, 0.4157, 0.3795, 0.3435, 0.3299],
                "Status": ["High Score (Too Broad)", "Highest Score (Too Broad)", "🎯 Selected (Optimal for Business)", "Suboptimal", "Suboptimal"]
            })
            st.dataframe(
                sil_data.style.format({"Silhouette Score": "{:.4f}"}),
                width="stretch",
                hide_index=True
            )
            st.caption(
                "**Note**: Although $k=2$ has the highest raw Silhouette score, it only splits customers into two broad groups. "
                "Selecting $k=4$ provides a much more actionable segmentation for e-commerce marketing (separating At-Risk from Occasional customers)."
            )

        with col_a2:
            st.markdown("#### 👥 RFM Customer Clusters")
            st.write(
                "The scatter plot below illustrates the customer groupings along the Recency (x-axis) "
                "and Monetary spend (y-axis) dimensions. KMeans splits customers cleanly into 4 distinct behaviors."
            )
            st.image(FIGURES_DIR / "rfm_clusters.png", caption="Customer Segments Visualized (Recency vs Monetary)")

        st.markdown("---")
        st.markdown("#### 📁 Segment Profiles & Statistics")
        st.write("A breakdown of customer statistics and mean RFM scores across the four segments:")
        
        # Calculate group profiles
        segment_profiles = rfm_df.groupby("Segment").agg(
            Count=("CustomerID", "count"),
            Mean_Recency=("Recency", "mean"),
            Mean_Frequency=("Frequency", "mean"),
            Mean_Monetary=("Monetary", "mean")
        ).reset_index()
        
        # Rename columns nicely
        segment_profiles.columns = [
            "Segment Label", "Customer Count", "Average Recency (Days)", 
            "Average Frequency (Orders)", "Average Monetary Spend (£)"
        ]
        
        # Display table with formatting
        st.dataframe(
            segment_profiles.style.format({
                "Customer Count": "{:,}",
                "Average Recency (Days)": "{:.1f}",
                "Average Frequency (Orders)": "{:.1f}",
                "Average Monetary Spend (£)": "£{:,.2f}"
            }),
            width="stretch",
            hide_index=True
        )
        
        # Show segment details card
        st.markdown("#### 🔍 Segment Profiles & Business Action Plans")
        col_card1, col_card2 = st.columns(2)
        with col_card1:
            st.success(
                "🟢 **High-Value Spenders**\n\n"
                "- *Characteristics*: Highly active, frequent buyers who spend premium amounts and bought recently.\n"
                "- *Action*: Reward with VIP programs, exclusive access, and early product launches. Retain at all costs!"
            )
            st.info(
                "🔵 **Regular Customers**\n\n"
                "- *Characteristics*: Steady purchasers with moderate order frequency and total spends.\n"
                "- *Action*: Nurture with upsell recommendations, cross-sell opportunities, and loyalty programs."
            )
        with col_card2:
            st.warning(
                "🟡 **Occasional Buyers**\n\n"
                "- *Characteristics*: Infrequent shoppers who spend small amounts. Long interval between orders.\n"
                "- *Action*: Engage with targeted promotions, seasonal discount campaigns, and reactivation newsletters."
            )
            st.error(
                "🔴 **At-Risk / Inactive**\n\n"
                "- *Characteristics*: Customers who haven't ordered in a very long time, with minimal spend.\n"
                "- *Action*: Deploy win-back marketing emails, feedback questionnaires, or high-discount recovery deals."
            )

with tab3:
    st.subheader("🎯 Find similar products")
    st.write("Enter a product name to get the top 5 recommended products, based on what "
             "customers who bought it also tend to buy.")

    similarity_dict, lookup = load_recommendation_artifacts()
    all_products = sorted(set(lookup.values()))

    product_input = st.selectbox(
        "Choose or search a product",
        options=all_products,
        index=None,
        placeholder="Start typing a product name...",
        key="prod_select"
    )

    if st.button("Get Recommendations", type="primary", key="rec_btn"):
        if not product_input:
            st.warning("Please select a product first.")
        else:
            matched, recs = recommend_similar_products(product_input, similarity_dict, lookup, n=5)
            if not recs:
                st.error("No matching product found. Try a different name.")
            else:
                st.success(f"Because you looked at: **{matched}**")
                cols = st.columns(5)
                for col, (name, score) in zip(cols, recs):
                    with col:
                        st.markdown(f"""
                        <div class="prod-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between; min-height: 120px;">
                            <div style="font-size: 1.05em; font-weight: 600; color: #f8fafc; margin-bottom: 8px; line-height: 1.3;">{name}</div>
                            <div style="font-size: 0.85em; color: #818cf8; font-weight: 600;">similarity: {score}</div>
                        </div>
                        """, unsafe_allow_html=True)

st.divider()
st.caption("Shopper Spectrum project - built with Streamlit, scikit-learn, and pandas.")
