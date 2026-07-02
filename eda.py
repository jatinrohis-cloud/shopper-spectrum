"""
Shopper Spectrum - Step 2: Exploratory Data Analysis (EDA)
------------------------------------------------------------
Reads the cleaned dataset and generates business insight charts:
    - Top-selling products
    - Country-wise sales
    - Monthly sales trend
    - Customer purchase pattern (orders per customer)
    - Revenue analysis
    - Most active customers

All charts are saved as PNG files in outputs/figures/.

Run:
    python src/eda.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt

CLEAN_PATH = os.path.join("data", "cleaned_data.csv")
FIG_DIR = os.path.join("outputs", "figures")

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"


def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH, parse_dates=["InvoiceDate"])
    return df


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved: {path}")


def top_selling_products(df: pd.DataFrame, n=10):
    top = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots()
    top.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"Top {n} Best-Selling Products (by Quantity)")
    ax.set_xlabel("Quantity Sold")
    ax.set_ylabel("")
    save_fig(fig, "top_selling_products.png")
    return top


def country_wise_sales(df: pd.DataFrame, n=10):
    sales = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(n)
    fig, ax = plt.subplots()
    sales.sort_values().plot(kind="barh", ax=ax, color="#55A868")
    ax.set_title(f"Top {n} Countries by Revenue")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("")
    save_fig(fig, "country_wise_sales.png")
    return sales


def monthly_sales_trend(df: pd.DataFrame):
    monthly = df.set_index("InvoiceDate").resample("ME")["TotalPrice"].sum()
    fig, ax = plt.subplots()
    monthly.plot(ax=ax, marker="o", color="#C44E52")
    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    save_fig(fig, "monthly_sales_trend.png")
    return monthly


def customer_purchase_pattern(df: pd.DataFrame):
    orders_per_customer = df.groupby("CustomerID")["InvoiceNo"].nunique()
    fig, ax = plt.subplots()
    orders_per_customer.plot(kind="hist", bins=50, ax=ax, color="#8172B2")
    ax.set_title("Distribution of Orders per Customer")
    ax.set_xlabel("Number of Orders")
    ax.set_ylabel("Number of Customers")
    save_fig(fig, "orders_per_customer.png")
    return orders_per_customer


def revenue_analysis(df: pd.DataFrame):
    total_revenue = df["TotalPrice"].sum()
    avg_order_value = df.groupby("InvoiceNo")["TotalPrice"].sum().mean()
    print(f"\nTotal Revenue: {total_revenue:,.2f}")
    print(f"Average Order Value: {avg_order_value:,.2f}")
    return total_revenue, avg_order_value


def most_active_customers(df: pd.DataFrame, n=10):
    top_customers = (
        df.groupby("CustomerID")["TotalPrice"].sum().sort_values(ascending=False).head(n)
    )
    fig, ax = plt.subplots()
    top_customers.sort_values().plot(kind="barh", ax=ax, color="#CCB974")
    ax.set_title(f"Top {n} Customers by Spend")
    ax.set_xlabel("Total Spend")
    ax.set_ylabel("Customer ID")
    save_fig(fig, "top_customers.png")
    return top_customers


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_clean_data()
    print(f"Loaded cleaned data: {df.shape}")

    print("\n--- Top Selling Products ---")
    print(top_selling_products(df))

    print("\n--- Country-wise Sales ---")
    print(country_wise_sales(df))

    print("\n--- Monthly Sales Trend ---")
    print(monthly_sales_trend(df))

    print("\n--- Customer Purchase Pattern ---")
    print(customer_purchase_pattern(df).describe())

    print("\n--- Revenue Analysis ---")
    revenue_analysis(df)

    print("\n--- Most Active Customers ---")
    print(most_active_customers(df))

    print(f"\nAll EDA charts saved in: {FIG_DIR}")


if __name__ == "__main__":
    main()
