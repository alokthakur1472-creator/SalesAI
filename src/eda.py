import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# SALES AI - EXPLORATORY DATA ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "clean_sales_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("SALES AI - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------
    print("\n[1/6] Loading processed dataset...")

    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # 2. Overall statistics
    # --------------------------------------------------------
    print("\n[2/6] Calculating overall statistics...")

    print(f"\nTotal Sales: ${df['Weekly_Sales'].sum():,.2f}")
    print(f"Average Weekly Sales: ${df['Weekly_Sales'].mean():,.2f}")
    print(f"Maximum Weekly Sales: ${df['Weekly_Sales'].max():,.2f}")
    print(f"Minimum Weekly Sales: ${df['Weekly_Sales'].min():,.2f}")

    # --------------------------------------------------------
    # 3. Store performance
    # --------------------------------------------------------
    print("\n[3/6] Analyzing store performance...")

    store_sales = (
        df.groupby("Store")["Weekly_Sales"]
        .agg(["sum", "mean", "max"])
        .sort_values("sum", ascending=False)
    )

    print("\nTop 10 Stores by Total Sales:")
    print(store_sales.head(10).to_string())

    store_sales.to_csv(
        OUTPUT_DIR / "store_performance.csv"
    )

    # --------------------------------------------------------
    # 4. Monthly sales
    # --------------------------------------------------------
    print("\n[4/6] Analyzing monthly sales...")

    monthly_sales = (
        df.groupby(["Year", "Month"])["Weekly_Sales"]
        .sum()
        .reset_index()
    )

    monthly_sales["Date"] = pd.to_datetime(
        monthly_sales["Year"].astype(str)
        + "-"
        + monthly_sales["Month"].astype(str)
        + "-01"
    )

    monthly_sales = monthly_sales.sort_values("Date")

    monthly_sales.to_csv(
        OUTPUT_DIR / "monthly_sales.csv",
        index=False
    )

    # --------------------------------------------------------
    # 5. Holiday analysis
    # --------------------------------------------------------
    print("\n[5/6] Analyzing holiday impact...")

    holiday_sales = (
        df.groupby("Holiday_Flag")["Weekly_Sales"]
        .agg(["count", "mean", "sum"])
    )

    print("\nHoliday vs Non-Holiday Sales:")
    print(holiday_sales.to_string())

    # --------------------------------------------------------
    # 6. Generate visualizations
    # --------------------------------------------------------
    print("\n[6/6] Creating visualizations...")

    # ---- Chart 1: Total sales by store ----
    plt.figure(figsize=(14, 7))

    store_sales["sum"].sort_values().plot(
        kind="bar"
    )

    plt.title("Total Sales by Store")
    plt.xlabel("Store")
    plt.ylabel("Total Weekly Sales")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "sales_by_store.png",
        dpi=150
    )

    plt.close()

    # ---- Chart 2: Monthly sales trend ----
    plt.figure(figsize=(14, 7))

    plt.plot(
        monthly_sales["Date"],
        monthly_sales["Weekly_Sales"],
        marker="o"
    )

    plt.title("Monthly Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "monthly_sales_trend.png",
        dpi=150
    )

    plt.close()

    # ---- Chart 3: Holiday comparison ----
    plt.figure(figsize=(8, 6))

    holiday_sales["mean"].plot(
        kind="bar"
    )

    plt.title("Average Weekly Sales: Holiday vs Non-Holiday")
    plt.xlabel("Holiday Flag")
    plt.ylabel("Average Weekly Sales")
    plt.xticks(
        [0, 1],
        ["Non-Holiday", "Holiday"],
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "holiday_sales_comparison.png",
        dpi=150
    )

    plt.close()

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nEDA files saved to:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- store_performance.csv")
    print("- monthly_sales.csv")
    print("- sales_by_store.png")
    print("- monthly_sales_trend.png")
    print("- holiday_sales_comparison.png")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()