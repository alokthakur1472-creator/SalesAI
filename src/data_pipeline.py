import pandas as pd
from pathlib import Path


# ============================================================
# SALES AI - DATA PIPELINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "sales_data.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "clean_sales_data.csv"


def main():

    print("=" * 60)
    print("SALES AI - DATA PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    print("\n[1/7] Loading dataset...")

    df = pd.read_csv(RAW_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    # --------------------------------------------------------
    # 2. Standardize column names
    # --------------------------------------------------------
    print("\n[2/7] Standardizing column names...")

    df.columns = df.columns.str.strip()

    print("Columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # 3. Convert Date column
    # --------------------------------------------------------
    print("\n[3/7] Converting Date column...")

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    invalid_dates = df["Date"].isna().sum()

    print(f"Invalid dates: {invalid_dates}")

    if invalid_dates > 0:
        raise ValueError("Invalid dates found in dataset.")

    # --------------------------------------------------------
    # 4. Check missing values and duplicates
    # --------------------------------------------------------
    print("\n[4/7] Checking data quality...")

    missing_values = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    print(f"Total missing values: {missing_values}")
    print(f"Duplicate rows: {duplicates}")

    if missing_values > 0:
        print("\nMissing values by column:")
        print(df.isnull().sum())

    # Remove exact duplicate rows if any
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Duplicates removed: {duplicates}")

    # --------------------------------------------------------
    # 5. Feature engineering
    # --------------------------------------------------------
    print("\n[5/7] Creating time-based features...")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day

    # Weekend/day information is not meaningful for weekly data,
    # so we intentionally do not create it.

    # --------------------------------------------------------
    # 6. Sort data
    # --------------------------------------------------------
    print("\n[6/7] Sorting dataset...")

    df = df.sort_values(
        by=["Store", "Date"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 7. Save processed dataset
    # --------------------------------------------------------
    print("\n[7/7] Saving processed dataset...")

    PROCESSED_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nProcessed rows: {len(df):,}")
    print(f"Processed columns: {len(df.columns)}")

    print(
        f"\nDate range: "
        f"{df['Date'].min().strftime('%d-%m-%Y')} "
        f"to "
        f"{df['Date'].max().strftime('%d-%m-%Y')}"
    )

    print(
        f"\nTotal Weekly Sales: "
        f"${df['Weekly_Sales'].sum():,.2f}"
    )

    print(
        f"Average Weekly Sales: "
        f"${df['Weekly_Sales'].mean():,.2f}"
    )

    print(
        f"Number of Stores: "
        f"{df['Store'].nunique()}"
    )

    print("\nFinal columns:")
    print(df.columns.tolist())

    print(f"\nSaved to:")
    print(PROCESSED_FILE)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()