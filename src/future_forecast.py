import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# ============================================================
# SALES AI - FUTURE SALES FORECAST
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "clean_sales_data.csv"
MODEL_FILE = BASE_DIR / "models" / "best_sales_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs" / "forecasts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "Store",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Year",
    "Month",
    "Quarter",
    "Week",
    "Lag_1",
    "Lag_2",
    "Lag_4",
    "Lag_8",
    "Rolling_Mean_4",
    "Rolling_Mean_8"
]


def create_features(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    df["Lag_1"] = (
        df.groupby("Store")["Weekly_Sales"]
        .shift(1)
    )

    df["Lag_2"] = (
        df.groupby("Store")["Weekly_Sales"]
        .shift(2)
    )

    df["Lag_4"] = (
        df.groupby("Store")["Weekly_Sales"]
        .shift(4)
    )

    df["Lag_8"] = (
        df.groupby("Store")["Weekly_Sales"]
        .shift(8)
    )

    df["Rolling_Mean_4"] = (
        df.groupby("Store")["Weekly_Sales"]
        .transform(
            lambda x: x.shift(1).rolling(4).mean()
        )
    )

    df["Rolling_Mean_8"] = (
        df.groupby("Store")["Weekly_Sales"]
        .transform(
            lambda x: x.shift(1).rolling(8).mean()
        )
    )

    return df


def main():

    print("=" * 70)
    print("SALES AI - FUTURE SALES FORECAST")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    print("\n[1/6] Loading data...")

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Rows: {len(df):,}")
    print(f"Stores: {df['Store'].nunique()}")

    # --------------------------------------------------------
    # 2. Load model
    # --------------------------------------------------------

    print("\n[2/6] Loading XGBoost model...")

    model = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # 3. Prepare historical data
    # --------------------------------------------------------

    print("\n[3/6] Preparing historical data...")

    df = create_features(df)

    last_date = df["Date"].max()

    print(
        f"Last historical date: "
        f"{last_date.date()}"
    )

    # --------------------------------------------------------
    # 4. Generate future dates
    # --------------------------------------------------------

    print("\n[4/6] Generating future dates...")

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=7),
        periods=12,
        freq="7D"
    )

    print(
        f"Forecast period: "
        f"{future_dates[0].date()} "
        f"to "
        f"{future_dates[-1].date()}"
    )

    # --------------------------------------------------------
    # 5. Recursive forecasting
    # --------------------------------------------------------

    print("\n[5/6] Generating 12-week forecasts...")

    all_forecasts = []

    for store in sorted(df["Store"].unique()):

        store_history = df[
            df["Store"] == store
        ].copy()

        sales_history = list(
            store_history["Weekly_Sales"]
        )

        # Use latest economic/weather values as fallback
        latest_row = store_history.iloc[-1]

        for forecast_date in future_dates:

            lag_1 = sales_history[-1]
            lag_2 = sales_history[-2]
            lag_4 = sales_history[-4]
            lag_8 = sales_history[-8]

            rolling_4 = np.mean(
                sales_history[-4:]
            )

            rolling_8 = np.mean(
                sales_history[-8:]
            )

            row = {
                "Store": store,
                "Holiday_Flag": 0,
                "Temperature": latest_row["Temperature"],
                "Fuel_Price": latest_row["Fuel_Price"],
                "CPI": latest_row["CPI"],
                "Unemployment": latest_row["Unemployment"],
                "Year": forecast_date.year,
                "Month": forecast_date.month,
                "Quarter": forecast_date.quarter,
                "Week": int(forecast_date.isocalendar().week),
                "Lag_1": lag_1,
                "Lag_2": lag_2,
                "Lag_4": lag_4,
                "Lag_8": lag_8,
                "Rolling_Mean_4": rolling_4,
                "Rolling_Mean_8": rolling_8
            }

            X_future = pd.DataFrame(
                [row]
            )[FEATURES]

            prediction = model.predict(
                X_future
            )[0]

            prediction = max(
                0,
                prediction
            )

            all_forecasts.append(
                {
                    "Store": store,
                    "Date": forecast_date,
                    "Forecasted_Sales": prediction
                }
            )

            sales_history.append(
                prediction
            )

    forecast_df = pd.DataFrame(
        all_forecasts
    )

    # --------------------------------------------------------
    # 6. Save forecast
    # --------------------------------------------------------

    print("\n[6/6] Saving forecast...")

    forecast_file = (
        OUTPUT_DIR / "future_forecast.csv"
    )

    forecast_df.to_csv(
        forecast_file,
        index=False
    )

    # Overall weekly forecast
    overall_forecast = (
        forecast_df
        .groupby("Date")["Forecasted_Sales"]
        .sum()
        .reset_index()
    )

    overall_forecast.to_csv(
        OUTPUT_DIR / "overall_future_forecast.csv",
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUTURE FORECAST COMPLETED")
    print("=" * 70)

    print(
        f"\nForecast rows: "
        f"{len(forecast_df):,}"
    )

    print(
        f"Stores forecasted: "
        f"{forecast_df['Store'].nunique()}"
    )

    print(
        f"Weeks forecasted: "
        f"{forecast_df['Date'].nunique()}"
    )

    print(
        f"\nTotal forecasted sales: "
        f"${forecast_df['Forecasted_Sales'].sum():,.2f}"
    )

    print(
        f"Average forecast per store/week: "
        f"${forecast_df['Forecasted_Sales'].mean():,.2f}"
    )

    print("\nForecast files saved:")
    print(f"- {forecast_file}")
    print(
        f"- {OUTPUT_DIR / 'overall_future_forecast.csv'}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()