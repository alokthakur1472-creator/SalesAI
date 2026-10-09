import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# SALES AI - FORECASTING ENGINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "clean_sales_data.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "forecasts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Evaluation functions
# ------------------------------------------------------------

def calculate_mape(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    return np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100


def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = calculate_mape(
        actual,
        predicted
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

def create_features(df):

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])

    # Sort by store and date
    df = df.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    # Calendar features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    # Lag features
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

    # Rolling features
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SALES AI - FORECASTING ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    print("\n[1/8] Loading processed dataset...")

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Rows: {len(df):,}")
    print(f"Stores: {df['Store'].nunique()}")
    print(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # 2. Create features
    # --------------------------------------------------------

    print("\n[2/8] Creating forecasting features...")

    df = create_features(df)

    # Remove rows without sufficient lag history
    df_model = df.dropna().copy()

    print(
        f"Rows available after feature engineering: "
        f"{len(df_model):,}"
    )

    # --------------------------------------------------------
    # 3. Time-based train/test split
    # --------------------------------------------------------

    print("\n[3/8] Creating time-based train/test split...")

    # Last 12 weeks are used as test data
    unique_dates = sorted(
        df_model["Date"].unique()
    )

    test_dates = unique_dates[-12:]

    train = df_model[
        ~df_model["Date"].isin(test_dates)
    ].copy()

    test = df_model[
        df_model["Date"].isin(test_dates)
    ].copy()

    print(
        f"Training rows: {len(train):,}"
    )

    print(
        f"Testing rows: {len(test):,}"
    )

    print(
        f"Test period: "
        f"{test['Date'].min().date()} "
        f"to "
        f"{test['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # 4. Select features
    # --------------------------------------------------------

    print("\n[4/8] Preparing model features...")

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

    TARGET = "Weekly_Sales"

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # --------------------------------------------------------
    # 5. Baseline
    # --------------------------------------------------------

    print("\n[5/8] Evaluating baseline model...")

    baseline_predictions = test["Lag_1"]

    results = []

    results.append(
        evaluate_model(
            "Previous Week Baseline",
            y_test,
            baseline_predictions
        )
    )

    # --------------------------------------------------------
    # 6. Random Forest
    # --------------------------------------------------------

    print("\n[6/8] Training Random Forest...")

    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(
        X_train,
        y_train
    )

    rf_predictions = rf_model.predict(
        X_test
    )

    results.append(
        evaluate_model(
            "Random Forest",
            y_test,
            rf_predictions
        )
    )

    # --------------------------------------------------------
    # 7. XGBoost
    # --------------------------------------------------------

    print("\n[7/8] Training XGBoost...")

    xgb_model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(
        X_train,
        y_train,
        verbose=False
    )

    xgb_predictions = xgb_model.predict(
        X_test
    )

    results.append(
        evaluate_model(
            "XGBoost",
            y_test,
            xgb_predictions
        )
    )

    # --------------------------------------------------------
    # 8. Results
    # --------------------------------------------------------

    print("\n[8/8] Comparing models...")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "MAPE"
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    for _, row in results_df.iterrows():

        print(
            f"\n{row['Model']}"
        )

        print(
            f"  MAE  : ${row['MAE']:,.2f}"
        )

        print(
            f"  RMSE : ${row['RMSE']:,.2f}"
        )

        print(
            f"  MAPE : {row['MAPE']:.2f}%"
        )

        print(
            f"  R²   : {row['R2']:.4f}"
        )

    # Save model comparison
    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    best_model_name = results_df.iloc[0]["Model"]

    print("\n" + "=" * 70)
    print(
        f"BEST MODEL: {best_model_name}"
    )
    print("=" * 70)

    if best_model_name == "XGBoost":
        best_model = xgb_model
        best_predictions = xgb_predictions

    elif best_model_name == "Random Forest":
        best_model = rf_model
        best_predictions = rf_predictions

    else:
        best_model = None
        best_predictions = baseline_predictions

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    predictions_df = test[
        ["Store", "Date", "Weekly_Sales"]
    ].copy()

    predictions_df["Predicted_Sales"] = best_predictions

    predictions_df["Error"] = (
        predictions_df["Weekly_Sales"]
        - predictions_df["Predicted_Sales"]
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if best_model is not None:

        model_path = MODEL_DIR / "best_sales_model.pkl"

        joblib.dump(
            best_model,
            model_path
        )

        print(
            f"\nModel saved to:"
        )

        print(model_path)

    # Save feature list
    with open(
        MODEL_DIR / "feature_list.txt",
        "w"
    ) as f:

        for feature in FEATURES:
            f.write(feature + "\n")

    print(
        "\nForecast outputs saved to:"
    )

    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- model_comparison.csv")
    print("- test_predictions.csv")
    print("- best_sales_model.pkl")
    print("- feature_list.txt")

    print("\n" + "=" * 70)
    print("FORECASTING ENGINE COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()