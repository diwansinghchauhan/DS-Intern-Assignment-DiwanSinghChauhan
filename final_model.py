import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return df


def preprocess_data(df):
    """Clean and preprocess Smart Factory dataset for energy prediction."""

    # 🧹 Remove duplicates
    df = df.drop_duplicates()
    print("After removing duplicates:", df.shape)

    # ⏱️ Parse timestamp early for feature extraction
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

    # 🕒 Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Now drop timestamp
    df.drop(columns=["timestamp"], inplace=True)

    # 🔁 Replace 'unknown' with NaN if present (safe fallback)
    df.replace("unknown", np.nan, inplace=True)

    # 🔢 Convert object columns to numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 📉 Handle missing values
    df.fillna(method='bfill', inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 🧽 Sensor value filters
    temp_cols = [f"zone{i}_temperature" for i in range(1, 10)]
    humidity_cols = [f"zone{i}_humidity" for i in range(1, 10)]

    for col in temp_cols:
        df = df[df[col].between(10, 30)]
    for col in humidity_cols:
        df = df[df[col].between(0, 100)]

    df = df[df["outdoor_temperature"].between(0, 40)]
    df = df[df["outdoor_humidity"].between(0, 100)]
    df = df[df["atmospheric_pressure"].between(720, 800)]
    df = df[df["wind_speed"].between(0, 15)]
    df = df[df["visibility_index"].between(0, 100)]
    df = df[df["dew_point"].between(-10, 25)]

    # 🎲 Special feature filters
    df = df[df["random_variable1"].between(0, 50)]
    df = df[df["lighting_energy"] >= 0]

    # ⚠️ Remove rows with invalid target
    df = df[df["equipment_energy_consumption"] >= 0]

    # 🔄 Log-transform target to reduce skew
    df["equipment_energy_consumption"] = np.log1p(df["equipment_energy_consumption"])

    # 📊 Zone-level feature engineering
    df['mean_zone_temp'] = df[temp_cols].mean(axis=1)
    df['std_zone_temp'] = df[temp_cols].std(axis=1)
    df['min_zone_temp'] = df[temp_cols].min(axis=1)
    df['max_zone_temp'] = df[temp_cols].max(axis=1)

    df['mean_zone_humidity'] = df[humidity_cols].mean(axis=1)
    df['std_zone_humidity'] = df[humidity_cols].std(axis=1)
    df['min_zone_humidity'] = df[humidity_cols].min(axis=1)
    df['max_zone_humidity'] = df[humidity_cols].max(axis=1)

    # 🌡️ Thermal comfort index
    df['temp_humidity_index'] = df['mean_zone_temp'] * df['mean_zone_humidity'] / 100

    # 🎯 Separate features and target
    y = df["equipment_energy_consumption"]
    X = df.drop(columns=["equipment_energy_consumption"])

    return X, y

