import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (train_test_split,GridSearchCV,cross_validate)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(filepath):
    """Load CSV with a timestamp column."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return df


def preprocess_data(df):
    """Clean, filter, feature-engineer, and return X, y (log1p target)."""
    df = df.copy()
    # Timestamp → cyclical features
    df.loc[:, "timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.loc[:, "hour"]        = df["timestamp"].dt.hour
    df.loc[:, "day"]         = df["timestamp"].dt.day
    df.loc[:, "month"]       = df["timestamp"].dt.month
    df.loc[:, "day_of_week"] = df["timestamp"].dt.dayofweek
    df.loc[:, "hour_sin"]    = np.sin(2 * np.pi * df["hour"]/24)
    df.loc[:, "hour_cos"]    = np.cos(2 * np.pi * df["hour"]/24)
    df.drop(columns=["timestamp"], inplace=True)

    # Convert objects → numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing values without deprecated downcasting
    df = df.bfill().infer_objects(copy=False)
    df = df.fillna(df.mean(numeric_only=True)).infer_objects(copy=False)

    # Sensor & environment filters
    temp_cols     = [f"zone{i}_temperature" for i in range(1, 10)]
    humidity_cols = [f"zone{i}_humidity"    for i in range(1, 10)]
    for col in temp_cols:
        df = df[(df[col] >= 10) & (df[col] <= 30)]
    for col in humidity_cols:
        df = df[(df[col] >= 0) & (df[col] <= 100)]
    df = df[(df['outdoor_temperature'] >= 0) & (df['outdoor_temperature'] <= 40)]
    df = df[(df['outdoor_humidity'] >= 0) & (df['outdoor_humidity'] <= 100)]
    df = df[(df['atmospheric_pressure'] >= 720) & (df['atmospheric_pressure'] <= 800)]
    df = df[(df['wind_speed'] >= 0) & (df['wind_speed'] <= 15)]
    df = df[(df['visibility_index'] >= 0) & (df['visibility_index'] <= 100)]
    df = df[(df['dew_point'] >= -10) & (df['dew_point'] <= 25)]
    df = df[(df['random_variable1'] >= 0) & (df['random_variable1'] <= 50)]
    df = df[df["lighting_energy"] >= 0]
    df = df[df["equipment_energy_consumption"] >= 0]

    # Log-transform the target
    df.loc[:, "equipment_energy_consumption"] = np.log1p(df["equipment_energy_consumption"])

    # Zone-level statistics
    df.loc[:, 'mean_zone_temp']     = df[temp_cols].mean(axis=1)
    df.loc[:, 'std_zone_temp']      = df[temp_cols].std(axis=1)
    df.loc[:, 'min_zone_temp']      = df[temp_cols].min(axis=1)
    df.loc[:, 'max_zone_temp']      = df[temp_cols].max(axis=1)
    df.loc[:, 'mean_zone_humidity'] = df[humidity_cols].mean(axis=1)
    df.loc[:, 'std_zone_humidity']  = df[humidity_cols].std(axis=1)
    df.loc[:, 'min_zone_humidity']  = df[humidity_cols].min(axis=1)
    df.loc[:, 'max_zone_humidity']  = df[humidity_cols].max(axis=1)
    df.loc[:, 'temp_humidity_index'] = df['mean_zone_temp'] * df['mean_zone_humidity'] / 100

    # Select features & target
    feature_cols = [
        'lighting_energy', 'outdoor_temperature', 'wind_speed', 'dew_point', 'random_variable1',
        'mean_zone_temp', 'std_zone_temp', 'min_zone_temp', 'max_zone_temp',
        'mean_zone_humidity', 'std_zone_humidity', 'min_zone_humidity', 'max_zone_humidity',
        'temp_humidity_index', 'hour_sin', 'hour_cos', 'day_of_week', 'month'
    ]
    X = df[feature_cols].copy()
    y = df['equipment_energy_consumption'].copy()
    return X, y


def train_evaluate_rf_pipeline(
    X, y,
    test_size: float = 0.2,
    random_state: int = 42,
    param_grid: dict = None,
    cv: int = 5
):
    """
    Split into train/test, fit a Pipeline(StandardScaler -> RF), optionally tune via GridSearchCV,
    compute cross-validation metrics, then evaluate on test. Returns the fitted model, test metrics,
    and a dict of CV metrics (on log scale).
    """
    # 1) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2) Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=random_state))
    ])

    # 3) hyperparameter tuning with GridSearchCV
    if param_grid:
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print("Best params:", search.best_params_)
    else:
        pipeline.fit(X_train, y_train)
        model = pipeline

    # 4) Cross-validation metrics on the training set (log scale)
    scoring = {
        'RMSE': 'neg_root_mean_squared_error',
        'MAE':  'neg_mean_absolute_error',
        'R2':   'r2'
    }
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
    )
    cv_rmse = -cv_results['test_RMSE']
    cv_mae  = -cv_results['test_MAE']
    cv_r2   =  cv_results['test_R2']
    cv_metrics = {
        'CV RMSE Mean': np.mean(cv_rmse),
        'CV RMSE Std':  np.std(cv_rmse),
        'CV MAE Mean':  np.mean(cv_mae),
        'CV MAE Std':   np.std(cv_mae),
        'CV R2 Mean':   np.mean(cv_r2),
        'CV R2 Std':    np.std(cv_r2)
    }
    print("Cross-validation metrics (log1p scale):", cv_metrics)

    # 5) Final evaluation on the test set (invert log1p)
    y_pred_log = model.predict(X_test)
    y_true = np.expm1(y_test)
    y_pred = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R²:   {r2:.4f}")

    test_metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    return model, test_metrics, cv_metrics


def main():
    # Adjust filepath as needed
    filepath = 'data/data.csv'
    df = load_data(filepath)
    X, y = preprocess_data(df)

    # Hyperparameter grid for GridSearchCV
    param_grid = {
        'rf__n_estimators': [50, 100, 200, 300, 400, 500],
        'rf__max_depth':    [5, 10, 15, 20, None],
        'rf__max_features': ['sqrt', 'log2'],
        'rf__min_samples_leaf': [1, 2, 4, 6]
    }

    model, test_metrics, cv_metrics = train_evaluate_rf_pipeline(
        X, y,
        test_size=0.2,
        random_state=42,
        param_grid=param_grid,
        cv=5
    )

    print("\nFinal Test Metrics:", test_metrics)
    print("Final CV Metrics:", cv_metrics)


if __name__ == '__main__':
    main()
