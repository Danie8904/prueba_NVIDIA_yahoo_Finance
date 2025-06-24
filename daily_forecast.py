import mlflow
import mlflow.sklearn
import yfinance as yf
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Yahoo Finance - Pronosticos 
TICKER = "NVDA"
YEARS_OF_DATA = 10
FORECAST_DAYS = 30
TEST_DAYS = 30

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de evaluación"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(np.sqrt(mse), 4),
        'MAPE_percent': round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 4)
    }

def download_stock_data():
    """Descarga datos históricos del ticker"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(
            TICKER, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=True
        )
    return data[['Close']].dropna()

def create_features(df):
    """Genera variables predictoras"""
    df = df.copy()
    df['Date'] = df.index
    df['Day'] = df['Date'].dt.day.astype(int)
    df['Month'] = df['Date'].dt.month.astype(int)
    df['Year'] = df['Date'].dt.year.astype(int)
    df['DayOfWeek'] = df['Date'].dt.weekday.astype(int)
    df['Close_Lag1'] = df['Close'].shift(1).astype(float)
    return df.dropna()

def split_data(df):
    """Divide el conjunto de datos en entrenamiento y prueba"""
    train = df[:-TEST_DAYS]
    test = df[-TEST_DAYS:]
    
    features = ['Day', 'Month', 'Year', 'DayOfWeek', 'Close_Lag1']
    X_train = train[features].values
    y_train = train['Close'].values.ravel()
    X_test = test[features].values
    y_test = test['Close'].values.ravel()
    
    return X_train, y_train, X_test, y_test, test.index

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Configuración de MLflow dentro del main
    #mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("NVDA_Forecast")

    with mlflow.start_run():
        try:
            print("Descargando datos...")
            data = download_stock_data()

            print("Generando características...")
            df = create_features(data)

            print("Dividiendo datos...")
            X_train, y_train, X_test, y_test, test_dates = split_data(df)

            print("Entrenando modelo...")
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)

            print("Generando predicciones...")
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)

            # Registro en MLflow
            mlflow.log_params({
                "ticker": TICKER,
                "years": YEARS_OF_DATA,
                "test_days": TEST_DAYS,
                "n_estimators": 100,
                "max_depth": 10
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            # Guardar archivo CSV con resultados
            os.makedirs("forecasts", exist_ok=True)
            today = datetime.now().strftime("%Y%m%d")
            results = pd.DataFrame({
                'Date': test_dates,
                'Actual': y_test,
                'Predicted': y_pred
            })
            results.to_csv(f"forecasts/nvda_forecast_{today}.csv", index=False)

            print("\n Métricas:")
            for k, v in metrics.items():
                print(f"{k}: {v}")

            print(f"\n Resultados guardados en:")
            print(f"- MLflow: http://localhost:5000")
            print(f"- Archivo: forecasts/nvda_forecast_{today}.csv")

        except Exception as e:
            mlflow.set_tag("error", str(e))
            print(f" Error: {e}")
            raise

     


