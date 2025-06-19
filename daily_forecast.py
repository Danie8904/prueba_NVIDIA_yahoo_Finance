import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuración
TICKER = "NVDA"
YEARS_OF_DATA = 10
FORECAST_DAYS = 30
TEST_DAYS = 30

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAPE (%)': round(mape, 4)
    }

def download_stock_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=YEARS_OF_DATA * 365)
    data = yf.download(TICKER, start=start_date, end=end_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    print(f"📄 Columnas descargadas: {data.columns}")
    print(f"🔢 Tamaño del dataset: {data.shape}")
    price_column = 'Close' if 'Close' in data.columns else 'Adj Close'
    return data, price_column

def prepare_data(data, price_column):
    df = data[[price_column]].copy()
    df = df.reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(inplace=True)
    return df

def build_features(df):
    # Convertimos la fecha a ordinal para que RF la entienda
    df['ds_ordinal'] = df['ds'].map(datetime.toordinal)
    return df[['ds', 'ds_ordinal', 'y']]

def forecast_future_dates(df, model):
    last_date = df['ds'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, FORECAST_DAYS + 1)]
    future_ordinals = [d.toordinal() for d in future_dates]
    
    X_future = pd.DataFrame({'ds': future_dates, 'ds_ordinal': future_ordinals})
    X_pred = X_future[['ds_ordinal']]
    y_pred = model.predict(X_pred)
    
    future_forecast = X_future.copy()
    future_forecast['yhat'] = y_pred
    return future_forecast

def save_results(forecast, metrics):
    os.makedirs("forecasts", exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    
    forecast.to_csv(f"forecasts/nvda_forecast_rf_{today}.csv", index=False)
    pd.DataFrame([metrics]).to_csv(f"forecasts/nvda_metrics_rf_{today}.csv", index=False)
    with open(f"forecasts/metrics_log_rf_{today}.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    try:
        print("🔍 Descargando datos...")
        data, price_column = download_stock_data()
        print(f"📊 Usando columna de precios: {price_column}")
        
        df = prepare_data(data, price_column)
        df = build_features(df)

        print("🧪 Separando en entrenamiento y prueba...")
        train = df[:-TEST_DAYS]
        test = df[-TEST_DAYS:]

        X_train = train[['ds_ordinal']]
        y_train = train['y']
        X_test = test[['ds_ordinal']]
        y_test = test['y']

        print("🌲 Entrenando modelo Random Forest...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        print("🔮 Generando predicciones...")
        y_pred_test = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred_test)

        # Forecast de los próximos 30 días
        future_forecast = forecast_future_dates(df, model)

        print("\n📈 Métricas de Error:")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        save_results(future_forecast, metrics)
        print("\n✅ Resultados guardados en /forecasts/")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")



