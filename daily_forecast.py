import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

# Configuración
TICKER = "NVDA"
YEARS_OF_DATA = 10
TEST_DAYS = 30

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 4)
    }

def download_stock_data():
    """Descarga datos históricos de la acción"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * YEARS_OF_DATA)
    data = yf.download(TICKER, start=start_date, end=end_date, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    return data[['Close']].dropna()

def create_features(df):
    """Crea variables predictoras"""
    df['Fecha'] = df.index
    df['Dia'] = df['Fecha'].dt.day
    df['Mes'] = df['Fecha'].dt.month
    df['Año'] = df['Fecha'].dt.year
    df['Dia_Semana'] = df['Fecha'].dt.weekday
    df['Close_Lag1'] = df['Close'].shift(1)
    df = df.dropna()
    return df

def split_data(df):
    """Divide en train y test"""
    df_train = df[:-TEST_DAYS]
    df_test = df[-TEST_DAYS:]

    X_train = df_train[['Dia', 'Mes', 'Año', 'Dia_Semana', 'Close_Lag1']]
    y_train = df_train['Close']
    X_test = df_test[['Dia', 'Mes', 'Año', 'Dia_Semana', 'Close_Lag1']]
    y_test = df_test['Close']

    return X_train, y_train, X_test, y_test, df_test

def ensure_folder(path):
    os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    print("🔍 Descargando datos...")
    data = download_stock_data()
    print(f"📄 Columnas descargadas: {data.columns}")
    print(f"🔢 Tamaño del dataset: {data.shape}")

    df = create_features(data)
    print("🧪 Separando en entrenamiento y prueba...")
    X_train, y_train, X_test, y_test, df_test = split_data(df)

    ensure_folder("forecasts")

    with mlflow.start_run():
        print("🌲 Entrenando modelo Random Forest...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        print("🔮 Generando predicciones...")
        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)

        print("\n📈 Métricas de Error:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            mlflow.log_metric(k, v)

        # Guardar predicciones
        df_resultados = pd.DataFrame({
            "Fecha": df_test['Fecha'].dt.strftime("%Y-%m-%d"),
            "Real": y_test.values,
            "Predicción": y_pred
        })
        resultados_path = "forecasts/predicciones_mlflow.csv"
        df_resultados.to_csv(resultados_path, index=False)

        # Log de parámetros y artefactos
        mlflow.log_param("modelo", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("test_days", TEST_DAYS)
        mlflow.sklearn.log_model(model, "modelo_random_forest")
        mlflow.log_artifact(resultados_path)

        print("✅ Resultados guardados y experimento registrado en MLflow.")




