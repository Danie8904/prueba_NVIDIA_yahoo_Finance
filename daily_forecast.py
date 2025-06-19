# daily_forecast.py
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- CONFIGURACIÓN ---
# Fechas: últimos 10 años desde hoy
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * 10)

# 1. Descargar datos de NVDA últimos 10 años
nvda = yf.download("NVDA", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

# 2. Preparar los datos para Prophet
df = nvda.reset_index()[['Date', 'Close']]
df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
df['ds'] = df['ds'].dt.tz_localize(None)

# 3. Entrenar modelo Prophet
model = Prophet(daily_seasonality=True)
model.fit(df)

# 4. Predecir para mañana
future = model.make_future_dataframe(periods=1)
forecast = model.predict(future)

# 5. Extraer predicción del día siguiente
prediccion_manana = forecast.iloc[-1][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# 6. Calcular errores sobre los datos históricos
forecast_trim = forecast[forecast['ds'].isin(df['ds'])].copy()
df_eval = df.merge(forecast_trim[['ds', 'yhat']], on='ds')
y_true = df_eval['y']
y_pred = df_eval['yhat']

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 7. Crear tabla resumen de errores
errores_df = pd.DataFrame({
    'Métrica': ['MAE', 'MSE', 'RMSE', 'MAPE'],
    'Valor': [mae, mse, rmse, mape]
}).round(2)

# 8. Guardar resultados
hoy = datetime.now().strftime("%Y-%m-%d")
os.makedirs("forecasts", exist_ok=True)

pred_path = f"forecasts/prediccion_{hoy}.csv"
err_path  = f"forecasts/errores_{hoy}.csv"

prediccion_manana.to_frame().T.to_csv(pred_path, index=False)
errores_df.to_csv(err_path, index=False)

# 9. Mostrar en consola
print("✅ Predicción guardada en:", pred_path)
print(prediccion_manana)
print("\n📊 Tabla de errores:")
print(errores_df)
