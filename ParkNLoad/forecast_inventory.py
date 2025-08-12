import pandas as pd
import os
import pickle
from prophet import Prophet
import matplotlib.pyplot as plt

# Paths
plots_dir = "ParkNLoad/plots"
os.makedirs(plots_dir, exist_ok=True)

model_file = "ParkNLoad/forecast_model.pkl"
forecast_summary_file = "ParkNLoad/forecast_summary.csv"

# Load data
df = pd.read_csv("ParkNLoad/inventory_data_large.csv")
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# Unique warehouse IDs
warehouses = df['warehouse_id'].unique()

# Try loading existing models safely
warehouse_models = {}
if os.path.exists(model_file):
    try:
        with open(model_file, "rb") as f:
            warehouse_models = pickle.load(f)
        print("✅ Loaded saved models.")
    except EOFError:
        print("⚠️ Model file is empty or corrupted. Retraining all models...")
        warehouse_models = {}
else:
    print("📊 No saved model found. Training new models...")

# Train missing models
for wh in warehouses:
    if wh not in warehouse_models:
        wh_df = df[df['warehouse_id'] == wh].groupby('datetime').agg({'stock': 'mean'}).reset_index()
        wh_df.rename(columns={'datetime': 'ds', 'stock': 'y'}, inplace=True)

        model = Prophet()
        model.fit(wh_df)

        warehouse_models[wh] = model
        print(f"🆕 Trained model for {wh}")

# Save updated models
with open(model_file, "wb") as f:
    pickle.dump(warehouse_models, f)
print("💾 All models saved/updated.")

# Forecast & save
forecast_summaries = []
for wh in warehouses:
    wh_info = df[df['warehouse_id'] == wh].iloc[0]
    wh_name = wh_info['warehouse_name']
    dims = f"{wh_info['x_length_m']}×{wh_info['y_length_m']}×{wh_info['z_height_m']} m"

    model = warehouse_models[wh]
    future = model.make_future_dataframe(periods=30, freq='D')  # 1-month forecast
    forecast = model.predict(future)

    # Plot forecast
    fig = model.plot(forecast)
    plt.title(f"{wh_name} Forecast (Size: {dims})")
    plt.xlabel("Date")
    plt.ylabel("Stock Level")
    plot_path = os.path.join(plots_dir, f"{wh_name.replace(' ', '_')}_forecast.png")
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"📈 Saved plot for {wh_name} at {plot_path}")

    # Forecast summary in English
    summary_text = f"WAREHOUSE: {wh_name} (Size: {dims})\n"
    for _, row in forecast.tail(30).iterrows():
        date_str = row['ds'].strftime('%Y-%m-%d')
        yhat = round(row['yhat'])
        low = round(row['yhat_lower'])
        high = round(row['yhat_upper'])
        summary_text += f"On {date_str}, expected stock is around {yhat} units (range: {low}–{high}).\n"

    forecast_summaries.append(summary_text + "\n\n")  # 2-line gap between warehouses

# Save summaries to file
with open(forecast_summary_file, "w") as f:
    f.writelines(forecast_summaries)

print(f"✅ Forecast summaries saved in '{forecast_summary_file}'")
