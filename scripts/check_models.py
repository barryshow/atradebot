import joblib, os

mdir = "/root/atradebot/models"
for sym in ["btcusdt", "ethusdt", "solusdt"]:
    path = os.path.join(mdir, f"{sym}_fast_entry.pkl")
    if os.path.exists(path):
        b = joblib.load(path)
        model_type = type(b.get("model", "NONE")).__name__
        scaler_type = type(b.get("scaler", "NONE")).__name__
        print(f"{sym}: OK model={model_type} scaler={scaler_type}")
    else:
        print(f"{sym}: NOT FOUND at {path}")