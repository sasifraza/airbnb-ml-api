from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schema import AirbnbInput
from app.preprocess import FEATURE_COLUMNS

app = FastAPI(title="Airbnb Price Prediction API")

model = None


@app.on_event("startup")
def load_model():
    global model

    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run training first.")

    model = joblib.load(model_path)


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(payload: AirbnbInput):
    input_df = pd.DataFrame([payload.model_dump()])
    input_df = input_df[FEATURE_COLUMNS]

    prediction = model.predict(input_df)[0]

    return {"predicted_price": float(prediction)}