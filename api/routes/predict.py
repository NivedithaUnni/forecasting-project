from fastapi import APIRouter
import pandas as pd

from api.services.predictor import forecast

router = APIRouter()

df = pd.read_csv("data/processed/cleaned_data.csv")


@router.get("/predict")
def predict(state: str):

    result = forecast(state, df)

    return {
        "state": state,
        "forecast_next_8_weeks": result
    }