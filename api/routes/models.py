from fastapi import APIRouter
import json

router = APIRouter()

@router.get("/models")
def get_models():

    with open("models/final_model_selection.json", "r") as f:
        data = json.load(f)

    return data