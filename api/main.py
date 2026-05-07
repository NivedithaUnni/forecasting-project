from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="End-to-End Time Series Forecasting API",
    version="1.0",
    description="Multi-model forecasting system with automatic model selection"
)

app.include_router(router)


@app.get("/")
def home():
    return {
        "message": "Forecasting API is running 🚀",
        "docs": "/docs"
    }