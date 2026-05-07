import numpy as np
from src.models.registry import ModelRegistry

registry = ModelRegistry()


class ModelSelector:

    def select_best_for_state(self, state, xgb_mae=None, prophet_results=None, arima_results=None, lstm_mae=None):

        print(f"\n📊 Comparing Models for {state}...\n")

        # -----------------------------
        # SAFE MAE CALCULATION
        # -----------------------------
        prophet_mae = np.mean(list(prophet_results.values())) if prophet_results else float("inf")
        arima_mae = np.mean(list(arima_results.values())) if arima_results else float("inf")

        scores = {
            "XGBoost": xgb_mae if xgb_mae is not None else float("inf"),
            "Prophet": prophet_mae,
            "ARIMA": arima_mae,
            "LSTM": lstm_mae if lstm_mae is not None else float("inf")
        }

        # -----------------------------
        # BEST MODEL SELECTION
        # -----------------------------
        best_model = min(scores, key=scores.get)
        best_mae = scores[best_model]

        # -----------------------------
        # PRINT RESULTS
        # -----------------------------
        print("📊 Model Scores:")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")

        print(f"\n🏆 BEST MODEL for {state}: {best_model}")

        # -----------------------------
        # SAVE TO REGISTRY (IMPORTANT)
        # -----------------------------
        registry.save_best_model(state, best_model, best_mae)

        return {
            "state": state,
            "best_model": best_model,
            "best_mae": best_mae,
            "scores": scores
        }