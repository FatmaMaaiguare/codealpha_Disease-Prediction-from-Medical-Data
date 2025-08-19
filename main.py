from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import os
from typing import Literal

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Diagnosis API", version="1.0")

# Charger le modèle
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Modèle chargé avec succès")
    if hasattr(model, 'n_features_in_'):
        logger.info(f"Nombre de features attendues : {model.n_features_in_}")
except Exception as e:
    logger.error(f"Erreur de chargement : {e}")
    raise RuntimeError("Modèle introuvable ou incompatible !")

class PatientData(BaseModel):
    features: list[float]  # Doit correspondre à model.n_features_in_

class PredictionResult(BaseModel):
    status: Literal["sick", "healthy"]
    confidence: float
    probabilities: dict[str, float]
    risk_level: Literal["low", "medium", "high"]
    message: str | None = None

@app.post("/predict", response_model=PredictionResult)
async def predict(data: PatientData):
    try:
        # Validation des données
        features_array = np.array(data.features).reshape(1, -1)
        
        if hasattr(model, 'n_features_in_') and features_array.shape[1] != model.n_features_in_:
            raise ValueError(
                f"Nombre incorrect de features. Reçu {features_array.shape[1]}, "
                f"attendu {model.n_features_in_}"
            )
            
        # Prédiction
        prediction = int(model.predict(features_array)[0])
        proba = model.predict_proba(features_array)[0]
        
        # Conversion en résultat lisible
        status = "sick" if prediction == 1 else "healthy"
        confidence = max(proba)
        risk_level = (
            "high" if proba[1] > 0.7 else
            "medium" if proba[1] > 0.3 else
            "low"
        )
        
        return {
            "status": status,
            "confidence": confidence,
            "probabilities": {
                "healthy": float(proba[0]),
                "sick": float(proba[1])
            },
            "risk_level": risk_level,
            "message": "Consultez un médecin" if status == "sick" else None
        }
        
    except Exception as e:
        logger.error(f"Erreur de prédiction : {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "expected_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "Non disponible"
            }
        )

@app.get("/", summary="Health Check")
def health_check():
    return {
        "status": "running",
        "model_loaded": True,
        "expected_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else None
    }