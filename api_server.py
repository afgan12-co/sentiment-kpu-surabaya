"""
FastAPI server for Sentiment Analysis API endpoints
Run with: uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess_text
import uvicorn
from datetime import datetime

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis predictions and data synchronization",
    version="1.0.0"
)

# CORS middleware for web admin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your admin domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and vectorizer
try:
    model_nb = joblib.load("models/model_naive_bayes.pkl")
    model_svm = joblib.load("models/model_svm.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    models_loaded = True
except:
    models_loaded = False
    print("Warning: Models not found. Please train models first.")

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    text: str
    model: str = "nb"  # "nb" or "svm"

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    model: str = "nb"

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    model_used: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    model_used: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    timestamp: str

class StatsResponse(BaseModel):
    total_predictions: int
    model_stats: Dict[str, Any]
    timestamp: str

# Global stats
prediction_stats = {
    "total_predictions": 0,
    "nb_predictions": 0,
    "svm_predictions": 0,
    "last_prediction": None
}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if models_loaded else "models_not_loaded",
        models_loaded=models_loaded,
        timestamp=datetime.now().isoformat()
    )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API usage statistics"""
    return StatsResponse(
        total_predictions=prediction_stats["total_predictions"],
        model_stats={
            "naive_bayes": prediction_stats["nb_predictions"],
            "svm": prediction_stats["svm_predictions"],
            "last_prediction": prediction_stats["last_prediction"]
        },
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """Single text prediction endpoint"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Please train models first.")

    try:
        # Preprocess text
        cleaned_text = preprocess_text(request.text)

        # Vectorize
        text_vectorized = vectorizer.transform([cleaned_text])

        # Select model
        if request.model.lower() == "svm":
            model = model_svm
            model_name = "svm"
            prediction_stats["svm_predictions"] += 1
        else:
            model = model_nb
            model_name = "naive_bayes"
            prediction_stats["nb_predictions"] += 1

        # Predict
        prediction = model.predict(text_vectorized)[0]

        # Get confidence (probability for NB, decision function for SVM)
        if hasattr(model, 'predict_proba'):
            confidence = float(max(model.predict_proba(text_vectorized)[0]))
        else:
            confidence = float(abs(model.decision_function(text_vectorized)[0]))

        prediction_stats["total_predictions"] += 1
        prediction_stats["last_prediction"] = datetime.now().isoformat()

        return PredictionResponse(
            text=request.text,
            sentiment=prediction,
            confidence=round(confidence, 4),
            model_used=model_name,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Please train models first.")

    try:
        predictions = []

        # Select model
        if request.model.lower() == "svm":
            model = model_svm
            model_name = "svm"
        else:
            model = model_nb
            model_name = "naive_bayes"

        # Process each text
        for text in request.texts:
            cleaned_text = preprocess_text(text)
            text_vectorized = vectorizer.transform([cleaned_text])
            prediction = model.predict(text_vectorized)[0]

            # Get confidence
            if hasattr(model, 'predict_proba'):
                confidence = float(max(model.predict_proba(text_vectorized)[0]))
            else:
                confidence = float(abs(model.decision_function(text_vectorized)[0]))

            predictions.append(PredictionResponse(
                text=text,
                sentiment=prediction,
                confidence=round(confidence, 4),
                model_used=model_name,
                timestamp=datetime.now().isoformat()
            ))

        prediction_stats["total_predictions"] += len(request.texts)
        if model_name == "svm":
            prediction_stats["svm_predictions"] += len(request.texts)
        else:
            prediction_stats["nb_predictions"] += len(request.texts)
        prediction_stats["last_prediction"] = datetime.now().isoformat()

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(request.texts),
            model_used=model_name,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload dataset for processing"""
    try:
        # Save uploaded file
        os.makedirs("data/uploads", exist_ok=True)
        file_path = f"data/uploads/{file.filename}"

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Basic validation
        df = pd.read_csv(file_path)
        return {
            "message": "Dataset uploaded successfully",
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@app.get("/models-info")
async def get_models_info():
    """Get information about loaded models"""
    if not models_loaded:
        return {"error": "Models not loaded"}

    try:
        nb_info = {
            "type": "MultinomialNB",
            "classes": list(model_nb.classes_),
            "features": model_nb.n_features_in_
        }

        svm_info = {
            "type": "LinearSVC",
            "classes": list(model_svm.classes_),
            "features": model_svm.n_features_in_
        }

        return {
            "naive_bayes": nb_info,
            "svm": svm_info,
            "vectorizer": {
                "max_features": vectorizer.max_features,
                "vocabulary_size": len(vectorizer.vocabulary_)
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)