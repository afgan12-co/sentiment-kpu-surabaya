"""
REST API untuk Sistem Analisis Sentimen
Menggunakan FastAPI framework

Cara menjalankan:
    uvicorn api:app --reload --port 8000

Dokumentasi:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import joblib
from utils import preprocess_text, lexicon_label
import io
import os

# Inisialisasi FastAPI
app = FastAPI(
    title="Sentiment Analysis API - KPU Surabaya",
    description="API untuk Analisis Sentimen menggunakan Naive Bayes dan SVM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware - untuk akses dari frontend/aplikasi lain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dalam production, ganti dengan domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models saat aplikasi start
def load_models():
    """Load trained models dan vectorizer"""
    models = {
        'nb_model': None,
        'svm_model': None,
        'vectorizer': None
    }
    
    try:
        if os.path.exists('models/naive_bayes_model.pkl'):
            models['nb_model'] = joblib.load('models/naive_bayes_model.pkl')
            print("✅ Naive Bayes model loaded")
    except Exception as e:
        print(f"⚠️ Naive Bayes model not loaded: {e}")
    
    try:
        if os.path.exists('models/svm_model.pkl'):
            models['svm_model'] = joblib.load('models/svm_model.pkl')
            print("✅ SVM model loaded")
    except Exception as e:
        print(f"⚠️ SVM model not loaded: {e}")
    
    try:
        if os.path.exists('models/tfidf_vectorizer.pkl'):
            models['vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
            print("✅ TF-IDF vectorizer loaded")
    except Exception as e:
        print(f"⚠️ TF-IDF vectorizer not loaded: {e}")
    
    return models

# Load models
MODELS = load_models()

# ============= PYDANTIC MODELS (Request/Response Schemas) =============

class TextInput(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Pemerintah gak becus ngurus korupsi!"
            }
        }

class TextListInput(BaseModel):
    texts: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Pemerintah bagus banget!",
                    "KPU korup parah!",
                    "Rapat membahas anggaran"
                ]
            }
        }

class PreprocessResponse(BaseModel):
    original_text: str
    cleaned_text: str

class LexiconResponse(BaseModel):
    original_text: str
    cleaned_text: str
    sentiment: str

class PredictResponse(BaseModel):
    original_text: str
    cleaned_text: str
    predicted_sentiment: str
    model_used: str
    confidence: Optional[float] = None

class BatchPredictResponse(BaseModel):
    total_processed: int
    results: List[PredictResponse]

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict

# ============= API ENDPOINTS =============

@app.get("/", tags=["Info"])
def root():
    """
    Root endpoint - Informasi API
    """
    return {
        "message": "Sentiment Analysis API - KPU Surabaya",
        "version": "1.0.0",
        "author": "Skripsi System",
        "documentation": "/docs",
        "endpoints": {
            "GET /": "API Information",
            "GET /health": "Health check & model status",
            "POST /preprocess": "Text preprocessing",
            "POST /preprocess/batch": "Batch text preprocessing",
            "POST /label/lexicon": "Lexicon-based sentiment labeling",
            "POST /label/lexicon/batch": "Batch lexicon labeling",
            "POST /predict/naive-bayes": "Predict sentiment with Naive Bayes",
            "POST /predict/svm": "Predict sentiment with SVM",
            "POST /predict/batch/naive-bayes": "Batch prediction with Naive Bayes",
            "POST /predict/batch/svm": "Batch prediction with SVM",
            "POST /upload-dataset": "Upload & process CSV dataset"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check():
    """
    Health check endpoint - Cek status API dan model yang ter-load
    """
    return {
        "status": "healthy",
        "models_loaded": {
            "naive_bayes": MODELS['nb_model'] is not None,
            "svm": MODELS['svm_model'] is not None,
            "tfidf_vectorizer": MODELS['vectorizer'] is not None
        }
    }

# ============= PREPROCESSING ENDPOINTS =============

@app.post("/preprocess", response_model=PreprocessResponse, tags=["Preprocessing"])
def preprocess_text_endpoint(input_data: TextInput):
    """
    **Preprocessing teks tunggal**
    
    Tahapan:
    1. Case Folding
    2. Cleaning (URL, mention, hashtag, special chars)
    3. Tokenization
    4. Normalization (slang → baku)
    5. Stopwords Removal
    6. Stemming
    
    **Input:** Text yang akan dipreprocess
    
    **Output:** Original text dan cleaned text
    """
    try:
        cleaned = preprocess_text(input_data.text)
        return {
            "original_text": input_data.text,
            "cleaned_text": cleaned
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

@app.post("/preprocess/batch", tags=["Preprocessing"])
def preprocess_batch_endpoint(input_data: TextListInput):
    """
    **Batch preprocessing untuk multiple teks**
    
    **Input:** List of texts
    
    **Output:** List of preprocessed results
    """
    try:
        results = []
        for text in input_data.texts:
            cleaned = preprocess_text(text)
            results.append({
                "original_text": text,
                "cleaned_text": cleaned
            })
        
        return {
            "total_processed": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch preprocessing error: {str(e)}")

# ============= LEXICON LABELING ENDPOINTS =============

@app.post("/label/lexicon", response_model=LexiconResponse, tags=["Labeling"])
def lexicon_labeling_endpoint(input_data: TextInput):
    """
    **Lexicon-based sentiment labeling**
    
    Metode: Pembobotan kata berdasarkan kamus lexicon positif dan negatif
    
    **Label:**
    - **positif**: Score > 0
    - **negatif**: Score < 0
    - **netral**: Score = 0
    
    **Input:** Text yang akan dilabeli
    
    **Output:** Original text, cleaned text, dan sentiment label
    """
    try:
        # Preprocess
        cleaned = preprocess_text(input_data.text)
        
        # Label with lexicon
        sentiment = lexicon_label(cleaned)
        
        return {
            "original_text": input_data.text,
            "cleaned_text": cleaned,
            "sentiment": sentiment
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lexicon labeling error: {str(e)}")

@app.post("/label/lexicon/batch", tags=["Labeling"])
def lexicon_batch_labeling_endpoint(input_data: TextListInput):
    """
    **Batch lexicon labeling untuk multiple teks**
    
    **Input:** List of texts
    
    **Output:** List of labeling results
    """
    try:
        results = []
        for text in input_data.texts:
            cleaned = preprocess_text(text)
            sentiment = lexicon_label(cleaned)
            results.append({
                "original_text": text,
                "cleaned_text": cleaned,
                "sentiment": sentiment
            })
        
        # Count distribution
        distribution = {}
        for r in results:
            sentiment = r['sentiment']
            distribution[sentiment] = distribution.get(sentiment, 0) + 1
        
        return {
            "total_processed": len(results),
            "distribution": distribution,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch labeling error: {str(e)}")

# ============= PREDICTION ENDPOINTS =============

@app.post("/predict/naive-bayes", response_model=PredictResponse, tags=["Prediction"])
def predict_naive_bayes(input_data: TextInput):
    """
    **Prediksi sentiment menggunakan Naive Bayes**
    
    Model: Multinomial Naive Bayes
    Features: TF-IDF
    
    **Input:** Text untuk diprediksi
    
    **Output:** Predicted sentiment (positif/negatif/netral)
    """
    if MODELS['nb_model'] is None or MODELS['vectorizer'] is None:
        raise HTTPException(
            status_code=503, 
            detail="Naive Bayes model belum tersedia. Silakan train model terlebih dahulu."
        )
    
    try:
        # Preprocess
        cleaned = preprocess_text(input_data.text)
        
        if not cleaned.strip():
            raise HTTPException(status_code=400, detail="Text menjadi kosong setelah preprocessing")
        
        # Transform with TF-IDF
        X = MODELS['vectorizer'].transform([cleaned])
        
        # Predict
        prediction = MODELS['nb_model'].predict(X)[0]
        
        # Get probability (confidence)
        probabilities = MODELS['nb_model'].predict_proba(X)[0]
        confidence = float(max(probabilities))
        
        return {
            "original_text": input_data.text,
            "cleaned_text": cleaned,
            "predicted_sentiment": prediction,
            "model_used": "Naive Bayes",
            "confidence": round(confidence, 4)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/svm", response_model=PredictResponse, tags=["Prediction"])
def predict_svm(input_data: TextInput):
    """
    **Prediksi sentiment menggunakan SVM**
    
    Model: Support Vector Machine (Linear kernel)
    Features: TF-IDF
    
    **Input:** Text untuk diprediksi
    
    **Output:** Predicted sentiment (positif/negatif/netral)
    """
    if MODELS['svm_model'] is None or MODELS['vectorizer'] is None:
        raise HTTPException(
            status_code=503, 
            detail="SVM model belum tersedia. Silakan train model terlebih dahulu."
        )
    
    try:
        # Preprocess
        cleaned = preprocess_text(input_data.text)
        
        if not cleaned.strip():
            raise HTTPException(status_code=400, detail="Text menjadi kosong setelah preprocessing")
        
        # Transform with TF-IDF
        X = MODELS['vectorizer'].transform([cleaned])
        
        # Predict
        prediction = MODELS['svm_model'].predict(X)[0]
        
        # Get decision function (confidence score)
        decision = MODELS['svm_model'].decision_function(X)
        confidence = float(max(abs(decision[0]))) if hasattr(decision, '__iter__') else float(abs(decision))
        
        return {
            "original_text": input_data.text,
            "cleaned_text": cleaned,
            "predicted_sentiment": prediction,
            "model_used": "SVM",
            "confidence": round(confidence, 4)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch/naive-bayes", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch_naive_bayes(input_data: TextListInput):
    """
    **Batch prediction dengan Naive Bayes**
    
    **Input:** List of texts
    
    **Output:** Batch prediction results
    """
    if MODELS['nb_model'] is None or MODELS['vectorizer'] is None:
        raise HTTPException(status_code=503, detail="Naive Bayes model belum tersedia")
    
    try:
        results = []
        for text in input_data.texts:
            cleaned = preprocess_text(text)
            
            if cleaned.strip():
                X = MODELS['vectorizer'].transform([cleaned])
                prediction = MODELS['nb_model'].predict(X)[0]
                probabilities = MODELS['nb_model'].predict_proba(X)[0]
                confidence = float(max(probabilities))
                
                results.append({
                    "original_text": text,
                    "cleaned_text": cleaned,
                    "predicted_sentiment": prediction,
                    "model_used": "Naive Bayes",
                    "confidence": round(confidence, 4)
                })
            else:
                results.append({
                    "original_text": text,
                    "cleaned_text": "",
                    "predicted_sentiment": "netral",
                    "model_used": "Naive Bayes",
                    "confidence": 0.0
                })
        
        return {
            "total_processed": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict/batch/svm", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch_svm(input_data: TextListInput):
    """
    **Batch prediction dengan SVM**
    
    **Input:** List of texts
    
    **Output:** Batch prediction results
    """
    if MODELS['svm_model'] is None or MODELS['vectorizer'] is None:
        raise HTTPException(status_code=503, detail="SVM model belum tersedia")
    
    try:
        results = []
        for text in input_data.texts:
            cleaned = preprocess_text(text)
            
            if cleaned.strip():
                X = MODELS['vectorizer'].transform([cleaned])
                prediction = MODELS['svm_model'].predict(X)[0]
                decision = MODELS['svm_model'].decision_function(X)
                confidence = float(max(abs(decision[0]))) if hasattr(decision, '__iter__') else float(abs(decision))
                
                results.append({
                    "original_text": text,
                    "cleaned_text": cleaned,
                    "predicted_sentiment": prediction,
                    "model_used": "SVM",
                    "confidence": round(confidence, 4)
                })
            else:
                results.append({
                    "original_text": text,
                    "cleaned_text": "",
                    "predicted_sentiment": "netral",
                    "model_used": "SVM",
                    "confidence": 0.0
                })
        
        return {
            "total_processed": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# ============= DATASET UPLOAD ENDPOINT =============

@app.post("/upload-dataset", tags=["Dataset"])
async def upload_dataset(file: UploadFile = File(...)):
    """
    **Upload dan proses dataset CSV**
    
    **Requirements:**
    - File format: CSV
    - Harus memiliki kolom 'text'
    
    **Process:**
    1. Preprocessing semua text
    2. Lexicon-based labeling
    3. Menampilkan distribusi label
    
    **Output:** Dataset summary dan sample
    """
    # Validasi file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File harus berformat CSV")
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validasi kolom
        text_col = None
        for col in ['text', 'Text', 'komentar', 'content']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            raise HTTPException(
                status_code=400, 
                detail="CSV harus memiliki kolom 'text' (atau 'Text', 'komentar', 'content')"
            )
        
        # Preprocess all
        df['cleaned_text'] = df[text_col].apply(preprocess_text)
        
        # Label with lexicon
        df['label'] = df['cleaned_text'].apply(lexicon_label)
        
        # Statistics
        total_rows = len(df)
        empty_count = df['cleaned_text'].apply(lambda x: len(x) == 0).sum()
        distribution = df['label'].value_counts().to_dict()
        
        # Sample data
        sample = df[[text_col, 'cleaned_text', 'label']].head(5).to_dict('records')
        
        return {
            "message": "Dataset berhasil diproses",
            "filename": file.filename,
            "total_rows": total_rows,
            "empty_after_preprocessing": empty_count,
            "label_distribution": distribution,
            "sample_data": sample
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

# ============= ERROR HANDLERS =============

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Endpoint not found",
        "message": f"Endpoint '{request.url.path}' tidak ditemukan",
        "available_endpoints": "/docs"
    }

# Run with: uvicorn api:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
