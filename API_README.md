# 🚀 Quick Start - Sentiment Analysis API

## Cara Menjalankan API

### 1️⃣ Install Dependencies Baru

```bash
pip install fastapi uvicorn[standard] python-multipart
```

Atau install semua dependencies:

```bash
pip install -r requirements.txt
```

### 2️⃣ Jalankan API Server

```bash
uvicorn api:app --reload --port 8000
```

### 3️⃣ Akses API

✅ **API Server**: http://localhost:8000

✅ **Dokumentasi Interaktif (Swagger UI)**: http://localhost:8000/docs

✅ **Alternative Docs (ReDoc)**: http://localhost:8000/redoc

---

## 📋 Daftar API Endpoints

### 🔹 Info & Health
1. `GET /` - API Information
2. `GET /health` - Health Check

### 🔹 Preprocessing
3. `POST /preprocess` - Preprocess single text
4. `POST /preprocess/batch` - Preprocess multiple texts

### 🔹 Lexicon Labeling
5. `POST /label/lexicon` - Label single text with lexicon
6. `POST /label/lexicon/batch` - Label multiple texts

### 🔹 ML Prediction
7. `POST /predict/naive-bayes` - Predict with Naive Bayes
8. `POST /predict/svm` - Predict with SVM
9. `POST /predict/batch/naive-bayes` - Batch prediction (NB)
10. `POST /predict/batch/svm` - Batch prediction (SVM)

### 🔹 Dataset
11. `POST /upload-dataset` - Upload & process CSV

---

## 🧪 Test Cepat

### Test dengan Browser
Buka: http://localhost:8000/docs

### Test dengan cURL

```bash
# Health check
curl http://localhost:8000/health

# Preprocess
curl -X POST "http://localhost:8000/preprocess" \
  -H "Content-Type: application/json" \
  -d '{"text": "Pemerintah gak becus!"}'

# Lexicon labeling
curl -X POST "http://localhost:8000/label/lexicon" \
  -H "Content-Type: application/json" \
  -d '{"text": "Pembangunan bagus sekali!"}'
```

### Test dengan Python

```python
import requests

# Test preprocessing
response = requests.post(
    "http://localhost:8000/preprocess",
    json={"text": "Pemerintah gak becus!"}
)
print(response.json())

# Test lexicon labeling
response = requests.post(
    "http://localhost:8000/label/lexicon",
    json={"text": "Pembangunan bagus!"}
)
print(response.json())
```

---

## ⚠️ Catatan Penting

> **Untuk ML Prediction** (`/predict/*` endpoints):
> 
> Model harus sudah di-training terlebih dahulu melalui aplikasi Streamlit:
> 1. Jalankan Streamlit: `streamlit run app.py`
> 2. Login dan train model di menu **Klasifikasi Naive Bayes** atau **Klasifikasi SVM**
> 3. Model akan otomatis tersimpan di folder `models/`
> 4. API akan otomatis load model saat restart

---

## 📚 Dokumentasi Lengkap

Lihat file **API_ENDPOINTS.md** untuk dokumentasi lengkap semua endpoint dengan contoh request/response.

---

## 🎯 Use Cases

- **Mobile App Integration**: Sentiment analysis untuk app Android/iOS
- **Web Dashboard**: Real-time sentiment monitoring
- **Batch Processing**: Analisis ribuan komentar sekaligus
- **Third-party Integration**: Integrasi dengan sistem lain via REST API
