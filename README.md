# Sentiment Analysis Application with API Integration

Aplikasi analisis sentimen lengkap dengan Streamlit web app dan REST API untuk integrasi dengan web admin berbasis Tailwind CSS.

## 🚀 **Fitur Utama**

- ✅ **Streamlit Web App**: Interface lengkap untuk preprocessing, training, dan evaluasi
- ✅ **REST API**: Endpoints untuk prediction real-time dan batch
- ✅ **Machine Learning**: Naive Bayes + SVM dengan SMOTE
- ✅ **Text Processing**: Pipeline preprocessing lengkap
- ✅ **Visualization**: Charts dan wordclouds interaktif
- ✅ **Authentication**: Login system
- ✅ **Web Admin Demo**: Contoh integrasi dengan Tailwind CSS

## 📁 **Struktur Project**

```
sentiment-kpu-surabaya/
├── app.py                    # Main Streamlit application
├── api_server.py           # FastAPI server untuk REST endpoints
├── run_servers.py          # Script untuk menjalankan kedua server
├── web_admin_demo.html     # Demo web admin dengan Tailwind CSS
├── API_DOCUMENTATION.md    # Dokumentasi lengkap API
├── requirements.txt        # Dependencies Python
├── utils.py               # Utility functions & preprocessing
├── users.json            # User authentication data
├── page_*.py             # Individual Streamlit pages
├── data/
│   ├── clean/           # Preprocessed datasets
│   └── labeled/         # Labeled datasets
├── models/              # Trained ML models & vectorizers
└── results/             # Evaluation results
```

## 🛠️ **Installation & Setup**

### **1. Clone Repository**
```bash
git clone https://github.com/afgan12-co/sentiment-kpu-surabaya
cd sentiment-kpu-surabaya
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Jalankan Aplikasi**

#### **Option A: Jalankan Semua (Streamlit + API)**
```bash
python run_servers.py
```
- Streamlit App: http://localhost:8501
- FastAPI Server: http://localhost:8000
- API Docs: http://localhost:8000/docs

#### **Option B: Jalankan Terpisah**

**Terminal 1 - Streamlit:**
```bash
streamlit run app.py
```

**Terminal 2 - API Server:**
```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### **4. Demo Web Admin**
Buka file `web_admin_demo.html` di browser untuk melihat demo integrasi dengan Tailwind CSS.

## 📊 **Workflow Lengkap**

1. **Login** → Akses aplikasi
2. **Text Processing** → Upload CSV → Preprocessing
3. **Lexicon Labeling** → Auto-label sentimen
4. **TF-IDF Vectorization** → Feature extraction
5. **Model Training** → Train NB & SVM
6. **Evaluation** → Bandingkan performa model
7. **Visualization** → Lihat hasil & insights
8. **API Integration** → Gunakan endpoints untuk web admin

## 🔌 **API Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Single text prediction |
| POST | `/predict-batch` | Batch prediction |
| POST | `/upload-dataset` | Upload dataset |
| GET | `/stats` | API usage statistics |
| GET | `/models-info` | Model information |

### **Example API Usage:**

```javascript
// Single prediction
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: "Saya sangat senang dengan pelayanan ini",
    model: "nb"
  })
});
const result = await response.json();
// result: { sentiment: "positif", confidence: 0.9876, ... }
```

## 🎨 **Web Admin Integration**

File `web_admin_demo.html` menunjukkan contoh integrasi lengkap dengan:
- ✅ **Tailwind CSS** styling
- ✅ **Real-time API calls** dengan JavaScript
- ✅ **Interactive UI** untuk single & batch prediction
- ✅ **Statistics dashboard**
- ✅ **Error handling**

## 📈 **Model Performance**

Aplikasi menggunakan:
- **Naive Bayes**: MultinomialNB untuk text classification
- **SVM**: LinearSVC dengan SMOTE untuk handle imbalance
- **TF-IDF**: Feature extraction dengan max_features=1000
- **Preprocessing**: Case folding, cleaning, stemming, stopwords

## 🚀 **Deployment**

## 🧪 Testing Live Server Sebelum Deploy

Gunakan alur ini agar sistem bisa diuji seperti kondisi live sebelum deployment final.

### 1) Jalankan server aplikasi

```bash
python run_servers.py
```

Output default:
- Streamlit: `http://localhost:8503`
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

### 2) Jalankan smoke test otomatis (langsung cek live sekarang)

Di terminal terpisah:

```bash
python scripts/live_smoke_test.py
```

Jika berhasil, akan muncul status:
- ✅ Streamlit OK
- ✅ API health OK

Jika Anda hanya ingin cek UI Streamlit saja (tanpa API), gunakan:

```bash
python scripts/live_smoke_test.py --streamlit-only
```

> Catatan: jika muncul error `No module named uvicorn`, install dependensi API dulu:
> `pip install uvicorn fastapi`

Fokus pengujian ini adalah validasi langsung di mesin saat ini sebelum deploy.

### **Streamlit Cloud:**
1. Connect GitHub repository
2. Deploy otomatis dari branch `main`

### **API Server (Production):**
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### **Docker:**
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## 📚 **Dokumentasi Lengkap**

- **API Documentation**: `API_DOCUMENTATION.md`
- **Interactive API Docs**: http://localhost:8000/docs (Swagger UI)
- **Web Admin Demo**: `web_admin_demo.html`

## 🔧 **Troubleshooting**

### **Models not loaded error:**
1. Jalankan Streamlit app terlebih dahulu
2. Lakukan training model melalui interface
3. Restart API server

### **CORS error:**
Tambahkan domain web admin ke `CORS_ORIGINS` di `api_server.py`

### **Port conflicts:**
Ubah port di `run_servers.py` atau `api_server.py`

## 🤝 **Integration dengan Web Admin**

Untuk integrasi dengan web admin Tailwind:

1. **Copy JavaScript functions** dari `web_admin_demo.html`
2. **Sesuaikan API_BASE URL** dengan domain production
3. **Implement authentication** jika diperlukan
4. **Add error handling** dan loading states
5. **Style dengan Tailwind CSS** sesuai design system

## 📞 **Support**

Jika ada pertanyaan atau masalah:
1. Cek `API_DOCUMENTATION.md` untuk detail endpoints
2. Lihat `web_admin_demo.html` untuk contoh integrasi
3. Pastikan models sudah ditraining sebelum menggunakan API

---

**🎉 Aplikasi siap digunakan untuk analisis sentimen dan integrasi dengan web admin!**
