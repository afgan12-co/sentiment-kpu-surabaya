# Sentiment Analysis API Documentation

## 🚀 **Overview**

Aplikasi ini menyediakan REST API untuk sentiment analysis yang dapat diintegrasikan dengan web admin berbasis Tailwind CSS.

## 📡 **API Endpoints**

### **Base URL:** `http://localhost:8000` (development) / `https://your-domain.com` (production)

### **1. Health Check**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": "2025-12-23T10:00:00"
}
```

### **2. Single Text Prediction**
```http
POST /predict
Content-Type: application/json

{
  "text": "Saya sangat senang dengan pelayanan ini",
  "model": "nb"
}
```

**Parameters:**
- `text`: Text to analyze (required)
- `model`: "nb" (Naive Bayes) or "svm" (default: "nb")

**Response:**
```json
{
  "text": "Saya sangat senang dengan pelayanan ini",
  "sentiment": "positif",
  "confidence": 0.9876,
  "model_used": "naive_bayes",
  "timestamp": "2025-12-23T10:00:00"
}
```

### **3. Batch Prediction**
```http
POST /predict-batch
Content-Type: application/json

{
  "texts": [
    "Pelayanan sangat baik",
    "Produk jelek sekali",
    "Biasa saja"
  ],
  "model": "svm"
}
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "Pelayanan sangat baik",
      "sentiment": "positif",
      "confidence": 0.9456,
      "model_used": "svm",
      "timestamp": "2025-12-23T10:00:00"
    }
  ],
  "total_processed": 3,
  "model_used": "svm",
  "timestamp": "2025-12-23T10:00:00"
}
```

### **4. Upload Dataset**
```http
POST /upload-dataset
Content-Type: multipart/form-data

file: dataset.csv
```

### **5. Get Statistics**
```http
GET /stats
```

**Response:**
```json
{
  "total_predictions": 150,
  "model_stats": {
    "naive_bayes": 89,
    "svm": 61,
    "last_prediction": "2025-12-23T10:00:00"
  },
  "timestamp": "2025-12-23T10:00:00"
}
```

### **6. Get Model Information**
```http
GET /models-info
```

## 🛠️ **Setup & Installation**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Train Models (via Streamlit)**
```bash
streamlit run app.py
```
- Upload dataset
- Lakukan preprocessing
- Buat TF-IDF matrix
- Train Naive Bayes & SVM models

### **3. Run API Server**
```bash
# Option 1: Run API only
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# Option 2: Run both Streamlit + API
python run_servers.py
```

## 🌐 **Integration dengan Web Admin (Tailwind)**

### **JavaScript Fetch Examples:**

#### **Single Prediction:**
```javascript
async function predictSentiment(text, model = 'nb') {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: text,
      model: model
    })
  });

  const result = await response.json();
  return result;
}

// Usage
predictSentiment("Saya suka produk ini").then(result => {
  console.log(result.sentiment); // "positif"
  console.log(result.confidence); // 0.9876
});
```

#### **Batch Prediction:**
```javascript
async function predictBatch(texts, model = 'nb') {
  const response = await fetch('http://localhost:8000/predict-batch', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      texts: texts,
      model: model
    })
  });

  const result = await response.json();
  return result;
}

// Usage
const texts = ["Bagus", "Jelek", "Biasa"];
predictBatch(texts, 'svm').then(result => {
  result.predictions.forEach(pred => {
    console.log(`${pred.text}: ${pred.sentiment}`);
  });
});
```

#### **Health Check:**
```javascript
async function checkAPIHealth() {
  const response = await fetch('http://localhost:8000/health');
  const health = await response.json();
  return health.status === 'healthy' && health.models_loaded;
}
```

### **React/Tailwind Integration Example:**

```jsx
import { useState } from 'react';

function SentimentAnalyzer() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeSentiment = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model: 'nb' })
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  return (
    <div className="max-w-md mx-auto bg-white rounded-xl shadow-md p-6">
      <h2 className="text-xl font-bold mb-4">Sentiment Analysis</h2>

      <textarea
        className="w-full p-3 border rounded-lg mb-4"
        placeholder="Masukkan teks..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button
        onClick={analyzeSentiment}
        disabled={loading || !text}
        className="w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600 disabled:opacity-50"
      >
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>

      {result && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <p><strong>Sentiment:</strong>
            <span className={`ml-2 px-2 py-1 rounded text-sm ${
              result.sentiment === 'positif' ? 'bg-green-100 text-green-800' :
              result.sentiment === 'negatif' ? 'bg-red-100 text-red-800' :
              'bg-yellow-100 text-yellow-800'
            }`}>
              {result.sentiment}
            </span>
          </p>
          <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%</p>
        </div>
      )}
    </div>
  );
}
```

## 📊 **API Documentation**

Kunjungi `http://localhost:8000/docs` untuk interactive API documentation (Swagger UI).

## 🔒 **Security Notes**

- Untuk production, implementasikan authentication
- Gunakan HTTPS
- Limit rate requests
- Validate input data
- Monitor API usage

## 🚀 **Deployment**

### **Production Server:**
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### **Docker:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```