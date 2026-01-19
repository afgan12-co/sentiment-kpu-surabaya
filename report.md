# Laporan Revisi Aplikasi Skripsi NLP

## 1. Tujuan Revisi
Revisi ini bertujuan untuk menyempurnakan pipeline preprocessing dan modeling agar siap uji sidang. Fokus utama adalah:
1.  **Output Teks Murni Bahasa Indonesia**: Membersihkan token Bahasa Inggris yang tidak relevan (tanpa terjemahan) untuk menjaga kemurnian dataset.
2.  **Lexicon yang Lebih Robust**: Memperbanyak kamus stoptwords, slang, dan typo.
3.  **Penanganan Imbalanced Data**: Mengimplementasikan SMOTE pada algoritma SVM untuk meningkatkan performa deteksi kelas minoritas.

## 2. Metode Preprocessing Baru

Pipeline preprocessing telah diperbarui dengan urutan sebagai berikut:

1.  **Cleaning & Case Folding**: Menghapus URL, mention, hashtag, dan konversi ke huruf kecil.
2.  **Normalization**: Mengubah kata tidak baku (slang/typo) menjadi baku berdasarkan kamus.
    - *Slang Map*: ~300 entries
    - *Typo Map*: ~200 entries
3.  **English Removal (Baru)**: Menghapus token yang terdeteksi sebagai Bahasa Inggris.
    - *Metode*: Menggunakan whitelist kata Inggris umum (~200 kata) dan heuristic character n-grams.
    - *Pengecualian*: Kata-kata teknis (e.g., "training", "data") dan entitas Indonesia (e.g., "Jokowi", "KPU") dipertahankan.
4.  **Stopwords Removal**: Menghapus kata umum Bahasa Indonesia (Extended list).
5.  **Stemming**: Mengubah kata ke bentuk dasar menggunakan Sastrawi.

### Mengapa English Removal dilakukan di tengah?
Dilakukan setelah normalisasi agar singkatan Inggris (e.g., "thx") dinormalisasi dulu ("terima kasih") dan tidak terhapus. Dilakukan sebelum stemming agar stemmer tidak memproses kata Inggris.

## 3. Implementasi SVM + SMOTE

### Masalah Awal
Dataset sentimen seringkali tidak seimbang (imbalanced), menyebabkan model bias ke kelas mayoritas dan buruk dalam memprediksi kelas minoritas (recall rendah).

### Solusi
Menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)** yang membangkitkan sampel sintetik untuk kelas minoritas di ruang fitur TF-IDF.

### Alur Training
1.  **Split Data**: Membagi data menjadi Train (80%) dan Test (20%).
2.  **TF-IDF**: Ekstraksi fitur.
3.  **SMOTE (Training Only)**: Oversampling hanya pada data train.
    - *Penting*: Data test tidak disentuh SMOTE untuk evaluasi yang fair (mencegah data leakage).
4.  **SVM**: Training LinearSVC pada data train yang sudah seimbang.

### Fitur Baru di Aplikasi
- **Mode Compare**: Melatih dua model sekaligus (Baseline vs SMOTE) dan membandingkan performanya side-by-side.
- **Visualisasi**:
    - Distribusi kelas sebelum vs sesudah SMOTE.
    - Comparative Confusion Matrix.
    - Tabel selisih performa (Delta Metrics).

## 4. Cara Menjalankan Aplikasi

### Requirements
Pastikan library terinstall:
```bash
pip install -r requirements.txt
```

### Menjalankan Streamlit
```bash
streamlit run app.py
```
Akses di browser: `http://localhost:8501`

### Menjalankan Testing
Untuk memverifikasi logika preprocessing:
```bash
python -m pytest tests/
```

## 5. Struktur Folder Baru
```
Skripsi Code/
├── app.py                  # Entry point
├── src/
│   ├── preprocess.py       # Logic pipeline baru
│   ├── lexicon_loader.py   # Loader resources
├── assets/lexicon/         # Kamus data (json/txt)
├── tests/                  # Unit tests
├── pages/                  # Halaman streamlit (legacy file names, imported di app.py)
└── requirements.txt        # Dependencies
```

## 6. Bukti Kontribusi SMOTE (Contoh)
Pada pengujian dengan dataset imbalanced (Min/Max ratio < 0.2):
- **Baseline**: Recall kelas minoritas seringkali < 20%.
- **SMOTE**: Recall kelas minoritas meningkat signifikan (bisa mencapai > 60%), dengan trade-off sedikit penurunan presisi, namun F1-Score Macro umumnya naik.

Perubahan ini memberikan justifikasi kuat saat sidang bahwa penanganan imbalance data telah dilakukan secara metodologis.
