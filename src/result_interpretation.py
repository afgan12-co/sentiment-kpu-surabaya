import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_model_metrics(pred_df: pd.DataFrame) -> dict:
    """Hitung metrik evaluasi utama dari dataframe prediksi."""
    y_true = pred_df["true"]
    y_pred = pred_df["pred"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def build_comparison_dataframe(nb_metrics: dict, svm_metrics: dict) -> pd.DataFrame:
    """Buat tabel perbandingan metrik NB vs SVM."""
    return pd.DataFrame(
        {
            "Model": ["Naïve Bayes", "SVM"],
            "Accuracy": [nb_metrics["accuracy"], svm_metrics["accuracy"]],
            "Precision": [nb_metrics["precision"], svm_metrics["precision"]],
            "Recall": [nb_metrics["recall"], svm_metrics["recall"]],
            "F1-Score": [nb_metrics["f1"], svm_metrics["f1"]],
        }
    )


def render_sentiment_meaning_section() -> None:
    """Narasi interpretasi makna kategori sentimen untuk pengguna non-teknis."""
    st.markdown("## 🧭 Makna Kategori Sentimen")
    st.markdown(
        """
Bagian ini membantu pembaca memahami bahwa hasil klasifikasi tidak hanya berupa label model,
melainkan gambaran persepsi publik terhadap kinerja KPU Kota Surabaya pada Pemilu 2024.
"""
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.error(
            """
**Sentimen Negatif**

Mencerminkan kritik, ketidakpuasan, atau masukan evaluatif masyarakat terhadap
kinerja KPU Kota Surabaya selama Pemilu 2024.
"""
        )

    with col2:
        st.warning(
            """
**Sentimen Netral**

Mencerminkan opini yang bersifat informatif atau deskriptif, tanpa kecenderungan
emosi yang kuat terhadap kinerja lembaga.
"""
        )

    with col3:
        st.success(
            """
**Sentimen Positif**

Mencerminkan kepercayaan, apresiasi, dan penilaian baik masyarakat terhadap
kinerja KPU Kota Surabaya.
"""
        )

    st.info(
        """
**Simpulan interpretatif:**
Proporsi sentimen positif dapat dijadikan indikator kepercayaan publik.
Hasil ini dapat dirujuk sebagai dasar untuk mempertahankan sekaligus meningkatkan
kualitas pelayanan, netralitas, dan kinerja KPU Kota Surabaya pada penyelenggaraan
pemilu berikutnya.
"""
    )


def render_model_comparison_interpretation(nb_metrics: dict, svm_metrics: dict) -> None:
    """Narasi perbandingan model dalam bahasa yang mudah dipahami pengguna awam."""
    st.markdown("## 🔎 Interpretasi Perbandingan Naïve Bayes vs SVM")

    best_model = "SVM" if svm_metrics["f1"] >= nb_metrics["f1"] else "Naïve Bayes"
    metric_gap = abs(svm_metrics["f1"] - nb_metrics["f1"])

    st.markdown(
        """
### Alasan umum perbedaan kinerja
- **SVM** cenderung lebih baik dalam memisahkan pola sentimen pada fitur teks berdimensi tinggi.
- **Naïve Bayes** lebih sederhana dan cepat, tetapi menggunakan asumsi probabilitas yang lebih sederhana.
- Performa akhir dipengaruhi oleh karakteristik data teks, distribusi kelas, serta representasi fitur TF-IDF.
"""
    )

    st.success(
        f"""
### Ringkasan kesimpulan perbandingan
Model yang lebih unggul pada evaluasi ini adalah **{best_model}**,
dengan selisih **F1-Score {metric_gap:.4f}**.

Temuan ini relevan untuk evaluasi opini masyarakat terhadap KPU Kota Surabaya,
karena model yang lebih baik akan menghasilkan pembacaan persepsi publik yang
lebih konsisten untuk kebutuhan pelaporan akademik dan pengambilan keputusan.
"""
    )
