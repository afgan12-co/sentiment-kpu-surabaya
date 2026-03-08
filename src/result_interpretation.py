"""
Modul interpretasi hasil analisis sentimen untuk website TA.

Berisi fungsi-fungsi helper untuk menampilkan narasi interpretatif,
perbandingan model, dan kesimpulan akademik yang sesuai dengan konteks
penelitian analisis sentimen terhadap KPU Kota Surabaya Pemilu 2024.
"""

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
    """
    Narasi interpretasi makna kategori sentimen untuk pengguna non-teknis.
    Menampilkan penjelasan akademik yang sesuai untuk sidang dan buku TA.
    """
    st.markdown("## 🧭 Makna dan Interpretasi Kategori Sentimen")
    st.markdown(
        """
Bagian ini membantu pembaca — termasuk pihak KPU Kota Surabaya, dosen penguji, dan peneliti berikutnya —
untuk memahami bahwa hasil klasifikasi sentimen bukan sekadar output teknis model,
melainkan **gambaran nyata persepsi publik** terhadap kinerja KPU Kota Surabaya
dalam penyelenggaraan Pemilu 2024.
"""
    )

    # Sentiment category cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
<div style="background-color:#fdecea;border-left:5px solid #e53935;border-radius:8px;padding:16px;height:200px;">
<h4 style="color:#b71c1c;">🔴 Sentimen Negatif</h4>
<p style="font-size:0.92rem;color:#333;">
Mencerminkan <strong>kritik, ketidakpuasan, atau masukan evaluatif</strong>
masyarakat terhadap kinerja KPU Kota Surabaya selama Pemilu 2024.
Opini ini mengandung ekspresi emosi yang cenderung negatif, seperti kekecewaan
terhadap proses administrasi, transparansi, atau teknis penyelenggaraan.
</p>
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
<div style="background-color:#fffde7;border-left:5px solid #f9a825;border-radius:8px;padding:16px;height:200px;">
<h4 style="color:#e65100;">🟡 Sentimen Netral</h4>
<p style="font-size:0.92rem;color:#333;">
Mencerminkan <strong>opini yang bersifat informatif, deskriptif,</strong>
atau pernyataan faktual yang tidak menunjukkan kecenderungan emosi yang kuat,
baik positif maupun negatif. Komentar netral umumnya bersifat observatif
terhadap proses pemilu tanpa ekspresi penilaian yang tajam.
</p>
</div>
""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
<div style="background-color:#e8f5e9;border-left:5px solid #43a047;border-radius:8px;padding:16px;height:200px;">
<h4 style="color:#1b5e20;">🟢 Sentimen Positif</h4>
<p style="font-size:0.92rem;color:#333;">
Mencerminkan <strong>kepercayaan, apresiasi, dan penilaian baik</strong>
masyarakat terhadap kinerja KPU Kota Surabaya. Opini positif menunjukkan
kepuasan terhadap proses pemilu, netralitas lembaga, pelayanan publik,
serta pengelolaan suara yang dianggap transparan dan profesional.
</p>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Concluding interpretation box
    st.markdown(
        """
<div style="background-color:#e3f2fd;border-left:6px solid #1565c0;border-radius:8px;padding:20px;margin-top:8px;">
<h4 style="color:#0d47a1;">📌 Simpulan Interpretatif Hasil Klasifikasi</h4>
<p style="font-size:0.95rem;color:#212121;line-height:1.7;">
Proporsi <strong>sentimen positif</strong> yang teridentifikasi pada data komentar publik di media sosial
dapat dijadikan sebagai <strong>indikator tingkat kepercayaan masyarakat</strong> terhadap KPU Kota Surabaya
dalam penyelenggaraan Pemilu 2024.
</p>
<p style="font-size:0.95rem;color:#212121;line-height:1.7;">
Sementara itu, <strong>sentimen negatif</strong> tidak semata-mata bersifat merusak citra lembaga,
melainkan merupakan <strong>masukan evaluatif yang konstruktif</strong> yang dapat digunakan
sebagai dasar perbaikan kualitas layanan, transparansi, dan penyelenggaraan pemilu berikutnya.
</p>
<p style="font-size:0.95rem;color:#212121;line-height:1.7;">
Hasil analisis sentimen ini relevan untuk dirujuk oleh KPU Kota Surabaya sebagai bahan refleksi kinerja,
bagi dosen penguji sebagai validasi kontribusi akademik penelitian,
serta bagi peneliti berikutnya sebagai baseline untuk perbandingan studi lanjutan.
</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)


def render_model_comparison_interpretation(nb_metrics: dict, svm_metrics: dict) -> None:
    """
    Narasi perbandingan model dalam bahasa yang mudah dipahami pengguna awam.
    Menampilkan penjelasan per-metrik dan simpulan akademik akhir.
    """
    st.markdown("## 🔎 Interpretasi Perbandingan Naïve Bayes vs SVM")
    st.markdown(
        """
Bagian ini menjelaskan perbedaan kinerja antara dua metode klasifikasi yang digunakan
agar hasil angka metrik evaluasi dapat dipahami secara kualitatif oleh semua kalangan pembaca.
"""
    )

    best_model = "SVM" if svm_metrics["f1"] >= nb_metrics["f1"] else "Naïve Bayes"
    other_model = "Naïve Bayes" if best_model == "SVM" else "SVM"
    metric_gap = abs(svm_metrics["f1"] - nb_metrics["f1"])

    # Per-metric explanation in 4 columns
    st.markdown("### 📐 Penjelasan Metrik Evaluasi")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
<div style="background:#f5f5f5;border-radius:8px;padding:12px;text-align:center;">
<h5>🎯 Accuracy</h5>
<p style="font-size:0.85rem;color:#555;">Persentase prediksi yang benar dari keseluruhan data uji. Semakin tinggi, semakin banyak data yang diklasifikasikan dengan tepat.</p>
</div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
<div style="background:#f5f5f5;border-radius:8px;padding:12px;text-align:center;">
<h5>🔍 Precision</h5>
<p style="font-size:0.85rem;color:#555;">Dari semua prediksi positif model, seberapa banyak yang benar-benar positif. Tinggi berarti model jarang salah mendeteksi sentimen.</p>
</div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
<div style="background:#f5f5f5;border-radius:8px;padding:12px;text-align:center;">
<h5>📡 Recall</h5>
<p style="font-size:0.85rem;color:#555;">Dari semua data yang sebenarnya positif, seberapa banyak berhasil ditemukan. Tinggi berarti model tidak banyak melewatkan data penting.</p>
</div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """
<div style="background:#f5f5f5;border-radius:8px;padding:12px;text-align:center;">
<h5>⚖️ F1-Score</h5>
<p style="font-size:0.85rem;color:#555;">Rata-rata harmonis Precision dan Recall. Metrik terpenting saat distribusi kelas tidak seimbang, seperti pada data sentimen publik.</p>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Qualitative comparison
    st.markdown("### 🤔 Mengapa Ada Perbedaan Kinerja?")
    st.markdown(
        f"""
- **Support Vector Machine (SVM)** bekerja dengan mencari batas pemisah optimal
  (*hyperplane*) di antara kelas-kelas sentimen pada ruang fitur berdimensi tinggi yang dihasilkan oleh TF-IDF.
  Pendekatan ini sangat efektif untuk data teks karena mampu menangani hubungan non-linear antar fitur.

- **Naïve Bayes** menggunakan probabilitas kemunculan kata per kategori sentimen.
  Metode ini lebih sederhana dan cepat, namun mengasumsikan bahwa setiap kata bersifat saling bebas
  (*independent*) — asumsi yang sering tidak terpenuhi pada teks nyata di media sosial.

- Perbedaan performa kedua model dipengaruhi oleh:
  - **Distribusi kelas sentimen** pada data — jika satu kelas lebih dominan, F1 makro bisa lebih rendah.
  - **Kualitas fitur TF-IDF** — semakin baik preprocessing, semakin baik performa kedua model.
  - **Karakteristik bahasa** — komentar di media sosial seringkali mengandung bahasa informal,
    singkatan, dan slang yang membutuhkan normalisasi yang baik.
"""
    )

    # Winner summary box
    gap_label = "relatif kecil" if metric_gap < 0.05 else "cukup signifikan"
    winner_color = "#1b5e20" if best_model == "SVM" else "#1a237e"
    winner_bg = "#e8f5e9" if best_model == "SVM" else "#e8eaf6"
    winner_border = "#43a047" if best_model == "SVM" else "#3949ab"

    st.markdown(
        f"""
<div style="background-color:{winner_bg};border-left:6px solid {winner_border};border-radius:8px;padding:20px;margin-top:8px;">
<h4 style="color:{winner_color};">🏆 Ringkasan Kesimpulan Perbandingan Model</h4>
<p style="font-size:0.97rem;color:#212121;line-height:1.7;">
Berdasarkan seluruh metrik evaluasi yang digunakan — Accuracy, Precision, Recall, dan F1-Score —
model <strong>{best_model}</strong> menunjukkan kinerja yang lebih unggul dibandingkan <strong>{other_model}</strong>
pada dataset komentar publik tentang KPU Kota Surabaya dalam Pemilu 2024.
</p>
<p style="font-size:0.97rem;color:#212121;line-height:1.7;">
Selisih F1-Score antara kedua model adalah <strong>{metric_gap:.4f}</strong> ({gap_label}).
Hal ini menunjukkan bahwa <strong>{best_model}</strong> lebih konsisten dalam mengklasifikasikan
ketiga kategori sentimen (positif, netral, negatif) secara merata.
</p>
<p style="font-size:0.97rem;color:#212121;line-height:1.7;">
Temuan ini relevan dan dapat dipertanggungjawabkan secara akademik, karena model dengan F1
lebih tinggi akan menghasilkan pembacaan persepsi publik yang lebih akurat
untuk keperluan pelaporan, evaluasi kinerja lembaga, maupun penelitian lanjutan.
</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
