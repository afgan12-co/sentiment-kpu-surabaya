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


def render_model_comparison_interpretation(nb_metrics: dict, svm_metrics: dict) -> None:
    """
    Narasi perbandingan model dalam bahasa yang mudah dipahami pengguna awam.
    Hanya ditampilkan pada halaman Evaluasi Model.
    """
    st.markdown("## 🔎 Interpretasi Perbandingan Naïve Bayes vs SVM")
    st.markdown(
        "Bagian ini menjelaskan perbedaan kinerja antara dua metode klasifikasi "
        "agar angka metrik evaluasi dapat dipahami secara kualitatif oleh semua kalangan pembaca."
    )

    best_model = "SVM" if svm_metrics["f1"] >= nb_metrics["f1"] else "Naïve Bayes"
    other_model = "Naïve Bayes" if best_model == "SVM" else "SVM"
    metric_gap = abs(svm_metrics["f1"] - nb_metrics["f1"])

    st.markdown("### 📐 Penjelasan Metrik Evaluasi")
    c1, c2, c3, c4 = st.columns(4)
    for col, title, desc in [
        (c1, "🎯 Accuracy",  "Persentase prediksi yang benar dari keseluruhan data uji."),
        (c2, "🔍 Precision", "Dari semua prediksi suatu kelas, seberapa banyak yang benar."),
        (c3, "📡 Recall",    "Dari semua data suatu kelas, seberapa banyak yang berhasil ditemukan."),
        (c4, "⚖️ F1-Score",  "Rata-rata harmonis Precision dan Recall — metrik terpenting untuk data tidak seimbang."),
    ]:
        col.markdown(
            f"<div style='background:#f5f5f5;border-radius:8px;padding:12px;text-align:center;'>"
            f"<h5>{title}</h5><p style='font-size:0.83rem;color:#555;'>{desc}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🤔 Mengapa Ada Perbedaan Kinerja?")
    st.markdown(
        f"""
- **SVM** mencari batas pemisah optimal (*hyperplane*) di ruang fitur TF-IDF berdimensi tinggi 
  — sangat efektif untuk teks yang bersifat *sparse*.
- **Naïve Bayes** menggunakan probabilitas kemunculan kata, lebih sederhana namun mengasumsikan 
  setiap kata bersifat saling bebas (*independent*) — asumsi yang sering tidak terpenuhi pada teks nyata.
- Perbedaan performa juga dipengaruhi distribusi kelas sentimen, kualitas preprocessing, 
  dan representasi fitur TF-IDF yang digunakan.
"""
    )

    gap_label = "relatif kecil (kedua model berkinerja setara)" if metric_gap < 0.05 else "cukup signifikan"
    winner_color  = "#1b5e20" if best_model == "SVM" else "#1a237e"
    winner_bg     = "#e8f5e9" if best_model == "SVM" else "#e8eaf6"
    winner_border = "#43a047" if best_model == "SVM" else "#3949ab"

    st.markdown(
        f"""
<div style="background:{winner_bg};border-left:6px solid {winner_border};border-radius:8px;padding:20px;margin-top:8px;">
<h4 style="color:{winner_color};">🏆 Kesimpulan Perbandingan Model</h4>
<p style="font-size:0.97rem;color:#212121;line-height:1.7;">
Berdasarkan seluruh metrik evaluasi, model <strong>{best_model}</strong> menunjukkan kinerja
yang lebih unggul dibandingkan <strong>{other_model}</strong> dengan selisih F1-Score
sebesar <strong>{metric_gap:.4f}</strong> ({gap_label}).
</p>
<p style="font-size:0.95rem;color:#212121;line-height:1.7;">
Model <strong>{best_model}</strong> lebih konsisten dalam mengklasifikasikan sentimen
publik secara merata di ketiga kategori, sehingga lebih andal digunakan sebagai dasar
pembacaan persepsi masyarakat terhadap kinerja KPU Kota Surabaya.
</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)


def render_final_conclusion(best_pred_df: pd.DataFrame, best_model_name: str) -> None:
    """
    Kesimpulan akhir dinamis berbasis hasil prediksi model terbaik.
    Menampilkan distribusi nyata sentimen dan rekomendasi actionable untuk KPU.
    Hanya dipanggil satu kali di halaman Visualisasi Hasil (akhir alur).
    """
    st.markdown("---")
    st.markdown("## 🏁 Kesimpulan Akhir Hasil Analisis Sentimen")
    st.markdown(
        f"Berikut adalah hasil final analisis sentimen berdasarkan prediksi model "
        f"**{best_model_name}** (model terbaik) terhadap dataset komentar publik "
        "mengenai KPU Kota Surabaya pada Pemilu 2024."
    )

    # Hitung distribusi sentimen nyata
    pred_counts = best_pred_df["pred"].value_counts()
    total = pred_counts.sum()

    label_map = {
        "positif": "Positif", "negatif": "Negatif", "netral": "Netral",
        "1": "Positif", "-1": "Negatif", "0": "Netral",
        1: "Positif", -1: "Negatif", 0: "Netral",
    }

    counts = {}
    for key, label in label_map.items():
        if key in pred_counts.index:
            counts[label] = pred_counts[key]

    positif_n  = counts.get("Positif", 0)
    netral_n   = counts.get("Netral", 0)
    negatif_n  = counts.get("Negatif", 0)
    positif_pct = (positif_n / total * 100) if total > 0 else 0
    netral_pct  = (netral_n  / total * 100) if total > 0 else 0
    negatif_pct = (negatif_n / total * 100) if total > 0 else 0

    # Distribusi nyata sebagai metric cards
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Positif",  f"{positif_n:,} komentar", f"{positif_pct:.1f}% dari total")
    c2.metric("🟡 Netral",   f"{netral_n:,} komentar",  f"{netral_pct:.1f}% dari total")
    c3.metric("🔴 Negatif",  f"{negatif_n:,} komentar", f"{negatif_pct:.1f}% dari total")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tentukan sentimen dominan
    dominant_label = max(counts, key=lambda k: counts.get(k, 0)) if counts else "Netral"

    # Narasi per kategori
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""<div style="background:#fdecea;border-left:5px solid #e53935;border-radius:8px;padding:14px;">
<h4 style="color:#b71c1c;">🔴 Sentimen Negatif</h4>
<p style="font-size:0.88rem;color:#333;line-height:1.6;">
Mencerminkan <strong>kritik dan ketidakpuasan</strong> masyarakat terhadap aspek tertentu
dalam penyelenggaraan Pemilu 2024 oleh KPU Kota Surabaya — seperti proses administrasi,
transparansi, atau teknis pelaksanaan.
<br><br>Komentar negatif harus dibaca sebagai <strong>masukan evaluatif konstruktif</strong>
yang dapat dijadikan prioritas perbaikan layanan.
</p></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div style="background:#fffde7;border-left:5px solid #f9a825;border-radius:8px;padding:14px;">
<h4 style="color:#e65100;">🟡 Sentimen Netral</h4>
<p style="font-size:0.88rem;color:#333;line-height:1.6;">
Mencerminkan opini yang bersifat <strong>informatif dan deskriptif</strong>,
tanpa ekspresi emosi yang kuat. Umumnya berupa laporan situasi, pertanyaan,
atau observasi tentang proses pemilu.
<br><br>Sentimen netral menunjukkan bahwa sebagian masyarakat
<strong>belum membentuk sikap yang tegas</strong> — peluang untuk meningkatkan komunikasi publik.
</p></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div style="background:#e8f5e9;border-left:5px solid #43a047;border-radius:8px;padding:14px;">
<h4 style="color:#1b5e20;">🟢 Sentimen Positif</h4>
<p style="font-size:0.88rem;color:#333;line-height:1.6;">
Mencerminkan <strong>kepercayaan, apresiasi, dan penilaian baik</strong>
masyarakat terhadap kinerja KPU Kota Surabaya — mencakup kepuasan atas
proses pemilu, netralitas lembaga, dan kualitas pelayanan.
<br><br>Sentimen positif merupakan <strong>indikator kepercayaan publik</strong>
yang harus dipertahankan dan ditingkatkan.
</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Kotak rekomendasi utama untuk KPU — berbasis data nyata
    if dominant_label == "Positif":
        box_color = "#e8f5e9"
        border_color = "#43a047"
        title_color = "#1b5e20"
        icon = "✅"
        headline = "Mayoritas Opini Publik Bersifat Positif"
        body = (
            f"Sebanyak <strong>{positif_n:,} komentar ({positif_pct:.1f}%)</strong> dari total "
            f"<strong>{total:,} data</strong> diklasifikasikan sebagai sentimen <strong>positif</strong>. "
            "Ini menunjukkan bahwa masyarakat secara umum <strong>memberikan kepercayaan dan apresiasi</strong> "
            "terhadap kinerja KPU Kota Surabaya dalam penyelenggaraan Pemilu 2024."
        )
        rekomendasi = (
            "📌 <strong>Rekomendasi untuk KPU Kota Surabaya:</strong> Pertahankan aspek-aspek yang sudah "
            "mendapat respons positif dari masyarakat — seperti transparansi, netralitas, dan kualitas pelayanan. "
            "Tingkatkan komunikasi publik untuk mengubah sentimen netral menjadi positif, "
            "dan jadikan komentar negatif sebagai bahan evaluasi perbaikan layanan berikutnya."
        )
    elif dominant_label == "Negatif":
        box_color = "#fdecea"
        border_color = "#e53935"
        title_color = "#b71c1c"
        icon = "⚠️"
        headline = "Mayoritas Opini Publik Bersifat Negatif — Perlu Perhatian Khusus"
        body = (
            f"Sebanyak <strong>{negatif_n:,} komentar ({negatif_pct:.1f}%)</strong> dari total "
            f"<strong>{total:,} data</strong> diklasifikasikan sebagai sentimen <strong>negatif</strong>. "
            "Hasil ini mengindikasikan adanya <strong>ketidakpuasan atau kritik signifikan</strong> "
            "dari masyarakat terhadap aspek tertentu dalam penyelenggaraan Pemilu 2024."
        )
        rekomendasi = (
            "📌 <strong>Rekomendasi untuk KPU Kota Surabaya:</strong> Lakukan kajian mendalam terhadap "
            "komentar negatif untuk mengidentifikasi aspek yang paling banyak dipermasalahkan. "
            "Buat program peningkatan layanan dan komunikasi publik yang terstruktur "
            "untuk memperbaiki persepsi masyarakat pada penyelenggaraan pemilu berikutnya."
        )
    else:  # Netral dominan
        box_color = "#fffde7"
        border_color = "#f9a825"
        title_color = "#e65100"
        icon = "ℹ️"
        headline = "Mayoritas Opini Publik Bersifat Netral"
        body = (
            f"Sebanyak <strong>{netral_n:,} komentar ({netral_pct:.1f}%)</strong> dari total "
            f"<strong>{total:,} data</strong> diklasifikasikan sebagai sentimen <strong>netral</strong>. "
            "Hal ini menunjukkan bahwa sebagian besar masyarakat menyampaikan informasi atau observasi "
            "tanpa ekspresi penilaian yang kuat — positif maupun negatif."
        )
        rekomendasi = (
            "📌 <strong>Rekomendasi untuk KPU Kota Surabaya:</strong> Tingkatkan strategi komunikasi publik "
            "yang proaktif agar masyarakat lebih terlibat dan memberikan penilaian yang positif. "
            "Transparansi informasi dan pelibatan masyarakat sejak dini dapat mendorong "
            "pergeseran sentimen dari netral ke positif."
        )

    st.markdown(
        f"""
<div style="background:{box_color};border-left:6px solid {border_color};border-radius:10px;padding:24px;margin-top:8px;">
<h3 style="color:{title_color};">{icon} {headline}</h3>
<p style="font-size:0.97rem;color:#212121;line-height:1.8;">{body}</p>
<hr style="border-color:{border_color};opacity:0.3;margin:14px 0;">
<p style="font-size:0.95rem;color:#212121;line-height:1.8;">{rekomendasi}</p>
<hr style="border-color:{border_color};opacity:0.3;margin:14px 0;">
<p style="font-size:0.88rem;color:#555;line-height:1.6;">
<em>Hasil analisis ini dihasilkan secara otomatis berdasarkan data komentar publik dari platform X (Twitter)
yang diproses melalui pipeline NLP dan diklasifikasikan menggunakan model <strong>{best_model_name}</strong>.
Temuan ini dapat digunakan sebagai referensi evaluasi kinerja lembaga, bahan laporan akademik,
serta dasar pengembangan penelitian lanjutan.</em>
</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
