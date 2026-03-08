import streamlit as st


def show_home():
    st.title("📊 Sistem Analisis Sentimen Kinerja KPU Kota Surabaya")
    st.markdown("**Pemilu 2024 · Metode Naïve Bayes & Support Vector Machine (SVM)**")

    st.markdown("---")

    st.markdown(
        """
### 🎯 Tentang Sistem Ini

Aplikasi ini merupakan sistem berbasis web yang dikembangkan sebagai bagian dari penelitian
**Tugas Akhir** untuk menganalisis **sentimen opini publik** terhadap kinerja
**Komisi Pemilihan Umum (KPU) Kota Surabaya** pada penyelenggaraan Pemilu 2024.

Data yang digunakan bersumber dari platform **X (Twitter)**, yang dikumpulkan menggunakan
kata kunci terkait KPU Kota Surabaya selama periode pemilihan umum berlangsung.

---

### 🔄 Alur Sistem Analisis Sentimen

Sistem ini mengimplementasikan alur analisis lengkap yang terdiri dari:
"""
    )

    steps = [
        ("1️⃣", "Text Processing", "Pembersihan teks, normalisasi, tokenisasi, penghapusan stopword, dan stemming untuk mempersiapkan data teks sebelum analisis."),
        ("2️⃣", "Lexicon Labeling", "Pemberian label sentimen (Positif / Netral / Negatif) berdasarkan kamus sentimen (lexicon) bahasa Indonesia."),
        ("3️⃣", "Pembobotan TF-IDF", "Konversi teks ke representasi numerik menggunakan Term Frequency-Inverse Document Frequency sebagai fitur input model."),
        ("4️⃣", "Klasifikasi Naïve Bayes", "Pelatihan dan prediksi sentimen menggunakan metode probabilistik Multinomial Naïve Bayes."),
        ("5️⃣", "Klasifikasi SVM", "Pelatihan dan prediksi sentimen menggunakan metode Support Vector Machine dengan kernel linear."),
        ("6️⃣", "Evaluasi Model", "Perbandingan kinerja kedua model menggunakan metrik Accuracy, Precision, Recall, dan F1-Score."),
        ("7️⃣", "Visualisasi Hasil", "Tampilan interpretatif hasil analisis mencakup distribusi sentimen, WordCloud, Confusion Matrix, dan simpulan akademik."),
    ]

    for icon, title, desc in steps:
        with st.container():
            col_icon, col_text = st.columns([1, 9])
            with col_icon:
                st.markdown(f"<div style='font-size:2rem;text-align:center;padding-top:8px;'>{icon}</div>", unsafe_allow_html=True)
            with col_text:
                st.markdown(f"**{title}**  \n{desc}")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            """
<div style="background-color:#e3f2fd;border-radius:10px;padding:18px;">
<h4 style="color:#0d47a1;">🎓 Konteks Akademik</h4>
<ul style="color:#333;font-size:0.93rem;line-height:1.8;">
<li>Program Studi Teknik Informatika</li>
<li>Fokus: Analisis Sentimen NLP Bahasa Indonesia</li>
<li>Dataset: Komentar Twitter tentang KPU Surabaya</li>
<li>Tahun: Pemilu 2024</li>
</ul>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown(
            """
<div style="background-color:#e8f5e9;border-radius:10px;padding:18px;">
<h4 style="color:#1b5e20;">🚀 Cara Menggunakan</h4>
<ol style="color:#333;font-size:0.93rem;line-height:1.8;">
<li>Unggah dataset atau gunakan data yang tersedia</li>
<li>Ikuti alur menu dari kiri ke kanan secara berurutan</li>
<li>Lihat hasil akhir pada menu Visualisasi & Evaluasi</li>
<li>Unduh laporan untuk kebutuhan dokumentasi TA</li>
</ol>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.info(
        "ℹ️ **Petunjuk:** Gunakan menu navigasi di sidebar kiri untuk mulai memproses data. "
        "Pastikan mengikuti urutan alur dari **Text Processing** hingga **Visualisasi Hasil** "
        "agar hasil analisis dapat ditampilkan dengan benar."
    )
