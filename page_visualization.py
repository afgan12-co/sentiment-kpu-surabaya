import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from src.result_interpretation import (
    build_comparison_dataframe,
    compute_model_metrics,
    render_final_conclusion,
)


def create_wordcloud(text_list):
    """Create WordCloud from list of texts."""
    text = " ".join(text_list)
    if not text.strip():
        return None
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="RdYlGn",
        max_words=80,
        collocations=False,
    ).generate(text)
    return wc


def show_visualization():
    st.title("📊 Visualisasi Hasil Analisis Sentimen")
    st.markdown(
        "Halaman ini menyajikan hasil analisis sentimen opini publik terhadap kinerja "
        "**KPU Kota Surabaya** pada **Pemilu 2024** secara visual dan interpretatif, "
        "mencakup distribusi sentimen, perbandingan model, visualisasi kata kunci, "
        "serta simpulan akademik."
    )

    if "nb_pred" not in st.session_state or "svm_pred" not in st.session_state:
        st.error("⚠️ Latih model Naïve Bayes dan SVM terlebih dahulu.")
        return

    df_nb = st.session_state["nb_pred"]
    df_svm = st.session_state["svm_pred"]

    nb_metrics = compute_model_metrics(df_nb)
    svm_metrics = compute_model_metrics(df_svm)

    # ── 1. Sentiment Distribution ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📈 Distribusi Sentimen Hasil Prediksi")
    st.markdown(
        "Grafik berikut menunjukkan sebaran kategori sentimen yang dihasilkan oleh tiap model klasifikasi. "
        "Perhatikan perbandingan proporsi antara sentimen **Positif**, **Netral**, dan **Negatif**."
    )

    col1, col2 = st.columns(2)

    label_colors = {
        "positif": "#4CAF50", "negatif": "#F44336", "netral": "#FFC107",
        "1": "#4CAF50", "-1": "#F44336", "0": "#FFC107",
    }

    with col1:
        st.subheader("🔵 Naïve Bayes")
        counts_nb = df_nb["pred"].value_counts()
        colors_nb = [label_colors.get(str(l), "#90A4AE") for l in counts_nb.index]
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        bars = ax1.bar(counts_nb.index.astype(str), counts_nb.values, color=colors_nb, edgecolor="white", linewidth=0.8)
        ax1.set_xlabel("Kategori Sentimen")
        ax1.set_ylabel("Jumlah Data")
        ax1.set_title("Distribusi Prediksi — Naïve Bayes")
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                     str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig1)
        st.caption(
            "Distribusi sentimen hasil prediksi Naïve Bayes. "
            f"Total: {len(df_nb)} data."
        )

    with col2:
        st.subheader("🟢 Support Vector Machine (SVM)")
        counts_svm = df_svm["pred"].value_counts()
        colors_svm = [label_colors.get(str(l), "#90A4AE") for l in counts_svm.index]
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        bars = ax2.bar(counts_svm.index.astype(str), counts_svm.values, color=colors_svm, edgecolor="white", linewidth=0.8)
        ax2.set_xlabel("Kategori Sentimen")
        ax2.set_ylabel("Jumlah Data")
        ax2.set_title("Distribusi Prediksi — SVM")
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                     str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig2)
        st.caption(
            "Distribusi sentimen hasil prediksi SVM. "
            f"Total: {len(df_svm)} data."
        )

    # ── 2. Model Comparison ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🏆 Perbandingan Kinerja Naïve Bayes vs SVM")
    st.markdown(
        "Grafik berikut membandingkan tingkat akurasi kedua model secara visual, "
        "dilengkapi tabel metrik evaluasi lengkap."
    )

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    models = ["Naïve Bayes", "SVM"]
    metrics_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    nb_vals  = [nb_metrics[k]  for k in metrics_keys]
    svm_vals = [svm_metrics[k] for k in metrics_keys]

    x = range(len(metric_labels))
    width = 0.35
    bars_nb  = ax3.bar([i - width/2 for i in x], nb_vals,  width, label="Naïve Bayes", color="#2196F3", alpha=0.85)
    bars_svm = ax3.bar([i + width/2 for i in x], svm_vals, width, label="SVM",         color="#4CAF50", alpha=0.85)

    for bar in list(bars_nb) + list(bars_svm):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax3.set_ylabel("Nilai Metrik")
    ax3.set_ylim([0, 1.12])
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(metric_labels)
    ax3.set_title("Perbandingan Metrik Evaluasi — NB vs SVM")
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)
    st.caption("Batang lebih tinggi menandakan performa yang lebih baik pada metrik tersebut.")

    st.markdown("### 📋 Tabel Ringkasan Metrik Evaluasi")
    comparison_df = build_comparison_dataframe(nb_metrics, svm_metrics)
    st.dataframe(
        comparison_df.style
        .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"})
        .highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1-Score"]),
        use_container_width=True,
    )
    st.caption("Sel yang disorot hijau menunjukkan model dengan nilai tertinggi pada metrik tersebut.")

    # ── 3. WordClouds ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## ☁️ WordCloud per Kategori Sentimen")
    st.markdown(
        "WordCloud memvisualisasikan kata-kata yang paling sering muncul pada tiap kategori sentimen. "
        "Semakin besar ukuran kata, semakin sering kata tersebut muncul dalam komentar dengan kategori tersebut."
    )

    if "tfidf_dataset" not in st.session_state:
        st.warning("⚠️ Dataset asli tidak ditemukan. Jalankan TF-IDF terlebih dahulu.")
        st.info("Untuk menampilkan WordCloud, pastikan Anda telah menjalankan preprocessing dan TF-IDF.")
    else:
        df_original = st.session_state["tfidf_dataset"]

        viz_column = None
        if "text_final" in df_original.columns:
            viz_column = "text_final"
            st.success("✅ WordCloud menggunakan kolom `text_final` (bebas kata Bahasa Inggris)")
        elif "cleaned_text" in df_original.columns:
            viz_column = "cleaned_text"
            st.warning("⚠️ Kolom `text_final` tidak ditemukan — menggunakan `cleaned_text`.")

        if viz_column is None:
            st.error("❌ Kolom teks tidak ditemukan. Silakan re-process data di halaman Text Processing.")
        else:
            from src.lexicon_loader import load_lexicons
            from src.auto_english_detector import MultiSignalEnglishDetector
            lexicons = load_lexicons(force_reload=True)
            english_detector = MultiSignalEnglishDetector(
                lexicons["english_words"],
                lexicons["indo_whitelist"],
                lexicons["english_keep"],
            )

            sentiments = sorted(df_original["label"].unique())
            label_map = {
                "positif": ("🟢 Sentimen Positif", "Greens"),
                "netral":  ("🟡 Sentimen Netral",  "YlOrBr"),
                "negatif": ("🔴 Sentimen Negatif", "Reds"),
                "1":       ("🟢 Sentimen Positif", "Greens"),
                "0":       ("🟡 Sentimen Netral",  "YlOrBr"),
                "-1":      ("🔴 Sentimen Negatif", "Reds"),
                 1:        ("🟢 Sentimen Positif", "Greens"),
                 0:        ("🟡 Sentimen Netral",  "YlOrBr"),
                -1:        ("🔴 Sentimen Negatif", "Reds"),
            }

            cols = st.columns(min(len(sentiments), 3))

            for idx, sentiment in enumerate(sentiments):
                display_name, cmap_name = label_map.get(
                    sentiment,
                    label_map.get(str(sentiment), (f"Sentimen {str(sentiment).capitalize()}", "Blues"))
                )
                with cols[idx % 3]:
                    st.subheader(display_name)
                    raw_texts = df_original[df_original["label"] == sentiment][viz_column].tolist()

                    if raw_texts:
                        raw_combined = " ".join(str(t) for t in raw_texts if isinstance(t, str) and t.strip())
                        final_tokens = [
                            tok for tok in raw_combined.split()
                            if not english_detector.is_english(tok)[0]
                        ]
                        text_combined = " ".join(final_tokens)

                        if text_combined.strip():
                            wc = WordCloud(
                                width=700, height=350,
                                background_color="white",
                                max_words=70,
                                collocations=False,
                            ).generate(text_combined)
                            fig_wc, ax_wc = plt.subplots(figsize=(7, 3.5))
                            ax_wc.imshow(wc.to_image(), interpolation="bilinear")
                            ax_wc.axis("off")
                            ax_wc.set_title(display_name, fontsize=11, pad=8)
                            plt.tight_layout()
                            st.pyplot(fig_wc)
                            st.caption(f"{len(final_tokens):,} token · {len(raw_texts):,} komentar")
                        else:
                            st.info(f"Tidak ada teks Indonesia tersisa untuk {display_name}.")
                    else:
                        st.info(f"Tidak ada data untuk {display_name}.")

            # Audit Panel (collapsible)
            st.markdown("---")
            with st.expander("🔍 Audit Panel: Deteksi Kebocoran Token Bahasa Inggris"):
                all_tokens = []
                for txt in df_original[viz_column].head(200):
                    if isinstance(txt, str):
                        all_tokens.extend(txt.split())

                leaked = [
                    f"{tok} ({method})"
                    for tok in all_tokens[:500]
                    for is_eng, method in [english_detector.is_english(tok)]
                    if is_eng
                ]
                if leaked:
                    st.warning(f"⚠️ Terdeteksi {len(leaked)} token Bahasa Inggris yang lolos.")
                    st.write(", ".join(list(set(leaked))[:50]))
                else:
                    st.success("✅ Tidak ada kebocoran token Bahasa Inggris terdeteksi pada sampel data.")

                from collections import Counter
                top_tokens = Counter(all_tokens).most_common(30)
                st.write("**Top 30 Token Terbanyak:**")
                st.write(", ".join(f"{w} ({c})" for w, c in top_tokens))

    # ── 4. Confusion Matrix ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔥 Confusion Matrix")
    st.markdown(
        "Confusion Matrix menampilkan rincian prediksi model dibandingkan label sebenarnya. "
        "Diagonal utama menunjukkan prediksi yang tepat; sel di luar diagonal menunjukkan kesalahan klasifikasi."
    )

    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        st.subheader("Naïve Bayes")
        cm_nb = confusion_matrix(df_nb["true"], df_nb["pred"])
        fig_cm1, ax_cm1 = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm_nb, annot=True, fmt="d", cmap="Blues", ax=ax_cm1,
            xticklabels=sorted(df_nb["true"].unique()),
            yticklabels=sorted(df_nb["true"].unique()),
        )
        ax_cm1.set_xlabel("Prediksi")
        ax_cm1.set_ylabel("Aktual")
        ax_cm1.set_title("Confusion Matrix — NB")
        plt.tight_layout()
        st.pyplot(fig_cm1)
        st.caption("Confusion Matrix Naïve Bayes — prediksi vs label aktual.")

    with col_cm2:
        st.subheader("SVM")
        cm_svm = confusion_matrix(df_svm["true"], df_svm["pred"])
        fig_cm2, ax_cm2 = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm_svm, annot=True, fmt="d", cmap="Greens", ax=ax_cm2,
            xticklabels=sorted(df_svm["true"].unique()),
            yticklabels=sorted(df_svm["true"].unique()),
        )
        ax_cm2.set_xlabel("Prediksi")
        ax_cm2.set_ylabel("Aktual")
        ax_cm2.set_title("Confusion Matrix — SVM")
        plt.tight_layout()
        st.pyplot(fig_cm2)
        st.caption("Confusion Matrix SVM — prediksi vs label aktual.")

    # ── 5. Final Conclusion ───────────────────────────────────────────────────
    # Gunakan model terbaik berdasarkan F1-Score untuk kesimpulan akhir
    if svm_metrics["f1"] >= nb_metrics["f1"]:
        best_pred_df   = df_svm
        best_model_name = "Support Vector Machine (SVM)"
    else:
        best_pred_df   = df_nb
        best_model_name = "Naïve Bayes"

    render_final_conclusion(best_pred_df, best_model_name)
