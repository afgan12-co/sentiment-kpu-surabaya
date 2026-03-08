import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.result_interpretation import (
    build_comparison_dataframe,
    compute_model_metrics,
    render_model_comparison_interpretation,
)


def _metric_delta(val_a: float, val_b: float) -> str:
    """Kembalikan string delta metrik dengan tanda dan format."""
    delta = val_a - val_b
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def show_evaluation():
    st.title("📈 Evaluasi Model Klasifikasi Sentimen")
    st.markdown(
        "Halaman ini menyajikan hasil evaluasi dua metode klasifikasi — "
        "**Naïve Bayes** dan **Support Vector Machine (SVM)** — "
        "yang digunakan untuk mengklasifikasikan sentimen opini publik "
        "terhadap KPU Kota Surabaya pada Pemilu 2024."
    )

    if "nb_pred" not in st.session_state or "svm_pred" not in st.session_state:
        st.error("⚠️ Latih model Naïve Bayes dan SVM terlebih dahulu di halaman masing-masing.")
        return

    nb = st.session_state["nb_pred"]
    svm = st.session_state["svm_pred"]

    nb_metrics = compute_model_metrics(nb)
    svm_metrics = compute_model_metrics(svm)

    # ── Naive Bayes Section ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🔵 Evaluasi Naïve Bayes")
    st.markdown(
        "Naïve Bayes adalah metode klasifikasi berbasis probabilitas yang menghitung kemungkinan "
        "tiap komentar termasuk dalam kategori sentimen tertentu berdasarkan kemunculan kata-kata "
        "dalam data latih."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{nb_metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{nb_metrics['precision']:.4f}")
    c3.metric("Recall",    f"{nb_metrics['recall']:.4f}")
    c4.metric("F1-Score",  f"{nb_metrics['f1']:.4f}")

    with st.expander("📊 Classification Report Lengkap — Naïve Bayes"):
        st.text(classification_report(nb["true"], nb["pred"], zero_division=0))

    st.subheader("Confusion Matrix — Naïve Bayes")
    cm_nb = confusion_matrix(nb["true"], nb["pred"])
    fig_nb, ax_nb = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm_nb, annot=True, fmt="d", cmap="Blues", ax=ax_nb,
        xticklabels=sorted(nb["true"].unique()),
        yticklabels=sorted(nb["true"].unique()),
    )
    ax_nb.set_xlabel("Prediksi")
    ax_nb.set_ylabel("Aktual")
    ax_nb.set_title("Confusion Matrix — Naïve Bayes")
    st.pyplot(fig_nb)
    st.caption(
        "Setiap sel menunjukkan jumlah data yang diklasifikasikan. "
        "Diagonal utama adalah prediksi yang benar."
    )

    # ── SVM Section ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🟢 Evaluasi Support Vector Machine (SVM)")
    st.markdown(
        "SVM bekerja dengan mencari batas pemisah optimal (*hyperplane*) antara kelas-kelas sentimen "
        "di ruang fitur berdimensi tinggi yang dihasilkan oleh TF-IDF. "
        "Metode ini umumnya lebih kuat dalam menangani data teks yang bersifat *sparse*."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Accuracy",  f"{svm_metrics['accuracy']:.4f}",
        delta=_metric_delta(svm_metrics["accuracy"], nb_metrics["accuracy"]),
    )
    c2.metric(
        "Precision", f"{svm_metrics['precision']:.4f}",
        delta=_metric_delta(svm_metrics["precision"], nb_metrics["precision"]),
    )
    c3.metric(
        "Recall",    f"{svm_metrics['recall']:.4f}",
        delta=_metric_delta(svm_metrics["recall"], nb_metrics["recall"]),
    )
    c4.metric(
        "F1-Score",  f"{svm_metrics['f1']:.4f}",
        delta=_metric_delta(svm_metrics["f1"], nb_metrics["f1"]),
    )
    st.caption(
        "△ menunjukkan selisih kinerja SVM dibanding Naïve Bayes. "
        "Nilai positif berarti SVM lebih unggul pada metrik tersebut."
    )

    with st.expander("📊 Classification Report Lengkap — SVM"):
        st.text(classification_report(svm["true"], svm["pred"], zero_division=0))

    st.subheader("Confusion Matrix — SVM")
    cm_svm = confusion_matrix(svm["true"], svm["pred"])
    fig_svm, ax_svm = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm_svm, annot=True, fmt="d", cmap="Greens", ax=ax_svm,
        xticklabels=sorted(svm["true"].unique()),
        yticklabels=sorted(svm["true"].unique()),
    )
    ax_svm.set_xlabel("Prediksi")
    ax_svm.set_ylabel("Aktual")
    ax_svm.set_title("Confusion Matrix — SVM")
    st.pyplot(fig_svm)
    st.caption(
        "Confusion Matrix SVM — diagonal utama menunjukkan prediksi yang tepat "
        "per kelas sentimen."
    )

    # ── Comparison Section ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🏆 Perbandingan Kinerja Model")
    st.markdown(
        "Tabel berikut merangkum seluruh metrik evaluasi kedua model secara berdampingan. "
        "Sel yang di-*highlight* menunjukkan nilai terbaik untuk setiap metrik."
    )

    comparison = build_comparison_dataframe(nb_metrics, svm_metrics)
    st.dataframe(
        comparison.style
        .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"})
        .highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1-Score"]),
        use_container_width=True,
    )

    render_model_comparison_interpretation(nb_metrics, svm_metrics)

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Unduh Hasil Evaluasi")

    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        csv_comparison = comparison.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Tabel Perbandingan (.csv)",
            csv_comparison,
            file_name="evaluasi_model.csv",
            mime="text/csv",
        )

    with col_exp2:
        cm_df = pd.DataFrame({
            "NB_CM": cm_nb.flatten(),
            "SVM_CM": cm_svm.flatten(),
        })
        csv_cm = cm_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Confusion Matrices (.csv)",
            csv_cm,
            file_name="confusion_matrices.csv",
            mime="text/csv",
        )
