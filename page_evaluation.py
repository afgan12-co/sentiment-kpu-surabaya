import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.result_interpretation import (
    build_comparison_dataframe,
    compute_model_metrics,
    render_model_comparison_interpretation,
    render_sentiment_meaning_section,
)

def show_evaluation():
    st.title("📈 Modul Evaluasi Model")

    if "nb_pred" not in st.session_state or "svm_pred" not in st.session_state:
        st.error("⚠️ Latih model Naïve Bayes dan SVM terlebih dahulu di halaman masing-masing.")
        return

    # Naive Bayes Evaluation
    st.markdown("## 🔵 Evaluasi Naïve Bayes")
    nb = st.session_state["nb_pred"]
    
    nb_metrics = compute_model_metrics(nb)
    nb_acc = nb_metrics["accuracy"]
    nb_prec = nb_metrics["precision"]
    nb_rec = nb_metrics["recall"]
    nb_f1 = nb_metrics["f1"]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{nb_acc:.4f}")
    col2.metric("Precision", f"{nb_prec:.4f}")
    col3.metric("Recall", f"{nb_rec:.4f}")
    col4.metric("F1-Score", f"{nb_f1:.4f}")
    
    with st.expander("📊 Classification Report (NB)"):
        st.text(classification_report(nb["true"], nb["pred"], zero_division=0))

    st.subheader("Confusion Matrix - Naive Bayes")
    cm_nb = confusion_matrix(nb["true"], nb["pred"])
    fig_nb, ax_nb = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues", ax=ax_nb, 
                xticklabels=sorted(nb["true"].unique()),
                yticklabels=sorted(nb["true"].unique()))
    ax_nb.set_xlabel("Predicted")
    ax_nb.set_ylabel("Actual")
    st.pyplot(fig_nb)

    # SVM Evaluation
    st.markdown("---")
    st.markdown("## 🟢 Evaluasi SVM")
    svm = st.session_state["svm_pred"]
    
    svm_metrics = compute_model_metrics(svm)
    svm_acc = svm_metrics["accuracy"]
    svm_prec = svm_metrics["precision"]
    svm_rec = svm_metrics["recall"]
    svm_f1 = svm_metrics["f1"]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{svm_acc:.4f}")
    col2.metric("Precision", f"{svm_prec:.4f}")
    col3.metric("Recall", f"{svm_rec:.4f}")
    col4.metric("F1-Score", f"{svm_f1:.4f}")
    
    with st.expander("📊 Classification Report (SVM)"):
        st.text(classification_report(svm["true"], svm["pred"], zero_division=0))

    st.subheader("Confusion Matrix - SVM")
    cm_svm = confusion_matrix(svm["true"], svm["pred"])
    fig_svm, ax_svm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens", ax=ax_svm,
                xticklabels=sorted(svm["true"].unique()),
                yticklabels=sorted(svm["true"].unique()))
    ax_svm.set_xlabel("Predicted")
    ax_svm.set_ylabel("Actual")
    st.pyplot(fig_svm)

    # Comparison
    st.markdown("---")
    st.markdown("## 🏆 Perbandingan Model")
    
    comparison = build_comparison_dataframe(nb_metrics, svm_metrics)
    
    st.dataframe(comparison.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']))

    # Interpretation
    render_model_comparison_interpretation(nb_metrics, svm_metrics)

    st.markdown("---")
    st.subheader("🧾 Kesimpulan Akhir Hasil Klasifikasi")
    render_sentiment_meaning_section()

    # Export
    st.markdown("---")
    st.subheader("💾 Export Hasil")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        csv_comparison = comparison.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Tabel Perbandingan",
            csv_comparison,
            file_name="evaluasi_model.csv",
            mime="text/csv"
        )
    
    with col_exp2:
        # Export confusion matrices
        cm_df = pd.DataFrame({
            'NB_CM': cm_nb.flatten(),
            'SVM_CM': cm_svm.flatten()
        })
        csv_cm = cm_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Confusion Matrices",
            csv_cm,
            file_name="confusion_matrices.csv",
            mime="text/csv"
        )
