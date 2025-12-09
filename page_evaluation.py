import streamlit as st
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def show_evaluation():
    st.title("📈 Modul Evaluasi Model")

    if "nb_pred" not in st.session_state or "svm_pred" not in st.session_state:
        st.error("⚠️ Latih model Naïve Bayes dan SVM terlebih dahulu di halaman masing-masing.")
        return

    # Naive Bayes Evaluation
    st.markdown("## 🔵 Evaluasi Naïve Bayes")
    nb = st.session_state["nb_pred"]
    
    nb_acc = accuracy_score(nb["true"], nb["pred"])
    nb_prec = precision_score(nb["true"], nb["pred"], average='macro', zero_division=0)
    nb_rec = recall_score(nb["true"], nb["pred"], average='macro', zero_division=0)
    nb_f1 = f1_score(nb["true"], nb["pred"], average='macro', zero_division=0)
    
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
    
    svm_acc = accuracy_score(svm["true"], svm["pred"])
    svm_prec = precision_score(svm["true"], svm["pred"], average='macro', zero_division=0)
    svm_rec = recall_score(svm["true"], svm["pred"], average='macro', zero_division=0)
    svm_f1 = f1_score(svm["true"], svm["pred"], average='macro', zero_division=0)
    
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
    
    comparison = pd.DataFrame({
        "Model": ["Naïve Bayes", "SVM"],
        "Accuracy": [nb_acc, svm_acc],
        "Precision": [nb_prec, svm_prec],
        "Recall": [nb_rec, svm_rec],
        "F1-Score": [nb_f1, svm_f1]
    })
    
    st.dataframe(comparison.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']))

    # Interpretation
    st.subheader("📖 Interpretasi")
    best_model = "SVM" if svm_acc > nb_acc else "Naïve Bayes"
    st.info(f"""
    Model **{best_model}** memiliki performa lebih baik dengan akurasi **{max(nb_acc, svm_acc):.4f}**.
    
    - Jika SVM lebih unggul: Kemungkinan karena kemampuannya menangani data berdimensi tinggi dan penggunaan SMOTE untuk mengatasi ketidakseimbangan kelas.
    - Jika NB lebih unggul: Kemungkinan karena dataset relatif sederhana dan asumsi independensi fitur cukup terpenuhi.
    """)

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
