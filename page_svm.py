import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter

def show_svm():
    st.title("⚡ Modul Klasifikasi SVM + SMOTE")
    
    st.markdown("""
    **Support Vector Machine (SVM)** mencari hyperplane terbaik yang memisahkan kelas sentimen.
    Modul ini dilengkapi dengan **SMOTE (Synthetic Minority Over-sampling Technique)** untuk menangani ketidakseimbangan kelas.
    """)

    # Check session state
    if 'X_train_tfidf' not in st.session_state or 'y_train' not in st.session_state:
        st.error("❌ TF-IDF Matrix belum dibuat!")
        st.warning("""
        **Langkah yang harus dilakukan:**
        1. Lakukan **Text Processing** untuk mendapatkan `cleaned_text`
        2. Lakukan **Lexicon Labeling** untuk mendapatkan `label`
        3. Buka halaman **Pembobotan TF-IDF** dan buat TF-IDF matrix
        4. Kembali ke halaman ini untuk training model
        """)
        return
    
    # Load data
    X_train = st.session_state['X_train_tfidf']
    X_test = st.session_state['X_test_tfidf']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    
    # --- SECTION 1: Class Distribution Analysis ---
    st.subheader("1. Analisis Distribusi Kelas")
    
    counter = Counter(y_train)
    dist_df = pd.DataFrame.from_dict(counter, orient='index', columns=['Count']).reset_index()
    dist_df.columns = ['Label', 'Count']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Distribusi Awal (Sebelum SMOTE):**")
        st.dataframe(dist_df)
    with col2:
        # Check imbalance ratio
        min_class = min(counter.values())
        max_class = max(counter.values())
        ratio = min_class / max_class
        
        st.metric("Imbalance Ratio (Min/Max)", f"{ratio:.2f}")
        
        if ratio < 0.6:
            st.warning("⚠️ Data Imbalanced! Disarankan menggunakan SMOTE.")
            
            # Preview prediction of SMOTE
            st.markdown("**Estimasi Setelah SMOTE:**")
            # SMOTE usually balances to the majority class count
            estimated_counts = {k: max_class for k in counter.keys()}
            est_df = pd.DataFrame.from_dict(estimated_counts, orient='index', columns=['Count (Est)']).reset_index()
            est_df.columns = ['Label', 'Count (Est)']
            st.dataframe(est_df)
            st.caption("*SMOTE akan menyetarakan jumlah data minoritas dengan kelas mayoritas.*")
        else:
            st.success("✅ Data cukup seimbang.")

    st.markdown("---")

    # --- SECTION 2: Konfigurasi Training ---
    st.sidebar.markdown("### 🛠️ Konfigurasi SVM & SMOTE")
    
    smote_mode = st.sidebar.radio(
        "Mode SMOTE",
        ["Compare (Baseline vs SMOTE)", "Auto", "On", "Off"],
        index=0,
        help="Compare: Latih 2 model sekaligus untuk perbandingan."
    )
    
    k_neighbors = st.sidebar.slider("K-Neighbors for SMOTE", 1, 10, 5)
    
    st.subheader("2. Training Model")
    
    if st.button("🚀 Mulai Training Model", type="primary"):
        with st.spinner("Sedang melatih model..."):
            
            reports = {}
            
            # Helper to train
            def train_model(use_smote, name):
                pipeline_steps = []
                if use_smote:
                    pipeline_steps.append(('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)))
                
                pipeline_steps.append(('svm', LinearSVC(random_state=42, max_iter=2000)))
                
                model = ImbPipeline(steps=pipeline_steps)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                acc = accuracy_score(y_test, y_pred)
                return model, y_pred, acc
            
            # Determine execution plan
            exec_list = []
            if smote_mode == "Compare (Baseline vs SMOTE)":
                exec_list = [(False, "Baseline"), (True, "SMOTE")]
            elif smote_mode == "Auto":
                use_smote = ratio < 0.6
                name = "SMOTE" if use_smote else "Baseline"
                exec_list = [(use_smote, name)]
            elif smote_mode == "On":
                exec_list = [(True, "SMOTE")]
            else: # Off
                exec_list = [(False, "Baseline")]
            
            # Execute training
            cols = st.columns(len(exec_list))
            
            for idx, (use_smote, name) in enumerate(exec_list):
                model, y_pred, acc = train_model(use_smote, name)
                
                # Save to session (overwrite if single, store both if compare)
                st.session_state["svm_model"] = model # Last one wins as 'main' model
                st.session_state["svm_pred"] = pd.DataFrame({"true": y_test, "pred": y_pred})
                
                if name == "Baseline":
                    st.session_state["svm_baseline_pred"] = y_pred
                else:
                    st.session_state["svm_smote_pred"] = y_pred
                
                with cols[idx]:
                    st.success(f"✅ Model {name} Selesai")
                    st.metric(f"Akurasi {name}", f"{acc:.4f}")
                    
            # --- SECTION 3: Visualization & Comparison ---
            st.markdown("---")
            st.subheader("3. Evaluasi & Kontribusi SMOTE")
            
            if smote_mode == "Compare (Baseline vs SMOTE)":
                y_base = st.session_state["svm_baseline_pred"]
                y_smote = st.session_state["svm_smote_pred"]
                
                # 1. Metrics Comparison Table
                metrics_data = []
                # Prepare metrics comparison
                metrics_data = []
                for label, pred in [("Baseline", y_base), ("SMOTE", y_smote)]:
                    metrics_data.append({
                        "Model": label,
                        "Accuracy": accuracy_score(y_test, pred),
                        "Recall (Macro)": recall_score(y_test, pred, average='macro'),
                        "Precision (Macro)": precision_score(y_test, pred, average='macro'),
                        "F1 (Macro)": f1_score(y_test, pred, average='macro')
                    })
                
                metrics_df = pd.DataFrame(metrics_data).set_index("Model")
                metrics_df.loc["Diff"] = metrics_df.loc["SMOTE"] - metrics_df.loc["Baseline"]
                
                # Use modern map instead of applymap
                st.table(metrics_df.style.format("{:.4f}").map(
                    lambda v: 'color: green' if v > 0 else ('color: red' if v < 0 else ''), 
                    subset=pd.IndexSlice[['Diff'], :]
                ))
                
                st.info("💡 **Interpretasi:** Baris 'Diff' menunjukkan kenaikan/penurunan performa akibat SMOTE.")

                # 2. Confusion Matrices
                st.write("**Perbandingan Confusion Matrix:**")
                cm_base = confusion_matrix(y_test, y_base)
                cm_smote = confusion_matrix(y_test, y_smote)
                
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                
                labels = sorted(set(y_test))
                sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=ax[0], xticklabels=labels, yticklabels=labels)
                ax[0].set_title("Baseline (Tanpa SMOTE)")
                
                sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens', ax=ax[1], xticklabels=labels, yticklabels=labels)
                ax[1].set_title("Dengan SMOTE")
                
                st.pyplot(fig)
                
                # 3. Minority Class Analysis
                minority_class = min(counter, key=counter.get)
                st.write(f"**Analisis Kelas Minoritas ({minority_class}):**")
                
                rec_base = recall_score(y_test, y_base, labels=[minority_class], average=None)[0]
                rec_smote = recall_score(y_test, y_smote, labels=[minority_class], average=None)[0]
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric(f"Recall Baseline ({minority_class})", f"{rec_base:.2%}")
                col_b.metric(f"Recall SMOTE ({minority_class})", f"{rec_smote:.2%}")
                col_c.metric("Improvement", f"{rec_smote - rec_base:+.2%}")
                
                if rec_smote > rec_base:
                    st.success(f"✅ SMOTE berhasil meningkatkan kemampuan model mengenali kelas minoritas **{minority_class}**.")
                else:
                    st.warning(f"⚠️ SMOTE tidak memberikan peningkatan signifikan pada kelas minoritas ini.")

            else:
                # Single model view
                st.text("Mode single model aktif. Lihat detail di menu Evaluasi Model.")
                
            # Code Snippet
            with st.expander("📄 Lihat Implementation Code"):
                st.code("""
# Pipeline Implementation
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('svm', LinearSVC(random_state=42))
])
pipeline.fit(X_train, y_train)
                """, language="python")
    
    # --- Glosarium Section ---
    st.markdown("---")
    with st.expander("📚 Glosarium & Cara Membaca (Panduan Sidang)"):
        st.markdown("""
        ### Istilah Penting
        1.  **Imbalance Ratio**: Perbandingan jumlah kelas terkecil dengan kelas terbesar. Jika < 0.6, data dianggap tidak seimbang.
        2.  **SMOTE (Synthetic Minority Over-sampling Technique)**: Teknik untuk menambah data latih kelas minoritas secara sintetis agar seimbang dengan kelas mayoritas.
        3.  **Baseline vs SMOTE**: `Baseline` adalah model SVM tanpa penanganan imbalance. `SMOTE` adalah model dengan penanganan imbalance.
        4.  **Recall (Sensitivitas)**: Kemampuan model menemukan data positif yang sebenarnya. Recall tinggi pada kelas minoritas berarti model berhasil mengatasi bias.
        5.  **Data Leakage**: Kebocoran informasi data uji ke data latih. Di aplikasi ini, **SMOTE hanya diterapkan pada data latih (training set)**, sehingga pengujian (testing set) tetap murni dan valid.
        
        ### Cara Membaca Confusion Matrix
        - **Diagonal Utama (Warna Pekat)**: Jumlah prediksi benar. Semakin gelap semakin bagus.
        - **Model Baseline**: Biasanya gelap di kelas mayoritas (misal 'netral') tapi terang/kosong di kelas minoritas (misal 'negatif').
        - **Model SMOTE**: Warna di kelas minoritas akan menjadi lebih gelap (prediksi benar bertambah), menunjukkan SMOTE bekerja.
        """)

    # Disclaimer
    st.caption("Catatan: SMOTE hanya diterapkan pada Training Set untuk mencegah data leakage.")
