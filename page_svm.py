import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from datetime import datetime

def show_svm():
    st.title("⚡ Modul Klasifikasi SVM")
    
    st.markdown("""
    **Support Vector Machine (SVM)** mencari hyperplane terbaik yang memisahkan kelas sentimen dengan margin maksimum.
    
    **Pipeline:** TF-IDF → SMOTE → LinearSVC
    
    ⚠️ **Penting**: Halaman ini menggunakan TF-IDF matrix yang dibuat di halaman **Pembobotan TF-IDF**.
    Jangan upload file di sini!
    """)

    # Code snippet
    with st.expander("📄 Lihat Kode Pipeline"):
        st.code("""
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC

# Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Train SVM
model_svm = LinearSVC(random_state=42, max_iter=2000)
model_svm.fit(X_train_resampled, y_train_resampled)
y_pred_svm = model_svm.predict(X_test_tfidf)
        """, language='python')

    st.markdown("---")
    
    # Check if TF-IDF matrices exist in session state
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
    
    # Show TF-IDF info
    st.success("✅ TF-IDF Matrix ditemukan di memori!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", st.session_state['X_train_tfidf'].shape[0])
    with col2:
        st.metric("Testing Samples", st.session_state['X_test_tfidf'].shape[0])
    with col3:
        st.metric("Features", st.session_state['X_train_tfidf'].shape[1])
    
    # Check class distribution
    st.markdown("---")
    st.subheader("📊 Class Distribution")
    y_train = st.session_state['y_train']
    unique, counts = pd.Series(y_train).value_counts().index, pd.Series(y_train).value_counts().values
    
    dist_df = pd.DataFrame({'Label': unique, 'Count': counts})
    st.dataframe(dist_df)
    
    # Check imbalance
    if len(counts) > 1 and counts.min() / counts.max() < 0.5:
        st.warning("⚠️ Terdeteksi ketidakseimbangan kelas. SMOTE akan diterapkan untuk oversampling.")
    
    st.markdown("---")
    st.subheader("📊 Training Model SVM")
    
    # Train model button
    if st.button("🚀 Latih Model SVM (dengan SMOTE)", type="primary"):
        with st.spinner("Melatih model SVM..."):
            try:
                # Get data from session state
                X_train = st.session_state['X_train_tfidf']
                X_test = st.session_state['X_test_tfidf']
                y_train = st.session_state['y_train']
                y_test = st.session_state['y_test']
                
                # Apply SMOTE
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                
                st.info(f"SMOTE: {X_train.shape[0]} → {X_train_resampled.shape[0]} samples")
                
                # Create and train model
                model_svm = LinearSVC(random_state=42, max_iter=2000)
                model_svm.fit(X_train_resampled, y_train_resampled)
                
                # Predict
                y_pred = model_svm.predict(X_test)
                
                # Save to session state
                st.session_state["svm_model"] = model_svm
                st.session_state["svm_pred"] = pd.DataFrame({
                    "true": y_test,
                    "pred": y_pred
                })
                
                from sklearn.metrics import accuracy_score
                accuracy = accuracy_score(y_test, y_pred)
                
                st.success(f"✅ Model SVM berhasil dilatih! Accuracy: {accuracy:.4f}")
                
                # Show sample predictions
                st.write("**Contoh Hasil Prediksi:**")
                sample_df = st.session_state["svm_pred"].head(10)
                st.dataframe(sample_df)
                
                # Save model
                os.makedirs("models", exist_ok=True)
                model_path = "models/model_svm.pkl"
                joblib.dump(model_svm, model_path)
                
                with open(model_path, "rb") as f:
                    st.download_button(
                        "💾 Download Model SVM",
                        f,
                        file_name="model_svm.pkl",
                        mime="application/octet-stream"
                    )

                # ==========================================
                # AUTO-SAVE RESULTS FOR DASHBOARD
                # ==========================================
                try:
                    with st.spinner("Mengupdate data dashboard..."):
                        # Load full dataset for dashboard population
                        if 'tfidf_dataset' in st.session_state:
                             df_full = st.session_state['tfidf_dataset']
                             
                             # Ensure we use the vectorizer to transform all data
                             if 'tfidf_vectorizer' in st.session_state:
                                 vectorizer = st.session_state['tfidf_vectorizer']
                                 X_full = df_full['cleaned_text'].fillna("").values
                                 X_full_tfidf = vectorizer.transform(X_full)
                                 
                                 # Predict all
                                 y_full_pred = model_svm.predict(X_full_tfidf)
                                 
                                 # Get confidence (using decision function + sigmoid)
                                 decision = model_svm.decision_function(X_full_tfidf)
                                 # Handle binary vs multiclass
                                 if decision.ndim == 1:
                                     # Binary: dist < 0 class 0, dist > 0 class 1
                                     confidences = 1 / (1 + np.exp(-np.abs(decision)))
                                 else:
                                     # Multiclass: take max score
                                     confidences = 1 / (1 + np.exp(-np.max(decision, axis=1)))
                                 
                                 # Prepare results list
                                 from dashboard_utils import detect_category, save_bulk_analysis_results
                                 
                                 dashboard_results = []
                                 for i, (text, pred, conf) in enumerate(zip(df_full['cleaned_text'], y_full_pred, confidences)):
                                     # Normalize sentiment
                                     sentiment_map = {'positif': 'positive', 'negatif': 'negative', 'netral': 'neutral'}
                                     sentiment = sentiment_map.get(pred.lower(), 'neutral')
                                     
                                     dashboard_results.append({
                                         "id": i + 1,
                                         "text": text,
                                         "cleaned_text": text,
                                         "sentiment": sentiment,
                                         "confidence": float(round(conf, 4)),
                                         "category": detect_category(text),
                                         "timestamp": datetime.now().isoformat() + 'Z'
                                     })
                                 
                                 # Save to dashboard JSON
                                 save_bulk_analysis_results(dashboard_results)
                                 st.success(f"✅ Dashboard Statistics berhasil diupdate dengan {len(dashboard_results)} data baru!")
                except Exception as e:
                    st.warning(f"⚠️ Gagal mengupdate dashboard: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ Error saat training: {e}")
    
    # Show model info if already trained
    if 'svm_model' in st.session_state:
        st.markdown("---")
        st.info("✅ Model SVM sudah dilatih dan tersimpan di memori. Lanjut ke halaman **Evaluasi Model** untuk melihat performa.")

    st.markdown("---")
    st.subheader("🎯 Kenapa SVM?")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Kelebihan:**
        - Sangat kuat untuk data berdimensi tinggi
        - Akurasi biasanya lebih tinggi dari Naive Bayes
        - Tidak terpengaruh outlier kecil
        """)
    
    with col2:
        st.info("""
        **Catatan:**
        - SMOTE membantu mengatasi imbalance class
        - Parameter `C` dapat dituning dengan GridSearchCV
        """)
