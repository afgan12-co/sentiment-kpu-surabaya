import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime

def show_naive_bayes():
    st.title("🤖 Modul Klasifikasi Naïve Bayes")
    
    st.markdown("""
    **Multinomial Naïve Bayes** adalah algoritma probabilistik yang bekerja dengan asumsi independensi antar fitur.
    
    ⚠️ **Penting**: Halaman ini menggunakan TF-IDF matrix yang dibuat di halaman **Pembobotan TF-IDF**.
    Jangan upload file di sini!
    """)

    # Code snippet
    with st.expander("📄 Lihat Kode Pipeline"):
        st.code("""
from sklearn.naive_bayes import MultinomialNB

# Data sudah dalam bentuk TF-IDF matrix
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)
y_pred_nb = model_nb.predict(X_test_tfidf)
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
    
    st.markdown("---")
    st.subheader("📊 Training Model Naive Bayes")
    
    # Train model button
    if st.button("🚀 Latih Model Naïve Bayes", type="primary"):
        with st.spinner("Melatih model Naive Bayes..."):
            # Get data from session state
            X_train = st.session_state['X_train_tfidf']
            X_test = st.session_state['X_test_tfidf']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            
            # Create and train model
            model_nb = MultinomialNB()
            model_nb.fit(X_train, y_train)
            
            # Predict
            y_pred = model_nb.predict(X_test)
            
            # Save to session state
            st.session_state["nb_model"] = model_nb
            st.session_state["nb_pred"] = pd.DataFrame({
                "true": y_test,
                "pred": y_pred
            })
            
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_test, y_pred)
            
            st.success(f"✅ Model berhasil dilatih! Accuracy: {accuracy:.4f}")
            
            # Show sample predictions
            st.write("**Contoh Hasil Prediksi:**")
            sample_df = st.session_state["nb_pred"].head(10)
            st.dataframe(sample_df)
            
            # Save model
            os.makedirs("models", exist_ok=True)
            model_path = "models/model_naive_bayes.pkl"
            joblib.dump(model_nb, model_path)
            
            with open(model_path, "rb") as f:
                st.download_button(
                    "💾 Download Model Naive Bayes",
                    f,
                    file_name="model_nb.pkl",
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
                             y_full_pred = model_nb.predict(X_full_tfidf)
                             
                             # Get confidence
                             y_full_proba = model_nb.predict_proba(X_full_tfidf)
                             confidences = [max(prob) for prob in y_full_proba]
                             
                             # Prepare results list
                             from dashboard_utils import detect_category, save_bulk_analysis_results
                             
                             dashboard_results = []
                             for i, (text, pred, conf) in enumerate(zip(df_full['cleaned_text'], y_full_pred, confidences)):
                                 # Normalize sentiment
                                 sentiment_map = {'positif': 'positive', 'negatif': 'negative', 'netral': 'neutral'}
                                 sentiment = sentiment_map.get(pred.lower(), 'neutral')
                                 
                                 dashboard_results.append({
                                     "id": i + 1,
                                     "text": text,  # Use cleaned text or original if available
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

    
    # Show model info if already trained
    if 'nb_model' in st.session_state:
        st.markdown("---")
        st.info("✅ Model Naive Bayes sudah dilatih dan tersimpan di memori. Lanjut ke halaman **Evaluasi Model** untuk melihat performa.")

    st.markdown("---")
    st.subheader("✅ Kelebihan & ⚠️ Keterbatasan")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Kelebihan:**
        - Cepat dan efisien
        - Cocok untuk dataset besar
        - Tidak membutuhkan komputasi berat
        """)
    
    with col2:
        st.warning("""
        **Keterbatasan:**
        - Asumsi independensi fitur sering tidak realistis
        - Kurang optimal untuk data berdimensi sangat tinggi
        """)
