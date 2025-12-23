import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

def show_tfidf():
    st.title("📊 Pembobotan Kata TF-IDF")
    
    st.markdown("""
    **TF-IDF (Term Frequency - Inverse Document Frequency)** mengubah teks menjadi representasi numerik untuk machine learning.
    
    ⚠️ **Penting**: 
    - Halaman ini membutuhkan dataset yang sudah **dilabeli** (memiliki kolom `cleaned_text` dan `label`)
    - Hasil TF-IDF akan disimpan di memori untuk digunakan oleh model ML
    """)
    
    # Formula display
    with st.expander("📐 Rumus TF-IDF"):
        st.latex(r"TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)")
        st.latex(r"IDF(t) = \log\frac{N}{df(t)}")
    
    st.markdown("---")
    st.subheader("📂 Pilih Dataset Berlabel")
    
    # Load from data/labeled
    labeled_dir = "data/labeled"
    clean_dir = "data/clean"
    
    if not os.path.exists(labeled_dir):
        os.makedirs(labeled_dir, exist_ok=True)
    
    files = [f for f in os.listdir(labeled_dir) if f.endswith('.csv')]
    
    if not files:
        st.warning("⚠️ Tidak ada file di 'data/labeled'.")
        st.info("""
        **Untuk menggunakan halaman ini, Anda harus:**
        1. ✅ Lakukan **Text Processing** (menghasilkan `cleaned_text`)
        2. ✅ Lakukan **Lexicon Labeling** (menghasilkan `label`)
        3. 💾 Simpan hasil labeling ke `data/labeled`
        
        Setelah itu, kembali ke halaman ini.
        """)
        
        # Check if there are files in data/clean
        if os.path.exists(clean_dir):
            clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.csv')]
            if clean_files:
                st.info(f"""
                **Tip**: Terdeteksi {len(clean_files)} file di `data/clean`. 
                Silakan ke halaman **Lexicon Labeling** untuk melabeli file tersebut terlebih dahulu.
                """)
        return
    
    selected_file = st.selectbox("Pilih file berlabel:", files)
    
    if st.button("📥 Load Dataset"):
        df = pd.read_csv(os.path.join(labeled_dir, selected_file))
        
        # Validate columns
        if 'cleaned_text' not in df.columns or 'label' not in df.columns:
            st.error("❌ Dataset harus memiliki kolom 'cleaned_text' dan 'label'")
            st.warning("Pastikan file ini hasil dari Lexicon Labeling")
            return
        
        st.session_state['tfidf_dataset'] = df
        st.session_state['tfidf_source_file'] = selected_file
        st.success(f"✅ Dataset loaded: {len(df)} baris")
    
    if 'tfidf_dataset' in st.session_state:
        df = st.session_state['tfidf_dataset']
        
        st.subheader("Preview Dataset:")
        st.dataframe(df[['cleaned_text', 'label']].head())
        
        # Show label distribution
        st.write("**Distribusi Label:**")
        st.write(df['label'].value_counts())
        
        # TF-IDF Parameters
        st.markdown("---")
        st.subheader("⚙️ Konfigurasi TF-IDF & Split Data")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            max_features = st.number_input("Max Features", min_value=100, max_value=10000, value=1000, step=100)
        with col2:
            min_df = st.slider("Min Document Frequency", 1, 5, 1)
        with col3:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        
        if st.button("🚀 Buat TF-IDF Matrix & Split Data", type="primary"):
            with st.spinner("Membuat TF-IDF matrix..."):
                # Prepare X and y
                X = df['cleaned_text'].fillna("").values
                y = df['label'].values
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Create and fit TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    ngram_range=(1, 1)
                )
                
                # Fit only on training data
                X_train_tfidf = vectorizer.fit_transform(X_train)
                X_test_tfidf = vectorizer.transform(X_test)
                
                # Store in session state for ML models
                st.session_state['X_train_tfidf'] = X_train_tfidf
                st.session_state['X_test_tfidf'] = X_test_tfidf
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['tfidf_vectorizer'] = vectorizer
                st.session_state['feature_names'] = vectorizer.get_feature_names_out()

                # Save vectorizer for API usage
                os.makedirs("models", exist_ok=True)
                joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
                st.info("💾 TF-IDF Vectorizer disimpan untuk API usage")

                st.success("✅ TF-IDF Matrix berhasil dibuat!")
                
                # Show info
                st.info(f"""
                **Data Split:**
                - Training: {len(X_train)} samples
                - Testing: {len(X_test)} samples
                
                **TF-IDF Matrix Shape:**
                - Training: {X_train_tfidf.shape}
                - Testing: {X_test_tfidf.shape}
                
                **Jumlah Features:** {len(vectorizer.get_feature_names_out())}
                """)
                
                st.success("🎯 Data siap digunakan untuk training model Naive Bayes dan SVM!")
        
        # Show TF-IDF info if already created
        if 'X_train_tfidf' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Informasi TF-IDF Matrix")
            
            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                st.metric("Training Samples", st.session_state['X_train_tfidf'].shape[0])
            with col_i2:
                st.metric("Testing Samples", st.session_state['X_test_tfidf'].shape[0])
            with col_i3:
                st.metric("Total Features", st.session_state['X_train_tfidf'].shape[1])
            
            # Show top features
            st.subheader("🔝 Top 20 Features (by avg TF-IDF)")
            
            avg_tfidf = np.asarray(st.session_state['X_train_tfidf'].mean(axis=0)).flatten()
            top_indices = avg_tfidf.argsort()[-20:][::-1]
            feature_names = st.session_state['feature_names']
            
            top_features_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_indices],
                'Avg TF-IDF': [avg_tfidf[i] for i in top_indices]
            })
            
            st.dataframe(top_features_df)
            st.bar_chart(top_features_df.set_index('Feature')['Avg TF-IDF'])
            
            st.success("✅ TF-IDF Matrix siap! Lanjut ke halaman **Naive Bayes** atau **SVM** untuk training model.")
