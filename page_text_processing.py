import streamlit as st
import pandas as pd
import os
from utils import preprocess_text

def show_text_processing():
    st.title("⚙️ Text Processing Pipeline")
    
    st.markdown("""
    Halaman ini melakukan **preprocessing teks lengkap** untuk menghasilkan kolom `cleaned_text`.
    
    **Pipeline:** Case Folding → Cleaning → Tokenization → Normalization → Stopwords Removal → Stemming
    """)
    
    uploaded_file = st.file_uploader("Upload CSV Dataset (harus ada kolom 'text')", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Dataset loaded successfully!")
            
            st.subheader("Preview Dataset:")
            st.dataframe(df.head())
            
            # Check for text column
            text_col = None
            for col in ['text', 'Text', 'komentar', 'content']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                st.error("❌ Dataset harus memiliki kolom 'text' (atau 'Text', 'komentar', 'content')")
                return
            
            st.info(f"📝 Menggunakan kolom: **{text_col}**")
            
            # Show preprocessing info
            with st.expander("ℹ️ Tahapan Preprocessing"):
                st.markdown("""
                1. **Case Folding**: Huruf kecil semua
                2. **Cleaning**: Hapus URL, mention, hashtag, karakter khusus
                3. **Tokenization**: Pisah menjadi kata-kata
                4. **Normalization**: Ubah slang → kata baku (contoh: 'gak' → 'tidak')
                5. **Stopwords Removal**: Hapus kata umum ('yang', 'dan', dll)
                6. **Stemming**: Ubah ke kata dasar (contoh: 'memakan' → 'makan')
                """)
            
            if st.button("🚀 Jalankan Preprocessing", type="primary"):
                with st.spinner("Sedang memproses teks..."):
                    # Apply preprocessing to entire text column
                    df['cleaned_text'] = df[text_col].apply(preprocess_text)
                    
                    st.success("✅ Preprocessing selesai!")
                    
                    # Show results
                    st.subheader("Hasil Preprocessing:")
                    st.dataframe(df[[text_col, 'cleaned_text']].head(10))
                    
                    # Statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Data", len(df))
                    with col2:
                        empty_count = df['cleaned_text'].apply(lambda x: len(x) == 0).sum()
                        st.metric("Teks Kosong Setelah Cleaning", empty_count)
                    
                    if empty_count > 0:
                        st.warning(f"⚠️ {empty_count} baris menjadi kosong setelah preprocessing. Ini normal untuk teks pendek atau hanya berisi stopwords.")
                    
                    # Prepare output
                    output_cols = ['cleaned_text']
                    if 'label' in df.columns:
                        output_cols.append('label')
                        st.info("✅ Kolom 'label' ditemukan dan akan disimpan.")
                    
                    df_output = df[output_cols]
                    
                    # Save to data/clean
                    os.makedirs("data/clean", exist_ok=True)
                    save_path = os.path.join("data/clean",f"preprocessed_{uploaded_file.name}")
                    df_output.to_csv(save_path, index=False)
                    
                    st.success(f"💾 File tersimpan di: **{save_path}**")
                    st.info(f"📊 Kolom output: {', '.join(output_cols)}")
                    
                    # Download button
                    csv = df_output.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Hasil Preprocessing",
                        csv,
                        file_name=f"preprocessed_{uploaded_file.name}",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
