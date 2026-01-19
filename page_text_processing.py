import streamlit as st
import pandas as pd
import os
from src.pipeline import run_pipeline, PipelineConfig

def show_text_processing():
    st.title("⚙️ Text Processing Pipeline")
    
    st.markdown("""
    Halaman ini melakukan **preprocessing teks lengkap** dengan pipeline yang terstruktur dan konsisten.
    
    **Pipeline Runut:**
    1. **Cleaning** → hapus URL, email, simbol
    2. **Normalisasi** → slang & typo correction
    3. **English Removal** → buang token Bahasa Inggris
    4. **Stopword Removal** → buang kata umum
    5. **Stemming** → bentuk kata dasar
    """)
    
    # Sidebar Configuration
    st.sidebar.markdown("### 🔧 Konfigurasi Pipeline")
    
    config = PipelineConfig(
        remove_english=st.sidebar.checkbox("Hapus Token Bahasa Inggris", value=True,
                                           help="WAJIB ON agar tidak bocor ke visualisasi"),
        normalize_slang=st.sidebar.checkbox("Normalisasi Slang/Typo", value=True,
                                             help="Ubah kata gaul ke bentuk baku"),
        remove_stopwords=st.sidebar.checkbox("Hapus Stopwords", value=True,
                                              help="Hapus kata umum (yang, dan, di...)"),
        apply_stemming=st.sidebar.checkbox("Lakukan Stemming", value=True,
                                            help="Ubah ke bentuk dasar (Sastrawi)"),
        compute_sentiment=False  # Will be done in lexicon labeling page
    )
    
    st.sidebar.warning("⚠️ English Removal harus ON untuk mencegah kebocoran ke WordCloud!")
    
    with st.expander("ℹ️ Penjelasan Pipeline"):
        st.markdown("""
        **Mengapa urutan ini penting?**
        
        1. **Normalisasi sebelum English Removal**: Agar slang Inggris (e.g., "thx") dinormalisasi ke Indonesia ("terima kasih") dulu
        2. **English Removal sebelum Stopword**: Agar stopword Inggris ("the", "and") terhapus
        3. **Semua tahap disimpan**: `text_clean`, `text_normalized`, `text_id_only`, `text_final` → traceable!
        
        **KRITIS**: Visualisasi (WordCloud) dan Lexicon Labeling **WAJIB** pakai kolom `text_final` (hasil akhir pipeline).
        """)
        
    # Preview Section
    st.subheader("🔍 Preview Pipeline")
    test_text = st.text_area(
        "Coba masukkan teks (campuran Indo-Inggris):", 
        value="Saya sangat happy dengan friend lockdown yall milu dibawabawa fuck aplikasi ini bagus banget!"
    )
    
    if st.button("Preview Hasil"):
        result = run_pipeline(test_text, config)
        
        st.markdown("### Hasil Pipeline:")
        
        # Show each stage
        st.markdown("**📝 Text Raw:**")
        st.code(result.text_raw)
        
        st.markdown("**🧹 Text Clean:**")
        st.code(result.text_clean)
        
        st.markdown("**📚 Text Normalized:**")
        st.code(result.text_normalized)
        
        st.markdown("**🇮🇩 Text ID Only (English Removed):**")
        st.code(result.text_id_only)
        st.caption(f"Dihapus {result.english_removed_count} token Inggris ({result.english_removed_ratio:.1%})")
        if result.english_removed_tokens:
            st.warning(f"Token Inggris dihapus: {', '.join(result.english_removed_tokens)}")
        
        st.markdown("**✅ Text Final (Output Akhir):**")
        st.success(result.text_final)
        st.caption(f"Stopwords dihapus: {result.stopwords_removed_count}")
        
    st.markdown("---")
    
    # Process Full Dataset
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
            
            if st.button("🚀 Jalankan Preprocessing Full", type="primary"):
                with st.spinner("Sedang memproses teks... ini mungkin memakan waktu..."):
                    # Force reload lexicons to pick up any changes
                    from src.lexicon_loader import load_lexicons
                    load_lexicons(force_reload=True)
                    
                    results = []
                    english_removed_total = 0
                    stopwords_removed_total = 0
                    
                    progress_bar = st.progress(0)
                    total_rows = len(df)
                    
                    for i, row in df.iterrows():
                        text = row[text_col]
                        result = run_pipeline(str(text), config)
                        results.append(result)
                        
                        english_removed_total += result.english_removed_count
                        stopwords_removed_total += result.stopwords_removed_count
                        
                        if i % 10 == 0:
                            progress_bar.progress((i + 1) / total_rows)
                            
                    progress_bar.progress(1.0)
                    
                    # Convert results to DataFrame columns
                    df_processed = pd.DataFrame([r.to_dict() for r in results])
                    
                    # Preserve original label if exists
                    if 'label' in df.columns:
                        df_processed['label'] = df['label']
                    
                    st.success("✅ Preprocessing selesai!")
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Data", len(df_processed))
                    with col2:
                        st.metric("Token Inggris Dibuang", english_removed_total)
                    with col3:
                        st.metric("Stopwords Dibuang", stopwords_removed_total)
                    
                    # Warning for empty results
                    empty_count = df_processed['text_final'].apply(lambda x: len(str(x)) == 0).sum()
                    if empty_count > 0:
                        st.warning(f"⚠️ {empty_count} baris menjadi kosong setelah preprocessing.")
                    
                    # Show Comparison Table
                    st.subheader("📊 Perbandingan Sebelum vs Sesudah (20 sample)")
                    comparison_df = df_processed[['text_raw', 'text_final', 'english_removed_count']].head(20)
                    st.table(comparison_df)
                    
                    # Save to session state for next stages
                    st.session_state['df_preprocessed'] = df_processed
                    st.info("💾 Data disimpan ke session state. Lanjutkan ke halaman **Lexicon Labeling**.")
                    
                    # Save Output
                    os.makedirs("data/clean", exist_ok=True)
                    save_path = os.path.join("data/clean", f"preprocessed_{uploaded_file.name}")
                    
                    # Save all columns for full lineage
                    df_processed.to_csv(save_path, index=False)
                    
                    st.success(f"💾 File tersimpan di: **{save_path}**")
                    
                    csv = df_processed.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Hasil Preprocessing",
                        csv,
                        file_name=f"preprocessed_{uploaded_file.name}",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())
