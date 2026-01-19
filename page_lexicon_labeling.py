import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from src.sentiment_lexicon import load_sentiment_lexicons, compute_sentiment, validate_lexicons

def show_labeling():
    st.title("🏷️ Lexicon Based Labeling")
    
    st.markdown("""
    **Lexicon Based Labeling** menggunakan kamus kata positif dan negatif untuk menentukan sentimen.
    
    **Cara Kerja (Updated):**
    - Menggunakan **500+ kata** per kategori (positif, negatif, netral)
    - Deteksi **negasi** (tidak bagus → negatif, bukan jelek → positif)
    - Deteksi **intensifier** (sangat bagus → boost skor)
    - Label ditentukan: Positif (score>0), Negatif (score<0), Netral (score==0)
    
    ⚠️ **PENTING**: Dataset harus dari halaman **Text Processing** dengan kolom `text_final` (sudah bersih dari Bahasa Inggris)
    """)
    
    # Show lexicon stats
    with st.expander("📊 Statistik Kamus Sentimen"):
        lex = load_sentiment_lexicons()
        col1, col2,col3 = st.columns(3)
        with col1:
            st.metric("Kata Positif", len(lex.positive))
        with col2:
            st.metric("Kata Negatif", len(lex.negative))
        with col3:
            st.metric("Kata Netral", len(lex.neutral))
        
        st.caption(f"Negasi: {len(lex.negation)} kata | Intensifier: {len(lex.intensifier)} kata")
        
        # Validate lexicons
        validation = validate_lexicons(lex)
        if validation.is_valid:
            st.success("✅ Kamus valid, tidak ada konflik")
        else:
            st.warning(f"⚠️ Ditemukan {len(validation.conflicts)} konflik kata!")
            with st.expander("Lihat Konflik"):
                for word, sources in validation.conflicts[:10]:
                    st.write(f"- `{word}` muncul di: {', '.join(sources)}")
    
    # Check for preprocessed data in session state
    if 'df_preprocessed' in st.session_state:
        st.info("✅ Data sudah ada di session state (dari Text Processing)")
        df = st.session_state['df_preprocessed']
        use_session = st.checkbox("Gunakan data dari session?", value=True)
        
        if not use_session:
            df = None
    else:
        df = None
    
    # If not in session, load from file
    if df is None:
        clean_dir = "data/clean"
        if not os.path.exists(clean_dir):
            st.warning("Folder data/clean belum ada. Silakan lakukan Text Processing terlebih dahulu.")
            return
            
        files = [f for f in os.listdir(clean_dir) if f.endswith(".csv")]
        if not files:
            st.warning("Belum ada file bersih. Silakan lakukan Text Processing terlebih dahulu.")
            return
            
        selected_file = st.selectbox("Pilih File Dataset yang Sudah Dipreprocess:", files)
        
        if st.button("Load Dataset"):
            df = pd.read_csv(os.path.join(clean_dir, selected_file))
            st.session_state['df_lb'] = df
            st.success(f"✅ Loaded {selected_file}")

    if df is None and 'df_lb' in st.session_state:
        df = st.session_state['df_lb']
    
    if df is not None:
        st.dataframe(df.head())
        
        # Check for text_final column (critical!)
        if 'text_final' not in df.columns:
            st.error("""
            ❌ **Kolom `text_final` tidak ditemukan!**
            
            Dataset ini belum melalui pipeline baru. Silakan:
            1. Kembali ke halaman **Text Processing**
            2. Upload dataset dan jalankan preprocessing
            3. Kembali ke halaman ini
            
            (Kolom `text_final` adalah output AKHIR pipeline yang sudah bebas Bahasa Inggris)
            """)
            return
        
        st.markdown("---")
        st.subheader("Jalankan Pelabelan Lexicon")
        
        if st.button("🏷️ Label Sentimen (Gunakan text_final)", type="primary"):
            with st.spinner("Sedang melabeli dengan sentiment scoring..."):
                lex = load_sentiment_lexicons()
                
                # Apply sentiment labeling to text_final
                results = []
                for _, row in df.iterrows():
                    text_final = str(row['text_final'])
                    tokens = text_final.split()  # Already tokenized and cleaned
                    
                    sentiment_result = compute_sentiment(tokens, lex)
                    results.append(sentiment_result)
                
                # Add sentiment columns
                df['pos_count'] = [r.pos_count for r in results]
                df['neg_count'] = [r.neg_count for r in results]
                df['neu_count'] = [r.neu_count for r in results]
                df['sentiment_score'] = [r.sentiment_score for r in results]
                df['label'] = [r.sentiment_label for r in results] # Standardized to 'label'
                df['sentiment_details'] = [r.details for r in results]
                
                st.session_state['df_labeled'] = df
                st.success("✅ Pelabelan selesai!")
        
        if 'df_labeled' in st.session_state:
            df_final = st.session_state['df_labeled']
            
            # Show results
            st.subheader("Hasil Pelabelan:")
            st.dataframe(df_final[['text_final', 'pos_count', 'neg_count', 'sentiment_score', 'label']].head(10))
            
            # Distribution
            st.subheader("Distribusi Label:")
            label_counts = df_final['label'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Jumlah per Label:**")
                st.write(label_counts)
            
            with col2:
                st.write("**Persentase:**")
                st.write((label_counts / len(df_final) * 100).round(2).astype(str) + '%')
            
            # Save to data/labeled
            st.markdown("---")
            st.subheader("Simpan Hasil")
            
            save_filename = st.text_input("Nama file output:", "labeled_dataset.csv")
            
            if st.button("💾 Simpan ke data/labeled"):
                os.makedirs("data/labeled", exist_ok=True)
                save_path = os.path.join("data/labeled", save_filename)
                df_final.to_csv(save_path, index=False)
                st.success(f"✅ File tersimpan di: **{save_path}**")
            
            # Download
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Hasil Labeling",
                csv,
                file_name=save_filename,
                mime="text/csv"
            )
            
            # Visualizations
            st.markdown("---")
            st.subheader("📊 Visualisasi")
            
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                st.write("**Distribusi Label (Bar Chart)**")
                fig1, ax1 = plt.subplots()
                label_counts.plot(kind='bar', ax=ax1, color=['#4CAF50', '#F44336', '#FFC107'])
                ax1.set_xlabel("Label")
                ax1.set_ylabel("Jumlah")
                plt.xticks(rotation=45)
                st.pyplot(fig1)
            
            with col_v2:
                st.write("**Distribusi Label (Pie Chart)**")
                fig2, ax2 = plt.subplots()
                label_counts.plot.pie(autopct='%1.1f%%', ax=ax2, colors=['#4CAF50', '#F44336', '#FFC107'])
                ax2.set_ylabel("")
                st.pyplot(fig2)
            
            # WordClouds (CRITICAL: Use text_final, NOT cleaned_text)
            st.subheader("☁️ WordCloud per Sentimen (Dari text_final)")
            st.caption("WordCloud dibuat dari `text_final` yang sudah bebas Bahasa Inggris")
            
            sentiments = df_final['label'].unique()
            
            cols_wc = st.columns(min(len(sentiments), 3))
            for idx, sentiment in enumerate(sorted(sentiments)):
                with cols_wc[idx % 3]:
                    st.write(f"**{str(sentiment).capitalize()}**")
                    subset = df_final[df_final['label'] == sentiment]
                    
                    # CRITICAL: Use text_final column
                    text_wc = " ".join(subset['text_final'].astype(str).tolist())
                    
                    if text_wc.strip():
                        wc = WordCloud(width=400, height=300, background_color='white').generate(text_wc)
                        fig_wc, ax_wc = plt.subplots()
                        ax_wc.imshow(wc.to_image(), interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    else:
                        st.info("Tidak ada teks")
