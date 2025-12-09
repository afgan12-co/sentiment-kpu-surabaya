import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import lexicon_label, validate_columns

def show_labeling():
    st.title("🏷️ Lexicon Based Labeling")
    
    st.markdown("""
    **Lexicon Based Labeling** menggunakan kamus kata positif dan negatif untuk menentukan sentimen.
    
    **Cara Kerja:**
    - Setiap kata diberi skor positif atau negatif
    - Total skor dihitung untuk setiap teks
    - Label ditentukan: Positif (>0), Negatif (<0), Netral (=0)
    
    ⚠️ **Penting**: Dataset harus sudah melalui preprocessing (memiliki kolom `cleaned_text`)
    """)
    
    # Select clean file
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
        
        # Validate that cleaned_text column exists
        is_valid, error_msg = validate_columns(df, ['cleaned_text'])
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return
        
        st.session_state['df_lb'] = df
        st.success(f"✅ Loaded {selected_file}")

    if 'df_lb' in st.session_state:
        df = st.session_state['df_lb']
        st.dataframe(df.head())
        
        st.markdown("---")
        st.subheader("Jalankan Pelabelan Lexicon")
        
        if st.button("🏷️ Label Sentimen"):
            with st.spinner("Sedang melabeli..."):
                # Apply lexicon labeling
                df['label'] = df['cleaned_text'].apply(lexicon_label)
                
                st.session_state['df_labeled'] = df
                st.success("✅ Pelabelan selesai!")
        
        if 'df_labeled' in st.session_state:
            df_final = st.session_state['df_labeled']
            
            # Show results
            st.subheader("Hasil Pelabelan:")
            st.dataframe(df_final[['cleaned_text', 'label']].head(10))
            
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
            
            save_filename = st.text_input("Nama file output:", f"labeled_{selected_file}")
            
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
                label_counts.plot(kind='bar', ax=ax1, color=['#4CAF50', '#FFC107', '#F44336'])
                ax1.set_xlabel("Label")
                ax1.set_ylabel("Jumlah")
                plt.xticks(rotation=45)
                st.pyplot(fig1)
            
            with col_v2:
                st.write("**Distribusi Label (Pie Chart)**")
                fig2, ax2 = plt.subplots()
                label_counts.plot.pie(autopct='%1.1f%%', ax=ax2, colors=['#4CAF50', '#FFC107', '#F44336'])
                ax2.set_ylabel("")
                st.pyplot(fig2)
            
            # WordClouds
            st.subheader("☁️ WordCloud per Sentimen")
            sentiments = df_final['label'].unique()
            
            cols_wc = st.columns(min(len(sentiments), 3))
            for idx, sentiment in enumerate(sorted(sentiments)):
                with cols_wc[idx % 3]:
                    st.write(f"**{sentiment.capitalize()}**")
                    subset = df_final[df_final['label'] == sentiment]
                    text_wc = " ".join(subset['cleaned_text'].astype(str).tolist())
                    
                    if text_wc.strip():
                        wc = WordCloud(width=400, height=300, background_color='white').generate(text_wc)
                        fig_wc, ax_wc = plt.subplots()
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    else:
                        st.info("Tidak ada teks")
