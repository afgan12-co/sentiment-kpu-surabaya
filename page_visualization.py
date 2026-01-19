import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from src.preprocess import EnglishDetector
from src.lexicon_loader import load_lexicons
from src.pipeline import validate_no_english_leakage

def create_wordcloud(text_list):
    """Create WordCloud from list of texts"""
    text = " ".join(text_list)
    if not text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wc

def show_visualization():
    st.title("📊 Visualisasi Hasil Analisis Sentimen")
    
    st.info("""
    **📋 Data Lineage:**
    - WordCloud menggunakan kolom: `text_final` (hasil akhir pipeline, bebas Bahasa Inggris)
    - Label sentimen dari: `sentiment_label` (lexicon-based scoring)
    """)

    if "nb_pred" not in st.session_state or "svm_pred" not in st.session_state:
        st.error("⚠️ Latih model Naïve Bayes dan SVM terlebih dahulu.")
        return

    df_nb = st.session_state["nb_pred"]
    df_svm = st.session_state["svm_pred"]

    # 1. Sentiment Distribution
    st.markdown("## 📈 Distribusi Sentimen")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Naïve Bayes")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        df_nb['pred'].value_counts().plot(kind='bar', ax=ax1, color=['#4CAF50', '#F44336', '#FFC107'])
        ax1.set_xlabel("Sentimen")
        ax1.set_ylabel("Jumlah")
        ax1.set_title("Distribusi Prediksi NB")
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        st.caption("Distribusi hasil prediksi menggunakan Naive Bayes")
    
    with col2:
        st.subheader("SVM")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        df_svm['pred'].value_counts().plot(kind='bar', ax=ax2, color=['#4CAF50', '#F44336', '#FFC107'])
        ax2.set_xlabel("Sentimen")
        ax2.set_ylabel("Jumlah")
        ax2.set_title("Distribusi Prediksi SVM")
        plt.xticks(rotation=45)
        st.pyplot(fig2)
        st.caption("Distribusi hasil prediksi menggunakan SVM")

    # 2. Model Comparison
    st.markdown("---")
    st.markdown("## 🏆 Perbandingan Akurasi Model")
    
    nb_acc = accuracy_score(df_nb["true"], df_nb["pred"])
    svm_acc = accuracy_score(df_svm["true"], df_svm["pred"])
    
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    models = ["Naïve Bayes", "SVM"]
    accuracies = [nb_acc, svm_acc]
    colors = ['#2196F3', '#4CAF50']
    
    bars = ax3.bar(models, accuracies, color=colors)
    ax3.set_ylabel("Accuracy")
    ax3.set_ylim([0, 1])
    ax3.set_title("Perbandingan Accuracy NB vs SVM")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    st.pyplot(fig3)
    st.caption("Perbandingan tingkat akurasi antara Naive Bayes dan SVM")

    # 3. WordClouds per Sentiment - CRITICAL: Use text_final
    st.markdown("---")
    st.markdown("## ☁️ WordCloud per Sentimen")
    
    # Check if we have the original dataset with text_final
    if 'tfidf_dataset' in st.session_state:
        df_original = st.session_state['tfidf_dataset']
        
        # DETERMINE COLUMN: Prioritize text_final, then fallback to cleaned_text
        viz_column = None
        if 'text_final' in df_original.columns:
            viz_column = 'text_final'
            st.success("✅ WordCloud dibuat dari kolom `text_final` (Bebas Bahasa Inggris)")
        elif 'cleaned_text' in df_original.columns:
            viz_column = 'cleaned_text'
            st.warning("⚠️ Kolom `text_final` tidak ditemukan. Menggunakan `cleaned_text` (Mungkin masih ada Bahasa Inggris).")
            st.info("💡 Untuk hasil terbaik, silakan re-process data di halaman Text Processing agar mendapatkan kolom `text_final`.")
        
        if viz_column is None:
            st.error("""
            ❌ **Kolom teks tidak ditemukan di dataset!**
            
            Dataset di session state (`tfidf_dataset`) tidak memiliki kolom `text_final` atau `cleaned_text`.
            
            Silakan:
            1. Kembali ke halaman **Text Processing**
            2. Jalankan ulang preprocessing
            3. Lanjutkan ke TF-IDF agar dataset terbaru terdaftar di sistem.
            """)
            return
        
        # Create wordclouds based on true labels using selected column
        sentiments = df_original['label'].unique()
        
        if len(sentiments) == 0:
            st.warning("Tidak ada sentimen untuk divisualisasikan")
        else:
            # Load English detector once for safety filter and audit
            from src.lexicon_loader import load_lexicons
            from src.auto_english_detector import MultiSignalEnglishDetector
            lexicons = load_lexicons(force_reload=True)
            english_detector = MultiSignalEnglishDetector(
                lexicons['english_words'],
                lexicons['indo_whitelist'],
                lexicons['english_keep']
            )
            
            # Create columns for wordclouds
            cols = st.columns(min(len(sentiments), 3))
            
            # Mapping for display names
            label_map = {
                'positif': 'Sentimen Positif',
                'netral': 'Sentimen Netral',
                'negatif': 'Sentimen Negatif',
                '1': 'Sentimen Positif',
                '0': 'Sentimen Netral',
                '-1': 'Sentimen Negatif',
                 1: 'Sentimen Positif',
                 0: 'Sentimen Netral',
                -1: 'Sentimen Negatif'
            }
            
            for idx, sentiment in enumerate(sorted(sentiments)):
                # Handle numeric or string labels
                display_name = label_map.get(sentiment, label_map.get(str(sentiment), f"Sentimen {str(sentiment).capitalize()}"))
                
                with cols[idx % 3]:
                    st.subheader(display_name)
                    
                    # USE DETECTED COLUMN
                    raw_text_list = df_original[df_original["label"] == sentiment][viz_column].tolist()
                    
                    if len(raw_text_list) > 0:
                        raw_combined = " ".join([str(t) for t in raw_text_list if isinstance(t, str) and t.strip()])
                        
                        # === SAFETY NET: Filter English tokens on-the-fly for WordCloud ===
                        tokens = raw_combined.split()
                        final_tokens = []
                        for token in tokens:
                            is_eng, _ = english_detector.is_english(token)
                            if not is_eng:
                                final_tokens.append(token)
                        
                        text_combined = " ".join(final_tokens)
                        
                        if text_combined.strip():
                            wc = create_wordcloud([text_combined])
                            if wc:
                                fig_wc = plt.figure(figsize=(8, 4))
                                plt.imshow(wc.to_image(), interpolation='bilinear')
                                plt.axis("off")
                                plt.title(display_name)
                                st.pyplot(fig_wc)
                                st.caption(f"Kata-kata yang sering muncul pada {display_name} ({len(final_tokens)} token)")
                            else:
                                st.info(f"Tidak ada teks untuk {display_name}")
                        else:
                            st.info(f"Tidak ada teks Indonesia yang tersisa untuk {display_name}")
                    else:
                        st.info(f"Tidak ada data untuk {display_name}")
            
            # === AUDIT PANEL: English Leakage Detection ===
            st.markdown("---")
            st.subheader("🔍 Audit Panel: English Leakage Detection")
            
            with st.expander("Lihat Hasil Audit"):
                # Note: english_detector and lexicons are initialized above
                
                # Sample tokens from detected column
                all_tokens = []
                for txt in df_original[viz_column].head(200):  # Sample more for audit
                    if isinstance(txt, str):
                        all_tokens.extend(txt.split())
                
                # Validate with detailed report
                leaked_with_methods = []
                for token in all_tokens[:500]: # Check up to 500 tokens
                    is_eng, method = english_detector.is_english(token)
                    if is_eng:
                        leaked_with_methods.append(f"{token} ({method})")
                
                if leaked_with_methods:
                    st.warning(f"⚠️ Terdeteksi {len(leaked_with_methods)} token Inggris yang lolos!")
                    st.write("**Token Inggris & Metode Deteksi:**")
                    st.write(", ".join(list(set(leaked_with_methods))[:50])) # Unique leaked tokens
                    st.error("💡 **Tips**: Jika banyak kata Indonesia yang lolos, tambahkan kata tersebut ke `indo_vocab_whitelist.txt`. Jika kata Inggris lolos, tambahkan ke `english_common.txt`.")
                    
                    if st.button("🔄 Refresh Audit"):
                        st.rerun()
                else:
                    st.success("✅ Tidak ada kebocoran token Bahasa Inggris terdeteksi pada sampel data!")
                
                # Show top tokens
                from collections import Counter
                token_counts = Counter(all_tokens)
                
                st.write("**Top 30 Token Terbanyak:**")
                top_tokens = [f"{word} ({count})" for word, count in token_counts.most_common(30)]
                st.write(", ".join(top_tokens))
                
                # Show removed English tokens if available in df_original
                if 'english_removed_count' in df_original.columns:
                    total_removed = df_original['english_removed_count'].sum()
                    st.info(f"📊 Total token Inggris yang dihapus di preprocessing: **{total_removed:,}**")
    else:
        st.warning("⚠️ Dataset asli tidak ditemukan. WordCloud tidak dapat ditampilkan.")
        st.info("Untuk menampilkan WordCloud, pastikan Anda telah membuat TF-IDF matrix di halaman Pembobotan TF-IDF.")

    # 4. Confusion Matrix Heatmaps
    st.markdown("---")
    st.markdown("## 🔥 Confusion Matrix Heatmaps")
    
    col_cm1, col_cm2 = st.columns(2)
    
    with col_cm1:
        st.subheader("Naive Bayes")
        cm_nb = confusion_matrix(df_nb["true"], df_nb["pred"])
        fig_cm1, ax_cm1 = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues", ax=ax_cm1,
                    xticklabels=sorted(df_nb["true"].unique()),
                    yticklabels=sorted(df_nb["true"].unique()))
        ax_cm1.set_xlabel("Predicted")
        ax_cm1.set_ylabel("Actual")
        ax_cm1.set_title("Confusion Matrix - NB")
        st.pyplot(fig_cm1)
        st.caption("Matriks konfusi Naive Bayes menunjukkan perbandingan prediksi vs aktual")
    
    with col_cm2:
        st.subheader("SVM")
        cm_svm = confusion_matrix(df_svm["true"], df_svm["pred"])
        fig_cm2, ax_cm2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens", ax=ax_cm2,
                    xticklabels=sorted(df_svm["true"].unique()),
                    yticklabels=sorted(df_svm["true"].unique()))
        ax_cm2.set_xlabel("Predicted")
        ax_cm2.set_ylabel("Actual")
        ax_cm2.set_title("Confusion Matrix - SVM")
        st.pyplot(fig_cm2)
        st.caption("Matriks konfusi SVM menunjukkan perbandingan prediksi vs aktual")
