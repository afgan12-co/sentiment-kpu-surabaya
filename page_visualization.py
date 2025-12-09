import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

def create_wordcloud(text_list):
    """Create WordCloud from list of texts"""
    text = " ".join(text_list)
    if not text.strip():
        return None
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wc

def show_visualization():
    st.title("📊 Visualisasi Hasil Analisis Sentimen")

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
        df_nb['pred'].value_counts().plot(kind='bar', ax=ax1, color=['#4CAF50', '#FFC107', '#F44336'])
        ax1.set_xlabel("Sentimen")
        ax1.set_ylabel("Jumlah")
        ax1.set_title("Distribusi Prediksi NB")
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        st.caption("Distribusi hasil prediksi menggunakan Naive Bayes")
    
    with col2:
        st.subheader("SVM")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        df_svm['pred'].value_counts().plot(kind='bar', ax=ax2, color=['#4CAF50', '#FFC107', '#F44336'])
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

    # 3. WordClouds per Sentiment - Use original dataset
    st.markdown("---")
    st.markdown("## ☁️ WordCloud per Sentimen")
    
    # Check if we have the original dataset with text
    if 'tfidf_dataset' in st.session_state:
        df_original = st.session_state['tfidf_dataset']
        
        # Create wordclouds based on true labels
        st.info("WordCloud dibuat berdasarkan label asli dari dataset")
        
        sentiments = df_original['label'].unique()
        
        if len(sentiments) == 0:
            st.warning("Tidak ada sentimen untuk divisualisasikan")
        else:
            # Create columns for wordclouds
            cols = st.columns(min(len(sentiments), 3))
            
            for idx, sentiment in enumerate(sorted(sentiments)):
                with cols[idx % 3]:
                    st.subheader(f"{sentiment.capitalize()}")
                    text_list = df_original[df_original["label"] == sentiment]["cleaned_text"].tolist()
                    
                    if len(text_list) > 0:
                        text_combined = " ".join([str(t) for t in text_list if isinstance(t, str) and t.strip()])
                        if text_combined.strip():
                            wc = create_wordcloud([text_combined])
                            if wc:
                                fig_wc = plt.figure(figsize=(8, 4))
                                plt.imshow(wc, interpolation='bilinear')
                                plt.axis("off")
                                plt.title(f"WordCloud {sentiment.capitalize()}")
                                st.pyplot(fig_wc)
                                st.caption(f"Kata-kata yang sering muncul pada sentimen {sentiment}")
                            else:
                                st.info(f"Tidak ada teks untuk sentimen {sentiment}")
                        else:
                            st.info(f"Tidak ada teks untuk sentimen {sentiment}")
                    else:
                        st.info(f"Tidak ada data untuk sentimen {sentiment}")
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
