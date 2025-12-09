import streamlit as st

def show_home():
    st.title("Selamat Datang di Analisis Sentimen")
    
    st.markdown("""
    ### Deskripsi Aplikasi
    Aplikasi ini dirancang untuk memudahkan Anda dalam melakukan **analisis sentimen** dari platform media sosial. 
    Dengan pendekatan berbasis **Naive Bayes** dan **Support Vector Machine (SVM)**, Anda dapat mengklasifikasikan opini menjadi **Positif**, **Netral**, atau **Negatif**.
    
    ### Fokus Sistem
    > "Fokus sistem ini adalah melakukan analisis sentimen terhadap teks yang dikumpulkan dari platform X (Twitter)."
    
    ---
    Silakan gunakan menu di sidebar untuk mulai memproses data Anda.
    """)
    
    st.image("https://img.freepik.com/free-vector/sentiment-analysis-concept-illustration_114360-5182.jpg", caption="Sentiment Analysis Illustration", width=400)
