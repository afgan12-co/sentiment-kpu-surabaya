import streamlit as st
from page_login import show_login
from page_register import show_register
from page_home import show_home
from page_text_processing import show_text_processing
from page_lexicon_labeling import show_labeling
from page_tfidf import show_tfidf
from page_naive_bayes import show_naive_bayes
from page_svm import show_svm
from page_evaluation import show_evaluation
from page_visualization import show_visualization
from page_statistics import show_statistics
import os

# Create data directories
os.makedirs("data/clean", exist_ok=True)
os.makedirs("data/labeled", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

st.set_page_config(page_title="Sistem Analisis Sentimen", page_icon="📊", layout="wide")

# Session State Initialization
if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

# Sidebar Navigation Logic
if not st.session_state['is_logged_in']:
    st.sidebar.title("Navigasi")
    choice = st.sidebar.radio("Menu", ["Login", "Registrasi"])
    
    if choice == "Login":
        show_login()
    elif choice == "Registrasi":
        show_register()
else:
    st.sidebar.title(f"Hi, {st.session_state.get('username', 'User')}")
    menu = st.sidebar.radio("Menu Utama", [
        "Halaman Utama", 
        "Text Processing", 
        "Lexicon Labeling",
        "Pembobotan TF-IDF",
        "Klasifikasi Naive Bayes",
        "Klasifikasi SVM",
        "Evaluasi Model",
        "Visualisasi Hasil",
        "Dashboard Statistics",
        "Logout"
    ])
    
    if menu == "Halaman Utama":
        show_home()
    elif menu == "Text Processing":
        show_text_processing()
    elif menu == "Lexicon Labeling":
        show_labeling()
    elif menu == "Pembobotan TF-IDF":
        show_tfidf()
    elif menu == "Klasifikasi Naive Bayes":
        show_naive_bayes()
    elif menu == "Klasifikasi SVM":
        show_svm()
    elif menu == "Evaluasi Model":
        show_evaluation()
    elif menu == "Visualisasi Hasil":
        show_visualization()
    elif menu == "Dashboard Statistics":
        show_statistics()
    elif menu == "Logout":
        st.session_state['is_logged_in'] = False
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Sistem Skripsi © 2025")
