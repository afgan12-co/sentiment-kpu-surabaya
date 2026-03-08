import streamlit as st
import nltk

# Download NLTK resources for deployment (Streamlit Cloud)
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

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
import os

# Create data directories
os.makedirs("data/clean", exist_ok=True)
os.makedirs("data/labeled", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

st.set_page_config(page_title="Sistem Analisis Sentimen", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .stImage img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 50%;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    [data-testid="stSidebar"] * {
        color: #f0f0f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Session State Initialization
if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

# Sidebar Header with Logo - proporsional dengan lebar tetap
with st.sidebar:
    col_pad1, col_logo, col_pad2 = st.columns([1, 3, 1])
    with col_logo:
        st.image("assets/kpu_logo_fix.png", width=130)
    st.markdown(
        "<h2 style='text-align: center; color: #ffcc00; margin-top: 4px; margin-bottom: 0px; font-size: 1rem; font-weight: bold;'>KPU KOTA SURABAYA</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #aaaaaa; font-size: 0.78rem; margin-top: 2px;'>Sistem Analisis Sentimen Pemilu 2024</p>",
        unsafe_allow_html=True,
    )
st.sidebar.markdown("---")

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
    elif menu == "Logout":
        st.session_state['is_logged_in'] = False
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Sistem Skripsi © 2025")
