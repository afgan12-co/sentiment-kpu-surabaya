import streamlit as st
import json
import os


def load_users():
    """Load users from users.json"""
    users_file = 'users.json'
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}


def show_login():
    st.title("🔐 Login")
    
    st.markdown("""
    Login ke aplikasi NLP Sentiment Analysis untuk mengakses fitur analisis teks.
    """)
    
    with st.form("login_form"):
        username = st.text_input("🔑 Username", placeholder="Masukkan username Anda")
        password = st.text_input("🔒 Password", type="password", placeholder="Masukkan password Anda")
        
        submitted = st.form_submit_button("🚀 Login", type="primary", width='stretch')
        
        if submitted:
            if not username or not password:
                st.error("❌ Username dan password harus diisi!")
            else:
                users = load_users()
                
                if username not in users:
                    st.error("❌ Username tidak ditemukan!")
                elif users[username]['password'] != password:
                    st.error("❌ Password salah!")
                else:
                    # Login successful
                    st.session_state['is_logged_in'] = True
                    st.session_state['username'] = username
                    st.success(f"✅ Login berhasil! Selamat datang, {username}.")
                    st.balloons()
                    st.rerun()
    
    st.markdown("---")
    st.info("💡 **Belum punya akun?** Silakan daftar di halaman Register.")


