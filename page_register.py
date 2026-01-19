import streamlit as st
import json
import os
from datetime import datetime


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


def save_users(users):
    """Save users to users.json"""
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=2)


def show_register():
    """Simple registration - username and password only"""
    st.title("📝 Registrasi Akun Baru")
    
    st.markdown("""
    Daftarkan akun Anda untuk menggunakan aplikasi analisis sentimen NLP.
    Setelah registrasi berhasil, Anda dapat langsung login.
    """)
    
    with st.form("registration_form"):
        username = st.text_input("🔑 Username", placeholder="Masukkan username Anda")
        password = st.text_input("🔒 Password", type="password", placeholder="Min. 6 karakter")
        password_confirm = st.text_input("🔒 Konfirmasi Password", type="password", placeholder="Ulangi password")
        
        submitted = st.form_submit_button("✅ Daftar Sekarang", type="primary", width='stretch')
        
        if submitted:
            # Validate inputs
            if not username or not password:
                st.error("❌ Username dan password harus diisi!")
            elif len(username) < 3:
                st.error("❌ Username minimal 3 karakter!")
            elif len(password) < 6:
                st.error("❌ Password minimal 6 karakter!")
            elif password != password_confirm:
                st.error("❌ Password dan konfirmasi tidak cocok!")
            else:
                # Check if user exists
                users = load_users()
                if username in users:
                    st.error("❌ Username sudah terdaftar!")
                else:
                    # Register user (no email, no verification)
                    users[username] = {
                        'password': password,
                        'created_at': datetime.now().isoformat()
                    }
                    save_users(users)
                    
                    st.success("✅ Registrasi berhasil! Silakan login.")
                    st.balloons()
                    
                    st.markdown("---")
                    st.info("⏩ **Lanjutkan ke halaman Login** untuk masuk ke aplikasi.")
    
    st.markdown("---")
    st.info("💡 **Sudah punya akun?** Silakan login di halaman Login.")



