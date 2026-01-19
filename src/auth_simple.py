"""
Simple Registration without OTP
For testing when Gmail SMTP is not configured
"""

import streamlit as st
import json
import os
from datetime import datetime


def load_users():
    """Load users from users.json"""
    users_file = 'users.json'
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            return json.load(f)
    return {}


def save_users(users):
    """Save users to users.json"""
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=2)


def register_simple(username: str, password: str, email: str):
    """
    Register user without OTP (for testing/demo)
    
    Returns:
        (success: bool, message: str)
    """
    # Validate inputs
    if not username or not password or not email:
        return (False, "Semua field harus diisi.")
    
    if len(username) < 3:
        return (False, "Username minimal 3 karakter.")
    
    if len(password) < 6:
        return (False, "Password minimal 6 karakter.")
    
    if '@' not in email or '.' not in email:
        return (False, "Format email tidak valid.")
    
    # Check if user exists
    users = load_users()
    if username in users:
        return (False, "Username sudah terdaftar.")
    
    # Check email already used
    for user_data in users.values():
        if user_data.get('email') == email:
            return (False, "Email sudah terdaftar.")
    
    # Register user (already verified, no OTP)
    users[username] = {
        'password': password,
        'email': email,
        'verified': True,  # Auto-verified for simple mode
        'created_at': datetime.now().isoformat(),
        'registration_method': 'simple'  # Mark as simple registration
    }
    save_users(users)
    
    return (True, "✅ Registrasi berhasil! Silakan login.")


def show_register_simple():
    """Simple registration form without OTP"""
    st.title("📝 Registrasi Akun")
    
    st.info("""
    **Mode: Registrasi Sederhana**  
    Gmail OTP belum dikonfigurasi. Menggunakan mode registrasi langsung.
    """)
    
    with st.form("simple_registration_form"):
        username = st.text_input("🔑 Username", placeholder="johndoe")
        email = st.text_input("📧 Email", placeholder="your.email@example.com")
        password = st.text_input("🔒 Password", type="password", placeholder="Min. 6 karakter")
        password_confirm = st.text_input("🔒 Konfirmasi Password", type="password", placeholder="Ulangi password")
        
        submitted = st.form_submit_button("✅ Daftar Sekarang", type="primary")
        
        if submitted:
            # Validate inputs
            if password != password_confirm:
                st.error("❌ Password dan  konfirmasi tidak cocok!")
            elif len(password) < 6:
                st.error("❌ Password minimal 6 karakter!")
            elif not email:
                st.error("❌ Email harus diisi!")
            elif not username:
                st.error("❌ Username harus diisi!")
            else:
                success, message = register_simple(username, password, email)
                
                if success:
                    st.success(message)
                    st.balloons()
                    st.markdown("### ➡️ Silakan login di halaman Login")
                else:
                    st.error(f"❌ {message}")
    
    st.markdown("---")
    st.info("💡 **Sudah punya akun?** Silakan login di halaman Login.")
    
    # Gmail configuration help
    with st.expander("⚙️ Cara Aktifkan OTP via Gmail (Opsional)"):
        st.markdown("""
        **Untuk mengaktifkan OTP via Gmail:**
        
        1. **Buat Gmail App Password** di [Google Account Settings](https://myaccount.google.com/apppasswords)
        2. **Edit file** `.streamlit/secrets.toml` di folder aplikasi:
        
        ```toml
        [email]
        smtp_user = "your_email@gmail.com"
        smtp_password = "your_16_char_app_password"
        ```
        
        3. **Restart aplikasi**
        
        Setelah dikonfigurasi, sistem akan otomatis menggunakan OTP via Gmail.
        """)
