import streamlit as st
import json
import os
import random

def save_user(name, username, email, password):
    if not os.path.exists('users.json'):
        users = {}
    else:
        with open('users.json', 'r') as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                users = {}
                
    if username in users:
        return False, "Username already exists"
        
    users[username] = {
        "name": name,
        "email": email,
        "password": password
    }
    
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=4)
        
    return True, "Registration successful"

def show_register():
    st.title("Registrasi Akun")
    
    name = st.text_input("Nama Lengkap")
    username = st.text_input("Username")
    email = st.text_input("Gmail (untuk OTP)")
    password = st.text_input("Password (min 5 karakter)", type="password")
    
    # OTP Simulation
    if 'otp_sent' not in st.session_state:
        st.session_state['otp_sent'] = False
        st.session_state['generated_otp'] = None
        
    if st.button("Kirim OTP"):
        if not email:
            st.error("Masukkan email terlebih dahulu.")
        else:
            otp = str(random.randint(1000, 9999))
            st.session_state['generated_otp'] = otp
            st.session_state['otp_sent'] = True
            # Simulate sending email by showing it
            st.info(f"[SIMULASI] Kode OTP Anda dikirim ke {email}: **{otp}**")
            
    if st.session_state['otp_sent']:
        otp_input = st.text_input("Masukkan Kode OTP")
        
        if st.button("Verifikasi & Daftar"):
            if len(password) < 5:
                st.error("Password minimal 5 karakter")
            elif otp_input == st.session_state['generated_otp']:
                success, msg = save_user(name, username, email, password)
                if success:
                    st.success("Registrasi berhasil! Silakan login.")
                    # Reset
                    st.session_state['otp_sent'] = False
                    st.session_state['generated_otp'] = None
                else:
                    st.error(msg)
            else:
                st.error("Kode OTP salah")
