"""
Gmail-based OTP Authentication System
Real Gmail SMTP integration (NOT simulation)

Features:
- Generate 6-digit OTP with expiration
- Send OTP via Gmail SMTP
- Rate limiting (max 3 attempts per hour)
- Email verification required for login
"""

import json
import random
import time
import smtplib
import streamlit as st
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import os


class OTPManager:
    """
    Manage OTP generation, sending, and verification
    """
    
    def __init__(self):
        self.otp_expiry_minutes = 5
        self.max_attempts = 3
        self.rate_limit_window = 3600  # 1 hour in seconds
        self.max_requests_per_hour = 3
        
    def generate_otp(self) -> str:
        """Generate 6-digit OTP"""
        return str(random.randint(100000, 999999))
    
    def get_smtp_config(self) -> Dict[str, str]:
        """
        Get SMTP configuration from multiple sources
        
        Configuration priority:
        1. Session state (UI input - for testing)
        2. Streamlit secrets (for deployed app)
        3. Environment variables (for local dev)
        4. Empty (will use skip OTP mode)
        """
        # Priority 1: Check session state (UI input)
        if 'gmail_config' in st.session_state:
            cfg = st.session_state['gmail_config']
            if cfg.get('smtp_user') and cfg.get('smtp_password'):
                return {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'smtp_user': cfg['smtp_user'],
                    'smtp_password': cfg['smtp_password'],
                    'app_url': 'http://localhost:8501'
                }
        
        # Priority 2: Try Streamlit secrets
        try:
            if hasattr(st, 'secrets') and 'email' in st.secrets:
                smtp_user = st.secrets['email'].get('smtp_user', '')
                smtp_pass = st.secrets['email'].get('smtp_password', '')
                
                # Only use if both are configured (not default values)
                if smtp_user and smtp_pass and smtp_user != 'your_email@gmail.com':
                    return {
                        'smtp_server': 'smtp.gmail.com',
                        'smtp_port': 587,
                        'smtp_user': smtp_user,
                        'smtp_password': smtp_pass,
                        'app_url': st.secrets.get('app', {}).get('url', 'http://localhost:8501')
                    }
        except Exception as e:
            print(f"Could not load from Streamlit secrets: {e}")
        
        # Priority 3: Environment variables
        env_user = os.getenv('GMAIL_SMTP_USER', '')
        env_pass = os.getenv('GMAIL_SMTP_PASSWORD', '')
        if env_user and env_pass:
            return {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'smtp_user': env_user,
                'smtp_password': env_pass,
                'app_url': os.getenv('APP_URL', 'http://localhost:8501')
            }
        
        # Priority 4: Return empty (will skip OTP)
        return {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'smtp_user': '',
            'smtp_password': '',
            'app_url': 'http://localhost:8501'
        }
    
    def send_otp_email(self, to_email: str, otp: str, username: str) -> Tuple[bool, str]:
        """
        Send OTP via Gmail SMTP (REAL EMAIL, NOT SIMULATION)
        
        Args:
            to_email: Recipient email address
            otp: 6-digit OTP code
            username: Username for personalization
            
        Returns:
            (success: bool, message: str)
        """
        config = self.get_smtp_config()
        
        # Validate configuration
        if not config['smtp_user'] or not config['smtp_password']:
            return (False, "Email configuration not found. Please configure Gmail SMTP in Streamlit secrets or environment variables.")
        
        try:
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['From'] = config['smtp_user']
            msg['To'] = to_email
            msg['Subject'] = '🔐 OTP Verification - NLP Sentiment Analysis App'
            
            # HTML email body
            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5;">
                    <div style="max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h2 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">
                            🔐 Verifikasi Email Anda
                        </h2>
                        
                        <p style="font-size: 16px; color: #34495e;">
                            Halo <strong>{username}</strong>,
                        </p>
                        
                        <p style="font-size: 16px; color: #34495e;">
                            Terima kasih telah mendaftar di <strong>NLP Sentiment Analysis App</strong>.<br>
                            Gunakan kode OTP berikut untuk memverifikasi akun Anda:
                        </p>
                        
                        <div style="background-color: #ecf0f1; padding: 20px; text-align: center; border-radius: 5px; margin: 20px 0;">
                            <p style="font-size: 14px; color: #7f8c8d; margin: 0;">Kode OTP Anda:</p>
                            <h1 style="font-size: 48px; color: #e74c3c; margin: 10px 0; letter-spacing: 8px; font-weight: bold;">
                                {otp}
                            </h1>
                            <p style="font-size: 14px; color: #7f8c8d; margin: 0;">
                                ⏰ Kode ini akan kadaluarsa dalam <strong>{self.otp_expiry_minutes} menit</strong>
                            </p>
                        </div>
                        
                        <p style="font-size: 14px; color: #e74c3c; background-color: #fadbd8; padding: 15px; border-radius: 5px; border-left: 4px solid #e74c3c;">
                            <strong>⚠️ Penting:</strong><br>
                            Jangan bagikan kode ini kepada siapapun. Tim kami tidak akan pernah meminta OTP Anda.
                        </p>
                        
                        <p style="font-size: 14px; color: #7f8c8d; margin-top: 30px;">
                            Jika Anda tidak mendaftar di aplikasi ini, abaikan email ini.
                        </p>
                        
                        <hr style="border: none; border-top: 1px solid #ecf0f1; margin: 30px 0;">
                        
                        <p style="font-size: 12px; color: #95a5a6; text-align: center;">
                            Email otomatis dari NLP Sentiment Analysis App<br>
                            {config['app_url']}
                        </p>
                    </div>
                </body>
            </html>
            """
            
            # Plain text fallback
            text_body = f"""
            Verifikasi Email Anda
            
            Halo {username},
            
            Terima kasih telah mendaftar di NLP Sentiment Analysis App.
            
            Kode OTP Anda: {otp}
            
            Kode ini akan kadaluarsa dalam {self.otp_expiry_minutes} menit.
            
            PENTING: Jangan bagikan kode ini kepada siapapun.
            
            Jika Anda tidak mendaftar, abaikan email ini.
            
            ---
            NLP Sentiment Analysis App
            {config['app_url']}
            """
            
            # Attach both versions
            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))
            
            # Connect to Gmail SMTP server
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['smtp_user'], config['smtp_password'])
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            return (True, "OTP berhasil dikirim ke email Anda. Silakan cek inbox (atau folder spam).")
            
        except smtplib.SMTPAuthenticationError:
            return (False, "Autentikasi Gmail gagal. Pastikan App Password sudah dikonfigurasi dengan benar.")
        except smtplib.SMTPException as e:
            return (False, f"Gagal mengirim email: {str(e)}")
        except Exception as e:
            return (False, f"Error: {str(e)}")
    
    def store_otp(self, email: str, otp: str):
        """
        Store OTP in session state with expiration
        """
        if 'otp_storage' not in st.session_state:
            st.session_state['otp_storage'] = {}
        
        expiry_time = time.time() + (self.otp_expiry_minutes * 60)
        
        st.session_state['otp_storage'][email] = {
            'otp': otp,
            'expires_at': expiry_time,
            'attempts': 0,
            'created_at': time.time()
        }
    
    def verify_otp(self, email: str, entered_otp: str) -> Tuple[bool, str]:
        """
        Verify OTP code
        
        Returns:
            (success: bool, message: str)
        """
        if 'otp_storage' not in st.session_state:
            return (False, "Tidak ada OTP yang ditemukan. Silakan minta OTP baru.")
        
        if email not in st.session_state['otp_storage']:
            return (False, "Tidak ada OTP untuk email ini. Silakan minta OTP baru.")
        
        otp_data = st.session_state['otp_storage'][email]
        
        # Check expiration
        if time.time() > otp_data['expires_at']:
            del st.session_state['otp_storage'][email]
            return (False, "OTP sudah kadaluarsa. Silakan minta OTP baru.")
        
        # Check attempts
        if otp_data['attempts'] >= self.max_attempts:
            del st.session_state['otp_storage'][email]
            return (False, f"Terlalu banyak percobaan gagal. Silakan minta OTP baru.")
        
        # Verify OTP
        if entered_otp == otp_data['otp']:
            # Success - clear OTP
            del st.session_state['otp_storage'][email]
            return (True, "✅ Verifikasi berhasil!")
        else:
            # Increment attempts
            otp_data['attempts'] += 1
            remaining = self.max_attempts - otp_data['attempts']
            return (False, f"❌ OTP salah. Sisa percobaan: {remaining}")
    
    def check_rate_limit(self, email: str) -> Tuple[bool, str]:
        """
        Check if email exceeded rate limit for OTP requests
        
        Returns:
            (allowed: bool, message: str)
        """
        if 'otp_rate_limits' not in st.session_state:
            st.session_state['otp_rate_limits'] = {}
        
        current_time = time.time()
        
        if email not in st.session_state['otp_rate_limits']:
            st.session_state['otp_rate_limits'][email] = []
        
        # Remove old timestamps (outside 1 hour window)
        st.session_state['otp_rate_limits'][email] = [
            ts for ts in st.session_state['otp_rate_limits'][email]
            if current_time - ts < self.rate_limit_window
        ]
        
        # Check if limit exceeded
        if len(st.session_state['otp_rate_limits'][email]) >= self.max_requests_per_hour:
            return (False, f"Terlalu banyak permintaan OTP. Silakan coba lagi dalam 1 jam.")
        
        # Add current request
        st.session_state['otp_rate_limits'][email].append(current_time)
        
        return (True, "OK")


def load_users() -> Dict:
    """Load users from users.json"""
    users_file = 'users.json'
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            return json.load(f)
    return {}


def save_users(users: Dict):
    """Save users to users.json"""
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=2)


def register_user(username: str, password: str, email: str, otp_manager: OTPManager) -> Tuple[bool, str, Optional[str]]:
    """
    Register new user and send OTP
    
    Returns:
        (success: bool, message: str, otp: Optional[str])
    """
    # Validate inputs
    if not username or not password or not email:
        return (False, "Semua field harus diisi.", None)
    
    if len(username) < 3:
        return (False, "Username minimal 3 karakter.", None)
    
    if len(password) < 6:
        return (False, "Password minimal 6 karakter.", None)
    
    if '@' not in email or '.' not in email:
        return (False, "Format email tidak valid.", None)
    
    # Check if user exists
    users = load_users()
    if username in users:
        return (False, "Username sudah terdaftar.", None)
    
    # Check email already used
    for user_data in users.values():
        if user_data.get('email') == email:
            return (False, "Email sudah terdaftar.", None)
    
    # Check rate limit
    allowed, msg = otp_manager.check_rate_limit(email)
    if not allowed:
        return (False, msg, None)
    
    # Generate and send OTP
    otp = otp_manager.generate_otp()
    success, msg = otp_manager.send_otp_email(email, otp, username)
    
    if not success:
        return (False, msg, None)
    
    # Store OTP
    otp_manager.store_otp(email, otp)
    
    # Store user temporarily (will be marked verified after OTP)
    users[username] = {
        'password': password,  # In production, use bcrypt.hashpw()
        'email': email,
        'verified': False,
        'created_at': datetime.now().isoformat()
    }
    save_users(users)
    
    return (True, msg, otp)


def verify_user_email(username: str, email: str, otp: str, otp_manager: OTPManager) -> Tuple[bool, str]:
    """
    Verify user email with OTP
    
    Returns:
        (success: bool, message: str)
    """
    # Verify OTP
    success, msg = otp_manager.verify_otp(email, otp)
    
    if not success:
        return (False, msg)
    
    # Mark user as verified
    users = load_users()
    if username in users:
        users[username]['verified'] = True
        users[username]['verified_at'] = datetime.now().isoformat()
        save_users(users)
        return (True, "✅ Email berhasil diverifikasi! Silakan login.")
    else:
        return (False, "User tidak ditemukan.")


def login_user(username: str, password: str) -> Tuple[bool, str]:
    """
    Login user with verification check
    
    Returns:
        (success: bool, message: str)
    """
    users = load_users()
    
    if username not in users:
        return (False, "Username tidak ditemukan.")
    
    user_data = users[username]
    
    # Check email verification
    if not user_data.get('verified', False):
        return (False, "Email belum diverifikasi. Silakan verifikasi email Anda terlebih dahulu.")
    
    # Check password
    if user_data['password'] != password:  # In production, use bcrypt.checkpw()
        return (False, "Password salah.")
    
    return (True, f"Login berhasil! Selamat datang, {username}.")
