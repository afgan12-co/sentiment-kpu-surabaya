import streamlit as st
import json
import os

def check_login(username, password):
    if not os.path.exists('users.json'):
        return False
    
    with open('users.json', 'r') as f:
        try:
            users = json.load(f)
        except json.JSONDecodeError:
            users = {}
            
    if username in users and users[username]['password'] == password:
        return True
    return False

def show_login():
    st.title("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if check_login(username, password):
            st.session_state['is_logged_in'] = True
            st.session_state['username'] = username
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")
