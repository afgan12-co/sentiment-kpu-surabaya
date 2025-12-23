import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard_utils import load_analysis_results, aggregate_statistics
from datetime import datetime

def show_statistics():
    st.title("Admin Dashboard - KPU Kota Surabaya")
    st.markdown("Ringkasan opini masyarakat terhadap KPU Kota Surabaya pada Pemilu 2024")
    
    # Load data
    results = load_analysis_results()
    stats = aggregate_statistics(results)
    
    # Check if we have data
    if not results:
        st.warning("⚠️ Belum ada data analisis. Silakan jalankan training model Naive Bayes atau SVM terlebih dahulu.")
    
    # Since only Dashboard remains, we don't need Tabs anymore.
    # Just show the Dashboard content directly.
    
    st.subheader("Dashboard Analisis Sentimen")
    
    # 1. Statistics Cards (Overall)
    col1, col2, col3 = st.columns(3)
    sd = stats['sentiment_distribution']
    
    with col1:
        st.container(border=True)
        st.markdown("**Sentimen Positif**")
        st.markdown(f"<h2 style='color: #22c55e;'>{sd['positive']}</h2>", unsafe_allow_html=True)
    
    with col2:
        st.container(border=True)
        st.markdown("**Sentimen Negatif**")
        st.markdown(f"<h2 style='color: #ef4444;'>{sd['negative']}</h2>", unsafe_allow_html=True)
        
    with col3:
        st.container(border=True)
        st.markdown("**Sentimen Netral**")
        st.markdown(f"<h2 style='color: #6b7280;'>{sd['neutral']}</h2>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. Charts
    c_col1, c_col2 = st.columns(2)
    
    with c_col1:
        st.container(border=True)
        st.markdown("**Distribusi Sentimen Keseluruhan**")
        
        # Pie Chart
        if stats['total_data'] > 0:
            fig_pie = px.pie(
                values=[sd['positive'], sd['negative'], sd['neutral']],
                names=['Positif', 'Negatif', 'Netral'],
                color=['Positif', 'Negatif', 'Netral'],
                color_discrete_map={
                    'Positif': '#22c55e',
                    'Negatif': '#ef4444',
                    'Netral': '#6b7280'
                },
                hole=0.4
            )
            fig_pie.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
            # Fix use_container_width deprecation
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data")
    
    with c_col2:
        st.container(border=True)
        st.markdown("**Opini Terkini**")
        
        # Replaced Category Bar Chart with Recent Data List (Total)
        # Since we removed categories, showing recent data here makes sense for "Dashboard" view
        
        # Extract all recent data from all categories or raw results
        # To be clean, let's take the latest 5 from 'results' directly
        
        recent_all = sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
        
        if recent_all:
            # Create a scrolling container or just list them
             for item in recent_all:
                sent = item.get('sentiment', 'neutral').title()
                # Normalize sentiment string if needed (e.g. 'positif' -> 'Positif')
                if sent.lower() == 'positif': 
                     sent = 'Positif'
                     color = '#22c55e'
                     icon = "📈"
                elif sent.lower() == 'negatif': 
                     sent = 'Negatif'
                     color = '#ef4444'
                     icon = "📉"
                else: 
                     sent = 'Netral' 
                     color = '#6b7280'
                     icon = "➖"
                     
                text = item.get('text', '') or item.get('cleaned_text', '')
                
                # Simplified list item
                st.markdown(f"""
                <div style="margin-bottom: 8px; border-bottom: 1px solid #eee; padding-bottom: 4px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:bold; font-size:14px;">{icon} {sent}</span>
                        <span style="color:gray; font-size:10px;">{item.get('timestamp', '')[:10]}</span>
                    </div>
                    <div style="font-size:12px; color:#333; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                        {text[:50]}...
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Belum ada data opini terkini")

    # 3. Summary Section
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("ℹ️ Ringkasan Analisis")
        
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("**Total Data Dianalisis**")
            st.markdown(f"<h3 style='color: #3b82f6;'>{stats['total_data']}</h3>", unsafe_allow_html=True)
            st.caption(f"Terakhir diupdate: {stats['last_updated'][:16].replace('T', ' ')}")
            
        with s_col2:
            st.markdown("**Dominasi Sentimen**")
            # Determine dominant sentiment
            if stats['total_data'] > 0:
                dominant = max(sd, key=sd.get)
                dom_count = sd[dominant]
                dom_pct = (dom_count / stats['total_data']) * 100
                
                color_map = {'positive': '#22c55e', 'negative': '#ef4444', 'neutral': '#6b7280'}
                dom_label = {'positive': 'Positif', 'negative': 'Negatif', 'neutral': 'Netral'}
                
                st.markdown(f"<h3 style='color: {color_map.get(dominant, 'black')};'>{dom_label.get(dominant, dominant).title()} ({dom_pct:.1f}%)</h3>", unsafe_allow_html=True)
            else:
                 st.markdown("-")
