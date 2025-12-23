import streamlit as st
import pandas as pd
from dashboard_utils import load_analysis_results, aggregate_statistics
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

def show_statistics():
    st.title("📊 Dashboard Statistics")
    
    st.markdown("""
    Halaman ini menampilkan statistik hasil analisis sentimen yang sama dengan endpoint API `/statistics`.
    Data ini dapat diakses oleh aplikasi frontend lain melalui REST API.
    """)
    
    # Load data
    try:
        results = load_analysis_results()
        
        if not results:
            st.warning("⚠️ Belum ada data analisis. Silakan lakukan prediksi terlebih dahulu melalui:")
            st.info("""
            - **Menu Klasifikasi Naive Bayes** atau **SVM** untuk training & prediksi
            - **API Endpoint** `/predict` atau `/predict-batch`
            """)
            return
        
        # Generate statistics
        stats = aggregate_statistics(results)
        
        # ============= HEADER METRICS =============
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Data", stats['total_data'])
        
        with col2:
            positive_pct = (stats['sentiment_distribution']['positive'] / stats['total_data'] * 100) if stats['total_data'] > 0 else 0
            st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
        
        with col3:
            st.metric("Last Updated", stats['last_updated'].split('T')[0])
        
        # ============= OVERALL SENTIMENT DISTRIBUTION =============
        st.markdown("---")
        st.subheader("📈 Overall Sentiment Distribution")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Pie Chart
            sentiment_data = stats['sentiment_distribution']
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Positive', 'Negative', 'Neutral'],
                values=[sentiment_data['positive'], sentiment_data['negative'], sentiment_data['neutral']],
                marker_colors=['#4CAF50', '#F44336', '#FFC107']
            )])
            fig_pie.update_layout(title="Sentiment Distribution", height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            # Bar Chart
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=['Positive', 'Negative', 'Neutral'],
                    y=[sentiment_data['positive'], sentiment_data['negative'], sentiment_data['neutral']],
                    marker_color=['#4CAF50', '#F44336', '#FFC107']
                )
            ])
            fig_bar.update_layout(title="Sentiment Counts", height=300, yaxis_title="Count")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # ============= CATEGORY BREAKDOWN =============
        st.markdown("---")
        st.subheader("🏷️ Breakdown by Category")
        
        # Category tabs
        tab1, tab2, tab3 = st.tabs(["📋 Kinerja", "⚖️ Netralitas", "📜 Kebijakan"])
        
        for tab, category in zip([tab1, tab2, tab3], ['kinerja', 'netralitas', 'kebijakan']):
            with tab:
                cat_data = stats['categories'][category]
                
                # Category metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Total", cat_data['total'])
                with col_m2:
                    st.metric("Positive", cat_data['positive'], delta=None, delta_color="normal")
                with col_m3:
                    st.metric("Negative", cat_data['negative'], delta=None, delta_color="inverse")
                with col_m4:
                    st.metric("Neutral", cat_data['neutral'])
                
                if cat_data['total'] > 0:
                    # Sentiment trend
                    if cat_data['sentiment_trend']:
                        st.markdown("**📊 Sentiment Trend**")
                        trend_df = pd.DataFrame(cat_data['sentiment_trend'])
                        
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(
                            x=trend_df['date'], y=trend_df['positive'],
                            name='Positive', line=dict(color='#4CAF50'), mode='lines+markers'
                        ))
                        fig_trend.add_trace(go.Scatter(
                            x=trend_df['date'], y=trend_df['negative'],
                            name='Negative', line=dict(color='#F44336'), mode='lines+markers'
                        ))
                        fig_trend.add_trace(go.Scatter(
                            x=trend_df['date'], y=trend_df['neutral'],
                            name='Neutral', line=dict(color='#FFC107'), mode='lines+markers'
                        ))
                        fig_trend.update_layout(
                            height=300,
                            xaxis_title="Date",
                            yaxis_title="Count",
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Top keywords
                    if cat_data['top_keywords']:
                        st.markdown("**🔤 Top Keywords**")
                        keywords_df = pd.DataFrame(cat_data['top_keywords'])
                        
                        fig_keywords = go.Figure(data=[
                            go.Bar(
                                x=keywords_df['count'][:10],
                                y=keywords_df['word'][:10],
                                orientation='h',
                                marker_color='#2196F3'
                            )
                        ])
                        fig_keywords.update_layout(
                            height=300,
                            xaxis_title="Frequency",
                            yaxis_title="",
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        st.plotly_chart(fig_keywords, use_container_width=True)
                    
                    # Recent data
                    if cat_data['recent_data']:
                        st.markdown("**📝 Recent Data**")
                        recent_df = pd.DataFrame(cat_data['recent_data'])
                        recent_df = recent_df[['id', 'text', 'sentiment', 'confidence', 'timestamp']]
                        st.dataframe(recent_df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"Belum ada data untuk kategori **{category.title()}**")
        
        # ============= RAW DATA TABLE =============
        st.markdown("---")
        st.subheader("📋 All Analysis Results")
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Display filters
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment",
                options=['positive', 'negative', 'neutral', 'positif', 'negatif', 'netral'],
                default=None
            )
        
        with col_f2:
            category_filter = st.multiselect(
                "Filter by Category",
                options=['kinerja', 'netralitas', 'kebijakan'],
                default=None
            )
        
        # Apply filters
        filtered_df = df_results.copy()
        if sentiment_filter:
            filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]
        if category_filter:
            filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
        
        # Display table
        st.write(f"Showing {len(filtered_df)} of {len(df_results)} results")
        
        display_cols = ['id', 'text', 'sentiment', 'category', 'confidence', 'timestamp']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[available_cols].sort_values('id', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # ============= API ENDPOINT INFO =============
        st.markdown("---")
        st.subheader("🔌 API Endpoint")
        
        st.info("""
        **Data statistics ini juga tersedia melalui REST API:**
        
        ```
        GET http://localhost:8000/statistics
        ```
        
        Frontend aplikasi lain dapat mengakses endpoint ini untuk mendapatkan data dashboard secara real-time.
        
        Dokumentasi lengkap: http://localhost:8000/docs
        """)
        
        # Download button
        st.markdown("---")
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Data (CSV)",
            data=csv,
            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Error loading statistics: {str(e)}")
        st.exception(e)
