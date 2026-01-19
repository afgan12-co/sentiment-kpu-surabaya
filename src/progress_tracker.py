"""
Progress Tracking Module
Tracks user progress through the entire NLP pipeline workflow

This module creates sidebar progress checklist for user visibility 
of their workflow completion status.  
"""

import streamlit as st
from typing import List, Dict


class ProgressTracker:
    """
    Track user progress through NLP workflow
    """
    
    STEPS = [
        {'id': 'upload_data', 'name': '📁 Upload Data', 'description': 'Upload dataset CSV'},
        {'id': 'preprocessing', 'name': '🧹 Text Preprocessing', 'description': 'Clean and normalize text'},
        {'id': 'english_removal', 'name': '🇮🇩 English Removal', 'description': 'Auto-detect and remove English'},
        {'id': 'sentiment_labeling', 'name': '💬 Sentiment Labeling', 'description': 'Lexicon-based sentiment scoring'},
        {'id': 'tfidf', 'name': '📊 TF-IDF Vectorization', 'description': 'Convert text to features'},
        {'id': 'svm_training', 'name': '🤖 SVM Training', 'description': 'Train SVM classifier'},
        {'id': 'smote_applied', 'name': '⚖️ SMOTE (optional)', 'description': 'Balance dataset if imbalanced'},
        {'id': 'visualization', 'name': '📈 Visualization', 'description': 'View results and wordcloud'}
    ]
    
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize progress tracking in session state"""
        if 'progress' not in st.session_state:
            st.session_state['progress'] = {
                step['id']: False for step in self.STEPS
            }
    
    def mark_complete(self, step_id: str):
        """
        Mark a step as complete
        
        Args:
            step_id: ID of the step (e.g., 'preprocessing', 'english_removal')
        """
        if step_id in st.session_state['progress']:
            st.session_state['progress'][step_id] = True
    
    def mark_incomplete(self, step_id: str):
        """Mark a step as incomplete"""
        if step_id in st.session_state['progress']:
            st.session_state['progress'][step_id] = False
    
    def is_complete(self, step_id: str) -> bool:
        """Check if a step is complete"""
        return st.session_state['progress'].get(step_id, False)
    
    def get_completion_percentage(self) -> float:
        """Get overall completion percentage"""
        total_steps = len(self.STEPS)
        completed = sum(1 for step in self.STEPS if self.is_complete(step['id']))
        return (completed / total_steps) * 100 if total_steps > 0 else 0.0
    
    def reset_progress(self):
        """Reset all progress"""
        for step in self.STEPS:
            self.mark_incomplete(step['id'])
    
    def reset_from_step(self, step_id: str):
        """
        Reset progress from a specific step onwards
        Useful when user re-does an earlier step
        """
        start_reset = False
        for step in self.STEPS:
            if step['id'] == step_id:
                start_reset = True
            if start_reset:
                self.mark_incomplete(step['id'])
    
    def render_sidebar(self, title: str = "📋 Progress Tracker"):
        """
        Render progress tracker in sidebar
        
        Args:
            title: Title for the progress section
        """
        with st.sidebar:
            st.markdown(f"### {title}")
            
            # Overall progress bar
            completion_pct = self.get_completion_percentage()
            st.progress(completion_pct / 100)
            st.caption(f"Progres: {completion_pct:.0f}%")
            
            st.markdown("---")
            
            # Individual steps
            for step in self.STEPS:
                is_done = self.is_complete(step['id'])
                icon = "✅" if is_done else "⏸️"
                
                # Different styling for complete vs incomplete
                if is_done:
                    st.markdown(f"{icon} **{step['name']}**")
                else:
                    st.markdown(f"{icon} {step['name']}")
                
                # Show description on hover (via caption)
                if not is_done:
                    st.caption(f"   ↳ {step['description']}")
            
            st.markdown("---")
            
            # Reset button (in expander to avoid accidental clicks)
            with st.expander("🔄 Reset Progress"):
                if st.button("Reset All Progress", type="secondary"):
                    self.reset_progress()
                    st.success("Progress reset!")
                    st.rerun()
    
    def render_page_header(self, current_step_id: str):
        """
        Render page header showing current step context
        
        Args:
            current_step_id: ID of current step
        """
        current_step = next((s for s in self.STEPS if s['id'] == current_step_id), None)
        
        if current_step:
            # Find step number
            step_num = next((i + 1 for i, s in enumerate(self.STEPS) if s['id'] == current_step_id), 0)
            
            st.markdown(f"""
            <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 20px;">
                <div style="font-size: 14px; color: #666;">
                    Step {step_num} of {len(self.STEPS)}
                </div>
                <div style="font-size: 20px; font-weight: bold; margin-top: 5px;">
                    {current_step['name']}
                </div>
                <div style="font-size: 14px; color: #666; margin-top: 5px;">
                    {current_step['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)


def get_tracker() -> ProgressTracker:
    """
    Get or create progress tracker instance
    Singleton pattern using session state
    """
    if 'progress_tracker_instance' not in st.session_state:
        st.session_state['progress_tracker_instance'] = ProgressTracker()
    
    return st.session_state['progress_tracker_instance']


# Example usage in pages:
# from src.progress_tracker import get_tracker
#
# tracker = get_tracker()
# tracker.render_sidebar()
# tracker.mark_complete('preprocessing')
