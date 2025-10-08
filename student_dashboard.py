import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def show_student_dashboard():
    """Main student dashboard interface."""
    st.title("📚 Student Dashboard")
    
    if 'user_id' not in st.session_state or st.session_state.user_id is None:
        st.error("Please log in to access the student dashboard.")
        return
    
    db_manager = st.session_state.db_manager
    student_id = st.session_state.user_id
    
    # Get student information
    with st.spinner("Loading your profile..."):
        student_info = db_manager.get_student_profile(student_id)
    if student_info.empty:
        st.error("Student profile not found.")
        return
    
    student_name = student_info['name'].iloc[0]
    st.subheader(f"Welcome back, {student_name}! 👋")
    
    # Main dashboard content
    show_dashboard_overview(student_id, db_manager)
    
    st.divider()
    
    # Quick actions section
    show_quick_actions(student_id, db_manager)

def show_dashboard_overview(student_id, db_manager):
    """Show overview metrics and charts."""
    
    with st.spinner("Fetching your learning data..."):
        # Get progress data
        progress_data = db_manager.get_student_progress(student_id)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    if not progress_data.empty:
        avg_score = progress_data['score'].mean()
        completed_items = len(progress_data)
        total_time = progress_data['time_spent'].sum()
        
        # Calculate streak (simplified)
        recent_days = (datetime.now() - pd.to_datetime(progress_data['timestamp']).min()).days
        streak_days = min(recent_days, 30)  # Cap at 30 for demo
        
        with col1:
            st.metric(
                "Average Score", 
                f"{avg_score:.1f}%",
                delta=f"{avg_score - 70:.1f}%" if avg_score > 70 else None
            )
        
        with col2:
            st.metric("Completed Activities", completed_items)
        
        with col3:
            st.metric("Learning Hours", f"{total_time/60:.1f}h")
        
        with col4:
            st.metric("Day Streak", f"{streak_days}d")
    else:
        with col1:
            st.metric("Average Score", "No data")
        with col2:
            st.metric("Completed Activities", "0")
        with col3:
            st.metric("Learning Hours", "0h")
        with col4:
            st.metric("Day Streak", "0d")
    
    # Charts section
    if not progress_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Score trend over time
            fig_trend = px.line(
                progress_data.sort_values('timestamp'), 
                x='timestamp', 
                y='score',
                title='📈 Score Trend Over Time',
                labels={'score': 'Score (%)', 'timestamp': 'Date'}
            )
            fig_trend.update_layout(showlegend=False)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Subject performance
            if 'subject' in progress_data.columns:
                subject_avg = progress_data.groupby('subject')['score'].mean().reset_index()
                fig_subjects = px.bar(
                    subject_avg, 
                    x='subject', 
                    y='score',
                    title='📊 Performance by Subject',
                    color='score',
                    color_continuous_scale='viridis'
                )
                fig_subjects.update_layout(showlegend=False)
                st.plotly_chart(fig_subjects, use_container_width=True)
            else:
                st.info("Complete more content to see subject analysis")
        
        # Recent activity table
        st.subheader("📋 Recent Activity")
        recent_activities = progress_data.head(10)[['title', 'subject', 'score', 'timestamp']].copy()
        recent_activities['timestamp'] = pd.to_datetime(recent_activities['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        recent_activities['score'] = recent_activities['score'].round(1).astype(str) + '%'
        
        st.dataframe(
            recent_activities,
            column_config={
                "title": "Content",
                "subject": "Subject", 
                "score": "Score",
                "timestamp": "Completed At"
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("🎯 Start learning to see your progress and analytics here!")

def show_quick_actions(student_id, db_manager):
    """Show quick action buttons and recommendations."""
    
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Get New Recommendations", use_container_width=True):
            st.switch_page("pages/recommendations.py")
            
            with st.spinner("Generating personalized recommendations..."):
                import time
                time.sleep(2)# Simulate a delay
            st.success("New recommendations generated!")
            st.rerun()
    
    with col2:
        if st.button("📈 View Detailed Progress", use_container_width=True):
            st.switch_page("pages/progress_tracking.py")
    
    with col3:
        if st.button("👤 Update Profile", use_container_width=True):
            st.switch_page("pages/student_profile.py")
    
    # Show top recommendations inline
    st.subheader("🎯 Recommended for You")
    
    try:
        from recommendation_engine import RecommendationEngine
        rec_engine = RecommendationEngine(db_manager)
        recommendations = rec_engine.get_recommendations(student_id, limit=3)
        
        if not recommendations.empty:
            for idx, row in recommendations.iterrows():
                with st.expander(f"📚 {row['title']} - {row['subject']} ({row['difficulty_level']})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Type:** {row['content_type']}")
                        st.write(f"**Duration:** {row['duration_minutes']} minutes")
                        st.write(f"**Description:** {row['description']}")
                    
                    with col2:
                        if st.button("Start Learning", key=f"quick_start_{idx}"):
                            # Record interaction
                            db_manager.record_content_interaction(
                                student_id, row['content_id'], 'started'
                            )
                            st.success("Started! Good luck! 🍀")
                        
                        if st.button("Mark Complete", key=f"quick_complete_{idx}"):
                            # Simulate completion with random score
                            import numpy as np
                            score = np.random.randint(75, 95)
                            db_manager.record_progress(
                                student_id, row['content_id'], score, 
                                row['duration_minutes']
                            )
                            st.success(f"Completed! Score: {score}%")
                            st.rerun()
        else:
            st.info("Complete some content to get personalized recommendations!")
            
    except Exception as e:
        st.error(f"Error loading recommendations: {e}")

def show_performance_insights(student_id, db_manager):
    """Show AI-generated performance insights."""
    
    st.subheader("🔮 AI Insights")
    
    try:
        from data_processor import DataProcessor
        from performance_predictor import PerformancePredictor
        
        data_processor = DataProcessor()
        predictor = PerformancePredictor(db_manager)
        
        # Get student data
        progress_data = db_manager.get_student_progress(student_id)
        student_profile = db_manager.get_student_profile(student_id)
        
        # Generate insights
        insights = data_processor.generate_learning_insights(
            student_id, progress_data, student_profile
        )
        
        if insights:
            for insight in insights[:3]:  # Show top 3 insights
                if insight['type'] == 'success':
                    st.success(insight['message'])
                elif insight['type'] == 'warning':
                    st.warning(insight['message'])
                else:
                    st.info(insight['message'])
        
        # Performance prediction
        if predictor.is_trained:
            prediction = predictor.predict_performance(student_id)
            if prediction is not None:
                st.subheader("🔮 Next Performance Prediction")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Next Score", f"{prediction:.1f}%")
                
                with col2:
                    if prediction >= 80:
                        st.success("Excellent trajectory! Keep it up! 🌟")
                    elif prediction >= 60:
                        st.info("Good progress. Stay consistent! 💪")
                    else:
                        st.warning("Consider reviewing fundamentals 📚")
        
    except Exception as e:
        st.error(f"Error generating insights: {e}")

# Additional utility functions for student dashboard
def calculate_progress_percentage(student_id, db_manager):
    """Calculate overall progress percentage."""
    try:
        total_content = len(db_manager.get_all_content())
        completed_content = len(db_manager.get_student_progress(student_id))
        
        if total_content == 0:
            return 0
        
        return min(100, (completed_content / total_content) * 100)
    
    except Exception:
        return 0

def get_learning_streaks(progress_data):
    """Calculate learning streaks from progress data."""
    try:
        if progress_data.empty:
            return 0
        
        # Convert timestamps to dates
        progress_data['date'] = pd.to_datetime(progress_data['timestamp']).dt.date
        unique_dates = sorted(progress_data['date'].unique(), reverse=True)
        
        if not unique_dates:
            return 0
        
        # Calculate consecutive days
        streak = 0
        current_date = datetime.now().date()
        
        for date in unique_dates:
            if (current_date - date).days == streak:
                streak += 1
            else:
                break
        
        return streak
    
    except Exception:
        return 0
