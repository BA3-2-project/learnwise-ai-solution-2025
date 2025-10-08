import streamlit as st
import pandas as pd
import numpy as np
import os
from db_manager import DBManager
from data_processor import DataProcessor
from recommendation_engine import RecommendationEngine
from performance_predictor import PerformancePredictor
from student_dashboard import show_student_dashboard
from educator_dashboard import show_educator_dashboard
import plotly.express as px
import plotly.graph_objects as go
from chatbot_tutor import show_tutor_chatbot 

# Initialize session state
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
    
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DBManager()

def main():
    st.set_page_config(
        page_title="AI-Powered Personalized Learning System",
        page_icon="🎓",
        layout="wide"
    )
    
    # Initialize database and load sample data if needed
    with st.spinner("Connecting to the learning system..."):
        db_manager = st.session_state.db_manager
        db_manager.initialize_database()
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("🎓 Learning System")
        
        if st.session_state.user_type is None:
            st.subheader("Login")
            user_type = st.selectbox("Select User Type", ["Student", "Educator", "Virtual Tutor"])
            
            if user_type == "Student":
                students = db_manager.get_all_students()
                if not students.empty:
                    student_names = students['name'].tolist()
                    selected_student = st.selectbox("Select Student", student_names)
                    if st.button("Login as Student"):
                        student_id = students[students['name'] == selected_student]['student_id'].iloc[0]
                        st.session_state.user_type = "Student"
                        st.session_state.user_id = student_id
                        st.success("Login successful! 🎉")
                        st.balloons()
                        st.rerun()
                else:
                    st.warning("No students found. Please contact your administrator.")
            
            elif user_type == "Educator":
                if st.button("Login as Educator"):
                    st.session_state.user_type = "Educator"
                    st.success("Logged in as Educator")
                    st.rerun()
                    
            elif user_type == "Virtual Tutor":
                if st.button("Access Virtual Tutor"):
                    st.session_state.user_type = "Get Help From Virtual Tutor"
                    st.success("Accessing Virtual Tutor")
                    st.rerun()
                    show_tutor_chatbot() 
        
        else:
            st.success(f"Logged in to: {st.session_state.user_type}")
            if st.button("Logout"):
                st.session_state.user_type = None
                st.session_state.user_id = None
                st.rerun()

            # Only show page selection if logged in
           # page = st.sidebar.selectbox("Go to...", ["Virtual Tutor"])
           # if page == "Virtual Tutor":
              #  show_tutor_chatbot()

    # Main content based on user type (only shown if not using sidebar navigation)
    if st.session_state.user_type is None:
        show_welcome_page()
    else:
        # Show dashboards in main area, not sidebar
        if st.session_state.user_type == "Student":
            show_student_dashboard()
        elif st.session_state.user_type == "Educator":
            show_educator_dashboard()
        elif st.session_state.user_type == "Get Help From Virtual Tutor":
            show_tutor_chatbot()

def show_welcome_page():
    st.title("🎓 AI-Powered Personalized Learning System")
    st.markdown("""
    Welcome to the AI-Powered Personalized Learning and Educational Content Recommendation System!
    
    ## Features:
    - 📊 **Personalized Learning Paths**: AI-driven content recommendations based on your learning style and progress
    - 📈 **Progress Tracking**: Real-time monitoring of your learning journey
    - 🎯 **Performance Prediction**: Advanced analytics to predict and improve learning outcomes
    - 👨‍🏫 **Educator Dashboard**: Comprehensive tools for teachers to monitor and support students
    - 📚 **Content Management**: Curated educational resources tailored to individual needs
    
    ## How it works:
    1. **Student Profiling**: The system analyzes your learning behavior and preferences
    2. **AI Recommendations**: Machine learning algorithms suggest the best content for you
    3. **Progress Monitoring**: Track your performance and identify areas for improvement
    4. **Adaptive Learning**: The system continuously adapts to your learning style
    
    Please select your user type from the sidebar to get started!
    """)
    
    # Display system statistics
    db_manager = st.session_state.db_manager
    students_count = len(db_manager.get_all_students())
    content_count = len(db_manager.get_all_content())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Students", students_count)
    with col2:
        st.metric("Learning Resources", content_count)
    with col3:
        st.metric("AI Models", "3 Active")

def show_student_dashboard():
    st.title("📚 Student Dashboard")
    
    db_manager = st.session_state.db_manager
    student_id = st.session_state.user_id
    
    # Get student information
    student_info = db_manager.get_student_profile(student_id)
    if student_info.empty:
        st.error("Student profile not found.")
        return
    
    student_name = student_info['name'].iloc[0]
    st.subheader(f"Welcome back, {student_name}!")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Recommendations", "📈 Progress", "👤 Profile"])
    
    with tab1:
        show_student_overview(student_id)
    
    with tab2:
        show_recommendations(student_id)
    
    with tab3:
        show_progress_tracking(student_id)
    
    with tab4:
        show_student_profile(student_id)

def show_student_overview(student_id):
    st.subheader("📚 Your Learning Overview")
    
    # Get progress data
    progress_data = st.session_state.db_manager.get_student_progress(student_id)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    if not progress_data.empty:
        # Convert timestamp to datetime objects for calculations
        progress_data['timestamp'] = pd.to_datetime(progress_data['timestamp'], errors='coerce')
        
        # Now find the most recent activity
        recent_activity = progress_data['timestamp'].max()
        avg_score = progress_data['score'].mean()
        total_time_spent = progress_data['time_spent'].sum()
        
        # Display key metrics
        with col1:
            st.metric("Average Score", f"{avg_score:.1f}")
        with col2:
            st.metric("Total Time Spent", f"{total_time_spent} min")
        with col3:
            st.metric("Activities Completed", len(progress_data))
        with col4:
            st.metric("Last Activity", recent_activity.strftime('%Y-%m-%d') if pd.notnull(recent_activity) else "N/A")
            
        # Display learning style
        student_info = st.session_state.db_manager.get_student_profile(student_id)
        learning_style = student_info['learning_style'].iloc[0] if not student_info.empty else "N/A"
        st.info(f"💡 Your learning style is: **{learning_style}**")
        
    else:
        st.info("No learning data available yet. Start a new learning activity!")
    
    # Recent activity chart
    if not progress_data.empty:
        st.subheader("Recent Performance")
        fig = px.line(progress_data, x='timestamp', y='score', 
                     title='Score Trend Over Time',
                     labels={'score': 'Score (%)', 'timestamp': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Get New Recommendations", use_container_width=True):
            st.success("New recommendations generated!")
    
    with col2:
        if st.button("📝 Take Practice Quiz", use_container_width=True):
            st.info("Redirecting to practice quiz...")
    
    with col3:
        if st.button("📊 View Detailed Progress", use_container_width=True):
            st.info("Loading detailed progress report...")

def show_recommendations(student_id):
    st.subheader("🎯 Personalized Recommendations")
    
    db_manager = st.session_state.db_manager
    rec_engine = RecommendationEngine(db_manager)
    
    # Get recommendations
    recommendations = rec_engine.get_recommendations(student_id, limit=10)
    
    if not recommendations.empty:
        st.info(f"Found {len(recommendations)} personalized recommendations for you!")
        
        for idx, row in recommendations.iterrows():
            with st.expander(f"📚 {row['title']} - {row['subject']} ({row['difficulty_level']})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Type:** {row['content_type']}")
                    st.write(f"**Description:** {row['description']}")
                    st.write(f"**Estimated Time:** {row['duration_minutes']} minutes")
                    st.write(f"**Prerequisites:** {row['prerequisites']}")
                
                with col2:
                    if st.button(f"Start Learning", key=f"start_{idx}"):
                        # Record interaction
                        db_manager.record_content_interaction(
                            student_id, row['content_id'], 'started'
                        )
                        st.success("Started learning! Progress will be tracked.")
                    
                    if st.button(f"Mark Complete", key=f"complete_{idx}"):
                        # Record completion with random score for demo
                        score = np.random.randint(70, 100)
                        db_manager.record_progress(
                            student_id, row['content_id'], score, 
                            row['duration_minutes']
                        )
                        st.success(f"Completed with score: {score}%")
                        st.rerun()
    else:
        st.warning("No recommendations available. Complete some content to get personalized suggestions!")

def show_progress_tracking(student_id):
    st.subheader("📈 Progress Tracking")
    
    db_manager = st.session_state.db_manager
    progress_data = db_manager.get_student_progress(student_id)
    
    
    
    if not progress_data.empty:
        # Performance by subject
        if 'subject' in progress_data.columns:
            subject_performance = progress_data.groupby('subject')['score'].agg(['mean', 'count']).reset_index()
            subject_performance.columns = ['Subject', 'Average Score', 'Completed Items']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(subject_performance, x='Subject', y='Average Score',
                           title='Average Score by Subject')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(subject_performance, values='Completed Items', names='Subject',
                           title='Content Completion by Subject')
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed progress table
        st.subheader("Detailed Progress History")
        display_progress = progress_data[['title', 'subject', 'score', 'time_spent', 'timestamp']].copy()
        display_progress['timestamp'] = pd.to_datetime(display_progress['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
        st.dataframe(display_progress, use_container_width=True)
        
        # Performance prediction
        predictor = PerformancePredictor(db_manager)
        prediction = predictor.predict_performance(student_id)
        
        if prediction is not None:
            st.subheader("🔮 Performance Prediction")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Next Score", f"{prediction:.1f}%")
            
            with col2:
                if prediction >= 80:
                    st.success("Excellent trajectory! Keep up the great work!")
                elif prediction >= 60:
                    st.info("Good progress. Consider focusing on challenging areas.")
                else:
                    st.warning("May need additional support. Recommended to review fundamentals.")
    else:
        st.info("No progress data available yet. Start learning to see your progress!")

def show_student_profile(student_id):
    st.subheader("👤 Student Profile")
    
    db_manager = st.session_state.db_manager
    student_info = db_manager.get_student_profile(student_id)
    
    if not student_info.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Personal Information**")
            st.write(f"Name: {student_info['name'].iloc[0]}")
            st.write(f"Grade Level: {student_info['grade_level'].iloc[0]}")
            st.write(f"Learning Style: {student_info['learning_style'].iloc[0]}")
            st.write(f"Preferred Subjects: {student_info['preferred_subjects'].iloc[0]}")
        
        with col2:
            st.write("**Learning Preferences**")
            
            # Update learning preferences
            new_learning_style = st.selectbox(
                "Learning Style",
                ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"],
                index=["Visual", "Auditory", "Kinesthetic", "Reading/Writing"].index(student_info['learning_style'].iloc[0])
            )
            
            new_preferred_subjects = st.text_input(
                "Preferred Subjects (comma separated)",
                value=student_info['preferred_subjects'].iloc[0]
            )
            
            if st.button("Update Profile"):
                db_manager.update_student_profile(
                    student_id, new_learning_style, new_preferred_subjects
                )
                st.success("Profile updated successfully!")
                st.rerun()
    else:
        st.warning("Student profile not found.")

def show_educator_dashboard():
    st.title("👨‍🏫 Educator Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "👥 Students", "📚 Content", "📈 Analytics"])
    
    with tab1:
        show_educator_overview()
    
    with tab2:
        show_student_management()
    
    with tab3:
        show_content_management()
    
    with tab4:
        show_analytics()

def show_educator_overview():
    st.subheader("System Overview")
    
    db_manager = st.session_state.db_manager
    
    # Key metrics
    students = db_manager.get_all_students()
    content = db_manager.get_all_content()
    all_progress = db_manager.get_all_progress()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(students))
    
    with col2:
        st.metric("Learning Resources", len(content))
    
    with col3:
        if not all_progress.empty:
            avg_performance = all_progress['score'].mean()
            st.metric("Avg Performance", f"{avg_performance:.1f}%")
        else:
            st.metric("Avg Performance", "No data")
    
    with col4:
        active_students = len(all_progress['student_id'].unique()) if not all_progress.empty else 0
        st.metric("Active Students", active_students)
    
    # Recent activity overview
    if not all_progress.empty:
        st.subheader("Recent Learning Activity")
        
        # Activity by day
        all_progress['date'] = pd.to_datetime(all_progress['timestamp'], errors='coerce').dt.date
        daily_activity = all_progress.groupby('date').size().reset_index(name='activities')
        
        fig = px.line(daily_activity, x='date', y='activities',
                     title='Daily Learning Activities')
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance distribution
        fig = px.histogram(all_progress, x='score', nbins=20,
                         title='Score Distribution Across All Students')
        st.plotly_chart(fig, use_container_width=True)

def show_student_management():
    st.subheader("Student Management")
    
    db_manager = st.session_state.db_manager
    students = db_manager.get_all_students()
    
    if not students.empty:
        # Student overview table
        st.subheader("All Students")
        
        # Get progress summary for each student
        student_summary = []
        for _, student in students.iterrows():
            progress = db_manager.get_student_progress(student['student_id'])
            
            if not progress.empty:
                progress['timestamp'] = pd.to_datetime(progress['timestamp'], errors='coerce')
                avg_score = progress['score'].mean()
                completed_items = len(progress)
                last_activity = progress['timestamp'].max()
                last_activity_str = last_activity.strftime('%Y-%m-%d') if pd.notnull(last_activity) else "Never"
            else:
                avg_score = 0
                completed_items = 0
                last_activity_str = "Never"
            
            student_summary.append({
                'Name': student.get('name', ''),
                'Grade': student.get('grade_level', ''),
                'Learning Style': student.get('learning_style', ''),
                'Avg Score': f"{avg_score:.1f}%",
                'Completed Items': completed_items,
                'Last Activity': last_activity_str
            })
        
        summary_df = pd.DataFrame(student_summary, columns=[
            'Name', 'Grade', 'Learning Style', 'Avg Score', 'Completed Items', 'Last Activity'
        ])
        st.dataframe(summary_df, use_container_width=True)
        
        # Individual student details
        st.subheader("Individual Student Analysis")
        selected_student = st.selectbox(
            "Select Student for Detailed View",
            students['name'].tolist()
        )
        
        if selected_student:
            student_id = students[students['name'] == selected_student]['student_id'].iloc[0]
            progress = db_manager.get_student_progress(student_id)
            
            if not progress.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Score trend
                    fig = px.line(progress, x='timestamp', y='score',
                                title=f'{selected_student} - Score Trend')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Subject performance
                    if 'subject' in progress.columns:
                        subject_perf = progress.groupby('subject')['score'].mean().reset_index()
                        fig = px.bar(subject_perf, x='subject', y='score',
                                   title=f'{selected_student} - Subject Performance')
                        st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations for this student
                rec_engine = RecommendationEngine(db_manager)
                recommendations = rec_engine.get_recommendations(student_id, limit=5)
                
                if not recommendations.empty:
                    st.subheader(f"AI Recommendations for {selected_student}")
                    rec_display = recommendations[['title', 'subject', 'difficulty_level', 'content_type']]
                    st.dataframe(rec_display, use_container_width=True)
            else:
                st.info(f"No activity data available for {selected_student}")
    else:
        st.warning("No students found in the system.")

def show_content_management():
    st.subheader("Content Management")
    
    db_manager = st.session_state.db_manager
    content = db_manager.get_all_content()
    
    if not content.empty:
        # Content overview
        col1, col2 = st.columns(2)
        
        with col1:
            # Content by subject
            subject_counts = content['subject'].value_counts()
            fig = px.pie(values=subject_counts.values, names=subject_counts.index,
                        title='Content Distribution by Subject')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Content by difficulty
            difficulty_counts = content['difficulty_level'].value_counts()
            fig = px.bar(x=difficulty_counts.index, y=difficulty_counts.values,
                        title='Content by Difficulty Level')
            st.plotly_chart(fig, use_container_width=True)
        
        # Content table
        st.subheader("All Content")
        display_content = content[['title', 'subject', 'content_type', 'difficulty_level', 'duration_minutes']]
        st.dataframe(display_content, use_container_width=True)
        
        # Add new content form
        with st.expander("Add New Content"):
            with st.form("add_content_form"):
                title = st.text_input("Title")
                subject = st.selectbox("Subject", ["Mathematics", "Science", "English", "History", "Art"])
                content_type = st.selectbox("Content Type", ["Video", "Article", "Quiz", "Interactive"])
                difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])
                duration = st.number_input("Duration (minutes)", min_value=1, value=30)
                description = st.text_area("Description")
                prerequisites = st.text_input("Prerequisites")
                
                if st.form_submit_button("Add Content"):
                    if title and description:
                        # Generate new content ID
                        new_id = f"content_{len(content) + 1}"
                        
                        # Add to database (this would need to be implemented)
                        st.success(f"Content '{title}' added successfully!")
                    else:
                        st.error("Please fill in all required fields.")
    else:
        st.warning("No content found in the system.")

def show_analytics():
    st.subheader("Learning Analytics")
    
    db_manager = st.session_state.db_manager
    all_progress = db_manager.get_all_progress()
    
    if not all_progress.empty:
        # Overall performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_score = all_progress['score'].mean()
            st.metric("System Average Score", f"{avg_score:.1f}%")
        
        with col2:
            total_time = all_progress['time_spent'].sum()
            st.metric("Total Learning Time", f"{total_time:.0f} min")
        
        with col3:
            completion_rate = len(all_progress) / (len(db_manager.get_all_students()) * len(db_manager.get_all_content())) * 100
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        # Advanced analytics
        st.subheader("Performance Analysis")
        
        # Score distribution by subject
        if 'subject' in all_progress.columns:
            fig = px.box(all_progress, x='subject', y='score',
                        title='Score Distribution by Subject')
            st.plotly_chart(fig, use_container_width=True)
        
        # Learning patterns
        all_progress['hour'] = pd.to_datetime(all_progress['timestamp']).dt.hour
        hourly_activity = all_progress.groupby('hour').size().reset_index(name='activities')
        
        fig = px.bar(hourly_activity, x='hour', y='activities',
                    title='Learning Activity by Hour of Day')
        st.plotly_chart(fig, use_container_width=True)
        
        # Student performance ranking
        student_performance = all_progress.groupby('student_id').agg({
            'score': 'mean',
            'time_spent': 'sum'
        }).reset_index()
        
        # Get student names
        students = db_manager.get_all_students()
        student_performance = student_performance.merge(
            students[['student_id', 'name']], on='student_id', how='left'
        )
        
        student_performance = student_performance.sort_values('score', ascending=False)
        
        st.subheader("Student Performance Ranking")
        st.dataframe(
            student_performance[['name', 'score', 'time_spent']].round(2),
            use_container_width=True
        )
    else:
        st.info("No analytics data available yet. Student activity will generate insights.")

if __name__ == "__main__":
    main()
