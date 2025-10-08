import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from content_management import show_content_management

def show_educator_dashboard():
    """Main educator dashboard interface."""
    st.title("👨‍🏫 Educator Dashboard")
    
    if 'user_type' not in st.session_state or st.session_state.user_type != 'Educator':
        st.error("Access denied. Educator login required.")
        return
    
    db_manager = st.session_state.db_manager
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "👥 Students", "📚 Content", "📈 Analytics"])
    
    with tab1:
        with st.spinner("Loading system overview..."):
            show_system_overview(db_manager)
    
    with tab2:
        with st.spinner("Loading student management data..."):
            show_student_management(db_manager)
    with tab3:
        with st.spinner("Loading content overview..."):
            show_content_overview(db_manager)
    
    with tab4:
        with st.spinner("Generating detailed analytics..."):
            show_detailed_analytics(db_manager)
 

def show_system_overview(db_manager):
    """Show high-level system metrics and overview."""
    
    st.subheader("System Overview")
    
    # Get data
    students_df = db_manager.get_all_students()
    content_df = db_manager.get_all_content()
    progress_df = db_manager.get_all_progress()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Students", 
            len(students_df),
            help="Number of registered students"
        )
    
    with col2:
        st.metric(
            "Learning Resources", 
            len(content_df),
            help="Total available content items"
        )
    
    with col3:
        if not progress_df.empty:
            avg_performance = progress_df['score'].mean()
            st.metric(
                "Avg Performance", 
                f"{avg_performance:.1f}%",
                delta=f"{avg_performance - 75:.1f}%" if avg_performance != 75 else None
            )
        else:
            st.metric("Avg Performance", "No data")
    
    with col4:
        active_students = len(progress_df['student_id'].unique()) if not progress_df.empty else 0
        st.metric(
            "Active Students", 
            active_students,
            help="Students with recent activity"
        )
    
    if not progress_df.empty:
        # Recent activity overview
        st.subheader("📊 Recent Learning Activity")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily activity trend
            progress_df['date'] = pd.to_datetime(progress_df['timestamp']).dt.date
            daily_activity = progress_df.groupby('date').size().reset_index(name='activities')
            daily_activity = daily_activity.tail(14)  # Last 14 days
            
            fig_daily = px.line(
                daily_activity, 
                x='date', 
                y='activities',
                title='Daily Learning Activities (Last 14 Days)',
                markers=True
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with col2:
            # Score distribution
            fig_scores = px.histogram(
                progress_df, 
                x='score', 
                nbins=20,
                title='Score Distribution Across All Students',
                labels={'score': 'Score (%)', 'count': 'Number of Activities'}
            )
            fig_scores.update_layout(showlegend=False)
            st.plotly_chart(fig_scores, use_container_width=True)
        
        # Performance by subject
        if 'subject' in progress_df.columns:
            st.subheader("📚 Performance by Subject")
            
            subject_stats = progress_df.groupby('subject').agg({
                'score': ['mean', 'count', 'std']
            }).round(2)
            subject_stats.columns = ['Average Score', 'Total Activities', 'Score Std Dev']
            subject_stats = subject_stats.sort_values('Average Score', ascending=False)
            
            # Display as metrics
            cols = st.columns(len(subject_stats))
            for i, (subject, stats) in enumerate(subject_stats.iterrows()):
                with cols[i % len(cols)]:
                    st.metric(
                        subject,
                        f"{stats['Average Score']:.1f}%",
                        delta=f"±{stats['Score Std Dev']:.1f}",
                        help=f"{stats['Total Activities']} activities completed"
                    )
        
        # At-risk students alert
        show_at_risk_students(db_manager, progress_df)
    
    else:
        st.info("No learning activity data available yet.")

def show_at_risk_students(db_manager, progress_df):
    """Identify and show at-risk students."""
    
    if progress_df.empty:
        return
    
    # Calculate student performance metrics
    student_performance = progress_df.groupby('student_id').agg({
        'score': ['mean', 'count'],
        'timestamp': 'max'
    })
    student_performance.columns = ['avg_score', 'activity_count', 'last_activity']
    
    # Get student names
    students_df = db_manager.get_all_students()
    student_performance = student_performance.merge(
        students_df[['student_id', 'name']], 
        left_index=True, 
        right_on='student_id', 
        how='left'
    )
    
    # Identify at-risk students
    at_risk_students = []
    
    for _, student in student_performance.iterrows():
        risk_factors = []
        
        # Low performance
        if student['avg_score'] < 60:
            risk_factors.append(f"Low avg score ({student['avg_score']:.1f}%)")
        
        # Low activity
        if student['activity_count'] < 3:
            risk_factors.append(f"Low activity ({student['activity_count']} items)")
        
        # Inactive for long time
        days_since_activity = (datetime.now() - pd.to_datetime(student['last_activity'])).days
        if days_since_activity > 7:
            risk_factors.append(f"Inactive for {days_since_activity} days")
        
        if risk_factors:
            at_risk_students.append({
                'name': student['name'],
                'risk_factors': ', '.join(risk_factors),
                'avg_score': student['avg_score'],
                'activity_count': student['activity_count']
            })
    
    if at_risk_students:
        st.subheader("⚠️ Students Needing Attention")
        
        risk_df = pd.DataFrame(at_risk_students)
        st.dataframe(
            risk_df,
            column_config={
                "name": "Student Name",
                "risk_factors": "Risk Factors",
                "avg_score": st.column_config.NumberColumn(
                    "Avg Score (%)",
                    format="%.1f%%"
                ),
                "activity_count": "Activities"
            },
            use_container_width=True,
            hide_index=True
        )

def show_student_management(db_manager):
    """Show detailed student management interface."""
    
    st.subheader("Student Management")
    
    students_df = db_manager.get_all_students()
    
    if students_df.empty:
        st.warning("No students found in the system.")
        return
    
    # Student selection
    selected_student = st.selectbox(
        "Select Student for Detailed Analysis",
        options=students_df['name'].tolist(),
        key="student_selector"
    )
    
    if selected_student:
        student_id = students_df[students_df['name'] == selected_student]['student_id'].iloc[0]
        show_individual_student_analysis(student_id, selected_student, db_manager)
    
    st.divider()
    
    # All students overview table
    st.subheader("All Students Overview")
    show_students_overview_table(db_manager)

def show_individual_student_analysis(student_id, student_name, db_manager):
    """Show detailed analysis for individual student."""
    
    st.subheader(f"📊 Analysis for {student_name}")
    
    # Get student data
    with st.spinner(f"Loading data for {student_name}..."):
        student_info = db_manager.get_student_profile(student_id)
        progress_data = db_manager.get_student_progress(student_id)
    
    # Student info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Student Information**")
        if not student_info.empty:
            st.write(f"📚 Grade Level: {student_info['grade_level'].iloc[0]}")
            st.write(f"🧠 Learning Style: {student_info['learning_style'].iloc[0]}")
            st.write(f"❤️ Preferred Subjects: {student_info['preferred_subjects'].iloc[0]}")
    
    if not progress_data.empty:
        with col2:
            st.write("**Performance Summary**")
            avg_score = progress_data['score'].mean()
            total_time = progress_data['time_spent'].sum()
            completed_count = len(progress_data)
            
            st.write(f"📈 Average Score: {avg_score:.1f}%")
            st.write(f"⏱️ Total Learning Time: {total_time:.0f} minutes")
            st.write(f"✅ Completed Items: {completed_count}")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Score timeline
            fig_timeline = px.line(
                progress_data.sort_values('timestamp'),
                x='timestamp',
                y='score',
                title=f'{student_name} - Score Timeline',
                markers=True
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with col2:
            # Subject performance
            if 'subject' in progress_data.columns:
                subject_perf = progress_data.groupby('subject')['score'].mean().reset_index()
                fig_subjects = px.bar(
                    subject_perf,
                    x='subject',
                    y='score',
                    title=f'{student_name} - Subject Performance',
                    color='score',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_subjects, use_container_width=True)
        
        # AI Recommendations for this student
        show_student_recommendations(student_id, student_name, db_manager)
        
        # Performance prediction
        show_student_prediction(student_id, student_name, db_manager)
    
    else:
        st.info(f"{student_name} has not completed any content yet.")

def show_student_recommendations(student_id, student_name, db_manager):
    """Show AI recommendations for specific student."""
    
    st.subheader(f"🎯 AI Recommendations for {student_name}")
    
    try:
        from recommendation_engine import RecommendationEngine
        rec_engine = RecommendationEngine(db_manager)
        with st.spinner("Generating personalized recommendations..."):
            recommendations = rec_engine.get_recommendations(student_id, limit=5)
         
        if not recommendations.empty:
            rec_display = recommendations[[
                'title', 'subject', 'difficulty_level', 'content_type', 'duration_minutes'
            ]].copy()
            rec_display.columns = ['Title', 'Subject', 'Difficulty', 'Type', 'Duration (min)']
            
            st.dataframe(rec_display, use_container_width=True, hide_index=True)
        else:
            st.info("No specific recommendations available for this student.")
    
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")

def show_student_prediction(student_id, student_name, db_manager):
    """Show performance prediction for student."""
    
    try:
        from performance_predictor import PerformancePredictor
        predictor = PerformancePredictor(db_manager)
        
        if predictor.is_trained:
            prediction = predictor.predict_performance(student_id)
            
            if prediction is not None:
                st.subheader(f"🔮 Performance Prediction for {student_name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Next Score", f"{prediction:.1f}%")
                
                with col2:
                    if prediction >= 80:
                        st.success("Expected to perform excellently")
                    elif prediction >= 60:
                        st.info("Expected to perform adequately")
                    else:
                        st.warning("May need additional support")
    
    except Exception as e:
        st.error(f"Error generating prediction: {e}")

def show_students_overview_table(db_manager):
    """Show overview table of all students."""
    
    students_df = db_manager.get_all_students()
    progress_df = db_manager.get_all_progress()
    
    # Create summary for each student
    student_summary = []
    
    for _, student in students_df.iterrows():
        student_progress = progress_df[progress_df['student_id'] == student['student_id']]
        
        if not student_progress.empty:
            avg_score = student_progress['score'].mean()
            completed_items = len(student_progress)
            last_activity = student_progress['timestamp'].max()
            total_time = student_progress['time_spent'].sum()
        else:
            avg_score = 0
            completed_items = 0
            last_activity = "Never"
            total_time = 0
        
        student_summary.append({
            'Name': student['name'],
            'Grade': student['grade_level'],
            'Learning Style': student['learning_style'],
            'Avg Score (%)': round(avg_score, 1) if avg_score > 0 else 0,
            'Completed': completed_items,
            'Total Time (min)': int(total_time),
            'Last Activity': last_activity
        })
    
    summary_df = pd.DataFrame(student_summary)
    
    st.dataframe(
        summary_df,
        column_config={
            "Avg Score (%)": st.column_config.ProgressColumn(
                "Avg Score (%)",
                help="Average score across all completed content",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
            "Total Time (min)": st.column_config.NumberColumn(
                "Total Time (min)",
                help="Total time spent learning"
            )
        },
        use_container_width=True,
        hide_index=True
    )

def show_content_overview(db_manager):
    """Show content management overview."""
    
    st.subheader("Content Overview")
    
    content_df = db_manager.get_all_content()
    progress_df = db_manager.get_all_progress()
    
    if content_df.empty:
        st.warning("No content found in the system.")
        return
    
    # Content statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Content Items", len(content_df))
    
    with col2:
        unique_subjects = content_df['subject'].nunique()
        st.metric("Subjects Covered", unique_subjects)
    
    with col3:
        if not progress_df.empty:
            popular_content = progress_df['content_id'].value_counts().iloc[0] if len(progress_df) > 0 else 0
            st.metric("Most Popular Item Usage", popular_content)
        else:
            st.metric("Content Usage", "No data")
    
    with col4:
        avg_duration = content_df['duration_minutes'].mean()
        st.metric("Avg Duration", f"{avg_duration:.0f} min")
    
    # Content distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Content by subject
        subject_counts = content_df['subject'].value_counts()
        fig_subjects = px.pie(
            values=subject_counts.values,
            names=subject_counts.index,
            title='Content Distribution by Subject'
        )
        st.plotly_chart(fig_subjects, use_container_width=True)
    
    with col2:
        # Content by difficulty
        difficulty_counts = content_df['difficulty_level'].value_counts()
        fig_difficulty = px.bar(
            x=difficulty_counts.index,
            y=difficulty_counts.values,
            title='Content by Difficulty Level',
            labels={'x': 'Difficulty Level', 'y': 'Number of Items'}
        )
        st.plotly_chart(fig_difficulty, use_container_width=True)
    
    # Content performance analysis
    if not progress_df.empty:
        st.subheader("📊 Content Performance Analysis")
        
        # Merge progress with content info
        content_performance = progress_df.merge(
            content_df[['content_id', 'title', 'subject', 'difficulty_level']],
            on='content_id',
            how='left'
        )
        
        # Aggregate by content
        content_stats = content_performance.groupby(['content_id', 'title']).agg({
            'score': ['mean', 'count'],
            'time_spent': 'mean'
        }).round(2)
        
        content_stats.columns = ['Avg Score', 'Completions', 'Avg Time (min)']
        content_stats = content_stats.sort_values('Completions', ascending=False)
        
        st.dataframe(
            content_stats.head(10),
            column_config={
                "Avg Score": st.column_config.NumberColumn(
                    "Avg Score (%)",
                    format="%.1f%%"
                ),
                "Avg Time (min)": st.column_config.NumberColumn(
                    "Avg Time (min)",
                    format="%.1f"
                )
            },
            use_container_width=True
        )
    
    # Detailed content table
    st.subheader("📚 All Content")
    
    display_content = content_df[[
        'title', 'subject', 'content_type', 'difficulty_level', 'duration_minutes'
    ]].copy()
    display_content.columns = ['Title', 'Subject', 'Type', 'Difficulty', 'Duration (min)']
    
    st.dataframe(display_content, use_container_width=True, hide_index=True)

def show_detailed_analytics(db_manager):
    """Show detailed analytics and insights."""
    
    st.subheader("📈 Detailed Analytics")
    
    progress_df = db_manager.get_all_progress()
    
    if progress_df.empty:
        st.info("No analytics data available yet.")
        return
    
    # Time-based analysis
    st.subheader("⏰ Time-based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Learning activity by hour
        progress_df['timestamp'] = pd.to_datetime(progress_df['timestamp'], errors='coerce')
        progress_df['hour'] = progress_df['timestamp'].dt.hour
        hourly_activity = progress_df.groupby('hour').size().reset_index(name='activities')
        
        fig_hourly = px.bar(
            hourly_activity,
            x='hour',
            y='activities',
            title='Learning Activity by Hour of Day',
            labels={'hour': 'Hour of Day', 'activities': 'Number of Activities'}
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Learning activity by day of week
        progress_df['day_of_week'] = pd.to_datetime(progress_df['timestamp']).dt.day_name()
        daily_activity = progress_df.groupby('day_of_week').size().reset_index(name='activities')
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_activity['day_of_week'] = pd.Categorical(daily_activity['day_of_week'], categories=day_order, ordered=True)
        daily_activity = daily_activity.sort_values('day_of_week')
        
        fig_daily = px.bar(
            daily_activity,
            x='day_of_week',
            y='activities',
            title='Learning Activity by Day of Week',
            labels={'day_of_week': 'Day of Week', 'activities': 'Number of Activities'}
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    
    # Performance analysis
    st.subheader("📊 Performance Analysis")
    
    # Score distribution by various factors
    if 'subject' in progress_df.columns:
        fig_box = px.box(
            progress_df,
            x='subject',
            y='score',
            title='Score Distribution by Subject'
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Learning Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Time spent vs Score correlation
        if 'time_spent' in progress_df.columns:
            fig_scatter = px.scatter(
                progress_df,
                x='time_spent',
                y='score',
                title='Score vs Time Spent',
                labels={'time_spent': 'Time Spent (minutes)', 'score': 'Score (%)'},
                trendline="ols"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Difficulty level performance
        if 'difficulty_level' in progress_df.columns:
            difficulty_perf = progress_df.groupby('difficulty_level')['score'].mean().reset_index()
            
            fig_difficulty = px.bar(
                difficulty_perf,
                x='difficulty_level',
                y='score',
                title='Average Score by Difficulty Level',
                color='score',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_difficulty, use_container_width=True)
    
    # Learning efficiency metrics
    show_learning_efficiency_metrics(progress_df)

def show_learning_efficiency_metrics(progress_df):
    """Show learning efficiency and engagement metrics."""
    
    st.subheader("⚡ Learning Efficiency Metrics")
    
    # Calculate efficiency metrics
    metrics = []
    
    # Overall metrics
    total_time = progress_df['time_spent'].sum()
    total_activities = len(progress_df)
    avg_score = progress_df['score'].mean()
    
    # Efficiency score (score per minute)
    if total_time > 0:
        efficiency_score = (avg_score * total_activities) / total_time
        metrics.append(("Learning Efficiency", f"{efficiency_score:.2f}", "Score points per minute"))
    
    # Completion rate (activities scoring >60%)
    passing_activities = len(progress_df[progress_df['score'] >= 60])
    completion_rate = (passing_activities / total_activities) * 100 if total_activities > 0 else 0
    metrics.append(("Completion Rate", f"{completion_rate:.1f}%", "Activities scoring 60%+"))
    
    # Average time per activity
    avg_time_per_activity = total_time / total_activities if total_activities > 0 else 0
    metrics.append(("Avg Time per Activity", f"{avg_time_per_activity:.1f} min", "Time efficiency"))
    
    # Display metrics
    cols = st.columns(len(metrics))
    for i, (label, value, help_text) in enumerate(metrics):
        with cols[i]:
            st.metric(label, value, help=help_text)
    
    # Engagement trends
    if len(progress_df) >= 7:  # Need at least a week of data
        st.subheader("📈 Engagement Trends")
        
        # Daily engagement over time
        progress_df['date'] = pd.to_datetime(progress_df['timestamp']).dt.date
        daily_engagement = progress_df.groupby('date').agg({
            'score': 'mean',
            'time_spent': 'sum',
            'student_id': 'nunique'
        }).reset_index()
        daily_engagement.columns = ['Date', 'Avg Score', 'Total Time', 'Active Students']
        
        # Plot engagement trend
        fig_engagement = px.line(
            daily_engagement,
            x='Date',
            y='Avg Score',
            title='Daily Average Performance Trend',
            markers=True
        )
        st.plotly_chart(fig_engagement, use_container_width=True)
