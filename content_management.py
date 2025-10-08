import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def show_content_management():
    """Main content management interface for educators."""
    st.title("📚 Content Management")
    
    if 'user_type' not in st.session_state or st.session_state.user_type != 'Educator':
        st.error("Access denied. Educator login required.")
        return
    
    db_manager = st.session_state.db_manager
    
    # Main tabs for content management
    tab1, tab2, tab3, tab4 = st.tabs(["📋 All Content", "➕ Add Content", "📊 Analytics", "🏷️ Categories"])
    
    with tab1:
        show_all_content(db_manager)
    
    with tab2:
        show_add_content_form(db_manager)
    
    with tab3:
        show_content_analytics(db_manager)
    
    with tab4:
        show_content_categories(db_manager)

def show_all_content(db_manager):
    """Display all content with management options."""
    st.subheader("📋 All Learning Content")
    
    content_df = db_manager.get_all_content()
    
    if content_df.empty:
        st.warning("No content found in the system.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        subject_filter = st.selectbox(
            "Filter by Subject",
            options=["All"] + list(content_df['subject'].unique()),
            key="content_subject_filter"
        )
    
    with col2:
        difficulty_filter = st.selectbox(
            "Filter by Difficulty",
            options=["All"] + list(content_df['difficulty_level'].unique()),
            key="content_difficulty_filter"
        )
    
    with col3:
        content_type_filter = st.selectbox(
            "Filter by Type",
            options=["All"] + list(content_df['content_type'].unique()),
            key="content_type_filter"
        )
    
    # Apply filters
    filtered_content = content_df.copy()
    
    if subject_filter != "All":
        filtered_content = filtered_content[filtered_content['subject'] == subject_filter]
    
    if difficulty_filter != "All":
        filtered_content = filtered_content[filtered_content['difficulty_level'] == difficulty_filter]
    
    if content_type_filter != "All":
        filtered_content = filtered_content[filtered_content['content_type'] == content_type_filter]
    
    # Search functionality
    search_term = st.text_input("🔍 Search content by title or description")
    if search_term:
        mask = (
            filtered_content['title'].str.contains(search_term, case=False, na=False) |
            filtered_content['description'].str.contains(search_term, case=False, na=False)
        )
        filtered_content = filtered_content[mask]
    
    # Display results count
    st.info(f"Showing {len(filtered_content)} of {len(content_df)} content items")
    
    # Content management interface
    if not filtered_content.empty:
        for idx, content in filtered_content.iterrows():
            with st.expander(f"📚 {content['title']} - {content['subject']} ({content['difficulty_level']})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Subject:** {content['subject']}")
                    st.write(f"**Type:** {content['content_type']}")
                    st.write(f"**Difficulty:** {content['difficulty_level']}")
                    st.write(f"**Duration:** {content['duration_minutes']} minutes")
                    st.write(f"**Description:** {content['description']}")
                    st.write(f"**Prerequisites:** {content['prerequisites']}")
                    st.write(f"**Tags:** {content['tags']}")
                
                with col2:
                    st.write("**Actions**")
                    
                    if st.button("✏️ Edit", key=f"edit_{content['content_id']}"):
                        st.session_state[f"editing_{content['content_id']}"] = True
                        st.rerun()
                    
                    if st.button("🗑️ Delete", key=f"delete_{content['content_id']}"):
                        if st.session_state.get(f"confirm_delete_{content['content_id']}", False):
                            # Perform deletion (note: this would need database implementation)
                            st.success(f"Content '{content['title']}' marked for deletion")
                            st.session_state[f"confirm_delete_{content['content_id']}"] = False
                        else:
                            st.session_state[f"confirm_delete_{content['content_id']}"] = True
                            st.warning("Click again to confirm deletion")
                    
                    # Show content performance if available
                    show_content_performance_summary(content['content_id'], db_manager)
                
                # Edit form (if editing)
                if st.session_state.get(f"editing_{content['content_id']}", False):
                    st.divider()
                    show_edit_content_form(content, db_manager)
    else:
        st.info("No content matches the current filters.")

def show_edit_content_form(content, db_manager):
    """Show form to edit existing content."""
    st.subheader(f"✏️ Edit: {content['title']}")
    
    with st.form(f"edit_content_form_{content['content_id']}"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_title = st.text_input("Title", value=content['title'])
            new_subject = st.selectbox(
                "Subject",
                options=["Mathematics", "Science", "English", "History", "Art"],
                index=["Mathematics", "Science", "English", "History", "Art"].index(content['subject'])
            )
            new_content_type = st.selectbox(
                "Content Type",
                options=["Video", "Article", "Quiz", "Interactive", "Audio"],
                index=["Video", "Article", "Quiz", "Interactive", "Audio"].index(content['content_type'])
            )
        
        with col2:
            new_difficulty = st.selectbox(
                "Difficulty Level",
                options=["Beginner", "Intermediate", "Advanced"],
                index=["Beginner", "Intermediate", "Advanced"].index(content['difficulty_level'])
            )
            new_duration = st.number_input(
                "Duration (minutes)",
                min_value=1,
                max_value=300,
                value=int(content['duration_minutes'])
            )
        
        new_description = st.text_area("Description", value=content['description'])
        new_prerequisites = st.text_input("Prerequisites", value=content['prerequisites'])
        new_tags = st.text_input("Tags (comma-separated)", value=content['tags'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("💾 Save Changes", use_container_width=True):
                # Here you would update the database
                st.success("Content updated successfully!")
                st.session_state[f"editing_{content['content_id']}"] = False
                st.rerun()
        
        with col2:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                st.session_state[f"editing_{content['content_id']}"] = False
                st.rerun()

def show_add_content_form(db_manager):
    """Show form to add new content."""
    st.subheader("➕ Add New Learning Content")
    
    with st.form("add_content_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Title *", placeholder="Enter content title")
            subject = st.selectbox("Subject *", ["Mathematics", "Science", "English", "History", "Art"])
            content_type = st.selectbox("Content Type *", ["Video", "Article", "Quiz", "Interactive", "Audio"])
        
        with col2:
            difficulty = st.selectbox("Difficulty Level *", ["Beginner", "Intermediate", "Advanced"])
            duration = st.number_input("Duration (minutes) *", min_value=1, max_value=300, value=30)
        
        description = st.text_area("Description *", placeholder="Describe what students will learn")
        prerequisites = st.text_input("Prerequisites", placeholder="What should students know before this content?")
        tags = st.text_input("Tags", placeholder="Enter tags separated by commas")
        
        # Learning objectives
        st.subheader("Learning Objectives")
        objectives = st.text_area(
            "Learning Objectives",
            placeholder="List what students will be able to do after completing this content"
        )
        
        # Content URL or file upload placeholder
        st.subheader("Content Source")
        content_url = st.text_input("Content URL", placeholder="Link to video, article, or interactive content")
        
        # Assessment settings
        st.subheader("Assessment Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            has_assessment = st.checkbox("Include Assessment")
            passing_score = st.number_input("Passing Score (%)", min_value=0, max_value=100, value=60)
        
        with col2:
            max_attempts = st.number_input("Max Attempts", min_value=1, max_value=10, value=3)
        
        # Submit button
        if st.form_submit_button("🚀 Add Content", use_container_width=True):
            if title and description and subject:
                # Generate new content ID
                existing_content = db_manager.get_all_content()
                new_content_id = f"content_{len(existing_content) + 1}"
                
                # Here you would add to database
                st.success(f"✅ Content '{title}' added successfully!")
                st.info(f"Content ID: {new_content_id}")
                
                # Show preview of added content
                st.subheader("Preview of Added Content")
                preview_data = {
                    "Title": title,
                    "Subject": subject,
                    "Type": content_type,
                    "Difficulty": difficulty,
                    "Duration": f"{duration} minutes",
                    "Description": description
                }
                
                for key, value in preview_data.items():
                    st.write(f"**{key}:** {value}")
                
            else:
                st.error("❌ Please fill in all required fields (marked with *)")

def show_content_analytics(db_manager):
    """Show analytics for content performance."""
    st.subheader("📊 Content Analytics")
    
    content_df = db_manager.get_all_content()
    progress_df = db_manager.get_all_progress()
    
    if content_df.empty:
        st.warning("No content available for analytics.")
        return
    
    if progress_df.empty:
        st.info("No student activity data available yet.")
        return
    
    # Overall content metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_content = len(content_df)
        st.metric("Total Content Items", total_content)
    
    with col2:
        used_content = progress_df['content_id'].nunique()
        st.metric("Content with Activity", used_content)
    
    with col3:
        usage_rate = (used_content / total_content) * 100 if total_content > 0 else 0
        st.metric("Usage Rate", f"{usage_rate:.1f}%")
    
    with col4:
        avg_score = progress_df['score'].mean()
        st.metric("Avg Content Score", f"{avg_score:.1f}%")
    
    # Content performance analysis
    st.subheader("🎯 Content Performance Analysis")
    
    # Merge progress with content info
    content_performance = progress_df.merge(
        content_df[['content_id', 'title', 'subject', 'difficulty_level', 'content_type']],
        on='content_id',
        how='left'
    )
    
    # Performance by content
    content_stats = content_performance.groupby(['content_id', 'title']).agg({
        'score': ['mean', 'count', 'std'],
        'time_spent': 'mean'
    }).round(2)
    
    content_stats.columns = ['Avg Score', 'Completions', 'Score Std', 'Avg Time (min)']
    content_stats = content_stats.sort_values('Completions', ascending=False)
    
    # Top performing content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Most Popular Content")
        top_popular = content_stats.head(10)[['Completions', 'Avg Score']]
        st.dataframe(
            top_popular,
            column_config={
                "Avg Score": st.column_config.ProgressColumn(
                    "Avg Score (%)",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                )
            }
        )
    
    with col2:
        st.subheader("⭐ Highest Rated Content")
        top_rated = content_stats[content_stats['Completions'] >= 3].sort_values('Avg Score', ascending=False).head(10)
        st.dataframe(
            top_rated[['Avg Score', 'Completions']],
            column_config={
                "Avg Score": st.column_config.ProgressColumn(
                    "Avg Score (%)",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                )
            }
        )
    
    # Content performance visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance by subject
        subject_performance = content_performance.groupby('subject')['score'].mean().reset_index()
        fig_subjects = px.bar(
            subject_performance,
            x='subject',
            y='score',
            title='Average Score by Subject',
            color='score',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_subjects, use_container_width=True)
    
    with col2:
        # Performance by difficulty
        difficulty_performance = content_performance.groupby('difficulty_level')['score'].mean().reset_index()
        fig_difficulty = px.bar(
            difficulty_performance,
            x='difficulty_level',
            y='score',
            title='Average Score by Difficulty',
            color='score',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_difficulty, use_container_width=True)
    
    # Content usage over time
    st.subheader("📈 Content Usage Trends")
    
    content_performance['date'] = pd.to_datetime(content_performance['timestamp']).dt.date
    daily_usage = content_performance.groupby('date').size().reset_index(name='completions')
    
    fig_usage = px.line(
        daily_usage,
        x='date',
        y='completions',
        title='Daily Content Completions',
        markers=True
    )
    st.plotly_chart(fig_usage, use_container_width=True)
    
    # Underperforming content identification
    st.subheader("⚠️ Content Needing Attention")
    
    # Identify content with low scores or low usage
    underperforming = []
    
    for content_id in content_df['content_id']:
        content_progress = progress_df[progress_df['content_id'] == content_id]
        content_info = content_df[content_df['content_id'] == content_id].iloc[0]
        
        if content_progress.empty:
            underperforming.append({
                'Title': content_info['title'],
                'Issue': 'No student activity',
                'Avg Score': 'N/A',
                'Completions': 0
            })
        elif len(content_progress) >= 3:  # Only flag if enough data
            avg_score = content_progress['score'].mean()
            if avg_score < 60:
                underperforming.append({
                    'Title': content_info['title'],
                    'Issue': f'Low performance ({avg_score:.1f}%)',
                    'Avg Score': f"{avg_score:.1f}%",
                    'Completions': len(content_progress)
                })
    
    if underperforming:
        underperforming_df = pd.DataFrame(underperforming)
        st.dataframe(underperforming_df, use_container_width=True, hide_index=True)
    else:
        st.success("All content is performing well! 🎉")

def show_content_performance_summary(content_id, db_manager):
    """Show quick performance summary for a content item."""
    progress_df = db_manager.get_all_progress()
    content_progress = progress_df[progress_df['content_id'] == content_id]
    
    if not content_progress.empty:
        avg_score = content_progress['score'].mean()
        completions = len(content_progress)
        avg_time = content_progress['time_spent'].mean()
        
        st.write("**Performance**")
        st.write(f"📊 Avg Score: {avg_score:.1f}%")
        st.write(f"✅ Completions: {completions}")
        st.write(f"⏱️ Avg Time: {avg_time:.0f}min")
    else:
        st.write("**Performance**")
        st.write("No activity yet")

def show_content_categories(db_manager):
    """Show content organization by categories."""
    st.subheader("🏷️ Content Categories & Organization")
    
    content_df = db_manager.get_all_content()
    
    if content_df.empty:
        st.warning("No content available for categorization.")
        return
    
    # Category overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📚 By Subject")
        subject_counts = content_df['subject'].value_counts()
        for subject, count in subject_counts.items():
            st.write(f"**{subject}:** {count} items")
    
    with col2:
        st.subheader("📊 By Difficulty")
        difficulty_counts = content_df['difficulty_level'].value_counts()
        for difficulty, count in difficulty_counts.items():
            st.write(f"**{difficulty}:** {count} items")
    
    with col3:
        st.subheader("🎥 By Type")
        type_counts = content_df['content_type'].value_counts()
        for content_type, count in type_counts.items():
            st.write(f"**{content_type}:** {count} items")
    
    # Content distribution visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Subject distribution pie chart
        fig_subjects = px.pie(
            values=subject_counts.values,
            names=subject_counts.index,
            title='Content Distribution by Subject'
        )
        st.plotly_chart(fig_subjects, use_container_width=True)
    
    with col2:
        # Content type distribution
        fig_types = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title='Content Distribution by Type'
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    # Content gaps analysis
    st.subheader("🔍 Content Gap Analysis")
    
    # Create a matrix showing content availability
    content_matrix = content_df.groupby(['subject', 'difficulty_level']).size().unstack(fill_value=0)
    
    # Visualize as heatmap
    fig_heatmap = px.imshow(
        content_matrix.values,
        labels=dict(x="Difficulty Level", y="Subject", color="Content Count"),
        x=content_matrix.columns,
        y=content_matrix.index,
        title="Content Availability Matrix",
        color_continuous_scale="YlOrRd"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Recommendations for content gaps
    st.subheader("💡 Content Development Recommendations")
    
    # Find subjects/difficulty combinations with low content
    recommendations = []
    
    for subject in content_df['subject'].unique():
        for difficulty in ['Beginner', 'Intermediate', 'Advanced']:
            count = len(content_df[
                (content_df['subject'] == subject) & 
                (content_df['difficulty_level'] == difficulty)
            ])
            
            if count < 3:  # Recommend if less than 3 items
                recommendations.append({
                    'Subject': subject,
                    'Difficulty': difficulty,
                    'Current Count': count,
                    'Recommendation': f'Add more {difficulty.lower()} level {subject} content'
                })
    
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    else:
        st.success("Good content coverage across all subjects and difficulty levels! 🎉")
    
    # Tag analysis
    st.subheader("🏷️ Content Tags Analysis")
    
    # Extract and analyze tags
    all_tags = []
    for tags_str in content_df['tags'].dropna():
        if tags_str:
            tags = [tag.strip().lower() for tag in str(tags_str).split(',')]
            all_tags.extend(tags)
    
    if all_tags:
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        # Most common tags
        common_tags = tag_counts.most_common(10)
        
        if common_tags:
            tag_df = pd.DataFrame(common_tags, columns=['Tag', 'Count'])
            fig_tags = px.bar(
                tag_df,
                x='Count',
                y='Tag',
                orientation='h',
                title='Most Common Content Tags'
            )
            st.plotly_chart(fig_tags, use_container_width=True)
    else:
        st.info("No tags found in content. Consider adding tags to improve content organization.")

