import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import re

class DataProcessor:
    def __init__(self):
        self.learning_styles = self._load_learning_styles()
    
    def _load_learning_styles(self):
        """Load learning styles configuration."""
        try:
            with open('data/learning_styles.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default learning styles if file doesn't exist
            return {
                "Visual": {
                    "description": "Learn best through seeing and visualizing",
                    "preferred_content": ["Video", "Interactive", "Infographic"],
                    "characteristics": ["Good with charts", "Remembers faces", "Likes visual demonstrations"]
                },
                "Auditory": {
                    "description": "Learn best through hearing and speaking",
                    "preferred_content": ["Audio", "Video", "Discussion"],
                    "characteristics": ["Good with verbal instructions", "Likes music", "Talks through problems"]
                },
                "Kinesthetic": {
                    "description": "Learn best through doing and moving",
                    "preferred_content": ["Interactive", "Quiz", "Hands-on"],
                    "characteristics": ["Likes physical activity", "Hands-on learner", "Good with building things"]
                },
                "Reading/Writing": {
                    "description": "Learn best through reading and writing",
                    "preferred_content": ["Article", "Text", "Quiz"],
                    "characteristics": ["Likes taking notes", "Good with written instructions", "Enjoys reading"]
                }
            }
    
    def clean_student_data(self, student_df):
        """Clean and validate student data."""
        try:
            if student_df.empty:
                return student_df
            
            # Clean student names
            if 'name' in student_df.columns:
                student_df['name'] = student_df['name'].str.strip().str.title()
            
            # Validate learning styles
            if 'learning_style' in student_df.columns:
                valid_styles = list(self.learning_styles.keys())
                student_df['learning_style'] = student_df['learning_style'].apply(
                    lambda x: x if x in valid_styles else 'Visual'
                )
            
            # Clean preferred subjects
            if 'preferred_subjects' in student_df.columns:
                student_df['preferred_subjects'] = student_df['preferred_subjects'].apply(
                    self._clean_subject_list
                )
            
            # Validate grade levels
            if 'grade_level' in student_df.columns:
                student_df['grade_level'] = student_df['grade_level'].apply(
                    self._standardize_grade_level
                )
            
            return student_df
            
        except Exception as e:
            print(f"Error cleaning student data: {e}")
            return student_df
    
    def clean_content_data(self, content_df):
        """Clean and validate content data."""
        try:
            if content_df.empty:
                return content_df
            
            # Clean titles
            if 'title' in content_df.columns:
                content_df['title'] = content_df['title'].str.strip().str.title()
            
            # Standardize subjects
            if 'subject' in content_df.columns:
                content_df['subject'] = content_df['subject'].apply(
                    self._standardize_subject
                )
            
            # Validate difficulty levels
            if 'difficulty_level' in content_df.columns:
                valid_difficulties = ['Beginner', 'Intermediate', 'Advanced']
                content_df['difficulty_level'] = content_df['difficulty_level'].apply(
                    lambda x: x if x in valid_difficulties else 'Beginner'
                )
            
            # Validate content types
            if 'content_type' in content_df.columns:
                valid_types = ['Video', 'Article', 'Quiz', 'Interactive', 'Audio']
                content_df['content_type'] = content_df['content_type'].apply(
                    lambda x: x if x in valid_types else 'Article'
                )
            
            # Validate duration
            if 'duration_minutes' in content_df.columns:
                content_df['duration_minutes'] = pd.to_numeric(
                    content_df['duration_minutes'], errors='coerce'
                ).fillna(30).astype(int)
                # Ensure reasonable duration bounds
                content_df['duration_minutes'] = content_df['duration_minutes'].clip(1, 300)
            
            # Clean descriptions
            if 'description' in content_df.columns:
                content_df['description'] = content_df['description'].fillna('').str.strip()
            
            # Process tags
            if 'tags' in content_df.columns:
                content_df['tags'] = content_df['tags'].apply(self._process_tags)
            
            return content_df
            
        except Exception as e:
            print(f"Error cleaning content data: {e}")
            return content_df
    
    def clean_progress_data(self, progress_df):
        """Clean and validate progress data."""
        try:
            if progress_df.empty:
                return progress_df
            
            # Validate scores (0-100)
            if 'score' in progress_df.columns:
                progress_df['score'] = pd.to_numeric(
                    progress_df['score'], errors='coerce'
                ).fillna(0).clip(0, 100)
            
            # Validate time spent
            if 'time_spent' in progress_df.columns:
                progress_df['time_spent'] = pd.to_numeric(
                    progress_df['time_spent'], errors='coerce'
                ).fillna(0).clip(0, 1000)
            
            # Clean timestamps
            if 'timestamp' in progress_df.columns:
                progress_df['timestamp'] = pd.to_datetime(
                    progress_df['timestamp'], errors='coerce'
                ).fillna(datetime.now())
            
            return progress_df
            
        except Exception as e:
            print(f"Error cleaning progress data: {e}")
            return progress_df
    
    def _clean_subject_list(self, subjects_str):
        """Clean and standardize subject list."""
        if pd.isna(subjects_str) or subjects_str == '':
            return 'Mathematics'
        
        # Split by common delimiters
        subjects = re.split(r'[,;|]', str(subjects_str))
        cleaned_subjects = []
        
        for subject in subjects:
            cleaned_subject = self._standardize_subject(subject.strip())
            if cleaned_subject and cleaned_subject not in cleaned_subjects:
                cleaned_subjects.append(cleaned_subject)
        
        return ', '.join(cleaned_subjects) if cleaned_subjects else 'Mathematics'
    
    def _standardize_subject(self, subject):
        """Standardize subject names."""
        if pd.isna(subject):
            return 'Mathematics'
        
        subject = str(subject).strip().lower()
        
        # Subject mapping
        subject_mapping = {
            'math': 'Mathematics',
            'mathematics': 'Mathematics',
            'maths': 'Mathematics',
            'science': 'Science',
            'sciences': 'Science',
            'biology': 'Science',
            'chemistry': 'Science',
            'physics': 'Science',
            'english': 'English',
            'language': 'English',
            'history': 'History',
            'social studies': 'History',
            'art': 'Art',
            'arts': 'Art',
            'drawing': 'Art',
            'music': 'Art'
        }
        
        return subject_mapping.get(subject, subject.title())
    
    def _standardize_grade_level(self, grade):
        """Standardize grade level format."""
        if pd.isna(grade):
            return 'Grade 9'
        
        grade_str = str(grade).strip()
        
        # Extract number from grade string
        numbers = re.findall(r'\d+', grade_str)
        if numbers:
            grade_num = int(numbers[0])
            if 1 <= grade_num <= 12:
                return f'Grade {grade_num}'
        
        return 'Grade 9'  # Default grade
    
    def _process_tags(self, tags_str):
        """Process and clean tags."""
        if pd.isna(tags_str) or tags_str == '':
            return ''
        
        # Split tags and clean them
        tags = re.split(r'[,;|]', str(tags_str))
        cleaned_tags = []
        
        for tag in tags:
            cleaned_tag = tag.strip().lower()
            if cleaned_tag and len(cleaned_tag) > 1:
                cleaned_tags.append(cleaned_tag)
        
        return ', '.join(cleaned_tags)
    
    def calculate_learning_metrics(self, progress_df):
        """Calculate learning metrics from progress data."""
        try:
            if progress_df.empty:
                return {}
            
            metrics = {}
            
            # Basic metrics
            metrics['total_activities'] = len(progress_df)
            metrics['average_score'] = progress_df['score'].mean()
            metrics['total_time_spent'] = progress_df['time_spent'].sum()
            metrics['completion_rate'] = len(progress_df[progress_df['score'] >= 60]) / len(progress_df) * 100
            
            # Performance trends
            if len(progress_df) >= 3:
                # Sort by timestamp
                sorted_progress = progress_df.sort_values('timestamp')
                scores = sorted_progress['score'].values
                
                # Calculate trend (simple linear regression slope)
                x = np.arange(len(scores))
                if len(scores) > 1:
                    slope = np.polyfit(x, scores, 1)[0]
                    metrics['performance_trend'] = 'Improving' if slope > 1 else 'Declining' if slope < -1 else 'Stable'
                else:
                    metrics['performance_trend'] = 'Stable'
            else:
                metrics['performance_trend'] = 'Insufficient data'
            
            # Subject-wise performance
            if 'subject' in progress_df.columns:
                subject_performance = progress_df.groupby('subject')['score'].agg(['mean', 'count'])
                metrics['subject_performance'] = subject_performance.to_dict('index')
                
                # Best and worst subjects
                if not subject_performance.empty:
                    metrics['best_subject'] = subject_performance['mean'].idxmax()
                    metrics['worst_subject'] = subject_performance['mean'].idxmin()
            
            # Difficulty analysis
            if 'difficulty_level' in progress_df.columns:
                difficulty_performance = progress_df.groupby('difficulty_level')['score'].mean()
                metrics['difficulty_performance'] = difficulty_performance.to_dict()
            
            # Recent performance (last 7 days)
            if 'timestamp' in progress_df.columns:
                recent_cutoff = datetime.now() - timedelta(days=7)
                progress_df['timestamp'] = pd.to_datetime(progress_df['timestamp'])
                recent_progress = progress_df[progress_df['timestamp'] >= recent_cutoff]
                
                if not recent_progress.empty:
                    metrics['recent_average_score'] = recent_progress['score'].mean()
                    metrics['recent_activities'] = len(recent_progress)
                else:
                    metrics['recent_average_score'] = 0
                    metrics['recent_activities'] = 0
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating learning metrics: {e}")
            return {}
    
    def generate_learning_insights(self, student_id, progress_df, student_profile):
        """Generate personalized learning insights."""
        try:
            insights = []
            
            if progress_df.empty:
                insights.append({
                    'type': 'info',
                    'message': 'Start your learning journey! Complete some content to get personalized insights.',
                    'priority': 'high'
                })
                return insights
            
            metrics = self.calculate_learning_metrics(progress_df)
            
            # Performance insights
            avg_score = metrics.get('average_score', 0)
            if avg_score >= 85:
                insights.append({
                    'type': 'success',
                    'message': f'Excellent performance! Your average score is {avg_score:.1f}%. Keep up the great work!',
                    'priority': 'medium'
                })
            elif avg_score >= 70:
                insights.append({
                    'type': 'info',
                    'message': f'Good progress with an average score of {avg_score:.1f}%. Consider challenging yourself with harder content.',
                    'priority': 'medium'
                })
            else:
                insights.append({
                    'type': 'warning',
                    'message': f'Your average score is {avg_score:.1f}%. Focus on fundamentals and consider reviewing easier content.',
                    'priority': 'high'
                })
            
            # Trend insights
            trend = metrics.get('performance_trend', 'Stable')
            if trend == 'Improving':
                insights.append({
                    'type': 'success',
                    'message': 'Your performance is improving over time! Great learning progress.',
                    'priority': 'low'
                })
            elif trend == 'Declining':
                insights.append({
                    'type': 'warning',
                    'message': 'Your recent performance shows a declining trend. Consider taking a break or reviewing fundamentals.',
                    'priority': 'high'
                })
            
            # Subject-specific insights
            subject_performance = metrics.get('subject_performance', {})
            if subject_performance:
                best_subject = metrics.get('best_subject')
                worst_subject = metrics.get('worst_subject')
                
                if best_subject and worst_subject and best_subject != worst_subject:
                    best_score = subject_performance[best_subject]['mean']
                    worst_score = subject_performance[worst_subject]['mean']
                    
                    insights.append({
                        'type': 'info',
                        'message': f'You excel in {best_subject} ({best_score:.1f}%) but might need more practice in {worst_subject} ({worst_score:.1f}%).',
                        'priority': 'medium'
                    })
            
            # Activity insights
            recent_activities = metrics.get('recent_activities', 0)
            if recent_activities == 0:
                insights.append({
                    'type': 'warning',
                    'message': 'No recent learning activity detected. Try to maintain consistent study habits.',
                    'priority': 'high'
                })
            elif recent_activities >= 5:
                insights.append({
                    'type': 'success',
                    'message': f'Great activity level with {recent_activities} recent completions!',
                    'priority': 'low'
                })
            
            # Learning style recommendations
            if 'learning_style' in student_profile.columns:
                learning_style = student_profile['learning_style'].iloc[0]
                style_info = self.learning_styles.get(learning_style, {})
                preferred_content = style_info.get('preferred_content', [])
                
                if preferred_content:
                    insights.append({
                        'type': 'info',
                        'message': f'As a {learning_style} learner, you might prefer {", ".join(preferred_content)} content types.',
                        'priority': 'low'
                    })
            
            return sorted(insights, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)
            
        except Exception as e:
            print(f"Error generating learning insights: {e}")
            return []
    
    def export_student_report(self, student_id, db_manager):
        """Generate comprehensive student report."""
        try:
            # Get student data
            student_profile = db_manager.get_student_profile(student_id)
            progress_data = db_manager.get_student_progress(student_id)
            
            if student_profile.empty:
                return None
            
            # Calculate metrics
            metrics = self.calculate_learning_metrics(progress_data)
            insights = self.generate_learning_insights(student_id, progress_data, student_profile)
            
            report = {
                'student_info': {
                    'name': student_profile['name'].iloc[0],
                    'grade_level': student_profile['grade_level'].iloc[0],
                    'learning_style': student_profile['learning_style'].iloc[0],
                    'preferred_subjects': student_profile['preferred_subjects'].iloc[0]
                },
                'performance_metrics': metrics,
                'insights': insights,
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            print(f"Error generating student report: {e}")
            return None
        
    def _preprocess_data_for_training(self, progress_data, students_data, content_data):
        """
        Preprocesses and merges data for model training.

        This function handles:
        1. Merging data from three sources (progress, students, content).
        2. Handling missing data by dropping rows with NaN values.
        3. One-hot encoding categorical features.

        Args:
            progress_data (pd.DataFrame): DataFrame with student progress.
            students_data (pd.DataFrame): DataFrame with student profiles.
            content_data (pd.DataFrame): DataFrame with content details.

        Returns:
            pd.DataFrame: A preprocessed DataFrame ready for model training,
                          or an empty DataFrame if preprocessing fails.
        """
        try:
            # Merge data from the three tables
            merged_data = progress_data.merge(
                students_data, on='student_id', how='left'
            ).merge(
                content_data, on='content_id', how='left'
            )

            # Drop rows with any missing values, as they can't be used for training
            merged_data = merged_data.dropna(
                subset=['score', 'learning_style', 'subject', 'difficulty_level', 'content_type']
            )

            if merged_data.empty:
                print("After cleaning, no data remains for training.")
                return pd.DataFrame()

            # One-hot encode categorical features
            categorical_features = ['learning_style', 'subject', 'difficulty_level', 'content_type']
            merged_data = pd.get_dummies(merged_data, columns=categorical_features, drop_first=True)

            # Select relevant features for the model
            training_features = merged_data.select_dtypes(include=[np.number, 'bool'])
            if 'time_spent' in merged_data.columns:
                training_features['time_spent'] = merged_data['time_spent']

            # Make sure no duplicate columns from merges
            training_features = training_features.loc[:, ~training_features.columns.duplicated()]

            return training_features

        except Exception as e:
            print(f"Error during data preprocessing for training: {e}")
            return pd.DataFrame()
