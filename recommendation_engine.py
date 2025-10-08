import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import json

class RecommendationEngine:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.content_similarity_matrix = None
        self.user_profiles = {}
        self.content_features = None
        self._build_models()
    
    def _build_models(self):
        """Build recommendation models based on current data."""
        try:
            # Load content and student data
            self.content_df = self.db_manager.get_all_content()
            self.students_df = self.db_manager.get_all_students()
            self.progress_df = self.db_manager.get_all_progress()
            
            if not self.content_df.empty:
                self._build_content_similarity()
                self._build_user_profiles()
        except Exception as e:
            print(f"Error building recommendation models: {e}")
    
    def _build_content_similarity(self):
        """Build content-based similarity matrix."""
        if self.content_df.empty:
            return
        
        try:
            # Create content features by combining text fields
            content_features = []
            for _, row in self.content_df.iterrows():
                features = f"{row['subject']} {row['content_type']} {row['difficulty_level']} {row['description']} {row['tags']}"
                content_features.append(features)
            
            # Create TF-IDF matrix
            tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = tfidf.fit_transform(content_features)
            
            # Calculate cosine similarity
            self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
            self.content_features = content_features
            
        except Exception as e:
            print(f"Error building content similarity: {e}")
    
    def _build_user_profiles(self):
        """Build user profiles based on learning history."""
        if self.progress_df.empty or self.students_df.empty:
            return
        
        try:
            for _, student in self.students_df.iterrows():
                student_id = student['student_id']
                
                # Get student's progress
                student_progress = self.progress_df[self.progress_df['student_id'] == student_id]
                
                if not student_progress.empty:
                    # Calculate preferences based on performance
                    subject_performance = student_progress.groupby('subject')['score'].agg(['mean', 'count'])
                    difficulty_performance = student_progress.groupby('difficulty_level')['score'].agg(['mean', 'count'])
                    
                    # Create user profile
                    profile = {
                        'learning_style': student['learning_style'],
                        'preferred_subjects': student['preferred_subjects'].split(', ') if student['preferred_subjects'] else [],
                        'subject_performance': subject_performance.to_dict('index'),
                        'difficulty_performance': difficulty_performance.to_dict('index'),
                        'avg_score': student_progress['score'].mean(),
                        'completed_count': len(student_progress)
                    }
                    
                    self.user_profiles[student_id] = profile
                else:
                    # Profile for new students
                    profile = {
                        'learning_style': student['learning_style'],
                        'preferred_subjects': student['preferred_subjects'].split(', ') if student['preferred_subjects'] else [],
                        'subject_performance': {},
                        'difficulty_performance': {},
                        'avg_score': 0,
                        'completed_count': 0
                    }
                    
                    self.user_profiles[student_id] = profile
                    
        except Exception as e:
            print(f"Error building user profiles: {e}")
    
    def get_recommendations(self, student_id, limit=10):
        """Get personalized recommendations for a student."""
        try:
            if self.content_df.empty:
                return pd.DataFrame()
            
            # Get student profile
            if student_id not in self.user_profiles:
                self._build_user_profiles()
            
            if student_id not in self.user_profiles:
                return self._get_popular_content(limit)
            
            user_profile = self.user_profiles[student_id]
            
            # Get content that student hasn't completed yet
            completed_content = set()
            if not self.progress_df.empty:
                student_progress = self.progress_df[self.progress_df['student_id'] == student_id]
                completed_content = set(student_progress['content_id'].tolist())
            
            available_content = self.content_df[~self.content_df['content_id'].isin(completed_content)]
            
            if available_content.empty:
                return pd.DataFrame()
            
            # Calculate recommendation scores
            recommendations = []
            
            for _, content in available_content.iterrows():
                score = self._calculate_recommendation_score(content, user_profile, student_id)
                recommendations.append({
                    'content_id': content['content_id'],
                    'title': content['title'],
                    'subject': content['subject'],
                    'content_type': content['content_type'],
                    'difficulty_level': content['difficulty_level'],
                    'duration_minutes': content['duration_minutes'],
                    'description': content['description'],
                    'prerequisites': content['prerequisites'],
                    'recommendation_score': score
                })
            
            # Sort by recommendation score and return top recommendations
            recommendations = sorted(recommendations, key=lambda x: x['recommendation_score'], reverse=True)
            recommendations_df = pd.DataFrame(recommendations[:limit])
            
            return recommendations_df
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return pd.DataFrame()
    
    def _calculate_recommendation_score(self, content, user_profile, student_id):
        """Calculate recommendation score for a content item."""
        score = 0.0
        
        try:
            # Subject preference score (30%)
            if content['subject'] in user_profile['preferred_subjects']:
                score += 0.3
            
            # Performance-based scoring (25%)
            subject_perf = user_profile['subject_performance'].get(content['subject'])
            if subject_perf:
                # Higher score for subjects where student performs well
                perf_score = subject_perf['mean'] / 100.0  # Normalize to 0-1
                score += 0.25 * perf_score
            
            # Difficulty matching (20%)
            difficulty_perf = user_profile['difficulty_performance'].get(content['difficulty_level'])
            if difficulty_perf:
                # Recommend content at appropriate difficulty level
                if difficulty_perf['mean'] >= 80:  # If doing well, can handle harder content
                    if content['difficulty_level'] in ['Intermediate', 'Advanced']:
                        score += 0.2
                elif difficulty_perf['mean'] >= 60:  # Moderate performance
                    if content['difficulty_level'] in ['Beginner', 'Intermediate']:
                        score += 0.2
                else:  # Struggling
                    if content['difficulty_level'] == 'Beginner':
                        score += 0.2
            else:
                # For new students, start with beginner content
                if content['difficulty_level'] == 'Beginner':
                    score += 0.2
            
            # Learning style matching (15%)
            learning_style = user_profile['learning_style']
            content_type = content['content_type']
            
            style_match = {
                'Visual': ['Video', 'Interactive'],
                'Auditory': ['Video', 'Audio'],
                'Kinesthetic': ['Interactive', 'Quiz'],
                'Reading/Writing': ['Article', 'Quiz']
            }
            
            if learning_style in style_match and content_type in style_match[learning_style]:
                score += 0.15
            
            # Content similarity (10%)
            if self.content_similarity_matrix is not None:
                # Find similar content based on what student has completed with high scores
                student_progress = self.progress_df[
                    (self.progress_df['student_id'] == student_id) & 
                    (self.progress_df['score'] >= 80)
                ]
                
                if not student_progress.empty:
                    # Get indices of high-performing content
                    high_perf_content = student_progress['content_id'].tolist()
                    content_indices = self.content_df.reset_index()
                    
                    current_idx = content_indices[content_indices['content_id'] == content['content_id']].index
                    if len(current_idx) > 0:
                        current_idx = current_idx[0]
                        max_similarity = 0
                        
                        for completed_content_id in high_perf_content:
                            completed_idx = content_indices[content_indices['content_id'] == completed_content_id].index
                            if len(completed_idx) > 0:
                                completed_idx = completed_idx[0]
                                if completed_idx < len(self.content_similarity_matrix) and current_idx < len(self.content_similarity_matrix):
                                    similarity = self.content_similarity_matrix[current_idx][completed_idx]
                                    max_similarity = max(max_similarity, similarity)
                        
                        score += 0.1 * max_similarity
            
            # Add small random factor to introduce diversity
            score += np.random.uniform(0, 0.05)
            
        except Exception as e:
            print(f"Error calculating recommendation score: {e}")
            score = np.random.uniform(0.1, 0.5)  # Fallback to random score
        
        return score
    
    def _get_popular_content(self, limit=10):
        """Get popular content as fallback for new users."""
        try:
            if self.content_df.empty:
                return pd.DataFrame()
            
            # For new users, recommend popular beginner content
            popular_content = self.content_df[
                self.content_df['difficulty_level'] == 'Beginner'
            ].head(limit)
            
            if popular_content.empty:
                popular_content = self.content_df.head(limit)
            
            return popular_content
            
        except Exception as e:
            print(f"Error getting popular content: {e}")
            return pd.DataFrame()
    
    def update_models(self):
        """Update recommendation models with new data."""
        self._build_models()
    
    def get_content_recommendations_by_similarity(self, content_id, limit=5):
        """Get similar content recommendations based on content similarity."""
        try:
            if self.content_similarity_matrix is None or self.content_df.empty:
                return pd.DataFrame()
            
            # Find content index
            content_indices = self.content_df.reset_index()
            content_idx = content_indices[content_indices['content_id'] == content_id].index
            
            if len(content_idx) == 0:
                return pd.DataFrame()
            
            content_idx = content_idx[0]
            
            # Get similarity scores
            similarity_scores = list(enumerate(self.content_similarity_matrix[content_idx]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar content (excluding the content itself)
            similar_indices = [i for i, _ in similarity_scores[1:limit+1]]
            similar_content = self.content_df.iloc[similar_indices]
            
            return similar_content
            
        except Exception as e:
            print(f"Error getting similar content: {e}")
            return pd.DataFrame()
