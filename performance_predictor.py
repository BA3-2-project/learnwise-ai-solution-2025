import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from data_processor import DataProcessor # Import DataProcessor

class PerformancePredictor:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.data_processor = DataProcessor() # Initialize DataProcessor
        self.model = None
        self.feature_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self._train_model()
    
    def _create_features(self, df, is_training=True):
        """Helper function to create and encode features."""
        
        # 1. Drop rows with missing values for required features
        required_cols = ['score', 'time_spent', 'subject', 'difficulty_level', 
                         'content_type', 'learning_style', 'grade_level']
        df = df.dropna(subset=[col for col in required_cols if col in df.columns])
        
        if df.empty:
            return pd.DataFrame()
        
        # Define features
        categorical_features = ['subject', 'difficulty_level', 'content_type', 'learning_style', 'grade_level']
        numerical_features = ['time_spent', 'duration_minutes']
        
        processed_data = df.copy()

        # 2. Encode Categorical Features
        for col in categorical_features:
            if is_training:
                # Fit LabelEncoder on training data
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                self.feature_encoders[col] = le
            else:
                # Transform data using fitted encoder, handling unseen labels
                le = self.feature_encoders.get(col)
                if le:
                    # Use a mapping to handle unseen labels gracefully (e.g., assign a default value)
                    le_mapping = {label: idx for idx, label in enumerate(le.classes_)}
                    processed_data[col] = processed_data[col].astype(str).map(le_mapping).fillna(len(le.classes_)).astype(int)
                else:
                    processed_data[col] = 0 # Default to 0 if encoder is missing (shouldn't happen)
        
        # 3. Define the final feature list
        self.feature_columns = numerical_features + categorical_features
        
        # Ensure all columns exist before scaling
        features_df = processed_data[self.feature_columns]

        return features_df
    
    def _train_model(self):
        """Train the performance prediction model."""
        try:
            # Get training data
            progress_df = self.db_manager.get_all_progress()
            students_df = self.db_manager.get_all_students()
            content_df = self.db_manager.get_all_content()
            
            if progress_df.empty or students_df.empty or content_df.empty:
                print("Insufficient data for training performance predictor (initial check)")
                return
            
            # Merge data to create features
            training_data = progress_df.merge(
                students_df[['student_id', 'learning_style', 'grade_level']], 
                on='student_id', how='left'
            ).merge(
                content_df[['content_id', 'subject', 'difficulty_level', 'content_type', 'duration_minutes']], 
                on='content_id', how='left'
            )
            
            # CRITICAL: Drop rows with ANY missing value in the required columns
            required_cols = ['score', 'time_spent', 'subject', 'difficulty_level', 
                             'content_type', 'learning_style', 'grade_level', 'duration_minutes']
            training_data.dropna(subset=[col for col in required_cols if col in training_data.columns], inplace=True)

            if len(training_data) < 5:
                print("Insufficient processed data for training.")
                return

            # CRITICAL FIX: Define the final feature columns used for both training and prediction
            self.feature_columns = [
                'learning_style', 'grade_level', 'subject', 'difficulty_level', 
                'content_type', 'duration_minutes', 'time_spent'
            ]

            # Create features
            features_df = self._create_features(training_data, is_training=True)

            if features_df.empty:
                return
            
            # Define X (Features) and y (Target)
            X = features_df.values
            y = training_data['score'].values

            # Split data: 80% for training, 20% for testing
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # 4. Scale numerical features (fit ONLY on training data, transform both)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test) # Do NOT fit on test data!

            # 5. Train the model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # 6. Evaluate (Testing)
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
           
            print(f"Training Features (X_train) shape: {X_train.shape}")
            print(f"Testing Features (X_test) shape: {X_test.shape}")
            print(f"Training Target (y_train) shape: {y_train.shape}")
            print(f"Testing Target (y_test) shape: {y_test.shape}")
            print(f"Model Training Complete. Test MSE: {mse:.2f}, Test R2: {r2:.2f}")
            
        except Exception as e:
            print(f"Error training performance predictor: {e}")
            self.is_trained = False
            return

    def _create_features(self, data, is_training=False):
        """Create feature matrix from the data."""
        
        # We use a copy to avoid modifying the original data frame
        df = data.copy()
        
        # Categorical features to encode
        categorical_features = ['learning_style', 'grade_level', 'subject', 'difficulty_level', 'content_type']
        
        for feature in categorical_features:
            if feature in df.columns:
                if is_training:
                    # Fit LabelEncoder only on training data
                    encoder = LabelEncoder()
                    df[feature] = encoder.fit_transform(df[feature].astype(str))
                    self.feature_encoders[feature] = encoder
                else:
                    # Transform data using fitted encoder
                    if feature in self.feature_encoders:
                        encoder = self.feature_encoders[feature]
                        # Use a mapping to handle unseen labels gracefully
                        le_mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
                        df[feature] = df[feature].astype(str).map(le_mapping).fillna(len(encoder.classes_)).astype(int)
                    else:
                        # Fallback for missing encoder
                        df[feature] = 0
            
        # Numerical features need filling
        if 'duration_minutes' in df.columns:
            df['duration_minutes'] = df['duration_minutes'].fillna(30)
        if 'time_spent' in df.columns:
            df['time_spent'] = df['time_spent'].fillna(0) # Use 0 for prediction

        # CRITICAL FIX: Reindex the DataFrame to match the stored feature columns
        # This ensures the prediction data frame (when not training) has the same columns and order as the training data
        
        # Create the final feature matrix
        # Ensure only the columns in self.feature_columns are kept
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        
        # Add missing columns (should only happen for prediction where columns might be synthetic)
        for col in missing_cols:
            df[col] = 0
            
        try:
            # Return the DataFrame, filtered and ordered by the consistent feature_columns list
            return df[self.feature_columns]
        except KeyError as e:
            print(f"Error aligning features: Missing column {e}. Check feature_columns list.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def predict_performance(self, student_id, content_id=None):
        """Predict performance for a student on next content or specific content."""
        try:
            if not self.is_trained:
                return None
            
            # Get student information
            student_info = self.db_manager.get_student_profile(student_id)
            if student_info.empty:
                return None
            
            # If no specific content, predict for typical content
            if content_id is None:
                # Get student's most common subject and difficulty
                student_progress = self.db_manager.get_student_progress(student_id)
                
                if not student_progress.empty:
                    most_common_subject = student_progress['subject'].mode().iloc[0] if not student_progress['subject'].mode().empty else 'Mathematics'
                    
                    # Determine appropriate difficulty based on recent performance
                    recent_scores = student_progress.head(5)['score'].mean()
                    if recent_scores >= 80:
                        difficulty = 'Intermediate'
                    elif recent_scores >= 60:
                        difficulty = 'Beginner'
                    else:
                        difficulty = 'Beginner'
                else:
                    most_common_subject = 'Mathematics'
                    difficulty = 'Beginner'
                
                # Create typical content data
                content_data = {
                    'subject': most_common_subject,
                    'difficulty_level': difficulty,
                    'content_type': 'Video',
                    'duration_minutes': 30
                }
            else:
                # Get specific content information
                content_info = self.db_manager.get_all_content()
                content_info = content_info[content_info['content_id'] == content_id]
                
                if content_info.empty:
                    return None
                
                content_data = {
                    'subject': content_info['subject'].iloc[0],
                    'difficulty_level': content_info['difficulty_level'].iloc[0],
                    'content_type': content_info['content_type'].iloc[0],
                    'duration_minutes': content_info['duration_minutes'].iloc[0]
                }
            
            # Create prediction data
            prediction_data = pd.DataFrame([{
                'learning_style': student_info['learning_style'].iloc[0],
                'grade_level': student_info['grade_level'].iloc[0],
                'subject': content_data['subject'],
                'difficulty_level': content_data['difficulty_level'],
                'content_type': content_data['content_type'],
                'duration_minutes': content_data['duration_minutes']
            }])
            
            # Create features
            features_df = self._create_features(prediction_data, is_training=False)
            
            if features_df.empty:
                return None
            
            # Scale features
            features_scaled = self.scaler.transform(features_df.values)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Ensure prediction is within reasonable bounds
            prediction = max(0, min(100, prediction))
            
            return prediction
            
        except Exception as e:
            print(f"Error predicting performance: {e}")
            return None
    
    def get_student_performance_trend(self, student_id, days=30):
        """Analyze student's performance trend over time."""
        try:
            progress_data = self.db_manager.get_student_progress(student_id)
            
            if progress_data.empty:
                return {
                    'trend': 'No data',
                    'avg_score': 0,
                    'improvement': 0
                }
            
            # Filter recent data
            progress_data['timestamp'] = pd.to_datetime(progress_data['timestamp'])
            cutoff_date = progress_data['timestamp'].max() - pd.Timedelta(days=days)
            recent_data = progress_data[progress_data['timestamp'] >= cutoff_date]
            
            if len(recent_data) < 2:
                return {
                    'trend': 'Insufficient data',
                    'avg_score': progress_data['score'].mean(),
                    'improvement': 0
                }
            
            # Calculate trend
            recent_data = recent_data.sort_values('timestamp')
            scores = recent_data['score'].values
            
            # Simple linear trend
            x = np.arange(len(scores))
            slope = np.polyfit(x, scores, 1)[0]
            
            avg_score = scores.mean()
            
            # Determine trend category
            if slope > 2:
                trend = 'Improving'
            elif slope < -2:
                trend = 'Declining'
            else:
                trend = 'Stable'
            
            return {
                'trend': trend,
                'avg_score': avg_score,
                'improvement': slope
            }
            
        except Exception as e:
            print(f"Error analyzing performance trend: {e}")
            return {
                'trend': 'Error',
                'avg_score': 0,
                'improvement': 0
            }
    
    def get_subject_difficulty_recommendations(self, student_id):
        """Recommend optimal subject and difficulty combinations for a student."""
        try:
            if not self.is_trained:
                return []
            
            student_info = self.db_manager.get_student_profile(student_id)
            if student_info.empty:
                return []
            
            # Get all possible combinations
            subjects = ['Mathematics', 'Science', 'English', 'History', 'Art']
            difficulties = ['Beginner', 'Intermediate', 'Advanced']
            content_types = ['Video', 'Article', 'Quiz', 'Interactive']
            
            recommendations = []
            
            # Get student's profile information
            student_profile = self.db_manager.get_student_profile(student_id)
            if student_profile.empty:
                return []
            
            student_style = student_profile['learning_style'].iloc[0]
            student_grade = student_profile['grade_level'].iloc[0]

            for subject in subjects:
                for difficulty in difficulties:
                    for content_type in content_types:
                        # Create a mock data point for prediction
                        prediction_data = pd.DataFrame([{
                            'subject': subject,
                            'difficulty_level': difficulty,
                            'content_type': content_type,
                            'duration_minutes': 30,
                            # CRITICAL FIX: Add student features for prediction consistency
                            'learning_style': student_style,
                            'grade_level': student_grade,
                            'time_spent': 0 # Time spent is set to 0 as it is a prediction for new content
                        }])
                        
                        features_df = self._create_features(prediction_data, is_training=False)
                        
                        if not features_df.empty:
                            features_scaled = self.scaler.transform(features_df.values)
                            predicted_score = self.model.predict(features_scaled)[0]
                            
                            recommendations.append({
                                'subject': subject,
                                'difficulty_level': difficulty,
                                'content_type': content_type,
                                'predicted_score': max(0, min(100, predicted_score))
                            })
            
            # Sort by predicted score and return top recommendations
            recommendations = sorted(recommendations, key=lambda x: x['predicted_score'], reverse=True)
            return recommendations[:10]
            
        except Exception as e:
            print(f"Error getting subject difficulty recommendations: {e}")
            return []
    
    def retrain_model(self):
        """Retrain the model with updated data."""
        self.is_trained = False
        self.model = None
        self.feature_encoders = {}
        self.scaler = StandardScaler()
        self._train_model()
