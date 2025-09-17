import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
import numpy as np
import os

class  DBManager:
    def __init__(self, db_path="learning_system.db"):
        self.db_path = db_path
        self.init_connection()
    
    def init_connection(self):
        """Initialize database connection and create tables if they don't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.create_tables()
        except Exception as e:
            print(f"Database connection error: {e}")
    
    def create_tables(self):
        """Create all necessary tables for the learning system."""
        cursor = self.conn.cursor()
        
        # Students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                grade_level TEXT,
                learning_style TEXT,
                preferred_subjects TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Content table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS content (
                content_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                subject TEXT,
                content_type TEXT,
                difficulty_level TEXT,
                duration_minutes INTEGER,
                description TEXT,
                prerequisites TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS progress (
                progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                content_id TEXT,
                score REAL,
                time_spent INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students (student_id),
                FOREIGN KEY (content_id) REFERENCES content (content_id)
            )
        ''')
        
        # Interactions table (for tracking user behavior)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                content_id TEXT,
                interaction_type TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students (student_id),
                FOREIGN KEY (content_id) REFERENCES content (content_id)
            )
        ''')
        
        self.conn.commit()
    
    def initialize_database(self):
        """Initialize database with sample data if empty."""
        if self.is_database_empty():
            self.populate_sample_data()
    
    def is_database_empty(self):
        """Check if the key tables are empty."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM students")
            students_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM content")
            content_count = cursor.fetchone()[0]
            return students_count == 0 or content_count == 0
        except Exception as e:
            print(f"Error checking if database is empty: {e}")
            return True # Assume empty on error to trigger data creation

    def populate_sample_data(self):
        """Populate the database with sample data for demonstration."""
        if not self.is_database_empty():
            print("Database is not empty. Skipping sample data population.")
            return

        print("Populating database with sample data...")

        # 1. Create sample students data
        student_data = [
            ('student_1', 'Bethuel Sonyoni', 'Grade 10', 'Visual', 'Mathematics, Science'),
            ('student_2', 'Vutlhari Masinga', 'Grade 9', 'Auditory', 'English, History'),
            ('student_3', 'Siyabonga Hlope', 'Grade 11', 'Kinesthetic', 'Science, Art'),
            ('student_4', 'Dimbanyika Phindulo', 'Grade 10', 'Reading/Writing', 'Mathematics, English'),
            ('student_5', 'Andile Maluleke', 'Grade 9', 'Visual', 'Art, History'),
            ('student_6', 'Glad Maimele', 'Grade 12', 'Auditory', 'Physics, Chemistry'),
            ('student_7', 'Thando Mthembu', 'Grade 11', 'Kinesthetic', 'Biology, Geography'),
            ('student_8', 'Lebo Moloi', 'Grade 10', 'Reading/Writing', 'Literature, Drama'),
            ('student_9', 'Sipho Dlamini', 'Grade 9', 'Visual', 'Computer Science, Robotics'),
            ('student_10', 'Nandi Ndlovu', 'Grade 12', 'Auditory', 'Music, History')
        ]
        students_df = pd.DataFrame(student_data, columns=['student_id', 'name', 'grade_level', 'learning_style', 'preferred_subjects'])
        students_df.to_sql('students', self.conn, if_exists='replace', index=False)

        # 2. Create sample content data
        content_data = [
            {'content_id': 'c_101', 'title': 'Introduction to Algebra', 'subject': 'Mathematics', 'content_type': 'Video', 'difficulty_level': 'Beginner', 'duration_minutes': 15, 'description': 'Basic concepts of algebra.', 'prerequisites': 'None', 'tags': 'algebra,math'},
            {'content_id': 'c_102', 'title': 'The Scientific Method', 'subject': 'Science', 'content_type': 'Article', 'difficulty_level': 'Beginner', 'duration_minutes': 10, 'description': 'Learn how to think like a scientist.', 'prerequisites': 'None', 'tags': 'science,methodology'},
            {'content_id': 'c_103', 'title': 'World War II History', 'subject': 'History', 'content_type': 'Video', 'difficulty_level': 'Intermediate', 'duration_minutes': 25, 'description': 'A detailed overview of WWII.', 'prerequisites': 'c_102', 'tags': 'history,wwii'},
            {'content_id': 'c_104', 'title': 'Poetry Analysis', 'subject': 'English', 'content_type': 'Text', 'difficulty_level': 'Intermediate', 'duration_minutes': 20, 'description': 'How to analyze poetry for deeper meaning.', 'prerequisites': 'None', 'tags': 'english,poetry'},
            {'content_id': 'c_105', 'title': 'Interactive Science Quiz', 'subject': 'Science', 'content_type': 'Quiz', 'difficulty_level': 'Beginner', 'duration_minutes': 5, 'description': 'Test your knowledge on basic science facts.', 'prerequisites': 'None', 'tags': 'science,quiz,interactive'},
            {'content_id': 'c_106', 'title': 'Advanced Calculus', 'subject': 'Mathematics', 'content_type': 'Video', 'difficulty_level': 'Advanced', 'duration_minutes': 45, 'description': 'Complex concepts in calculus.', 'prerequisites': 'c_101', 'tags': 'calculus,math'},
            {'content_id': 'c_107', 'title': 'Introduction to Python', 'subject': 'Computer Science', 'content_type': 'Interactive', 'difficulty_level': 'Beginner', 'duration_minutes': 60, 'description': 'Learn the basics of coding with Python.', 'prerequisites': 'None', 'tags': 'coding,python,interactive'},
            {'content_id': 'c_108', 'title': 'Biology Lab Simulation', 'subject': 'Biology', 'content_type': 'Interactive', 'difficulty_level': 'Intermediate', 'duration_minutes': 30, 'description': 'Virtual lab for cell biology.', 'prerequisites': 'c_102', 'tags': 'biology,lab,interactive'},
            {'content_id': 'c_109', 'title': 'The Romantic Period', 'subject': 'English', 'content_type': 'Text', 'difficulty_level': 'Advanced', 'duration_minutes': 35, 'description': 'A deep dive into Romantic literature.', 'prerequisites': 'c_104', 'tags': 'english,literature'},
            {'content_id': 'c_110', 'title': 'Chemistry Basics Quiz', 'subject': 'Chemistry', 'content_type': 'Quiz', 'difficulty_level': 'Beginner', 'duration_minutes': 10, 'description': 'Test your basic chemistry knowledge.', 'prerequisites': 'None', 'tags': 'chemistry,quiz'},
            {'content_id': 'c_111', 'title': 'Intro to Web Dev', 'subject': 'Computer Science', 'content_type': 'Interactive', 'difficulty_level': 'Intermediate', 'duration_minutes': 45, 'description': 'Basic web dev with HTML, CSS', 'prerequisites': 'c_107', 'tags': 'coding,webdev'},
            {'content_id': 'c_112', 'title': 'The French Revolution', 'subject': 'History', 'content_type': 'Article', 'difficulty_level': 'Intermediate', 'duration_minutes': 15, 'description': 'Key events of the French Revolution', 'prerequisites': 'None', 'tags': 'history,frenchrevolution'}
        ]
        content_df = pd.DataFrame(content_data)
        content_df.to_sql('content', self.conn, if_exists='replace', index=False)
        
        # 3. Create sample progress and interaction data
        progress_data = []
        for i in range(1, 11):
            for j in range(10):
                student_id = f'student_{i}'
                content_id = np.random.choice([f'c_10{k}' for k in range(1, 13) if k != 11 and k != 12] + ['c_111', 'c_112'])
                score = np.random.randint(50, 101)
                time_spent = np.random.randint(5, 61)
                timestamp = datetime.now() - timedelta(days=np.random.randint(1, 31), hours=np.random.randint(1, 24))
                progress_data.append({
                    'student_id': student_id,
                    'content_id': content_id,
                    'score': float(score),
                    'time_spent': time_spent,
                    'timestamp': timestamp
                })
        
        # Add some more data to ensure diversity
        progress_data.extend([
            {'student_id': 'student_1', 'content_id': 'c_106', 'score': 85.0, 'time_spent': 40, 'timestamp': datetime.now() - timedelta(days=5)},
            {'student_id': 'student_2', 'content_id': 'c_109', 'score': 92.0, 'time_spent': 30, 'timestamp': datetime.now() - timedelta(days=7)},
            {'student_id': 'student_4', 'content_id': 'c_108', 'score': 78.0, 'time_spent': 25, 'timestamp': datetime.now() - timedelta(days=10)},
            {'student_id': 'student_6', 'content_id': 'c_110', 'score': 95.0, 'time_spent': 8, 'timestamp': datetime.now() - timedelta(days=12)}
        ])
        
        progress_df = pd.DataFrame(progress_data)
        progress_df.to_sql('progress', self.conn, if_exists='replace', index=False)
        
        # 4. Add interactions
        interactions_data = []
        for i in range(1, 11):
            for j in range(5):
                student_id = f'student_{i}'
                content_id = np.random.choice(content_df['content_id'])
                interaction_type = np.random.choice(['viewed', 'completed', 'skipped'])
                interactions_data.append({
                    'student_id': student_id,
                    'content_id': content_id,
                    'interaction_type': interaction_type
                })

        interactions_df = pd.DataFrame(interactions_data)
        interactions_df.to_sql('interactions', self.conn, if_exists='replace', index=False)
        
        self.conn.commit()
        print("Sample data population complete.")
    
    def get_all_students(self):
        """Get all students from database."""
        try:
            return pd.read_sql_query("SELECT * FROM students", self.conn)
        except Exception as e:
            print(f"Error fetching students: {e}")
            return pd.DataFrame()
    
    def get_student_profile(self, student_id):
        """Get specific student profile."""
        try:
            query = "SELECT * FROM students WHERE student_id = ?"
            return pd.read_sql_query(query, self.conn, params=(student_id,))
        except Exception as e:
            print(f"Error fetching student profile: {e}")
            return pd.DataFrame()
    
    def get_all_content(self):
        """Get all content from database."""
        try:
            return pd.read_sql_query("SELECT * FROM content", self.conn)
        except Exception as e:
            print(f"Error fetching content: {e}")
            return pd.DataFrame()
    
    def get_student_progress(self, student_id):
        """Get progress data for a specific student with content details."""
        try:
            query = '''
                SELECT p.*, c.title, c.subject, c.difficulty_level
                FROM progress p
                JOIN content c ON p.content_id = c.content_id
                WHERE p.student_id = ?
                ORDER BY p.timestamp DESC
            '''
            return pd.read_sql_query(query, self.conn, params=(student_id,))
        except Exception as e:
            print(f"Error fetching student progress: {e}")
            return pd.DataFrame()
    
    def get_all_progress(self):
        """Get all progress data with student and content details."""
        try:
            query = '''
                SELECT p.*, s.name as student_name, c.title, c.subject, c.difficulty_level
                FROM progress p
                JOIN students s ON p.student_id = s.student_id
                JOIN content c ON p.content_id = c.content_id
                ORDER BY p.timestamp DESC
            '''
            return pd.read_sql_query(query, self.conn, params=())
        except Exception as e:
            print(f"Error fetching all progress: {e}")
            return pd.DataFrame()
    
    def record_progress(self, student_id, content_id, score, time_spent):
        """Record new progress entry."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO progress (student_id, content_id, score, time_spent)
                VALUES (?, ?, ?, ?)
            ''', (student_id, content_id, score, time_spent))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error recording progress: {e}")
            return False
    
    def record_content_interaction(self, student_id, content_id, interaction_type):
        """Record user interaction with content."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO interactions (student_id, content_id, interaction_type)
                VALUES (?, ?, ?)
            ''', (student_id, content_id, interaction_type))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error recording interaction: {e}")
            return False
    
    def update_student_profile(self, student_id, learning_style, preferred_subjects):
        """Update student profile information."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE students 
                SET learning_style = ?, preferred_subjects = ?
                WHERE student_id = ?
            ''', (learning_style, preferred_subjects, student_id))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating student profile: {e}")
            return False
    
    def get_content_by_subject(self, subject):
        """Get content filtered by subject."""
        try:
            query = "SELECT * FROM content WHERE subject = ?"
            return pd.read_sql_query(query, self.conn, params=(subject,))
        except Exception as e:
            print(f"Error fetching content by subject: {e}")
            return pd.DataFrame()
    
    def get_student_interactions(self, student_id):
        """Get interaction history for a student."""
        try:
            query = '''
                SELECT i.*, c.title, c.subject
                FROM interactions i
                JOIN content c ON i.content_id = c.content_id
                WHERE i.student_id = ?
                ORDER BY i.timestamp DESC
            '''
            return pd.read_sql_query(query, self.conn, params=(student_id,))
        except Exception as e:
            print(f"Error fetching student interactions: {e}")
            return pd.DataFrame()
    
    def close_connection(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
