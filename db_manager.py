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
        """Check if database has any data."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        student_count = cursor.fetchone()[0]
        return student_count == 0
    
    def populate_sample_data(self):
        """Populate database with sample data."""
        # Sample students
        students_data = [
            ("student_1", "Bethuel Sonyoni", "Grade 10", "Visual", "Mathematics, Science"),
            ("student_2", "Vutlhari Masinga", "Grade 9", "Auditory", "English, History"),
            ("student_3", "Siyabonga Hlope", "Grade 11", "Kinesthetic", "Science, Art"),
            ("student_4", "Dimbanyika Phindulo", "Grade 10", "Reading/Writing", "Mathematics, English"),
            ("student_5", "Andile Maluleke", "Grade 9", "Visual", "Art, History")
        ]
        
        cursor = self.conn.cursor()
        cursor.executemany('''
            INSERT OR IGNORE INTO students (student_id, name, grade_level, learning_style, preferred_subjects)
            VALUES (?, ?, ?, ?, ?)
        ''', students_data)
        
        # Load and insert content data
        content_df = self.load_content_from_csv()
        if not content_df.empty:
            content_data = content_df.to_records(index=False)
            cursor.executemany('''
                INSERT OR IGNORE INTO content 
                (content_id, title, subject, content_type, difficulty_level, duration_minutes, description, prerequisites, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', content_data)
        
        # Generate sample progress data
        self.generate_sample_progress()
        
        self.conn.commit()
    
    def load_content_from_csv(self):
        """Load content data from CSV file."""
        try:
            if os.path.exists("data/educational_content.csv"):
                return pd.read_csv("data/educational_content.csv")
            else:
                # Return empty DataFrame if file doesn't exist
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading content CSV: {e}")
            return pd.DataFrame()
    
    def generate_sample_progress(self):
        """Generate sample progress data for demonstration."""
        cursor = self.conn.cursor()
        
        # Get all students and content
        cursor.execute("SELECT student_id FROM students")
        students = [row[0] for row in cursor.fetchall()]
        
        cursor.execute("SELECT content_id FROM content")
        content_items = [row[0] for row in cursor.fetchall()]
        
        if not students or not content_items:
            return
        
        # Generate random progress data
        np.random.seed(42)  # For reproducible results
        
        for student_id in students:
            # Each student completes 5-15 content items
            num_completed = np.random.randint(5, 16)
            completed_content = np.random.choice(content_items, num_completed, replace=False)
            
            for content_id in completed_content:
                # Generate realistic scores based on student and content
                base_score = np.random.normal(75, 15)  # Normal distribution around 75%
                score = max(0, min(100, base_score))  # Clamp between 0-100
                
                # Random time spent (10-120 minutes)
                time_spent = np.random.randint(10, 121)
                
                # Random timestamp in the last 30 days
                days_ago = np.random.randint(0, 31)
                timestamp = datetime.now() - timedelta(days=days_ago)
                
                cursor.execute('''
                    INSERT INTO progress (student_id, content_id, score, time_spent, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (student_id, content_id, score, time_spent, timestamp))
        
        self.conn.commit()
    
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
