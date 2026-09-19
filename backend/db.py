import os
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import g, current_app

def get_db():
    """
    Opens a new database connection if there is none yet for the
    current application context.
    """
    if 'db' not in g:
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            raise ValueError("DATABASE_URL environment variable is not set.")
        g.db = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        # Ensure autocommit or manage transactions explicitly. 
        # For simplicity in this app, we'll use autocommit.
        g.db.autocommit = True
    return g.db

def close_db(e=None):
    """Closes the database again at the end of the request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """
    Creates the necessary tables if they don't exist.
    """
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("WARNING: DATABASE_URL not set. Skipping DB initialization.")
        return
        
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cars (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                make TEXT NOT NULL,
                model TEXT NOT NULL,
                year INTEGER NOT NULL,
                mileage INTEGER NOT NULL,
                condition TEXT,
                predicted_price REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
    finally:
        cursor.close()
        conn.close()

def create_user_if_new(user_id):
    """Inserts a new user ID if it doesn't already exist."""
    db = get_db()
    with db.cursor() as cursor:
        cursor.execute("""
            INSERT INTO users (id) 
            VALUES (%s) 
            ON CONFLICT (id) DO NOTHING
        """, (user_id,))

def save_car(user_id, make, model, year, mileage, condition, predicted_price):
    """Saves a prediction result to the cars table."""
    car_id = str(uuid.uuid4())
    db = get_db()
    with db.cursor() as cursor:
        cursor.execute("""
            INSERT INTO cars (id, user_id, make, model, year, mileage, condition, predicted_price)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (car_id, user_id, make, model, year, mileage, condition, predicted_price))
    return car_id

def get_user_cars(user_id):
    """Retrieves all saved cars for a specific user."""
    db = get_db()
    with db.cursor() as cursor:
        cursor.execute("""
            SELECT * FROM cars 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """, (user_id,))
        return cursor.fetchall()
