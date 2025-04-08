import os
import psycopg2

def get_connection():
    try:
        conn = psycopg2.connect(
            dbname="ollama",
            user=os.getenv('POSTGRES_DATABASE_USERNAME'),
            password=os.getenv('POSTGRES_DATABASE_PASSWORD'),
            host="localhost",
            port="5555"
        )
        return conn
    except psycopg2.Error as e:
        print(f"Failed to connect to database: {e}")
        return None

