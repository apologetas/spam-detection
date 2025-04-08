import psycopg2
from psycopg2 import Binary
from utils.db_utils import model_to_binary, vectorizer_to_binary

def store_model(conn, model, vectorizer, model_name="spam_detector_v1"):

    try:
        cursor = conn.cursor()
        model_binary = Binary(model_to_binary(model))
        vectorizer_binary = Binary(vectorizer_to_binary(vectorizer))

        cursor.execute(
            "INSERT INTO spam_model (model_name, model_weights, vectorizer) VALUES (%s, %s, %s) RETURNING id",
            (model_name, model_binary, vectorizer_binary)
        )

        conn.commit()

    except psycopg2.Error as e:
        print(f"Failed to store model: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()