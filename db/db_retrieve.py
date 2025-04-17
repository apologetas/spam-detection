from utils.db_utils import binary_to_model, binary_to_vectorizer


def get_model(conn):

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT model_weights, vectorizer FROM spam ORDER BY created_at DESC LIMIT 1")

        result = cursor.fetchone()
        if result is None:
            print("No model found in database")
            return None, None

        model_binary, vectorizer_binary = result
        model = binary_to_model(model_binary)
        vectorizer = binary_to_vectorizer(vectorizer_binary)

        return model, vectorizer

    except Exception as e:
        print(f"Error retrieving model: {e}")
        return None, None
    finally:
        cursor.close()