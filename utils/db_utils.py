import io
import pickle
import tempfile
import os


def model_to_binary(model):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_model_path = os.path.join(temp_dir, "model.h5")
        model.save(temp_model_path)

        with open(temp_model_path, "rb") as f:
            binary_data = f.read()

    return binary_data


def vectorizer_to_binary(vectorizer):
    buffer = io.BytesIO()
    pickle.dump(vectorizer, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def binary_to_vectorizer(binary_data):
    buffer = io.BytesIO(binary_data)
    vectorizer = pickle.load(buffer)
    return vectorizer


def binary_to_model(binary_data):

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_model_path = os.path.join(temp_dir, "model.h5")
        with open(temp_model_path, "wb") as f:
            f.write(binary_data)

        from tensorflow.keras.models import load_model
        model = load_model(temp_model_path)

    return model