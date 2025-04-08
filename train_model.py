import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from db import db_connection
from db import db_retrieve
from db import db_store
from utils import graph
from utils import matrix


def predict_with_db_model(conn, new_texts):
    model, vectorizer = db_retrieve.get_model(conn)

    if model is None or vectorizer is None:
        return None

    X_new = vectorizer.transform(new_texts).toarray()

    predictions = model.predict(X_new)

    return (predictions > 0.5).astype(int)

def classify_emails(emails):

    conn = db_connection.get_connection()
    if not conn:
        return "Error: Could not connect to database"

    try:
        import numpy as np


        model, vectorizer = db_retrieve.get_model(conn)

        if model is None or vectorizer is None:
            return "Error: No model found in database"

        email_vectors = vectorizer.transform(emails).toarray()

        predictions = model.predict(email_vectors)
        results = (predictions > 0.5).astype(int)


        classified_emails = []
        for email, pred, raw_pred in zip(emails, results, predictions):
            label = "SPAM" if pred[0] == 1 else "HAM"
            confidence = float(raw_pred[0]) if pred[0] == 1 else 1 - float(raw_pred[0])
            classified_emails.append({
                "email": email[:50] + "..." if len(email) > 50 else email,
                "classification": label,
                "confidence": f"{confidence:.2%}"
            })

        return classified_emails

    except Exception as e:
        return f"Error classifying emails: {str(e)}"
    finally:
        conn.close()


def main():

    conn = db_connection.get_connection()
    if not conn:
        print("Exiting due to database connection failure")
        return


    df = pd.read_csv("spam_ham_dataset.csv")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=9000,
        min_df=3,
        max_df=0.7,
        stop_words='english'
    )

    X = tfidf_vectorizer.fit_transform(df['text']).toarray()
    y = df['label_num'].values


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )


    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001
    )

    class_weights = {
        0: (1 / (y_train == 0).sum()) * (len(y_train) / 2),
        1: (1 / (y_train == 1).sum()) * (len(y_train) / 2)
    }

    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    matrix.get_confusion_matrix(y_test, y_pred)
    graph.plot_training_history(history)

    db_store.store_model(conn, model, tfidf_vectorizer)


    test_emails = [
        "Subject: Meeting tomorrow at 10 AM. Please bring your reports.",
        "Subject: URGENT - You have WON a Free Vacation! Click now to claim your prize!",
        "Subject: Important project update from the team lead",
        "Subject: Make MONEY FAST! Work from home opportunity. 100% GUARANTEED!",
        "Subject: Quarterly financial reports now available",
        "Subject: FREE BITCOIN! Limited time offer - CLICK HERE!!!",
    ]

    print("Testing model retrieval and predictions...")
    results = classify_emails(test_emails)

    if isinstance(results, list):
        for result in results:
            print(f"Email: {result['email']}")
            print(f"Classification: {result['classification']} (Confidence: {result['confidence']})")
            print("-" * 60)
    else:
        print(results)

    conn.close()


if __name__ == "__main__":
    main()