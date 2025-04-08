from db import db_connection
from db import db_retrieve

def classify_emails(emails):

    conn = db_connection.get_connection()

    try:
        import numpy as np
        model, vectorizer = db_retrieve.get_model(conn)

        email_vectors = vectorizer.transform(emails).toarray()
        predictions = model.predict(email_vectors)
        print("predictions:", predictions)
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


if __name__ == "__main__":
    test_emails = [
        "Subject: Meeting tomorrow at 10 AM. Please bring your reports.",
        "Subject: URGENT - You have WON a Free Vacation! Click now to claim your prize!",
        "Subject: Important project update from the team lead",
        "Subject: Make MONEY FAST! Work from home opportunity. 100% GUARANTEED!",
        "Subject: A Nigerian prince has left you an inheritance.",
        "Subject: an important message from the bank",
        "Subject: Lecture time change information",
    ]

    results = classify_emails(test_emails)

    if isinstance(results, list):
        for result in results:
            print(f"Email: {result['email']}")
            print(f"Classification: {result['classification']} (Confidence: {result['confidence']})")
            print("-" * 60)
    else:
        print(results)