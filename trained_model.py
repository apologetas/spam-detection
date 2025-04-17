from db import db_connection
from db import db_retrieve
from train_model import classify_emails



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