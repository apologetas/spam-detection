from db import db_connection
from db import db_retrieve
from train_model import classify_emails
from sklearn.metrics import matthews_corrcoef



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

  #print(matthews_corrcoef(y_true=y_test, y_pred=model.predict(X_test)))
    y_true = [+1, +1, +1, -1]
    y_pred = [+1, -1, +1, +1]

    result =  matthews_corrcoef(y_true, y_pred)
    print(result)