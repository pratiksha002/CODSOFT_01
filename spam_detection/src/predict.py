import joblib

model = joblib.load("../model/spam_model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

from train import clean_text
text = input("ENTER A MESSAGE:")
text = clean_text(text)

text_vect = vectorizer.transform([text])
pred = model.predict(text_vect)

if pred[0] == "spam":
    print("This message is SPAM")
else:
    print("This message is NOT SPAM")

