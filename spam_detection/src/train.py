import pandas as pd
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

data = pd.read_csv("../data/spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['message'] = data['message'].apply(clean_text)

x = data['message']
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
x_train_vect = vectorizer.fit_transform(x_train)
x_test_vect = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(x_train_vect, y_train)

y_pred = model.predict(x_test_vect)

print("Acccuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, "../model/spam_model.pkl")
joblib.dump(vectorizer, "../model/vectorizer.pkl")

print("Saved")