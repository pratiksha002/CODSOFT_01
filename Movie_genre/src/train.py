import pandas as pd
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

data = pd.read_csv("../data/Genre Classification Dataset/train_data.txt", sep = " ::: ", engine = "python", header = None)

print(data.head())
print(data.columns)


data.columns = ['id', 'title', 'genre', 'description']
data = data[['genre', 'description']]
data['description'] = data['description'].apply(clean_text)

x = data['description']
y = data['genre']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words="english")

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = LogisticRegression(max_iter=2000)
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion metrics:", confusion_matrix(y_test, y_pred))

joblib.dump(model, "../model/movie_model.pkl")
joblib.dump(vectorizer, "../model/vectorizer.pkl")

print("Saved")