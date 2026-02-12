import joblib
import re
import string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

model = joblib.load("../model/movie_model.pkl")
vectorizer = joblib.load("../model/vectorizer.pkl")

plot = input("ENTER A MOVIE PLOT:")
plot = clean_text(plot)

plot_vect = vectorizer.transform([plot])
pred = model.predict(plot_vect)

print("PREDICTED GENRE IS:", pred[0])