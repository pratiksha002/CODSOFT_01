import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("../data/Churn_Modelling.csv")
print(data.columns)

data = data.drop(columns=["CustomerId","RowNumber","Surname"], errors="ignore")
print("Columns after drop:", data.columns)

data = data.dropna()

label_encoders = {}
for col in data.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le


print("\nData types after encoding:\n", data.dtypes)

x = data.drop("Exited", axis=1)
y = data["Exited"]

print("\nChecking for object columns:")
print(data.select_dtypes(include=["object"]).columns)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "../model/churn_model.pkl")
joblib.dump(label_encoders, "../model/encoders.pkl")

print("Saved")