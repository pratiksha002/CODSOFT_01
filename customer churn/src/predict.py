import joblib
import pandas as pd

model = joblib.load("../model/churn_model.pkl")
encoders = joblib.load("../model/encoders.pkl")

credit_score = int(input("Credit Score: "))
geography = input("Geography (France/Germany/Spain): ").strip().title()
gender = input("Gender (Male/Female): ").strip().title()
age = int(input("Age: "))
tenure = int(input("Tenure (years): "))
balance = float(input("Balance: "))
num_products = int(input("Number of Products: "))
has_card = int(input("Has Credit Card? (0/1): "))
active = int(input("Is Active Member? (0/1): "))
salary = float(input("Estimated Salary: "))

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": active,
    "EstimatedSalary": salary
}

df = pd.DataFrame([input_data])

for cols in df.columns:
    if cols in encoders:
        df[cols] = encoders[cols].transform(df[cols])

pred = model.predict(df)

if pred[0] == 1:
    print("\nCustomer is likely to churn")
else:
    print("\nCustomer is not likely to churn")