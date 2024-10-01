import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv("stroke.csv")
data = data.drop(["stroke"], axis=1)
data = pd.get_dummies(data)
X = data.drop("gender_Male", axis=1)
y = data["gender_Male"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
def get_user_input():
    age = int(input("Enter the age of the patient: "))
    hypertension = int(input("Enter the presence of hypertension (1 for Yes, 0 for No): "))
    heart_disease = int(input("Enter the presence of heart disease (1 for Yes, 0 for No): "))
    ever_married = int(input("Enter the presence of a previous marriage (1 for Yes, 0 for No): "))
    work_type = input("Enter the type of work the patient does (Private, Self-employed, Self-employed-not-working, Children, Never-worked): ")
    residential_area = input("Enter the type of residential area the patient lives in (Urban, Rural, Semiurban): ")
    avg_glucose_level = float(input("Enter the average glucose level of the patient: "))
    bmi = float(input("Enter the Body Mass Index (BMI) of the patient: "))
    smoking_status = input("Enter the smoking status of the patient (Currently smokes, Never smoked, Smokes, Ex-smoker): ")
    stroke_presence = input("Enter the presence of a stroke in the patient's family (1 for Yes, 0 for No): ")

    return [age, hypertension, heart_disease, ever_married,
            work_type == "Private", work_type == "Self-employed", work_type == "Self-employed-not-working", work_type == "Children", work_type == "Never-worked",
            residential_area == "Urban", residential_area == "Rural", residential_area == "Semiurban",
            avg_glucose_level, bmi,
            smoking_status == "Currently smokes", smoking_status == "Never smoked", smoking_status == "Smokes", smoking_status == "Ex-smoker",
            stroke_presence]

user_input = get_user_input()
user_input = np.array(user_input).reshape(1, -1)
user_input = pd.get_dummies(user_input).drop("gender_Male", axis=1)
prediction = model.predict(user_input)
print(f"Gender of the patient with stroke: {'Male' if prediction[0] == 1 else 'Female'}")