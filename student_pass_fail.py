import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Historical student data
data = {
    "attendance": [
        90, 85, 40, 35, 80, 30, 95, 60, 75, 45,
        88, 50, 92, 25, 70, 82, 55, 96, 65, 38
    ],
    "study_hours": [
        5, 4, 1, 2, 6, 1, 7, 3, 5, 2,
        6, 2, 8, 1, 4, 5, 3, 7, 4, 1
    ],
    "previous_marks": [
        80, 75, 30, 25, 70, 20, 90, 55, 65, 35,
        85, 40, 95, 15, 60, 78, 45, 92, 58, 28
    ],
    "result": [
        "Pass", "Pass", "Fail", "Fail", "Pass",
        "Fail", "Pass", "Pass", "Pass", "Fail",
        "Pass", "Fail", "Pass", "Fail", "Pass",
        "Pass", "Fail", "Pass", "Pass", "Fail"
    ]
}

df = pd.DataFrame(data)

# 2. Select input features and target
X = df[["attendance", "study_hours", "previous_marks"]]
y = df["result"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Create and train the machine learning model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("Student Pass/Fail Prediction System")
print("-----------------------------------")
print("Model training completed successfully!")

# 5. Test the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.2%}")

# 6. Get details for a new student
print("\nEnter New Student Details")

attendance = float(input("Attendance percentage (0-100): "))
study_hours = float(input("Daily study hours: "))
previous_marks = float(input("Previous marks percentage (0-100): "))

# 7. Validate input
if not (0 <= attendance <= 100):
    print("Error: Attendance must be between 0 and 100.")
elif not (0 <= previous_marks <= 100):
    print("Error: Previous marks must be between 0 and 100.")
elif study_hours < 0:
    print("Error: Study hours cannot be negative.")
else:
    # 8. Predict the student's result
    new_student = pd.DataFrame([{
        "attendance": attendance,
        "study_hours": study_hours,
        "previous_marks": previous_marks
    }])

    prediction = model.predict(new_student)[0]
    probabilities = model.predict_proba(new_student)[0]
    confidence = max(probabilities) * 100

    # 9. Display the result
    print("\n--- Prediction Result ---")
    print("Predicted outcome:", prediction)
    print(f"Model confidence: {confidence:.2f}%")
    print("Note: This is an educational estimate,")
    print("not a guarantee of the actual exam result.")