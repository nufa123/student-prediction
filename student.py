from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Historical student data
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

X = df[["attendance", "study_hours", "previous_marks"]]
y = df["result"]

# Train the model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            attendance = float(request.form["attendance"])
            study_hours = float(request.form["study_hours"])
            previous_marks = float(request.form["previous_marks"])

            values = {
                "attendance": attendance,
                "study_hours": study_hours,
                "previous_marks": previous_marks
            }

            if not 0 <= attendance <= 100:
                raise ValueError("Attendance must be between 0 and 100.")

            if study_hours < 0 or study_hours > 24:
                raise ValueError("Study hours must be between 0 and 24.")

            if not 0 <= previous_marks <= 100:
                raise ValueError("Previous marks must be between 0 and 100.")

            student = pd.DataFrame([values])
            prediction = model.predict(student)[0]

            probabilities = model.predict_proba(student)[0]
            confidence = round(max(probabilities) * 100, 2)

        except ValueError as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)