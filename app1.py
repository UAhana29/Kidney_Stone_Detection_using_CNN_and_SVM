from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("clinical_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    severity = None
    suggestions = []

    if request.method == "POST":
        # ---- MATCH TRAINING FEATURES EXACTLY ----
        gravity = float(request.form["gravity"])
        ph = float(request.form["ph"])
        osmo = float(request.form["osmo"])
        cond = float(request.form["cond"])
        urea = float(request.form["urea"])
        calc = float(request.form["calc"])

        # Order MUST match training data
        data = np.array([[gravity, ph, osmo, cond, urea, calc]])

        data = scaler.transform(data)
        result = model.predict(data)[0]

        # ---- SEVERITY LOGIC (DATASET-ALIGNED) ----
        score = 0
        if calc > 10:
            score += 2
        if ph < 5.5:
            score += 1
        if osmo > 800:
            score += 2
        if gravity > 1.020:
            score += 1

        if score <= 2:
            severity = "LOW"
            suggestions = [
                "Increase daily water intake (2.5–3L)",
                "Reduce salt consumption",
                "Eat citrus fruits"
            ]
        elif score <= 4:
            severity = "MEDIUM"
            suggestions = [
                "Avoid high-calcium and salty foods",
                "Increase hydration",
                "Monitor urine pH"
            ]
        else:
            severity = "HIGH"
            suggestions = [
                "Strict hydration (>3L/day)",
                "Dietary regulation required",
                "Consult a urologist"
            ]

        prediction = "Kidney Stone Detected" if result == 1 else "No Kidney Stone"

    return render_template(
        "index.html",
        prediction=prediction,
        severity=severity,
        suggestions=suggestions
    )

if __name__ == "__main__":
    app.run(debug=True)
