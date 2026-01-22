from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model/wine_cultivar_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["alcohol"]),
            float(request.form["malic_acid"]),
            float(request.form["ash"]),
            float(request.form["alcalinity_of_ash"]),
            float(request.form["flavanoids"]),
            float(request.form["proline"])
        ]

        scaled = scaler.transform([features])
        pred = model.predict(scaled)[0]
        prediction = f"Cultivar {pred + 1}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
