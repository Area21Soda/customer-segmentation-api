from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

CLUSTER_DESCRIPTION = {
    0: "Bajo ingreso / Bajo gasto",
    1: "Bajo ingreso / Alto gasto",
    2: "Ingreso medio / Gasto medio",
    3: "Alto ingreso / Alto gasto",
    4: "Alto ingreso / Bajo gasto"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    annual_income = float(data["annual_income"])
    spending_score = float(data["spending_score"])

    X = np.array([[annual_income, spending_score]])
    X_scaled = scaler.transform(X)

    cluster = int(model.predict(X_scaled)[0])

    return jsonify({
        "cluster": cluster,
        "description": CLUSTER_DESCRIPTION[cluster]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
