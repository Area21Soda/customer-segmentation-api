from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo y scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Etiquetas de clusters
cluster_labels = {
    0: "Cliente Conservador (Bajo ingreso / Bajo gasto)",
    1: "Cliente Impulsivo (Bajo ingreso / Alto gasto)",
    2: "Cliente Prudente (Alto ingreso / Bajo gasto)",
    3: "Cliente Premium (Alto ingreso / Alto gasto)",
    4: "Cliente Potencial (Ingreso medio / Gasto medio)"
}

@app.route("/", methods=["GET"])
def home():
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
        "segmento": cluster_labels[cluster]
    })

if __name__ == "__main__":
    app.run(debug=True)
