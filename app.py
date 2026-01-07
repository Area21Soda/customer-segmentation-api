from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# ==============================
# CARGA DE MODELO Y ESCALADOR
# ==============================
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# CENTROIDES NORMALIZADOS
# ==============================
centroids = model.cluster_centers_

# ==============================
# INTERPRETACIÓN DINÁMICA
# ==============================
def interpret_cluster(cluster_id):
    income, spending = centroids[cluster_id]

    # Umbrales sobre datos normalizados
    if income < -0.5 and spending < -0.5:
        return "Cliente Conservador (Bajo ingreso / Bajo gasto)"

    if income < -0.5 and spending >= 0.5:
        return "Cliente Impulsivo (Bajo ingreso / Alto gasto)"

    if income >= 0.5 and spending < -0.5:
        return "Cliente Prudente (Alto ingreso / Bajo gasto)"

    if income >= 0.5 and spending >= 0.5:
        return "Cliente Premium (Alto ingreso / Alto gasto)"

    return "Cliente Potencial (Ingreso medio / Gasto medio)"

# ==============================
# RUTAS
# ==============================
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
    segmento = interpret_cluster(cluster)

    return jsonify({
        "cluster": cluster,
        "segmento": segmento
    })

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
