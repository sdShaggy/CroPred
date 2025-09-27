from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("sihm_1.joblib")

# if hasattr(model, "set_params"):
#     model.set_params(tree_method="hist", predictor="cpu_predictor")

expected_features = [
    "district", "season", "crop", "soil_type",
    "ndvi_early", "ndvi_mid", "ndvi_late",
    "annual_avg_rainfall_mm", "monsoon_rainfall_mm"
]

@app.route("/")
def home():
    return render_template("front.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        df = pd.DataFrame([data])
        numeric_fields = ["ndvi_early", "ndvi_mid", "ndvi_late",
                          "annual_avg_rainfall_mm", "monsoon_rainfall_mm"]
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors="coerce")
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_features]
        prediction = model.predict(df)[0]

        return jsonify({"predicted_yield": round(float(prediction), 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
