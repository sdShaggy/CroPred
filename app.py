import os
import joblib
import random
import string
import pandas as pd
from flask import Flask, jsonify, request, render_template, flash, session, redirect, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret")

USERS = {
    "pred001": {"password": "password123", "name": "Sarvagya", "aadhar": "123456789012", "mobile": "9876543210"},
    "pred002": {"password": "securepass", "name": "Shivam", "aadhar": "234567890123", "mobile": "8765432109"}
}


XGB_MODEL_PATH = "models/xgb_model_1.joblib"
LSTM_MODEL_PATH = "models/lstm_model_1.joblib"
SCALER_PATH = "models/scaler_1.joblib"
FEATURE_COLUMNS_PATH = "models/feature_columns_r1.joblib"


if not os.path.exists(XGB_MODEL_PATH):
    raise FileNotFoundError(f"{XGB_MODEL_PATH} not found.")
if not os.path.exists(LSTM_MODEL_PATH):
    raise FileNotFoundError(f"{LSTM_MODEL_PATH} not found.")

xgb_model = joblib.load(XGB_MODEL_PATH)
lstm_model = joblib.load(LSTM_MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

if scaler is None:
    app.logger.warning("Scaler not found. Using raw features.")


if os.path.exists(FEATURE_COLUMNS_PATH):
    model_feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
else:
    model_feature_columns = []
    try:
        booster = xgb_model.get_booster()
        model_feature_columns = booster.feature_names or []
    except Exception:
        app.logger.warning("Could not extract feature names. Provide feature_columns.joblib.")


BASE_DEFAULTS = {
    "year": 2025,
    "soil_ph": 6.3,
    "growing_degree_days_GDD": 2250,
    "monsoon_onset_doy": 150,
    "monsoon_end_doy": 280,
}


DISTRICT_DEFAULTS = {
    "angul": {"annual_avg_rainfall_mm": 1450, "monsoon_rainfall_mm": 1150, "avg_annual_temp_C": 26,
              "avg_max_temp_C": 33, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 77,
              "ndvi.early": 0.51, "ndvi.mid": 0.65, "ndvi.late": 0.60,
              "organic_carbon": 0.75, "nitrogen_kg_per_ha": 65, "phosphorus_kg_per_ha": 32,
              "potassium_kg_per_ha": 42},
    "balangir": {"annual_avg_rainfall_mm": 1300, "monsoon_rainfall_mm": 1050, "avg_annual_temp_C": 27,
                 "avg_max_temp_C": 34, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 72,
                 "ndvi.early": 0.48, "ndvi.mid": 0.62, "ndvi.late": 0.58,
                 "organic_carbon": 0.70, "nitrogen_kg_per_ha": 60, "phosphorus_kg_per_ha": 30,
                 "potassium_kg_per_ha": 40},
    "balasore": {"annual_avg_rainfall_mm": 1600, "monsoon_rainfall_mm": 1300, "avg_annual_temp_C": 27,
                 "avg_max_temp_C": 33, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 78,
                 "ndvi.early": 0.55, "ndvi.mid": 0.68, "ndvi.late": 0.63,
                 "organic_carbon": 0.72, "nitrogen_kg_per_ha": 62, "phosphorus_kg_per_ha": 33,
                 "potassium_kg_per_ha": 43},
    "bargarh": {"annual_avg_rainfall_mm": 1400, "monsoon_rainfall_mm": 1100, "avg_annual_temp_C": 26,
                "avg_max_temp_C": 34, "avg_min_temp_C": 20, "avg_relative_humidity_pct": 74,
                "ndvi.early": 0.50, "ndvi.mid": 0.64, "ndvi.late": 0.59,
                "organic_carbon": 0.74, "nitrogen_kg_per_ha": 63, "phosphorus_kg_per_ha": 31,
                "potassium_kg_per_ha": 41},
    "bhadrak": {"annual_avg_rainfall_mm": 1500, "monsoon_rainfall_mm": 1200, "avg_annual_temp_C": 28,
                "avg_max_temp_C": 34, "avg_min_temp_C": 23, "avg_relative_humidity_pct": 76,
                "ndvi.early": 0.53, "ndvi.mid": 0.67, "ndvi.late": 0.61,
                "organic_carbon": 0.73, "nitrogen_kg_per_ha": 61, "phosphorus_kg_per_ha": 32,
                "potassium_kg_per_ha": 42},
    "boudh": {"annual_avg_rainfall_mm": 1250, "monsoon_rainfall_mm": 950, "avg_annual_temp_C": 27,
              "avg_max_temp_C": 33, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 71,
              "ndvi.early": 0.47, "ndvi.mid": 0.61, "ndvi.late": 0.57,
              "organic_carbon": 0.69, "nitrogen_kg_per_ha": 58, "phosphorus_kg_per_ha": 29,
              "potassium_kg_per_ha": 39},
    "cuttack": {"annual_avg_rainfall_mm": 1550, "monsoon_rainfall_mm": 1250, "avg_annual_temp_C": 27,
                "avg_max_temp_C": 33, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 75,
                "ndvi.early": 0.52, "ndvi.mid": 0.66, "ndvi.late": 0.61,
                "organic_carbon": 0.71, "nitrogen_kg_per_ha": 62, "phosphorus_kg_per_ha": 32,
                "potassium_kg_per_ha": 43},
    "deogarh": {"annual_avg_rainfall_mm": 1200, "monsoon_rainfall_mm": 900, "avg_annual_temp_C": 26,
                "avg_max_temp_C": 32, "avg_min_temp_C": 20, "avg_relative_humidity_pct": 70,
                "ndvi.early": 0.46, "ndvi.mid": 0.60, "ndvi.late": 0.56,
                "organic_carbon": 0.68, "nitrogen_kg_per_ha": 57, "phosphorus_kg_per_ha": 28,
                "potassium_kg_per_ha": 38},
    "dhenkanal": {"annual_avg_rainfall_mm": 1350, "monsoon_rainfall_mm": 1050, "avg_annual_temp_C": 26,
                  "avg_max_temp_C": 33, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 73,
                  "ndvi.early": 0.49, "ndvi.mid": 0.63, "ndvi.late": 0.58,
                  "organic_carbon": 0.72, "nitrogen_kg_per_ha": 61, "phosphorus_kg_per_ha": 30,
                  "potassium_kg_per_ha": 41},
    "gajapati": {"annual_avg_rainfall_mm": 1600, "monsoon_rainfall_mm": 1300, "avg_annual_temp_C": 28,
                 "avg_max_temp_C": 34, "avg_min_temp_C": 23, "avg_relative_humidity_pct": 78,
                 "ndvi.early": 0.54, "ndvi.mid": 0.68, "ndvi.late": 0.62,
                 "organic_carbon": 0.73, "nitrogen_kg_per_ha": 63, "phosphorus_kg_per_ha": 33,
                 "potassium_kg_per_ha": 44},
    "ganjam": {"annual_avg_rainfall_mm": 1550, "monsoon_rainfall_mm": 1250, "avg_annual_temp_C": 27,
               "avg_max_temp_C": 33, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 75,
               "ndvi.early": 0.52, "ndvi.mid": 0.66, "ndvi.late": 0.61,
               "organic_carbon": 0.71, "nitrogen_kg_per_ha": 62, "phosphorus_kg_per_ha": 32,
               "potassium_kg_per_ha": 43},
    "jagatsinghpur": {"annual_avg_rainfall_mm": 1500, "monsoon_rainfall_mm": 1200, "avg_annual_temp_C": 27,
                      "avg_max_temp_C": 33, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 74,
                      "ndvi.early": 0.51, "ndvi.mid": 0.65, "ndvi.late": 0.60,
                      "organic_carbon": 0.70, "nitrogen_kg_per_ha": 60, "phosphorus_kg_per_ha": 31,
                      "potassium_kg_per_ha": 42},
    "jajpur": {"annual_avg_rainfall_mm": 1450, "monsoon_rainfall_mm": 1150, "avg_annual_temp_C": 27,
               "avg_max_temp_C": 33, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 73,
               "ndvi.early": 0.50, "ndvi.mid": 0.64, "ndvi.late": 0.59,
               "organic_carbon": 0.70, "nitrogen_kg_per_ha": 60, "phosphorus_kg_per_ha": 30,
               "potassium_kg_per_ha": 41},
    "jharsuguda": {"annual_avg_rainfall_mm": 1200, "monsoon_rainfall_mm": 900, "avg_annual_temp_C": 26,
                   "avg_max_temp_C": 34, "avg_min_temp_C": 20, "avg_relative_humidity_pct": 70,
                   "ndvi.early": 0.46, "ndvi.mid": 0.60, "ndvi.late": 0.55,
                   "organic_carbon": 0.67, "nitrogen_kg_per_ha": 57, "phosphorus_kg_per_ha": 28,
                   "potassium_kg_per_ha": 39},
    "kalahandi": {"annual_avg_rainfall_mm": 1350, "monsoon_rainfall_mm": 1050, "avg_annual_temp_C": 27,
                  "avg_max_temp_C": 34, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 73,
                  "ndvi.early": 0.49, "ndvi.mid": 0.63, "ndvi.late": 0.58,
                  "organic_carbon": 0.70, "nitrogen_kg_per_ha": 60, "phosphorus_kg_per_ha": 31,
                  "potassium_kg_per_ha": 41},
    "kandhamal": {"annual_avg_rainfall_mm": 1600, "monsoon_rainfall_mm": 1300, "avg_annual_temp_C": 26,
                  "avg_max_temp_C": 33, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 77,
                  "ndvi.early": 0.52, "ndvi.mid": 0.66, "ndvi.late": 0.61,
                  "organic_carbon": 0.72, "nitrogen_kg_per_ha": 62, "phosphorus_kg_per_ha": 32,
                  "potassium_kg_per_ha": 42},
    "kendrapara": {"annual_avg_rainfall_mm": 1550, "monsoon_rainfall_mm": 1250, "avg_annual_temp_C": 27,
                   "avg_max_temp_C": 33, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 75,
                   "ndvi.early": 0.51, "ndvi.mid": 0.65, "ndvi.late": 0.60,
                   "organic_carbon": 0.71, "nitrogen_kg_per_ha": 61, "phosphorus_kg_per_ha": 32,
                   "potassium_kg_per_ha": 42},
    "kendujhar (keonjhar)": {"annual_avg_rainfall_mm": 1400, "monsoon_rainfall_mm": 1100, "avg_annual_temp_C": 26,
                             "avg_max_temp_C": 32, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 72,
                             "ndvi.early": 0.48, "ndvi.mid": 0.62, "ndvi.late": 0.57,
                             "organic_carbon": 0.69, "nitrogen_kg_per_ha": 59, "phosphorus_kg_per_ha": 30,
                             "potassium_kg_per_ha": 40},
    "khordha": {"annual_avg_rainfall_mm": 1500, "monsoon_rainfall_mm": 1200, "avg_annual_temp_C": 27,
                "avg_max_temp_C": 33, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 74,
                "ndvi.early": 0.50, "ndvi.mid": 0.64, "ndvi.late": 0.59,
                "organic_carbon": 0.70, "nitrogen_kg_per_ha": 61, "phosphorus_kg_per_ha": 31,
                "potassium_kg_per_ha": 41},
    "koraput": {"annual_avg_rainfall_mm": 1600, "monsoon_rainfall_mm": 1300, "avg_annual_temp_C": 26,
                "avg_max_temp_C": 32, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 77,
                "ndvi.early": 0.53, "ndvi.mid": 0.67, "ndvi.late": 0.61,
                "organic_carbon": 0.72, "nitrogen_kg_per_ha": 63, "phosphorus_kg_per_ha": 32,
                "potassium_kg_per_ha": 43},
    "malkangiri": {"annual_avg_rainfall_mm": 1400, "monsoon_rainfall_mm": 1100, "avg_annual_temp_C": 26,
                   "avg_max_temp_C": 32, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 74,
                   "ndvi.early": 0.50, "ndvi.mid": 0.64, "ndvi.late": 0.58,
                   "organic_carbon": 0.70, "nitrogen_kg_per_ha": 60, "phosphorus_kg_per_ha": 30,
                   "potassium_kg_per_ha": 41},
    "mayurbhanj": {"annual_avg_rainfall_mm": 1500, "monsoon_rainfall_mm": 1200, "avg_annual_temp_C": 27,
                   "avg_max_temp_C": 33, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 75,
                   "ndvi.early": 0.52, "ndvi.mid": 0.66, "ndvi.late": 0.60,
                   "organic_carbon": 0.71, "nitrogen_kg_per_ha": 62, "phosphorus_kg_per_ha": 32,
                   "potassium_kg_per_ha": 42},
    "nabarangpur": {"annual_avg_rainfall_mm": 1400, "monsoon_rainfall_mm": 1100, "avg_annual_temp_C": 26,
                    "avg_max_temp_C": 32, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 73,
                    "ndvi.early": 0.48, "ndvi.mid": 0.62, "ndvi.late": 0.57,
                    "organic_carbon": 0.69, "nitrogen_kg_per_ha": 59, "phosphorus_kg_per_ha": 30,
                    "potassium_kg_per_ha": 40},
    "nayagarh": {"annual_avg_rainfall_mm": 1350, "monsoon_rainfall_mm": 1050, "avg_annual_temp_C": 26,
                 "avg_max_temp_C": 32, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 72,
                 "ndvi.early": 0.47, "ndvi.mid": 0.61, "ndvi.late": 0.56,
                 "organic_carbon": 0.68, "nitrogen_kg_per_ha": 58, "phosphorus_kg_per_ha": 29,
                 "potassium_kg_per_ha": 39},
    "nuapada": {"annual_avg_rainfall_mm": 1250, "monsoon_rainfall_mm": 950, "avg_annual_temp_C": 27,
                "avg_max_temp_C": 33, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 71,
                "ndvi.early": 0.46, "ndvi.mid": 0.60, "ndvi.late": 0.55,
                "organic_carbon": 0.67, "nitrogen_kg_per_ha": 57, "phosphorus_kg_per_ha": 28,
                "potassium_kg_per_ha": 38},
    "puri": {"annual_avg_rainfall_mm": 1550, "monsoon_rainfall_mm": 1250, "avg_annual_temp_C": 27,
             "avg_max_temp_C": 33, "avg_min_temp_C": 22, "avg_relative_humidity_pct": 75,
             "ndvi.early": 0.52, "ndvi.mid": 0.66, "ndvi.late": 0.61,
             "organic_carbon": 0.71, "nitrogen_kg_per_ha": 62, "phosphorus_kg_per_ha": 32,
             "potassium_kg_per_ha": 43},
    "rayagada": {"annual_avg_rainfall_mm": 1400, "monsoon_rainfall_mm": 1100, "avg_annual_temp_C": 26,
                 "avg_max_temp_C": 32, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 73,
                 "ndvi.early": 0.49, "ndvi.mid": 0.63, "ndvi.late": 0.58,
                 "organic_carbon": 0.70, "nitrogen_kg_per_ha": 60, "phosphorus_kg_per_ha": 31,
                 "potassium_kg_per_ha": 41},
    "sambalpur": {"annual_avg_rainfall_mm": 1300, "monsoon_rainfall_mm": 1000, "avg_annual_temp_C": 26,
                  "avg_max_temp_C": 34, "avg_min_temp_C": 20, "avg_relative_humidity_pct": 70,
                  "ndvi.early": 0.47, "ndvi.mid": 0.61, "ndvi.late": 0.56,
                  "organic_carbon": 0.68, "nitrogen_kg_per_ha": 58, "phosphorus_kg_per_ha": 29,
                  "potassium_kg_per_ha": 39},
    "subarnapur (sonepur)": {"annual_avg_rainfall_mm": 1250, "monsoon_rainfall_mm": 950, "avg_annual_temp_C": 26,
                             "avg_max_temp_C": 33, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 71,
                             "ndvi.early": 0.46, "ndvi.mid": 0.60, "ndvi.late": 0.55,
                             "organic_carbon": 0.67, "nitrogen_kg_per_ha": 57, "phosphorus_kg_per_ha": 28,
                             "potassium_kg_per_ha": 38},
    "sundargarh": {"annual_avg_rainfall_mm": 1400, "monsoon_rainfall_mm": 1100, "avg_annual_temp_C": 26,
                   "avg_max_temp_C": 33, "avg_min_temp_C": 21, "avg_relative_humidity_pct": 73,
                   "ndvi.early": 0.49, "ndvi.mid": 0.63, "ndvi.late": 0.58,
                   "organic_carbon": 0.70, "nitrogen_kg_per_ha": 60, "phosphorus_kg_per_ha": 31,
                   "potassium_kg_per_ha": 41}
}



def get_default_features_for_district(district):
    key = district.strip().lower() if district else None
    defaults = DISTRICT_DEFAULTS.get(key, {})
    out = BASE_DEFAULTS.copy()
    out.update(defaults)
    return out

def find_onehot_name(feature_list, prefix, value):
    if not feature_list or not value:
        return None
    target = f"{prefix}{value}".lower()
    for feat in feature_list:
        if feat.lower() == target:
            return feat
    val_alt = value.replace(" ", "_").replace("(", "").replace(")", "")
    target2 = f"{prefix}{val_alt}".lower()
    for feat in feature_list:
        if feat.lower() == target2:
            return feat
    return None

def align_input_with_model(df, model_features):
    for feat in model_features:
        if feat not in df.columns:
            df[feat] = 0
    extra_cols = [c for c in df.columns if c not in model_features]
    if extra_cols:
        df = df.drop(columns=extra_cols)
    return df[model_features].copy()

def preprocess_for_model(district, season, crop, soil_type):
    defaults = get_default_features_for_district(district)
    data = defaults.copy()
    onehot_flags = {}
    if model_feature_columns:
        for prefix, val in [("district_", district), ("season_", season),
                            ("crop_", crop), ("soil_type_", soil_type)]:
            col_name = find_onehot_name(model_feature_columns, prefix, val)
            if col_name:
                onehot_flags[col_name] = 1
    data.update(onehot_flags)
    df = pd.DataFrame([data])
    if model_feature_columns:
        df = align_input_with_model(df, model_feature_columns)
    df = df.fillna(0)
    X_input_scaled = scaler.transform(df) if scaler else df.values.astype(float)
    return X_input_scaled, df


def generate_captcha(length=5):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choices(chars, k=length))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        pred_id = request.form.get("pred_id", "").strip()
        password = request.form.get("password", "").strip()
        name = request.form.get("name", "").strip()
        aadhar = request.form.get("aadhar", "").strip()
        mobile = request.form.get("mobile", "").strip()
        captcha_input = request.form.get("captcha_input", "").strip()
        captcha_expected = session.get("captcha_text", "")

        if captcha_input.upper() != captcha_expected.upper():
            captcha_text = generate_captcha()
            session["captcha_text"] = captcha_text
            return render_template("reg.html", captcha_text=captcha_text)

        if pred_id in USERS:
            captcha_text = generate_captcha()
            session["captcha_text"] = captcha_text
            return render_template("reg.html", captcha_text=captcha_text)

        USERS[pred_id] = {
            "password": password,
            "name": name,
            "aadhar": aadhar,
            "mobile": mobile
        }

        return redirect(url_for("login"))

    # GET request: show registration page with captcha
    captcha_text = generate_captcha()
    session["captcha_text"] = captcha_text
    return render_template("reg.html", captcha_text=captcha_text)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        pred_id = request.form.get("pred_id", "").strip()
        password = request.form.get("password", "").strip()
        captcha_input = request.form.get("captcha_input", "").strip()
        captcha_expected = session.get("captcha_text", "")

        if captcha_input.upper() != captcha_expected.upper():
            captcha_text = generate_captcha()
            session["captcha_text"] = captcha_text
            
            return render_template("auth.html", captcha_text=captcha_text)

        
        user = USERS.get(pred_id)
        if user and user["password"] == password:
            session["pred_id"] = pred_id
            session["name"] = user.get("name", pred_id)
            return redirect(url_for("index"))  
        else:
            # Invalid login, reload login page with new captcha
            captcha_text = generate_captcha()
            session["captcha_text"] = captcha_text
            return render_template("auth.html", captcha_text=captcha_text)

    
    captcha_text = generate_captcha()
    session["captcha_text"] = captcha_text
    return render_template("auth.html", captcha_text=captcha_text)


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    if request.method == "POST":
        return "", 204
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))



@app.route("/", methods=["GET"])
def index():
    pred_id = session.get("pred_id")
    if not pred_id:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    user_name = session.get("name", pred_id)
    districts_ui = [d.title() for d in DISTRICT_DEFAULTS.keys()]
    crops = ["Paddy", "Maize", "Moong", "Mustard", "Sunflower", "Vegetables", "Jute"]
    seasons = ["Kharif", "Rabi", "Zaid"]
    soil_types = ["Alluvial", "Red_Yellow", "Laterite", "Mixed_Red_Black"]

    return render_template(
        "front_1.html",
        user_name=user_name,
        districts=districts_ui,
        crops=crops,
        seasons=seasons,
        soil_types=soil_types
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "pred_id" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    try:
        district = request.form.get("district")
        season = request.form.get("season")
        crop = request.form.get("crop")
        soil_type = request.form.get("soil_type")
        district_key = district.strip().lower()
        defaults = DISTRICT_DEFAULTS.get(district_key, {})
        rainfall = defaults.get("annual_avg_rainfall_mm", 0)
        ndvi_avg = round((defaults.get("ndvi.early",0) + defaults.get("ndvi.mid",0) + defaults.get("ndvi.late",0))/3, 2)

        
        yhat_xgb = round(random.uniform(1000, 4500), 2)

        return render_template(
            "report.html",
            district=district,
            season=season,
            crop=crop,
            soil_type=soil_type,
            rainfall=rainfall,
            ndvi_avg=ndvi_avg,
            predicted_yield=yhat_xgb,
            message="Showing placeholder yields (model not active)."
        )
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

