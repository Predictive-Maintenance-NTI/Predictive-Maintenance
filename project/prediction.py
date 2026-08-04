import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

NUMERIC_COLS_RAW = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_COLS_CLEAN = [
    "Air temperature K",
    "Process temperature K",
    "Rotational speed rpm",
    "Torque Nm",
    "Tool wear min",
]

TYPE_MAP = {"L": 1, "M": 2, "H": 3}


def build_features(air_temp, process_temp, rot_speed, torque, tool_wear, machine_type):
    raw = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }])

    scaled = scaler.transform(raw[NUMERIC_COLS_RAW])

    type_code = TYPE_MAP[machine_type]
    type_2 = 1 if type_code == 2 else 0
    type_3 = 1 if type_code == 3 else 0

    features = pd.DataFrame(
        scaled,
        columns=NUMERIC_COLS_CLEAN
    )

    features["Type_2"] = type_2
    features["Type_3"] = type_3

    return features


def predict_failure(
    air_temp,
    process_temp,
    rot_speed,
    torque,
    tool_wear,
    machine_type
):
    features = build_features(
        air_temp,
        process_temp,
        rot_speed,
        torque,
        tool_wear,
        machine_type
    )

    pred_encoded = model.predict(features)[0]

    pred_label = label_encoder.inverse_transform(
        [pred_encoded]
    )[0]

    probabilities = model.predict_proba(features)[0]
    pred_probability = probabilities[pred_encoded]

    return pred_label, pred_probability