from flask import Flask, request, jsonify
import joblib, os
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
model = joblib.load(MODEL_PATH)

OPENAI_API_KEY = os.environ.get("AIzaSyBaxidtgfq85EIjdMBI-ANKAiOrBqCtQDk", "")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    xs = data.get("x", [])
    if not isinstance(xs, list) or len(xs) == 0:
        return jsonify({"error": "send JSON {\"x\": [numbers]}"}), 400

    X = np.array([[float(v)] for v in xs])
    preds = model.predict(X).tolist()
    return jsonify({"predictions": preds, "openai_key_present": bool(AIzaSyBaxidtgfq85EIjdMBI-ANKAiOrBqCtQDk)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

