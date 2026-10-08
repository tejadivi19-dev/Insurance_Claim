"""
Insurance Claim Prediction - Flask web app.

Run:
    python train_model.py     # once (creates model.pkl + metrics.json)
    python app.py             # then open http://127.0.0.1:5000
"""
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, url_for

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.pkl"
METRICS_PATH = BASE / "metrics.json"

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------- model loading
_model = None
_metrics = {}


def _try_load():
    """Load model + metrics. Raises if missing, unreadable or built with another scikit-learn."""
    import sklearn

    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError("model.pkl or metrics.json not found")
    metrics = json.loads(METRICS_PATH.read_text())
    if metrics.get("sklearn_version") != sklearn.__version__:
        raise RuntimeError(
            f"model built with scikit-learn {metrics.get('sklearn_version')}, installed is {sklearn.__version__}"
        )
    return joblib.load(MODEL_PATH), metrics


def load_artifacts():
    """Load the trained model once. If it is missing or incompatible, train a fresh one automatically.

    Returns (model, error_message)."""
    global _model, _metrics
    if _model is not None:
        return _model, None
    try:
        _model, _metrics = _try_load()
        return _model, None
    except Exception as first_error:
        app.logger.warning("Saved model unusable (%s). Training a new one - this takes about a minute...", first_error)
    try:
        import train_model

        train_model.main()
        _model, _metrics = _try_load()
        return _model, None
    except Exception as exc:
        app.logger.exception("Automatic training failed")
        _model = None
        return None, f"Could not prepare the model: {exc}. Run 'python train_model.py' and read the error it prints."


# ---------------------------------------------------------------- validation
# name -> (label, type, min, max)   (hard limits for sensible input)
FIELDS = {
    "age": ("Age", int, 18, 100),
    "sex": ("Sex", str, None, None),
    "bmi": ("BMI", float, 10, 70),
    "steps": ("Average daily steps", int, 0, 50000),
    "children": ("Number of children", int, 0, 10),
    "smoker": ("Smoker", str, None, None),
}
CHOICES = {"sex": {"male": 1, "female": 0}, "smoker": {"yes": 1, "no": 0}}


def parse_input(data):
    """Validate raw form/JSON data. Returns (clean_dict, errors_dict)."""
    clean, errors = {}, {}
    for name, (label, typ, lo, hi) in FIELDS.items():
        raw = str(data.get(name, "")).strip().lower()
        if raw == "":
            errors[name] = f"{label} is required."
            continue
        if name in CHOICES:
            if raw not in CHOICES[name]:
                errors[name] = f"{label} must be one of: {', '.join(CHOICES[name])}."
            else:
                clean[name] = CHOICES[name][raw]
            continue
        try:
            num = float(raw)
            if typ is int and not num.is_integer():
                raise ValueError
        except ValueError:
            errors[name] = f"{label} must be a {'whole number' if typ is int else 'number'}."
            continue
        value = int(num) if typ is int else num
        if not (lo <= value <= hi):
            errors[name] = f"{label} must be between {lo} and {hi}."
            continue
        clean[name] = value
    return clean, errors


def out_of_range_notes(clean):
    """Warn when an input is outside the range the model was trained on."""
    notes = []
    for name, (lo, hi) in _metrics.get("feature_ranges", {}).items():
        if name in clean and name not in CHOICES and not (lo <= clean[name] <= hi):
            label = FIELDS[name][0]
            notes.append(f"{label} is outside the training data range ({lo:g} to {hi:g}); treat this result with caution.")
    return notes


def predict_claim(clean):
    model, err = load_artifacts()
    if err:
        raise RuntimeError(err)
    features = _metrics.get("features", list(FIELDS))
    row = pd.DataFrame([[clean[f] for f in features]], columns=features)
    prob = float(model.predict_proba(row)[0][1])
    return prob


def _test_accuracy():
    """Best model's hold-out accuracy as a percentage string, or None."""
    try:
        return f"{_metrics['results'][_metrics['best_model']]['test_accuracy'] * 100:.0f}"
    except KeyError:
        return None


# ---------------------------------------------------------------- routes
@app.route("/")
def home():
    _, err = load_artifacts()
    return render_template("index.html", values={}, errors={}, model_error=err, acc=_test_accuracy())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":  # opening /predict directly -> back to the form
        return redirect(url_for("home"))

    clean, errors = parse_input(request.form)
    if errors:
        return render_template("index.html", values=request.form, errors=errors, model_error=None, acc=_test_accuracy()), 400
    try:
        prob = predict_claim(clean)
    except RuntimeError as exc:
        return render_template("index.html", values=request.form, errors={}, model_error=str(exc), acc=_test_accuracy()), 500

    claim = prob >= 0.5
    return render_template(
        "result.html",
        claim=claim,
        # tree models can output exactly 0 or 1; never show absolute certainty
        probability=round(min(max(prob * 100, 0.1), 99.9), 1),
        inputs=request.form,
        notes=out_of_range_notes(clean),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API.  curl -X POST -H "Content-Type: application/json" -d '{...}' /api/predict"""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Send a JSON object."), 400
    clean, errors = parse_input(data)
    if errors:
        return jsonify(errors=errors), 400
    try:
        prob = predict_claim(clean)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 500
    return jsonify(claim=bool(prob >= 0.5), claim_probability=round(prob, 4), notes=out_of_range_notes(clean))


@app.route("/performance")
def performance():
    load_artifacts()
    return render_template("performance.html", m=_metrics)


@app.route("/health")
def health():
    _, err = load_artifacts()
    return jsonify(status="ok" if not err else "error", detail=err), (200 if not err else 500)


if __name__ == "__main__":
    app.run(debug=True)
