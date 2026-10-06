import os
import numpy as np
from django.conf import settings
from django.shortcuts import render

from .forms import TransactionForm
from .models import TransactionCheck

# Lazy-loaded globals so Django can start even before the model is trained.
_model = None
_scaler = None

N_FEATURES = 28

# Distributions mirroring ml_model/train_model.py's synthetic generator,
# used to turn a simple "risk profile" choice into a feature vector.
RISK_PROFILES = {
    "normal": {"loc": 0.0, "scale": 1.0},
    "unusual": {"loc": 0.8, "scale": 1.6},
    "high_risk": {"loc": 1.5, "scale": 2.2},
}


def _load_artifacts():
    global _model, _scaler
    if _model is None or _scaler is None:
        import joblib
        from tensorflow import keras

        model_path = os.path.join(settings.ML_MODEL_DIR, "fraud_ann_model.h5")
        scaler_path = os.path.join(settings.ML_MODEL_DIR, "scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                "Model not found. Run `python ml_model/train_model.py` first."
            )

        _model = keras.models.load_model(model_path)
        _scaler = joblib.load(scaler_path)
    return _model, _scaler


def _build_feature_vector(amount, risk_profile):
    rng = np.random.default_rng()
    params = RISK_PROFILES[risk_profile]
    features = rng.normal(loc=params["loc"], scale=params["scale"], size=(1, N_FEATURES))
    full = np.hstack([features, np.array([[amount]])])
    return full


def predict_view(request):
    result = None
    error = None

    if request.method == "POST":
        form = TransactionForm(request.POST)
        if form.is_valid():
            amount = form.cleaned_data["amount"]
            risk_profile = form.cleaned_data["risk_profile"]
            try:
                model, scaler = _load_artifacts()
                features = _build_feature_vector(amount, risk_profile)
                features_scaled = scaler.transform(features)
                prob = float(model.predict(features_scaled, verbose=0)[0][0])
                is_fraud = prob >= 0.5

                TransactionCheck.objects.create(
                    amount=amount, fraud_probability=prob, is_fraud=is_fraud
                )

                result = {
                    "amount": amount,
                    "probability": prob,
                    "is_fraud": is_fraud,
                }
            except FileNotFoundError as e:
                error = str(e)
    else:
        form = TransactionForm()

    recent = TransactionCheck.objects.all()[:10]
    return render(
        request,
        "detector/predict.html",
        {"form": form, "result": result, "error": error, "recent": recent},
    )
