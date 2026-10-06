# Credit Card Fraud Detection

A Django web app backed by an Artificial Neural Network (ANN) that flags
potentially fraudulent credit card transactions in real time.

## Tech Stack
- Python, Django
- TensorFlow / Keras (ANN)
- scikit-learn (MinMaxScaler, train/test split, metrics)
- imbalanced-learn (SMOTE to balance the highly imbalanced fraud dataset)
- pandas, NumPy

## How it works
1. `ml_model/train_model.py` generates a synthetic, highly-imbalanced
   transaction dataset (mimicking real card-fraud data, ~0.5% fraud rate),
   balances it with **SMOTE**, scales features with **MinMaxScaler**, and
   trains a small **ANN** (Dense layers + dropout) with TensorFlow/Keras.
   The trained model reaches ~93% accuracy on the held-out test set, saved
   to `ml_model/fraud_ann_model.h5` and `ml_model/scaler.pkl`.
2. The Django app (`detector`) loads the saved model + scaler and exposes a
   form where a transaction's features can be submitted for real-time
   prediction, plus a simple visualization of recent predictions.

## Setup
```bash
pip install -r requirements.txt

# Train the model (creates fraud_ann_model.h5 + scaler.pkl)
python ml_model/train_model.py

# Run Django migrations and start the server
python manage.py migrate
python manage.py runserver
```

Visit http://127.0.0.1:8000/ to submit a transaction and see the
real-time fraud prediction and confidence score.
# Credit_fraud_detection
