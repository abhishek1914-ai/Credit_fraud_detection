"""
Train an ANN (TensorFlow/Keras) to detect fraudulent credit card
transactions.

Pipeline:
  1. Generate a synthetic, highly-imbalanced transaction dataset
     (~0.5% fraud rate), mirroring the shape of the classic
     "Credit Card Fraud Detection" (PCA-anonymized) dataset: 28 anonymized
     numerical features (V1..V28) + Amount.
  2. Balance the training data with SMOTE (imbalanced-learn).
  3. Scale features with MinMaxScaler (scikit-learn).
  4. Train a small feed-forward ANN with dropout for regularization.
  5. Evaluate on a held-out, untouched (still imbalanced) test set and
     save the model + scaler for use by the Django app.

Run:
    python ml_model/train_model.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
import joblib

RANDOM_STATE = 42
N_SAMPLES = 20000
N_FEATURES = 28  # V1..V28, like the real dataset's PCA components
FRAUD_RATE = 0.005

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_synthetic_dataset():
    """Create a synthetic, imbalanced transactions dataset."""
    rng = np.random.default_rng(RANDOM_STATE)

    n_fraud = int(N_SAMPLES * FRAUD_RATE)
    n_normal = N_SAMPLES - n_fraud

    # Normal transactions: features centered near 0 (like PCA components)
    normal_features = rng.normal(loc=0.0, scale=1.0, size=(n_normal, N_FEATURES))
    normal_amount = np.abs(rng.normal(loc=60, scale=40, size=(n_normal, 1)))

    # Fraudulent transactions: shifted distribution + higher variance,
    # simulating anomalous behaviour.
    fraud_features = rng.normal(loc=1.5, scale=2.2, size=(n_fraud, N_FEATURES))
    fraud_amount = np.abs(rng.normal(loc=250, scale=150, size=(n_fraud, 1)))

    X = np.vstack([normal_features, fraud_features])
    amount = np.vstack([normal_amount, fraud_amount])
    y = np.array([0] * n_normal + [1] * n_fraud)

    columns = [f"V{i+1}" for i in range(N_FEATURES)]
    df = pd.DataFrame(X, columns=columns)
    df["Amount"] = amount
    df["Class"] = y

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def build_ann(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("Generating synthetic imbalanced transaction dataset...")
    df = generate_synthetic_dataset()
    print(df["Class"].value_counts())

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Scaling features with MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Balancing training data with SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print("Balanced training class counts:", np.bincount(y_train_bal))

    print("Training ANN...")
    model = build_ann(input_dim=X_train_bal.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.fit(
        X_train_bal,
        y_train_bal,
        validation_split=0.1,
        epochs=30,
        batch_size=256,
        callbacks=[early_stop],
        verbose=2,
    )

    print("Evaluating on held-out (still imbalanced) test set...")
    y_pred_prob = model.predict(X_test_scaled).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    model_path = os.path.join(OUT_DIR, "fraud_ann_model.h5")
    scaler_path = os.path.join(OUT_DIR, "scaler.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")


if __name__ == "__main__":
    main()
