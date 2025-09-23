from pathlib import Path
import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wine_model.pkl"

def predict_data(X):
    """
    Predict class labels for the input data using the trained Wine model.
    Args:
        X (array-like): Input samples shape (n_samples, 13).
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    model = joblib.load(MODEL_PATH)
    X_arr = np.asarray(X, dtype=float)
    y_pred = model.predict(X_arr)
    return y_pred
