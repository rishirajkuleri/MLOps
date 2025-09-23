from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from data import load_data, split_data

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wine_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def fit_model(X_train, y_train):
    """
    Train a Decision Tree Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)
    joblib.dump(dt_classifier, MODEL_PATH)

if __name__ == "__main__":
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)

    # Optional quick sanity check
    clf = joblib.load(MODEL_PATH)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Test Accuracy: {acc:.3f}")
