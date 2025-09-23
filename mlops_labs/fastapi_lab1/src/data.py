import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the Wine dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the Wine dataset.
        y (numpy.ndarray): The target values of the Wine dataset.
        feature_names (list[str]): Ordered feature names.
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    return X, y, feature_names

def split_data(X: np.ndarray, y: np.ndarray):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=12, stratify=y
    )
    return X_train, X_test, y_train, y_test
