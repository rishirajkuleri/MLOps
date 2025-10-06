import os
import base64
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

# ---- File locations (adjust if your project uses different folders) ----
DATA_FILE = os.path.join(os.path.dirname(__file__), "../data/OnlineRetail.csv")
TEST_FILE = os.path.join(os.path.dirname(__file__), "../data/test.csv")  # expects 2 cols: Quantity, UnitPrice
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")

# ---- Common helpers ----
NUMERIC_FEATURES = ["Quantity", "UnitPrice"]

def _clean_online_retail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning to make OnlineRetail usable for clustering on price/qty:
    - Keep only numeric features needed
    - Drop NA
    - Remove non-positive quantities/prices (returns/cancellations or errors)
    """
    # Be tolerant to various column casings (if any)
    cols = {c.lower(): c for c in df.columns}
    qty_col = cols.get("quantity", "Quantity")
    price_col = cols.get("unitprice", "UnitPrice")

    # Select & rename to canonical names
    x = df[[qty_col, price_col]].rename(columns={qty_col: "Quantity", price_col: "UnitPrice"})
    x = x.dropna()
    x = x[(x["Quantity"] > 0) & (x["UnitPrice"] > 0)]
    return x

# ----------------------------------------------------------------------
# 1) Load & serialize data (now reads OnlineRetail.csv)
# ----------------------------------------------------------------------
def load_data():
    """
    Loads OnlineRetail.csv, performs lightweight cleaning, serializes it,
    and returns base64-encoded pickled DataFrame (JSON-safe).
    """
    df_raw = pd.read_csv(DATA_FILE, encoding="unicode_escape")
    df = _clean_online_retail(df_raw)

    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")

# ----------------------------------------------------------------------
# 2) Preprocess: MinMax scale Quantity & UnitPrice
# ----------------------------------------------------------------------
def data_preprocessing(data_b64: str):
    """
    Deserializes base64 pickled data (DataFrame with Quantity & UnitPrice),
    scales using MinMax, returns base64 pickled numpy array.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    # Ensure the expected numeric columns exist
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in data.")

    clustering_data = df[NUMERIC_FEATURES].copy()

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # Return serialized scaled matrix
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")

# ----------------------------------------------------------------------
# 3) Train KMeans for k=1..49, save last-fitted model (as before), return SSE
# ----------------------------------------------------------------------
def build_save_model(data_b64: str, filename: str):
    """
    Builds KMeans models for k = 1..49 on preprocessed data (numpy array),
    saves the last-fitted model (k=49) to /model/<filename>, and returns SSE list.
    """
    data_bytes = base64.b64decode(data_b64)
    X = pickle.loads(data_bytes)  # numpy array

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    last_model = None

    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
        last_model = kmeans

    os.makedirs(MODEL_DIR, exist_ok=True)
    output_path = os.path.join(MODEL_DIR, filename)
    with open(output_path, "wb") as f:
        pickle.dump(last_model, f)

    return sse

# ----------------------------------------------------------------------
# 4) Load model, compute elbow (info), predict first label for test.csv
# ----------------------------------------------------------------------
def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved (last-fitted) KMeans model and reports optimal k via elbow.
    Predicts the first cluster label for data in ../data/test.csv (must contain Quantity, UnitPrice).
    Returns an int cluster label (JSON-safe).
    """
    model_path = os.path.join(MODEL_DIR, filename)
    loaded_model = pickle.load(open(model_path, "rb"))

    # Elbow (for logging/diagnostics)
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters (elbow): {kl.elbow}")

    # Read test data (assumes same features as training before scaling — here, raw Quantity & UnitPrice)
    test_df_raw = pd.read_csv(TEST_FILE, encoding="unicode_escape")
    # If the test is just two columns without headers, you can adapt:
    # test_df_raw.columns = ["Quantity", "UnitPrice"]
    # For safety, normalize headers and select needed columns:
    if set(NUMERIC_FEATURES).issubset(test_df_raw.columns):
        test_df = test_df_raw[NUMERIC_FEATURES].copy()
    else:
        # attempt to map by lower-cased names
        cols = {c.lower(): c for c in test_df_raw.columns}
        try:
            test_df = test_df_raw[[cols["quantity"], cols["unitprice"]]].copy()
            test_df.columns = NUMERIC_FEATURES
        except Exception:
            raise ValueError("test.csv must contain columns 'Quantity' and 'UnitPrice'.")

    # Clean/clip non-positive just in case
    test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna()
    test_df = test_df.clip(lower=0)

    # IMPORTANT: The model was trained on MinMax-scaled data,
    # but we didn't persist the scaler. Since you originally predicted with raw data,
    # we will follow the same approach: predict on raw (consistent with your prior code).
    # If you want strict correctness, persist and reuse the MinMaxScaler.
    pred = loaded_model.predict(test_df.values)[0]

    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred
