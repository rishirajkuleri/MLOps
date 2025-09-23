
#  Wine Classification API

This project is a simple **MLOps-style FastAPI application** that trains and serves a machine learning model on the [Wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset) from scikit-learn.  
The model classifies wines into one of **three cultivars (0, 1, 2)** based on 13 chemical properties.

---

##  Project Structure

```

mlops\_labs/
└── fastapi\_lab1/
├── assets/                   # (optional assets, docs, screenshots, etc.)
├── fastapi\_lab1\_env/         # local virtual environment (not tracked in Git)
├── model/
│   └── wine\_model.pkl        # trained model artifact
├── src/
│   ├── **init**.py
│   ├── data.py               # load & split Wine dataset
│   ├── main.py               # FastAPI app (endpoints)
│   ├── predict.py            # prediction helper
│   └── train.py              # training script (saves model)
├── requirements.txt          # Python dependencies
└── README.md                 # project documentation

````

---

##  Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/rishirajkuleri/MLOps.git
   cd mlops_labs/fastapi_lab1

2. **Create a virtual environment**

   ```bash
   python -m venv fastapi_lab1_env
   ```

3. **Activate the environment**

   * Linux/macOS:

     ```bash
     source fastapi_lab1_env/bin/activate
     ```
   * Windows (cmd):

     ```cmd
     fastapi_lab1_env\Scripts\activate
     ```
   * Windows (PowerShell):

     ```powershell
     fastapi_lab1_env\Scripts\Activate.ps1
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

##  Train the Model

Run the training script to train and save the classifier:

```bash
python train.py
```

* A **DecisionTreeClassifier** is trained on the Wine dataset.
* The trained model is saved to `model/wine_model.pkl`.

---

##  Run the API

Start the FastAPI server with Uvicorn:

```bash
python uvicorn app:main --reload
```

* API will be available at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Interactive docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* Alternative docs (ReDoc): [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 API Endpoints

### Health Check

```http
GET /
```

**Response**

```json
{ "status": "healthy" }
```

### Predict Wine Class

```http
POST /predict
```

**Request Body** (all 13 wine features must be provided):

```json
{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26,
  "proanthocyanins": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.4,
  "proline": 1050
}
```

**Response**

```json
{
  "response": 2
}
```

Where:

* `0` → Cultivar 1
* `1` → Cultivar 2
* `2` → Cultivar 3

---

##  Requirements

Main dependencies:

* [fastapi](https://fastapi.tiangolo.com/)
* [uvicorn](https://www.uvicorn.org/)
* [scikit-learn](https://scikit-learn.org/)
* [joblib](https://joblib.readthedocs.io/)

Install them via:

```bash
pip install -r requirements.txt
```

---

##  Next Steps / Improvements

* Add endpoint to return **class probabilities** (`predict_proba`) instead of just the class label.
* Store training metrics and logs.
* Containerize with **Docker** for deployment.
* Add **GitHub Actions** CI/CD for automated testing and deployment.

---

##  Author

* **Rishi Raj Kuleri** – Northeastern University – *MLOps Lab 1*

```
