from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from predict import predict_data

app = FastAPI(title="Wine Classifier API", version="1.0.0")

# Wine dataset feature names (order matters and matches sklearn.load_wine().feature_names):
# ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
#  'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
#  'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

class WineData(BaseModel):
    alcohol: float = Field(..., description="Alcohol")
    malic_acid: float = Field(..., description="Malic acid")
    ash: float = Field(..., description="Ash")
    alcalinity_of_ash: float = Field(..., description="Alcalinity of ash")
    magnesium: float = Field(..., description="Magnesium")
    total_phenols: float = Field(..., description="Total phenols")
    flavanoids: float = Field(..., description="Flavanoids")
    nonflavanoid_phenols: float = Field(..., description="Nonflavanoid phenols")
    proanthocyanins: float = Field(..., description="Proanthocyanins")
    color_intensity: float = Field(..., description="Color intensity")
    hue: float = Field(..., description="Hue")
    od280_od315_of_diluted_wines: float = Field(..., description="OD280/OD315 of diluted wines")
    proline: float = Field(..., description="Proline")

class WineResponse(BaseModel):
    response: Literal[0, 1, 2]  # Wine dataset has 3 classes

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    try:
        features = [[
            wine_features.alcohol,
            wine_features.malic_acid,
            wine_features.ash,
            wine_features.alcalinity_of_ash,
            wine_features.magnesium,
            wine_features.total_phenols,
            wine_features.flavanoids,
            wine_features.nonflavanoid_phenols,
            wine_features.proanthocyanins,
            wine_features.color_intensity,
            wine_features.hue,
            wine_features.od280_od315_of_diluted_wines,
            wine_features.proline,
        ]]
        prediction = predict_data(features)
        return WineResponse(response=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
