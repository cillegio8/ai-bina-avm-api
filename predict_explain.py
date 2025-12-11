from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OR replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Pydantic models ----

class ListingFeatures(BaseModel):
    # use same names as your model features
    area_m2: float
    rooms: int
    floor: int | None = None
    floors_total: int | None = None
    property_type: str
    microlocation: str | None = None
    city: str
    təmir: str | None = None
    çıxarış: str | None = None
    # include any tag_* features as optional 0/1 fields if you want clients to pass them
    # or you compute them on backend from microlocation / extra_info


class ExplanationResponse(BaseModel):
    listing_id: str | None = None
    predictions: dict
    model_info: dict
    key_attributes: dict
    top_positive_contributors: list
    top_negative_contributors: list
    all_contributors: list


# ---- Load model & SHAP explainer once at startup ----
# model = CatBoostRegressor() ...
# model.load_model("catboost_ai_bina_price_per_m2.cbm")
# explainer = shap.TreeExplainer(model)

@app.post("/predict_explain", response_model=ExplanationResponse)
def predict_and_explain(features: ListingFeatures, listing_id: str | None = None):
    # Convert Pydantic model to DataFrame in correct column order
    data_dict = features.dict()
    # ensure all model feature columns exist (fill missing tags with 0, etc.)
    row_df = pd.DataFrame([data_dict])[X.columns]  # X.columns from training notebook

    explanation = build_explanation_json(row_df, listing_id=listing_id)
    return explanation

# If running locally:
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
