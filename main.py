from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from catboost import CatBoostRegressor
import pandas as pd
import numpy as np

app = FastAPI(title="AI-Bina AVM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load model once at startup
# =========================================================
model = CatBoostRegressor()
model.load_model("ai_bina_catboost_avm.cbm")


# =========================================================
# Schemas
# =========================================================
class PropertyFeatures(BaseModel):
    location_name: str
    city_name: str
    area_m2: float
    rooms: int
    floor: int | None = None
    floor_count: int | None = None
    leased: bool | None = None
    has_mortgage: bool | None = None
    has_bill_of_sale: bool | None = None
    has_repair: bool | None = None
    paid_daily: bool | None = None
    is_business: bool | None = None
    vipped: bool | None = None
    featured: bool | None = None
    photos_count: int | None = None


class ExplanationResponse(BaseModel):
    listing_id: Optional[str] = None
    predictions: Dict[str, Any]
    model_info: Dict[str, Any]
    key_attributes: Dict[str, Any]
    top_positive_contributors: List[Dict[str, Any]]
    top_negative_contributors: List[Dict[str, Any]]
    all_contributors: List[Dict[str, Any]]


# =========================================================
# Helper – build model row (same logic as before)
# =========================================================
def make_feature_row(p: PropertyFeatures) -> pd.DataFrame:
    floor = p.floor or 0
    floor_count = p.floor_count or 1
    floor_ratio = floor / max(floor_count, 1)

    row = {
        "location_name": p.location_name,
        "city_name": p.city_name,
        "area_m2": p.area_m2,
        "rooms": p.rooms,
        "floor": floor,
        "floor_count": floor_count,
        "floor_ratio": floor_ratio,
        "leased": int(p.leased) if p.leased is not None else 0,
        "has_mortgage": int(p.has_mortgage) if p.has_mortgage is not None else 0,
        "has_bill_of_sale": int(p.has_bill_of_sale) if p.has_bill_of_sale is not None else 0,
        "has_repair": int(p.has_repair) if p.has_repair is not None else 0,
        "paid_daily": int(p.paid_daily) if p.paid_daily is not None else 0,
        "is_business": int(p.is_business) if p.is_business is not None else 0,
        "vipped": int(p.vipped) if p.vipped is not None else 0,
        "featured": int(p.featured) if p.featured is not None else 0,
        "photos_count": p.photos_count or 0,
    }

    df = pd.DataFrame([row])
    return df


# =========================================================
# Endpoints
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: PropertyFeatures):
    df = make_feature_row(p)
    y_log = model.predict(df)[0]
    price_per_m2 = float(np.exp(y_log))
    total_price = price_per_m2 * p.area_m2

    return {
        "price_per_m2": price_per_m2,
        "total_price": total_price,
        "min_price": total_price * 0.9,
        "max_price": total_price * 1.1,
    }


@app.post("/predict_explain", response_model=ExplanationResponse)
def predict_explain(
    p: PropertyFeatures,
    listing_id: Optional[str] = None,
) -> ExplanationResponse:
    """
    SAFE stub: wraps /predict result into the explanation JSON structure.
    No SHAP, no extra imports – cannot crash from missing libs.
    """
    preds = predict(p)

    # Neutral model_info for now
    model_info = {
        "base_price_per_m2": preds["price_per_m2"],
        "delta_price_per_m2": 0.0,
        "relative_position": "same",
    }

    key_attributes = {
        "location_name": p.location_name,
        "city_name": p.city_name,
        "rooms": p.rooms,
        "floor": p.floor,
        "floor_count": p.floor_count,
        "leased": p.leased,
        "has_repair": p.has_repair,
        "has_bill_of_sale": p.has_bill_of_sale,
        "has_mortgage": p.has_mortgage,
    }

    return ExplanationResponse(
        listing_id=listing_id,
        predictions={
            "currency": "AZN",
            "price_per_m2": preds["price_per_m2"],
            "total_price": preds["total_price"],
            "min_price": preds["min_price"],
            "max_price": preds["max_price"],
            "area_m2": p.area_m2,
        },
        model_info=model_info,
        key_attributes=key_attributes,
        top_positive_contributors=[],
        top_negative_contributors=[],
        all_contributors=[],
    )
