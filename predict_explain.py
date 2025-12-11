# predict_explain.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Any, Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import shap

# ---------- Router ----------
router = APIRouter()

# ---------- Pydantic models ----------

class ListingFeatures(BaseModel):
    # Same logical fields as /predict, but simplified
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


# ---------- Load model & SHAP explainer ----------

model = CatBoostRegressor()
model.load_model("ai_bina_catboost_avm.cbm")

explainer = shap.TreeExplainer(model)


# ---------- Helper: build DataFrame in SAME order as /predict ----------

FEATURE_COLUMNS = [
    "location_name",
    "city_name",
    "area_m2",
    "rooms",
    "floor",
    "floor_count",
    "floor_ratio",
    "leased",
    "has_mortgage",
    "has_bill_of_sale",
    "has_repair",
    "paid_daily",
    "is_business",
    "vipped",
    "featured",
    "photos_count",
]


def build_feature_row(p: ListingFeatures) -> pd.DataFrame:
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

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return df


# ---------- Helper: build explanation JSON ----------

def build_explanation_json(p: ListingFeatures, listing_id: str | None = None) -> Dict[str, Any]:
    df = build_feature_row(p)

    # Predict (same as /predict)
    y_log = model.predict(df)[0]
    price_per_m2 = float(np.exp(y_log))
    total_price = price_per_m2 * p.area_m2
    min_price = total_price * 0.9
    max_price = total_price * 1.1

    # SHAP values for this row
    shap_vals = explainer.shap_values(df)[0]  # 1D array
    base_value = float(explainer.expected_value)  # avg log-price-per-m2
    base_price_per_m2 = float(np.exp(base_value))  # convert to AZN/m²
    delta_price_per_m2 = price_per_m2 - base_price_per_m2
    relative_position = "higher" if delta_price_per_m2 >= 0 else "lower"

    contrib_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "value": df.iloc[0].values,
        "shap_value": shap_vals,
    })
    contrib_df["abs_importance"] = contrib_df["shap_value"].abs()
    contrib_sorted = contrib_df.sort_values("abs_importance", ascending=False)

    def df_to_list(df, limit):
        out: List[Dict[str, Any]] = []
        for _, r in df.head(limit).iterrows():
            out.append({
                "feature": r["feature"],
                "display_name": r["feature"],  # frontend can map to human labels
                "value": r["value"],
                "shap_value_price_per_m2": float(r["shap_value"]),
                "abs_importance": float(r["abs_importance"]),
            })
        return out

    pos_df = contrib_sorted[contrib_sorted["shap_value"] > 0]
    neg_df = contrib_sorted[contrib_sorted["shap_value"] < 0]

    explanation = {
        "listing_id": listing_id,
        "predictions": {
            "currency": "AZN",
            "price_per_m2": price_per_m2,
            "total_price": total_price,
            "min_price": min_price,
            "max_price": max_price,
            "area_m2": p.area_m2,
        },
        "model_info": {
            "base_price_per_m2": base_price_per_m2,
            "delta_price_per_m2": delta_price_per_m2,
            "relative_position": relative_position,
        },
        "key_attributes": {
            "location_name": p.location_name,
            "city_name": p.city_name,
            "rooms": p.rooms,
            "floor": p.floor,
            "floor_count": p.floor_count,
            "leased": p.leased,
            "has_repair": p.has_repair,
            "has_bill_of_sale": p.has_bill_of_sale,
            "has_mortgage": p.has_mortgage,
        },
        "top_positive_contributors": df_to_list(pos_df, limit=6),
        "top_negative_contributors": df_to_list(neg_df, limit=6),
        "all_contributors": df_to_list(contrib_sorted, limit=30),
    }

    return explanation


# ---------- Endpoint ----------

@router.post("/predict_explain", response_model=ExplanationResponse)
def predict_and_explain(
    features: ListingFeatures,
    listing_id: str | None = None
) -> Dict[str, Any]:
    return build_explanation_json(features, listing_id=listing_id)

