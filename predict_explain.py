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
    all_co_
