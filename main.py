from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np

app = FastAPI(title="AI-Bina AVM API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https://.*\.local-credentialless\.webcontainer-api\.io$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load model once at startup
# =========================================================
model = CatBoostRegressor()
model.load_model("ai_bina_catboost_avm.cbm")

# =========================================================
# Features used in model (MUST match training)
# =========================================================
FEATURE_NUMERIC = ["area_m2", "rooms", "floor", "floors_total"]
FEATURE_CATEGORICAL = ["property_type", "microlocation", "city", "təmir", "çıxarış"]
FEATURE_COLUMNS = FEATURE_NUMERIC + FEATURE_CATEGORICAL
CAT_FEATURES = FEATURE_CATEGORICAL  # pass names to Pool

# =========================================================
# Schemas
# =========================================================
class PropertyFeatures(BaseModel):
    area_m2: float = Field(..., gt=0)
    rooms: int = Field(..., ge=0)
    floor: Optional[int] = None
    floors_total: Optional[int] = None

    property_type: str
    microlocation: str
    city: str
    təmir: str
    çıxarış: str


class ExplanationResponse(BaseModel):
    listing_id: Optional[str] = None
    predictions: Dict[str, Any]
    model_info: Dict[str, Any]
    key_attributes: Dict[str, Any]
    top_positive_contributors: List[Dict[str, Any]]
    top_negative_contributors: List[Dict[str, Any]]
    all_contributors: List[Dict[str, Any]]
    group_contributions: Dict[str, float]


# =========================================================
# Helpers
# =========================================================
def make_feature_row(p: PropertyFeatures) -> pd.DataFrame:
    row: Dict[str, Any] = {
        # numeric
        "area_m2": float(p.area_m2),
        "rooms": int(p.rooms),
        "floor": int(p.floor) if p.floor is not None else 0,
        "floors_total": int(p.floors_total) if p.floors_total is not None else 0,

        # categorical (strings)
        "property_type": str(p.property_type or ""),
        "microlocation": str(p.microlocation or ""),
        "city": str(p.city or ""),
        "təmir": str(p.təmir or ""),
        "çıxarış": str(p.çıxarış or ""),
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def catboost_contributions(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
