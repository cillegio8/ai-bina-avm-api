from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np

app = FastAPI(title="AI-Bina AVM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
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
# Features used in model (must match training)
# =========================================================
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
    group_contributions: Dict[str, float]


# =========================================================
# Helpers
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
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def catboost_contributions(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    Returns:
      base_log (bias term, log-space)
      contrib_df with per-feature contributions in log-space
    """
    pool = Pool(df)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")
    # shap_vals shape: (n_rows, n_features + 1) last column = expected value (bias) in same space as prediction
    row_vals = shap_vals[0]
    contribs = row_vals[:-1]          # per-feature contributions
    base_log = float(row_vals[-1])    # bias term

    contrib_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "value": df.iloc[0].values,
        "contrib_log": contribs
    })
    contrib_df["abs_importance"] = np.abs(contrib_df["contrib_log"])
    contrib_df = contrib_df.sort_values("abs_importance", ascending=False)
    return base_log, contrib_df


def group_contribs_from_df(contrib_df: pd.DataFrame) -> Dict[str, float]:
    groups = {
        "location": ["location_name", "city_name"],
        "size": ["area_m2", "rooms"],
        "building": ["floor", "floor_count", "floor_ratio"],
        "status": ["leased", "has_mortgage", "has_bill_of_sale", "has_repair", "paid_daily", "is_business"],
        "marketing": ["vipped", "featured", "photos_count"],
    }
    out: Dict[str, float] = {}
    for g, feats in groups.items():
        out[g] = float(contrib_df.loc[contrib_df["feature"].isin(feats), "contrib_log"].sum())
    return out


def build_explanation_json(p: PropertyFeatures, listing_id: Optional[str] = None) -> Dict[str, Any]:
    df = make_feature_row(p)

    # Predict in log space then exp
    y_log = float(model.predict(df)[0])
    price_per_m2 = float(np.exp(y_log))
    total_price = price_per_m2 * p.area_m2
    min_price = total_price * 0.9
    max_price = total_price * 1.1

    # CatBoost-native contributions (log-space)
    base_log, contrib_df = catboost_contributions(df)
    base_price_per_m2 = float(np.exp(base_log))
    delta_price_per_m2 = price_per_m2 - base_price_per_m2
    relative_position = "higher" if delta_price_per_m2 >= 0 else "lower"

    group_contribs_log = group_contribs_from_df(contrib_df)

    def df_to_list(sub_df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _, r in sub_df.head(limit).iterrows():
            out.append({
                "feature": r["feature"],
                "display_name": r["feature"],
                "value": r["value"],
                "contrib_log": float(r["contrib_log"]),
                # approx effect in AZN/m² (local linearization around prediction)
                "approx_impact_price_per_m2": float(price_per_m2 * (np.exp(float(r["contrib_log"])) - 1.0)),
                "abs_importance": float(r["abs_importance"]),
            })
        return out

    pos_df = contrib_df[contrib_df["contrib_log"] > 0]
    neg_df = contrib_df[contrib_df["contrib_log"] < 0]

    return {
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
            "note": "Contributions computed using CatBoost ShapValues in log-space (no shap python dependency).",
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
        "all_contributors": df_to_list(contrib_df, limit=30),
        # group contributions in log-space (sum of log contributions per group)
        "group_contributions": {k: float(v) for k, v in group_contribs_log.items()},
    }


# =========================================================
# Endpoints
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: PropertyFeatures):
    df = make_feature_row(p)
    y_log = float(model.predict(df)[0])
    price_per_m2 = float(np.exp(y_log))
    total_price = price_per_m2 * p.area_m2
    return {
        "price_per_m2": price_per_m2,
        "total_price": total_price,
        "min_price": total_price * 0.9,
        "max_price": total_price * 1.1,
    }


@app.post("/predict_explain", response_model=ExplanationResponse)
def predict_explain(p: PropertyFeatures, listing_id: Optional[str] = None) -> ExplanationResponse:
    return ExplanationResponse(**build_explanation_json(p, listing_id=listing_id))
