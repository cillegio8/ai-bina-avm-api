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

# ---------------------------------------------------------
# Optional SHAP (will not crash app if missing/broken)
# ---------------------------------------------------------
try:
    import shap
    explainer = shap.TreeExplainer(model)
    SHAP_AVAILABLE = True
    print("SHAP initialized")
except Exception as e:
    explainer = None
    SHAP_AVAILABLE = False
    print("SHAP disabled:", e)


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
# Features used in model
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
# Helpers
# =========================================================
def make_feature_row(p: PropertyFeatures) -> pd.DataFrame:
    """Build a single-row DataFrame in the same format used for training."""
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


def compute_shap_explanation(df: pd.DataFrame):
    """
    Returns:
      base_price_per_m2,
      contrib_sorted_df,
      group_contributions_dict
    or None if SHAP not available.
    """
    if not SHAP_AVAILABLE or explainer is None:
        return None

    try:
        shap_vals = explainer.shap_values(df)[0]  # 1D array
        base_value_log = float(explainer.expected_value)  # log(price_per_m2)
        base_price_per_m2 = float(np.exp(base_value_log))

        contrib_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "value": df.iloc[0].values,
            "shap_value": shap_vals,
        })
        contrib_df["abs_importance"] = contrib_df["shap_value"].abs()
        contrib_sorted = contrib_df.sort_values("abs_importance", ascending=False)

        # Grouping (Option C)
        groups = {
            "location": ["location_name", "city_name"],
            "size": ["area_m2", "rooms"],
            "building": ["floor", "floor_count", "floor_ratio"],
            "status": [
                "leased",
                "has_mortgage",
                "has_bill_of_sale",
                "has_repair",
                "paid_daily",
                "is_business",
            ],
            "marketing": ["vipped", "featured", "photos_count"],
        }

        group_contribs: Dict[str, float] = {}
        used_feats: List[str] = []

        for g, feats in groups.items():
            mask = contrib_df["feature"].isin(feats)
            used_feats.extend(feats)
            group_contribs[g] = float(contrib_df.loc[mask, "shap_value"].sum())

        # "other" group if there are any remaining features
        mask_other = ~contrib_df["feature"].isin(used_feats)
        if mask_other.any():
            group_contribs["other"] = float(
                contrib_df.loc[mask_other, "shap_value"].sum()
            )

        return base_price_per_m2, contrib_sorted, group_contribs

    except Exception as e:
        # Never crash app because of SHAP
        print("SHAP runtime error:", e)
        return None


def build_explanation_json(
    p: PropertyFeatures, listing_id: Optional[str] = None
) -> Dict[str, Any]:
    df = make_feature_row(p)

    # --- Prediction (same as /predict) ---
    y_log = model.predict(df)[0]
    price_per_m2 = float(np.exp(y_log))
    total_price = price_per_m2 * p.area_m2
    min_price = total_price * 0.9
    max_price = total_price * 1.1

    # Defaults (if SHAP not available)
    base_price_per_m2 = price_per_m2
    delta_price_per_m2 = 0.0
    relative_position = "same"
    top_pos: List[Dict[str, Any]] = []
    top_neg: List[Dict[str, Any]] = []
    all_contribs: List[Dict[str, Any]] = []
    group_contribs: Dict[str, float] = {}

    shap_result = compute_shap_explanation(df)
    if shap_result is not None:
        base_price_per_m2, contrib_sorted, group_contribs = shap_result
        delta_price_per_m2 = price_per_m2 - base_price_per_m2
        relative_position = "higher" if delta_price_per_m2 >= 0 else "lower"

        def df_to_list(sub_df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for _, r in sub_df.head(limit).iterrows():
                out.append(
                    {
                        "feature": r["feature"],
                        "display_name": r["feature"],  # map to pretty labels in frontend
                        "value": r["value"],
                        "shap_value": float(r["shap_value"]),
                        "abs_importance": float(r["abs_importance"]),
                    }
                )
            return out

        pos_df = contrib_sorted[contrib_sorted["shap_value"] > 0]
        neg_df = contrib_sorted[contrib_sorted["shap_value"] < 0]

        top_pos = df_to_list(pos_df, limit=6)
        top_neg = df_to_list(neg_df, limit=6)
        all_contribs = df_to_list(contrib_sorted, limit=30)

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
        "top_positive_contributors": top_pos,
        "top_negative_contributors": top_neg,
        "all_contributors": all_contribs,
        "group_contributions": group_contribs,
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
    Full valuation explanation:
    - prediction numbers
    - feature-level SHAP contributions (A)
    - grouped contributions (C) if SHAP is available.
    """
    return ExplanationResponse(**build_explanation_json(p, listing_id=listing_id))
