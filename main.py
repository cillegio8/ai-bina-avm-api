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
# Feature definitions (MUST match training)
# =========================================================
FEATURE_NUMERIC = ["area_m2", "rooms", "floor", "floors_total"]
FEATURE_CATEGORICAL = ["property_type", "microlocation", "city", "təmir", "çıxarış"]
FEATURE_COLUMNS = FEATURE_NUMERIC + FEATURE_CATEGORICAL

# CatBoost needs to know which columns are categorical
CAT_FEATURES = FEATURE_CATEGORICAL  # can be names


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
    # robust defaults
    floor = p.floor if p.floor is not None else 0
    floors_total = p.floors_total if p.floors_total is not None else 0

    row: Dict[str, Any] = {
        "area_m2": float(p.area_m2),
        "rooms": int(p.rooms),
        "floor": int(floor),
        "floors_total": int(floors_total),

        # categorical as strings (CatBoost handles categories)
        "property_type": str(p.property_type or ""),
        "microlocation": str(p.microlocation or ""),
        "city": str(p.city or ""),
        "təmir": str(p.təmir or ""),
        "çıxarış": str(p.çıxarış or ""),
    }

    # Ensure exact column order
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def catboost_contributions(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    pool = Pool(df, cat_features=CAT_FEATURES)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")

    row_vals = shap_vals[0]
    contribs = row_vals[:-1]
    base = float(row_vals[-1])

    contrib_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "value": df.iloc[0].values,
        "contrib": contribs
    })
    contrib_df["abs_importance"] = np.abs(contrib_df["contrib"])
    contrib_df = contrib_df.sort_values("abs_importance", ascending=False)
    return base, contrib_df


def group_contribs_from_df(contrib_df: pd.DataFrame) -> Dict[str, float]:
    groups = {
        "location": ["microlocation", "city"],
        "property": ["property_type"],
        "condition_docs": ["təmir", "çıxarış"],
        "size": ["area_m2", "rooms"],
        "building": ["floor", "floors_total"],
    }
    out: Dict[str, float] = {}
    for g, feats in groups.items():
        out[g] = float(contrib_df.loc[contrib_df["feature"].isin(feats), "contrib"].sum())
    return out


def build_explanation_json(p: PropertyFeatures, listing_id: Optional[str] = None) -> Dict[str, Any]:
    df = make_feature_row(p)

    pool = Pool(df, cat_features=CAT_FEATURES)
    pred = float(model.predict(pool)[0])  # note: this is whatever target space you trained on

    # If you trained on log(price_per_m2), keep exp here.
    # If you trained directly on AZN/m², remove exp.
    # -----
    # IMPORTANT: choose ONE and keep it consistent with training.
    price_per_m2 = float(np.exp(pred))  # <-- keep only if model trained on log
    # price_per_m2 = pred              # <-- use this if model trained directly on AZN/m²

    total_price = price_per_m2 * p.area_m2

    base, contrib_df = catboost_contributions(df)
    # base_price_per_m2 = float(np.exp(base))  # if log-trained
    base_price_per_m2 = float(np.exp(base))   # keep consistent with above

    group_contribs = group_contribs_from_df(contrib_df)

    def df_to_list(sub_df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _, r in sub_df.head(limit).iterrows():
            out.append({
                "feature": r["feature"],
                "value": r["value"],
                "contrib": float(r["contrib"]),
                "abs_importance": float(r["abs_importance"]),
            })
        return out

    pos_df = contrib_df[contrib_df["contrib"] > 0]
    neg_df = contrib_df[contrib_df["contrib"] < 0]

    return {
        "listing_id": listing_id,
        "predictions": {
            "currency": "AZN",
            "price_per_m2": price_per_m2,
            "total_price": total_price,
            "min_price": total_price * 0.9,
            "max_price": total_price * 1.1,
            "area_m2": p.area_m2,
        },
        "model_info": {
            "base_price_per_m2": base_price_per_m2,
            "note": "Contributions computed using CatBoost ShapValues.",
            "features": FEATURE_COLUMNS,
            "categorical_features": FEATURE_CATEGORICAL,
        },
        "key_attributes": {
            "property_type": p.property_type,
            "microlocation": p.microlocation,
            "city": p.city,
            "təmir": p.təmir,
            "çıxarış": p.çıxarış,
            "rooms": p.rooms,
            "floor": p.floor,
            "floors_total": p.floors_total,
            "area_m2": p.area_m2,
        },
        "top_positive_contributors": df_to_list(pos_df, limit=6),
        "top_negative_contributors": df_to_list(neg_df, limit=6),
        "all_contributors": df_to_list(contrib_df, limit=30),
        "group_contributions": {k: float(v) for k, v in group_contribs.items()},
    }


# =========================================================
# Endpoints
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "features": FEATURE_COLUMNS,
        "categorical_features": FEATURE_CATEGORICAL,
    }


@app.post("/predict")
def predict(p: PropertyFeatures):
    df = make_feature_row(p)
    pool = Pool(df, cat_features=CAT_FEATURES)
    pred = float(model.predict(pool)[0])

    price_per_m2 = float(np.exp(pred))  # keep consistent with training
    total_price = price_per_m2 * p.area_m2
    return {
        "price_per_m2": price_per_m2,
        "total_price": total_price,
        "min_price": total_price * 0.9,
        "max_price": total_price * 1.1,
    }


@app.post("/predict_explain", response_model=ExplanationResponse)
def predict_explain(p: PropertyFeatures, listing_id: Optional[str] = Query(default=None)) -> ExplanationResponse:
    return ExplanationResponse(**build_explanation_json(p, listing_id=listing_id))
