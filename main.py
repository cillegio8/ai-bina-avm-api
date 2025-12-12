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

# Use the model itself as the source of truth for feature order
MODEL_FEATURES: List[str] = list(model.feature_names_ or [])
if not MODEL_FEATURES:
    # fallback (rare), but better to fail loudly than mispredict
    raise RuntimeError("Model has no feature_names_. Re-train/export with feature names.")

# =========================================================
# Schemas (keep only what AVM model uses)
# =========================================================
class PropertyFeatures(BaseModel):
    location_name: str
    city_name: str
    area_m2: float
    rooms: int
    floor: int | None = None
    floor_count: int | None = None


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

    # Build only the base fields you know
    base_row: Dict[str, Any] = {
        "location_name": p.location_name,
        "city_name": p.city_name,
        "area_m2": p.area_m2,
        "rooms": p.rooms,
        "floor": floor,
        "floor_count": floor_count,
        "floor_ratio": floor_ratio,
    }

    # Now produce a row that matches MODEL_FEATURES exactly:
    # - keep only model-used keys
    # - fill missing ones with 0 (safe default for numeric/binary)
    row: Dict[str, Any] = {f: base_row.get(f, 0) for f in MODEL_FEATURES}

    return pd.DataFrame([row], columns=MODEL_FEATURES)


def catboost_contributions(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """
    Returns:
      base_log (bias term, log-space)
      contrib_df with per-feature contributions in log-space
    """
    pool = Pool(df)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")

    row_vals = shap_vals[0]
    contribs = row_vals[:-1]          # per-feature contributions
    base_log = float(row_vals[-1])    # bias term

    contrib_df = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "value": df.iloc[0].values,
        "contrib_log": contribs
    })
    contrib_df["abs_importance"] = np.abs(contrib_df["contrib_log"])
    contrib_df = contrib_df.sort_values("abs_importance", ascending=False)
    return base_log, contrib_df


def group_contribs_from_df(contrib_df: pd.DataFrame) -> Dict[str, float]:
    # Groups adjusted to the reduced feature set
    groups = {
        "location": ["location_name", "city_name"],
        "size": ["area_m2", "rooms"],
        "building": ["floor", "floor_count", "floor_ratio"],
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
            "note": "Contributions computed using CatBoost ShapValues in log-space.",
            "model_features": MODEL_FEATURES,
        },
        "key_attributes": {
            "location_name": p.location_name,
            "city_name": p.city_name,
            "rooms": p.rooms,
            "floor": p.floor,
            "floor_count": p.floor_count,
        },
        "top_positive_contributors": df_to_list(pos_df, limit=6),
        "top_negative_contributors": df_to_list(neg_df, limit=6),
        "all_contributors": df_to_list(contrib_df, limit=30),
        "group_contributions": {k: float(v) for k, v in group_contribs_log.items()},
    }


# =========================================================
# Endpoints
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok", "model_features": MODEL_FEATURES}


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
