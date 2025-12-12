from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple, Union
import traceback

from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np

# =========================================================
# App
# =========================================================
app = FastAPI(title="AI-Bina AVM API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https://.*\.local-credentialless\.webcontainer-api\.io$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load model (ONCE)
# =========================================================
model = CatBoostRegressor()
model.load_model("ai_bina_catboost_avm.cbm")

# =========================================================
# Feature definitions (MUST MATCH TRAINING)
# =========================================================
FEATURE_NUMERIC = [
    "area_m2",
    "rooms",
    "floor",
    "floors_total",
]

FEATURE_CATEGORICAL = [
    "property_type",
    "microlocation",
    "city",
    "təmir",
    "çıxarış",
]

FEATURE_COLUMNS = FEATURE_NUMERIC + FEATURE_CATEGORICAL

# Safer than passing names (works reliably across CatBoost versions)
CAT_FEATURE_INDICES = [FEATURE_COLUMNS.index(c) for c in FEATURE_CATEGORICAL]

# =========================================================
# Normalizers (frontend-friendly -> model-friendly)
# =========================================================
Boolish = Union[str, bool, int, None]


def _norm_str(x: Boolish) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def normalize_cixarish(value: Boolish) -> str:
    """
    Normalize 'çıxarış' into CatBoost expected categories:
      - "var" | "yoxdur"
    Accepts: Yes/No, true/false, 1/0, var/yoxdur, bəli/yox, etc.
    """
    v = _norm_str(value)

    if v in {"yes", "true", "1", "var", "bəli", "beli", "kupca var", "çıxarış var", "cixarish var"}:
        return "var"
    if v in {"no", "false", "0", "yoxdur", "yox", "kupca yoxdur", "çıxarış yoxdur", "cixarish yoxdur", ""}:
        return "yoxdur"

    # safe fallback
    return "yoxdur"


def normalize_temir(value: Boolish) -> str:
    """
    Normalize 'təmir' into CatBoost expected categories:
      - "təmirli" | "təmir tələb edir"
    Accepts: Yes/No, true/false, 1/0, təmirli/təmirsiz/natəmirsiz, etc.
    """
    v = _norm_str(value)

    # treated as renovated
    if v in {"yes", "true", "1", "təmirli", "temirli", "bəli", "beli", "əla təmirli", "yaxşı təmirli"}:
        return "təmirli"

    # treated as needs repair / not renovated
    if v in {
        "no", "false", "0",
        "təmir tələb edir", "temir teleb edir",
        "təmirsiz", "temirsiz",
        "natəmirsiz", "natemirsiz",
        "yarımçıq təmir", "temirsizdir",
        ""
    }:
        return "təmir tələb edir"

    # if unknown, default to conservative bucket
    return "təmir tələb edir"


def normalize_property_type(value: Optional[str]) -> str:
    """
    Normalize property_type into CatBoost expected categories.
    Model known: new_apartment / old_apartment / house
    MVP: keep only new_apartment / old_apartment (house -> old_apartment)
    """
    v = _norm_str(value)

    if v in {"new_apartment", "new", "yeni", "yeni tikili", "yeni tikili mənzil", "yeni tikili menzil"}:
        return "new_apartment"

    if v in {"old_apartment", "old", "köhnə", "kohne", "köhnə tikili", "kohne tikili", "köhnə tikili mənzil"}:
        return "old_apartment"

    if v in {"house", "həyət evi", "heyet evi", "bağ evi", "bag evi", "villa"}:
        # MVP does not use house; map to old_apartment for stability
        return "old_apartment"

    # fallback default (MVP)
    return "new_apartment"


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
    təmir: Boolish
    çıxarış: Boolish


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
    # normalize categorical fields to what CatBoost expects
    pt = normalize_property_type(p.property_type)
    temir = normalize_temir(p.təmir)
    cixarish = normalize_cixarish(p.çıxarış)

    row = {
        "area_m2": float(p.area_m2),
        "rooms": int(p.rooms),
        "floor": int(p.floor) if p.floor is not None else 0,
        "floors_total": int(p.floors_total) if p.floors_total is not None else 0,

        "property_type": pt,
        "microlocation": str(p.microlocation),
        "city": str(p.city),
        "təmir": temir,
        "çıxarış": cixarish,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_price_per_m2(df: pd.DataFrame) -> float:
    pool = Pool(df, cat_features=CAT_FEATURE_INDICES)
    return float(model.predict(pool)[0])


def catboost_contributions(df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """
    Returns:
      base_value: model expected value (bias)
      contrib_df: per-feature contributions (same units as target: AZN/m²)
    """
    pool = Pool(df, cat_features=CAT_FEATURE_INDICES)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")

    row_vals = shap_vals[0]               # n_features + 1
    contribs = row_vals[:-1]              # per-feature
    base_value = float(row_vals[-1])      # expected value

    contrib_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "value": df.iloc[0].values,
            "contrib": contribs,
        }
    )
    contrib_df["abs_importance"] = np.abs(contrib_df["contrib"])
    contrib_df = contrib_df.sort_values("abs_importance", ascending=False)
    return base_value, contrib_df


def group_contributions(contrib_df: pd.DataFrame) -> Dict[str, float]:
    groups = {
        "location": ["city", "microlocation"],
        "size": ["area_m2", "rooms"],
        "building": ["floor", "floors_total"],
        "legal": ["çıxarış"],
        "condition": ["təmir"],
        "type": ["property_type"],
    }

    out: Dict[str, float] = {}
    for group, feats in groups.items():
        out[group] = float(contrib_df.loc[contrib_df["feature"].isin(feats), "contrib"].sum())
    return out


def build_explanation_json(p: PropertyFeatures, listing_id: Optional[str] = None) -> Dict[str, Any]:
    df = make_feature_row(p)

    price_per_m2 = predict_price_per_m2(df)
    total_price = price_per_m2 * float(p.area_m2)

    base_value, contrib_df = catboost_contributions(df)

    delta = price_per_m2 - base_value
    relative_position = "higher" if delta > 0 else "lower" if delta < 0 else "same"

    def df_to_list(df_sub: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for _, r in df_sub.head(limit).iterrows():
            out.append(
                {
                    "feature": r["feature"],
                    "display_name": r["feature"],
                    "value": r["value"],
                    "contrib": float(r["contrib"]),
                    "contrib_azn_per_m2": float(r["contrib"]),
                    "abs_importance": float(r["abs_importance"]),
                }
            )
        return out

    pos_df = contrib_df[contrib_df["contrib"] > 0]
    neg_df = contrib_df[contrib_df["contrib"] < 0]

    # show normalized values in key_attributes (so frontend/debug sees what model actually used)
    key_attrs = {
        "city": str(p.city),
        "microlocation": str(p.microlocation),
        "property_type": normalize_property_type(p.property_type),
        "rooms": int(p.rooms),
        "floor": p.floor,
        "floors_total": p.floors_total,
        "təmir": normalize_temir(p.təmir),
        "çıxarış": normalize_cixarish(p.çıxarış),
    }

    return {
        "listing_id": listing_id,
        "predictions": {
            "currency": "AZN",
            "price_per_m2": price_per_m2,
            "total_price": total_price,
            "min_price": total_price * 0.9,
            "max_price": total_price * 1.1,
            "area_m2": float(p.area_m2),
        },
        "model_info": {
            "base_price_per_m2": base_value,
            "delta_price_per_m2": delta,
            "relative_position": relative_position,
            "note": "CatBoost native SHAP values. Target = unit_price (AZN/m²).",
        },
        "key_attributes": key_attrs,
        "top_positive_contributors": df_to_list(pos_df, 6),
        "top_negative_contributors": df_to_list(neg_df, 6),
        "all_contributors": df_to_list(contrib_df, 30),
        "group_contributions": group_contributions(contrib_df),
    }


# =========================================================
# Endpoints
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok", "model": "catboost-avm-unit-price"}


@app.post("/predict")
def predict(p: PropertyFeatures):
    try:
        df = make_feature_row(p)
        price_per_m2 = predict_price_per_m2(df)
        total_price = price_per_m2 * float(p.area_m2)

        return {
            "price_per_m2": price_per_m2,
            "total_price": total_price,
            "min_price": total_price * 0.9,
            "max_price": total_price * 1.1,
        }
    except Exception as e:
        tb = traceback.format_exc()
        print("❌ PREDICT ERROR:")
        print(tb)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_explain", response_model=ExplanationResponse)
def predict_explain(
    p: PropertyFeatures,
    listing_id: Optional[str] = Query(default=None),
):
    try:
        return ExplanationResponse(**build_explanation_json(p, listing_id=listing_id))
    except Exception as e:
        tb = traceback.format_exc()
        print("❌ PREDICT_EXPLAIN ERROR:")
        print(tb)
        raise HTTPException(status_code=500, detail=str(e))
