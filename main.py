from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple, Union
import traceback
import re

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
# Model-driven schema (version-safe)
# =========================================================
def _get_model_feature_names(m: CatBoostRegressor) -> List[str]:
    # Older CatBoost often has feature_names_
    names = getattr(m, "feature_names_", None)
    if names:
        return list(names)

    # Some builds have get_feature_names()
    if hasattr(m, "get_feature_names"):
        try:
            return list(m.get_feature_names())  # type: ignore[attr-defined]
        except Exception:
            pass

    return []


def _get_model_cat_indices(m: CatBoostRegressor) -> List[int]:
    if hasattr(m, "get_cat_feature_indices"):
        try:
            return list(m.get_cat_feature_indices())  # type: ignore[attr-defined]
        except Exception:
            pass
    return []


MODEL_FEATURES: List[str] = _get_model_feature_names(model)
MODEL_CAT_INDICES: List[int] = _get_model_cat_indices(model)

if not MODEL_FEATURES:
    raise RuntimeError(
        "Could not read feature names from CatBoost model. "
        "Your CatBoost build does not expose feature_names_. "
        "Fix by upgrading catboost or hardcoding FEATURE_NAMES."
    )

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
    Normalize 'çıxarış' into model categories:
      - "var" | "yoxdur"
    """
    v = _norm_str(value)

    if v in {"yes", "true", "1", "var", "bəli", "beli", "kupca var", "çıxarış var", "cixarish var"}:
        return "var"
    if v in {"no", "false", "0", "yoxdur", "yox", "kupca yoxdur", "çıxarış yoxdur", "cixarish yoxdur", ""}:
        return "yoxdur"
    return "yoxdur"


def normalize_temir(value: Boolish) -> str:
    """
    Normalize 'təmir' into model categories:
      - "təmirli" | "təmir tələb edir"
    """
    v = _norm_str(value)

    if v in {"yes", "true", "1", "təmirli", "temirli", "bəli", "beli", "əla təmirli", "yaxşı təmirli"}:
        return "təmirli"

    if v in {
        "no", "false", "0",
        "təmir tələb edir", "temir teleb edir",
        "təmirsiz", "temirsiz",
        "natəmirsiz", "natemirsiz",
        "yarımçıq təmir", "temirsizdir",
        ""
    }:
        return "təmir tələb edir"

    return "təmir tələb edir"


def normalize_property_type(value: Optional[str]) -> str:
    """
    Normalize property_type into model categories.
    Known: new_apartment / old_apartment / house
    MVP: map house -> old_apartment
    """
    v = _norm_str(value)

    if v in {"new_apartment", "new", "yeni", "yeni tikili", "yeni tikili mənzil", "yeni tikili menzil"}:
        return "new_apartment"

    if v in {"old_apartment", "old", "köhnə", "kohne", "köhnə tikili", "kohne tikili", "köhnə tikili mənzil"}:
        return "old_apartment"

    if v in {"house", "həyət evi", "heyet evi", "bağ evi", "bag evi", "villa"}:
        return "old_apartment"

    return "new_apartment"


def sanitize_tag(tag: str) -> str:
    """
    Must match training sanitizer that produced columns like:
      tag_yasamal_r
    """
    tag = str(tag).lower().strip()
    tag = re.sub(r"\W+", "_", tag)
    tag = tag.strip("_")
    return tag or "unknown_tag"


def is_cat_feature_name(col: str) -> bool:
    """Check if a model feature name is categorical (by index)."""
    if not MODEL_CAT_INDICES:
        # If cat indices are unknown, we cannot reliably know from model
        return False
    try:
        idx = MODEL_FEATURES.index(col)
        return idx in MODEL_CAT_INDICES
    except ValueError:
        return False


def extract_microlocation_tags(
    microlocation: Optional[str],
    microlocations: Optional[List[str]],
    max_tags: int = 6,
) -> List[str]:
    """
    Returns a cleaned list of tags:
      - prefers microlocations[] if provided
      - otherwise parses microlocation string like:
            "Yasamal r.* Elmlər m.*"
            "Yasamal r., Elmlər m."
            "Yasamal r.; Elmlər m."
    """
    tags: List[str] = []

    if isinstance(microlocations, list) and len(microlocations) > 0:
        for t in microlocations:
            s = str(t).strip()
            if s:
                tags.append(s)
    else:
        s = (microlocation or "").strip()
        if s:
            parts = re.split(r"[\*\|;,]+", s)
            for p in parts:
                pp = p.strip()
                if pp:
                    tags.append(pp)

    # de-duplicate but keep order
    seen = set()
    uniq: List[str] = []
    for t in tags:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(t.strip())

    return uniq[:max_tags]


# =========================================================
# Schemas (request payload)
# =========================================================
class PropertyFeatures(BaseModel):
    area_m2: float = Field(..., gt=0)
    rooms: int = Field(..., ge=0)
    floor: Optional[int] = None
    floors_total: Optional[int] = None

    property_type: str
    city: str

    microlocation: str
    microlocations: Optional[List[str]] = None  # ✅ allow 3–4 items

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
# Cat feature indices fallback (if model doesn't expose them)
# =========================================================
def _cat_indices_for_df(df: pd.DataFrame) -> List[int]:
    # Prefer model-provided indices if present
    if MODEL_CAT_INDICES:
        return MODEL_CAT_INDICES
    # Fallback: infer categorical columns by dtype (object)
    return [i for i, c in enumerate(df.columns) if df[c].dtype == "object"]


# =========================================================
# Helpers
# =========================================================
def make_feature_row(p: PropertyFeatures) -> pd.DataFrame:
    """
    Build a DataFrame that matches the deployed model schema EXACTLY:
      - same columns
      - same order
    Supports tag_* multi-hot microlocation columns if present in the model.
    """

    pt = normalize_property_type(p.property_type)
    temir = normalize_temir(p.təmir)
    cixarish = normalize_cixarish(p.çıxarış)

    # init row with correct defaults
    row: Dict[str, Any] = {}
    for f in MODEL_FEATURES:
        # If we don't know cat indices, default to numeric 0 and overwrite cats later
        if MODEL_CAT_INDICES:
            row[f] = "" if is_cat_feature_name(f) else 0
        else:
            row[f] = 0

    # numeric
    if "area_m2" in row:
        row["area_m2"] = float(p.area_m2)
    if "rooms" in row:
        row["rooms"] = int(p.rooms)
    if "floor" in row:
        row["floor"] = int(p.floor) if p.floor is not None else 0
    if "floors_total" in row:
        row["floors_total"] = int(p.floors_total) if p.floors_total is not None else 0

    # categorical/base
    if "property_type" in row:
        row["property_type"] = pt
    if "city" in row:
        row["city"] = str(p.city)
    if "təmir" in row:
        row["təmir"] = temir
    if "çıxarış" in row:
        row["çıxarış"] = cixarish

    # ---- microlocation: primary + multi-tags ----
    tags = extract_microlocation_tags(p.microlocation, p.microlocations, max_tags=6)
    primary = tags[0] if tags else (p.microlocation or "")

    # categorical microlocation (if present)
    if "microlocation" in row:
        row["microlocation"] = str(primary)

    # tag_* one-hot (if present in model)
    for t in tags:
        col = f"tag_{sanitize_tag(t)}"
        if col in row:
            row[col] = 1

    # If we inferred cats, ensure string columns are actually strings (dtype object)
    # This helps Pool(cat_features=inferred) behave correctly.
    df = pd.DataFrame([[row[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)
    for i in _cat_indices_for_df(df):
        c = df.columns[i]
        df[c] = df[c].astype(str)

    return df


def predict_price_per_m2(df: pd.DataFrame) -> float:
    pool = Pool(df, cat_features=_cat_indices_for_df(df))
    return float(model.predict(pool)[0])


def catboost_contributions(df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """
    Returns:
      base_value: model expected value (bias)
      contrib_df: per-feature contributions (same units as target: AZN/m²)
    """
    pool = Pool(df, cat_features=_cat_indices_for_df(df))
    shap_vals = model.get_feature_importance(pool, type="ShapValues")

    row_vals = shap_vals[0]          # n_features + 1
    contribs = row_vals[:-1]         # per-feature
    base_value = float(row_vals[-1]) # expected value

    contrib_df = pd.DataFrame(
        {"feature": MODEL_FEATURES, "value": df.iloc[0].values, "contrib": contribs}
    )
    contrib_df["abs_importance"] = np.abs(contrib_df["contrib"])
    contrib_df = contrib_df.sort_values("abs_importance", ascending=False)
    return base_value, contrib_df


def group_contributions(contrib_df: pd.DataFrame) -> Dict[str, float]:
    tag_cols = [c for c in MODEL_FEATURES if c.startswith("tag_")]

    groups = {
        "location": [c for c in ["city", "microlocation"] if c in MODEL_FEATURES] + tag_cols,
        "size": [c for c in ["area_m2", "rooms"] if c in MODEL_FEATURES],
        "building": [c for c in ["floor", "floors_total"] if c in MODEL_FEATURES],
        "legal": [c for c in ["çıxarış"] if c in MODEL_FEATURES],
        "condition": [c for c in ["təmir"] if c in MODEL_FEATURES],
        "type": [c for c in ["property_type"] if c in MODEL_FEATURES],
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

    tags = extract_microlocation_tags(p.microlocation, p.microlocations, max_tags=6)
    used_tag_cols = []
    for t in tags:
        col = f"tag_{sanitize_tag(t)}"
        if col in MODEL_FEATURES:
            used_tag_cols.append(col)

    key_attrs = {
        "city": str(p.city),
        "property_type": normalize_property_type(p.property_type),
        "təmir": normalize_temir(p.təmir),
        "çıxarış": normalize_cixarish(p.çıxarış),
        "microlocation_primary": (tags[0] if tags else str(p.microlocation)),
        "microlocations_sent": tags,
        "tag_columns_used": used_tag_cols,
        "rooms": int(p.rooms),
        "floor": p.floor,
        "floors_total": p.floors_total,
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
    return {
        "status": "ok",
        "model": "catboost-avm-unit-price",
        "n_features": len(MODEL_FEATURES),
        "cat_indices_source": "model" if bool(MODEL_CAT_INDICES) else "inferred_from_df",
        "n_cat_features": len(MODEL_CAT_INDICES),
        "has_tag_features": any(f.startswith("tag_") for f in MODEL_FEATURES),
    }


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
