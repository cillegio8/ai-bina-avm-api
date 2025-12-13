from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple, Union
import traceback
import re
import uuid
import json

from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np

# =========================================================
# App
# =========================================================
app = FastAPI(title="AI-Bina AVM API (MVP)")

# =========================================================
# JSON safety (FIX for numpy/pandas types)
# =========================================================
def to_py(v: Any) -> Any:
    """Convert numpy/pandas scalars to native JSON-safe Python types."""
    # numpy scalars
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)

    # pandas NA
    if v is pd.NA:
        return None

    # numpy arrays
    if isinstance(v, np.ndarray):
        return [to_py(x) for x in v.tolist()]

    return v


def safe_json(obj: Any) -> Any:
    """
    Recursively convert dict/list contents to JSON-safe types.
    Ensures FastAPI encoder never sees numpy/pandas scalars.
    """
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    return to_py(obj)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print("❌ UNHANDLED ERROR:", tb)
    # keep trace for now; remove later
    return JSONResponse(
        status_code=500,
        content=safe_json(
            {
                "error": str(exc),
                "path": str(request.url.path),
                "trace": tb,
            }
        ),
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load model (ONCE)
# =========================================================
model = CatBoostRegressor()
model.load_model("/app/ai_bina_catboost_avm.cbm")


# =========================================================
# Model metadata (version-safe)
# =========================================================
def _get_model_feature_names(m: CatBoostRegressor) -> List[str]:
    names = getattr(m, "feature_names_", None)
    if names:
        return list(names)

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
        "Fix by upgrading catboost or hardcoding MODEL_FEATURES."
    )

# =========================================================
# Normalizers
# =========================================================
Boolish = Union[str, bool, int, None]


def _norm_str(x: Boolish) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def normalize_cixarish(value: Boolish) -> str:
    v = _norm_str(value)
    if v in {"yes", "true", "1", "var", "bəli", "beli", "kupca var", "çıxarış var", "cixarish var"}:
        return "var"
    if v in {"no", "false", "0", "yoxdur", "yox", "kupca yoxdur", "çıxarış yoxdur", "cixarish yoxdur", ""}:
        return "yoxdur"
    return "yoxdur"


def normalize_temir(value: Boolish) -> str:
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
    v = _norm_str(value)
    if v in {"new_apartment", "new", "yeni", "yeni tikili", "yeni tikili mənzil", "yeni tikili menzil"}:
        return "new_apartment"
    if v in {"old_apartment", "old", "köhnə", "kohne", "köhnə tikili", "kohne tikili", "köhnə tikili mənzil"}:
        return "old_apartment"
    if v in {"house", "həyət evi", "heyet evi", "bağ evi", "bag evi", "villa"}:
        return "old_apartment"  # MVP mapping
    return "new_apartment"


def sanitize_tag(tag: str) -> str:
    tag = str(tag).lower().strip()
    tag = re.sub(r"\W+", "_", tag).strip("_")
    return tag or "unknown_tag"


def extract_microlocation_tags(
    microlocation: Optional[str],
    microlocations: Optional[List[str]],
    max_tags: int = 6,
) -> List[str]:
    tags: List[str] = []

    if isinstance(microlocations, list) and microlocations:
        tags = [str(t).strip() for t in microlocations if str(t).strip()]
    else:
        s = (microlocation or "").strip()
        if s:
            parts = re.split(r"[\*\|;,]+", s)
            tags = [p.strip() for p in parts if p.strip()]

    seen = set()
    uniq: List[str] = []
    for t in tags:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(t)

    return uniq[:max_tags]

# =========================================================
# Schemas
# =========================================================
class PropertyFeatures(BaseModel):
    area_m2: float = Field(..., gt=0)
    rooms: int = Field(..., ge=0)
    floor: Optional[int] = None
    floors_total: Optional[int] = None

    property_type: str
    city: str

    microlocation: str
    microlocations: Optional[List[str]] = None

    təmir: Boolish
    çıxarış: Boolish

# =========================================================
# Core helpers
# =========================================================
def _cat_indices_for_df(df: pd.DataFrame) -> List[int]:
    if MODEL_CAT_INDICES:
        return MODEL_CAT_INDICES
    return [i for i, c in enumerate(df.columns) if df[c].dtype == "object"]


def make_feature_row(p: PropertyFeatures) -> pd.DataFrame:
    pt = normalize_property_type(p.property_type)
    temir = normalize_temir(p.təmir)
    cixarish = normalize_cixarish(p.çıxarış)

    row: Dict[str, Any] = {f: 0 for f in MODEL_FEATURES}

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

    # microlocation + tags
    tags = extract_microlocation_tags(p.microlocation, p.microlocations, max_tags=6)
    primary = tags[0] if tags else (p.microlocation or "")

    if "microlocation" in row:
        row["microlocation"] = str(primary)

    for t in tags:
        col = f"tag_{sanitize_tag(t)}"
        if col in row:
            row[col] = 1

    df = pd.DataFrame([[row[f] for f in MODEL_FEATURES]], columns=MODEL_FEATURES)

    # make categorical safe
    cat_idx = _cat_indices_for_df(df)
    for i in cat_idx:
        c = df.columns[i]
        df[c] = df[c].fillna("unknown").astype(str)

    # numeric NaN safe
    for c in df.columns:
        if c not in df.columns[cat_idx]:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(0)

    return df


def _predict_unit_price(df: pd.DataFrame) -> float:
    pool = Pool(df, cat_features=_cat_indices_for_df(df))
    return float(model.predict(pool)[0])


def _shap_contributions(df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    pool = Pool(df, cat_features=_cat_indices_for_df(df))
    shap_vals = model.get_feature_importance(pool, type="ShapValues")

    row_vals = shap_vals[0]          # n_features + 1
    contribs = row_vals[:-1]
    base_value = float(row_vals[-1])

    contrib_df = pd.DataFrame(
        {"feature": MODEL_FEATURES, "value": df.iloc[0].values, "contrib": contribs}
    )
    contrib_df["abs_importance"] = np.abs(contrib_df["contrib"])
    contrib_df = contrib_df.sort_values("abs_importance", ascending=False)
    return base_value, contrib_df


def _group_contributions(contrib_df: pd.DataFrame) -> Dict[str, float]:
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
    for g, feats in groups.items():
        s = contrib_df.loc[contrib_df["feature"].isin(feats), "contrib"].sum()
        out[g] = float(s)  # ensure python float
    return out


def _to_items(df_sub: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for _, r in df_sub.head(limit).iterrows():
        items.append({
            "feature": str(to_py(r["feature"])),
            "value": safe_json(to_py(r["value"])),
            "contrib_azn_per_m2": float(to_py(r["contrib"])),
        })
    return items


def build_explain_response(p: PropertyFeatures, listing_id: Optional[str] = None) -> Dict[str, Any]:
    df = make_feature_row(p)

    price_per_m2 = float(_predict_unit_price(df))
    total_price = float(price_per_m2 * float(p.area_m2))

    base_value, contrib_df = _shap_contributions(df)

    pos_df = contrib_df[contrib_df["contrib"] > 0]
    neg_df = contrib_df[contrib_df["contrib"] < 0]

    tags = extract_microlocation_tags(p.microlocation, p.microlocations, max_tags=6)
    tags = [str(t) for t in tags]

    used_tag_cols = [
        str(f"tag_{sanitize_tag(t)}")
        for t in tags
        if f"tag_{sanitize_tag(t)}" in MODEL_FEATURES
    ]

    response = {
        "listing_id": listing_id,
        "predictions": {
            "currency": "AZN",
            "price_per_m2": price_per_m2,
            "total_price": total_price,
            "min_price": float(total_price * 0.9),
            "max_price": float(total_price * 1.1),
            "area_m2": float(p.area_m2),
        },
        "explain": {
            "base_price_per_m2": float(base_value),
            "delta_price_per_m2": float(price_per_m2 - float(base_value)),
            "top_positive": _to_items(pos_df, 6),
            "top_negative": _to_items(neg_df, 6),
            "groups": _group_contributions(contrib_df),
            "microlocations_sent": tags,
            "tag_columns_used": used_tag_cols,
            "note": "CatBoost native SHAP. Units = AZN/m².",
        },
        "debug_model": {
            "n_features": int(len(MODEL_FEATURES)),
            "cat_indices_source": "model" if bool(MODEL_CAT_INDICES) else "inferred_from_df",
            "has_tag_features": bool(any(f.startswith("tag_") for f in MODEL_FEATURES)),
        },
    }

    # Final guarantee: nothing numpy/pandas leaks into output
    return safe_json(response)

# =========================================================
# Endpoints
# =========================================================
@app.get("/health")
def health():
    return safe_json({
        "status": "ok",
        "model": "catboost-avm-unit-price",
        "n_features": len(MODEL_FEATURES),
        "cat_indices_source": "model" if bool(MODEL_CAT_INDICES) else "inferred_from_df",
        "n_cat_features": len(MODEL_CAT_INDICES) if MODEL_CAT_INDICES else None,
        "has_tag_features": any(f.startswith("tag_") for f in MODEL_FEATURES),
    })


@app.post("/predict/explain")
def predict_explain(p: PropertyFeatures):
    try:
        return build_explain_response(p)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content=safe_json({"error": str(e)}))


@app.post("/predict")
def predict(p: PropertyFeatures):
    request_id = str(uuid.uuid4())
    try:
        df = make_feature_row(p)
        price_per_m2 = float(_predict_unit_price(df))
        total_price = float(price_per_m2 * float(p.area_m2))
        return safe_json({
            "price_per_m2": price_per_m2,
            "total_price": total_price,
            "min_price": total_price * 0.9,
            "max_price": total_price * 1.1,
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(f"❌ /predict ERROR request_id={request_id}\n{tb}")
        raise HTTPException(
            status_code=500,
            detail=safe_json({"message": str(e), "request_id": request_id}),
        )
