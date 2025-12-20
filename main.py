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
from pathlib import Path
import os

from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np

# =========================================================
# App
# =========================================================
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2_multihot190")
app = FastAPI(title=f"AI-Bina AVM API ({MODEL_VERSION})")

# =========================================================
# JSON safety
# =========================================================
def to_py(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if v is pd.NA:
        return None
    if isinstance(v, np.ndarray):
        return [to_py(x) for x in v.tolist()]
    return v

def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    return to_py(obj)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print("❌ UNHANDLED ERROR:", tb)
    return JSONResponse(
        status_code=500,
        content=safe_json({"error": str(exc), "path": str(request.url.path), "trace": tb}),
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Load schema + vocab + model (from GitHub Release assets)
# =========================================================
ART_DIR = Path(f"/app/artifacts/{MODEL_VERSION}")

SCHEMA_PATH = ART_DIR / "training_schema.json"
if not SCHEMA_PATH.exists():
    raise RuntimeError(f"Missing artifact: {SCHEMA_PATH}")

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    SCHEMA = json.load(f)

MODEL_PATH = ART_DIR / SCHEMA["model_file"]
VOCAB_PATH = ART_DIR / SCHEMA["vocab_file"]

for p in [MODEL_PATH, VOCAB_PATH]:
    if not p.exists():
        raise RuntimeError(f"Missing artifact: {p}")

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    VOCAB = json.load(f)

FEATURES: List[str] = list(SCHEMA["feature_order"])
NUMERIC = set(SCHEMA["base_numeric"])
CATEG = set(SCHEMA["base_categorical"])
MEDIANS: Dict[str, float] = {k: float(v) for k, v in SCHEMA["numeric_medians"].items()}
PREFIX: str = str(SCHEMA["microlocation_prefix"])
CAT_FEATURE_INDICES: List[int] = list(SCHEMA["cat_feature_indices"])
VOCAB_INDEX: Dict[str, int] = {str(t): i for i, t in enumerate(VOCAB)}

model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))

# =========================================================
# Normalizers
# =========================================================
Boolish = Union[str, bool, int, None]

def _norm_str(x: Boolish) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()

def normalize_property_type(value: Optional[str]) -> str:
    v = _norm_str(value)
    if v in {"new_apartment", "new", "yeni", "yeni tikili", "yeni tikili mənzil", "yeni tikili menzil"}:
        return "new_apartment"
    if v in {"old_apartment", "old", "köhnə", "kohne", "köhnə tikili", "kohne tikili", "köhnə tikili mənzil"}:
        return "old_apartment"
    if v in {"house", "həyət evi", "heyet evi", "bağ evi", "bag evi", "villa"}:
        return "old_apartment"
    return "new_apartment"

def normalize_repair(value: Boolish) -> str:
    """
    Must match TRAINING df['repair'] labels.
    Your curated dataset shows: 'repaired' and 'unknown'
    """
    v = _norm_str(value)
    if v in {"yes", "true", "1", "təmirli", "temirli", "repaired", "good", "əla", "ela", "yaxşı", "yaxsi"}:
        return "repaired"
    return "unknown"

def extract_microlocation_tags(
    microlocation: Optional[str],
    microlocations: Optional[List[str]],
    max_tags: int = 20,
) -> List[str]:
    if isinstance(microlocations, list) and microlocations:
        tags = [str(t).strip() for t in microlocations if str(t).strip()]
    else:
        s = (microlocation or "").strip()
        if not s:
            return []
        parts = re.split(r"[\*\|;,]+", s)
        tags = [p.strip() for p in parts if p.strip()]

    seen = set()
    uniq = []
    for t in tags:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq[:max_tags]

# =========================================================
# Request schema (keep backward compatible)
# =========================================================
class PropertyFeatures(BaseModel):
    area_m2: float = Field(..., gt=0)
    rooms: int = Field(..., ge=0)
    floor: Optional[int] = None
    floors_total: Optional[int] = None

    property_type: str

    microlocation: str = ""
    microlocations: Optional[List[str]] = None

    # NEW preferred field (future)
    repair: Optional[str] = None

    # Backward compatible old fields
    city: Optional[str] = None
    təmir: Boolish = None
    çıxarış: Boolish = None

# =========================================================
# Feature builder (schema-driven)
# =========================================================
def make_feature_row(p: PropertyFeatures) -> pd.DataFrame:
    row: Dict[str, Any] = {c: 0 for c in FEATURES}

    # numeric (use medians for missing)
    row["area_m2"] = float(p.area_m2)
    row["rooms"] = int(p.rooms)
    row["floor"] = float(p.floor) if p.floor is not None else MEDIANS.get("floor", 0.0)
    row["floors_total"] = float(p.floors_total) if p.floors_total is not None else MEDIANS.get("floors_total", 0.0)

    # categoricals
    row["property_type"] = normalize_property_type(p.property_type)

    # repair: prefer explicit repair field, fallback to təmir
    repair_in = p.repair if (p.repair is not None and str(p.repair).strip() != "") else p.təmir
    row["repair"] = normalize_repair(repair_in)

    # microlocations -> ml__0000..ml__XXXX
    tags = extract_microlocation_tags(p.microlocation, p.microlocations, max_tags=20)
    for t in set(tags):
        j = VOCAB_INDEX.get(str(t))
        if j is not None:
            row[f"{PREFIX}{j:04d}"] = 1

    df = pd.DataFrame([row], columns=FEATURES)

    # enforce categorical types
    for idx in CAT_FEATURE_INDICES:
        c = df.columns[idx]
        df[c] = df[c].fillna("missing").astype(str)

    # enforce numeric
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(MEDIANS.get(c, 0.0))

    return df

def _predict_unit_price(df: pd.DataFrame) -> float:
    pool = Pool(df, cat_features=CAT_FEATURE_INDICES)
    return float(model.predict(pool)[0])

def _shap_contributions(df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    pool = Pool(df, cat_features=CAT_FEATURE_INDICES)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")

    row_vals = shap_vals[0]
    contribs = row_vals[:-1]
    base_value = float(row_vals[-1])

    contrib_df = pd.DataFrame({"feature": FEATURES, "value": df.iloc[0].values, "contrib": contribs})
    contrib_df["abs_importance"] = np.abs(contrib_df["contrib"])
    contrib_df = contrib_df.sort_values("abs_importance", ascending=False)
    return base_value, contrib_df

def _to_items(df_sub: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    items = []
    for _, r in df_sub.head(limit).iterrows():
        items.append({
            "feature": str(to_py(r["feature"])),
            "value": safe_json(to_py(r["value"])),
            "contrib_azn_per_m2": float(to_py(r["contrib"])),
        })
    return items

def _group_contributions(contrib_df: pd.DataFrame) -> Dict[str, float]:
    ml_cols = [c for c in FEATURES if c.startswith(PREFIX)]
    groups = {
        "location": ml_cols,
        "size": ["area_m2", "rooms"],
        "building": ["floor", "floors_total"],
        "type": ["property_type"],
        "condition": ["repair"],
    }
    out = {}
    for g, feats in groups.items():
        out[g] = float(contrib_df.loc[contrib_df["feature"].isin(feats), "contrib"].sum())
    return out

def build_explain_response(p: PropertyFeatures) -> Dict[str, Any]:
    df = make_feature_row(p)

    price_per_m2 = float(_predict_unit_price(df))
    total_price = float(price_per_m2 * float(p.area_m2))

    base_value, contrib_df = _shap_contributions(df)

    pos_df = contrib_df[contrib_df["contrib"] > 0]
    neg_df = contrib_df[contrib_df["contrib"] < 0]

    tags = extract_microlocation_tags(p.microlocation, p.microlocations, max_tags=20)
    used_ml_cols = []
    for t in set(tags):
        idx = VOCAB_INDEX.get(str(t))
        if idx is not None:
            used_ml_cols.append(f"{PREFIX}{idx:04d}")

    return safe_json({
        "predictions": {
            "currency": "AZN",
            "price_per_m2": price_per_m2,
            "total_price": total_price,
            "min_price": total_price * 0.9,
            "max_price": total_price * 1.1,
            "area_m2": float(p.area_m2),
        },
        "explain": {
            "base_price_per_m2": float(base_value),
            "delta_price_per_m2": float(price_per_m2 - float(base_value)),
            "top_positive": _to_items(pos_df, 6),
            "top_negative": _to_items(neg_df, 6),
            "groups": _group_contributions(contrib_df),
            "microlocations_sent": [str(t) for t in tags],
            "ml_columns_used": used_ml_cols,
            "note": "CatBoost native SHAP. Units = AZN/m².",
        },
        "debug_model": {
            "schema_version": str(SCHEMA.get("version")),
            "n_features": int(len(FEATURES)),
            "n_cat_features": int(len(CAT_FEATURE_INDICES)),
            "artifacts_dir": str(ART_DIR),
            "model_file": str(SCHEMA.get("model_file")),
            "vocab_len": int(len(VOCAB)),
        },
    })

# =========================================================
# Endpoints
# =========================================================
@app.get("/health")
def health():
    return safe_json({
        "status": "ok",
        "model": "catboost-avm-unit-price",
        "schema_version": str(SCHEMA.get("version")),
        "n_features": len(FEATURES),
        "n_cat_features": len(CAT_FEATURE_INDICES),
        "has_ml_features": any(f.startswith(PREFIX) for f in FEATURES),
        "artifacts_dir": str(ART_DIR),
        "model_file": str(SCHEMA.get("model_file")),
        "vocab_len": int(len(VOCAB)),
    })

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

@app.post("/predict/explain")
def predict_explain(p: PropertyFeatures):
    try:
        return build_explain_response(p)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content=safe_json({"error": str(e)}))
