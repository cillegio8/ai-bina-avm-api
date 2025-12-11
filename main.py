from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np

# --- IMPORT THE ROUTER (must be BEFORE app.include_router) ---
from predict_explain import router as predict_explain_router


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="AI-Bina AVM API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Load Model Once at Startup
# ============================================================
model = CatBoostRegressor()
model.load_model("ai_bina_catboost_avm.cbm")


# ============================================================
# Schemas
# ============================================================
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


# ====================

