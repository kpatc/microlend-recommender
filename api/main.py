from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import uvicorn
import numpy as np

app = FastAPI(
    title="MicroLend Recommender API",
    description="Financial product recommendation for African SMEs",
    version="1.0.0",
)

_base = Path(__file__).parent

models_loaded = False
cf_model = None
cold_start_solver = None
hybrid_model = None
sme_features = None
user_item_matrix = None
model_metrics = {}


class RecommendRequest(BaseModel):
    sme_id: Optional[str] = None  # e.g. "SME_00042"
    sme_profile: Optional[dict] = None
    n: int = 5
    model: str = "hybrid"


class RecommendationItem(BaseModel):
    product_id: int
    product_name: str
    score: float
    explanation: str
    risk_level: str
    confidence: float


class SimilarSMEItem(BaseModel):
    sme_id: str
    similarity: float


@app.on_event("startup")
async def load_models():
    global models_loaded, cf_model, cold_start_solver, hybrid_model
    global sme_features, user_item_matrix

    try:
        import yaml
        from src.data.loader import DataLoader
        from src.models.user_based_cf import UserBasedCF
        from src.cold_start.solver import ColdStartSolver

        with open("configs/config.yaml") as f:
            config = yaml.safe_load(f)

        loader = DataLoader(config)
        sme_features = loader.load_sme_features()
        user_item_matrix = loader.load_user_item_matrix()

        cf_model = UserBasedCF(n_neighbors=20)
        cf_model.fit(user_item_matrix, sme_features)

        cold_start_solver = ColdStartSolver(n_neighbors=20)
        cold_start_solver.fit(sme_features, user_item_matrix)

        models_loaded = True
    except Exception as e:
        print(f"Warning: Could not load models — {e}. Run `make generate` first.")


PRODUCT_NAMES = {
    1: "Microcredit 3 months", 2: "Microcredit 12 months",
    3: "Agricultural insurance", 4: "Equipment leasing",
    5: "Group savings", 6: "Mobile payment setup",
    7: "Invoice financing", 8: "Crop advance loan",
}

RISK_LEVELS = {
    1: "low", 2: "medium", 3: "low", 4: "medium",
    5: "low", 6: "low", 7: "high", 8: "medium",
}


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(_base / "templates" / "index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": models_loaded}


@app.post("/recommend", response_model=List[RecommendationItem])
def recommend(request: RecommendRequest):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded. Run `make generate` first.")

    results = []

    if request.sme_id is not None:
        try:
            preds = cf_model.predict(request.sme_id, n_recommendations=request.n)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        for pid, score in preds:
            explanation = cf_model.explain(request.sme_id, pid)
            results.append(RecommendationItem(
                product_id=int(pid),
                product_name=PRODUCT_NAMES.get(int(pid), f"Product {pid}"),
                score=round(float(score), 3),
                explanation=explanation,
                risk_level=RISK_LEVELS.get(int(pid), "medium"),
                confidence=round(min(1.0, float(score) / 5), 3),
            ))

    elif request.sme_profile is not None:
        preds = cold_start_solver.recommend_new_sme(request.sme_profile, n=request.n)
        for p in preds:
            results.append(RecommendationItem(
                product_id=p["product_id"],
                product_name=p["product_name"],
                score=p["score"],
                explanation=f"Based on {p['n_similar_smes']} similar SMEs in our database.",
                risk_level=RISK_LEVELS.get(p["product_id"], "medium"),
                confidence=p["confidence"],
            ))
    else:
        raise HTTPException(status_code=400, detail="Provide either sme_id or sme_profile.")

    return results


@app.get("/similar-smes/{sme_id}", response_model=List[SimilarSMEItem])
def get_similar_smes(sme_id: str, n: int = 10):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    try:
        similar = cf_model.get_similar_smes(sme_id, n=n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return [SimilarSMEItem(sme_id=str(s), similarity=round(sim, 4)) for s, sim in similar]


@app.get("/product-associations/{product_id}")
def get_product_associations(product_id: int, n: int = 5):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    from src.models.item_based_cf import ItemBasedCF
    item_cf = ItemBasedCF()
    item_cf.fit(user_item_matrix)
    try:
        assocs = item_cf.get_product_associations(product_id, n=n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return [{"product_id": int(p), "similarity": round(sim, 4), "co_adoption_rate": round(rate, 4)}
            for p, sim, rate in assocs]


@app.get("/model-stats")
def model_stats():
    return {
        "models_loaded": models_loaded,
        "n_smes": int(len(sme_features)) if sme_features is not None else 0,
        "n_products": 8,
        "metrics": model_metrics,
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
