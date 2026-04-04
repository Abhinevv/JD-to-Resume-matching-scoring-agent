"""
api/main_api.py
----------------
FastAPI backend exposing REST endpoints for the matching system.

Endpoints:
  POST /match       — upload JD + resumes, run pipeline, return results
  GET  /resumes     — list all stored resumes
  GET  /resume/{id} — get single resume details
  POST /sample      — load sample data and run pipeline
  GET  /health      — health check
"""

import os
import sys
import uuid
import json
import logging
import math
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Resolve project root so imports work regardless of CWD
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_ROOT = os.path.join(PROJECT_ROOT, "frontend")
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume–JD Matching API",
    description="Intelligent Resume Matching using NLP, ML, and Data Mining",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(FRONTEND_ROOT):
    app.mount("/assets", StaticFiles(directory=FRONTEND_ROOT), name="assets")


@app.get("/", include_in_schema=False)
def frontend():
    index_path = os.path.join(FRONTEND_ROOT, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend not found.")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/dashboard/meta")
def dashboard_meta():
    try:
        from database.db_manager import faiss_store, get_all_resumes
        resumes = get_all_resumes()
        return {
            "resume_count": len(resumes),
            "faiss_vectors": getattr(faiss_store, "size", len(getattr(faiss_store, "meta", []))),
        }
    except Exception as e:
        logger.error(e)
        return {"resume_count": 0, "faiss_vectors": 0}


# ---------------------------------------------------------------------------
# Main matching endpoint
# ---------------------------------------------------------------------------

@app.post("/match")
async def match_resumes(
    jd_text: str = Form(..., description="Job description text"),
    resumes: List[UploadFile] = File(..., description="Resume files (PDF/image/txt)"),
    n_clusters: int = Form(4),
    min_support: float = Form(0.2),
    sem_weight: float = Form(0.45),
    skill_weight: float = Form(0.35),
    exp_weight: float = Form(0.20),
):
    """
    Accept a job description + one or more resume files,
    run the full matching pipeline, and return ranked results.
    """
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    if not resumes:
        raise HTTPException(status_code=400, detail="Please upload at least one resume.")

    from utils.file_extractor import extract_text_from_bytes, save_uploaded_file
    from utils.matching_pipeline import run_matching_pipeline

    raw_resumes = []
    for upload in resumes:
        content = await upload.read()
        # Save to disk
        save_uploaded_file(content, upload.filename,
                           save_dir=os.path.join(PROJECT_ROOT, "data", "raw_resumes"))
        # Extract text
        text = extract_text_from_bytes(content, upload.filename)
        if not text.strip():
            logger.warning(f"No text extracted from {upload.filename}")
            text = f"[Could not extract text from {upload.filename}]"

        raw_resumes.append({
            "resume_id": str(uuid.uuid4())[:8],
            "name":      os.path.splitext(upload.filename)[0],
            "filename":  upload.filename,
            "raw_text":  text,
        })

    weights = {"semantic": sem_weight, "skill": skill_weight, "experience": exp_weight}

    result = run_matching_pipeline(
        raw_resumes=raw_resumes,
        jd_text=jd_text,
        n_clusters=n_clusters,
        min_support=min_support,
        score_weights=weights,
    )

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    # Serialise DataFrames to JSON-safe dicts
    result["skill_frequencies"] = _df_to_list(result.get("skill_frequencies"))
    result["apriori_itemsets"]  = _df_to_list(result.get("apriori_itemsets"))
    result["apriori_rules"]     = _df_to_list(result.get("apriori_rules"))
    result["skill_gap"]         = _df_to_list(result.get("skill_gap"))
    result.pop("embeddings", None)  # too large to return via REST

    return _clean_json_value(result)


# ---------------------------------------------------------------------------
# Sample data endpoint
# ---------------------------------------------------------------------------

@app.post("/sample")
async def run_sample(
    role: str = Form("data_scientist", description="'data_scientist' or 'data_engineer'"),
    n_clusters: int = Form(3),
    min_support: float = Form(0.25),
    sem_weight: float = Form(0.45),
    skill_weight: float = Form(0.35),
    exp_weight: float = Form(0.20),
):
    """Load built-in sample data and run the pipeline."""
    from utils.matching_pipeline import run_matching_pipeline
    from data.sample.sample_data import SAMPLE_RESUMES, SAMPLE_JOB_DESCRIPTIONS

    jd_text = SAMPLE_JOB_DESCRIPTIONS.get(role, SAMPLE_JOB_DESCRIPTIONS["data_scientist"])
    raw_resumes = [
        {
            "resume_id": r["id"],
            "name":      r["name"],
            "filename":  f"{r['name'].replace(' ', '_')}.txt",
            "raw_text":  r["text"],
            "role":      r["role"],
        }
        for r in SAMPLE_RESUMES
    ]

    result = run_matching_pipeline(
        raw_resumes=raw_resumes,
        jd_text=jd_text,
        n_clusters=n_clusters,
        min_support=min_support,
        score_weights={"semantic": sem_weight, "skill": skill_weight, "experience": exp_weight},
    )

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    result["skill_frequencies"] = _df_to_list(result.get("skill_frequencies"))
    result["apriori_itemsets"]  = _df_to_list(result.get("apriori_itemsets"))
    result["apriori_rules"]     = _df_to_list(result.get("apriori_rules"))
    result["skill_gap"]         = _df_to_list(result.get("skill_gap"))
    result.pop("embeddings", None)

    return _clean_json_value(result)


# ---------------------------------------------------------------------------
# Resume store endpoints
# ---------------------------------------------------------------------------

@app.get("/resumes")
def list_resumes():
    try:
        from database.db_manager import get_all_resumes
        return {"resumes": get_all_resumes()}
    except Exception as e:
        logger.error(e)
        return {"resumes": []}


@app.get("/resume/{resume_id}")
def get_resume(resume_id: str):
    try:
        from database.db_manager import get_resume_by_id
        r = get_resume_by_id(resume_id)
        if not r:
            raise HTTPException(status_code=404, detail="Resume not found.")
        return r
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_list(df) -> list:
    """Convert a pandas DataFrame to a JSON-safe list of dicts."""
    if df is None:
        return []
    try:
        # frozensets (from mlxtend) need converting
        if "itemsets" in df.columns:
            df = df.copy()
            df["itemsets"] = df["itemsets"].apply(lambda x: sorted(list(x)))
        if "antecedents" in df.columns:
            df = df.copy()
            df["antecedents"]  = df["antecedents"].apply(lambda x: sorted(list(x)))
            df["consequents"]  = df["consequents"].apply(lambda x: sorted(list(x)))
        return _clean_json_value(df.to_dict(orient="records"))
    except Exception:
        return []


def _clean_json_value(value):
    """Recursively replace non-finite floats with null for JSON responses."""
    if isinstance(value, dict):
        return {k: _clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.main_api:app", host="0.0.0.0", port=8000, reload=True)
