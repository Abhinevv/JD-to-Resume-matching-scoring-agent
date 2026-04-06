"""FastAPI entrypoint for the resume matching application."""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_ROOT = os.path.join(PROJECT_ROOT, "frontend")
RAW_RESUME_DIR = os.path.join(PROJECT_ROOT, "data", "raw_resumes")
sys.path.insert(0, PROJECT_ROOT)

from services.api_response import clean_json_value
from services.matching_service import (
    build_raw_resumes_from_uploads,
    build_sample_resumes,
    build_score_weights,
    get_sample_jd,
    run_match_workflow,
    run_training_workflow,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume-JD Matching API",
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
def frontend() -> FileResponse:
    """Serve the static frontend entrypoint."""
    index_path = os.path.join(FRONTEND_ROOT, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


@app.get("/health")
def health() -> dict:
    """Basic health endpoint for uptime checks."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/training/status")
def training_status_endpoint():
    """Report whether dataset-trained models and mined skills are present."""
    from utils.training_pipeline import training_status

    return training_status()


@app.post("/train")
async def train_on_dataset(
    csv_file: Optional[UploadFile] = File(None),
    n_clusters: int = Form(8),
    top_skills: int = Form(250),
):
    """Train dataset-backed models and mined-skill artifacts."""
    result = await run_training_workflow(
        csv_file=csv_file,
        n_clusters=n_clusters,
        top_skills=top_skills,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=422, detail=result.get("message", "Training failed."))
    return result


@app.get("/dashboard/meta")
def dashboard_meta():
    """Return lightweight dashboard counters."""
    try:
        from database.db_manager import faiss_store, get_all_resumes

        resumes = get_all_resumes()
        return {
            "resume_count": len(resumes),
            "faiss_vectors": getattr(faiss_store, "size", len(getattr(faiss_store, "meta", []))),
        }
    except Exception as exc:
        logger.error("Dashboard meta failed: %s", exc)
        return {"resume_count": 0, "faiss_vectors": 0}


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
    """Run the resume-matching workflow for uploaded files."""
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")
    if not resumes:
        raise HTTPException(status_code=400, detail="Please upload at least one resume.")

    raw_resumes = await build_raw_resumes_from_uploads(resumes, save_dir=RAW_RESUME_DIR)
    result = run_match_workflow(
        raw_resumes=raw_resumes,
        jd_text=jd_text,
        n_clusters=n_clusters,
        min_support=min_support,
        score_weights=build_score_weights(sem_weight, skill_weight, exp_weight),
    )
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.post("/sample")
async def run_sample(
    role: str = Form("data_scientist", description="'data_scientist' or 'data_engineer'"),
    n_clusters: int = Form(3),
    min_support: float = Form(0.25),
    sem_weight: float = Form(0.45),
    skill_weight: float = Form(0.35),
    exp_weight: float = Form(0.20),
):
    """Run the bundled sample dataset through the same matching workflow."""
    result = run_match_workflow(
        raw_resumes=build_sample_resumes(),
        jd_text=get_sample_jd(role),
        n_clusters=n_clusters,
        min_support=min_support,
        score_weights=build_score_weights(sem_weight, skill_weight, exp_weight),
    )
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.get("/resumes")
def list_resumes():
    """List all stored resume records."""
    try:
        from database.db_manager import get_all_resumes

        return {"resumes": get_all_resumes()}
    except Exception as exc:
        logger.error("List resumes failed: %s", exc)
        return {"resumes": []}


@app.get("/resume/{resume_id}")
def get_resume(resume_id: str):
    """Fetch one stored resume record."""
    try:
        from database.db_manager import get_resume_by_id

        resume = get_resume_by_id(resume_id)
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found.")
        return clean_json_value(resume)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("api.main_api:app", host="0.0.0.0", port=8000, reload=True)
