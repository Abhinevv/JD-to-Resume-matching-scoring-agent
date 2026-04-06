"""Thin orchestration helpers shared by API endpoints."""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional

from fastapi import UploadFile

from data.sample.sample_data import SAMPLE_JOB_DESCRIPTIONS, SAMPLE_RESUMES
from services.api_response import clean_json_value, dataframe_to_records
from utils.file_extractor import extract_text_from_bytes, save_uploaded_file
from utils.matching_pipeline import run_matching_pipeline
from utils.training_pipeline import run_training_pipeline


def build_score_weights(semantic: float, skill: float, experience: float) -> Dict[str, float]:
    """Normalize endpoint score weights into one dict."""
    return {
        "semantic": semantic,
        "skill": skill,
        "experience": experience,
    }


async def build_raw_resumes_from_uploads(
    uploads: List[UploadFile],
    save_dir: str,
) -> List[Dict[str, Any]]:
    """Save uploaded resumes and extract text for pipeline input."""
    raw_resumes: List[Dict[str, Any]] = []
    for upload in uploads:
        content = await upload.read()
        save_uploaded_file(content, upload.filename, save_dir=save_dir)
        extracted_text = extract_text_from_bytes(content, upload.filename)
        raw_resumes.append(
            {
                "resume_id": str(uuid.uuid4())[:8],
                "name": os.path.splitext(upload.filename)[0],
                "filename": upload.filename,
                "raw_text": extracted_text.strip() or f"[Could not extract text from {upload.filename}]",
            }
        )
    return raw_resumes


def build_sample_resumes() -> List[Dict[str, Any]]:
    """Create pipeline-ready sample resume payloads."""
    return [
        {
            "resume_id": resume["id"],
            "name": resume["name"],
            "filename": f"{resume['name'].replace(' ', '_')}.txt",
            "raw_text": resume["text"],
            "role": resume["role"],
        }
        for resume in SAMPLE_RESUMES
    ]


def get_sample_jd(role: str) -> str:
    """Return a bundled sample JD, defaulting to the data scientist template."""
    return SAMPLE_JOB_DESCRIPTIONS.get(role, SAMPLE_JOB_DESCRIPTIONS["data_scientist"])


def run_match_workflow(
    raw_resumes: List[Dict[str, Any]],
    jd_text: str,
    n_clusters: int,
    min_support: float,
    score_weights: Dict[str, float],
) -> Dict[str, Any]:
    """Run the matching pipeline and format the API payload."""
    result = run_matching_pipeline(
        raw_resumes=raw_resumes,
        jd_text=jd_text,
        n_clusters=n_clusters,
        min_support=min_support,
        score_weights=score_weights,
    )
    return serialize_pipeline_result(result)


def serialize_pipeline_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert pipeline output into a response-friendly payload."""
    if "error" in result:
        return result

    payload = dict(result)
    payload["skill_frequencies"] = dataframe_to_records(payload.get("skill_frequencies"))
    payload["apriori_itemsets"] = dataframe_to_records(payload.get("apriori_itemsets"))
    payload["apriori_rules"] = dataframe_to_records(payload.get("apriori_rules"))
    payload["skill_gap"] = dataframe_to_records(payload.get("skill_gap"))
    payload.pop("embeddings", None)
    return clean_json_value(payload)


async def run_training_workflow(
    csv_file: Optional[UploadFile],
    n_clusters: int,
    top_skills: int,
) -> Dict[str, Any]:
    """Run dataset training using either an uploaded CSV or the default file."""
    if csv_file is None or not csv_file.filename:
        return clean_json_value(
            run_training_pipeline(
                csv_path=None,
                n_clusters=n_clusters,
                top_skills=top_skills,
            )
        )

    suffix = os.path.splitext(csv_file.filename)[1] or ".csv"
    content = await csv_file.read()
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="train_")
    os.close(fd)

    try:
        with open(temp_path, "wb") as temp_file:
            temp_file.write(content)
        return clean_json_value(
            run_training_pipeline(
                csv_path=temp_path,
                n_clusters=n_clusters,
                top_skills=top_skills,
            )
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
