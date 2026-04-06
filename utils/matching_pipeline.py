"""End-to-end orchestration for resume-to-job matching."""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from schemas.domain import JobDescriptionRecord, ResumeRecord
from utils.data_mining import (
    get_skill_frequencies,
    profile_clusters,
    run_apriori,
    skill_gap_analysis,
)
from utils.ml_engine import (
    cluster_resumes,
    compute_ats_score,
    compute_cosine_similarity,
    compute_experience_score,
    compute_final_score,
    compute_skill_overlap,
    encode_texts,
    generate_explanation,
    role_classifier,
)
from utils.preprocessing import preprocess_job_description, preprocess_resume

logger = logging.getLogger(__name__)


def run_matching_pipeline(
    raw_resumes: List[Dict],
    jd_text: str,
    n_clusters: int = 4,
    min_support: float = 0.2,
    score_weights: Optional[Dict] = None,
) -> Dict:
    """Run the resume-to-job matching pipeline."""
    if not raw_resumes:
        return {"error": "No resumes provided."}
    if not jd_text.strip():
        return {"error": "Job description is empty."}

    logger.info("Step 1: Preprocessing ...")
    processed_resumes = _process_resumes(raw_resumes)
    jd_record = JobDescriptionRecord.from_dict(preprocess_job_description(jd_text))

    logger.info("Step 2: Encoding embeddings ...")
    resume_embeddings, jd_embedding = _build_embeddings(processed_resumes, jd_record)

    logger.info("Step 3: Role classification ...")
    _train_role_classifier(raw_resumes, processed_resumes)
    _assign_predicted_roles(processed_resumes)

    logger.info("Step 4: Clustering ...")
    cluster_labels = _assign_clusters(processed_resumes, resume_embeddings, n_clusters)

    logger.info("Step 5: Scoring ...")
    ranked_candidates = _rank_candidates(
        processed_resumes=processed_resumes,
        resume_embeddings=resume_embeddings,
        jd_record=jd_record,
        jd_embedding=jd_embedding,
        cluster_labels=cluster_labels,
        score_weights=score_weights,
    )

    _persist_to_db(ranked_candidates)

    logger.info("Step 6: Data mining ...")
    skill_lists = [resume.skills for resume in processed_resumes]
    frequent_itemsets, rules = run_apriori(skill_lists, min_support=min_support, min_confidence=0.4)
    skill_frequencies = get_skill_frequencies(skill_lists)
    cluster_profiles = profile_clusters([resume.to_dict() for resume in processed_resumes], cluster_labels)
    skill_gap = skill_gap_analysis(jd_record.required_skills, skill_lists)

    _store_faiss(resume_embeddings, [resume.resume_id for resume in processed_resumes])

    logger.info("Pipeline complete. %s candidates ranked.", len(ranked_candidates))
    return {
        "ranked_candidates": ranked_candidates,
        "jd_info": jd_record.to_dict(),
        "apriori_itemsets": frequent_itemsets,
        "apriori_rules": rules,
        "skill_frequencies": skill_frequencies,
        "cluster_profiles": cluster_profiles,
        "skill_gap": skill_gap,
        "embeddings": resume_embeddings,
        "cluster_labels": cluster_labels,
        "resume_names": [resume.name for resume in processed_resumes],
    }


def _process_resumes(raw_resumes: List[Dict]) -> List[ResumeRecord]:
    """Normalize incoming raw resumes into typed pipeline records."""
    processed_resumes: List[ResumeRecord] = []
    for raw_resume in raw_resumes:
        resume = ResumeRecord.from_raw(
            {
                "resume_id": raw_resume.get("resume_id", str(uuid.uuid4())[:8]),
                "name": raw_resume.get("name", "Unknown"),
                "filename": raw_resume.get("filename", ""),
                "raw_text": raw_resume.get("raw_text", ""),
            }
        )
        extracted = preprocess_resume(resume.raw_text)
        resume.processed_text = extracted.get("processed_text", "")
        resume.skills = extracted.get("skills", [])
        resume.experience_years = extracted.get("experience_years", 0.0)
        resume.education = extracted.get("education", "Not Specified")
        processed_resumes.append(resume)
    return processed_resumes


def _build_embeddings(
    processed_resumes: List[ResumeRecord],
    jd_record: JobDescriptionRecord,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode resume texts and the job description into embedding vectors."""
    all_texts = [resume.processed_text for resume in processed_resumes] + [jd_record.processed_text]
    all_embeddings = encode_texts(all_texts)
    return all_embeddings[:-1].astype("float32"), all_embeddings[-1].astype("float32")


def _train_role_classifier(raw_resumes: List[Dict], processed_resumes: List[ResumeRecord]) -> None:
    """Train the role classifier when labeled sample roles are available."""
    if role_classifier.trained:
        return

    labeled_pairs = [
        (resume.processed_text, raw_resume.get("role", ""))
        for raw_resume, resume in zip(raw_resumes, processed_resumes)
        if raw_resume.get("role")
    ]
    if len(labeled_pairs) < 2:
        return

    texts, labels = zip(*labeled_pairs)
    role_classifier.train(list(texts), list(labels))


def _assign_predicted_roles(processed_resumes: List[ResumeRecord]) -> None:
    """Populate predicted roles for each processed resume."""
    for resume in processed_resumes:
        resume.predicted_role = role_classifier.predict(resume.processed_text)


def _assign_clusters(
    processed_resumes: List[ResumeRecord],
    resume_embeddings: np.ndarray,
    n_clusters: int,
) -> List[int]:
    """Cluster resumes and store cluster labels on each record."""
    cluster_labels, _ = cluster_resumes(resume_embeddings, n_clusters=min(n_clusters, len(processed_resumes)))
    cluster_ids = cluster_labels.tolist()
    for resume, cluster_id in zip(processed_resumes, cluster_ids):
        resume.cluster_label = int(cluster_id)
    return cluster_ids


def _rank_candidates(
    processed_resumes: List[ResumeRecord],
    resume_embeddings: np.ndarray,
    jd_record: JobDescriptionRecord,
    jd_embedding: np.ndarray,
    cluster_labels: List[int],
    score_weights: Optional[Dict],
) -> List[Dict]:
    """Compute per-resume scores, explanations, and final ranking."""
    ranked_candidates: List[Dict] = []

    for resume, resume_embedding, cluster_id in zip(processed_resumes, resume_embeddings, cluster_labels):
        resume.semantic_similarity = compute_cosine_similarity(resume_embedding, jd_embedding)
        resume.skill_overlap = compute_skill_overlap(resume.skills, jd_record.required_skills)
        resume.experience_score = compute_experience_score(
            resume.experience_years,
            jd_record.required_experience,
        )
        resume.match_score = compute_final_score(
            resume.semantic_similarity,
            resume.skill_overlap,
            resume.experience_score,
            weights=score_weights,
        )
        ats_score, ats_breakdown = compute_ats_score(
            resume.to_dict(),
            jd_record.to_dict(),
            resume.semantic_similarity,
        )

        explanation = generate_explanation(
            resume.to_dict(),
            jd_record.to_dict(),
            resume.semantic_similarity,
            resume.match_score,
            ats_score,
            ats_breakdown,
            int(cluster_id),
        )
        ranked_candidates.append({**resume.to_dict(), **explanation})

    ranked_candidates.sort(key=lambda item: item["match_score"], reverse=True)
    return ranked_candidates


def _persist_to_db(ranked_resumes: List[Dict]) -> None:
    """Persist ranked resumes to SQLite without failing the pipeline."""
    try:
        from database.db_manager import save_resume

        for resume in ranked_resumes:
            save_resume(resume)
    except Exception as exc:
        logger.warning("DB persist skipped: %s", exc)


def _store_faiss(embeddings: np.ndarray, resume_ids: List[str]) -> None:
    """Store embeddings in FAISS without failing the pipeline."""
    try:
        from database.db_manager import faiss_store

        faiss_store.add_embeddings(embeddings.copy(), resume_ids)
    except Exception as exc:
        logger.warning("FAISS store skipped: %s", exc)
