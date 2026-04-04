"""
utils/matching_pipeline.py
---------------------------
Orchestrates the end-to-end resume matching workflow.
"""

import uuid
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.preprocessing import preprocess_job_description, preprocess_resume
from utils.ml_engine import (
    cluster_resumes,
    compute_cosine_similarity,
    compute_experience_score,
    compute_final_score,
    compute_skill_overlap,
    encode_texts,
    generate_explanation,
    role_classifier,
)
from utils.data_mining import (
    get_skill_frequencies,
    profile_clusters,
    run_apriori,
    skill_gap_analysis,
)

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
    jd_info = preprocess_job_description(jd_text)

    logger.info("Step 2: Encoding embeddings ...")
    resume_embeddings, jd_embedding = _build_embeddings(processed_resumes, jd_info)

    logger.info("Step 3: Role classification ...")
    _train_role_classifier(raw_resumes, processed_resumes)
    _assign_predicted_roles(processed_resumes)

    logger.info("Step 4: Clustering ...")
    cluster_labels = _assign_clusters(processed_resumes, resume_embeddings, n_clusters)

    logger.info("Step 5: Scoring ...")
    ranked = _rank_candidates(
        processed_resumes=processed_resumes,
        resume_embeddings=resume_embeddings,
        jd_info=jd_info,
        jd_embedding=jd_embedding,
        cluster_labels=cluster_labels,
        score_weights=score_weights,
    )

    _persist_to_db(ranked)

    logger.info("Step 6: Data mining ...")
    skill_lists = [resume.get("skills", []) for resume in processed_resumes]
    frequent_itemsets, rules = run_apriori(skill_lists, min_support=min_support, min_confidence=0.4)
    skill_frequencies = get_skill_frequencies(skill_lists)
    cluster_profiles = profile_clusters(processed_resumes, cluster_labels)
    skill_gap = skill_gap_analysis(jd_info.get("required_skills", []), skill_lists)

    _store_faiss(resume_embeddings, [resume["resume_id"] for resume in processed_resumes])

    logger.info("Pipeline complete. %s candidates ranked.", len(ranked))
    return {
        "ranked_candidates": ranked,
        "jd_info": jd_info,
        "apriori_itemsets": frequent_itemsets,
        "apriori_rules": rules,
        "skill_frequencies": skill_frequencies,
        "cluster_profiles": cluster_profiles,
        "skill_gap": skill_gap,
        "embeddings": resume_embeddings,
        "cluster_labels": cluster_labels,
        "resume_names": [resume["name"] for resume in processed_resumes],
    }


def _process_resumes(raw_resumes: List[Dict]) -> List[Dict]:
    """Normalize incoming raw resumes into processed resume records."""
    processed_resumes: List[Dict] = []
    for raw_resume in raw_resumes:
        processed = preprocess_resume(raw_resume["raw_text"])
        processed.update(
            {
                "resume_id": raw_resume.get("resume_id", str(uuid.uuid4())[:8]),
                "name": raw_resume.get("name", "Unknown"),
                "filename": raw_resume.get("filename", ""),
            }
        )
        processed_resumes.append(processed)
    return processed_resumes


def _build_embeddings(processed_resumes: List[Dict], jd_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Encode resume texts and job description into embedding vectors."""
    resume_texts = [resume["processed_text"] for resume in processed_resumes]
    all_texts = resume_texts + [jd_info["processed_text"]]
    all_embeddings = encode_texts(all_texts)
    return all_embeddings[:-1].astype("float32"), all_embeddings[-1].astype("float32")


def _train_role_classifier(raw_resumes: List[Dict], processed_resumes: List[Dict]) -> None:
    """Train the role classifier when labeled sample roles are available."""
    if role_classifier.trained:
        # Persisted model (e.g. trained on Kaggle CSV) — do not overwrite with a tiny upload batch.
        return
    labeled_pairs = [
        (processed["processed_text"], raw.get("role", ""))
        for raw, processed in zip(raw_resumes, processed_resumes)
        if raw.get("role")
    ]
    if len(labeled_pairs) < 2:
        return

    texts, labels = zip(*labeled_pairs)
    role_classifier.train(list(texts), list(labels))


def _assign_predicted_roles(processed_resumes: List[Dict]) -> None:
    """Populate predicted roles for each processed resume."""
    for resume in processed_resumes:
        resume["predicted_role"] = role_classifier.predict(resume["processed_text"])


def _assign_clusters(processed_resumes: List[Dict], resume_embeddings: np.ndarray, n_clusters: int) -> List[int]:
    """Cluster resumes and store cluster labels on each record."""
    k = min(n_clusters, len(processed_resumes))
    cluster_labels, _ = cluster_resumes(resume_embeddings, n_clusters=k)
    cluster_ids = cluster_labels.tolist()
    for resume, cluster_id in zip(processed_resumes, cluster_ids):
        resume["cluster_label"] = int(cluster_id)
    return cluster_ids


def _rank_candidates(
    processed_resumes: List[Dict],
    resume_embeddings: np.ndarray,
    jd_info: Dict,
    jd_embedding: np.ndarray,
    cluster_labels: List[int],
    score_weights: Optional[Dict],
) -> List[Dict]:
    """Compute scores, explanations, and final ranking for each resume."""
    required_experience = jd_info.get("required_experience", 0.0)
    jd_skills = jd_info.get("required_skills", [])
    ranked: List[Dict] = []

    for resume, resume_embedding, cluster_id in zip(processed_resumes, resume_embeddings, cluster_labels):
        semantic_score = compute_cosine_similarity(resume_embedding, jd_embedding)
        skill_score = compute_skill_overlap(resume.get("skills", []), jd_skills)
        experience_score = compute_experience_score(resume.get("experience_years", 0.0), required_experience)
        final_score = compute_final_score(
            semantic_score,
            skill_score,
            experience_score,
            weights=score_weights,
        )

        resume["semantic_similarity"] = semantic_score
        resume["skill_overlap"] = skill_score
        resume["experience_score"] = experience_score
        resume["match_score"] = final_score

        explanation = generate_explanation(resume, jd_info, semantic_score, final_score, int(cluster_id))
        ranked.append({**resume, **explanation})

    ranked.sort(key=lambda item: item["match_score"], reverse=True)
    return ranked


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
