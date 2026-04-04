"""
utils/matching_pipeline.py
---------------------------
Orchestrates the entire end-to-end matching workflow:
  1. Preprocess resumes + JD
  2. Encode embeddings
  3. Store in FAISS + SQLite
  4. Train / use role classifier
  5. K-Means clustering
  6. Score + rank candidates
  7. Run Apriori on skill patterns
  8. Return all results + artefacts for the UI
"""

import uuid
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

from utils.preprocessing import preprocess_resume, preprocess_job_description
from utils.ml_engine import (
    encode_texts,
    compute_cosine_similarity,
    compute_final_score,
    compute_skill_overlap,
    compute_experience_score,
    cluster_resumes,
    generate_explanation,
    role_classifier,
)
from utils.data_mining import (
    run_apriori,
    get_skill_frequencies,
    profile_clusters,
    skill_gap_analysis,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_matching_pipeline(
    raw_resumes: List[Dict],
    jd_text: str,
    n_clusters: int = 4,
    min_support: float = 0.2,
    score_weights: Optional[Dict] = None,
) -> Dict:
    """
    Full pipeline.

    Parameters
    ----------
    raw_resumes  : list of dicts with keys {resume_id, name, raw_text}
    jd_text      : raw job description string
    n_clusters   : K for K-Means
    min_support  : Apriori support threshold
    score_weights: optional dict {semantic, skill, experience}

    Returns
    -------
    dict with keys:
      ranked_candidates, jd_info, apriori_itemsets, apriori_rules,
      skill_frequencies, cluster_profiles, skill_gap,
      embeddings, cluster_labels, resume_names
    """

    if not raw_resumes:
        return {"error": "No resumes provided."}
    if not jd_text.strip():
        return {"error": "Job description is empty."}

    # ------------------------------------------------------------------
    # Step 1 – Preprocess
    # ------------------------------------------------------------------
    logger.info("Step 1: Preprocessing …")
    processed_resumes = []
    for r in raw_resumes:
        processed = preprocess_resume(r["raw_text"])
        processed["resume_id"] = r.get("resume_id", str(uuid.uuid4())[:8])
        processed["name"]      = r.get("name", "Unknown")
        processed["filename"]  = r.get("filename", "")
        processed_resumes.append(processed)

    jd_info = preprocess_job_description(jd_text)

    # ------------------------------------------------------------------
    # Step 2 – Embeddings
    # ------------------------------------------------------------------
    logger.info("Step 2: Encoding embeddings …")
    resume_texts = [r["processed_text"] for r in processed_resumes]
    jd_processed = jd_info["processed_text"]

    all_texts      = resume_texts + [jd_processed]
    all_embeddings = encode_texts(all_texts)

    resume_embeddings = all_embeddings[:-1].astype("float32")
    jd_embedding      = all_embeddings[-1].astype("float32")

    # ------------------------------------------------------------------
    # Step 3 – Train role classifier (use sample labels if available)
    # ------------------------------------------------------------------
    logger.info("Step 3: Role classification …")
    role_labels = [r.get("role", "") for r in raw_resumes]
    if any(role_labels):
        valid_pairs = [(rt, rl) for rt, rl in zip(resume_texts, role_labels) if rl]
        if len(valid_pairs) >= 2:
            vt, vl = zip(*valid_pairs)
            role_classifier.train(list(vt), list(vl))

    for r, emb in zip(processed_resumes, resume_embeddings):
        r["predicted_role"] = role_classifier.predict(r["processed_text"])

    # ------------------------------------------------------------------
    # Step 4 – K-Means clustering
    # ------------------------------------------------------------------
    logger.info("Step 4: Clustering …")
    k = min(n_clusters, len(processed_resumes))
    cluster_labels_arr, _ = cluster_resumes(resume_embeddings, n_clusters=k)
    cluster_labels = cluster_labels_arr.tolist()

    for r, lbl in zip(processed_resumes, cluster_labels):
        r["cluster_label"] = int(lbl)

    # ------------------------------------------------------------------
    # Step 5 – Score every resume
    # ------------------------------------------------------------------
    logger.info("Step 5: Scoring …")
    req_exp    = jd_info.get("required_experience", 0.0)
    jd_skills  = jd_info.get("required_skills", [])

    ranked = []
    for r, r_emb, cl in zip(processed_resumes, resume_embeddings, cluster_labels):
        sem   = compute_cosine_similarity(r_emb, jd_embedding)
        skill = compute_skill_overlap(r.get("skills", []), jd_skills)
        exp   = compute_experience_score(r.get("experience_years", 0.0), req_exp)
        final = compute_final_score(sem, skill, exp, weights=score_weights)

        r["semantic_similarity"] = sem
        r["skill_overlap"]       = skill
        r["experience_score"]    = exp
        r["match_score"]         = final

        explanation = generate_explanation(r, jd_info, sem, final, int(cl))
        ranked.append({**r, **explanation})

    # Sort descending
    ranked.sort(key=lambda x: x["match_score"], reverse=True)

    # ------------------------------------------------------------------
    # Step 6 – Persist to DB (non-blocking — skip on failure)
    # ------------------------------------------------------------------
    _persist_to_db(ranked)

    # ------------------------------------------------------------------
    # Step 7 – Data mining
    # ------------------------------------------------------------------
    logger.info("Step 7: Data mining …")
    skill_lists = [r.get("skills", []) for r in processed_resumes]

    freq_itemsets, rules = run_apriori(
        skill_lists, min_support=min_support, min_confidence=0.4
    )
    skill_freq   = get_skill_frequencies(skill_lists)
    cluster_prof = profile_clusters(processed_resumes, cluster_labels)
    skill_gap    = skill_gap_analysis(jd_skills, skill_lists)

    # ------------------------------------------------------------------
    # Step 8 – FAISS store (non-blocking)
    # ------------------------------------------------------------------
    _store_faiss(resume_embeddings, [r["resume_id"] for r in processed_resumes])

    logger.info(f"Pipeline complete. {len(ranked)} candidates ranked.")

    return {
        "ranked_candidates": ranked,
        "jd_info":           jd_info,
        "apriori_itemsets":  freq_itemsets,
        "apriori_rules":     rules,
        "skill_frequencies": skill_freq,
        "cluster_profiles":  cluster_prof,
        "skill_gap":         skill_gap,
        "embeddings":        resume_embeddings,
        "cluster_labels":    cluster_labels,
        "resume_names":      [r["name"] for r in processed_resumes],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _persist_to_db(ranked_resumes: List[Dict]):
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from database.db_manager import save_resume
        for r in ranked_resumes:
            save_resume(r)
    except Exception as e:
        logger.warning(f"DB persist skipped: {e}")


def _store_faiss(embeddings: np.ndarray, resume_ids: List[str]):
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from database.db_manager import faiss_store
        faiss_store.add_embeddings(embeddings.copy(), resume_ids)
    except Exception as e:
        logger.warning(f"FAISS store skipped: {e}")
