"""
utils/ml_engine.py
------------------
Machine-learning and scoring helpers used by the resume matching pipeline.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "role_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans.pkl")
MINED_SKILLS_PATH = os.path.join(MODEL_DIR, "mined_skills.json")
DATASET_KMEANS_PATH = os.path.join(MODEL_DIR, "dataset_kmeans.pkl")

_SENTENCE_MODEL = None


def get_sentence_model():
    """Lazy-load the sentence-transformer model."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer

            _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence-transformer model loaded.")
        except Exception as exc:
            logger.error("Could not load sentence-transformer: %s", exc)
    return _SENTENCE_MODEL


def encode_texts(texts: List[str]) -> np.ndarray:
    """Encode texts into dense vectors, falling back to TF-IDF if needed."""
    model = get_sentence_model()
    if model is not None:
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.astype("float32")

    logger.warning("Using TF-IDF fallback for embeddings.")
    tfidf = TfidfVectorizer(max_features=384)
    matrix = tfidf.fit_transform(texts).toarray().astype("float32")
    return matrix


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return cosine similarity in the range 0-1."""
    a = vec_a.reshape(1, -1)
    b = vec_b.reshape(1, -1)
    score = cosine_similarity(a, b)[0][0]
    return float(np.clip(score, 0.0, 1.0))


class RoleClassifier:
    """TF-IDF + Logistic Regression role classifier."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000, C=1.0)
        self.encoder = LabelEncoder()
        self.trained = False
        self._try_load()

    def _try_load(self):
        try:
            if all(os.path.exists(path) for path in [CLASSIFIER_PATH, VECTORIZER_PATH, ENCODER_PATH]):
                with open(CLASSIFIER_PATH, "rb") as file:
                    self.classifier = pickle.load(file)
                with open(VECTORIZER_PATH, "rb") as file:
                    self.vectorizer = pickle.load(file)
                with open(ENCODER_PATH, "rb") as file:
                    self.encoder = pickle.load(file)
                self.trained = True
                logger.info("Loaded persisted role classifier.")
        except Exception as exc:
            logger.warning("Could not load classifier: %s", exc)

    def train(self, texts: List[str], labels: List[str]):
        """Train and persist the role classifier."""
        if len(texts) < 2:
            logger.warning("Not enough samples to train classifier (need >= 2).")
            return
        if len(set(label for label in labels if str(label).strip())) < 2:
            logger.warning("Need at least two role classes to train classifier.")
            return
        try:
            encoded_labels = self.encoder.fit_transform(labels)
            features = self.vectorizer.fit_transform(texts)
            self.classifier.fit(features, encoded_labels)
            self.trained = True
            with open(CLASSIFIER_PATH, "wb") as file:
                pickle.dump(self.classifier, file)
            with open(VECTORIZER_PATH, "wb") as file:
                pickle.dump(self.vectorizer, file)
            with open(ENCODER_PATH, "wb") as file:
                pickle.dump(self.encoder, file)
            logger.info("Role classifier trained on %s samples.", len(texts))
        except Exception as exc:
            logger.error("Training failed: %s", exc)

    def predict(self, text: str) -> str:
        """Return the predicted role, or Unknown when unavailable."""
        if not self.trained:
            return "Unknown"
        try:
            features = self.vectorizer.transform([text])
            prediction = self.classifier.predict(features)
            return str(self.encoder.inverse_transform(prediction)[0])
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            return "Unknown"

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Return per-role probabilities."""
        if not self.trained:
            return {}
        try:
            features = self.vectorizer.transform([text])
            probabilities = self.classifier.predict_proba(features)[0]
            classes = self.encoder.inverse_transform(range(len(probabilities)))
            return {str(name): float(prob) for name, prob in zip(classes, probabilities)}
        except Exception:
            return {}


role_classifier = RoleClassifier()


def trained_classifier_ready() -> bool:
    """True if the persisted role model artifacts are present."""
    return all(os.path.exists(path) for path in (CLASSIFIER_PATH, VECTORIZER_PATH, ENCODER_PATH))


def reload_role_classifier() -> None:
    """Reload the global role classifier after training."""
    global role_classifier
    role_classifier = RoleClassifier()


def cluster_resumes(embeddings: np.ndarray, n_clusters: int = 4) -> Tuple[np.ndarray, KMeans]:
    """Cluster resume embeddings with K-Means."""
    cluster_count = min(n_clusters, len(embeddings))
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = model.fit_predict(embeddings)
    with open(KMEANS_PATH, "wb") as file:
        pickle.dump(model, file)
    return labels, model


def compute_final_score(
    semantic_similarity: float,
    skill_overlap: float,
    experience_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Combine sub-scores into one fit score."""
    if weights is None:
        weights = {"semantic": 0.45, "skill": 0.35, "experience": 0.20}

    score = (
        weights.get("semantic", 0.45) * semantic_similarity +
        weights.get("skill", 0.35) * skill_overlap +
        weights.get("experience", 0.20) * experience_score
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_skill_overlap(resume_skills: List[str], jd_skills: List[str]) -> float:
    """Compute Jaccard-like skill overlap."""
    if not jd_skills:
        return 0.5
    resume_skill_set = set(skill.lower() for skill in resume_skills)
    jd_skill_set = set(skill.lower() for skill in jd_skills)
    intersection = resume_skill_set & jd_skill_set
    union = resume_skill_set | jd_skill_set
    return len(intersection) / len(union) if union else 0.0


def compute_experience_score(resume_years: float, required_years: float) -> float:
    """Score how well the candidate meets the experience requirement."""
    if required_years <= 0:
        return 1.0
    ratio = resume_years / required_years
    return float(np.clip(ratio, 0.0, 1.0))


def compute_education_score(resume_education: str, required_education: str) -> float:
    """Score whether the candidate meets the minimum education requirement."""
    education_rank = {
        "Not Specified": 0,
        "High School": 1,
        "Diploma": 2,
        "Bachelor's": 3,
        "Master's": 4,
        "PhD": 5,
    }
    required = education_rank.get(required_education or "Not Specified", 0)
    found = education_rank.get(resume_education or "Not Specified", 0)
    if required <= 0:
        return 1.0
    if found >= required:
        return 1.0
    return float(np.clip(found / required, 0.0, 1.0))


def compute_keyword_coverage(resume_text: str, jd_skills: List[str]) -> float:
    """Measure how many JD skills appear directly in the resume text."""
    if not jd_skills:
        return 0.5
    text = (resume_text or "").lower()
    hits = sum(1 for skill in jd_skills if skill.lower().strip() and skill.lower() in text)
    return float(np.clip(hits / len(jd_skills), 0.0, 1.0))


def compute_ats_score(
    resume_data: Dict,
    jd_data: Dict,
    semantic_similarity: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute an ATS-style screening score.

    This is an explainable heuristic rather than a vendor-specific ATS formula:
      - required skill coverage: 55%
      - experience alignment:   20%
      - education alignment:    10%
      - keyword coverage:       15%
    """
    required_skills = jd_data.get("required_skills", [])
    resume_skills = resume_data.get("skills", [])

    if required_skills:
        matched_skill_count = len(
            set(skill.lower() for skill in resume_skills) &
            set(skill.lower() for skill in required_skills)
        )
        required_skill_coverage = matched_skill_count / len(required_skills)
    else:
        required_skill_coverage = 0.5

    experience_alignment = compute_experience_score(
        resume_data.get("experience_years", 0.0),
        jd_data.get("required_experience", 0.0),
    )
    education_alignment = compute_education_score(
        resume_data.get("education", "Not Specified"),
        jd_data.get("required_education", "Not Specified"),
    )
    keyword_coverage = compute_keyword_coverage(
        resume_data.get("raw_text", ""),
        required_skills,
    )

    ats_score = (
        0.55 * required_skill_coverage +
        0.20 * experience_alignment +
        0.10 * education_alignment +
        0.15 * max(keyword_coverage, semantic_similarity)
    )
    breakdown = {
        "required_skill_coverage": round(required_skill_coverage, 4),
        "experience_alignment": round(experience_alignment, 4),
        "education_alignment": round(education_alignment, 4),
        "keyword_coverage": round(keyword_coverage, 4),
    }
    return float(np.clip(ats_score, 0.0, 1.0)), breakdown


def compute_project_relevance_score(
    resume_data: Dict,
    jd_data: Dict,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a project relevance score using explicit project evidence.

    Formula:
      - project skill overlap: 50%
      - project keyword coverage: 30%
      - project implementation signal: 20%
    """
    project_text = resume_data.get("projects_text", "")
    project_skills = resume_data.get("project_skills", [])
    required_skills = jd_data.get("required_skills", [])

    if not project_text.strip():
        return 0.0, {
            "project_skill_overlap": 0.0,
            "project_keyword_coverage": 0.0,
            "project_signal": 0.0,
        }

    project_skill_overlap = compute_skill_overlap(project_skills, required_skills) if required_skills else 0.5
    project_keyword_coverage = compute_keyword_coverage(project_text, required_skills)
    project_signal = float(np.clip(resume_data.get("project_signal", 0.0), 0.0, 1.0))

    project_score = (
        0.50 * project_skill_overlap +
        0.30 * project_keyword_coverage +
        0.20 * project_signal
    )
    breakdown = {
        "project_skill_overlap": round(project_skill_overlap, 4),
        "project_keyword_coverage": round(project_keyword_coverage, 4),
        "project_signal": round(project_signal, 4),
    }
    return float(np.clip(project_score, 0.0, 1.0)), breakdown


def generate_explanation(
    resume_data: Dict,
    jd_data: Dict,
    semantic_score: float,
    final_score: float,
    ats_score: float = 0.0,
    ats_breakdown: Optional[Dict[str, float]] = None,
    project_score: float = 0.0,
    project_breakdown: Optional[Dict[str, float]] = None,
    cluster_label: int = -1,
) -> Dict:
    """Produce human-readable explanation details for a candidate."""
    resume_skills = set(skill.lower() for skill in resume_data.get("skills", []))
    jd_skills = set(skill.lower() for skill in jd_data.get("required_skills", []))

    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)
    extra_skills = sorted(resume_skills - jd_skills)

    required_experience = jd_data.get("required_experience", 0.0)
    found_experience = resume_data.get("experience_years", 0.0)
    experience_gap = max(0.0, required_experience - found_experience)

    if final_score >= 0.70:
        recommendation = "Strong Match"
    elif final_score >= 0.45:
        recommendation = "Moderate Match"
    else:
        recommendation = "Weak Match"

    if ats_score >= 0.75:
        ats_recommendation = "ATS Pass"
    elif ats_score >= 0.55:
        ats_recommendation = "ATS Borderline"
    else:
        ats_recommendation = "ATS Low Match"

    return {
        "match_score": round(final_score, 4),
        "ats_score": round(ats_score, 4),
        "semantic_similarity": round(semantic_score, 4),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills[:10],
        "experience_required": required_experience,
        "experience_found": found_experience,
        "experience_gap": round(experience_gap, 1),
        "education": resume_data.get("education", "Not Specified"),
        "education_required": jd_data.get("required_education", "Not Specified"),
        "predicted_role": resume_data.get("predicted_role", "Unknown"),
        "cluster_label": cluster_label,
        "ats_breakdown": ats_breakdown or {},
        "ats_recommendation": ats_recommendation,
        "project_score": round(project_score, 4),
        "project_breakdown": project_breakdown or {},
        "project_summary": resume_data.get("project_summary", ""),
        "project_skills": resume_data.get("project_skills", []),
        "recommendation": recommendation,
    }
