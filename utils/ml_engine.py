"""
utils/ml_engine.py
-------------------
Machine Learning core:
  - Sentence-transformer embeddings
  - Cosine similarity scoring
  - Logistic Regression role classifier
  - K-Means clustering
  - Weighted final scoring
  - Explainability report per resume
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Paths for persisted models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "role_classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH    = os.path.join(MODEL_DIR, "label_encoder.pkl")
KMEANS_PATH     = os.path.join(MODEL_DIR, "kmeans.pkl")
MINED_SKILLS_PATH = os.path.join(MODEL_DIR, "mined_skills.json")
DATASET_KMEANS_PATH = os.path.join(MODEL_DIR, "dataset_kmeans.pkl")


# ---------------------------------------------------------------------------
# Embedding engine
# ---------------------------------------------------------------------------

_SENTENCE_MODEL = None

def get_sentence_model():
    """Lazy-load the sentence-transformer model (cached after first load)."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence-transformer model loaded.")
        except Exception as e:
            logger.error(f"Could not load sentence-transformer: {e}")
    return _SENTENCE_MODEL


def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts into dense embeddings.
    Returns ndarray of shape (N, 384) — float32.
    Falls back to TF-IDF dense array if sentence-transformers unavailable.
    """
    model = get_sentence_model()
    if model is not None:
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.astype("float32")
    else:
        # Fallback: sparse TF-IDF → dense (lower quality but always works)
        logger.warning("Using TF-IDF fallback for embeddings.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=384)
        matrix = tfidf.fit_transform(texts).toarray().astype("float32")
        return matrix


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return cosine similarity (0-1) between two 1-D vectors."""
    a = vec_a.reshape(1, -1)
    b = vec_b.reshape(1, -1)
    score = cosine_similarity(a, b)[0][0]
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Role classifier  (Logistic Regression on TF-IDF)
# ---------------------------------------------------------------------------

class RoleClassifier:
    """
    Trains / loads a Logistic Regression classifier that maps
    resume text → predicted job role.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = LogisticRegression(max_iter=1000, C=1.0)
        self.encoder    = LabelEncoder()
        self.trained    = False
        self._try_load()

    def _try_load(self):
        try:
            if all(os.path.exists(p) for p in [CLASSIFIER_PATH, VECTORIZER_PATH, ENCODER_PATH]):
                with open(CLASSIFIER_PATH, "rb") as f:
                    self.classifier = pickle.load(f)
                with open(VECTORIZER_PATH, "rb") as f:
                    self.vectorizer = pickle.load(f)
                with open(ENCODER_PATH, "rb") as f:
                    self.encoder = pickle.load(f)
                self.trained = True
                logger.info("Loaded persisted role classifier.")
        except Exception as e:
            logger.warning(f"Could not load classifier: {e}")

    def train(self, texts: List[str], labels: List[str]):
        """Train on processed resume texts with known role labels."""
        if len(texts) < 2:
            logger.warning("Not enough samples to train classifier (need ≥ 2).")
            return
        try:
            y = self.encoder.fit_transform(labels)
            X = self.vectorizer.fit_transform(texts)
            self.classifier.fit(X, y)
            self.trained = True
            # Persist
            with open(CLASSIFIER_PATH, "wb") as f:
                pickle.dump(self.classifier, f)
            with open(VECTORIZER_PATH, "wb") as f:
                pickle.dump(self.vectorizer, f)
            with open(ENCODER_PATH, "wb") as f:
                pickle.dump(self.encoder, f)
            logger.info(f"Role classifier trained on {len(texts)} samples.")
        except Exception as e:
            logger.error(f"Training failed: {e}")

    def predict(self, text: str) -> str:
        """Return predicted role string, or 'Unknown' if not trained."""
        if not self.trained:
            return "Unknown"
        try:
            X = self.vectorizer.transform([text])
            y_pred = self.classifier.predict(X)
            return str(self.encoder.inverse_transform(y_pred)[0])
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return "Unknown"

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Return dict of {role: probability} for all classes."""
        if not self.trained:
            return {}
        try:
            X = self.vectorizer.transform([text])
            probs = self.classifier.predict_proba(X)[0]
            classes = self.encoder.inverse_transform(range(len(probs)))
            return {str(c): float(p) for c, p in zip(classes, probs)}
        except Exception:
            return {}


# Singleton
role_classifier = RoleClassifier()


def trained_classifier_ready() -> bool:
    """True if a persisted TF-IDF + Logistic Regression role model is on disk."""
    return all(
        os.path.exists(p) for p in (CLASSIFIER_PATH, VECTORIZER_PATH, ENCODER_PATH)
    )


def reload_role_classifier() -> None:
    """Reload the global role classifier after new training (picks up new pickles)."""
    global role_classifier
    role_classifier = RoleClassifier()


# ---------------------------------------------------------------------------
# K-Means clustering
# ---------------------------------------------------------------------------

def cluster_resumes(embeddings: np.ndarray, n_clusters: int = 4) -> Tuple[np.ndarray, KMeans]:
    """
    Cluster resume embeddings with K-Means.
    Returns (labels array, fitted KMeans object).
    """
    k = min(n_clusters, len(embeddings))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    # Persist
    with open(KMEANS_PATH, "wb") as f:
        pickle.dump(km, f)

    return labels, km


# ---------------------------------------------------------------------------
# Weighted scoring
# ---------------------------------------------------------------------------

def compute_final_score(
    semantic_similarity: float,
    skill_overlap: float,
    experience_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Combine sub-scores into a single final score (0-1).

    Default weights:
      - semantic_similarity : 0.45
      - skill_overlap        : 0.35
      - experience_score     : 0.20
    """
    if weights is None:
        weights = {"semantic": 0.45, "skill": 0.35, "experience": 0.20}

    score = (
        weights.get("semantic", 0.45)    * semantic_similarity +
        weights.get("skill", 0.35)       * skill_overlap       +
        weights.get("experience", 0.20)  * experience_score
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_skill_overlap(resume_skills: List[str], jd_skills: List[str]) -> float:
    """Jaccard-like overlap: |intersection| / |union|."""
    if not jd_skills:
        return 0.5   # neutral if JD has no skills listed
    r = set(s.lower() for s in resume_skills)
    j = set(s.lower() for s in jd_skills)
    intersection = r & j
    union = r | j
    return len(intersection) / len(union) if union else 0.0


def compute_experience_score(
    resume_years: float, required_years: float
) -> float:
    """
    Score how well the candidate's experience meets the requirement.
    Perfect if ≥ required; partial credit below; capped at 1.
    """
    if required_years <= 0:
        return 1.0
    ratio = resume_years / required_years
    # Reward meeting the bar, slight bonus for exceeding (up to 1.0 cap)
    return float(np.clip(ratio, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def generate_explanation(
    resume_data: Dict,
    jd_data: Dict,
    semantic_score: float,
    final_score: float,
    cluster_label: int = -1,
) -> Dict:
    """
    Produce a human-readable explanation for a resume's match score.

    Returns dict with:
      - match_score
      - semantic_similarity
      - matched_skills
      - missing_skills
      - extra_skills
      - experience_gap
      - experience_required / experience_found
      - education_match
      - recommendation  (Good / Average / Poor)
      - predicted_role
      - cluster_label
    """
    resume_skills = set(s.lower() for s in resume_data.get("skills", []))
    jd_skills     = set(s.lower() for s in jd_data.get("required_skills", []))

    matched_skills  = sorted(resume_skills & jd_skills)
    missing_skills  = sorted(jd_skills - resume_skills)
    extra_skills    = sorted(resume_skills - jd_skills)

    req_exp  = jd_data.get("required_experience", 0.0)
    res_exp  = resume_data.get("experience_years", 0.0)
    exp_gap  = max(0.0, req_exp - res_exp)

    if final_score >= 0.70:
        recommendation = "✅ Strong Match"
    elif final_score >= 0.45:
        recommendation = "⚠️  Moderate Match"
    else:
        recommendation = "❌ Weak Match"

    return {
        "match_score":           round(final_score, 4),
        "semantic_similarity":   round(semantic_score, 4),
        "matched_skills":        matched_skills,
        "missing_skills":        missing_skills,
        "extra_skills":          extra_skills[:10],   # top-10 to keep it readable
        "experience_required":   req_exp,
        "experience_found":      res_exp,
        "experience_gap":        round(exp_gap, 1),
        "education":             resume_data.get("education", "Not Specified"),
        "predicted_role":        resume_data.get("predicted_role", "Unknown"),
        "cluster_label":         cluster_label,
        "recommendation":        recommendation,
    }
