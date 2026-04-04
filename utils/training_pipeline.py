"""
utils/training_pipeline.py
---------------------------
Train ML models on real CSV datasets (e.g. from Kaggle): role classification,
skill-frequency mining, and optional K-Means on resume text (unsupervised).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from utils.ml_engine import (
    DATASET_KMEANS_PATH,
    MINED_SKILLS_PATH,
    MODEL_DIR,
    reload_role_classifier,
    role_classifier,
)

logger = logging.getLogger(__name__)

# Default location for user-provided Kaggle / CSV exports
DEFAULT_KAGGLE_DIR = os.path.join("data", "kaggle")
DEFAULT_CSV_NAME = "resumes.csv"

TRAINING_META_PATH = os.path.join(MODEL_DIR, "training_meta.json")

# Flexible column name detection (common Kaggle / resume dataset conventions)
TEXT_COLUMNS = (
    "resume_text",
    "Resume",
    "resume",
    "text",
    "Text",
    "cv_text",
    "CV",
    "description",
    "Description",
    "summary",
)
ROLE_COLUMNS = (
    "role",
    "Role",
    "category",
    "Category",
    "job_title",
    "Job Title",
    "label",
    "Label",
    "position",
    "Position",
    "title",
    "Title",
)
SKILL_COLUMNS = (
    "skills",
    "Skills",
    "skill",
    "tech_stack",
    "Tech Stack",
    "keywords",
    "Keywords",
)


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_default_csv() -> str:
    """Path to the default bundled / user CSV under data/kaggle/."""
    return os.path.join(_project_root(), DEFAULT_KAGGLE_DIR, DEFAULT_CSV_NAME)


def _pick_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
        if cand in df.columns:
            return cand
    return None


def load_training_dataframe(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load CSV and normalize text / role / skills columns.

    Returns (dataframe with at least resume_text, role, skills_raw), info dict.
    """
    path = os.path.abspath(csv_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV has no rows.")

    text_col = _pick_column(df, TEXT_COLUMNS)
    role_col = _pick_column(df, ROLE_COLUMNS)
    skill_col = _pick_column(df, SKILL_COLUMNS)

    info: Dict[str, Any] = {
        "path": path,
        "text_column": text_col,
        "role_column": role_col,
        "skill_column": skill_col,
    }

    if not text_col:
        raise ValueError(
            "Could not find a resume text column. Expected one of: "
            + ", ".join(TEXT_COLUMNS[:8])
            + " ..."
        )
    if not role_col:
        raise ValueError(
            "Could not find a role / label column. Expected one of: "
            + ", ".join(ROLE_COLUMNS[:8])
            + " ..."
        )

    out = pd.DataFrame(
        {
            "resume_text": df[text_col].astype(str).fillna(""),
            "role": df[role_col].astype(str).fillna("").str.strip(),
        }
    )
    if skill_col:
        out["skills_raw"] = df[skill_col].astype(str).fillna("")
    else:
        out["skills_raw"] = ""

    # Drop rows with empty text or role
    out = out[(out["resume_text"].str.len() > 10) & (out["role"].str.len() > 0)]
    out = out.reset_index(drop=True)
    info["n_rows_after_clean"] = len(out)
    return out, info


def _split_skills(cell: str) -> List[str]:
    if not cell or not str(cell).strip():
        return []
    parts = re.split(r"[,;|/\n]+", str(cell))
    return [p.strip().lower() for p in parts if p.strip() and len(p.strip()) > 1]


def mine_skills_from_dataset(df: pd.DataFrame, top_n: int = 250) -> List[str]:
    """
    Build an extended skill vocabulary from dataset skill columns and
    frequent n-grams in resume text (data-driven, not static list-only).
    """
    counter: Counter[str] = Counter()
    for raw in df.get("skills_raw", pd.Series(dtype=str)):
        for s in _split_skills(str(raw)):
            counter[s] += 1

    # Supplement with frequent word / bigrams from resume bodies
    texts = df["resume_text"].tolist()
    if len(texts) >= 3:
        try:
            vec = CountVectorizer(
                max_features=120,
                ngram_range=(1, 2),
                min_df=max(1, min(2, len(texts) // 4)),
                stop_words="english",
            )
            vec.fit(texts)
            for term in vec.get_feature_names_out():
                t = str(term).lower().strip()
                if len(t) < 2 or len(t) > 40:
                    continue
                counter[t] += 1
        except Exception as exc:
            logger.warning("CountVectorizer skill mining skipped: %s", exc)

    # Keep tokens that look like skills (letters/digits; allow + # - /)
    cleaned: List[str] = []
    for term, _ in counter.most_common(top_n * 2):
        if not re.match(r"^[a-z0-9][a-z0-9+\-/#\. ]{0,38}$", term, re.I):
            continue
        cleaned.append(term.lower())
        if len(cleaned) >= top_n:
            break
    return list(dict.fromkeys(cleaned))


def _save_mined_skills(skills: List[str]) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "skills": skills,
        "count": len(skills),
    }
    with open(MINED_SKILLS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_training_meta(meta: Dict[str, Any]) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(TRAINING_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_training_pipeline(
    csv_path: Optional[str] = None,
    n_clusters: int = 8,
    top_skills: int = 250,
    min_samples: int = 2,
) -> Dict[str, Any]:
    """
    Train Logistic Regression role classifier on CSV data, mine skill vocabulary,
    fit K-Means on TF-IDF vectors (unsupervised), persist artifacts.

    If ``csv_path`` is None, uses ``data/kaggle/resumes.csv`` (see README).
    """
    csv_path = csv_path or resolve_default_csv()
    result: Dict[str, Any] = {
        "ok": False,
        "csv_path": os.path.abspath(csv_path),
        "n_samples": 0,
        "n_roles": 0,
        "classifier_trained": False,
        "kmeans_fit": False,
        "mined_skills_count": 0,
        "message": "",
    }

    try:
        df, col_info = load_training_dataframe(csv_path)
    except Exception as exc:
        result["message"] = str(exc)
        logger.error("Training data load failed: %s", exc)
        return result

    if len(df) < min_samples:
        result["message"] = (
            f"Need at least {min_samples} labeled rows after cleaning; got {len(df)}."
        )
        return result

    texts = df["resume_text"].tolist()
    labels = df["role"].tolist()

    # --- Supervised: role classifier (TF-IDF + Logistic Regression) ---
    role_classifier.train(texts, labels)
    result["classifier_trained"] = role_classifier.trained

    # --- Unsupervised: K-Means on same TF-IDF space as classifier ---
    kmeans_fit = False
    cluster_sizes: Dict[str, int] = {}
    try:
        if role_classifier.trained:
            X = role_classifier.vectorizer.transform(texts)
            k = max(2, min(n_clusters, len(df)))
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            labels_k = km.labels_
            for lab in np.unique(labels_k):
                cluster_sizes[int(lab)] = int(np.sum(labels_k == lab))
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(DATASET_KMEANS_PATH, "wb") as f:
                pickle.dump(km, f)
            kmeans_fit = True
    except Exception as exc:
        logger.warning("Dataset K-Means skipped: %s", exc)

    result["kmeans_fit"] = kmeans_fit
    result["cluster_sizes"] = cluster_sizes

    # --- Skill vocabulary from data ---
    mined = mine_skills_from_dataset(df, top_n=top_skills)
    _save_mined_skills(mined)
    result["mined_skills_count"] = len(mined)

    meta = {
        "csv_path": result["csv_path"],
        "columns": col_info,
        "n_samples": len(df),
        "n_unique_roles": int(df["role"].nunique()),
        "kmeans_clusters": len(cluster_sizes) if cluster_sizes else 0,
        "cluster_sizes": cluster_sizes,
    }
    _save_training_meta(meta)

    reload_role_classifier()
    try:
        from utils.preprocessing import reload_mined_skills_vocab

        reload_mined_skills_vocab()
    except Exception as exc:
        logger.warning("Could not reload preprocessing vocab: %s", exc)

    result["ok"] = True
    result["n_samples"] = len(df)
    result["n_roles"] = int(df["role"].nunique())
    result["message"] = (
        f"Trained on {len(df)} resumes, {result['n_roles']} role classes; "
        f"{len(mined)} mined skill terms; "
        f"K-Means={'ok' if kmeans_fit else 'skipped'}."
    )
    logger.info(result["message"])
    return result


def training_status() -> Dict[str, Any]:
    """Return whether persisted training artifacts exist."""
    from utils.ml_engine import CLASSIFIER_PATH, ENCODER_PATH, VECTORIZER_PATH

    clf_ok = all(os.path.exists(p) for p in (CLASSIFIER_PATH, VECTORIZER_PATH, ENCODER_PATH))
    return {
        "classifier_ready": clf_ok,
        "mined_skills_ready": os.path.isfile(MINED_SKILLS_PATH),
        "dataset_kmeans_ready": os.path.isfile(DATASET_KMEANS_PATH),
        "meta_ready": os.path.isfile(TRAINING_META_PATH),
        "default_csv": resolve_default_csv(),
        "default_csv_exists": os.path.isfile(resolve_default_csv()),
    }
