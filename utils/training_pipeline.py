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
    "candidate_summary",
    "profile",
    "objective",
    "content",
    "raw_text",
    "about",
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
    "designation",
    "Designation",
    "current_role",
    "target_role",
    "job_category",
)
SKILL_COLUMNS = (
    "skills",
    "Skills",
    "skill",
    "tech_stack",
    "Tech Stack",
    "keywords",
    "Keywords",
    "technical_skills",
    "competencies",
    "tools",
    "tool_list",
    "technology",
    "technologies",
    "software",
)

ROLE_INFERENCE_RULES = {
    "Machine Learning Engineer": (
        "mlops", "model serving", "model monitoring", "kubeflow", "mlflow",
        "feature store", "inference api", "sagemaker",
    ),
    "Data Scientist": (
        "machine learning", "scikit-learn", "tensorflow", "pytorch",
        "statistics", "predictive model", "nlp", "computer vision",
    ),
    "Data Engineer": (
        "spark", "airflow", "kafka", "etl", "data warehouse", "snowflake",
        "databricks", "bigquery", "redshift", "data lake",
    ),
    "Cloud DevOps Engineer": (
        "kubernetes", "terraform", "jenkins", "ci/cd", "prometheus",
        "grafana", "linux", "helm", "infrastructure",
    ),
    "Full Stack Developer": (
        "react", "node.js", "typescript", "html", "css", "graphql",
        "frontend", "full stack", "web application",
    ),
    "Software Engineer": (
        "java", "rest api", "microservices", "backend", "software engineer",
        "spring", "django", "fastapi",
    ),
    "Data Analyst": (
        "excel", "tableau", "power bi", "dashboard", "business intelligence",
        "kpi", "reporting", "data visualization",
    ),
    "Business Analyst": (
        "requirements", "stakeholder", "user stories", "process mapping",
        "jira", "business analyst", "documentation",
    ),
}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_default_csv() -> str:
    """Path to the default bundled / user CSV under data/kaggle/."""
    return os.path.join(_project_root(), DEFAULT_KAGGLE_DIR, DEFAULT_CSV_NAME)


def _pick_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    lower_map = {_normalize_column_name(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_column_name(cand)
        if key in lower_map:
            return lower_map[key]
    for column in df.columns:
        normalized = _normalize_column_name(column)
        for cand in candidates:
            cand_key = _normalize_column_name(cand)
            if cand_key and (cand_key in normalized or normalized in cand_key):
                return column
    return None


def _normalize_column_name(name: Any) -> str:
    """Normalize CSV headers so different naming styles can be matched."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _text_quality_score(series: pd.Series) -> float:
    """Score whether a column looks like resume body text."""
    values = series.dropna().astype(str)
    if values.empty:
        return 0.0
    sample = values.head(80)
    avg_len = float(sample.str.len().mean())
    long_ratio = float((sample.str.len() > 120).mean())
    unique_ratio = float(sample.nunique() / max(len(sample), 1))
    joined = " ".join(sample.head(20).str.lower().tolist())
    resume_terms = (
        "experience", "skills", "education", "project", "summary", "developed",
        "built", "python", "sql", "engineer", "analyst", "developer",
    )
    hits = sum(1 for term in resume_terms if term in joined)
    return avg_len * 0.02 + long_ratio * 8 + unique_ratio * 3 + hits


def _pick_text_column(df: pd.DataFrame) -> Optional[str]:
    """Pick a resume text column by name first, then by content shape."""
    named = _pick_column(df, TEXT_COLUMNS)
    if named:
        return named
    object_columns = [column for column in df.columns if df[column].dtype == "object"]
    if not object_columns:
        return None
    scored = sorted(
        ((column, _text_quality_score(df[column])) for column in object_columns),
        key=lambda item: item[1],
        reverse=True,
    )
    return scored[0][0] if scored and scored[0][1] > 5 else None


def _pick_skill_column(df: pd.DataFrame, text_col: Optional[str], role_col: Optional[str]) -> Optional[str]:
    named = _pick_column(df, SKILL_COLUMNS)
    if named:
        return named
    candidates = []
    for column in df.columns:
        if column in {text_col, role_col}:
            continue
        if df[column].dtype != "object":
            continue
        header = _normalize_column_name(column)
        sample = " ".join(df[column].dropna().astype(str).head(50).str.lower().tolist())
        delimiter_hits = sample.count(",") + sample.count(";") + sample.count("|")
        skill_hits = sum(1 for term in ["python", "sql", "java", "react", "aws", "excel", "docker"] if term in sample)
        header_hits = 4 if any(token in header for token in ["skill", "tool", "tech", "keyword", "competenc"]) else 0
        if delimiter_hits + skill_hits >= 3:
            candidates.append((column, delimiter_hits + skill_hits + header_hits))
    return max(candidates, key=lambda item: item[1])[0] if candidates else None


def _pick_role_column(df: pd.DataFrame, text_col: Optional[str]) -> Optional[str]:
    named = _pick_column(df, ROLE_COLUMNS)
    if named:
        return named
    best: Optional[Tuple[str, float]] = None
    for column in df.columns:
        if column == text_col or df[column].dtype != "object":
            continue
        sample = df[column].dropna().astype(str).str.strip()
        if sample.empty:
            continue
        unique_ratio = sample.nunique() / max(len(sample), 1)
        avg_len = sample.str.len().mean()
        joined = " ".join(sample.head(80).str.lower().tolist())
        role_hits = sum(1 for role in ROLE_INFERENCE_RULES for token in role.lower().split() if token in joined)
        score = role_hits + (4 if unique_ratio <= 0.35 else 0) + (2 if avg_len <= 45 else 0)
        if score > 3 and (best is None or score > best[1]):
            best = (column, score)
    return best[0] if best else None


def _combine_resume_text(df: pd.DataFrame, text_col: Optional[str], role_col: Optional[str], skill_col: Optional[str]) -> pd.Series:
    """Use a named text column, or combine useful object columns for odd CSVs."""
    if text_col:
        base = df[text_col].astype(str).fillna("")
    else:
        base = pd.Series([""] * len(df), index=df.index, dtype=str)

    useful_columns = [
        column for column in df.columns
        if column not in {text_col, role_col, skill_col} and df[column].dtype == "object"
    ]
    if not text_col and useful_columns:
        base = df[useful_columns].fillna("").astype(str).agg(" ".join, axis=1)
    elif skill_col:
        base = (base + " Skills: " + df[skill_col].astype(str).fillna("")).str.strip()

    for column in useful_columns:
        header = _normalize_column_name(column)
        if any(token in header for token in ["skill", "tool", "tech", "keyword", "competenc"]):
            base = (base + " " + df[column].astype(str).fillna("")).str.strip()
    return base


def _infer_role_from_text(text: str, fallback: str = "General Candidate") -> str:
    lowered = str(text or "").lower()
    scores: Dict[str, int] = {}
    for role, terms in ROLE_INFERENCE_RULES.items():
        scores[role] = sum(1 for term in terms if term in lowered)
    best_role, best_score = max(scores.items(), key=lambda item: item[1])
    return best_role if best_score > 0 else fallback


def _normalize_roles(raw_roles: pd.Series, resume_texts: pd.Series) -> pd.Series:
    roles = raw_roles.astype(str).fillna("").str.strip()
    invalid = roles.str.len().eq(0) | roles.str.lower().isin({"nan", "none", "unknown", "resume", "candidate"})
    if invalid.any():
        inferred = resume_texts.apply(_infer_role_from_text)
        roles = roles.mask(invalid, inferred)
    return roles


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

    text_col = _pick_text_column(df)
    role_col = _pick_role_column(df, text_col)
    skill_col = _pick_skill_column(df, text_col, role_col)

    info: Dict[str, Any] = {
        "path": path,
        "text_column": text_col,
        "role_column": role_col,
        "skill_column": skill_col,
    }

    if not text_col and not any(df[column].dtype == "object" for column in df.columns):
        raise ValueError(
            "Could not find any text-like columns to build resume text from."
        )

    resume_text = _combine_resume_text(df, text_col, role_col, skill_col)
    if role_col:
        roles = _normalize_roles(df[role_col], resume_text)
        role_source = "column"
    else:
        roles = resume_text.apply(_infer_role_from_text)
        role_source = "inferred_from_resume_text"

    out = pd.DataFrame(
        {
            "resume_text": resume_text.astype(str).fillna(""),
            "role": roles,
        }
    )
    if skill_col:
        out["skills_raw"] = df[skill_col].astype(str).fillna("")
    else:
        out["skills_raw"] = ""

    info["role_source"] = role_source

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
    if len(set(labels)) >= 2:
        role_classifier.train(texts, labels)
        result["classifier_trained"] = role_classifier.trained
    else:
        logger.warning("Classifier skipped because only one role class was available.")
        result["classifier_trained"] = False

    # --- Unsupervised: K-Means on same TF-IDF space as classifier ---
    kmeans_fit = False
    cluster_sizes: Dict[str, int] = {}
    try:
        if role_classifier.trained:
            X = role_classifier.vectorizer.transform(texts)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            X = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)).fit_transform(texts)
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
