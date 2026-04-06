"""
utils/preprocessing.py
-----------------------
Handles all text cleaning, normalization, tokenization,
stopword removal, and lemmatization tasks.
"""

import json
import os
import re
import string
import unicodedata
from typing import Dict, List, Optional

# We use NLTK for tokenization and stopwords; spaCy is optional for lemmatization
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# One-time downloads (will be skipped if already present)
# ---------------------------------------------------------------------------
def _ensure_nltk_data():
    resources = ["punkt", "stopwords", "averaged_perceptron_tagger", "punkt_tab"]
    for r in resources:
        try:
            nltk.data.find(f"tokenizers/{r}")
        except Exception:
            try:
                nltk.download(r, quiet=True)
            except Exception:
                pass

_ensure_nltk_data()

# ---------------------------------------------------------------------------
# spaCy model loader — falls back gracefully if model not installed
# ---------------------------------------------------------------------------
_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            try:
                _NLP = spacy.load("en_core_web_sm")
            except OSError:
                # Model not installed — use a blank English model as a fallback
                _NLP = spacy.blank("en")
        except Exception:
            # If spaCy or one of its native deps fails to import, disable it.
            _NLP = False
    return _NLP


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
try:
    STOP_WORDS = set(stopwords.words("english")) if stopwords else set()
except Exception:
    STOP_WORDS = set()

# Skills taxonomy for extraction
SKILL_KEYWORDS = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "scala",
    "golang", "rust", "swift", "kotlin", "ruby", "php", "matlab",
    # ML / AI
    "machine learning", "deep learning", "neural network", "nlp",
    "computer vision", "reinforcement learning", "tensorflow", "pytorch",
    "keras", "scikit-learn", "xgboost", "lightgbm", "transformers", "bert",
    "gpt", "llm", "huggingface", "langchain", "rag",
    # Data engineering
    "spark", "hadoop", "kafka", "airflow", "flink", "databricks",
    "snowflake", "redshift", "bigquery", "dbt", "etl", "elt",
    # Databases
    "sql", "postgresql", "mysql", "mongodb", "redis", "cassandra",
    "elasticsearch", "neo4j", "dynamodb", "sqlite",
    # Cloud
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "jenkins", "ci/cd", "git", "linux",
    # Data / Analytics
    "pandas", "numpy", "matplotlib", "seaborn", "tableau", "power bi",
    "excel", "statistics", "a/b testing", "data visualization",
    "data warehouse", "data lake", "data modeling",
    # Web
    "react", "node.js", "fastapi", "flask", "django", "rest api",
    "graphql", "html", "css", "microservices",
    # Other
    "mlops", "faiss", "vector database", "agile", "scrum",
]

_MINED_EXTRA: Optional[List[str]] = None
NON_SKILL_TERMS = {
    "ability", "analysis", "candidate", "cloud", "communication", "company",
    "education", "engineering", "experience", "job", "knowledge", "management",
    "platform", "platforms", "plus", "preferred", "problem", "qualification",
    "qualifications", "required", "responsibilities", "role", "science",
    "skill", "skills", "solving", "team", "technology", "technologies",
    "tool", "tools", "understanding", "work", "working", "year", "years",
}


def reload_mined_skills_vocab() -> None:
    """Clear cache so ``mined_skills.json`` is re-read after model training."""
    global _MINED_EXTRA
    _MINED_EXTRA = None


def _load_mined_skill_terms() -> List[str]:
    """Terms mined from the training CSV (Kaggle / real dataset), if present."""
    global _MINED_EXTRA
    if _MINED_EXTRA is not None:
        return _MINED_EXTRA
    _MINED_EXTRA = []
    try:
        from utils.ml_engine import MINED_SKILLS_PATH

        if os.path.isfile(MINED_SKILLS_PATH):
            with open(MINED_SKILLS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("skills") or []
            _MINED_EXTRA = [
                term
                for term in (str(s).strip().lower() for s in raw)
                if _is_valid_skill_term(term)
            ]
    except Exception:
        _MINED_EXTRA = []
    return _MINED_EXTRA


def _skill_match_vocabulary() -> List[str]:
    """Static taxonomy plus dataset-mined terms (order preserved, deduped)."""
    extra = _load_mined_skill_terms()
    combined = list(dict.fromkeys(list(SKILL_KEYWORDS) + extra))
    return [term for term in combined if _is_valid_skill_term(term)]


def _is_valid_skill_term(term: str) -> bool:
    """Reject generic resume/JD words that should not be treated as skills."""
    if not term:
        return False
    cleaned = term.strip().lower()
    if cleaned in NON_SKILL_TERMS:
        return False
    if cleaned.isdigit():
        return False
    return len(cleaned) >= 2 or cleaned in {"r", "c", "c#", "c++"}


# ---------------------------------------------------------------------------
# Core cleaning functions
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Remove noise: URLs, emails, special characters; normalise whitespace.
    Returns lowercased cleaned string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Normalize unicode (e.g. fancy dashes, smart quotes)
    text = unicodedata.normalize("NFKD", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove phone numbers
    text = re.sub(r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]", " ", text)

    # Keep alphanumeric, spaces, hyphens (useful in tech stacks e.g. "ci/cd")
    text = re.sub(r"[^a-zA-Z0-9\s\-/\+#\.]", " ", text)

    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase
    return text.lower()


def remove_duplicates(text: str) -> str:
    """Remove duplicate sentences (same line appearing multiple times)."""
    lines = text.split(".")
    seen, unique = set(), []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique.append(stripped)
    return ". ".join(unique)


def tokenize(text: str) -> List[str]:
    """Word-tokenize using NLTK with a simple fallback."""
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter out common English stopwords."""
    return [t for t in tokens if t.lower() not in STOP_WORDS and len(t) > 1]


def lemmatize(tokens: List[str]) -> List[str]:
    """
    Lemmatize a token list using spaCy.
    Falls back to original token if spaCy is unavailable.
    """
    nlp = _get_nlp()
    if not nlp:
        return tokens
    if nlp.pipe_names:          # full model with tagger + lemmatizer
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]
    else:                       # blank model — return as-is
        return tokens


def full_preprocess(text: str) -> str:
    """
    End-to-end preprocessing pipeline:
        clean → dedup → tokenize → remove stopwords → lemmatize → rejoin
    Returns a single cleaned string suitable for embedding or TF-IDF.
    """
    cleaned = clean_text(text)
    deduped = remove_duplicates(cleaned)
    tokens = tokenize(deduped)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Structured-field extractors
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> List[str]:
    """
    Scan raw text for known skills using the static taxonomy plus any terms
    mined from the training dataset (see ``mined_skills.json``).
    """
    text_lower = text.lower()
    found = []
    for skill in _skill_match_vocabulary():
        # Use word-boundary matching so "r" doesn't match "architecture"
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(dict.fromkeys(found))   # deduplicate while preserving order


def extract_experience_years(text: str) -> float:
    """
    Heuristically estimate total years of experience from the resume text.

    Strategy (in priority order):
    1. Look for explicit "X years of experience" patterns.
    2. Sum durations from date ranges like "2018 - 2022" or "2020 – Present".
    3. Return 0 if nothing found.
    """
    text_lower = text.lower()

    # --- Pattern 1: "X years of experience" ---
    explicit = re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?experience", text_lower
    )
    if explicit:
        return max(float(y) for y in explicit)

    # --- Pattern 2: date ranges ---
    import datetime
    current_year = datetime.datetime.now().year

    ranges = re.findall(
        r"(\d{4})\s*[-–]\s*(present|\d{4})", text_lower
    )
    total = 0.0
    for start_str, end_str in ranges:
        start = int(start_str)
        end = current_year if end_str == "present" else int(end_str)
        if 1990 <= start <= current_year and start <= end:
            total += end - start
    if total > 0:
        return round(total, 1)

    return 0.0


def extract_education(text: str) -> str:
    """
    Return the highest detected education level from the resume text.
    """
    text_lower = text.lower()
    levels = [
        ("phd", "PhD"),
        ("doctor", "PhD"),
        ("m.tech", "Master's"),
        ("m.s.", "Master's"),
        ("mba", "Master's"),
        ("master", "Master's"),
        ("b.tech", "Bachelor's"),
        ("b.e.", "Bachelor's"),
        ("b.com", "Bachelor's"),
        ("bachelor", "Bachelor's"),
        ("diploma", "Diploma"),
        ("12th", "High School"),
    ]
    for pattern, label in levels:
        if pattern in text_lower:
            return label
    return "Not Specified"


def preprocess_resume(raw_text: str) -> Dict:
    """
    Full preprocessing + field extraction for a single resume.
    Returns a dict with all structured fields.
    """
    skills = extract_skills(raw_text)
    experience = extract_experience_years(raw_text)
    education = extract_education(raw_text)
    processed_text = full_preprocess(raw_text)

    return {
        "raw_text": raw_text,
        "processed_text": processed_text,
        "skills": skills,
        "experience_years": experience,
        "education": education,
    }


def preprocess_job_description(jd_text: str) -> Dict:
    """
    Preprocess a job description.
    Same pipeline as a resume so embeddings are in the same space.
    """
    return {
        "raw_text": jd_text,
        "processed_text": full_preprocess(jd_text),
        "required_skills": extract_skills(jd_text),
        "required_experience": extract_experience_years(jd_text),
        "required_education": extract_education(jd_text),
    }
