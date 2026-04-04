"""
utils/data_mining.py
---------------------
Simple data-mining helpers for skill analysis.
"""

import logging
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_apriori(
    skill_lists: List[List[str]],
    min_support: float = 0.2,
    min_confidence: float = 0.5,
    max_len: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Apriori on normalized skill transactions."""
    transactions = _normalize_skill_lists(skill_lists)
    if len(transactions) < 3:
        logger.warning("Apriori requires at least 3 resumes.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        encoder = TransactionEncoder()
        encoded = encoder.fit(transactions).transform(transactions)
        encoded_df = pd.DataFrame(encoded, columns=encoder.columns_)

        frequent_itemsets = apriori(
            encoded_df,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len,
        )
        if frequent_itemsets.empty:
            return frequent_itemsets, pd.DataFrame()

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
        )
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
        return frequent_itemsets, rules
    except ImportError:
        logger.error("mlxtend not installed. Run: pip install mlxtend")
    except Exception as exc:
        logger.error("Apriori failed: %s", exc)
    return pd.DataFrame(), pd.DataFrame()


def get_skill_frequencies(skill_lists: List[List[str]]) -> pd.DataFrame:
    """Count how many resumes mention each skill."""
    normalized_skills = _flatten_skills(skill_lists)
    counts = Counter(normalized_skills)
    total_resumes = max(len(skill_lists), 1)

    frequency_df = pd.DataFrame(counts.most_common(), columns=["skill", "count"])
    if frequency_df.empty:
        return frequency_df

    frequency_df["frequency_pct"] = (frequency_df["count"] / total_resumes * 100).round(1)
    return frequency_df


def top_skills(skill_lists: List[List[str]], n: int = 20) -> List[Tuple[str, int]]:
    """Return the top-N skills by occurrence count."""
    frequency_df = get_skill_frequencies(skill_lists)
    return list(zip(frequency_df["skill"].head(n), frequency_df["count"].head(n)))


def profile_clusters(resumes: List[Dict], cluster_labels: List[int]) -> Dict[int, Dict]:
    """Build a simple profile for each cluster."""
    if not resumes or not cluster_labels:
        return {}

    profiles: Dict[int, Dict] = {}
    for cluster_id in sorted(set(cluster_labels)):
        members = [resume for resume, label in zip(resumes, cluster_labels) if label == cluster_id]
        if not members:
            continue

        member_skills = _flatten_skills([member.get("skills", []) for member in members])
        predicted_roles = [member.get("predicted_role", "Unknown") for member in members if member.get("predicted_role")]

        profiles[cluster_id] = {
            "cluster_id": cluster_id,
            "size": len(members),
            "top_skills": [skill for skill, _ in Counter(member_skills).most_common(5)],
            "avg_experience": round(float(np.mean([member.get("experience_years", 0) for member in members])), 1),
            "dominant_role": Counter(predicted_roles).most_common(1)[0][0] if predicted_roles else "Unknown",
            "members": [member.get("resume_id", "") for member in members],
        }
    return profiles


def skill_gap_analysis(jd_skills: List[str], all_resume_skills: List[List[str]]) -> pd.DataFrame:
    """Measure how much of the candidate pool covers each required JD skill."""
    total_resumes = max(len(all_resume_skills), 1)
    normalized_resume_skills = [set(skills) for skills in _normalize_skill_lists(all_resume_skills)]

    rows = []
    for skill in _normalize_skills(jd_skills):
        coverage_count = sum(1 for skills in normalized_resume_skills if skill in skills)
        rows.append(
            {
                "skill": skill,
                "candidates_with_skill": coverage_count,
                "coverage_pct": round(coverage_count / total_resumes * 100, 1),
            }
        )

    gap_df = pd.DataFrame(rows)
    if gap_df.empty:
        return gap_df
    return gap_df.sort_values("coverage_pct", ascending=False).reset_index(drop=True)


def _normalize_skill_lists(skill_lists: List[List[str]]) -> List[List[str]]:
    """Normalize every skill to lowercase and remove blanks."""
    return [_normalize_skills(skills) for skills in skill_lists]


def _normalize_skills(skills: List[str]) -> List[str]:
    """Normalize one skill list."""
    return [skill.strip().lower() for skill in skills if skill and skill.strip()]


def _flatten_skills(skill_lists: List[List[str]]) -> List[str]:
    """Flatten nested skill lists into one normalized list."""
    return [skill for skills in _normalize_skill_lists(skill_lists) for skill in skills]
