"""
utils/data_mining.py
---------------------
Data Mining layer:
  - Apriori algorithm for frequent skill-set patterns
  - Skill frequency analysis
  - Cluster profiling
"""

import logging
from collections import Counter
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Apriori — frequent skill patterns
# ---------------------------------------------------------------------------

def run_apriori(
    skill_lists: List[List[str]],
    min_support: float = 0.2,
    min_confidence: float = 0.5,
    max_len: int = 4,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the Apriori algorithm on a collection of resume skill lists.

    Parameters
    ----------
    skill_lists   : list of skill lists, one per resume
    min_support   : minimum support threshold (0-1)
    min_confidence: minimum confidence for association rules
    max_len       : maximum itemset length

    Returns
    -------
    frequent_itemsets : pd.DataFrame  (itemsets + support)
    rules             : pd.DataFrame  (antecedents, consequents, confidence, lift)
    """
    if len(skill_lists) < 3:
        logger.warning("Apriori requires at least 3 resumes.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori, association_rules

        # Normalise to lowercase
        transactions = [[s.lower() for s in skills] for skills in skill_lists]

        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_te = pd.DataFrame(te_array, columns=te.columns_)

        frequent_itemsets = apriori(
            df_te,
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
        # Sort by lift descending
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

        return frequent_itemsets, rules

    except ImportError:
        logger.error("mlxtend not installed. Run: pip install mlxtend")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        logger.error(f"Apriori failed: {e}")
        return pd.DataFrame(), pd.DataFrame()


# ---------------------------------------------------------------------------
# Skill frequency analysis
# ---------------------------------------------------------------------------

def get_skill_frequencies(skill_lists: List[List[str]]) -> pd.DataFrame:
    """
    Count how many resumes mention each skill.
    Returns DataFrame with columns [skill, count, frequency_pct].
    """
    all_skills = [s.lower() for skills in skill_lists for s in skills]
    counts = Counter(all_skills)
    total_resumes = max(len(skill_lists), 1)

    df = pd.DataFrame(counts.most_common(), columns=["skill", "count"])
    df["frequency_pct"] = (df["count"] / total_resumes * 100).round(1)
    return df


def top_skills(skill_lists: List[List[str]], n: int = 20) -> List[Tuple[str, int]]:
    """Return the top-n skills by occurrence count."""
    freq_df = get_skill_frequencies(skill_lists)
    return list(zip(freq_df["skill"].head(n), freq_df["count"].head(n)))


# ---------------------------------------------------------------------------
# Cluster profiling
# ---------------------------------------------------------------------------

def profile_clusters(
    resumes: List[Dict],
    cluster_labels: List[int],
) -> Dict[int, Dict]:
    """
    Summarise each K-Means cluster:
      - size (number of resumes)
      - top skills
      - average experience
      - dominant role

    Parameters
    ----------
    resumes        : list of resume dicts (must have 'skills', 'experience_years', 'predicted_role')
    cluster_labels : list of integer cluster assignments (same order as resumes)

    Returns
    -------
    dict mapping cluster_id → profile dict
    """
    if not resumes or not cluster_labels:
        return {}

    profiles: Dict[int, Dict] = {}
    cluster_ids = sorted(set(cluster_labels))

    for cid in cluster_ids:
        members = [r for r, lbl in zip(resumes, cluster_labels) if lbl == cid]
        if not members:
            continue

        # Aggregate skills
        all_skills = [s for r in members for s in r.get("skills", [])]
        skill_counts = Counter(s.lower() for s in all_skills)
        top_3_skills = [s for s, _ in skill_counts.most_common(5)]

        # Average experience
        avg_exp = np.mean([r.get("experience_years", 0) for r in members])

        # Dominant role
        roles = [r.get("predicted_role", "Unknown") for r in members if r.get("predicted_role")]
        dominant_role = Counter(roles).most_common(1)[0][0] if roles else "Unknown"

        profiles[cid] = {
            "cluster_id":     cid,
            "size":           len(members),
            "top_skills":     top_3_skills,
            "avg_experience": round(float(avg_exp), 1),
            "dominant_role":  dominant_role,
            "members":        [r.get("resume_id", "") for r in members],
        }

    return profiles


# ---------------------------------------------------------------------------
# Skill gap analysis (JD vs candidate pool)
# ---------------------------------------------------------------------------

def skill_gap_analysis(
    jd_skills: List[str],
    all_resume_skills: List[List[str]],
) -> pd.DataFrame:
    """
    For each required skill in the JD, compute what percentage of
    the candidate pool has it.

    Returns DataFrame with columns [skill, candidates_with_skill, coverage_pct].
    """
    total = max(len(all_resume_skills), 1)
    rows = []
    for skill in jd_skills:
        skill_lower = skill.lower()
        count = sum(
            1 for skills in all_resume_skills
            if skill_lower in [s.lower() for s in skills]
        )
        rows.append({
            "skill":                skill,
            "candidates_with_skill": count,
            "coverage_pct":         round(count / total * 100, 1),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("coverage_pct", ascending=False).reset_index(drop=True)
    return df
