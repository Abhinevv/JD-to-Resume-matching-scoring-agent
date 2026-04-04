"""
utils/visualizer.py
--------------------
All chart-generation functions.
Each function returns a matplotlib Figure object for downstream rendering.
"""

import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logger = logging.getLogger(__name__)

# ── consistent theme ────────────────────────────────────────────────────────
PALETTE = ["#4F8EF7", "#F76C4F", "#4FD1A5", "#F7C94F", "#B84FF7",
           "#4FF7E0", "#F74FA0", "#8EF74F", "#F79B4F"]

def _apply_style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


# ---------------------------------------------------------------------------
# 1. Skill Frequency Bar Chart
# ---------------------------------------------------------------------------

def plot_skill_frequency(skill_data: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """
    Horizontal bar chart of the top-N most common skills.

    skill_data: DataFrame with columns [skill, count]
    """
    if skill_data.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No skill data available", ha="center", va="center")
        return fig

    df = skill_data.head(top_n).sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    bars = ax.barh(df["skill"], df["count"], color=colors, edgecolor="white", height=0.7)

    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{int(w)}", va="center", fontsize=9)

    _apply_style(ax, f"Top {top_n} Skills Across All Resumes",
                 xlabel="Number of Resumes", ylabel="Skill")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Job Role Distribution Pie / Bar Chart
# ---------------------------------------------------------------------------

def plot_role_distribution(roles: List[str]) -> plt.Figure:
    """Pie chart of predicted job-role distribution."""
    if not roles:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No role data", ha="center", va="center")
        return fig

    from collections import Counter
    counts = Counter(roles)
    labels = list(counts.keys())
    sizes  = list(counts.values())
    colors = PALETTE[:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(width=0.5, edgecolor="white"),
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Predicted Job Role Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Clustering Scatter (PCA 2-D)
# ---------------------------------------------------------------------------

def plot_clustering(embeddings: np.ndarray, labels: List[int],
                    resume_names: Optional[List[str]] = None) -> plt.Figure:
    """
    2-D PCA scatter plot coloured by cluster label.
    """
    if embeddings is None or len(embeddings) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough data for clustering plot",
                ha="center", va="center")
        return fig

    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)
    except Exception as e:
        logger.error(f"PCA failed: {e}")
        coords = embeddings[:, :2]

    unique_labels = sorted(set(labels))
    color_map = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(unique_labels)}
    point_colors = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[:, 0], coords[:, 1],
               c=point_colors, s=120, alpha=0.85,
               edgecolors="white", linewidths=0.8)

    # Annotate with names (if provided)
    if resume_names:
        for i, name in enumerate(resume_names):
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                        fontsize=8, alpha=0.75,
                        xytext=(4, 4), textcoords="offset points")

    # Legend
    legend_patches = [
        mpatches.Patch(color=color_map[lbl], label=f"Cluster {lbl}")
        for lbl in unique_labels
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9)
    _apply_style(ax, "Resume Clusters (PCA 2-D)",
                 xlabel="Principal Component 1",
                 ylabel="Principal Component 2")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Score Comparison Bar Chart
# ---------------------------------------------------------------------------

def plot_score_comparison(results: List[Dict], top_n: int = 10) -> plt.Figure:
    """
    Horizontal bar chart comparing final match scores for all candidates.

    results: list of dicts with keys 'name' and 'match_score'
    """
    if not results:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No results to display", ha="center", va="center")
        return fig

    df = (pd.DataFrame(results)[["name", "match_score"]]
            .sort_values("match_score", ascending=True)
            .tail(top_n))

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6)))
    bars = ax.barh(df["name"], df["match_score"], color=PALETTE[0],
                   edgecolor="white", height=0.65)

    # Colour code by threshold
    for bar, score in zip(bars, df["match_score"]):
        if score >= 0.70:
            bar.set_facecolor(PALETTE[2])   # green
        elif score >= 0.45:
            bar.set_facecolor(PALETTE[3])   # yellow
        else:
            bar.set_facecolor(PALETTE[1])   # red

    ax.set_xlim(0, 1.05)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.2%}", va="center", fontsize=9)

    _apply_style(ax, "Candidate Match Score Comparison",
                 xlabel="Final Match Score", ylabel="Candidate")

    # Legend
    patches = [
        mpatches.Patch(color=PALETTE[2], label="Strong (≥70%)"),
        mpatches.Patch(color=PALETTE[3], label="Moderate (45–70%)"),
        mpatches.Patch(color=PALETTE[1], label="Weak (<45%)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Skill Coverage Heatmap (JD skills vs candidates)
# ---------------------------------------------------------------------------

def plot_skill_coverage(
    resumes: List[Dict],
    jd_skills: List[str],
    top_n_skills: int = 15,
) -> plt.Figure:
    """
    Binary heatmap: rows = candidates, columns = top JD skills,
    cells = 1 if candidate has skill, 0 otherwise.
    """
    if not resumes or not jd_skills:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data for heatmap", ha="center", va="center")
        return fig

    skills = jd_skills[:top_n_skills]
    names  = [r.get("name", r.get("resume_id", "?")) for r in resumes]

    matrix = []
    for r in resumes:
        r_skills = set(s.lower() for s in r.get("skills", []))
        row = [1 if s.lower() in r_skills else 0 for s in skills]
        matrix.append(row)

    df_heat = pd.DataFrame(matrix, index=names, columns=skills)

    fig, ax = plt.subplots(figsize=(max(8, len(skills) * 0.7), max(4, len(names) * 0.55)))
    sns.heatmap(
        df_heat, ax=ax, cmap="YlGnBu", linewidths=0.5,
        linecolor="white", annot=True, fmt="d",
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Skill Coverage Heatmap (1 = Has Skill)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Required Skills", fontsize=10)
    ax.set_ylabel("Candidates", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Apriori frequent-itemsets visualisation
# ---------------------------------------------------------------------------

def plot_frequent_itemsets(frequent_itemsets: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Bar chart of the top-N frequent skill itemsets by support."""
    if frequent_itemsets is None or frequent_itemsets.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No frequent itemsets found\n(try lowering min_support)",
                ha="center", va="center", fontsize=11)
        return fig

    df = frequent_itemsets.copy()
    df["itemset_str"] = df["itemsets"].apply(lambda x: " + ".join(sorted(x)))
    df = df.sort_values("support", ascending=False).head(top_n)
    df = df.sort_values("support", ascending=True)   # flip for horizontal bar

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.6)))
    ax.barh(df["itemset_str"], df["support"],
            color=PALETTE[4], edgecolor="white", height=0.7)
    ax.set_xlim(0, 1.0)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["support"] + 0.01, i, f"{row['support']:.0%}",
                va="center", fontsize=9)
    _apply_style(ax, "Frequent Skill Patterns (Apriori)",
                 xlabel="Support", ylabel="Skill Combination")
    fig.tight_layout()
    return fig
