"""
app.py  —  Streamlit frontend for the Resume–JD Matching System
----------------------------------------------------------------
Run with:  streamlit run app.py
"""

import sys
import os
import uuid
import json
import logging

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.WARNING)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeAI Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #0d1117; }

.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

/* Cards */
.metric-card {
    background: linear-gradient(145deg, #161b22, #1c2433);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(79,142,247,0.15);
}

/* Rank badge */
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px; height: 36px;
    border-radius: 50%;
    font-weight: 700;
    font-size: 15px;
    margin-right: 10px;
}
.rank-1  { background: linear-gradient(135deg,#FFD700,#FFA500); color:#000; }
.rank-2  { background: linear-gradient(135deg,#C0C0C0,#A8A8A8); color:#000; }
.rank-3  { background: linear-gradient(135deg,#CD7F32,#8B4513); color:#fff; }
.rank-n  { background: #30363d; color:#8b949e; }

/* Score pill */
.score-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 14px;
    margin-left: 8px;
}
.score-strong   { background:#1a3a2a; color:#4FD1A5; border: 1px solid #4FD1A5; }
.score-moderate { background:#3a3010; color:#F7C94F; border: 1px solid #F7C94F; }
.score-weak     { background:#3a1010; color:#F76C4F; border: 1px solid #F76C4F; }

/* Skill tags */
.skill-tag {
    display: inline-block;
    padding: 3px 10px;
    margin: 3px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
}
.skill-matched  { background:#1a3a2a; color:#4FD1A5; border:1px solid #2a5a3a; }
.skill-missing  { background:#3a1010; color:#F76C4F; border:1px solid #5a2020; }
.skill-extra    { background:#1a2a3a; color:#4F8EF7; border:1px solid #2a4a6a; }

/* Section headers */
h1, h2, h3 { color: #e6edf3 !important; }
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4F8EF7;
    margin-bottom: 6px;
}

/* Progress bar override */
.stProgress > div > div { background: linear-gradient(90deg,#4F8EF7,#4FD1A5); border-radius: 4px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #c9d1d9; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #8b949e; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #4F8EF7 !important; border-bottom: 2px solid #4F8EF7 !important; }

/* Info / warning boxes */
.stAlert { border-radius: 8px; }

/* Dataframe */
.stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg,#4F8EF7,#4FD1A5);
    color: #0d1117;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def score_class(score: float) -> str:
    if score >= 0.70: return "score-strong"
    if score >= 0.45: return "score-moderate"
    return "score-weak"

def score_emoji(score: float) -> str:
    if score >= 0.70: return "✅"
    if score >= 0.45: return "⚠️"
    return "❌"

def rank_badge_class(i: int) -> str:
    return {1:"rank-1", 2:"rank-2", 3:"rank-3"}.get(i, "rank-n")

def skill_tags(skills: list, css_class: str) -> str:
    if not skills:
        return "<span style='color:#8b949e;font-size:12px;'>None</span>"
    return "".join(f'<span class="skill-tag {css_class}">{s}</span>' for s in skills)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 ResumeAI Matcher")
    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Settings")

    n_clusters = st.slider("K-Means Clusters", 2, 8, 4)

    st.markdown("**Score Weights**")
    sem_w   = st.slider("Semantic Similarity", 0.0, 1.0, 0.45, 0.05)
    skill_w = st.slider("Skill Match",         0.0, 1.0, 0.35, 0.05)
    exp_w   = st.slider("Experience Match",    0.0, 1.0, 0.20, 0.05)

    total_w = sem_w + skill_w + exp_w
    if abs(total_w - 1.0) > 0.05:
        st.warning(f"Weights sum = {total_w:.2f}. Ideal = 1.00")

    min_support = st.slider("Apriori Min Support", 0.1, 0.8, 0.2, 0.05)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    - **NLP**: spaCy, NLTK  
    - **ML**: sentence-transformers, scikit-learn  
    - **Mining**: Apriori (mlxtend), K-Means  
    - **Store**: SQLite + FAISS  
    - **Backend**: FastAPI  
    """)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem;">
  <h1 style="font-size:2.8rem; font-weight:700; 
             background:linear-gradient(135deg,#4F8EF7,#4FD1A5);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;
             margin:0;">
    🎯 Intelligent Resume Matcher
  </h1>
  <p style="color:#8b949e; font-size:1.1rem; margin-top:0.5rem;">
    Data Engineering · Machine Learning · Data Mining · Semantic NLP
  </p>
</div>
""", unsafe_allow_html=True)


# ── Input tabs ────────────────────────────────────────────────────────────────

input_tab1, input_tab2 = st.tabs(["📁 Upload Files", "🗂 Use Sample Data"])

with input_tab1:
    col_jd, col_res = st.columns([1, 1], gap="large")

    with col_jd:
        st.markdown("#### 📋 Job Description")
        jd_input_method = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)

        jd_text = ""
        if jd_input_method == "Paste text":
            jd_text = st.text_area("Paste job description here", height=280,
                                   placeholder="Senior Data Scientist role requiring Python, ML, NLP…")
        else:
            jd_file = st.file_uploader("Upload JD (txt / pdf)", type=["txt", "pdf"])
            if jd_file:
                from utils.file_extractor import extract_text_from_bytes
                jd_text = extract_text_from_bytes(jd_file.read(), jd_file.name)
                st.success(f"Loaded: {jd_file.name} ({len(jd_text)} chars)")
                with st.expander("Preview JD"):
                    st.text(jd_text[:800] + "…" if len(jd_text) > 800 else jd_text)

    with col_res:
        st.markdown("#### 📄 Resumes (Batch Upload)")
        resume_files = st.file_uploader(
            "Upload resumes (PDF, PNG, JPG, TXT)",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True,
        )
        if resume_files:
            st.info(f"📂 {len(resume_files)} file(s) uploaded")
            for f in resume_files:
                st.caption(f"• {f.name}  ({f.size / 1024:.1f} KB)")

    run_upload = st.button("🚀 Run Matching Pipeline", key="run_upload",
                           disabled=(not jd_text.strip() or not resume_files))

with input_tab2:
    st.markdown("#### 📦 Built-in Sample Dataset")
    st.markdown("8 synthetic resumes + 2 JD profiles — instant demo with no file uploads.")
    sample_role = st.selectbox("Target Role", ["data_scientist", "data_engineer"])
    run_sample  = st.button("▶️ Run on Sample Data", key="run_sample")


# ── Pipeline execution ────────────────────────────────────────────────────────

results = None

if run_sample:
    with st.spinner("⚡ Running pipeline on sample data…"):
        try:
            from data.sample.sample_data import SAMPLE_RESUMES, SAMPLE_JOB_DESCRIPTIONS
            from utils.matching_pipeline import run_matching_pipeline

            jd = SAMPLE_JOB_DESCRIPTIONS[sample_role]
            raw = [
                {"resume_id": r["id"], "name": r["name"],
                 "filename": r["name"] + ".txt", "raw_text": r["text"], "role": r["role"]}
                for r in SAMPLE_RESUMES
            ]
            results = run_matching_pipeline(
                raw, jd, n_clusters=n_clusters,
                min_support=min_support,
                score_weights={"semantic": sem_w, "skill": skill_w, "experience": exp_w},
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback; st.code(traceback.format_exc())

elif run_upload and jd_text.strip() and resume_files:
    with st.spinner("⚡ Processing resumes…"):
        try:
            from utils.file_extractor import extract_text_from_bytes
            from utils.matching_pipeline import run_matching_pipeline

            raw = []
            for f in resume_files:
                content = f.read()
                text = extract_text_from_bytes(content, f.name)
                raw.append({
                    "resume_id": str(uuid.uuid4())[:8],
                    "name":      os.path.splitext(f.name)[0],
                    "filename":  f.name,
                    "raw_text":  text or f"[No text extracted from {f.name}]",
                })

            results = run_matching_pipeline(
                raw, jd_text, n_clusters=n_clusters,
                min_support=min_support,
                score_weights={"semantic": sem_w, "skill": skill_w, "experience": exp_w},
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback; st.code(traceback.format_exc())


# ── Results display ───────────────────────────────────────────────────────────

if results and "ranked_candidates" in results:
    ranked     = results["ranked_candidates"]
    jd_info    = results.get("jd_info", {})
    skill_freq = results.get("skill_frequencies", pd.DataFrame())
    ap_items   = results.get("apriori_itemsets")
    ap_rules   = results.get("apriori_rules")
    clust_prof = results.get("cluster_profiles", {})
    skill_gap  = results.get("skill_gap")
    embeddings = results.get("embeddings")
    clust_lbls = results.get("cluster_labels", [])
    res_names  = results.get("resume_names", [])

    st.markdown("---")

    # ── Summary strip ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    strong   = sum(1 for r in ranked if r["match_score"] >= 0.70)
    moderate = sum(1 for r in ranked if 0.45 <= r["match_score"] < 0.70)
    weak     = len(ranked) - strong - moderate
    top_score = ranked[0]["match_score"] if ranked else 0

    c1.metric("👥 Total Candidates", len(ranked))
    c2.metric("✅ Strong Matches",   strong)
    c3.metric("⚠️  Moderate",        moderate)
    c4.metric("🏆 Top Score",        f"{top_score:.1%}")

    # ── Main result tabs ───────────────────────────────────────────────────
    tab_rank, tab_explain, tab_viz, tab_mining, tab_db = st.tabs([
        "🏆 Rankings", "🔍 Explainability",
        "📊 Visualizations", "⛏ Data Mining", "🗄 Database"
    ])

    # ── RANKINGS ──────────────────────────────────────────────────────────
    with tab_rank:
        st.markdown("### 🏆 Ranked Candidates")

        for i, r in enumerate(ranked, start=1):
            score = r["match_score"]
            s_class = score_class(score)
            b_class = rank_badge_class(i)

            with st.expander(
                f"#{i}  {r['name']}  —  {score:.1%}", expanded=(i <= 3)
            ):
                col_a, col_b = st.columns([2, 1])

                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                      <span class="rank-badge {b_class}">{i}</span>
                      <strong style="font-size:1.1rem;color:#e6edf3">{r['name']}</strong>
                      <span class="score-pill {s_class}">{score_emoji(score)} {score:.1%}</span>
                      <br/><br/>
                      <div class="section-label">Matched Skills ({len(r.get('matched_skills',[]))})</div>
                      {skill_tags(r.get('matched_skills',[]), 'skill-matched')}
                      <br/><br/>
                      <div class="section-label">Missing Skills ({len(r.get('missing_skills',[]))})</div>
                      {skill_tags(r.get('missing_skills',[]), 'skill-missing')}
                      <br/><br/>
                      <div class="section-label">Additional Skills</div>
                      {skill_tags(r.get('extra_skills',[])[:8], 'skill-extra')}
                    </div>
                    """, unsafe_allow_html=True)

                with col_b:
                    st.markdown(f"""
                    <div class="metric-card" style="height:100%">
                      <div class="section-label">Score Breakdown</div>
                      <p>🧠 Semantic: <b>{r.get('semantic_similarity',0):.1%}</b></p>
                      <p>🔧 Skills:   <b>{r.get('skill_overlap',0):.1%}</b></p>
                      <p>📅 Exp:      <b>{r.get('experience_score',0):.1%}</b></p>
                      <hr style="border-color:#30363d">
                      <p>🎓 Education: <b>{r.get('education','N/A')}</b></p>
                      <p>📆 Exp (yrs): <b>{r.get('experience_found',0):.1f}</b>
                         / required <b>{r.get('experience_required',0):.1f}</b></p>
                      <p>🏷 Role: <b>{r.get('predicted_role','N/A')}</b></p>
                      <p>🔵 Cluster: <b>{r.get('cluster_label','N/A')}</b></p>
                      <br/>
                      <div style="font-size:1rem">{r.get('recommendation','')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Score bars
                st.markdown("**Score Components**")
                cols_bars = st.columns(3)
                cols_bars[0].metric("Semantic", f"{r.get('semantic_similarity',0):.1%}")
                cols_bars[1].metric("Skills",   f"{r.get('skill_overlap',0):.1%}")
                cols_bars[2].metric("Exp",      f"{r.get('experience_score',0):.1%}")
                st.progress(score)

    # ── EXPLAINABILITY ────────────────────────────────────────────────────
    with tab_explain:
        st.markdown("### 🔍 Detailed Explainability Report")
        sel_name = st.selectbox("Select Candidate", [r["name"] for r in ranked])
        sel = next(r for r in ranked if r["name"] == sel_name)

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            st.markdown(f"""
            <div class="metric-card">
              <div class="section-label">Overall Result</div>
              <h2 style="color:#e6edf3">{sel['name']}</h2>
              <h3 style="color:#4F8EF7">{sel['match_score']:.1%} Match Score</h3>
              <p style="font-size:1.2rem">{sel.get('recommendation','')}</p>
              <hr style="border-color:#30363d">
              <p>🏷 Predicted Role: <b>{sel.get('predicted_role','N/A')}</b></p>
              <p>🎓 Education: <b>{sel.get('education','N/A')}</b></p>
              <p>📅 Experience: <b>{sel.get('experience_found',0):.1f} yrs</b>
                 (Required: <b>{sel.get('experience_required',0):.1f} yrs</b>,
                  Gap: <b>{sel.get('experience_gap',0):.1f} yrs</b>)</p>
              <p>🔵 Cluster: <b>{sel.get('cluster_label','N/A')}</b></p>
            </div>
            """, unsafe_allow_html=True)

        with col_e2:
            st.markdown(f"""
            <div class="metric-card">
              <div class="section-label">Score Breakdown</div>
              <br/>
            </div>
            """, unsafe_allow_html=True)

            components = {
                "Semantic Similarity": sel.get("semantic_similarity", 0),
                "Skill Match":         sel.get("skill_overlap", 0),
                "Experience Match":    sel.get("experience_score", 0),
            }
            for label, val in components.items():
                st.markdown(f"**{label}**: {val:.1%}")
                st.progress(float(np.clip(val, 0, 1)))

        st.markdown("#### Matched Skills")
        st.markdown(skill_tags(sel.get("matched_skills", []), "skill-matched"), unsafe_allow_html=True)

        st.markdown("#### Missing Skills (from JD)")
        st.markdown(skill_tags(sel.get("missing_skills", []), "skill-missing"), unsafe_allow_html=True)

        st.markdown("#### Additional Skills (not in JD)")
        st.markdown(skill_tags(sel.get("extra_skills", [])[:12], "skill-extra"), unsafe_allow_html=True)

        st.markdown("#### JD Required Skills")
        st.markdown(skill_tags(jd_info.get("required_skills", []), "skill-extra"), unsafe_allow_html=True)

    # ── VISUALIZATIONS ────────────────────────────────────────────────────
    with tab_viz:
        st.markdown("### 📊 Visual Analytics")

        from utils.visualizer import (
            plot_skill_frequency, plot_role_distribution,
            plot_clustering, plot_score_comparison,
            plot_skill_coverage, plot_frequent_itemsets,
        )

        row1_c1, row1_c2 = st.columns(2)

        with row1_c1:
            st.markdown("#### Skill Frequency")
            if isinstance(skill_freq, pd.DataFrame) and not skill_freq.empty:
                fig = plot_skill_frequency(skill_freq, top_n=15)
                st.pyplot(fig)
            else:
                st.info("No skill frequency data.")

        with row1_c2:
            st.markdown("#### Role Distribution")
            roles = [r.get("predicted_role", "Unknown") for r in ranked]
            fig = plot_role_distribution(roles)
            st.pyplot(fig)

        row2_c1, row2_c2 = st.columns(2)

        with row2_c1:
            st.markdown("#### Candidate Score Comparison")
            fig = plot_score_comparison(ranked)
            st.pyplot(fig)

        with row2_c2:
            st.markdown("#### Resume Clusters (PCA 2-D)")
            if embeddings is not None and len(clust_lbls) >= 2:
                fig = plot_clustering(embeddings, clust_lbls, res_names)
                st.pyplot(fig)
            else:
                st.info("Not enough data for clustering plot.")

        st.markdown("#### Skill Coverage Heatmap")
        jd_skills_for_heat = jd_info.get("required_skills", [])
        if jd_skills_for_heat:
            fig = plot_skill_coverage(ranked, jd_skills_for_heat, top_n_skills=12)
            st.pyplot(fig)
        else:
            st.info("No JD skills found for heatmap.")

        # Apriori chart
        if isinstance(ap_items, pd.DataFrame) and not ap_items.empty:
            st.markdown("#### Frequent Skill Combinations (Apriori)")
            fig = plot_frequent_itemsets(ap_items, top_n=10)
            st.pyplot(fig)

    # ── DATA MINING ───────────────────────────────────────────────────────
    with tab_mining:
        st.markdown("### ⛏ Data Mining Results")

        # Apriori
        st.markdown("#### 🔗 Frequent Skill Patterns (Apriori)")
        if isinstance(ap_items, pd.DataFrame) and not ap_items.empty:
            display_items = ap_items.copy()
            display_items["itemsets"] = display_items["itemsets"].apply(
                lambda x: ", ".join(sorted(list(x)))
            )
            st.dataframe(
                display_items[["itemsets", "support"]].sort_values(
                    "support", ascending=False
                ).head(20),
                use_container_width=True,
            )
        else:
            st.info("No frequent patterns found. Try lowering the Min Support threshold.")

        if isinstance(ap_rules, pd.DataFrame) and not ap_rules.empty:
            st.markdown("#### 📐 Association Rules")
            display_rules = ap_rules.copy()
            for col in ["antecedents", "consequents"]:
                if col in display_rules.columns:
                    display_rules[col] = display_rules[col].apply(
                        lambda x: ", ".join(sorted(list(x)))
                    )
            show_cols = [c for c in ["antecedents", "consequents", "support",
                                      "confidence", "lift"] if c in display_rules.columns]
            st.dataframe(
                display_rules[show_cols].round(3).head(15),
                use_container_width=True,
            )

        # Cluster profiles
        st.markdown("#### 🔵 Cluster Profiles")
        if clust_prof:
            for cid, prof in clust_prof.items():
                with st.expander(f"Cluster {cid}  —  {prof['size']} resume(s)"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Members",      prof["size"])
                    c2.metric("Avg Exp (yrs)",prof["avg_experience"])
                    c3.metric("Dominant Role", prof["dominant_role"])
                    st.markdown("**Top Skills:** " +
                                ", ".join(prof.get("top_skills", [])))

        # Skill gap
        st.markdown("#### 📉 Skill Gap Analysis (JD vs Candidate Pool)")
        if isinstance(skill_gap, pd.DataFrame) and not skill_gap.empty:
            st.dataframe(skill_gap.style.background_gradient(
                subset=["coverage_pct"], cmap="RdYlGn"
            ), use_container_width=True)
        else:
            st.info("No JD skills detected for gap analysis.")

    # ── DATABASE ──────────────────────────────────────────────────────────
    with tab_db:
        st.markdown("### 🗄 Stored Resume Records (SQLite)")
        try:
            from database.db_manager import get_all_resumes
            db_rows = get_all_resumes()
            if db_rows:
                df_db = pd.DataFrame(db_rows)
                show_cols = [c for c in [
                    "resume_id", "name", "experience", "education",
                    "predicted_role", "match_score", "cluster_label", "created_at"
                ] if c in df_db.columns]
                st.dataframe(df_db[show_cols].sort_values("match_score", ascending=False),
                             use_container_width=True)

                # Download
                csv = df_db.to_csv(index=False)
                st.download_button("⬇️ Download CSV", csv,
                                   "resume_results.csv", "text/csv")
            else:
                st.info("No records in DB yet — run the pipeline first.")
        except Exception as e:
            st.warning(f"DB not accessible: {e}")

        st.markdown("### 📦 FAISS Vector Index")
        try:
            from database.db_manager import faiss_store
            st.metric("Vectors Indexed", faiss_store.size)
        except Exception:
            st.info("FAISS index not initialised yet.")

elif not results:
    # Landing placeholder
    st.markdown("""
    <div style="text-align:center; padding:4rem; opacity:0.5;">
      <p style="font-size:4rem">🎯</p>
      <p style="font-size:1.2rem; color:#8b949e;">
        Upload resumes & a job description, then click
        <strong>Run Matching Pipeline</strong>.
      </p>
      <p style="font-size:0.9rem; color:#8b949e;">
        Or try the built-in sample data to explore the system instantly.
      </p>
    </div>
    """, unsafe_allow_html=True)
