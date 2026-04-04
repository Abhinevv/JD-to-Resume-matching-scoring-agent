# 🎯 Intelligent Resume–Job Description Matching System

A production-style AI system that accepts resumes and job descriptions, processes them through a full **Data Engineering → Data Mining → Machine Learning** pipeline, and outputs ranked candidates with explainable scores.

---

## 🗂 Project Structure

```
project/
├── app.py                        # Streamlit frontend (main UI)
├── main.py                       # Entry-point: setup + launch
├── requirements.txt              # All Python dependencies
│
├── api/
│   ├── __init__.py
│   └── main_api.py               # FastAPI REST backend
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py          # Text cleaning, tokenization, lemmatization
│   ├── file_extractor.py         # PDF (pdfplumber) + OCR (pytesseract)
│   ├── ml_engine.py              # Embeddings, cosine similarity, classifier, K-Means
│   ├── data_mining.py            # Apriori, skill freq, cluster profiling
│   ├── matching_pipeline.py      # End-to-end orchestrator
│   └── visualizer.py             # All matplotlib/seaborn charts
│
├── database/
│   ├── __init__.py
│   └── db_manager.py             # SQLite ORM (SQLAlchemy) + FAISS vector store
│
├── data/
│   ├── raw_resumes/              # Uploaded resume files saved here
│   └── sample/
│       └── sample_data.py        # 8 synthetic resumes + 2 JD profiles
│
└── models/                       # Persisted ML models (auto-created)
    ├── role_classifier.pkl
    ├── tfidf_vectorizer.pkl
    ├── label_encoder.pkl
    └── kmeans.pkl
```

---

## ⚡ Quick Start

### 1. Clone / unzip the project
```bash
cd project
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLP models
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

### 5. Run setup + smoke test
```bash
python main.py --test
```

### 6. Launch the Streamlit UI
```bash
streamlit run app.py
# → Open http://localhost:8501
```

### 7. (Optional) Launch FastAPI backend
```bash
uvicorn api.main_api:app --reload --port 8000
# → Swagger docs at http://localhost:8000/docs
```

---

## 🐳 Docker (Optional)

```bash
docker build -t resume-matcher .
docker run -p 8501:8501 resume-matcher
```

---

## 🔧 How to Use

### Option A — Sample Data (instant demo)
1. Open the app → tab **"🗂 Use Sample Data"**
2. Select role: `data_scientist` or `data_engineer`
3. Click **"▶️ Run on Sample Data"**

### Option B — Your Own Files
1. Tab **"📁 Upload Files"**
2. **Paste or upload** a Job Description (text or PDF)
3. **Upload resumes** in batch (PDF, PNG, JPG, TXT supported)
4. Adjust weights and settings in the sidebar
5. Click **"🚀 Run Matching Pipeline"**

### Results Tabs
| Tab | What you see |
|-----|-------------|
| 🏆 Rankings | Ranked candidates with score breakdown |
| 🔍 Explainability | Matched/missing skills, experience gap, recommendation |
| 📊 Visualizations | 5 charts: skill freq, role dist, clusters, scores, heatmap |
| ⛏ Data Mining | Apriori patterns, association rules, cluster profiles, skill gap |
| 🗄 Database | SQLite records + FAISS index stats + CSV download |

---

## 🧠 Technical Architecture

```
 INPUT                  PROCESSING                      OUTPUT
 ──────                 ──────────                      ──────
 JD text          →  1. Text Extraction (PDF/OCR)  →   Ranked candidates
 Resume files     →  2. NLP Preprocessing          →   Explainability report
                  →  3. Sentence Embeddings         →   Charts & visualizations
                  →  4. FAISS Vector Store          →   Apriori skill patterns
                  →  5. Cosine Similarity            →   Cluster profiles
                  →  6. Role Classifier (LR)        →   SQLite database
                  →  7. K-Means Clustering          →   CSV export
                  →  8. Weighted Final Score        →
                  →  9. Apriori Data Mining         →
```

### Scoring Formula
```
Final Score = 0.45 × Semantic Similarity
            + 0.35 × Skill Overlap (Jaccard)
            + 0.20 × Experience Score
```
*(Weights are adjustable in the sidebar)*

### Recommendation Thresholds
| Score | Label |
|-------|-------|
| ≥ 70% | ✅ Strong Match |
| 45–70% | ⚠️ Moderate Match |
| < 45% | ❌ Weak Match |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend API | FastAPI + Uvicorn |
| Text Extraction | pdfplumber, pytesseract |
| NLP | spaCy, NLTK |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| ML | scikit-learn (LogReg, K-Means, TF-IDF) |
| Data Mining | mlxtend (Apriori) |
| Vector DB | FAISS (faiss-cpu) |
| Relational DB | SQLite via SQLAlchemy |
| Visualization | matplotlib, seaborn, plotly |
| File I/O | pdfplumber, Pillow, pytesseract |

---

## 🔌 REST API Reference

```
POST /match          Upload JD + resumes → full results JSON
POST /sample         Run on built-in sample data
GET  /resumes        List all stored resume records
GET  /resume/{id}    Get single resume by ID
GET  /health         Health check
```

### Example API call
```bash
curl -X POST http://localhost:8000/sample \
  -F "role=data_scientist" \
  -F "n_clusters=3"
```

---

## 🛠 Configuration

All settings can be changed in the Streamlit sidebar at runtime:

| Setting | Default | Description |
|---------|---------|-------------|
| K-Means Clusters | 4 | Number of resume clusters |
| Semantic Weight | 0.45 | Weight for embedding similarity |
| Skill Weight | 0.35 | Weight for skill overlap |
| Experience Weight | 0.20 | Weight for experience score |
| Apriori Min Support | 0.20 | Minimum support for frequent patterns |

---

## 🗃 Database Schema

**SQLite table: `resumes`**

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| resume_id | TEXT | Unique identifier |
| name | TEXT | Candidate name |
| filename | TEXT | Original filename |
| skills | TEXT (JSON) | Extracted skill list |
| experience | FLOAT | Years of experience |
| education | TEXT | Highest education level |
| predicted_role | TEXT | ML-predicted job role |
| match_score | FLOAT | Final weighted score |
| cluster_label | INTEGER | K-Means cluster |
| created_at | DATETIME | Timestamp |

---

## 🧪 Running Tests

```bash
# Smoke test (pipeline on sample data)
python main.py --test

# Unit tests
python -m pytest tests/ -v
```

---

## 📝 Extending the System

- **Add new skills**: Edit `SKILL_KEYWORDS` in `utils/preprocessing.py`
- **Change embedding model**: Edit `SentenceTransformer(...)` in `utils/ml_engine.py`
- **Add a new chart**: Add a function to `utils/visualizer.py` and call it in `app.py`
- **Switch to PostgreSQL**: Change `DB_PATH` in `database/db_manager.py` to a Postgres connection string
- **Add authentication**: Use FastAPI's OAuth2 middleware in `api/main_api.py`

---

## 📄 License

MIT License — free for personal and commercial use.
# JD-to-Resume-matching-scoring-agent
