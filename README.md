# JD-to-Resume-matching-scoring-agent

Resume-to-job-description matching system with:

- FastAPI backend
- HTML/CSS/JavaScript frontend
- NLP preprocessing
- matching, ranking, clustering, and Apriori mining
- SQLite + FAISS persistence at runtime
- **Dataset training pipeline** (TF-IDF + Logistic Regression, K-Means on real CSV, mined skill vocabulary)

## Model training (Kaggle / real CSV)

The system can be trained on **real-world tabular data** (for example exports from [Kaggle](https://www.kaggle.com) resume or job-posting datasets) so evaluation can honestly cover **data engineering, data mining, and machine learning on a dataset**.

1. Place a CSV under `data/kaggle/resumes.csv` (or upload via the API). Expected columns (flexible names are auto-detected):
   - Resume text: e.g. `resume_text`, `Resume`, `text`, `description`
   - Role / label: e.g. `role`, `Category`, `Job Title`, `label`
   - Optional skills: e.g. `skills`, `tech_stack` (comma- or pipe-separated)
2. Train models:
   - **Web UI:** open the app and use **Train Model on Dataset** (uses `data/kaggle/resumes.csv` and the current K-Means slider).
   - **REST:** `POST /train` (optional multipart `csv_file` + form fields `n_clusters`, `top_skills`).
   - **Streamlit (optional):** `streamlit run streamlit_train.py`
3. Artifacts are written to `models/`: `role_classifier.pkl`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`, `mined_skills.json`, `dataset_kmeans.pkl`, `training_meta.json`.

What gets learned:

- **TF-IDF vectorization** and **Logistic Regression** for resume → role classification
- **K-Means** on TF-IDF vectors of the training corpus (unsupervised structure)
- **Skill frequency mining** from the dataset (extends rule-based skill extraction at runtime)

After training, the matching pipeline **uses the trained classifier** when present; it does **not** overwrite it with a small labeled upload batch. If no model is on disk, behavior falls back to training only when uploads include a `role` field (previous behavior).

### Viva talking point

You can describe the project as combining semantic matching with **supervised learning (role classifier) and unsupervised learning (K-Means)** on real CSV data, plus **skill distribution mining**, so the system learns patterns from data instead of relying only on static rules.

## Structure

```text
project/
├── api/                  # FastAPI app and REST endpoints
├── frontend/             # Static HTML/CSS/JS frontend
├── utils/                # Pipeline, preprocessing, ML, charts
│   └── training_pipeline.py   # Train on Kaggle-style CSV
├── database/             # Runtime SQLite / FAISS files
├── models/               # Runtime trained model artifacts
├── data/
│   ├── raw_resumes/      # Uploaded files at runtime
│   ├── kaggle/           # Place resumes.csv (Kaggle export) here
│   ├── sample/           # Built-in Python sample dataset
│   └── sample_text/      # Exported sample .txt files
├── streamlit_train.py    # Optional Streamlit trainer UI
├── tests/
├── main.py               # Main launcher
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

App URL:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Main API Endpoints

- `GET /`
- `GET /health`
- `GET /training/status`
- `POST /train`
- `GET /dashboard/meta`
- `POST /sample`
- `POST /match`
- `GET /resumes`
- `GET /resume/{resume_id}`

## Notes

- The main UI is FastAPI + HTML/JS; optional **Streamlit** trainer (`streamlit_train.py`) is available for training-only demos.
- A starter `data/kaggle/resumes.csv` is generated from the built-in sample resumes so training works before you add your own Kaggle file.
- Runtime-generated files are ignored from git where possible.
- On environments where `torch` DLLs fail, the matching pipeline falls back to TF-IDF embeddings.
