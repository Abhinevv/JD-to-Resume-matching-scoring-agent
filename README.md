# 🧠 JD-to-Resume Matching & Scoring System

An end-to-end intelligent system that matches resumes with job descriptions using **Natural Language Processing (NLP), Machine Learning, and Data Mining techniques**.

This project ranks candidates based on semantic similarity, skill alignment, and experience while providing insights like clustering, skill gaps, and ATS-style evaluation.

New recruiter workflow: the app now asks for the recruiter's domain before matching, adds candidate-specific recruiter chat, generates domain-aware interview scenario questions, and gives resume improvement feedback when a profile is not up to the mark.

---

## 🚀 Features

* 📄 Resume parsing (PDF/Image/Text)
* 🧠 NLP-based preprocessing (tokenization, cleaning, skill extraction)
* 📊 Semantic similarity using TF-IDF / embeddings
* 🤖 Role prediction using Logistic Regression
* 📌 Candidate ranking with weighted scoring
* 🔍 Skill gap analysis (matched, missing, extra skills)
* 📈 Clustering using K-Means
* 🧩 Apriori algorithm for skill pattern mining
* ⚡ Fast similarity search using FAISS
* 💾 Persistent storage using SQLite
* 🌐 Interactive frontend dashboard

---

## 🏗️ System Architecture

```
Frontend (HTML/CSS/JS)
        ↓
FastAPI Backend (API Layer)
        ↓
Service Layer (Workflow Controller)
        ↓
ML + NLP Pipeline (Core Engine)
        ↓
SQLite + FAISS (Storage & Vector Search)
```

---

## 🔁 Workflow

1. User uploads resumes and enters a Job Description
2. Text is preprocessed (cleaning, tokenization, skill extraction)
3. ML models compute:

   * Semantic similarity
   * Skill overlap
   * Experience match
4. Final score is calculated using weighted formula
5. Candidates are ranked and stored in database
6. Frontend displays:

   * Rankings
   * Skill gaps
   * Clusters
   * Analytics

---

## 📊 Scoring Formula

```
Final Score =
  (Semantic Similarity × W1) +
  (Skill Match × W2) +
  (Experience Match × W3)
```

> Weights are adjustable from the UI.

---

## 🧠 Machine Learning Components

### Supervised Learning

* TF-IDF Vectorization
* Logistic Regression (Role Prediction)

### Unsupervised Learning

* K-Means Clustering (Candidate grouping)

### Data Mining

* Apriori Algorithm (Frequent skill patterns & associations)

---

## 🗂️ Project Structure

```
project/
├── api/                # FastAPI endpoints
├── services/           # Workflow & business logic
├── utils/              # NLP, ML, pipeline logic
├── database/           # SQLite + FAISS storage
├── models/             # Trained model artifacts
├── data/               # Sample + training data
├── frontend/           # UI (HTML/CSS/JS)
├── main.py             # Entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

---

## 🌐 Access

* App: http://127.0.0.1:8000
* API Docs: http://127.0.0.1:8000/docs

---

## 📡 API Endpoints

* `POST /match` → Upload resumes and match
* `POST /sample` → Run sample dataset
* `POST /train` → Train model on dataset
* `GET /resumes` → View stored resumes
* `GET /dashboard/meta` → Dashboard statistics

---

## 📈 Output Insights

* Ranked candidates with scores
* ATS-style evaluation
* Skill gap analysis
* Role prediction
* Cluster profiles
* Frequent skill patterns

---

## 💡 Real-World Applications

* Recruitment automation
* Resume screening systems (ATS)
* HR analytics
* Talent recommendation systems

---

## 🔮 Future Scope

* Deep learning embeddings (BERT, Sentence Transformers)
* Resume feedback generation
* Interview recommendation system
* Real-time job matching platform

---

## 🧑‍💻 Tech Stack

* **Backend:** FastAPI
* **Frontend:** HTML, CSS, JavaScript
* **ML/NLP:** Scikit-learn, spaCy, NLTK
* **Database:** SQLite
* **Vector Search:** FAISS
* **Data Mining:** mlxtend (Apriori)

---

## 🏁 Conclusion

This project integrates multiple domains — **NLP, Machine Learning, Data Mining, and Full-Stack Development** — to build a scalable and intelligent resume matching system.

It goes beyond keyword matching by incorporating semantic understanding, predictive modeling, and data-driven insights.

---

