"""
database/db_manager.py
-----------------------
Manages two storage layers:
  1. SQLite (via SQLAlchemy) — structured metadata for every resume
  2. FAISS                  — vector index for semantic similarity search
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Text, DateTime, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLAlchemy setup
# ---------------------------------------------------------------------------

Base = declarative_base()
DB_PATH = os.path.join("database", "resumes.db")
FAISS_PATH = os.path.join("database", "faiss_index.pkl")
FAISS_META_PATH = os.path.join("database", "faiss_meta.json")

os.makedirs("database", exist_ok=True)


class Resume(Base):
    """ORM model for a single resume record."""
    __tablename__ = "resumes"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    resume_id     = Column(String(50), unique=True, nullable=False)
    name          = Column(String(200), default="Unknown")
    filename      = Column(String(300), default="")
    raw_text      = Column(Text, default="")
    processed_text= Column(Text, default="")
    skills        = Column(Text, default="[]")      # JSON list
    experience    = Column(Float, default=0.0)
    education     = Column(String(100), default="Not Specified")
    predicted_role= Column(String(100), default="")
    match_score   = Column(Float, default=0.0)
    cluster_label = Column(Integer, default=-1)
    created_at    = Column(DateTime, default=datetime.utcnow)


def _get_engine():
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)


def _get_session() -> Session:
    engine = _get_engine()
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def save_resume(data: Dict) -> int:
    """
    Insert or update a resume record.
    Returns the primary-key id.
    """
    session = _get_session()
    try:
        existing = session.query(Resume).filter_by(
            resume_id=data["resume_id"]
        ).first()

        skills_json = json.dumps(data.get("skills", []))

        if existing:
            existing.name           = data.get("name", "Unknown")
            existing.filename       = data.get("filename", "")
            existing.raw_text       = data.get("raw_text", "")
            existing.processed_text = data.get("processed_text", "")
            existing.skills         = skills_json
            existing.experience     = data.get("experience_years", 0.0)
            existing.education      = data.get("education", "Not Specified")
            existing.predicted_role = data.get("predicted_role", "")
            existing.match_score    = data.get("match_score", 0.0)
            existing.cluster_label  = data.get("cluster_label", -1)
            row_id = existing.id
        else:
            record = Resume(
                resume_id     = data["resume_id"],
                name          = data.get("name", "Unknown"),
                filename      = data.get("filename", ""),
                raw_text      = data.get("raw_text", ""),
                processed_text= data.get("processed_text", ""),
                skills        = skills_json,
                experience    = data.get("experience_years", 0.0),
                education     = data.get("education", "Not Specified"),
                predicted_role= data.get("predicted_role", ""),
                match_score   = data.get("match_score", 0.0),
                cluster_label = data.get("cluster_label", -1),
            )
            session.add(record)
            session.flush()
            row_id = record.id

        session.commit()
        return row_id
    except Exception as e:
        session.rollback()
        logger.error(f"save_resume failed: {e}")
        raise
    finally:
        session.close()


def get_all_resumes() -> List[Dict]:
    """Return all resume records as dicts."""
    session = _get_session()
    try:
        rows = session.query(Resume).all()
        result = []
        for r in rows:
            d = {c.name: getattr(r, c.name) for c in Resume.__table__.columns}
            try:
                d["skills"] = json.loads(d["skills"])
            except Exception:
                d["skills"] = []
            result.append(d)
        return result
    finally:
        session.close()


def get_resume_by_id(resume_id: str) -> Optional[Dict]:
    session = _get_session()
    try:
        r = session.query(Resume).filter_by(resume_id=resume_id).first()
        if not r:
            return None
        d = {c.name: getattr(r, c.name) for c in Resume.__table__.columns}
        try:
            d["skills"] = json.loads(d["skills"])
        except Exception:
            d["skills"] = []
        return d
    finally:
        session.close()


def update_scores(resume_id: str, match_score: float, cluster_label: int = -1,
                  predicted_role: str = ""):
    session = _get_session()
    try:
        r = session.query(Resume).filter_by(resume_id=resume_id).first()
        if r:
            r.match_score    = match_score
            r.cluster_label  = cluster_label
            r.predicted_role = predicted_role
            session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"update_scores failed: {e}")
    finally:
        session.close()


def clear_all_resumes():
    """Delete all resume records (used for fresh sessions)."""
    session = _get_session()
    try:
        session.query(Resume).delete()
        session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# FAISS vector store
# ---------------------------------------------------------------------------

class FAISSStore:
    """
    Lightweight FAISS wrapper that persists index + metadata to disk.
    Metadata maps integer FAISS position → resume_id.
    """

    def __init__(self):
        self.index = None
        self.meta: List[str] = []      # list of resume_ids in FAISS order
        self._load()

    def _load(self):
        try:
            import faiss
            if os.path.exists(FAISS_PATH) and os.path.exists(FAISS_META_PATH):
                with open(FAISS_PATH, "rb") as f:
                    self.index = pickle.load(f)
                with open(FAISS_META_PATH, "r") as f:
                    self.meta = json.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {e}")
            self.index = None
            self.meta = []

    def _save(self):
        try:
            with open(FAISS_PATH, "wb") as f:
                pickle.dump(self.index, f)
            with open(FAISS_META_PATH, "w") as f:
                json.dump(self.meta, f)
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def add_embeddings(self, embeddings: np.ndarray, resume_ids: List[str]):
        """
        Add a batch of embeddings to the FAISS index.
        embeddings: shape (N, D) float32
        resume_ids: list of N string IDs
        """
        try:
            import faiss
            dim = embeddings.shape[1]

            if self.index is None:
                self.index = faiss.IndexFlatIP(dim)   # Inner Product ≈ cosine for normalised vectors

            # L2-normalise for cosine similarity via inner product
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            self.meta.extend(resume_ids)
            self._save()
            logger.info(f"FAISS index now has {self.index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"FAISS add_embeddings failed: {e}")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for the top_k most similar resume embeddings.
        Returns list of (resume_id, score) tuples.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        try:
            import faiss
            query = query_embedding.reshape(1, -1).astype("float32")
            faiss.normalize_L2(query)
            scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.meta):
                    results.append((self.meta[idx], float(score)))
            return results
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

    def reset(self):
        """Clear the FAISS index and metadata."""
        self.index = None
        self.meta = []
        for p in [FAISS_PATH, FAISS_META_PATH]:
            if os.path.exists(p):
                os.remove(p)

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0


# Singleton instance
faiss_store = FAISSStore()
