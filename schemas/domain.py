"""Core domain objects for resume matching workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ResumeRecord:
    """Structured representation of one resume throughout the pipeline."""

    resume_id: str
    name: str
    filename: str
    raw_text: str
    processed_text: str = ""
    projects_text: str = ""
    project_summary: str = ""
    skills: List[str] = field(default_factory=list)
    project_skills: List[str] = field(default_factory=list)
    experience_years: float = 0.0
    education: str = "Not Specified"
    predicted_role: str = "Unknown"
    cluster_label: int = -1
    semantic_similarity: float = 0.0
    skill_overlap: float = 0.0
    experience_score: float = 0.0
    project_score: float = 0.0
    match_score: float = 0.0

    @classmethod
    def from_raw(cls, payload: Dict[str, Any]) -> "ResumeRecord":
        """Create a record from incoming raw upload or sample payload."""
        return cls(
            resume_id=str(payload.get("resume_id", "")),
            name=str(payload.get("name", "Unknown")),
            filename=str(payload.get("filename", "")),
            raw_text=str(payload.get("raw_text", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a serializable dict."""
        return asdict(self)


@dataclass
class JobDescriptionRecord:
    """Structured representation of the current job description."""

    raw_text: str
    processed_text: str
    required_skills: List[str]
    required_experience: float
    required_education: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "JobDescriptionRecord":
        return cls(
            raw_text=str(payload.get("raw_text", "")),
            processed_text=str(payload.get("processed_text", "")),
            required_skills=list(payload.get("required_skills", [])),
            required_experience=float(payload.get("required_experience", 0.0)),
            required_education=str(payload.get("required_education", "Not Specified")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
