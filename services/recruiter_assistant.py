"""Recruiter assistant helpers for domain-specific resume review.

The project intentionally keeps the frontend and runtime dependencies simple, so
this module provides deterministic LLM-style recruiter support without requiring
an external model API key. It creates domain skill expectations, candidate
feedback, interview prompts, and grounded chat answers from the parsed resume
payload.
"""

from __future__ import annotations

import io
import wave
from typing import Any, Dict, Iterable, List


DOMAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "data science": {
        "label": "Data Science / ML",
        "skills": [
            "python", "sql", "statistics", "machine learning", "pandas",
            "numpy", "scikit-learn", "data visualization", "mlops",
            "tensorflow", "pytorch",
        ],
        "scenarios": [
            "How would you build and validate a churn prediction model when the classes are highly imbalanced?",
            "A model performs well offline but poorly after deployment. What checks would you run first?",
            "How would you explain a complex model's prediction to a non-technical hiring manager?",
            "Design an experiment to measure whether a recommendation feature improves retention.",
        ],
    },
    "software engineering": {
        "label": "Software Engineering",
        "skills": [
            "java", "python", "javascript", "typescript", "react", "node.js",
            "rest api", "sql", "git", "docker", "microservices", "ci/cd",
        ],
        "scenarios": [
            "Design a reliable API for uploading and processing large resume files.",
            "How would you debug a production endpoint that is intermittently timing out?",
            "A feature request conflicts with existing architecture. How would you evaluate the trade-offs?",
            "How do you keep code maintainable when multiple developers touch the same module?",
        ],
    },
    "data engineering": {
        "label": "Data Engineering",
        "skills": [
            "python", "sql", "spark", "airflow", "kafka", "etl", "data warehouse",
            "data lake", "dbt", "snowflake", "bigquery", "aws",
        ],
        "scenarios": [
            "Design a daily pipeline that ingests unreliable third-party data and serves analytics by 9 AM.",
            "How would you handle schema drift in a streaming data source?",
            "A Spark job suddenly becomes twice as slow. What would you inspect?",
            "How would you design data quality checks for a business-critical dashboard?",
        ],
    },
    "cloud devops": {
        "label": "Cloud / DevOps",
        "skills": [
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "linux", "ci/cd", "jenkins", "monitoring", "microservices",
        ],
        "scenarios": [
            "A deployment succeeds but users report elevated errors. What is your incident workflow?",
            "How would you design a rollback strategy for a Kubernetes service?",
            "Explain how you would secure secrets across development, staging, and production.",
            "How would you reduce cloud cost without hurting reliability?",
        ],
    },
    "business analyst": {
        "label": "Business Analyst",
        "skills": [
            "sql", "excel", "power bi", "tableau", "statistics",
            "data visualization", "a/b testing", "communication", "agile",
        ],
        "scenarios": [
            "A stakeholder asks for a dashboard metric that is not clearly defined. What do you do?",
            "How would you investigate a sudden drop in conversion rate?",
            "Explain a time when data changed a business decision.",
            "How would you prioritize analytics requests from multiple teams?",
        ],
    },
}

DEFAULT_PROFILE = {
    "label": "General Technology",
    "skills": [
        "communication", "problem solving", "sql", "python", "git",
        "data visualization", "agile",
    ],
    "scenarios": [
        "Walk through a project where you had to learn a new technology quickly.",
        "Describe how you handle unclear requirements from a stakeholder.",
        "How would you debug a task when the root cause is not obvious?",
        "Tell us about a time you improved an existing process or system.",
    ],
}


def normalize_domain(domain: str | None) -> str:
    """Map free-form recruiter input onto the closest known domain key."""
    text = (domain or "").strip().lower()
    if not text:
        return "general"

    aliases = {
        "ai": "data science",
        "ml": "data science",
        "machine learning": "data science",
        "analytics": "business analyst",
        "business analytics": "business analyst",
        "software": "software engineering",
        "developer": "software engineering",
        "web": "software engineering",
        "backend": "software engineering",
        "frontend": "software engineering",
        "devops": "cloud devops",
        "cloud": "cloud devops",
        "data engineer": "data engineering",
    }
    for needle, mapped in aliases.items():
        if needle in text:
            return mapped
    for key in DOMAIN_PROFILES:
        if key in text or text in key:
            return key
    return text


def get_domain_profile(domain: str | None) -> Dict[str, Any]:
    """Return the closest profile, preserving custom domain labels."""
    key = normalize_domain(domain)
    if key in DOMAIN_PROFILES:
        return {"key": key, **DOMAIN_PROFILES[key]}
    if key == "general":
        return {"key": key, **DEFAULT_PROFILE}
    return {
        "key": key,
        "label": domain.strip() if domain else DEFAULT_PROFILE["label"],
        "skills": DEFAULT_PROFILE["skills"],
        "scenarios": DEFAULT_PROFILE["scenarios"],
    }


def build_domain_context(domain: str | None, jd_skills: Iterable[str] | None = None) -> Dict[str, Any]:
    """Build the domain context shown in the UI and attached to results."""
    profile = get_domain_profile(domain)
    jd_skill_list = [str(skill).lower() for skill in (jd_skills or []) if str(skill).strip()]
    required = list(dict.fromkeys(jd_skill_list + profile["skills"]))
    return {
        "domain": profile["label"],
        "domain_key": profile["key"],
        "recommended_skills": required,
        "scenario_questions": profile["scenarios"],
    }


def enrich_candidate_for_recruiter(candidate: Dict[str, Any], jd_record: Any, domain: str | None) -> Dict[str, Any]:
    """Attach recruiter-facing review assets to a ranked candidate."""
    jd_skills = getattr(jd_record, "required_skills", []) or []
    context = build_domain_context(domain, jd_skills)
    resume_skills = set(_lower_list(candidate.get("skills", [])))
    expected_skills = context["recommended_skills"]
    domain_gaps = [skill for skill in expected_skills if skill not in resume_skills]

    candidate_strengths = _candidate_strengths(candidate)
    feedback = _resume_feedback(candidate, domain_gaps)
    questions = _candidate_questions(candidate, context, domain_gaps)
    scenarios = _scenario_questions(candidate, context, domain_gaps)

    candidate["recruiter_assistant"] = {
        "domain": context["domain"],
        "domain_key": context["domain_key"],
        "recommended_skills": expected_skills,
        "domain_skill_gaps": domain_gaps[:10],
        "strengths": candidate_strengths,
        "resume_feedback": feedback,
        "interview_questions": questions,
        "scenario_questions": scenarios,
        "chat_starters": [
            "What are this candidate's strongest skills?",
            "Which skills are missing for this domain?",
            "What should I ask in the interview?",
            "Is this resume up to the mark?",
        ],
    }
    return candidate


def answer_candidate_question(
    candidate: Dict[str, Any],
    question: str,
    domain: str | None = None,
    jd_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Answer recruiter questions from the selected candidate payload."""
    query = (question or "").strip()
    if not query:
        return {"answer": "Ask a question about the selected candidate first."}

    assistant = candidate.get("recruiter_assistant") or {}
    if not assistant:
        context = build_domain_context(domain, (jd_info or {}).get("required_skills", []))
        temp_candidate = dict(candidate)
        temp_candidate["recruiter_assistant"] = {
            "domain": context["domain"],
            "recommended_skills": context["recommended_skills"],
            "domain_skill_gaps": [
                skill for skill in context["recommended_skills"]
                if skill not in set(_lower_list(candidate.get("skills", [])))
            ][:10],
            "strengths": _candidate_strengths(candidate),
            "resume_feedback": _resume_feedback(candidate, []),
            "interview_questions": _candidate_questions(candidate, context, []),
            "scenario_questions": context["scenario_questions"],
        }
        candidate = temp_candidate
        assistant = candidate["recruiter_assistant"]

    q = query.lower()
    name = candidate.get("name", "This candidate")

    if any(word in q for word in ["skill", "strength", "good", "strong"]):
        strengths = assistant.get("strengths", [])
        gaps = assistant.get("domain_skill_gaps", [])
        answer = (
            f"{name}'s strongest signals are {', '.join(strengths[:5]) or 'not clearly visible'}."
            f" For {assistant.get('domain', 'this domain')}, the main gaps are {', '.join(gaps[:6]) or 'minor based on extracted skills'}."
        )
    elif any(word in q for word in ["missing", "gap", "lack", "weak"]):
        gaps = assistant.get("domain_skill_gaps", [])
        feedback = assistant.get("resume_feedback", [])
        answer = (
            f"The biggest domain gaps are {', '.join(gaps[:8]) or 'not significant from the extracted resume text'}."
            f" Resume feedback: {' '.join(feedback[:2])}"
        )
    elif any(word in q for word in ["question", "interview", "ask", "scenario"]):
        questions = assistant.get("interview_questions", [])[:3]
        scenarios = assistant.get("scenario_questions", [])[:2]
        answer = "Suggested interview prompts: " + " ".join(f"{idx + 1}. {item}" for idx, item in enumerate(questions + scenarios))
    elif any(word in q for word in ["score", "rank", "ats", "match", "eligible", "shortlist"]):
        answer = _shortlist_answer(candidate, assistant)
    elif any(word in q for word in ["project", "portfolio", "built"]):
        project_summary = candidate.get("project_summary") or "No clear project section was detected."
        project_score = float(candidate.get("project_score") or candidate.get("projects_score") or 0)
        answer = f"Project signal is {project_score:.0%}. {project_summary}"
    elif any(word in q for word in ["experience", "year"]):
        found = float(candidate.get("experience_years") or candidate.get("experience_found") or 0)
        required = float(candidate.get("experience_required") or (jd_info or {}).get("required_experience") or 0)
        answer = f"{name} has about {found:.1f} extracted years of experience. Required experience is {required:.1f} years."
    else:
        answer = _candidate_snapshot(candidate, assistant)

    return {
        "answer": answer,
        "candidate": name,
        "domain": assistant.get("domain"),
    }


def analyze_voice_note(
    audio_bytes: bytes,
    filename: str = "",
    duration_seconds: float = 0.0,
    transcript: str = "",
    candidate_name: str = "",
) -> Dict[str, Any]:
    """Estimate introduction confidence from a voice note.

    This is a lightweight screening heuristic. It uses duration, audio energy
    when a WAV file is provided, and optional transcript signals. It should be
    treated as interview support, not a final hiring decision.
    """
    duration = max(float(duration_seconds or 0), 0.0)
    audio_metrics = _wav_metrics(audio_bytes, filename)
    if audio_metrics.get("duration_seconds"):
        duration = audio_metrics["duration_seconds"]

    transcript_metrics = _transcript_metrics(transcript)
    duration_score = _range_score(duration, 25, 90, 8, 160)
    energy_score = audio_metrics.get("energy_score", 0.55 if audio_bytes else 0.0)
    variation_score = audio_metrics.get("variation_score", 0.55 if audio_bytes else 0.0)
    transcript_score = transcript_metrics.get("transcript_score", 0.55 if not transcript else 0.0)
    filler_penalty = transcript_metrics.get("filler_ratio", 0.0)

    confidence_score = (
        duration_score * 0.30 +
        energy_score * 0.22 +
        variation_score * 0.18 +
        transcript_score * 0.25 +
        max(0.0, 1.0 - filler_penalty * 4) * 0.05
    )
    confidence_score = max(0.0, min(1.0, confidence_score))

    if confidence_score >= 0.72:
        label = "High confidence"
    elif confidence_score >= 0.48:
        label = "Moderate confidence"
    else:
        label = "Needs review"

    notes = []
    if duration < 15:
        notes.append("Introduction is very short; ask the candidate for a fuller summary.")
    elif duration > 150:
        notes.append("Introduction is long; check whether the candidate can communicate concisely.")
    else:
        notes.append("Duration is suitable for an introductory screening note.")
    if audio_metrics.get("parsed_wav"):
        notes.append("Audio clarity was estimated from WAV volume and variation.")
    else:
        notes.append("Audio format could not be deeply parsed; score uses duration, file presence, and transcript signals.")
    if transcript:
        notes.extend(transcript_metrics.get("notes", []))

    return {
        "candidate": candidate_name or "Selected candidate",
        "filename": filename,
        "confidence_score": round(confidence_score, 4),
        "confidence_label": label,
        "duration_seconds": round(duration, 2),
        "audio_metrics": audio_metrics,
        "transcript_metrics": transcript_metrics,
        "notes": notes,
        "disclaimer": "Use this as a recruiter aid only; do not make final hiring decisions from voice confidence alone.",
    }


def _lower_list(values: Iterable[Any]) -> List[str]:
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _wav_metrics(audio_bytes: bytes, filename: str) -> Dict[str, Any]:
    if not audio_bytes:
        return {"parsed_wav": False, "energy_score": 0.0, "variation_score": 0.0}
    if not filename.lower().endswith(".wav"):
        return {
            "parsed_wav": False,
            "size_kb": round(len(audio_bytes) / 1024, 2),
            "energy_score": 0.55,
            "variation_score": 0.55,
        }
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            frame_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
        duration = frames / frame_rate if frame_rate else 0.0
        rms = _pcm_rms(raw, sample_width) if raw else 0
        max_possible = float(2 ** (8 * sample_width - 1))
        energy = min(rms / max_possible * 10, 1.0) if max_possible else 0.0
        chunk_size = max(sample_width * channels * frame_rate // 2, sample_width * channels)
        chunks = [raw[index:index + chunk_size] for index in range(0, len(raw), chunk_size) if len(raw[index:index + chunk_size]) >= sample_width]
        chunk_rms = [_pcm_rms(chunk, sample_width) for chunk in chunks[:30] if chunk]
        if len(chunk_rms) > 1 and max(chunk_rms) > 0:
            variation = min((max(chunk_rms) - min(chunk_rms)) / max(chunk_rms), 1.0)
        else:
            variation = 0.35
        return {
            "parsed_wav": True,
            "duration_seconds": round(duration, 2),
            "channels": channels,
            "sample_rate": frame_rate,
            "rms": rms,
            "energy_score": round(max(0.1, energy), 4),
            "variation_score": round(max(0.15, variation), 4),
        }
    except Exception as exc:
        return {
            "parsed_wav": False,
            "parse_error": str(exc),
            "size_kb": round(len(audio_bytes) / 1024, 2),
            "energy_score": 0.5,
            "variation_score": 0.5,
        }


def _transcript_metrics(transcript: str) -> Dict[str, Any]:
    text = (transcript or "").strip().lower()
    if not text:
        return {"word_count": 0, "filler_ratio": 0.0, "transcript_score": 0.55, "notes": []}
    words = [word.strip(".,!?;:()[]{}\"'") for word in text.split() if word.strip()]
    filler_words = {"um", "uh", "like", "actually", "basically", "literally", "you know", "hmm"}
    filler_count = sum(1 for word in words if word in filler_words)
    word_count = len(words)
    action_terms = {"built", "developed", "led", "designed", "implemented", "improved", "managed", "delivered"}
    action_hits = sum(1 for word in words if word in action_terms)
    structure_hits = sum(1 for term in ["first", "second", "because", "therefore", "result", "impact"] if term in text)
    score = min(word_count / 90, 1.0) * 0.35 + min(action_hits / 4, 1.0) * 0.35 + min(structure_hits / 4, 1.0) * 0.30
    filler_ratio = filler_count / max(word_count, 1)
    notes = []
    if word_count < 35:
        notes.append("Transcript is short; ask for more detail on background and achievements.")
    if action_hits:
        notes.append("Candidate used action-oriented language in the introduction.")
    if filler_ratio > 0.08:
        notes.append("Filler words are relatively frequent; check communication clarity in live conversation.")
    return {
        "word_count": word_count,
        "filler_ratio": round(filler_ratio, 4),
        "action_word_hits": action_hits,
        "structure_hits": structure_hits,
        "transcript_score": round(max(0.0, min(1.0, score - filler_ratio)), 4),
        "notes": notes,
    }


def _pcm_rms(raw: bytes, sample_width: int) -> int:
    """Compute RMS for PCM bytes without external audio packages."""
    if not raw or sample_width <= 0:
        return 0
    sample_count = len(raw) // sample_width
    if sample_count <= 0:
        return 0
    total = 0
    usable = raw[: sample_count * sample_width]
    for index in range(0, len(usable), sample_width):
        sample_bytes = usable[index:index + sample_width]
        if sample_width == 1:
            sample = sample_bytes[0] - 128
        else:
            sample = int.from_bytes(sample_bytes, byteorder="little", signed=True)
        total += sample * sample
    return int((total / sample_count) ** 0.5)


def _range_score(value: float, ideal_min: float, ideal_max: float, hard_min: float, hard_max: float) -> float:
    if ideal_min <= value <= ideal_max:
        return 1.0
    if value < hard_min or value > hard_max:
        return 0.15
    if value < ideal_min:
        return max(0.15, (value - hard_min) / max(ideal_min - hard_min, 1))
    return max(0.15, (hard_max - value) / max(hard_max - ideal_max, 1))


def _candidate_strengths(candidate: Dict[str, Any]) -> List[str]:
    strengths: List[str] = []
    matched = candidate.get("matched_skills") or []
    if matched:
        strengths.extend(matched[:5])
    if float(candidate.get("semantic_similarity") or 0) >= 0.55:
        strengths.append("strong JD similarity")
    if float(candidate.get("experience_score") or 0) >= 0.75:
        strengths.append("experience alignment")
    if float(candidate.get("project_score") or 0) >= 0.55:
        strengths.append("project evidence")
    return list(dict.fromkeys(strengths)) or ["basic resume signal"]


def _resume_feedback(candidate: Dict[str, Any], domain_gaps: List[str]) -> List[str]:
    feedback: List[str] = []
    score = float(candidate.get("match_score") or 0)
    ats = float(candidate.get("ats_score") or 0)
    project_score = float(candidate.get("project_score") or 0)
    missing = candidate.get("missing_skills") or []

    if score < 0.45:
        feedback.append("Overall match is low; tailor the resume summary and project bullets to the job description.")
    elif score < 0.7:
        feedback.append("Resume is partially aligned; strengthen evidence for the most important role requirements.")
    else:
        feedback.append("Resume is broadly aligned; interview can focus on depth, ownership, and real project outcomes.")

    if ats < 0.55:
        feedback.append("ATS score is weak; add role keywords, measurable impact, and clearer section headings.")
    if missing:
        feedback.append(f"Add evidence for missing JD skills: {', '.join(missing[:6])}.")
    if domain_gaps:
        feedback.append(f"Domain skill gaps to verify: {', '.join(domain_gaps[:6])}.")
    if project_score < 0.4:
        feedback.append("Project section needs stronger implementation detail, tools used, and measurable results.")
    if not candidate.get("education") or candidate.get("education") == "Not Specified":
        feedback.append("Education is not clearly detected; confirm qualification during screening.")
    return feedback


def _candidate_questions(candidate: Dict[str, Any], context: Dict[str, Any], domain_gaps: List[str]) -> List[str]:
    name = candidate.get("name", "the candidate")
    matched = candidate.get("matched_skills") or candidate.get("skills") or []
    project_summary = candidate.get("project_summary") or "the most relevant project in your resume"
    offset = _candidate_offset(candidate)

    opening_templates = [
        f"{name}, explain the strongest project from your resume and how it connects to {context['domain']}.",
        f"Walk us through {project_summary}; what problem did you solve, and what was your personal contribution?",
        f"Pick one achievement from your resume and explain the impact, trade-offs, and follow-up work.",
    ]
    decision_templates = [
        "Which technical decision in your resume had the biggest impact, and how did you measure it?",
        "Describe a time you had two possible approaches. Why did you choose the one you used?",
        "What part of your work would you redesign today if you had more time?",
    ]
    behavior_templates = [
        "Tell me about a time a project failed or changed direction. What did you do next?",
        "Describe a situation where you had to explain a technical issue to a non-technical stakeholder.",
        "How do you handle incomplete requirements when the deadline is fixed?",
    ]

    questions = [
        _pick(opening_templates, offset),
        _pick(decision_templates, offset + 1),
    ]

    skill_templates = [
        "Give a concrete example where you used {skill}. What was the input, output, and result?",
        "What mistakes have you seen teams make with {skill}, and how do you avoid them?",
        "How would you evaluate whether your use of {skill} was successful in production?",
        "Explain {skill} to a junior teammate using a project from your resume.",
    ]
    for index, skill in enumerate(matched[:4]):
        questions.append(_pick(skill_templates, offset + index).format(skill=skill))

    gap_templates = [
        "Your resume has limited evidence of {skill}. How would you approach a task that requires it?",
        "If this role required {skill} in the first month, what would your learning and delivery plan be?",
        "What adjacent experience do you have that could transfer to {skill}?",
    ]
    for index, skill in enumerate(domain_gaps[:3]):
        questions.append(_pick(gap_templates, offset + index).format(skill=skill))

    questions.append(_pick(behavior_templates, offset + 2))
    return list(dict.fromkeys(questions))[:8]


def _scenario_questions(candidate: Dict[str, Any], context: Dict[str, Any], domain_gaps: List[str]) -> List[str]:
    """Create scenario prompts that vary by candidate strengths and gaps."""
    matched = candidate.get("matched_skills") or candidate.get("skills") or []
    primary_skill = matched[_candidate_offset(candidate) % len(matched)] if matched else "your strongest technical skill"
    gap_skill = domain_gaps[0] if domain_gaps else "a new tool required by the team"
    score = float(candidate.get("match_score") or 0)
    base = list(context.get("scenario_questions") or [])
    custom = [
        f"A production task requires {primary_skill} and a tight deadline. How would you plan, build, test, and communicate progress?",
        f"The team needs {gap_skill}, but your resume shows limited evidence of it. How would you close that gap while still delivering?",
        "A stakeholder challenges the result of your analysis or implementation. What evidence would you show?",
    ]
    if score < 0.45:
        custom.append("You are missing several role requirements. Which two would you learn first, and how would you prove readiness?")
    else:
        custom.append("You are assigned ownership of a feature related to your resume strengths. What would your first 30 days look like?")
    return list(dict.fromkeys(custom + base))[:6]


def _candidate_offset(candidate: Dict[str, Any]) -> int:
    seed = str(candidate.get("resume_id") or candidate.get("name") or candidate.get("filename") or "")
    return sum(ord(char) for char in seed) % 11


def _pick(options: List[str], offset: int) -> str:
    return options[offset % len(options)]


def _shortlist_answer(candidate: Dict[str, Any], assistant: Dict[str, Any]) -> str:
    score = float(candidate.get("match_score") or 0)
    ats = float(candidate.get("ats_score") or 0)
    if score >= 0.7 and ats >= 0.6:
        verdict = "shortlist"
    elif score >= 0.45:
        verdict = "keep as a backup or phone-screen"
    else:
        verdict = "do not shortlist unless the resume has context the parser missed"
    return (
        f"Recommendation: {verdict}. Match score is {score:.0%}, ATS score is {ats:.0%}, "
        f"and the domain is {assistant.get('domain', 'not specified')}."
    )


def _candidate_snapshot(candidate: Dict[str, Any], assistant: Dict[str, Any]) -> str:
    strengths = ", ".join((assistant.get("strengths") or [])[:4]) or "limited extracted strengths"
    gaps = ", ".join((assistant.get("domain_skill_gaps") or [])[:4]) or "no major extracted gaps"
    return (
        f"{candidate.get('name', 'This candidate')} has a {float(candidate.get('match_score') or 0):.0%} match score. "
        f"Strengths: {strengths}. Gaps to verify: {gaps}."
    )
