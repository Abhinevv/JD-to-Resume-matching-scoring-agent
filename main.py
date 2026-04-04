"""
main.py
-------
Project entry-point:
  - Sets up folder structure
  - Downloads NLP models
  - Optionally runs a quick smoke test on sample data
  - Launches the FastAPI app that serves the HTML/CSS/JS frontend

Usage:
  python main.py         # setup + launch web app
  python main.py --test  # setup + run pipeline smoke test only
  python main.py --api   # setup + launch FastAPI server
"""

import os
import sys
import argparse
import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def setup_directories():
    """Create required project directories."""
    dirs = [
        "data/raw_resumes",
        "data/sample",
        "data/kaggle",
        "models",
        "database",
        "api",
        "utils",
        "frontend",
    ]
    for d in dirs:
        os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)
    logger.info("Directory structure ready.")


def download_nlp_models():
    """Download spaCy and NLTK models if not present."""
    import nltk

    for resource in ["punkt", "stopwords", "averaged_perceptron_tagger", "punkt_tab"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    logger.info("NLTK resources ready.")

    try:
        import spacy
        spacy.load("en_core_web_sm")
        logger.info("spaCy en_core_web_sm already installed.")
    except OSError:
        logger.info("Downloading spaCy en_core_web_sm ...")
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
                capture_output=True,
            )
            logger.info("spaCy model downloaded.")
        except Exception as e:
            logger.warning(f"spaCy download failed (will use fallback): {e}")


def run_smoke_test():
    """Run the pipeline on sample data and print top-3 results."""
    logger.info("\n" + "=" * 60)
    logger.info("  SMOKE TEST - Sample Data Pipeline")
    logger.info("=" * 60)

    from data.sample.sample_data import SAMPLE_RESUMES, SAMPLE_JOB_DESCRIPTIONS
    from utils.matching_pipeline import run_matching_pipeline

    jd = SAMPLE_JOB_DESCRIPTIONS["data_scientist"]
    raw = [
        {
            "resume_id": r["id"],
            "name": r["name"],
            "filename": r["name"] + ".txt",
            "raw_text": r["text"],
            "role": r["role"],
        }
        for r in SAMPLE_RESUMES
    ]

    logger.info(f"Processing {len(raw)} resumes...")
    results = run_matching_pipeline(raw, jd, n_clusters=3, min_support=0.2)

    if "error" in results:
        logger.error(f"Pipeline error: {results['error']}")
        return False

    ranked = results["ranked_candidates"]
    logger.info(f"\nTop {min(3, len(ranked))} Candidates:\n")

    for i, r in enumerate(ranked[:3], 1):
        logger.info(
            f"  #{i} {r['name']:20s}  Score={r['match_score']:.1%}  "
            f"Role={r.get('predicted_role', '?'):18s}  "
            f"Exp={r.get('experience_found', 0):.1f}yrs  "
            f"{r.get('recommendation', '')}"
        )
        logger.info(
            f"      Matched skills: {', '.join(r.get('matched_skills', [])[:5])}"
        )

    logger.info(f"\nPipeline completed successfully. {len(ranked)} candidates ranked.\n")
    return True


def launch_fastapi():
    logger.info("Launching FastAPI server on http://localhost:8000 ...")
    import uvicorn

    uvicorn.run(
        "api.main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume-JD Matching System")
    parser.add_argument("--test", action="store_true", help="Run smoke test only")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI server")
    args = parser.parse_args()

    logger.info("Setting up project...")
    setup_directories()
    download_nlp_models()

    if args.test:
        ok = run_smoke_test()
        sys.exit(0 if ok else 1)
    else:
        launch_fastapi()
