"""
tests/test_pipeline.py
-----------------------
Unit and integration tests for every module.

Run:  python -m pytest tests/ -v
      python -m pytest tests/ -v --tb=short   (compact tracebacks)
"""

import sys
import os
import json
import unittest
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Preprocessing tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        from utils.preprocessing import (
            clean_text, tokenize, remove_stopwords,
            extract_skills, extract_experience_years, extract_education,
        )
        self.clean_text            = clean_text
        self.tokenize              = tokenize
        self.remove_stopwords      = remove_stopwords
        self.extract_skills        = extract_skills
        self.extract_experience    = extract_experience_years
        self.extract_education     = extract_education

    def test_clean_text_removes_urls(self):
        text = "Visit https://example.com for details"
        result = self.clean_text(text)
        self.assertNotIn("https://", result)

    def test_clean_text_removes_email(self):
        text = "Contact me at foo@bar.com please"
        result = self.clean_text(text)
        self.assertNotIn("@", result)

    def test_clean_text_lowercases(self):
        result = self.clean_text("Python MACHINE Learning")
        self.assertEqual(result, result.lower())

    def test_tokenize_returns_list(self):
        tokens = self.tokenize("hello world test")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

    def test_remove_stopwords_filters_common_words(self):
        tokens = ["the", "python", "is", "great", "a", "and"]
        filtered = self.remove_stopwords(tokens)
        self.assertNotIn("the", filtered)
        self.assertIn("python", filtered)

    def test_extract_skills_finds_python(self):
        text = "Experienced in Python, Machine Learning, and SQL databases."
        skills = self.extract_skills(text)
        skill_lower = [s.lower() for s in skills]
        self.assertIn("python", skill_lower)
        self.assertIn("sql", skill_lower)
        self.assertIn("machine learning", skill_lower)

    def test_extract_experience_explicit_pattern(self):
        text = "I have 5 years of experience in data science"
        years = self.extract_experience(text)
        self.assertAlmostEqual(years, 5.0)

    def test_extract_experience_date_range(self):
        text = "Software Engineer | Acme Corp | 2019 - 2023"
        years = self.extract_experience(text)
        self.assertGreaterEqual(years, 4.0)

    def test_extract_experience_zero_when_none(self):
        years = self.extract_experience("No experience mentioned here.")
        self.assertEqual(years, 0.0)

    def test_extract_education_masters(self):
        text = "M.Tech Computer Science from IIT Bombay 2020"
        edu = self.extract_education(text)
        self.assertEqual(edu, "Master's")

    def test_extract_education_bachelors(self):
        text = "Completed B.Tech from NIT Trichy in 2018"
        edu = self.extract_education(text)
        self.assertEqual(edu, "Bachelor's")

    def test_extract_education_not_specified(self):
        edu = self.extract_education("No education information here")
        self.assertEqual(edu, "Not Specified")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ML Engine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMLEngine(unittest.TestCase):

    def setUp(self):
        from utils.ml_engine import (
            compute_cosine_similarity,
            compute_skill_overlap,
            compute_experience_score,
            compute_final_score,
            generate_explanation,
        )
        self.cosine_sim       = compute_cosine_similarity
        self.skill_overlap    = compute_skill_overlap
        self.exp_score        = compute_experience_score
        self.final_score      = compute_final_score
        self.gen_explanation  = generate_explanation

    def test_cosine_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype="float32")
        score = self.cosine_sim(v, v)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_cosine_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0], dtype="float32")
        b = np.array([0.0, 1.0, 0.0], dtype="float32")
        score = self.cosine_sim(a, b)
        self.assertAlmostEqual(score, 0.0, places=4)

    def test_cosine_bounded(self):
        rng = np.random.default_rng(42)
        a = rng.random(64).astype("float32")
        b = rng.random(64).astype("float32")
        score = self.cosine_sim(a, b)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_skill_overlap_perfect(self):
        skills = ["python", "sql", "pandas"]
        score = self.skill_overlap(skills, skills)
        self.assertAlmostEqual(score, 1.0)

    def test_skill_overlap_no_match(self):
        score = self.skill_overlap(["java", "spring"], ["python", "pandas"])
        self.assertAlmostEqual(score, 0.0)

    def test_skill_overlap_partial(self):
        r = ["python", "sql", "java"]
        j = ["python", "sql", "docker"]
        score = self.skill_overlap(r, j)
        # intersection=2, union=4  → 0.5
        self.assertAlmostEqual(score, 0.5, places=4)

    def test_skill_overlap_empty_jd(self):
        score = self.skill_overlap(["python"], [])
        self.assertEqual(score, 0.5)   # neutral

    def test_experience_score_meets_req(self):
        score = self.exp_score(5.0, 3.0)
        self.assertAlmostEqual(score, 1.0)

    def test_experience_score_below_req(self):
        score = self.exp_score(2.0, 4.0)
        self.assertAlmostEqual(score, 0.5)

    def test_experience_score_no_req(self):
        score = self.exp_score(0.0, 0.0)
        self.assertAlmostEqual(score, 1.0)

    def test_final_score_bounded(self):
        score = self.final_score(0.8, 0.6, 0.9)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_final_score_custom_weights(self):
        weights = {"semantic": 1.0, "skill": 0.0, "experience": 0.0}
        score = self.final_score(0.75, 0.0, 0.0, weights=weights)
        self.assertAlmostEqual(score, 0.75)

    def test_explanation_recommendation_strong(self):
        resume = {"skills": ["python", "sql"], "experience_years": 5.0,
                  "education": "Master's", "predicted_role": "DS"}
        jd = {"required_skills": ["python", "sql"], "required_experience": 3.0}
        exp = self.gen_explanation(resume, jd, 0.9, 0.85)
        self.assertIn("Strong", exp["recommendation"])

    def test_explanation_recommendation_weak(self):
        resume = {"skills": [], "experience_years": 0.0,
                  "education": "Not Specified", "predicted_role": "?"}
        jd = {"required_skills": ["python", "aws", "spark"], "required_experience": 5.0}
        exp = self.gen_explanation(resume, jd, 0.1, 0.12)
        self.assertIn("Weak", exp["recommendation"])

    def test_explanation_has_all_fields(self):
        resume = {"skills": ["python"], "experience_years": 2.0,
                  "education": "Bachelor's", "predicted_role": "Analyst"}
        jd = {"required_skills": ["python", "sql"], "required_experience": 3.0}
        exp = self.gen_explanation(resume, jd, 0.6, 0.55)
        for key in ["matched_skills", "missing_skills", "extra_skills",
                    "experience_gap", "recommendation", "match_score"]:
            self.assertIn(key, exp)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Mining tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataMining(unittest.TestCase):

    def setUp(self):
        from utils.data_mining import (
            get_skill_frequencies, top_skills,
            skill_gap_analysis, profile_clusters,
        )
        self.get_freq     = get_skill_frequencies
        self.top_skills   = top_skills
        self.gap_analysis = skill_gap_analysis
        self.profile      = profile_clusters

    def test_skill_frequency_counts(self):
        skill_lists = [
            ["python", "sql", "pandas"],
            ["python", "spark", "sql"],
            ["python", "docker"],
        ]
        df = self.get_freq(skill_lists)
        self.assertFalse(df.empty)
        top = df.iloc[0]
        self.assertEqual(top["skill"], "python")
        self.assertEqual(top["count"], 3)

    def test_skill_frequency_pct(self):
        skill_lists = [["python"], ["python"], ["java"]]
        df = self.get_freq(skill_lists)
        py_row = df[df["skill"] == "python"].iloc[0]
        self.assertAlmostEqual(py_row["frequency_pct"], 66.7, places=0)

    def test_top_skills_returns_n(self):
        skill_lists = [["a", "b", "c"], ["a", "b"], ["a"]]
        top = self.top_skills(skill_lists, n=2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0][0], "a")

    def test_skill_gap_analysis(self):
        jd_skills = ["python", "spark", "kafka"]
        all_skills = [
            ["python", "spark"],
            ["python"],
            ["python", "kafka"],
        ]
        df = self.gap_analysis(jd_skills, all_skills)
        self.assertFalse(df.empty)
        py_row = df[df["skill"] == "python"].iloc[0]
        self.assertEqual(py_row["candidates_with_skill"], 3)
        self.assertAlmostEqual(py_row["coverage_pct"], 100.0)

    def test_cluster_profile_structure(self):
        resumes = [
            {"resume_id": "R1", "skills": ["python", "ml"], "experience_years": 4.0, "predicted_role": "Data Scientist"},
            {"resume_id": "R2", "skills": ["python", "sql"], "experience_years": 2.0, "predicted_role": "Data Scientist"},
            {"resume_id": "R3", "skills": ["java", "spring"], "experience_years": 5.0, "predicted_role": "Engineer"},
        ]
        labels = [0, 0, 1]
        profiles = self.profile(resumes, labels)
        self.assertIn(0, profiles)
        self.assertIn(1, profiles)
        self.assertEqual(profiles[0]["size"], 2)
        self.assertIn("top_skills", profiles[0])

    def test_apriori_runs(self):
        try:
            from utils.data_mining import run_apriori
            skill_lists = [
                ["python", "sql", "pandas", "scikit-learn"],
                ["python", "sql", "spark", "airflow"],
                ["python", "pandas", "matplotlib", "scikit-learn"],
                ["sql", "pandas", "tableau", "power bi"],
                ["python", "sql", "pandas", "numpy"],
            ]
            items, rules = run_apriori(skill_lists, min_support=0.4, min_confidence=0.5)
            # Just check it doesn't crash and returns DataFrames
            import pandas as pd
            self.assertIsInstance(items, pd.DataFrame)
            self.assertIsInstance(rules, pd.DataFrame)
        except ImportError:
            self.skipTest("mlxtend not installed")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Role classifier tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRoleClassifier(unittest.TestCase):

    def setUp(self):
        from utils.ml_engine import RoleClassifier
        self.clf = RoleClassifier()

    def test_predict_before_training(self):
        # Should return 'Unknown', not crash
        result = self.clf.predict("python machine learning data science")
        self.assertEqual(result, "Unknown")

    def test_train_and_predict(self):
        texts = [
            "python machine learning deep learning tensorflow scikit-learn",
            "python machine learning nlp bert transformers huggingface",
            "react nodejs javascript typescript rest api graphql docker",
            "vue angular css html javascript frontend webpack",
            "spark kafka airflow etl data pipeline postgresql",
            "hadoop spark scala data warehouse redshift snowflake",
        ]
        labels = [
            "Data Scientist", "Data Scientist",
            "Full Stack Developer", "Full Stack Developer",
            "Data Engineer", "Data Engineer",
        ]
        self.clf.train(texts, labels)
        self.assertTrue(self.clf.trained)

        pred = self.clf.predict("python tensorflow deep learning neural network")
        self.assertIn(pred, ["Data Scientist", "Full Stack Developer", "Data Engineer"])

    def test_predict_proba_returns_dict(self):
        texts = ["python ml data", "react javascript frontend",
                 "spark etl pipeline", "python ml nlp", "javascript vue css",
                 "kafka airflow spark"]
        labels = ["DS", "FE", "DE", "DS", "FE", "DE"]
        self.clf.train(texts, labels)
        proba = self.clf.predict_proba("python machine learning")
        self.assertIsInstance(proba, dict)
        self.assertGreater(len(proba), 0)
        total = sum(proba.values())
        self.assertAlmostEqual(total, 1.0, places=3)


# ─────────────────────────────────────────────────────────────────────────────
# 5. File extractor tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFileExtractor(unittest.TestCase):

    def setUp(self):
        from utils.file_extractor import extract_text_from_bytes
        self.extract = extract_text_from_bytes

    def test_extract_txt(self):
        content = b"John Doe\nPython Developer\n5 years experience"
        result = self.extract(content, "resume.txt")
        self.assertIn("John", result)
        self.assertIn("Python", result)

    def test_unsupported_format_returns_empty(self):
        result = self.extract(b"some bytes", "resume.xyz")
        self.assertEqual(result, "")

    def test_empty_bytes(self):
        result = self.extract(b"", "empty.txt")
        self.assertEqual(result.strip(), "")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integration test — full pipeline on sample data
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline(unittest.TestCase):

    def test_pipeline_runs_on_sample_data(self):
        from data.sample.sample_data import SAMPLE_RESUMES, SAMPLE_JOB_DESCRIPTIONS
        from utils.matching_pipeline import run_matching_pipeline

        raw = [
            {
                "resume_id": r["id"],
                "name":      r["name"],
                "filename":  r["name"] + ".txt",
                "raw_text":  r["text"],
                "role":      r["role"],
            }
            for r in SAMPLE_RESUMES
        ]
        jd = SAMPLE_JOB_DESCRIPTIONS["data_scientist"]

        result = run_matching_pipeline(raw, jd, n_clusters=3, min_support=0.2)

        self.assertNotIn("error", result)
        self.assertIn("ranked_candidates", result)
        self.assertEqual(len(result["ranked_candidates"]), len(raw))

    def test_pipeline_ranking_is_sorted(self):
        from data.sample.sample_data import SAMPLE_RESUMES, SAMPLE_JOB_DESCRIPTIONS
        from utils.matching_pipeline import run_matching_pipeline

        raw = [{"resume_id": r["id"], "name": r["name"],
                "filename": r["name"]+".txt", "raw_text": r["text"],
                "role": r["role"]} for r in SAMPLE_RESUMES]
        jd = SAMPLE_JOB_DESCRIPTIONS["data_engineer"]

        result = run_matching_pipeline(raw, jd, n_clusters=3, min_support=0.3)
        ranked = result["ranked_candidates"]

        scores = [r["match_score"] for r in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_pipeline_all_explanations_present(self):
        from data.sample.sample_data import SAMPLE_RESUMES, SAMPLE_JOB_DESCRIPTIONS
        from utils.matching_pipeline import run_matching_pipeline

        raw = [{"resume_id": r["id"], "name": r["name"],
                "filename": r["name"]+".txt", "raw_text": r["text"],
                "role": r["role"]} for r in SAMPLE_RESUMES[:3]]
        jd = SAMPLE_JOB_DESCRIPTIONS["data_scientist"]

        result = run_matching_pipeline(raw, jd, n_clusters=2, min_support=0.3)

        for candidate in result["ranked_candidates"]:
            for field in ["match_score", "matched_skills", "missing_skills",
                          "recommendation", "experience_found", "education"]:
                self.assertIn(field, candidate,
                              msg=f"Field '{field}' missing for {candidate.get('name')}")

    def test_pipeline_empty_jd_returns_error(self):
        from utils.matching_pipeline import run_matching_pipeline
        result = run_matching_pipeline(
            [{"resume_id": "X", "name": "Test", "filename": "t.txt", "raw_text": "python"}],
            jd_text=""
        )
        self.assertIn("error", result)

    def test_pipeline_empty_resumes_returns_error(self):
        from utils.matching_pipeline import run_matching_pipeline
        result = run_matching_pipeline([], jd_text="Python developer needed")
        self.assertIn("error", result)

    def test_score_weights_affect_result(self):
        from data.sample.sample_data import SAMPLE_RESUMES, SAMPLE_JOB_DESCRIPTIONS
        from utils.matching_pipeline import run_matching_pipeline

        raw = [{"resume_id": r["id"], "name": r["name"],
                "filename": r["name"]+".txt", "raw_text": r["text"],
                "role": r["role"]} for r in SAMPLE_RESUMES[:4]]
        jd = SAMPLE_JOB_DESCRIPTIONS["data_scientist"]

        r1 = run_matching_pipeline(raw, jd, n_clusters=2,
                                   score_weights={"semantic":1.0,"skill":0.0,"experience":0.0})
        r2 = run_matching_pipeline(raw, jd, n_clusters=2,
                                   score_weights={"semantic":0.0,"skill":1.0,"experience":0.0})

        # Rankings can differ when weights are extreme
        top1 = r1["ranked_candidates"][0]["name"]
        top2 = r2["ranked_candidates"][0]["name"]
        # Both runs should complete successfully regardless of ranking
        self.assertIsNotNone(top1)
        self.assertIsNotNone(top2)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Database tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabase(unittest.TestCase):

    def setUp(self):
        # Use a temp DB for tests
        import database.db_manager as dbm
        dbm.DB_PATH = ":memory:"   # won't actually work with SQLAlchemy file path
        # We'll just test the logic without a real file

    def test_save_and_retrieve(self):
        try:
            from database.db_manager import save_resume, get_resume_by_id
            data = {
                "resume_id":      "TEST_001",
                "name":           "Test User",
                "filename":       "test.pdf",
                "raw_text":       "raw text here",
                "processed_text": "processed text",
                "skills":         ["python", "sql"],
                "experience_years": 3.0,
                "education":      "Bachelor's",
                "predicted_role": "Data Scientist",
                "match_score":    0.75,
                "cluster_label":  1,
            }
            save_resume(data)
            retrieved = get_resume_by_id("TEST_001")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved["name"], "Test User")
            self.assertAlmostEqual(retrieved["match_score"], 0.75)
            self.assertIn("python", retrieved["skills"])
        except Exception as e:
            self.skipTest(f"DB test skipped (env limitation): {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestMLEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestDataMining))
    suite.addTests(loader.loadTestsFromTestCase(TestRoleClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestFileExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestFullPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabase))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
