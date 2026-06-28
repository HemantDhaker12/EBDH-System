import numpy as np
from datetime import date
from edhc.app.schemas.candidate import CandidateProfileNormalized, WorkExperience, Project
from edhc.app.schemas.features import CandidateFeatures
from edhc.app.features.generator import FeatureGenerator
from edhc.app.skills.validator import SkillsValidator
from edhc.app.ranking.ranker import LambdaMARTRanker
from edhc.app.jd.intelligence import JobDescriptionParser

def test_feature_generation_dimension():
    # Setup parser and default JD
    jd_parser = JobDescriptionParser()
    jd = jd_parser.parse(
        "Job Title: Senior ML Engineer. Required skills: Python, Machine Learning.",
        "Senior ML Engineer"
    )
    
    # Setup candidate normalized
    candidate = CandidateProfileNormalized(
        id="CAND_9999999",
        name="Test Candidate",
        summary="Experienced senior machine learning developer.",
        skills=["Python", "ML"],
        experiences=[
            WorkExperience(
                job_title="ML Engineer",
                company_name="Meta",
                start_date=date(2020, 1, 1),
                end_date=date(2023, 1, 1),
                description="Built ranking systems.",
                duration_years=3.0
            )
        ],
        projects=[],
        education=[]
    )
    
    generator = FeatureGenerator()
    features = generator.generate(candidate, jd)
    
    # Assert features array contains exactly 39 features
    feat_arr = features.to_array()
    assert feat_arr.shape == (39,)
    assert len(features.get_feature_names()) == 39
    assert len(features.to_dict()) == 39

def test_feature_determinism():
    jd_parser = JobDescriptionParser()
    jd = jd_parser.parse("Required: Python", "Title")
    candidate = CandidateProfileNormalized(id="1", name="A", skills=["Python"], experiences=[])
    
    generator = FeatureGenerator()
    f1 = generator.generate(candidate, jd).to_array()
    f2 = generator.generate(candidate, jd).to_array()
    
    assert np.allclose(f1, f2)

def test_feature_generation_missing_fields():
    jd_parser = JobDescriptionParser()
    jd = jd_parser.parse("Required: Python", "Title")
    
    # Extreme case: completely empty profile except for required id and name
    candidate = CandidateProfileNormalized(
        id="CAND_0000000",
        name="Empty Candidate",
        skills=[],
        experiences=[],
        projects=[],
        education=[]
    )
    
    generator = FeatureGenerator()
    # Ensure it generates without any exception
    features = generator.generate(candidate, jd)
    assert features.to_array().shape == (39,)

def test_synonym_mapping():
    validator = SkillsValidator()
    # Check ML/Machine Learning mapping
    assert validator.skill_matches("ML", "Machine Learning")
    assert validator.skill_matches("NLP", "Natural Language Processing")
    assert validator.skill_matches("k8s", "kubernetes")

def test_backward_compatibility_slicing():
    # Instantiate LightGBM ranker
    ranker = LambdaMARTRanker()
    
    # Train on 12 features (mimicking a legacy model)
    X_train_12 = np.random.randn(10, 12)
    y_train = np.random.randint(0, 5, size=(10,))
    qids = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    ranker.train(X_train_12, y_train, qids)
    
    # Predict using 39 features (new feature matrix)
    X_test_39 = np.random.randn(10, 39)
    # The ranker predict should slice the features automatically and succeed
    preds = ranker.predict(X_test_39)
    assert preds.shape == (10,)

def test_new_ranker_compatibility():
    ranker = LambdaMARTRanker()
    # Train on full 39 features
    X_train_39 = np.random.randn(10, 39)
    y_train = np.random.randint(0, 5, size=(10,))
    qids = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    ranker.train(X_train_39, y_train, qids)
    
    # Predict using full 39 features
    preds = ranker.predict(X_train_39)
    assert preds.shape == (10,)
