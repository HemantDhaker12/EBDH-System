import numpy as np
from edhc.app.ranking.ranker import LambdaMARTRanker, LinearRanker
from edhc.app.calibration.calibrator import ScoreCalibrator
from edhc.app.schemas.features import CandidateFeatures

def test_rankers():
    # Test that we can instantiate and fit our rankers with dummy data
    X = np.random.randn(10, 12)
    y = np.random.randint(0, 5, size=(10,))
    qids = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    
    # LambdaMART
    ranker = LambdaMARTRanker()
    ranker.train(X, y, qids)
    preds = ranker.predict(X)
    assert preds.shape == (10,)
    
    # Linear Fallback
    linear_ranker = LinearRanker()
    linear_ranker.train(X, y, qids)
    lin_preds = linear_ranker.predict(X)
    assert lin_preds.shape == (10,)

def test_score_calibration():
    calibrator = ScoreCalibrator()
    
    candidate_ids = ["cand_1", "cand_2", "cand_3"]
    raw_scores = [0.8, 0.4, 0.2]
    
    features_list = [
        CandidateFeatures(
            candidate_id="cand_1",
            job_id="job_1",
            contradiction_count=1.0,  # Should apply penalty
            relevant_years=5.0,
            confidence=0.8,
            credibility=0.7
        ),
        CandidateFeatures(
            candidate_id="cand_2",
            job_id="job_1",
            contradiction_count=0.0,
            relevant_years=3.0,
            confidence=0.6,
            credibility=1.0
        ),
        CandidateFeatures(
            candidate_id="cand_3",
            job_id="job_1",
            contradiction_count=0.0,
            relevant_years=10.0,
            confidence=0.9,
            credibility=1.0
        )
    ]
    
    records = calibrator.calibrate(candidate_ids, raw_scores, features_list)
    
    assert len(records) == 3
    # Check that rank keys are added
    ranks = [r["rank"] for r in records]
    assert sorted(ranks) == [1, 2, 3]
    
    # Candidate 1 raw score was 0.8 (norm = 1.0), but has contradiction_count=1 which applies 0.1 penalty.
    # Calibrated score is roughly 0.9. Let's make sure it is updated.
    c1_record = next(r for r in records if r["candidate_id"] == "cand_1")
    assert len(c1_record["penalties_applied"]) > 0
