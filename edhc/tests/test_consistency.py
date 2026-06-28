from datetime import date
from edhc.app.schemas.candidate import CandidateProfileNormalized, WorkExperience
from edhc.app.consistency.detector import ConsistencyDetector

def test_timeline_overlaps():
    detector = ConsistencyDetector()
    
    # Setup candidate with overlapping jobs
    experiences = [
        WorkExperience(
            job_title="Dev 1",
            company_name="Company A",
            start_date=date(2020, 1, 1),
            end_date=date(2022, 1, 1),
            description="",
            duration_years=2.0
        ),
        # Overlaps significantly with Dev 1
        WorkExperience(
            job_title="Dev 2",
            company_name="Company B",
            start_date=date(2021, 6, 1),
            end_date=date(2023, 6, 1),
            description="",
            duration_years=2.0
        )
    ]
    
    overlaps = detector._detect_timeline_overlaps(experiences)
    assert len(overlaps) == 1
    assert "Company A" in overlaps[0][0]
    assert "Company B" in overlaps[0][1]

def test_unrealistic_tech_dates():
    detector = ConsistencyDetector()
    
    candidate = CandidateProfileNormalized(
        id="test-id",
        name="Candidate A",
        summary="",
        skills=[],
        experiences=[
            WorkExperience(
                job_title="Engineer",
                company_name="Old Co",
                start_date=date(2010, 1, 1), # FastAPI released in 2018
                end_date=date(2015, 1, 1),
                description="Built APIs using FastAPI",
                duration_years=5.0
            )
        ],
        projects=[],
        education=[]
    )
    
    unrealistic = detector._detect_unrealistic_tech_durations(candidate)
    assert len(unrealistic) == 1
    assert unrealistic[0][0] == "fastapi"

def test_honeypots():
    detector = ConsistencyDetector()
    
    candidate_clean = CandidateProfileNormalized(
        id="test-id",
        name="Candidate A",
        summary="",
        skills=["Python", "FastAPI"],
        experiences=[], projects=[], education=[]
    )
    assert detector._check_honeypots(candidate_clean) is False
    
    candidate_trap = CandidateProfileNormalized(
        id="test-id",
        name="Candidate A",
        summary="",
        skills=["Python", "quantum-cobol"], # Honeypot skill
        experiences=[], projects=[], education=[]
    )
    assert detector._check_honeypots(candidate_trap) is True
